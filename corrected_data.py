"""Free-source provenance gates for raw prices, corporate actions and membership."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np

from universe_membership import membership_status, load_membership, filter_panel_point_in_time
from safe_io import atomic_write_csv


def _attributed(value):
    """A label alone is not provenance: require a URL and a real retrieval date."""
    return (isinstance(value, dict) and value.get("access_cost") == "free"
            and str(value.get("source_url", "")).startswith(("https://", "http://"))
            and bool(str(value.get("license", "")).strip())
            and pd.notna(pd.to_datetime(value.get("retrieved_at"), errors="coerce", utc=True)))


def raw_price_gaps(path, ticker, expected_sessions):
    """Inspect the data itself; a matching file hash cannot prove price coverage."""
    try:
        frame = pd.read_parquet(path)
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("A date index is required")
        dates = frame.index
        if dates.tz is not None:
            dates = dates.tz_convert("America/New_York").tz_localize(None)
        dates = dates.normalize()
        numeric = frame[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=float)
        if (dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing
                or not np.isfinite(numeric).all() or (numeric[:, :4] <= 0).any()
                or (numeric[:, 4] < 0).any()
                or (numeric[:, 1] < numeric[:, [0, 2, 3]].max(axis=1)).any()
                or (numeric[:, 2] > numeric[:, [0, 1, 3]].min(axis=1)).any()):
            raise ValueError("Invalid dates, prices or volume")
        missing = expected_sessions.difference(dates)
        if len(missing):
            return [{"reason": "raw_price_sessions_missing", "ticker": ticker,
                     "count": len(missing), "first": str(missing.min().date()),
                     "last": str(missing.max().date())}]
        return []
    except (OSError, ValueError, KeyError, TypeError):
        return [{"reason": "raw_price_content_invalid", "ticker": ticker}]


def action_gaps(actions):
    """Reject unsupported or incomplete events before they reach the cash ledger."""
    required = {"event_id", "ticker", "kind", "date", "value", "source"}
    if not required.issubset(actions):
        return [{"reason": "corporate_action_schema_or_duplicate"}]
    gaps = []
    if actions.event_id.duplicated().any() or actions[list(required)].isna().any().any():
        gaps.append({"reason": "corporate_action_schema_or_duplicate"})
    for row in actions.to_dict("records"):
        kind = row["kind"]
        value = pd.to_numeric(row["value"], errors="coerce")
        date = pd.to_datetime(row["date"], errors="coerce", utc=True)
        invalid = (kind not in {"split", "dividend", "symbol_change", "cash_liquidation"}
                   or pd.isna(date) or not np.isfinite(value) or value < 0
                   or any(not str(row.get(k, "")).strip() for k in ("event_id", "ticker", "source")))
        if kind == "split" and value <= 0:
            invalid = True
        if kind == "symbol_change" and (pd.isna(row.get("new_ticker")) or not str(row.get("new_ticker", "")).strip()):
            invalid = True
        if kind == "dividend":
            ex_date = pd.to_datetime(row.get("ex_date"), errors="coerce", utc=True)
            if pd.isna(ex_date) or pd.isna(date) or ex_date > date:
                gaps.append({"reason": "dividend_entitlement_dates_missing", "ticker": row["ticker"]})
        if invalid:
            gaps.append({"reason": "corporate_action_value_or_kind_invalid", "ticker": row["ticker"]})
    return gaps


def import_membership(source: Path, destination: Path) -> dict:
    """Import documented free-source rows without claiming the table is complete."""
    from universe_membership import REQUIRED_COLUMNS, PROVENANCE_COLUMNS
    frame = pd.read_csv(source)
    missing = (REQUIRED_COLUMNS | PROVENANCE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Membership provenance columns missing: {sorted(missing)}")
    if not frame.access_cost.eq("free").all() or not frame.source_url.fillna("").str.match(r"https?://").all():
        raise ValueError("Only attributed free sources can be imported")
    if frame.license.fillna("").str.strip().eq("").any() or pd.to_datetime(frame.retrieved_at, errors="coerce", utc=True).isna().any():
        raise ValueError("License and retrieval timestamp required on every membership row")
    starts = pd.to_datetime(frame.effective_from, errors="coerce")
    ends = pd.to_datetime(frame.effective_to, errors="coerce")
    supplied_ends = frame.effective_to.notna() & frame.effective_to.astype(str).str.strip().ne("")
    if starts.isna().any() or (supplied_ends & ends.isna()).any() or (ends.notna() & ends.lt(starts)).any():
        raise ValueError("Invalid membership effective interval")
    # Normalize before checking overlap, so BRK.B and BRK-B cannot hide two
    # conflicting histories for the same symbol.
    if frame.ticker.isna().any() or frame.ticker.astype(str).str.strip().eq("").any():
        raise ValueError("Missing membership ticker")
    frame["ticker"] = frame.ticker.astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    for ticker, group in frame.assign(_start=starts, _end=ends).groupby("ticker"):
        previous_end = None
        for row in group.sort_values("_start").to_dict("records"):
            if previous_end is not None and row["_start"] <= previous_end:
                raise ValueError(f"Overlapping membership intervals: {ticker}")
            previous_end = row["_end"] if pd.notna(row["_end"]) else pd.Timestamp.max
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(frame, destination)
    return {"imported_rows": len(frame), "complete": False, "next_step": "run full source/price coverage validation"}


def validate_sources(data_dir: Path, membership_path: Path, *, start, end) -> dict:
    """List concrete gaps; a zero-action history still needs verified coverage."""
    membership = membership_status(membership_path, data_dir=data_dir / "raw", coverage_start=start, coverage_end=end)
    gaps = [{"reason": reason, "path": str(membership_path)} for reason in membership.get("reasons", [])]
    raw = data_dir / "raw"
    manifest_path = raw / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if not isinstance(manifest, dict) or not isinstance(manifest.get("symbols", {}), dict):
            raise ValueError("Invalid manifest object")
    except (OSError, ValueError, TypeError):
        manifest = {}
        gaps.append({"reason": "source_manifest_unreadable"})
    try:
        table = load_membership(membership_path)
    except (ValueError, KeyError) as exc:
        return {"complete": False, "membership": membership, "gaps": gaps + [{"reason": "membership_invalid", "detail": str(exc)}],
                "price_fingerprints": {}, "actions_verified": False}
    if table.empty:
        gaps.append({"reason": "historical_membership_missing", "path": str(membership_path)})
    # A community reconstruction must not become verified merely by copying it
    # into the expected filename. Bind independent coverage evidence to its hash.
    proof = manifest.get("membership", {})
    proof = proof if isinstance(proof, dict) else {}
    proof_start = pd.to_datetime(proof.get("start"), errors="coerce", utc=True)
    proof_end = pd.to_datetime(proof.get("end"), errors="coerce", utc=True)
    if (not _attributed(proof) or proof.get("verified") is not True
            or not membership_path.exists()
            or proof.get("sha256") != hashlib.sha256(membership_path.read_bytes()).hexdigest()
            or pd.isna(proof_start) or pd.isna(proof_end)
            or proof_start > pd.to_datetime(start, utc=True) or proof_end < pd.to_datetime(end, utc=True)):
        gaps.append({"reason": "membership_coverage_unverified"})
    # Validate direct file loads too, not just files passed through the importer.
    for ticker, group in table.groupby("ticker"):
        previous_end = None
        for row in group.sort_values("effective_from").itertuples():
            if (pd.isna(row.effective_from)
                    or (pd.notna(row.effective_to) and row.effective_to < row.effective_from)
                    or (previous_end is not None and row.effective_from <= previous_end)):
                gaps.append({"reason": "membership_interval_invalid", "ticker": ticker})
            previous_end = row.effective_to if pd.notna(row.effective_to) else pd.Timestamp.max
    tickers = set(table.loc[(table.effective_from <= pd.Timestamp(end)) & (table.effective_to.isna() | (table.effective_to >= pd.Timestamp(start))), "ticker"])
    tickers |= {"SPY", "QQQ"}
    versions = {}
    from core_satellite_alpha import _nyse_sessions
    sessions = _nyse_sessions(pd.Timestamp(start), pd.Timestamp(end))
    for ticker in sorted(tickers):
        metadata = manifest.get("symbols", {}).get(ticker, {})
        metadata = metadata if isinstance(metadata, dict) else {}
        path = raw / f"{ticker}.parquet"
        if not path.exists():
            gaps.append({"reason": "raw_price_file_missing", "ticker": ticker, "path": str(path)})
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            versions[ticker] = digest
            if metadata.get("sha256") != digest:
                gaps.append({"reason": "raw_price_checksum_unverified", "ticker": ticker})
            # Stocks need every eligible session; benchmarks need the full era.
            expected = sessions if ticker in {"SPY", "QQQ"} else pd.DatetimeIndex([])
            for row in table.loc[table.ticker == ticker].itertuples():
                expected = expected.union(sessions[(sessions >= row.effective_from) &
                    (sessions <= (row.effective_to if pd.notna(row.effective_to) else pd.Timestamp(end)))])
            gaps.extend(raw_price_gaps(path, ticker, expected))
        for key in ("source_url", "retrieved_at", "license"):
            if not metadata.get(key):
                gaps.append({"reason": f"{key}_missing", "ticker": ticker})
        if metadata.get("access_cost") != "free" or metadata.get("adjustment_mode") != "raw_ohlcv":
            gaps.append({"reason": "free_raw_source_unverified", "ticker": ticker})
        if not _attributed(metadata):
            gaps.append({"reason": "raw_price_provenance_invalid", "ticker": ticker})
        if not metadata.get("feed") or not metadata.get("symbol_mapping"):
            gaps.append({"reason": "raw_price_feed_or_symbol_mapping_missing", "ticker": ticker})
        # Tickers can be reused by unrelated securities (for example SHLD).
        # Turning provider symbol mapping off does not establish company identity.
        identity = metadata.get("identity", {})
        identity = identity if isinstance(identity, dict) else {}
        if (not _attributed(identity) or identity.get("verified") is not True
                or not str(identity.get("security_id", "")).strip()
                or not metadata.get("sha256") or identity.get("sha256") != metadata.get("sha256")):
            gaps.append({"reason": "raw_price_security_identity_unverified", "ticker": ticker})
        coverage = metadata.get("actions_coverage", {})
        coverage = coverage if isinstance(coverage, dict) else {}
        coverage_start = pd.to_datetime(coverage.get("start"), errors="coerce", utc=True)
        coverage_end = pd.to_datetime(coverage.get("end"), errors="coerce", utc=True)
        if (coverage.get("verified") is not True or not _attributed(coverage) or
            pd.isna(coverage_start) or pd.isna(coverage_end) or
            coverage_start > pd.to_datetime(start, utc=True) or coverage_end < pd.to_datetime(end, utc=True)):
            gaps.append({"reason": "corporate_action_coverage_missing", "ticker": ticker, "start": str(start), "end": str(end)})
    action_path = raw / "actions.csv"
    if not action_path.exists() or manifest.get("actions_sha256") != hashlib.sha256(action_path.read_bytes()).hexdigest():
        gaps.append({"reason": "corporate_action_file_unverified", "path": str(action_path)})
    elif action_path.exists():
        try:
            gaps.extend(action_gaps(pd.read_csv(action_path)))
        except (OSError, ValueError, TypeError):
            gaps.append({"reason": "corporate_action_file_unreadable"})
    return {"complete": membership.get("complete", False) and not gaps, "membership": membership,
            "gaps": gaps, "tickers": sorted(tickers), "price_fingerprints": versions,
            "adjustment_mode": "raw_ohlcv", "actions_verified": not any("action" in g["reason"] for g in gaps)}


def load_raw_panel(data_dir, membership_path, *, start, end):
    """Load every historical member, including removed names, only after verification."""
    report = validate_sources(Path(data_dir), Path(membership_path), start=start, end=end)
    if not report["complete"]:
        raise ValueError("Corrected data blocked: " + json.dumps(report["gaps"], default=str))
    pieces = []
    for ticker in report["tickers"]:
        frame = pd.read_parquet(Path(data_dir) / "raw" / f"{ticker}.parquet")
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame.index)
        frame["ticker"] = ticker
        pieces.append(frame.reset_index(drop=True))
    # Keep prices after index removal for liquidating held positions. Membership
    # filters candidates separately; it must not erase a held asset's price.
    return pd.concat(pieces, ignore_index=True), report


def eligible_candidates(panel, membership_path):
    membership = load_membership(Path(membership_path))
    return filter_panel_point_in_time(panel, membership[["ticker", "effective_from", "effective_to"]])


def validate_dated_inputs(frame):
    """A fact used at the close needs a free source and an earlier publication."""
    required = {"date", "ticker", "published_at", "source_url", "access_cost"}
    if not required.issubset(frame):
        raise ValueError(f"Dated feature/context provenance missing: {sorted(required - set(frame.columns))}")
    if frame.empty or frame.duplicated(["date", "ticker"]).any() or frame.ticker.isna().any():
        raise ValueError("Empty or duplicate dated evidence")
    published = pd.to_datetime(frame.published_at, utc=True, errors="coerce")
    from core_satellite_alpha import _nyse_calendar
    dates = pd.to_datetime(frame.date).dt.normalize()
    calendar = _nyse_calendar(dates.min().year, dates.max().year)
    # Early-close sessions end before 16:00, so use the exchange's actual close.
    closes = pd.to_datetime(dates.map(lambda date: calendar.session_close(date) if calendar.is_session(date) else pd.NaT), utc=True)
    invalid = published.isna() | closes.isna() | (published > closes) | ~frame.access_cost.eq("free") | ~frame.source_url.fillna("").str.match(r"https?://")
    if invalid.any():
        raise ValueError("Unavailable or future feature/context evidence: " + str(frame.loc[invalid, ["date", "ticker"]].to_dict("records")[:20]))


def validate_context_coverage(context, expected, configurations):
    """Missing history must not silently become the shared OTHER sector bucket."""
    required = set()
    for config in configurations:
        if int(config.get("max_per_sector", 2)) > 0:
            required.add("sector")
        if int(config.get("earnings_blackout_days", 0)) > 0:
            required.add("days_to_next_earnings")
        if config.get("regime_mode", "static") != "static":
            required.add("vix_inverted")
    if not required:
        return
    if context is None or not required.issubset(context):
        raise ValueError("Required dated context fields missing: " + ", ".join(sorted(required)))
    validate_dated_inputs(context)
    joined = expected[["date", "ticker"]].drop_duplicates().merge(
        context, on=["date", "ticker"], how="left", validate="one_to_one")
    if joined.empty or joined[sorted(required)].isna().any().any():
        raise ValueError("Dated context coverage incomplete")
    if "sector" in required and joined.sector.astype(str).str.strip().eq("").any():
        raise ValueError("Dated sector classification empty")
    for column in required - {"sector"}:
        if not np.isfinite(pd.to_numeric(joined[column], errors="coerce")).all():
            raise ValueError("Invalid dated context values: " + column)


def build_raw_features(bars: pd.DataFrame, actions: pd.DataFrame, *, horizon=20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small fixed candidate family before fitting any directions/weights.

    Raw prices remain intact. A separate cumulative return index handles known
    splits and ex-date dividends for momentum and volatility calculations.
    """
    pieces = []
    entity = {ticker: ticker for ticker in bars.ticker.unique()}
    # Verified symbol changes connect one business's old and new price rows;
    # they do not manufacture observations or change the raw execution symbol.
    if not actions.empty:
        for event in actions.loc[actions.kind == "symbol_change"].sort_values("date").to_dict("records"):
            old, new = event["ticker"], event["new_ticker"]
            previous = entity.get(new, new)
            replacement = entity.get(old, old)
            entity = {ticker: replacement if identity == previous else identity for ticker, identity in entity.items()}
    for _, frame in bars.groupby(bars.ticker.map(entity)):
        symbols = set(frame.ticker)
        frame = frame.sort_values("date").copy().set_index("date")
        if frame.index.has_duplicates:
            raise ValueError(f"Conflicting symbol-change price rows for {sorted(symbols)}")
        split = pd.Series(1., index=frame.index)
        dividend = pd.Series(0., index=frame.index)
        for event in actions.loc[actions.ticker.isin(symbols)].to_dict("records") if not actions.empty else []:
            date = pd.Timestamp(event.get("ex_date") if event["kind"] == "dividend" else event["date"])
            if date in frame.index:
                if event["kind"] == "split":
                    split.loc[date] *= float(event["value"])
                elif event["kind"] == "dividend":
                    dividend.loc[date] += float(event["value"])
        growth = (frame.Close * split + dividend * split) / frame.Close.shift(1)
        growth.iloc[0] = 1.
        frame["signal_close"] = growth.cumprod()
        daily = frame.signal_close.pct_change(fill_method=None)
        frame["causal_momentum_20"] = frame.signal_close.pct_change(20, fill_method=None)
        frame["causal_momentum_60"] = frame.signal_close.pct_change(60, fill_method=None)
        frame["causal_volatility_20"] = daily.rolling(20).std()
        frame["causal_dollar_volume_20"] = (frame.Close * frame.Volume).rolling(20).mean()
        label = f"forward_return_{horizon}d"
        effective_open = frame.signal_close * frame.Open / frame.Close
        frame[label] = frame.signal_close.shift(-horizon) / effective_open.shift(-1) - 1
        dates = pd.Series(frame.index, index=frame.index)
        frame[f"{label}_entry_date"] = dates.shift(-1)
        frame[f"{label}_end_date"] = dates.shift(-horizon)
        # Sector identity must be supplied as of the decision date. A missing
        # classification uses one shared bucket and never bypasses sector caps.
        if "sector" not in frame:
            frame["sector"] = "OTHER"
        pieces.append(frame.reset_index())
    joined = pd.concat(pieces, ignore_index=True)
    return joined, joined[["date", "ticker", "Open", "High", "Low", "Close", "Volume", "signal_close"]].copy()
