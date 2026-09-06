"""Free-source provenance gates for raw prices, corporate actions and membership."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from universe_membership import membership_status, load_membership, filter_panel_point_in_time
from safe_io import atomic_write_csv


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
    if starts.isna().any() or (ends.notna() & ends.lt(starts)).any():
        raise ValueError("Invalid membership effective interval")
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    try:
        table = load_membership(membership_path)
    except (ValueError, KeyError) as exc:
        return {"complete": False, "membership": membership, "gaps": gaps + [{"reason": "membership_invalid", "detail": str(exc)}],
                "price_fingerprints": {}, "actions_verified": False}
    if table.empty:
        gaps.append({"reason": "historical_membership_missing", "path": str(membership_path)})
    tickers = set(table.loc[(table.effective_from <= pd.Timestamp(end)) & (table.effective_to.isna() | (table.effective_to >= pd.Timestamp(start))), "ticker"])
    tickers |= {"SPY", "QQQ"}
    versions = {}
    for ticker in sorted(tickers):
        metadata = manifest.get("symbols", {}).get(ticker, {})
        path = raw / f"{ticker}.parquet"
        if not path.exists():
            gaps.append({"reason": "raw_price_file_missing", "ticker": ticker, "path": str(path)})
        else:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            versions[ticker] = digest
            if metadata.get("sha256") != digest:
                gaps.append({"reason": "raw_price_checksum_unverified", "ticker": ticker})
        for key in ("source_url", "retrieved_at", "license"):
            if not metadata.get(key):
                gaps.append({"reason": f"{key}_missing", "ticker": ticker})
        if metadata.get("access_cost") != "free" or metadata.get("adjustment_mode") != "raw_ohlcv":
            gaps.append({"reason": "free_raw_source_unverified", "ticker": ticker})
        coverage = metadata.get("actions_coverage", {})
        coverage_start = pd.to_datetime(coverage.get("start"), errors="coerce")
        coverage_end = pd.to_datetime(coverage.get("end"), errors="coerce")
        if (not coverage.get("verified") or not coverage.get("source_url") or
            pd.isna(coverage_start) or pd.isna(coverage_end) or
            coverage_start > pd.Timestamp(start) or coverage_end < pd.Timestamp(end)):
            gaps.append({"reason": "corporate_action_coverage_missing", "ticker": ticker, "start": str(start), "end": str(end)})
    action_path = raw / "actions.csv"
    if not action_path.exists() or manifest.get("actions_sha256") != hashlib.sha256(action_path.read_bytes()).hexdigest():
        gaps.append({"reason": "corporate_action_file_unverified", "path": str(action_path)})
    elif action_path.exists():
        actions = pd.read_csv(action_path)
        required = {"event_id", "ticker", "kind", "date", "value", "source"}
        if not required.issubset(actions) or ("event_id" in actions and actions.event_id.duplicated().any()):
            gaps.append({"reason": "corporate_action_schema_or_duplicate", "path": str(action_path)})
        elif not actions.empty:
            dividends = actions.loc[actions.kind == "dividend"]
            if not dividends.empty and ("ex_date" not in dividends or pd.to_datetime(dividends.ex_date, errors="coerce").isna().any()):
                gaps.append({"reason": "dividend_entitlement_dates_missing", "path": str(action_path)})
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
    published = pd.to_datetime(frame.published_at, utc=True, errors="coerce")
    from core_satellite_alpha import _nyse_calendar
    dates = pd.to_datetime(frame.date).dt.normalize()
    calendar = _nyse_calendar(dates.min().year, dates.max().year)
    # Early-close sessions end before 16:00, so use the exchange's actual close.
    closes = pd.to_datetime(dates.map(lambda date: calendar.session_close(date) if calendar.is_session(date) else pd.NaT), utc=True)
    invalid = published.isna() | closes.isna() | (published > closes) | ~frame.access_cost.eq("free") | ~frame.source_url.fillna("").str.match(r"https?://")
    if invalid.any():
        raise ValueError("Unavailable or future feature/context evidence: " + str(frame.loc[invalid, ["date", "ticker"]].to_dict("records")[:20]))


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
