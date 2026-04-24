from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PREDICTION_DIR = BASE_DIR / "signals"
REQUIRED_PRED_COLS = {"ticker", "confidence", "conf_threshold", "expected_return", "actionable"}
TRADE_FALLBACK_COLS = {"ticker", "date", "net_pnl"}
DATE_CANDIDATES = ["date", "signal_date", "entry_date", "Unnamed: 0"]


def _read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _find_csv_files() -> list[Path]:
    roots = [PREDICTION_DIR, BASE_DIR]
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if path in seen:
                continue
            seen.add(path)
            out.append(path)
    return sorted(out)


def _coerce_actionable(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "yes", "y"])


def _prepare_date_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    for col in DATE_CANDIDATES:
        if col in df.columns:
            tmp = df.copy()
            tmp[col] = pd.to_datetime(tmp[col], errors="coerce")
            tmp = tmp.dropna(subset=[col]).copy()
            if not tmp.empty:
                tmp = tmp.rename(columns={col: "date"})
                return tmp, "date"
    return df, None


def load_prediction_files() -> tuple[dict[str, pd.DataFrame], list[str]]:
    out: dict[str, pd.DataFrame] = {}
    scanned: list[str] = []

    for path in _find_csv_files():
        scanned.append(str(path))
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if not REQUIRED_PRED_COLS.issubset(df.columns):
            continue

        df, date_col = _prepare_date_column(df)
        if date_col is None or df.empty:
            continue

        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["actionable"] = _coerce_actionable(df["actionable"])
        df["year"] = df["date"].dt.year

        for ticker, sub in df.groupby("ticker"):
            out[ticker] = pd.concat([out.get(ticker, pd.DataFrame()), sub], ignore_index=True)

    return out, scanned


def load_trade_fallback(explicit_trades_file: str | None = None) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    candidate_paths: list[Path] = []

    if explicit_trades_file:
        candidate_paths.append(Path(explicit_trades_file).expanduser().resolve())

    candidate_paths.extend(_find_csv_files())

    seen: set[Path] = set()
    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)

        df = _read_csv(path)
        if df is None:
            continue
        if not TRADE_FALLBACK_COLS.issubset(df.columns):
            continue

        df, date_col = _prepare_date_column(df)
        if date_col is None or df.empty:
            continue

        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["year"] = df["date"].dt.year
        rows.append(df[["ticker", "date", "year", "net_pnl"]].copy())

    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "year", "net_pnl"])
    return pd.concat(rows, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (ticker, year), sub in df.groupby(["ticker", "year"]):
        n_preds = len(sub)
        n_actionable = int(sub["actionable"].sum())

        rows.append(
            {
                "ticker": ticker,
                "year": int(year),
                "predictions": n_preds,
                "actionable": n_actionable,
                "actionable_rate": round(n_actionable / max(n_preds, 1), 4),
                "avg_confidence": round(float(sub["confidence"].mean()), 2),
                "avg_threshold": round(float(sub["conf_threshold"].mean()), 2),
                "avg_expected_return": round(float(sub["expected_return"].mean()), 4),
            }
        )

    return pd.DataFrame(rows).sort_values(["ticker", "year"]).reset_index(drop=True)


def summarize_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for ticker, sub in df.groupby("ticker"):
        n_preds = len(sub)
        n_actionable = int(sub["actionable"].sum())

        rows.append(
            {
                "ticker": ticker,
                "predictions": n_preds,
                "actionable": n_actionable,
                "actionable_rate": round(n_actionable / max(n_preds, 1), 4),
                "avg_confidence": round(float(sub["confidence"].mean()), 2),
                "avg_threshold": round(float(sub["conf_threshold"].mean()), 2),
                "avg_expected_return": round(float(sub["expected_return"].mean()), 4),
                "first_date": str(sub["date"].min().date()),
                "last_date": str(sub["date"].max().date()),
            }
        )

    return pd.DataFrame(rows).sort_values("actionable_rate", ascending=False).reset_index(drop=True)


def summarize_trades_by_year(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ticker, year), sub in df.groupby(["ticker", "year"]):
        rows.append(
            {
                "ticker": ticker,
                "year": int(year),
                "trades": int(len(sub)),
                "win_rate": round(float((sub["net_pnl"] > 0).mean()), 4),
                "avg_pnl": round(float(sub["net_pnl"].mean()), 4),
                "total_pnl": round(float(sub["net_pnl"].sum()), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(["ticker", "year"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-file", default=None, help="Optional explicit path to a trades CSV such as walkforward_long_only_16tickers_trades.csv")
    args = parser.parse_args()

    loaded, scanned = load_prediction_files()

    if loaded:
        combined = pd.concat(loaded.values(), ignore_index=True)

        yearly = summarize(combined)
        ticker_level = summarize_by_ticker(combined)

        yearly_out = PREDICTION_DIR / "prediction_coverage_by_year.csv"
        ticker_out = PREDICTION_DIR / "prediction_coverage_by_ticker.csv"

        yearly.to_csv(yearly_out, index=False)
        ticker_level.to_csv(ticker_out, index=False)

        print("\n=== BY TICKER ===")
        print(ticker_level.to_string(index=False))

        print("\n=== BY YEAR (first 50 rows) ===")
        print(yearly.head(50).to_string(index=False))

        print(f"\nSaved -> {yearly_out}")
        print(f"Saved -> {ticker_out}")
        return

    print("No prediction CSVs found with columns: ticker, confidence, conf_threshold, expected_return, actionable")
    print("Scanned files:")
    for path in scanned[:50]:
        print(f"  - {path}")

    trades = load_trade_fallback(args.trades_file)
    if trades.empty:
        print("\nNo fallback trade CSVs found.")
        print("Use --trades-file with the exact CSV path, for example:")
        print("  python analyze_prediction_coverage.py --trades-file signals/walkforward_long_only_16tickers_trades.csv")
        raise SystemExit("You likely did not save prediction DataFrames during backtest generation, and the trades CSV is not in the searched paths.")

    trade_yearly = summarize_trades_by_year(trades)
    trade_out = PREDICTION_DIR / "trade_coverage_by_year.csv"
    trade_yearly.to_csv(trade_out, index=False)

    print("\nNo raw prediction files were saved, so falling back to trade coverage only.")
    print("This can tell you WHEN trading stopped, but not whether predictions existed and were filtered out.")
    print("\n=== TRADE COVERAGE BY YEAR ===")
    print(trade_yearly.to_string(index=False))
    print(f"\nSaved -> {trade_out}")


if __name__ == "__main__":
    main()