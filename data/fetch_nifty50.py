"""
Fetch a Nifty 50 OHLCV dataset for the secondary trading-env dataset.

Background
----------
The original Hackathon notebook (TradeMark-2.ipynb) used 3 Indian large-cap
tickers (RELIANCE, TCS, HDFCBANK) for 2020 and computed 8 hand-rolled
features (log_ret, rsi, sma_5, sma_20, vol_norm, volat, Daily_VWAP,
Time_to_Close).

Our codebase already has a robust ``data/preprocess.py`` pipeline that
computes the canonical **7-feature** view (multicollinearity-audited Apr
2026: log_return, sma5_dist, sma20_dist, rsi, norm_volume, volatility,
vwap_dist) with proper rolling semantics. So instead of reproducing the
notebook's 8 hand-rolled features, we just download real Nifty 50 OHLCV
bars and let ``load_and_preprocess`` derive the canonical 7-feature view.
That way the secondary dataset is fully schema-compatible with the
existing SPY pipeline, and the agent sees the same indicator distribution
it was trained on.

The default canonical ticker is RELIANCE.NS — it's the most liquid
Nifty 50 component and has continuous OHLCV history, so its indicator
distributions are well-behaved.

Usage:
    python data/fetch_nifty50.py
    python data/fetch_nifty50.py --ticker TCS.NS --years 7 \
        --output data/nifty50_prices.csv
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def fetch_nifty50_data(
    ticker: str = "RELIANCE.NS",
    years: int = 7,
    output_path: str = "data/nifty50_prices.csv",
) -> pd.DataFrame:
    """Download daily OHLCV for ``ticker`` and write a CSV in the codebase schema."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "yfinance is required. Install it via `pip install yfinance`."
        ) from exc

    end = datetime.today()
    start = end - timedelta(days=years * 365)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]

    if "adj close" in df.columns:
        df["close"] = df["adj close"]

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns from upstream data: {missing}. "
            f"Got: {list(df.columns)}"
        )

    df = df[required].copy()
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[fetch_nifty50] Saved {len(df)} rows to {output_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="RELIANCE.NS")
    parser.add_argument("--years", type=int, default=7)
    parser.add_argument("--output", default="data/nifty50_prices.csv")
    args = parser.parse_args()
    fetch_nifty50_data(
        ticker=args.ticker,
        years=args.years,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
