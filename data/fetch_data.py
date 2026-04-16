"""
Download and prepare 42 years of price data.

Strategy:
  - QQQ real data: 1999-03-10 onwards (Yahoo Finance)
  - NDX/^COMP proxy: 1985 onwards for pre-QQQ era
  - Synthetic TQQQ  = daily_return(QQQ) * 3  (with daily vol drag approximation)
  - Synthetic SQQQ  = daily_return(QQQ) * -3
  - Real TQQQ/SQQQ stitched in from 2010-02-11

Usage:
    python data/fetch_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from config.settings import (
    BACKTEST_START, BACKTEST_END, TQQQ_INCEPTION,
    DATA_RAW_DIR, DATA_PROCESSED_DIR
)

Path(DATA_RAW_DIR).mkdir(parents=True, exist_ok=True)
Path(DATA_PROCESSED_DIR).mkdir(parents=True, exist_ok=True)


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    print(f"  Downloading {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance >= 0.2.x may return MultiIndex columns — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


def synthesize_leveraged(qqq: pd.DataFrame, leverage: float) -> pd.DataFrame:
    """
    Approximate a leveraged ETF from QQQ returns.
    Applies daily compounding with a simple expense-ratio drag.
    leverage = +3.0 for TQQQ, -3.0 for SQQQ
    """
    EXPENSE_RATIO_DAILY = 0.0095 / 252   # ~0.95 % annual
    daily_ret = qqq["close"].pct_change().fillna(0)
    leveraged_ret = daily_ret * leverage - EXPENSE_RATIO_DAILY

    # Reconstruct price series starting at 100
    price = 100 * (1 + leveraged_ret).cumprod()
    df = pd.DataFrame({
        "open":  price,
        "high":  price * (1 + daily_ret.abs() * 0.5),
        "low":   price * (1 - daily_ret.abs() * 0.5),
        "close": price,
        "volume": qqq["volume"],
    }, index=qqq.index)
    return df


def build_full_history():
    # ── 1. Download raw data ─────────────────────────────────────────────────
    print("Fetching raw data...")
    qqq  = download_ticker("QQQ",  "1999-03-01", BACKTEST_END)
    ndx  = download_ticker("^NDX", BACKTEST_START, "1999-03-31")   # pre-QQQ proxy
    tqqq = download_ticker("TQQQ", TQQQ_INCEPTION, BACKTEST_END)
    sqqq = download_ticker("SQQQ", TQQQ_INCEPTION, BACKTEST_END)
    spy  = download_ticker("SPY",  "1993-01-29", BACKTEST_END)     # for dual momentum
    vix  = download_ticker("^VIX", BACKTEST_START, BACKTEST_END)   # VIX fear gauge

    # Save raws
    qqq.to_csv(f"{DATA_RAW_DIR}/QQQ.csv")
    ndx.to_csv(f"{DATA_RAW_DIR}/NDX.csv")
    tqqq.to_csv(f"{DATA_RAW_DIR}/TQQQ_real.csv")
    sqqq.to_csv(f"{DATA_RAW_DIR}/SQQQ_real.csv")
    spy.to_csv(f"{DATA_RAW_DIR}/SPY.csv")
    vix.to_csv(f"{DATA_RAW_DIR}/VIX.csv")

    # ── 2. Build unified QQQ-equivalent back to 1985 ─────────────────────────
    print("Building unified QQQ history...")
    # Normalise NDX to QQQ price level at overlap
    overlap_start = "1999-03-10"
    qqq_start_price = qqq.loc[qqq.index >= overlap_start, "close"].iloc[0]
    ndx_end_price   = ndx.loc[ndx.index < overlap_start,  "close"].iloc[-1]
    scale = qqq_start_price / ndx_end_price

    ndx_scaled = ndx[ndx.index < overlap_start].copy()
    for col in ["open", "high", "low", "close"]:
        ndx_scaled[col] *= scale

    qqq_full = pd.concat([ndx_scaled, qqq[qqq.index >= overlap_start]])
    qqq_full = qqq_full[~qqq_full.index.duplicated(keep="last")]
    qqq_full.to_csv(f"{DATA_PROCESSED_DIR}/QQQ_full.csv")

    # ── 3. Synthesize TQQQ/SQQQ for pre-inception period ────────────────────
    print("Synthesizing pre-inception TQQQ / SQQQ...")
    pre_period = qqq_full[qqq_full.index < TQQQ_INCEPTION]
    synth_tqqq = synthesize_leveraged(pre_period, +3.0)
    synth_sqqq = synthesize_leveraged(pre_period, -3.0)

    # Stitch: scale synthetic so it ends at real ETF's starting price
    real_tqqq_start = tqqq["close"].iloc[0]
    real_sqqq_start = sqqq["close"].iloc[0]
    synth_tqqq_end  = synth_tqqq["close"].iloc[-1]
    synth_sqqq_end  = synth_sqqq["close"].iloc[-1]

    for col in ["open", "high", "low", "close"]:
        synth_tqqq[col] *= real_tqqq_start / synth_tqqq_end
        synth_sqqq[col] *= real_sqqq_start / synth_sqqq_end

    tqqq_full = pd.concat([synth_tqqq, tqqq[tqqq.index >= TQQQ_INCEPTION]])
    sqqq_full = pd.concat([synth_sqqq, sqqq[sqqq.index >= TQQQ_INCEPTION]])
    tqqq_full = tqqq_full[~tqqq_full.index.duplicated(keep="last")]
    sqqq_full = sqqq_full[~sqqq_full.index.duplicated(keep="last")]

    tqqq_full.to_csv(f"{DATA_PROCESSED_DIR}/TQQQ_full.csv")
    sqqq_full.to_csv(f"{DATA_PROCESSED_DIR}/SQQQ_full.csv")

    # ── 4. Save SPY and VIX (aligned to trading days, no synthesis needed) ───
    spy.to_csv(f"{DATA_PROCESSED_DIR}/SPY_full.csv")
    vix.to_csv(f"{DATA_PROCESSED_DIR}/VIX_full.csv")

    print(f"\nDone. Data ranges:")
    print(f"  QQQ  full: {qqq_full.index[0].date()} → {qqq_full.index[-1].date()}  ({len(qqq_full)} bars)")
    print(f"  TQQQ full: {tqqq_full.index[0].date()} → {tqqq_full.index[-1].date()}  ({len(tqqq_full)} bars)")
    print(f"  SQQQ full: {sqqq_full.index[0].date()} → {sqqq_full.index[-1].date()}  ({len(sqqq_full)} bars)")
    print(f"  SPY  full: {spy.index[0].date()} → {spy.index[-1].date()}  ({len(spy)} bars)")
    print(f"  VIX  full: {vix.index[0].date()} → {vix.index[-1].date()}  ({len(vix)} bars)")


if __name__ == "__main__":
    build_full_history()
