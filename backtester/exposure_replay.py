"""
Exposure Replay — per-strategy exposure state for live execution
=================================================================
The backtested system's drawdown protection lives INSIDE the strategies
(MA/VIX guard exits, vol scaling, staggered exits, crash brake). The live
executor cannot reproduce that from regime weights alone: blending
weight × max_position_pct floors TQQQ exposure at ~66% even when the
backtest is 100% in cash.

This module closes that gap by replaying each locked strategy through the
SAME Backtester code path used for validation, over the processed data up
to the signal date, and reading out the strategy's current TQQQ exposure
(fraction of its sub-portfolio equity). daily_signal.py writes these as
exposure_a / exposure_b in signal_history.csv; the executors then target

    target_pct = weight_a × exposure_a + weight_b × exposure_b

which equals the blended allocation the backtest actually holds.

Determinism: same input CSVs → same exposures (the replay is the backtest).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from backtester.engine import Backtester
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from config.settings import DATA_PROCESSED_DIR, TQQQ_INCEPTION
from config.strategy_config import STRATEGY_A_CONFIG, STRATEGY_B_CONFIG

logger = logging.getLogger("exposure_replay")

REPLAY_CAPITAL = 10_000  # arbitrary — exposures are fractions, capital cancels out


def build_strategy(cfg: dict) -> LongOnlyGuardV2:
    """Construct a LongOnlyGuardV2 from a strategy_config dict (locked params)."""
    return LongOnlyGuardV2(
        ma_long=cfg["ma_long"],
        vix_exit=cfg["vix_exit"],
        vix_reentry=cfg["vix_reentry"],
        confirm_bars=cfg["confirm_bars"],
        max_position_pct=cfg["max_position_pct"],
        vol_scale=cfg["vol_scale"],
        stagger_exit=cfg["stagger_exit"],
        crash_brake_pct=cfg["crash_brake_pct"],
    )


def load_processed_data() -> tuple:
    """Load the processed CSVs the same way the backtest runners do."""
    tqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/TQQQ_full.csv", index_col=0, parse_dates=True)
    sqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/SQQQ_full.csv", index_col=0, parse_dates=True)
    qqq  = pd.read_csv(f"{DATA_PROCESSED_DIR}/QQQ_full.csv",  index_col=0, parse_dates=True)
    vix  = pd.read_csv(f"{DATA_PROCESSED_DIR}/VIX_full.csv",  index_col=0, parse_dates=True)
    tqqq = tqqq[tqqq.index >= TQQQ_INCEPTION]
    sqqq = sqqq[sqqq.index >= TQQQ_INCEPTION]
    qqq  = qqq[qqq.index   >= TQQQ_INCEPTION]
    return tqqq, sqqq, qqq, vix


def _replay_one(cfg: dict, tqqq, sqqq, qqq, vix) -> dict:
    bt  = Backtester(tqqq, sqqq, qqq, initial_capital=REPLAY_CAPITAL, vix=vix)
    res = bt.run(build_strategy(cfg), close_at_end=False)
    fs  = res["final_state"]
    pos = fs["position"]
    return {
        "exposure":    round(float(fs["exposure_pct"]), 4),
        "in_position": pos is not None,
        "ticker":      pos["ticker"] if pos else None,
        "entry_price": pos["entry_price"] if pos else None,
        "entry_date":  str(pd.Timestamp(pos["entry_date"]).date()) if pos else None,
        "state_date":  str(pd.Timestamp(fs["date"]).date()),
    }


def compute_exposures(
    as_of: Optional[pd.Timestamp] = None,
    data: Optional[tuple] = None,
) -> dict:
    """
    Replay both locked strategies and return their current exposure state.

    Args:
        as_of: include bars up to and including this date (default: all data).
        data:  optional (tqqq, sqqq, qqq, vix) tuple — loads processed CSVs
               when omitted. Used by tests / back-calculation.

    Returns dict:
        exposure_a, exposure_b   — fraction of each sub-portfolio in TQQQ [0, ~1)
        state_date               — last bar the state reflects
        detail_a, detail_b       — position diagnostics for the audit log
    """
    tqqq, sqqq, qqq, vix = data if data is not None else load_processed_data()
    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        tqqq = tqqq[tqqq.index <= as_of]
        sqqq = sqqq[sqqq.index <= as_of]
        qqq  = qqq[qqq.index   <= as_of]
        vix  = vix[vix.index   <= as_of]

    detail_a = _replay_one(STRATEGY_A_CONFIG, tqqq, sqqq, qqq, vix)
    detail_b = _replay_one(STRATEGY_B_CONFIG, tqqq, sqqq, qqq, vix)

    logger.info(
        f"Exposure replay (through {detail_a['state_date']}): "
        f"A={detail_a['exposure']:.1%} ({'in' if detail_a['in_position'] else 'out of'} market)  "
        f"B={detail_b['exposure']:.1%} ({'in' if detail_b['in_position'] else 'out of'} market)"
    )

    return {
        "exposure_a": detail_a["exposure"],
        "exposure_b": detail_b["exposure"],
        "state_date": detail_a["state_date"],
        "detail_a":   detail_a,
        "detail_b":   detail_b,
    }
