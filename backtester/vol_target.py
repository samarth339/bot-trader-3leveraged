"""
Volatility-Targeted Exposure Overlay
====================================
Computes a per-day exposure SCALAR in [floor, cap] that scales the blended
TQQQ target down when realized volatility is high and toward `cap` when it is
low, targeting a roughly constant portfolio volatility.

    realized_3x_vol_t = leverage * annualized_std( QQQ daily returns, lookback )   (T-1)
    scalar_t          = clip( target_annual_vol / realized_3x_vol_t, floor, cap )

Applied identically in two places so backtest and live never diverge:
  • backtester/dual_portfolio.py — multiplies each day's blended return
    (daily-rebalanced fraction: portfolio_ret = scalar_t * blended_ret_t).
  • daily_signal.py — multiplies the blended live target
    (target = scalar_t * (wa*exposure_a + wb*exposure_b)).

T-1 guard: QQQ returns are shifted by one bar before the rolling std, so the
scalar for day t uses only data available at t-1. Never uses same-bar data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.strategy_config import VOL_TARGET_CONFIG

TRADING_DAYS = 252


def compute_vol_scalar(
    qqq: pd.DataFrame,
    target_annual_vol: float = None,
    lookback: int = None,
    leverage: float = None,
    floor: float = None,
    cap: float = None,
    t1: bool = True,
) -> pd.Series:
    """
    Return a pd.Series (indexed like qqq) of exposure scalars in [floor, cap].

    Parameters default to VOL_TARGET_CONFIG when omitted. Before enough history
    exists for the rolling window, the scalar is `cap` (no de-risking claimed
    from data we don't have).
    """
    cfg    = VOL_TARGET_CONFIG
    target = cfg["target_annual_vol"] if target_annual_vol is None else target_annual_vol
    lb     = cfg["lookback"]          if lookback          is None else lookback
    lev    = cfg["leverage"]          if leverage          is None else leverage
    flo    = cfg["floor"]             if floor             is None else floor
    ceil   = cfg["cap"]               if cap               is None else cap

    close = qqq["close"].astype(float)
    rets  = close.pct_change()
    if t1:
        rets = rets.shift(1)          # T-1: scalar for day t uses returns through t-1

    realized_qqq = rets.rolling(lb, min_periods=lb).std() * np.sqrt(TRADING_DAYS)
    realized_3x  = realized_qqq * lev

    scalar = target / realized_3x
    scalar = scalar.clip(lower=flo, upper=ceil)
    # Warm-up (insufficient history) → cap (do not de-risk on missing data)
    scalar = scalar.fillna(ceil)
    scalar.name = "vol_scalar"
    return scalar


def latest_scalar(qqq: pd.DataFrame, as_of: pd.Timestamp = None, **kwargs) -> float:
    """
    Scalar for the most recent bar strictly before `as_of` (or the last bar if
    as_of is None). Used by the live signal path.
    """
    s = compute_vol_scalar(qqq, **kwargs)
    if as_of is not None:
        s = s[s.index < pd.Timestamp(as_of)]
    if len(s) == 0:
        return float(kwargs.get("cap") or VOL_TARGET_CONFIG["cap"])
    return float(s.iloc[-1])
