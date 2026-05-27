"""
data_loader.py — Single source of truth for all dashboard data.

Sources (read-only):
  logs/signal_history.csv   — daily regime signals
  data/processed/*.csv      — OHLCV price history (TQQQ, QQQ, VIX, SPY)
  logs/ibkr_state.json      — paper execution state (may not exist yet)
  logs/ibkr_orders.csv      — paper fills (may not exist yet)

Computes:
  Equity curve (post-2010 backtest, normalised to $5 K at inception)
  Shadow equity (2026-03-27 window, normalised to $5 K at shadow start)
  Paper trading overlay (when ibkr_orders.csv exists)
  Rolling Sharpe (7d, 30d), max drawdown curve, daily returns
  VaR 95/99 (historical, current allocation), realised volatility
  Signal outcome annotation (5-day forward TQQQ return per signal)
  Regime-change blocks for chart shading
"""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

LOGS = ROOT / "logs"
DATA = ROOT / "data" / "processed"

TQQQ_INCEPTION = pd.Timestamp("2010-02-11")
SHADOW_START   = pd.Timestamp("2026-03-27")
SEED_CAPITAL   = 5_000.0
RF_DAILY       = 0.05 / 252   # 5 % annual risk-free rate, daily


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class DashboardData:
    # Equity curves (all normalised to $5 K at their respective start)
    backtest_equity: pd.Series                   # 2010 → today
    shadow_equity:   pd.Series                   # 2026-03-27 → today
    paper_equity:    Optional[pd.Series] = None  # paper fills (None until fills exist)

    # Raw price data
    tqqq: pd.DataFrame = field(default_factory=pd.DataFrame)
    qqq:  pd.DataFrame = field(default_factory=pd.DataFrame)
    vix:  pd.Series    = field(default_factory=pd.Series)

    # Signal log with computed outcome column
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Equity curve with regime column (for chart shading)
    equity_with_regime: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Derived time-series
    daily_returns:     pd.Series = field(default_factory=pd.Series)
    drawdown:          pd.Series = field(default_factory=pd.Series)
    rolling_sharpe_7:  pd.Series = field(default_factory=pd.Series)
    rolling_sharpe_30: pd.Series = field(default_factory=pd.Series)

    # Regime change blocks [(start, end, regime), ...]  for vrect shading
    regime_blocks: list = field(default_factory=list)

    # Current state snapshots
    current_signal:  dict = field(default_factory=dict)
    paper_state:     dict = field(default_factory=dict)

    # Scalar summary values (shadow window)
    portfolio_value:  float = SEED_CAPITAL
    total_return_pct: float = 0.0
    max_dd_pct:       float = 0.0
    sharpe_30:        float = 0.0
    var_95:           float = 0.0     # dollar VaR 95 %
    var_99:           float = 0.0     # dollar VaR 99 %
    realized_vol_14:  float = 0.0    # annualised
    realized_vol_30:  float = 0.0
    mtd_return:       float = 0.0
    ytd_return:       float = 0.0
    regime_days:      dict  = field(default_factory=dict)   # {bull:N, uncertain:N, high_vol:N}
    current_alloc:    float = 0.85   # blended TQQQ target pct


# ── Public entry point ─────────────────────────────────────────────────────────

def load() -> DashboardData:
    """Load and compute all dashboard data. Call once at app startup."""

    # ── Price data ─────────────────────────────────────────────────────────────
    tqqq = pd.read_csv(DATA / "TQQQ_full.csv", index_col=0, parse_dates=True)
    sqqq = pd.read_csv(DATA / "SQQQ_full.csv", index_col=0, parse_dates=True)
    qqq  = pd.read_csv(DATA / "QQQ_full.csv",  index_col=0, parse_dates=True)
    vix  = pd.read_csv(DATA / "VIX_full.csv",  index_col=0, parse_dates=True)

    tqqq_close = tqqq["close"]

    # ── Signal history ─────────────────────────────────────────────────────────
    signals = _load_signals(tqqq_close)
    current_signal = signals.iloc[-1].to_dict() if not signals.empty else {}

    # Blended target allocation: A×85% + B×60%
    wa  = float(current_signal.get("weight_a", 0.9))
    wb  = float(current_signal.get("weight_b", 0.1))
    alloc = wa * 0.85 + wb * 0.60

    # ── Run backtester (normalised equity curve + regime column) ───────────────
    ec_full, regime_blocks = _run_backtest(tqqq, sqqq, qqq, vix)

    # Slice to real data only (post TQQQ inception) and normalise to $5 K
    ec_real = ec_full[ec_full.index >= TQQQ_INCEPTION].copy()
    bt_equity = _normalise(ec_real["equity"])

    # Shadow window (subset of backtest)
    ec_shadow  = ec_real[ec_real.index >= SHADOW_START]
    sh_equity  = _normalise(ec_shadow["equity"])

    equity_with_regime = ec_real[["equity", "regime", "drawdown"]].copy()
    equity_with_regime["equity_norm"] = bt_equity.values

    # ── Daily returns & derived series (post-2010) ─────────────────────────────
    daily_ret = bt_equity.pct_change().dropna()
    dd        = _drawdown_series(bt_equity)
    rs7       = _rolling_sharpe(daily_ret, 7)
    rs30      = _rolling_sharpe(daily_ret, 30)

    # ── Paper trading ──────────────────────────────────────────────────────────
    paper_state  = _load_json(LOGS / "ibkr_state.json")
    paper_equity = _build_paper_equity(tqqq_close, paper_state)

    # ── Scalar summary (shadow window) ────────────────────────────────────────
    shadow_value      = float(sh_equity.iloc[-1]) if not sh_equity.empty else SEED_CAPITAL
    total_return_pct  = (shadow_value / SEED_CAPITAL) - 1.0
    shadow_dd         = _drawdown_series(sh_equity)
    max_dd_pct        = float(shadow_dd.min()) if not shadow_dd.empty else 0.0
    sharpe_30_val     = float(rs30.dropna().iloc[-1]) if rs30.dropna().shape[0] > 0 else 0.0

    # VaR: 1-day dollar loss at 95/99 % confidence for current position
    tqqq_ret   = tqqq_close.pct_change().dropna()
    var95_pct, var99_pct = _hist_var(tqqq_ret, lookback=252)
    position_val = shadow_value * alloc
    var95_dollar = position_val * abs(var95_pct)
    var99_dollar = position_val * abs(var99_pct)

    rv14 = float(daily_ret.iloc[-14:].std() * np.sqrt(252)) if len(daily_ret) >= 14 else 0.0
    rv30 = float(daily_ret.iloc[-30:].std() * np.sqrt(252)) if len(daily_ret) >= 30 else 0.0

    # MTD / YTD using shadow equity
    mtd = _period_return(sh_equity, pd.Timestamp.today().replace(day=1))
    ytd = _period_return(sh_equity, pd.Timestamp.today().replace(month=1, day=1))

    # Regime day counts from shadow state
    from config.strategy_config import REGIME_CONFIG  # noqa: F401 – just checking it loads
    regime_days = {
        "bull":     int(signals[signals["regime"] == "bull"].shape[0]),
        "uncertain": int(signals[signals["regime"] == "uncertain"].shape[0]),
        "high_vol": int(signals[signals["regime"] == "high_vol"].shape[0]),
    }

    return DashboardData(
        backtest_equity    = bt_equity,
        shadow_equity      = sh_equity,
        paper_equity       = paper_equity,
        tqqq               = tqqq,
        qqq                = qqq,
        vix                = vix["close"],
        signals            = signals,
        equity_with_regime = equity_with_regime,
        daily_returns      = daily_ret,
        drawdown           = dd,
        rolling_sharpe_7   = rs7,
        rolling_sharpe_30  = rs30,
        regime_blocks      = regime_blocks,
        current_signal     = current_signal,
        paper_state        = paper_state,
        portfolio_value    = shadow_value,
        total_return_pct   = total_return_pct,
        max_dd_pct         = max_dd_pct,
        sharpe_30          = sharpe_30_val,
        var_95             = var95_dollar,
        var_99             = var99_dollar,
        realized_vol_14    = rv14,
        realized_vol_30    = rv30,
        mtd_return         = mtd,
        ytd_return         = ytd,
        regime_days        = regime_days,
        current_alloc      = alloc,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalise(equity: pd.Series, seed: float = SEED_CAPITAL) -> pd.Series:
    first = float(equity.iloc[0])
    return (equity / first * seed) if first != 0 else equity


def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    excess = returns - RF_DAILY
    mu  = excess.rolling(window).mean()
    sig = returns.rolling(window).std()
    # Annualise: multiply by sqrt(252)
    return (mu / sig) * np.sqrt(252)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Returns fraction below rolling peak. 0 = at ATH, -0.3 = 30 % below peak."""
    rolling_max = equity.cummax()
    return (equity / rolling_max) - 1.0


def _hist_var(returns: pd.Series, lookback: int = 252) -> tuple[float, float]:
    """Historical VaR percentiles (negative = loss)."""
    recent = returns.iloc[-lookback:].dropna()
    return float(np.percentile(recent, 5)), float(np.percentile(recent, 1))


def _period_return(equity: pd.Series, start: pd.Timestamp) -> float:
    sub = equity[equity.index >= start]
    if len(sub) < 2:
        return 0.0
    return float(sub.iloc[-1] / sub.iloc[0] - 1.0)


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _regime_blocks(regime_series: pd.Series) -> list[tuple]:
    """
    Collapse consecutive same-regime days into (start, end, regime) blocks
    for use as vrect background shading on the equity chart.
    """
    if regime_series.empty:
        return []
    blocks = []
    cur_regime = regime_series.iloc[0]
    cur_start  = regime_series.index[0]
    for dt, reg in regime_series.items():
        if reg != cur_regime:
            blocks.append((cur_start, dt, cur_regime))
            cur_regime = reg
            cur_start  = dt
    blocks.append((cur_start, regime_series.index[-1], cur_regime))
    return blocks


def _load_signals(tqqq_close: pd.Series) -> pd.DataFrame:
    signals = pd.read_csv(
        LOGS / "signal_history.csv",
        parse_dates=["as_of_date", "signal_date"],
    )
    signals = signals.sort_values("as_of_date").reset_index(drop=True)

    # 5-day forward TQQQ return for each signal date
    fwd = tqqq_close.pct_change().shift(-1).rolling(5).sum()
    signals["tqqq_5d_fwd"] = signals["as_of_date"].map(
        lambda d: float(fwd.get(d, np.nan))
    )

    def _outcome(row: pd.Series) -> str:
        r = row["tqqq_5d_fwd"]
        if pd.isna(r):
            return "—"
        if row["regime"] == "bull":
            return "✓" if r > 0 else "✗"
        if row["regime"] == "high_vol":
            return "✓" if r < 0 else "✗"
        return "~"   # uncertain: no directional call

    signals["outcome"] = signals.apply(_outcome, axis=1)
    return signals


def _run_backtest(tqqq, sqqq, qqq, vix):
    """Run DualPortfolioBacktester. Returns (equity_curve_df, regime_blocks)."""
    try:
        from backtester.dual_portfolio import DualPortfolioBacktester
        from strategies.long_only_guard_v2 import LongOnlyGuardV2
        from config.strategy_config import (
            STRATEGY_A_CONFIG, STRATEGY_B_CONFIG, PORTFOLIO_DEFAULTS,
        )

        sa = LongOnlyGuardV2(**{k: v for k, v in STRATEGY_A_CONFIG.items() if k != "name"})
        sb = LongOnlyGuardV2(**{k: v for k, v in STRATEGY_B_CONFIG.items() if k != "name"})

        dp = DualPortfolioBacktester(
            tqqq, sqqq, qqq, vix,
            strategy_a=sa, strategy_b=sb,
            initial_capital=SEED_CAPITAL,
            **PORTFOLIO_DEFAULTS,
        )
        result = dp.run()
        ec = result["equity_curve"]
        blocks = _regime_blocks(ec["regime"])
        return ec, blocks

    except Exception as exc:
        print(f"[data_loader] Backtester unavailable ({exc}). Falling back to signal reconstruction.")
        return _reconstruct_from_signals(tqqq), []


def _reconstruct_from_signals(tqqq: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal fallback: reconstruct an equity curve from signal_history.csv
    and TQQQ daily returns. Less accurate than the full backtester but ensures
    the dashboard always renders even if backtester imports fail.
    """
    signals = pd.read_csv(
        LOGS / "signal_history.csv", parse_dates=["as_of_date"]
    ).set_index("as_of_date").sort_index()

    tqqq_ret = tqqq["close"].pct_change()
    val = SEED_CAPITAL
    rows = {}
    for dt, row in signals.iterrows():
        wa, wb = float(row["weight_a"]), float(row["weight_b"])
        alloc = wa * 0.85 + wb * 0.60
        ret   = float(tqqq_ret.get(dt, 0.0))
        val   = val * (1.0 + alloc * ret)
        rows[dt] = {"equity": val, "regime": row["regime"], "drawdown": 0.0}

    return pd.DataFrame.from_dict(rows, orient="index")


def _build_paper_equity(
    tqqq_close: pd.Series,
    paper_state: dict,
) -> Optional[pd.Series]:
    """
    Build a mark-to-market equity series from ibkr_orders.csv.
    Returns None if no paper trading data exists yet.
    """
    orders_path = LOGS / "ibkr_orders.csv"
    if not orders_path.exists():
        return None

    try:
        orders = pd.read_csv(orders_path, parse_dates=["date"]).sort_values("date")
        if orders.empty:
            return None

        # Expected columns: date, shares, fill_price, direction (buy/sell)
        # Walk forward through fills, computing mark-to-market daily
        dates  = pd.date_range(orders["date"].min(), tqqq_close.index[-1], freq="B")
        shares = 0
        cash   = float(paper_state.get("last_fill_price", 0)) * 0  # start from first fill
        equity_vals = {}

        # Simple: track shares and value them at daily close
        for fill_date, group in orders.groupby("date"):
            for _, fill in group.iterrows():
                delta = int(fill.get("shares", 0))
                price = float(fill.get("fill_price", tqqq_close.get(fill_date, 0)))
                if str(fill.get("direction", "buy")).lower() == "sell":
                    cash   += delta * price
                    shares -= delta
                else:
                    cash   -= delta * price
                    shares += delta

        if shares == 0 and cash == 0:
            return None

        start_date = orders["date"].min()
        for dt in tqqq_close.index[tqqq_close.index >= start_date]:
            price = float(tqqq_close.get(dt, 0))
            equity_vals[dt] = cash + shares * price

        return pd.Series(equity_vals) if equity_vals else None

    except Exception:
        return None
