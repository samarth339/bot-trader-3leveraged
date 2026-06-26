"""
AUDIT: as-designed (backtest) vs as-implemented (live executor logic).

(a) As-designed: DualPortfolioBacktester with the LOCKED config from
    config/strategy_config.py (per-strategy exits, vol scaling, etc.)
(b) As-implemented: replicates paper_trade.py / ibkr executor logic:
    target = wa*max_pos_a + wb*max_pos_b, 5% drift gate, 10bps slippage,
    7%-below-last-fill stop (full exit, next-day re-entry), optional 35% DD
    kill switch that FREEZES the portfolio (no trades, position kept).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from backtester.dual_portfolio import DualPortfolioBacktester
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from config.strategy_config import (
    PORTFOLIO_DEFAULTS, STRATEGY_A_CONFIG, STRATEGY_B_CONFIG, RISK_CONFIG,
)
from config.settings import DATA_PROCESSED_DIR, TQQQ_INCEPTION

def load():
    out = {}
    for t in ["TQQQ", "SQQQ", "QQQ", "VIX"]:
        df = pd.read_csv(f"{DATA_PROCESSED_DIR}/{t}_full.csv", index_col=0, parse_dates=True)
        out[t] = df
    for t in ["TQQQ", "SQQQ", "QQQ"]:
        out[t] = out[t][out[t].index >= TQQQ_INCEPTION]
    return out

def mk_strat(cfg):
    return LongOnlyGuardV2(
        ma_long=cfg["ma_long"], vix_exit=cfg["vix_exit"], vix_reentry=cfg["vix_reentry"],
        confirm_bars=cfg["confirm_bars"], max_position_pct=cfg["max_position_pct"],
        vol_scale=cfg["vol_scale"], stagger_exit=cfg["stagger_exit"],
        crash_brake_pct=cfg["crash_brake_pct"],
    )

def metrics(eq: pd.Series, label):
    ret = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = (eq.cummax() - eq) / eq.cummax()
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    print(f"{label:42s} CAGR {cagr*100:6.1f}%  MaxDD {dd.max()*100:5.1f}%  "
          f"Calmar {cagr/dd.max():5.2f}  Sharpe {sharpe:5.2f}  Final ${eq.iloc[-1]:,.0f}")
    return dd

def window(eq, dd, a, b, label):
    sub = eq.loc[a:b]
    if len(sub) < 2:
        return
    print(f"    {label:24s} return {(sub.iloc[-1]/sub.iloc[0]-1)*100:7.1f}%   "
          f"maxDD-in-window {((sub.cummax()-sub)/sub.cummax()).max()*100:5.1f}%")

d = load()
data = (d["TQQQ"], d["SQQQ"], d["QQQ"], d["VIX"])

# ── (a) as-designed, locked config ───────────────────────────────────────────
dp = DualPortfolioBacktester(
    *data,
    strategy_a=mk_strat(STRATEGY_A_CONFIG),
    strategy_b=mk_strat(STRATEGY_B_CONFIG),
    initial_capital=10_000,
    alloc_bull=PORTFOLIO_DEFAULTS["alloc_bull"],
    alloc_mid=PORTFOLIO_DEFAULTS["alloc_mid"],
    alloc_hi_vol=PORTFOLIO_DEFAULTS["alloc_hi_vol"],
    vix_bull=PORTFOLIO_DEFAULTS["vix_bull"],
    vix_hi_vol=PORTFOLIO_DEFAULTS["vix_hi_vol"],
    ma_window=PORTFOLIO_DEFAULTS["ma_window"],
    t1=True, confirm_days=PORTFOLIO_DEFAULTS["confirm_days"],
    vix_smooth=PORTFOLIO_DEFAULTS["vix_smooth"],
)
res = dp.run()
eq_design = res["equity_curve"]["equity"]
regimes = dp.compute_regime_series()

# pct_vs_sma series for dynamic uncertain weights (same as live)
sc, ss, sv = dp._build_signal_inputs()
pct_sma = (sc - ss) / ss * 100

# ── (b) as-implemented executor sim ──────────────────────────────────────────
def run_executor_sim(kill_switch=True, daily_stop=True):
    tq = d["TQQQ"]["close"]
    dates = eq_design.index  # same trading days
    cash, shares = 10_000.0, 0.0
    last_fill, peak, frozen = 0.0, 10_000.0, False
    slip = 0.001
    drift_gate = RISK_CONFIG["alloc_drift_rebalance"]
    rows, n_trades, frozen_date = [], 0, None
    for dt in dates:
        px = float(tq.loc[dt])
        nlv = shares * px + cash
        peak = max(peak, nlv)
        # kill switch: freeze everything at 35% DD (mirrors PaperSafetyGuard)
        if kill_switch and not frozen and peak > 0 and (peak - nlv) / peak >= RISK_CONFIG["max_drawdown_halt"]:
            frozen, frozen_date = True, dt
        if not frozen:
            # daily stop: 7% below last fill → full exit at close - slip
            if daily_stop and shares > 0 and last_fill > 0 and (last_fill - px) / last_fill >= RISK_CONFIG["daily_stop_loss"]:
                fill = px * (1 - slip)
                cash += shares * fill
                shares = 0.0
                last_fill = fill
                n_trades += 1
            else:
                reg = regimes.loc[dt] if dt in regimes.index else "uncertain"
                ps = float(pct_sma.get(dt, np.nan))
                wa, wb = dp._weights(reg, ps)
                target = wa * STRATEGY_A_CONFIG["max_position_pct"] + wb * STRATEGY_B_CONFIG["max_position_pct"]
                cur = shares * px / nlv if nlv > 0 else 0.0
                if abs(target - cur) >= drift_gate:
                    tgt_sh = np.floor(nlv * target / px)
                    delta = tgt_sh - shares
                    if delta != 0:
                        fill = px * (1 + slip) if delta > 0 else px * (1 - slip)
                        cash -= delta * fill
                        shares = tgt_sh
                        last_fill = fill
                        n_trades += 1
        nlv = shares * px + cash
        rows.append((dt, nlv))
    eq = pd.Series(dict(rows)).sort_index()
    return eq, n_trades, frozen_date

print("=" * 100)
dd1 = metrics(eq_design, "(a) AS-DESIGNED (locked config, dual-strat)")
eq_b, nt_b, fz = run_executor_sim(kill_switch=False)
dd2 = metrics(eq_b, f"(b) AS-IMPLEMENTED, no kill switch ({nt_b} trades)")
eq_c, nt_c, fz_c = run_executor_sim(kill_switch=True)
dd3 = metrics(eq_c, f"(c) AS-IMPLEMENTED + 35% DD freeze ({nt_c} trades)")
if fz_c:
    print(f"    >>> kill switch froze portfolio on {fz_c.date()} — held position, never traded again")
eq_d, nt_d, _ = run_executor_sim(kill_switch=False, daily_stop=False)
dd4 = metrics(eq_d, f"(d) AS-IMPLEMENTED, no stop/no kill ({nt_d} trades)")

# ── (e) AS-IMPLEMENTED v2 — exposure-state targets + new guards ──────────────
# Daily target = wa*exp_a + wb*exp_b, where exp_a/exp_b are the per-strategy
# daily TQQQ exposures from the backtest engine (the FIX). Kill switch flattens
# then freezes; crash days block buys only.
def run_executor_v2(permanent_freeze: bool = False):
    # permanent_freeze models the manual-reset DD kill switch. Over a 16-year
    # replay it fires once and halts forever (unrealistic for a multi-year
    # backtest), so default off to isolate the exposure-state targeting effect.
    from backtester.engine import Backtester
    bt_a = Backtester(d["TQQQ"], d["SQQQ"], d["QQQ"], initial_capital=10_000, vix=d["VIX"])
    bt_b = Backtester(d["TQQQ"], d["SQQQ"], d["QQQ"], initial_capital=10_000, vix=d["VIX"])
    ra = bt_a.run(mk_strat(STRATEGY_A_CONFIG))
    rb = bt_b.run(mk_strat(STRATEGY_B_CONFIG))
    exp_a = ra["equity_curve"]["exposure"]
    exp_b = rb["equity_curve"]["exposure"]

    tq = d["TQQQ"]["close"]
    dates = eq_design.index
    cash, shares = 10_000.0, 0.0
    peak, frozen = 10_000.0, False
    slip = 0.001
    drift_gate = RISK_CONFIG["alloc_drift_rebalance"]
    daily_stop = RISK_CONFIG["daily_stop_loss"]
    rows, n_trades = [], 0
    prev_px = None
    for dt in dates:
        px = float(tq.loc[dt])
        nlv = shares * px + cash
        peak = max(peak, nlv)
        # kill switch: FLATTEN then freeze (only when modeling the manual halt)
        if permanent_freeze and not frozen and peak > 0 and (peak - nlv) / peak >= RISK_CONFIG["max_drawdown_halt"]:
            if shares > 0:
                fill = px * (1 - slip); cash += shares * fill; shares = 0.0; n_trades += 1
            frozen = True
        if not frozen:
            reg = regimes.loc[dt] if dt in regimes.index else "uncertain"
            ps = float(pct_sma.get(dt, np.nan))
            wa, wb = dp._weights(reg, ps)
            ea = float(exp_a.get(dt, 0.0)); eb = float(exp_b.get(dt, 0.0))
            target = wa * ea + wb * eb
            cur = shares * px / nlv if nlv > 0 else 0.0
            crash = prev_px is not None and (px - prev_px) / prev_px <= -daily_stop
            tgt_sh = np.floor(nlv * target / px)
            delta = tgt_sh - shares
            # crash day blocks BUYS only; sells always allowed
            if delta > 0 and crash:
                delta = 0
            if abs(target - cur) >= drift_gate and delta != 0:
                fill = px * (1 + slip) if delta > 0 else px * (1 - slip)
                cash -= delta * fill; shares += delta; n_trades += 1
        nlv = shares * px + cash
        rows.append((dt, nlv))
        prev_px = px
    return pd.Series(dict(rows)).sort_index(), n_trades

eq_e, nt_e = run_executor_v2(permanent_freeze=False)
dd5 = metrics(eq_e, f"(e) AS-IMPLEMENTED v2: exposure-state + new guards ({nt_e} trades)")
eq_f, nt_f = run_executor_v2(permanent_freeze=True)
dd6 = metrics(eq_f, f"(f) v2 + manual DD-halt freeze ({nt_f} trades)")

print("\nStress windows:")
for a, b, lab in [("2020-02-19", "2020-03-23", "COVID crash"),
                  ("2021-11-19", "2022-10-13", "2022 rate-hike bear"),
                  ("2025-01-01", "2026-06-01", "recent 18mo")]:
    print(f"  {lab}:")
    window(eq_design, dd1, a, b, "as-designed")
    window(eq_b, dd2, a, b, "as-impl (OLD, max-cap)")
    window(eq_e, dd5, a, b, "as-impl v2 (FIXED)")

# exposure comparison during high_vol days
print("\nHigh-vol regime days: as-implemented holds "
      f"{0.25*STRATEGY_A_CONFIG['max_position_pct'] + 0.75*STRATEGY_B_CONFIG['max_position_pct']:.1%} TQQQ "
      "(strategy exits never applied live)")
