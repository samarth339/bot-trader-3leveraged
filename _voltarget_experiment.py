"""
Honest A/B: locked baseline vs volatility-targeted exposure overlay.
Reports train / val / OOS / full for each target vol. No cherry-picking:
every period and every setting is printed. Selection (if any) must be
justified by OUT-OF-SAMPLE robustness, not the in-sample max.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from backtester.dual_portfolio import DualPortfolioBacktester
from backtester.exposure_replay import build_strategy
from config.strategy_config import PORTFOLIO_DEFAULTS, STRATEGY_A_CONFIG, STRATEGY_B_CONFIG
from config.settings import DATA_PROCESSED_DIR, TQQQ_INCEPTION

d = {t: pd.read_csv(f"{DATA_PROCESSED_DIR}/{t}_full.csv", index_col=0, parse_dates=True)
     for t in ["TQQQ","SQQQ","QQQ","VIX"]}
for t in ["TQQQ","SQQQ","QQQ"]:
    d[t] = d[t][d[t].index >= TQQQ_INCEPTION]

def slice_(a, b):
    out = {}
    for k, df in d.items():
        out[k] = df[(df.index >= a) & (df.index <= b)]
    return out

def run(data, vt=False, target=None):
    params = {"target_annual_vol": target} if target is not None else None
    dp = DualPortfolioBacktester(
        data["TQQQ"], data["SQQQ"], data["QQQ"], data["VIX"],
        strategy_a=build_strategy(STRATEGY_A_CONFIG),
        strategy_b=build_strategy(STRATEGY_B_CONFIG),
        initial_capital=10_000,
        alloc_bull=PORTFOLIO_DEFAULTS["alloc_bull"], alloc_mid=PORTFOLIO_DEFAULTS["alloc_mid"],
        alloc_hi_vol=PORTFOLIO_DEFAULTS["alloc_hi_vol"], vix_bull=PORTFOLIO_DEFAULTS["vix_bull"],
        vix_hi_vol=PORTFOLIO_DEFAULTS["vix_hi_vol"], ma_window=PORTFOLIO_DEFAULTS["ma_window"],
        t1=True, confirm_days=PORTFOLIO_DEFAULTS["confirm_days"], vix_smooth=PORTFOLIO_DEFAULTS["vix_smooth"],
        vol_target_override=vt, vol_target_params=params,
    )
    m = dp.run()["metrics"]
    return m

PERIODS = [
    ("TRAIN 2010-2022", "2010-02-11", "2022-01-01"),
    ("VAL   2022-2025", "2022-01-01", "2025-01-01"),
    ("OOS   2019-2025", "2019-01-01", "2025-01-01"),
    ("FULL  2010-2026", "2010-02-11", "2026-12-31"),
]
TARGETS = [None, 0.45, 0.55, 0.65]   # None = baseline (overlay off)

def fmt(m):
    return (f"CAGR {m['cagr']*100:5.1f}%  MaxDD {m['max_drawdown']*100:4.1f}%  "
            f"Calmar {m['calmar']:.2f}  Sharpe {m['sharpe']:.2f}  Final ${m['final_equity']:>10,.0f}")

for name, a, b in PERIODS:
    data = slice_(a, b)
    print(f"\n{'='*104}\n{name}")
    for tgt in TARGETS:
        label = "BASELINE (overlay OFF)     " if tgt is None else f"vol-target {int(tgt*100)}% ann     "
        m = run(data, vt=(tgt is not None), target=tgt)
        print(f"  {label} {fmt(m)}")
