"""
Fine-Tuning & Micro-Optimization Study
======================================
Locked production config:  T-1  |  VIX smooth=5  |  75/25 bull  |  50/50 mid  |  30/70 hi-vol
                           VIX thresholds: bull=18  hi-vol=25  |  MA window=150

Methodology
-----------
• Strategy A (Best-Calmar) and Strategy B (Near-Miss) equity curves are FIXED.
  Their trading signals never change between sweeps.
• ALL parameter sweeps only touch the *blending layer*:
    VIX thresholds · smoothing window · allocation splits · MA window
• This lets us cache returns once and sweep 1,000+ combos in seconds.

Parts
-----
A : VIX threshold sweep            (vix_bull, vix_hi_vol)
B : VIX smoothing window sweep     (3 – 10 days)
C : Allocation micro-tuning        (bull / mid / hi-vol splits)
D : MA window sweep                (120 – 210 days)
E : Execution & slippage analysis  (0 – 50 bps)
F : Stability / plateau analysis   (neighbourhood density)
G : Final verdict & locked params

Usage:  python fine_tune.py [--no-chart]
"""

import sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")

# Resolve project root (scripts/dev/ → scripts/ → root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

NO_CHART = "--no-chart" in sys.argv

from backtester.engine import Backtester
from strategies.long_only_guard_v2 import LongOnlyGuardV2
from config.settings import DATA_PROCESSED_DIR, TQQQ_INCEPTION, INITIAL_CAPITAL


# ══════════════════════════════════════════════════════════════════════════════
#  LOCKED BASELINE (never changes — reference point for all deltas)
# ══════════════════════════════════════════════════════════════════════════════
LOCKED = dict(
    vix_bull      = 18.0,
    vix_hi_vol    = 25.0,
    ma_window     = 150,
    vix_smooth    = 5,
    alloc_bull    = (0.75, 0.25),
    alloc_mid     = (0.50, 0.50),
    alloc_hi_vol  = (0.30, 0.70),
    t1            = True,
    confirm_days  = 1,
)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADER
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    tqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/TQQQ_full.csv", index_col=0, parse_dates=True)
    sqqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/SQQQ_full.csv", index_col=0, parse_dates=True)
    qqq  = pd.read_csv(f"{DATA_PROCESSED_DIR}/QQQ_full.csv",  index_col=0, parse_dates=True)
    vix  = pd.read_csv(f"{DATA_PROCESSED_DIR}/VIX_full.csv",  index_col=0, parse_dates=True)
    for df in (tqqq, sqqq, qqq, vix):
        df.index = pd.to_datetime(df.index)
    tqqq = tqqq[tqqq.index >= TQQQ_INCEPTION]
    sqqq = sqqq[sqqq.index >= TQQQ_INCEPTION]
    qqq  = qqq[qqq.index   >= TQQQ_INCEPTION]
    return tqqq, sqqq, qqq, vix


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY INSTANCES (fixed — never changed between sweeps)
# ══════════════════════════════════════════════════════════════════════════════
def make_strategies():
    strat_a = LongOnlyGuardV2(
        ma_long=200, vix_exit=25, vix_reentry=22, confirm_bars=2,
        max_position_pct=0.90, vol_scale=False, stagger_exit=True, crash_brake_pct=0.0,
    )
    strat_b = LongOnlyGuardV2(
        ma_long=150, vix_exit=28, vix_reentry=22, confirm_bars=4,
        max_position_pct=0.70, vol_scale=False, stagger_exit=True, crash_brake_pct=0.30,
    )
    return strat_a, strat_b


# ══════════════════════════════════════════════════════════════════════════════
#  CACHE STRATEGY RETURNS (run once, reuse for all sweeps)
# ══════════════════════════════════════════════════════════════════════════════
def cache_returns(tqqq, sqqq, qqq, vix, strat_a, strat_b, slippage_pct=0.001):
    """Run both strategies independently; return aligned daily-return Series."""
    bt_a = Backtester(tqqq, sqqq, qqq, initial_capital=INITIAL_CAPITAL,
                      vix=vix, slippage_pct=slippage_pct)
    bt_b = Backtester(tqqq, sqqq, qqq, initial_capital=INITIAL_CAPITAL,
                      vix=vix, slippage_pct=slippage_pct)
    res_a = bt_a.run(strat_a)
    res_b = bt_b.run(strat_b)

    ec_a = res_a["equity_curve"]["equity"]
    ec_b = res_b["equity_curve"]["equity"]
    common = ec_a.index.intersection(ec_b.index)
    ec_a, ec_b = ec_a.loc[common], ec_b.loc[common]

    ret_a = ec_a.pct_change().fillna(0).values
    ret_b = ec_b.pct_change().fillna(0).values
    n_trades_a = len(res_a["trades"])
    n_trades_b = len(res_b["trades"])
    return ret_a, ret_b, common, n_trades_a, n_trades_b


# ══════════════════════════════════════════════════════════════════════════════
#  FAST BLENDER  (no strategy re-run — just re-weight cached returns)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_regime_array(common, qqq_close, vix_close,
                          vix_bull, vix_hi_vol, ma_window,
                          vix_smooth, t1, confirm_days):
    """Returns numpy array of regime ints: 0=uncertain, 1=bull, 2=high_vol."""
    # Smooth VIX
    if vix_smooth > 1:
        vix_s = vix_close.rolling(vix_smooth, min_periods=1).mean()
    else:
        vix_s = vix_close

    # Pre-compute rolling MA for every QQQ date (vectorised)
    sma = qqq_close.rolling(ma_window, min_periods=ma_window).mean()

    # Map common dates to raw regime
    raw = np.zeros(len(common), dtype=np.int8)   # 0 = uncertain
    for i, date in enumerate(common):
        if date not in qqq_close.index:
            continue
        sma_val = sma.get(date, np.nan)
        if np.isnan(sma_val):
            continue
        price   = float(qqq_close.loc[date])
        vix_val = float(vix_s.loc[date]) if date in vix_s.index else 20.0
        if price < sma_val or vix_val >= vix_hi_vol:
            raw[i] = 2   # high_vol
        elif price > sma_val and vix_val < vix_bull:
            raw[i] = 1   # bull
        # else 0 = uncertain

    # Confirmation inertia
    if confirm_days > 1:
        confirmed  = raw.copy()
        current    = raw[0]
        pending    = raw[0]
        cnt        = 0
        for i in range(len(raw)):
            if raw[i] == pending:
                cnt += 1
            else:
                pending = raw[i]
                cnt     = 1
            if cnt >= confirm_days:
                current = pending
            confirmed[i] = current
        raw = confirmed

    # T-1 shift
    if t1:
        raw = np.roll(raw, 1)
        raw[0] = 0   # uncertain

    return raw


def blend(ret_a, ret_b, regime_arr, alloc_bull, alloc_mid, alloc_hi_vol):
    """Fast numpy blend of return streams given a regime array."""
    wa = np.where(regime_arr == 1, alloc_bull[0],
         np.where(regime_arr == 2, alloc_hi_vol[0], alloc_mid[0]))
    wb = 1.0 - wa
    blended = wa * ret_a + wb * ret_b
    equity  = INITIAL_CAPITAL * np.cumprod(1.0 + blended)
    return equity


def metrics_from_equity(equity, common):
    """Compute CAGR, max DD, Calmar, Sharpe from equity array."""
    returns = np.diff(equity) / equity[:-1]
    peak    = np.maximum.accumulate(equity)
    dd      = (peak - equity) / peak
    max_dd  = dd.max()
    n_years = (common[-1] - common[0]).days / 365.25
    total_r = equity[-1] / INITIAL_CAPITAL - 1
    cagr    = (1 + total_r) ** (1 / n_years) - 1 if n_years > 0 else 0
    calmar  = cagr / max_dd if max_dd > 0 else 0
    sharpe  = (returns.mean() / returns.std() * np.sqrt(252)
               if returns.std() > 0 else 0)
    return dict(cagr=round(cagr*100,2), max_dd=round(max_dd*100,2),
                calmar=round(calmar,3), sharpe=round(sharpe,3),
                final=round(equity[-1],0))


def run_combo(ret_a, ret_b, common, qqq_close, vix_close,
              vix_bull=18, vix_hi_vol=25, ma_window=150,
              alloc_bull=(0.75,0.25), alloc_mid=(0.50,0.50),
              alloc_hi_vol=(0.30,0.70), vix_smooth=5, t1=True,
              confirm_days=1):
    reg = _compute_regime_array(common, qqq_close, vix_close,
                                vix_bull, vix_hi_vol, ma_window,
                                vix_smooth, t1, confirm_days)
    eq  = blend(ret_a, ret_b, reg, alloc_bull, alloc_mid, alloc_hi_vol)
    m   = metrics_from_equity(eq, common)
    # Regime split
    n   = len(reg)
    m["pct_bull"]    = round((reg == 1).sum() / n * 100, 1)
    m["pct_hiVol"]   = round((reg == 2).sum() / n * 100, 1)
    m["pct_uncert"]  = round((reg == 0).sum() / n * 100, 1)
    m["n_flips"]     = int(np.diff(reg).astype(bool).sum())
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════════
W = 110

def divider(char="═"): return char * W
def section(title, subtitle=""):
    print(f"\n{divider()}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(divider())

def row(label, m, base=None, width=46):
    d_cagr = f"({m['cagr']-base['cagr']:+.1f})" if base else ""
    d_dd   = f"({m['max_dd']-base['max_dd']:+.1f})" if base else ""
    d_cal  = f"({m['calmar']-base['calmar']:+.3f})" if base else ""
    star   = " ★" if base and m['calmar'] > base['calmar'] + 0.005 else ""
    print(f"  {label:<{width}} "
          f"CAGR {m['cagr']:>5.1f}% {d_cagr:<8} "
          f"DD {m['max_dd']:>5.1f}% {d_dd:<8} "
          f"Calmar {m['calmar']:>5.3f} {d_cal:<10} "
          f"Sharpe {m['sharpe']:>5.3f}  "
          f"${m['final']:>11,.0f}{star}")

def grid_row(label, m):
    bar = "█" * int(m['calmar'] * 10)
    print(f"  {label:<38} "
          f"{m['cagr']:>5.1f}%  {m['max_dd']:>5.1f}%  {m['calmar']:>5.3f}  "
          f"{bar}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading data and running strategy A & B...")
    tqqq, sqqq, qqq, vix = load_data()
    strat_a, strat_b = make_strategies()
    ret_a, ret_b, common, n_trades_a, n_trades_b = \
        cache_returns(tqqq, sqqq, qqq, vix, strat_a, strat_b)
    qqq_close = qqq["close"]
    vix_close = vix["close"]
    print(f"  Period: {common[0].date()} → {common[-1].date()}  ({len(common)} bars)")
    print(f"  Strategy A trades: {n_trades_a}  |  Strategy B trades: {n_trades_b}")

    def rc(**kwargs):
        p = {**LOCKED, **kwargs}
        return run_combo(ret_a, ret_b, common, qqq_close, vix_close, **p)

    # ── Locked baseline ────────────────────────────────────────────────────────
    BASE = rc()
    print(f"\n{'─'*W}")
    print(f"  LOCKED BASELINE ►  "
          f"CAGR {BASE['cagr']}%  |  DD {BASE['max_dd']}%  |  "
          f"Calmar {BASE['calmar']}  |  Sharpe {BASE['sharpe']}  |  "
          f"${BASE['final']:,.0f}")
    print(f"  Regime split:  Bull {BASE['pct_bull']}%  "
          f"Uncertain {BASE['pct_uncert']}%  "
          f"High-Vol {BASE['pct_hiVol']}%  "
          f"Flips: {BASE['n_flips']}")
    print(f"{'─'*W}")

    all_results = []   # collect ALL combos for stability analysis

    # ══════════════════════════════════════════════════════════════════════════
    #  PART A — VIX THRESHOLD SWEEP
    # ══════════════════════════════════════════════════════════════════════════
    section("PART A — VIX THRESHOLD SWEEP",
            f"Fixed: vix_smooth={LOCKED['vix_smooth']}  alloc_bull={LOCKED['alloc_bull']}  "
            f"ma_window={LOCKED['ma_window']}")

    # A1: vix_bull sweep (vix_hi_vol fixed)
    print("\n  A1 › vix_bull threshold  (vix_hi_vol fixed = 25)")
    print(f"  {'Label':<46} {'CAGR':>9} {'':7} {'DD':>7} {'':7} {'Calmar':>9} {'':10} {'Sharpe':>8}  {'Final':>13}")
    print(f"  {'─'*46} {'─'*9} {'─'*7} {'─'*7} {'─'*7} {'─'*9} {'─'*10} {'─'*8}  {'─'*13}")
    for vb in [14, 15, 16, 17, 18, 19, 20, 21, 22]:
        m = rc(vix_bull=vb)
        row(f"vix_bull={vb}  {'◄ locked' if vb==18 else ''}", m, BASE)
        all_results.append({"param": "vix_bull", "value": vb, **m})

    # A2: vix_hi_vol sweep (vix_bull fixed)
    print("\n  A2 › vix_hi_vol threshold  (vix_bull fixed = 18)")
    print(f"  {'Label':<46} {'CAGR':>9} {'':7} {'DD':>7} {'':7} {'Calmar':>9} {'':10} {'Sharpe':>8}  {'Final':>13}")
    print(f"  {'─'*46} {'─'*9} {'─'*7} {'─'*7} {'─'*7} {'─'*9} {'─'*10} {'─'*8}  {'─'*13}")
    for vh in [20, 21, 22, 23, 24, 25, 26, 27, 28, 30]:
        m = rc(vix_hi_vol=vh)
        row(f"vix_hi_vol={vh}  {'◄ locked' if vh==25 else ''}", m, BASE)
        all_results.append({"param": "vix_hi_vol", "value": vh, **m})

    # A3: 2D grid  vix_bull × vix_hi_vol
    print("\n  A3 › Grid: vix_bull (rows) × vix_hi_vol (cols)  —  Calmar values")
    vb_vals  = [15, 16, 17, 18, 19, 20, 21]
    vh_vals  = [22, 23, 24, 25, 26, 27, 28]
    col_lbl = "vix_bull/hi_vol"
    header = f"  {col_lbl:>18}" + "".join(f"  {v:>6}" for v in vh_vals)
    print(header)
    print(f"  {'─'*18}" + "─"*8*len(vh_vals))
    best_a3_calmar, best_a3_cfg = 0, {}
    for vb in vb_vals:
        vals_str = ""
        for vh in vh_vals:
            if vh <= vb:
                vals_str += f"  {'N/A':>6}"
                continue
            m = rc(vix_bull=vb, vix_hi_vol=vh)
            mark = " ◄" if (vb == 18 and vh == 25) else ""
            vals_str += f"  {m['calmar']:>5.3f}{mark[0] if mark else ' '}"
            if m['calmar'] > best_a3_calmar:
                best_a3_calmar = m['calmar']
                best_a3_cfg = {"vix_bull": vb, "vix_hi_vol": vh, **m}
        locked = " ◄" if vb == 18 else ""
        print(f"  {f'vix_bull={vb}':>18}{locked}{vals_str}")
    print(f"\n  Grid best: vix_bull={best_a3_cfg['vix_bull']}  "
          f"vix_hi_vol={best_a3_cfg['vix_hi_vol']}  "
          f"Calmar={best_a3_calmar:.3f}  "
          f"(vs locked {BASE['calmar']:.3f}, Δ={best_a3_calmar-BASE['calmar']:+.3f})")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART B — VIX SMOOTHING WINDOW
    # ══════════════════════════════════════════════════════════════════════════
    section("PART B — VIX SMOOTHING WINDOW",
            "How sensitive is the system to the VIX noise-reduction window?")
    print(f"  {'Label':<46} {'CAGR':>9} {'':7} {'DD':>7} {'':7} {'Calmar':>9} {'':10} {'Sharpe':>8}  {'Flips':>6}")
    print(f"  {'─'*46} {'─'*9} {'─'*7} {'─'*7} {'─'*7} {'─'*9} {'─'*10} {'─'*8}  {'─'*6}")
    for ws in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]:
        m = rc(vix_smooth=ws)
        lbl = f"vix_smooth={ws:>2}d  {'◄ locked' if ws==5 else ''}"
        row(lbl, m, BASE)
        print(f"  {'':<46} {'':>9} {'':7} {'':>7} {'':7} {'':>9} {'':10} {'':>8}  {m['n_flips']:>6} flips", end="\r")
        all_results.append({"param": "vix_smooth", "value": ws, **m})

    # ══════════════════════════════════════════════════════════════════════════
    #  PART C — ALLOCATION MICRO-TUNING
    # ══════════════════════════════════════════════════════════════════════════
    section("PART C — ALLOCATION MICRO-TUNING")

    # C1: Bull regime split
    print("  C1 › Bull regime split  (mid=50/50, hi-vol=30/70 fixed)")
    print(f"  {'Label':<46} {'CAGR':>9} {'':7} {'DD':>7} {'':7} {'Calmar':>9} {'':10} {'Sharpe':>8}  {'Final':>13}")
    print(f"  {'─'*46} {'─'*9} {'─'*7} {'─'*7} {'─'*7} {'─'*9} {'─'*10} {'─'*8}  {'─'*13}")
    best_bull_calmar, best_bull_cfg = 0, {}
    for wa in [0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]:
        m = rc(alloc_bull=(wa, 1-wa))
        lbl = f"bull={int(wa*100)}/{int((1-wa)*100)}  {'◄ locked' if wa==0.75 else ''}"
        row(lbl, m, BASE)
        all_results.append({"param": "alloc_bull_wa", "value": wa, **m})
        if m['calmar'] > best_bull_calmar:
            best_bull_calmar, best_bull_cfg = m['calmar'], {"wa": wa, **m}

    # C2: Mid regime split
    print("\n  C2 › Mid regime split  (bull=75/25, hi-vol=30/70 fixed)")
    print(f"  {'Label':<46} {'CAGR':>9} {'':7} {'DD':>7} {'':7} {'Calmar':>9} {'':10} {'Sharpe':>8}  {'Final':>13}")
    print(f"  {'─'*46} {'─'*9} {'─'*7} {'─'*7} {'─'*7} {'─'*9} {'─'*10} {'─'*8}  {'─'*13}")
    for wa in [0.40, 0.43, 0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]:
        m = rc(alloc_mid=(wa, 1-wa))
        lbl = f"mid={int(wa*100)}/{int((1-wa)*100)}  {'◄ locked' if wa==0.50 else ''}"
        row(lbl, m, BASE)
        all_results.append({"param": "alloc_mid_wa", "value": wa, **m})

    # C3: Hi-vol regime split
    print("\n  C3 › Hi-vol regime split  (bull=75/25, mid=50/50 fixed)")
    print(f"  {'Label':<46} {'CAGR':>9} {'':7} {'DD':>7} {'':7} {'Calmar':>9} {'':10} {'Sharpe':>8}  {'Final':>13}")
    print(f"  {'─'*46} {'─'*9} {'─'*7} {'─'*7} {'─'*7} {'─'*9} {'─'*10} {'─'*8}  {'─'*13}")
    for wa in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        m = rc(alloc_hi_vol=(wa, 1-wa))
        lbl = f"hi-vol={int(wa*100)}/{int((1-wa)*100)}  {'◄ locked' if wa==0.30 else ''}"
        row(lbl, m, BASE)
        all_results.append({"param": "alloc_hv_wa", "value": wa, **m})

    # ══════════════════════════════════════════════════════════════════════════
    #  PART D — MA WINDOW SWEEP
    # ══════════════════════════════════════════════════════════════════════════
    section("PART D — MA WINDOW SWEEP",
            "Does a different regime MA window help?")
    print(f"  {'Label':<46} {'CAGR':>9} {'':7} {'DD':>7} {'':7} {'Calmar':>9} {'':10} {'Sharpe':>8}  {'Bull%':>7}")
    print(f"  {'─'*46} {'─'*9} {'─'*7} {'─'*7} {'─'*7} {'─'*9} {'─'*10} {'─'*8}  {'─'*7}")
    for mw in [100, 120, 130, 140, 150, 160, 170, 180, 200, 220, 250]:
        m = rc(ma_window=mw)
        lbl = f"ma_window={mw}  {'◄ locked' if mw==150 else ''}"
        row(lbl, m, BASE)
        print(f"                                                  "
              f"bull={m['pct_bull']}%")
        all_results.append({"param": "ma_window", "value": mw, **m})

    # ══════════════════════════════════════════════════════════════════════════
    #  PART E — EXECUTION & SLIPPAGE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    section("PART E — EXECUTION & SLIPPAGE ANALYSIS",
            "Strategy A makes ~{na} trades, B makes ~{nb} trades over 15 years."
            .format(na=n_trades_a, nb=n_trades_b))

    print(f"  Strategy A: {n_trades_a} trades over {(common[-1]-common[0]).days/365.25:.1f} years "
          f"= {n_trades_a / ((common[-1]-common[0]).days/365.25):.1f} trades/year")
    print(f"  Strategy B: {n_trades_b} trades over {(common[-1]-common[0]).days/365.25:.1f} years "
          f"= {n_trades_b / ((common[-1]-common[0]).days/365.25):.1f} trades/year")
    print(f"\n  Slippage is applied at the STRATEGY level (each entry/exit).")
    print(f"  Current assumption: 10 bps (0.1%) per fill.")
    print(f"  Additional portfolio rebalancing (regime shifts) at 0 bps (allocations, not actual trades).\n")

    slippage_results = {}
    print(f"  {'Slippage':>12}  {'A trades':>8}  {'B trades':>8}  "
          f"{'Drag/yr A':>10}  {'Drag/yr B':>10}  "
          f"{'CAGR':>7}  {'DD':>7}  {'Calmar':>7}  {'Final':>13}")
    print(f"  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  "
          f"{'─'*7}  {'─'*7}  {'─'*7}  {'─'*13}")

    n_years = (common[-1] - common[0]).days / 365.25
    for slip_bps in [0, 5, 10, 20, 30, 50]:
        slip_pct = slip_bps / 10_000
        ra, rb, _, nta, ntb = cache_returns(tqqq, sqqq, qqq, vix,
                                            make_strategies()[0], make_strategies()[1],
                                            slippage_pct=slip_pct)
        reg = _compute_regime_array(common, qqq_close, vix_close,
                                    **{k: LOCKED[k] for k in
                                       ['vix_bull','vix_hi_vol','ma_window',
                                        'vix_smooth','t1','confirm_days']})
        eq  = blend(ra, rb, reg, LOCKED['alloc_bull'], LOCKED['alloc_mid'], LOCKED['alloc_hi_vol'])
        m   = metrics_from_equity(eq, common)
        drag_a = slip_bps * nta / n_years / 100
        drag_b = slip_bps * ntb / n_years / 100
        mark   = " ◄ locked" if slip_bps == 10 else ""
        print(f"  {slip_bps:>9} bps{mark:<12}  {nta:>8}  {ntb:>8}  "
              f"{drag_a:>9.2f}%  {drag_b:>9.2f}%  "
              f"{m['cagr']:>6.1f}%  {m['max_dd']:>6.1f}%  "
              f"{m['calmar']:>7.3f}  ${m['final']:>12,.0f}")
        slippage_results[slip_bps] = m

    print(f"\n  Key takeaway: At 50 bps slippage, CAGR impact = "
          f"{slippage_results[50]['cagr'] - slippage_results[0]['cagr']:+.1f}% "
          f"vs zero-slippage baseline.")

    # Regime flip cost (portfolio rebalancing at alloc level)
    print(f"\n  Regime flips in locked config: {BASE['n_flips']} over {n_years:.1f} years "
          f"= {BASE['n_flips']/n_years:.1f} flips/year")
    print(f"  Each flip shifts ~25% of capital (bull↔uncertain) or ~45% (bull↔hi-vol).")
    print(f"  At 5 bps ETF spread: ~{BASE['n_flips']/n_years * 0.35 * 0.05:.2f}% annual drag from regime rebalancing.")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART F — STABILITY / PLATEAU ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    section("PART F — STABILITY & PLATEAU ANALYSIS",
            "Is the locked config at a performance plateau, or is it a sharp spike?")

    df_all = pd.DataFrame(all_results)

    print(f"\n  Total configs evaluated across all sweeps: {len(df_all)}")
    print(f"  Locked Calmar:  {BASE['calmar']:.3f}")

    for threshold_pct in [2, 5, 10]:
        threshold = BASE['calmar'] * (1 - threshold_pct/100)
        count = (df_all['calmar'] >= threshold).sum()
        print(f"  Configs within {threshold_pct:2d}% of locked Calmar "
              f"(≥ {threshold:.3f}): {count:4d} / {len(df_all)} "
              f"({count/len(df_all)*100:.0f}%)")

    best_overall = df_all.loc[df_all['calmar'].idxmax()]
    print(f"\n  Best single-point Calmar in any sweep: {best_overall['calmar']:.3f}")
    print(f"  Parameter: {best_overall['param']}={best_overall['value']}")
    print(f"  CAGR={best_overall['cagr']:.1f}%  DD={best_overall['max_dd']:.1f}%")

    # Calmar distribution
    print(f"\n  Calmar Distribution (all {len(df_all)} configs):")
    bins = [0.0, 0.40, 0.50, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80, 1.0]
    labels = [f"{int(b*100)}–{int(bins[i+1]*100)}" for i, b in enumerate(bins[:-1])]
    for i, lbl in enumerate(labels):
        count = ((df_all['calmar'] >= bins[i]) & (df_all['calmar'] < bins[i+1])).sum()
        bar   = "█" * count
        mark  = " ◄ locked" if bins[i] <= BASE['calmar'] < bins[i+1] else ""
        print(f"  {lbl:>8}  {bar:<40} {count:3d}{mark}")

    # Sensitivity table per parameter
    print(f"\n  Parameter sensitivity (std dev of Calmar within each param sweep):")
    print(f"  {'Parameter':<20} {'N configs':>10} {'Calmar min':>11} {'Calmar max':>11} "
          f"{'Std dev':>9} {'Verdict':>20}")
    print(f"  {'─'*20} {'─'*10} {'─'*11} {'─'*11} {'─'*9} {'─'*20}")
    for param in df_all['param'].unique():
        sub = df_all[df_all['param'] == param]['calmar']
        std = sub.std()
        verdict = "STABLE" if std < 0.02 else ("MODERATE" if std < 0.05 else "SENSITIVE")
        print(f"  {param:<20} {len(sub):>10} {sub.min():>11.3f} {sub.max():>11.3f} "
              f"{std:>9.4f} {verdict:>20}")

    # ══════════════════════════════════════════════════════════════════════════
    #  PART G — FINAL VERDICT
    # ══════════════════════════════════════════════════════════════════════════
    section("PART G — FINAL VERDICT & LOCKED PARAMETERS")

    # Check if any single-parameter change improves Calmar meaningfully (>1%)
    meaningful_threshold = BASE['calmar'] * 1.01   # 1% better
    improvements = df_all[df_all['calmar'] > meaningful_threshold].copy()
    improvements_sorted = improvements.sort_values('calmar', ascending=False)

    if len(improvements) == 0:
        print(f"\n  ✅  VERDICT: SYSTEM IS AT PLATEAU")
        print(f"     No single parameter change improves Calmar by >1% vs locked config.")
        print(f"     Recommendation: DEPLOY AS-IS.")
    else:
        print(f"\n  ⚙️   VERDICT: MARGINAL IMPROVEMENTS FOUND")
        print(f"     {len(improvements)} configs beat the locked Calmar by >1%:")
        print(f"\n  {'Param':<20} {'Value':>8} {'CAGR':>7} {'DD':>7} "
              f"{'Calmar':>9} {'Δ Calmar':>10} {'Sharpe':>8}")
        print(f"  {'─'*20} {'─'*8} {'─'*7} {'─'*7} {'─'*9} {'─'*10} {'─'*8}")
        for _, r in improvements_sorted.head(10).iterrows():
            delta = r['calmar'] - BASE['calmar']
            print(f"  {r['param']:<20} {str(r['value']):>8} "
                  f"{r['cagr']:>6.1f}%  {r['max_dd']:>6.1f}%  "
                  f"{r['calmar']:>9.3f}  {delta:>+9.3f}  {r['sharpe']:>8.3f}")

    # Final recommendation
    print(f"\n{'═'*W}")
    print(f"  FINAL LOCKED CONFIGURATION")
    print(f"{'═'*W}")
    print(f"  T-1 execution           : YES  (regime from previous day close)")
    print(f"  VIX smoothing           : {LOCKED['vix_smooth']}-day rolling average")

    # Check if any vix_bull improvement is meaningful and stable
    vb_improvements = df_all[(df_all['param']=='vix_bull') & (df_all['calmar'] > meaningful_threshold)]
    rec_vb = int(best_a3_cfg.get('vix_bull', LOCKED['vix_bull'])) \
             if best_a3_calmar > meaningful_threshold else LOCKED['vix_bull']
    rec_vh = int(best_a3_cfg.get('vix_hi_vol', LOCKED['vix_hi_vol'])) \
             if best_a3_calmar > meaningful_threshold else LOCKED['vix_hi_vol']

    best_bull_split_wa = best_bull_cfg.get('wa', LOCKED['alloc_bull'][0]) \
                         if best_bull_calmar > meaningful_threshold \
                         else LOCKED['alloc_bull'][0]

    print(f"  VIX bull threshold      : {rec_vb}")
    print(f"  VIX hi-vol threshold    : {rec_vh}")
    print(f"  MA window (regime)      : {LOCKED['ma_window']}")
    print(f"  Bull allocation         : {int(best_bull_split_wa*100)}/{int((1-best_bull_split_wa)*100)}")
    print(f"  Mid allocation          : 50/50")
    print(f"  Hi-vol allocation       : 30/70")
    print(f"  Slippage assumption     : 10 bps per fill (conservative)")
    print(f"\n  Expected live performance (10 bps slippage, T-1 execution):")
    print(f"    CAGR        {slippage_results[10]['cagr']:.1f}%")
    print(f"    Max DD      {slippage_results[10]['max_dd']:.1f}%")
    print(f"    Calmar      {slippage_results[10]['calmar']:.3f}")
    print(f"    Sharpe      {slippage_results[10]['sharpe']:.3f}")
    print(f"    Final $     ${slippage_results[10]['final']:,.0f}")
    print(f"\n  Overfitting risk assessment: ", end="")
    if len(improvements) == 0:
        print("LOW — system sits in a broad flat plateau. No cherry-picked peaks.")
    elif len(improvements) < 5:
        print("LOW-MEDIUM — very few configs beat the locked Calmar. Small stable zone.")
    else:
        print("MEDIUM — multiple configs beat locked config. Verify improvements hold out-of-sample.")
    print(f"{'═'*W}\n")


if __name__ == "__main__":
    main()
