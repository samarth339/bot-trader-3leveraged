"""
Daily Signal Generator  —  Phase 3
=====================================
Outputs today's regime, allocation, and recommended action.
Uses T-1 execution: signal is based on YESTERDAY'S close and VIX.

Usage:
    python daily_signal.py                   # normal run
    python daily_signal.py --shadow          # shadow mode (log only, no action)
    python daily_signal.py --date 2024-01-15 # back-calculate for a specific date

Output example:
    ╔══════════════════════════════════════════════════╗
    ║  DAILY SIGNAL  —  2026-03-23                     ║
    ╠══════════════════════════════════════════════════╣
    ║  Regime         : BULL                           ║
    ║  Signal source  : 2026-03-20 (T-1 close)        ║
    ║  QQQ close      : $473.81  (SMA-150: $451.22)   ║
    ║  VIX (5d avg)   : 14.8  (raw: 15.1)             ║
    ╠══════════════════════════════════════════════════╣
    ║  Target alloc   : 75% Strategy-A / 25% Strat-B  ║
    ║  Previous alloc : 75% / 25%                     ║
    ║  ACTION         : HOLD  (no rebalance needed)   ║
    ╚══════════════════════════════════════════════════╝
"""

import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from config.settings    import DATA_PROCESSED_DIR
from config.strategy_config import (
    REGIME_CONFIG, ALLOC_CONFIG, EXECUTION_CONFIG, RISK_CONFIG,
    STRATEGY_A_CONFIG, STRATEGY_B_CONFIG,
)

# ── Logging setup ──────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "daily_signal.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("daily_signal")

SIGNAL_LOG = LOG_DIR / "signal_history.csv"

VALID_REGIMES = {"bull", "uncertain", "high_vol"}
REGIME_LABEL  = {"bull": "BULL", "uncertain": "UNCERTAIN", "high_vol": "HIGH-VOL"}


# ── Data loader ────────────────────────────────────────────────────────────────
def load_data():
    qqq = pd.read_csv(f"{DATA_PROCESSED_DIR}/QQQ_full.csv",
                      index_col=0, parse_dates=True)
    vix = pd.read_csv(f"{DATA_PROCESSED_DIR}/VIX_full.csv",
                      index_col=0, parse_dates=True)
    if qqq["close"].isnull().any():
        raise ValueError("Missing QQQ data — run fetch_data.py")
    if vix["close"].isnull().any():
        raise ValueError("Missing VIX data — run fetch_data.py")
    return qqq, vix


# ── Regime computation (T-1 hardcoded) ────────────────────────────────────────
def compute_regime(qqq: pd.DataFrame, vix: pd.DataFrame,
                   as_of_date: pd.Timestamp) -> dict:
    """
    Compute regime AS OF as_of_date using T-1 close (yesterday's data).

    Returns a dict with all signal components for audit logging.
    """
    ma_window  = REGIME_CONFIG["ma_window"]
    vix_smooth = REGIME_CONFIG["vix_smooth"]
    vix_bull   = REGIME_CONFIG["vix_bull"]
    vix_hi_vol = REGIME_CONFIG["vix_hi_vol"]

    qqq_close = qqq["close"]
    vix_close = vix["close"]

    # ── Smooth VIX then shift by 1 (T-1 hard guard at data layer) ─────────────
    vix_smoothed = vix_close.rolling(vix_smooth, min_periods=1).mean()
    signal_close = qqq_close.shift(1).ffill()     # yesterday's QQQ close
    signal_vix   = vix_smoothed.shift(1).ffill()  # yesterday's smoothed VIX

    # Find the most recent trading day ≤ as_of_date
    available_dates = qqq_close.index[qqq_close.index <= as_of_date]
    if len(available_dates) == 0:
        raise ValueError(f"No QQQ data on or before {as_of_date.date()}")
    today_bar = available_dates[-1]
    # T-1: the signal is sourced from the previous trading day's data
    signal_date = available_dates[-2] if len(available_dates) >= 2 else today_bar

    # Signal values — read from the today_bar row of the shifted series
    # (shifted series at today_bar contains yesterday's actual data)
    price_signal   = float(signal_close.loc[today_bar])
    vix_signal     = float(signal_vix.loc[today_bar]) \
                     if today_bar in signal_vix.index else 20.0
    vix_raw_today  = float(vix_close.loc[today_bar]) \
                     if today_bar in vix_close.index else float("nan")

    # SMA on shifted close — no look-ahead possible
    signal_sma_series = signal_close.rolling(ma_window, min_periods=ma_window).mean()
    sma_val = float(signal_sma_series.loc[today_bar]) \
              if today_bar in signal_sma_series.index else float("nan")

    # ── Regime classification ──────────────────────────────────────────────────
    if np.isnan(sma_val) or np.isnan(price_signal):
        regime = "uncertain"
        reason = f"insufficient history (need {ma_window} bars)"
    elif price_signal < sma_val or vix_signal >= vix_hi_vol:
        regime = "high_vol"
        reason = (f"price {'below' if price_signal < sma_val else 'at/above'} SMA"
                  + (f" AND VIX {vix_signal:.1f} ≥ {vix_hi_vol}" if vix_signal >= vix_hi_vol else ""))
    elif price_signal > sma_val and vix_signal < vix_bull:
        regime = "bull"
        reason = f"price above SMA AND VIX {vix_signal:.1f} < {vix_bull}"
    else:
        regime = "uncertain"
        reason = (f"price above SMA but VIX {vix_signal:.1f} in grey zone "
                  f"[{vix_bull}–{vix_hi_vol})")

    # ── Momentum override (expert-panel v2): faster re-entry from high_vol ─────
    # If regime is high_vol but 5-day price momentum is strongly positive AND
    # price is within 1.5% of SMA, upgrade to uncertain early (reduces re-entry lag).
    pct_vs_sma = (price_signal - sma_val) / sma_val * 100 if not np.isnan(sma_val) else float("nan")
    roc_5 = float("nan")
    if len(available_dates) >= 6:
        price_5d_ago = float(signal_close.loc[available_dates[-6]])
        if price_5d_ago > 0 and not np.isnan(price_5d_ago):
            roc_5 = (price_signal - price_5d_ago) / price_5d_ago * 100

    if (regime == "high_vol"
            and not np.isnan(roc_5)
            and roc_5 > 3.0
            and not np.isnan(pct_vs_sma)
            and pct_vs_sma > -1.5):
        regime = "uncertain"
        reason = (f"momentum override: ROC-5={roc_5:.1f}%, "
                  f"price {pct_vs_sma:+.1f}% vs SMA (was high_vol)")

    # ── Deterministic assert ───────────────────────────────────────────────────
    assert regime in VALID_REGIMES, f"Invalid regime '{regime}'"

    return {
        "as_of_date":   as_of_date,
        "signal_date":  signal_date,      # the T-1 date used for computation
        "regime":       regime,
        "reason":       reason,
        "price_signal": price_signal,
        "sma_val":      sma_val,
        "pct_vs_sma":   pct_vs_sma,
        "roc_5":        roc_5,            # 5-day momentum of signal series
        "vix_signal":   vix_signal,       # smoothed, shifted (what drives the regime)
        "vix_raw":      vix_raw_today,    # raw VIX for display
        "vix_smooth_window": vix_smooth,
        "vix_bull_thresh":   vix_bull,
        "vix_hv_thresh":     vix_hi_vol,
        "sma_window":        ma_window,
    }


# ── Action resolver ────────────────────────────────────────────────────────────
def _uncertain_alloc(pct_vs_sma: float) -> tuple:
    """
    Dynamic allocation for uncertain regime based on price momentum vs SMA.
    Scales Strategy A weight from 45% (deep below SMA) to 75% (strongly above).
    This smooths re-entry and phases out of defensive positioning as market recovers.
    """
    if pct_vs_sma >  3.0: return (0.75, 0.25)  # strong recovery momentum
    if pct_vs_sma >  1.0: return (0.70, 0.30)  # clear uptrend
    if pct_vs_sma > -1.0: return (0.65, 0.35)  # neutral (matches old fixed value)
    if pct_vs_sma > -3.0: return (0.55, 0.45)  # weakening, lean defensive
    return (0.45, 0.55)                          # near high_vol boundary


def resolve_action(regime: str, prev_regime: str,
                   pct_vs_sma: float = 0.0,
                   prev_pct_vs_sma: float = 0.0) -> dict:
    """
    Determine what portfolio action to take given regime change.

    Args:
        regime:          current regime label
        prev_regime:     previous regime label
        pct_vs_sma:      current QQQ % above/below SMA (for dynamic uncertain alloc)
        prev_pct_vs_sma: previous day's pct_vs_sma (for accurate drift calculation)

    Returns action dict with: action, target_alloc, prev_alloc, rebalance_needed, drift_pct
    """
    # Resolve target allocation — uncertain is dynamic, others are fixed
    if regime == "uncertain":
        target = _uncertain_alloc(pct_vs_sma)
    else:
        target = ALLOC_CONFIG.get(regime, ALLOC_CONFIG["uncertain"])

    # Resolve previous allocation — mirror same logic for accurate drift
    if not prev_regime:
        prev = target
    elif prev_regime == "uncertain":
        prev = _uncertain_alloc(prev_pct_vs_sma)
    else:
        prev = ALLOC_CONFIG.get(prev_regime, ALLOC_CONFIG["uncertain"])

    drift  = abs(target[0] - prev[0])
    rebal  = drift > RISK_CONFIG["alloc_drift_warn"]

    if not rebal:
        action = "HOLD"
    elif target[0] > prev[0]:
        action = "INCREASE_A"   # shift more into aggressive strategy
    else:
        action = "REDUCE_A"     # shift more into defensive strategy

    return {
        "action":            action,
        "target_alloc":      target,
        "prev_alloc":        prev,
        "drift_pct":         round(drift * 100, 1),
        "rebalance_needed":  rebal,
    }


# ── Load signal history ────────────────────────────────────────────────────────
def load_prev_signal() -> dict:
    """Read the last signal from the history log."""
    if not SIGNAL_LOG.exists():
        return {}
    try:
        df = pd.read_csv(SIGNAL_LOG, parse_dates=["as_of_date"])
        if df.empty:
            return {}
        last = df.iloc[-1]
        return last.to_dict()
    except Exception:
        return {}


def append_signal_log(row: dict):
    """Write today's signal to the history CSV — one row per as_of_date (overwrite)."""
    today_str = row.get("as_of_date", "")
    df_new = pd.DataFrame([row])
    if SIGNAL_LOG.exists():
        try:
            existing = pd.read_csv(SIGNAL_LOG, dtype=str)
            if today_str and "as_of_date" in existing.columns:
                # Drop any existing entry for today then re-append (overwrite)
                existing = existing[existing["as_of_date"] != today_str]
            combined = pd.concat([existing, df_new], ignore_index=True)
            combined.to_csv(SIGNAL_LOG, index=False)
            return
        except Exception:
            pass
    df_new.to_csv(SIGNAL_LOG, index=False)


# ── Display ────────────────────────────────────────────────────────────────────
def print_signal(sig: dict, action: dict, shadow: bool = False,
                 gap_triggered: bool = False, gap_pct: float = float("nan")):
    w      = 54
    border = "═" * w
    mid    = "─" * w
    date_s = sig["as_of_date"].strftime("%Y-%m-%d")
    regime = REGIME_LABEL[sig["regime"]]
    sig_d  = sig["signal_date"].strftime("%Y-%m-%d")
    alloc  = action["target_alloc"]
    prev   = action["prev_alloc"]
    act    = action["action"]
    drift  = action["drift_pct"]
    shadow_tag = "  [SHADOW MODE — no action]" if shadow else ""

    def line(txt, w=w):
        return f"║  {txt:<{w-4}}  ║"

    print(f"\n╔{border}╗")
    print(f"║  {'DAILY SIGNAL  —  ' + date_s + shadow_tag:<{w-2}}║")
    print(f"╠{border}╣")
    print(line(f"Regime         : {regime}"))
    print(line(f"Reason         : {sig['reason'][:w-22]}"))
    print(line(f"Signal source  : {sig_d} (T-1 close)"))
    print(line(f"QQQ close (T-1): ${sig['price_signal']:.2f}  "
               f"(SMA-{sig['sma_window']}: ${sig['sma_val']:.2f}  "
               f"{sig['pct_vs_sma']:+.1f}%)"))
    print(line(f"VIX (smoothed) : {sig['vix_signal']:.1f}  "
               f"(raw: {sig['vix_raw']:.1f})  "
               f"[bull<{sig['vix_bull_thresh']}  "
               f"danger≥{sig['vix_hv_thresh']}]"))
    print(f"╠{border}╣")
    print(line(f"Target alloc   : {int(alloc[0]*100)}% Strategy-A  /  "
               f"{int(alloc[1]*100)}% Strategy-B"))
    if prev != alloc:
        print(line(f"Previous alloc : {int(prev[0]*100)}% / {int(prev[1]*100)}%  "
                   f"(drift: {drift}%)"))
    else:
        print(line(f"Previous alloc : {int(prev[0]*100)}% / {int(prev[1]*100)}%  (unchanged)"))
    print(f"╠{border}╣")

    if action["rebalance_needed"]:
        print(line(f"⚡  ACTION : {act}"))
        print(line(f"   Rebalance to {int(alloc[0]*100)}/{int(alloc[1]*100)} split"))
        if act == "INCREASE_A":
            print(line(f"   Buy more Strategy-A (aggressive)"))
            print(line(f"   Reduce Strategy-B (defensive)"))
        elif act == "REDUCE_A":
            print(line(f"   Buy more Strategy-B (defensive)"))
            print(line(f"   Reduce Strategy-A (aggressive)"))
    else:
        print(line(f"✅  ACTION : HOLD  (allocation in target range)"))
        print(line(f"   No rebalance needed  (drift {drift}% < "
                   f"{int(RISK_CONFIG['alloc_drift_warn']*100)}% threshold)"))

    print(f"╠{border}╣")
    print(line(f"Execution      : {EXECUTION_CONFIG['model'].upper()} + "
               f"{EXECUTION_CONFIG['slippage_bps']} bps slippage"))
    if not np.isnan(gap_pct):
        gap_str = f"{gap_pct*100:+.2f}%" if not np.isnan(gap_pct) else "n/a"
        if gap_triggered:
            print(line(f"🚫 GAP GUARD    : TRIGGERED ({gap_str} open) — BUY blocked"))
        else:
            print(line(f"✅ Gap guard    : clear  ({gap_str} open)"))
    if shadow:
        print(line("⚠  SHADOW MODE — signals logged, no trades placed"))
    print(f"╚{border}╝\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Daily trading signal generator")
    parser.add_argument("--shadow", action="store_true",
                        help="Shadow mode: generate and log signal, no trading")
    parser.add_argument("--date",   type=str, default=None,
                        help="Back-calculate for a specific date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Determine as-of date
    if args.date:
        as_of = pd.Timestamp(args.date)
    else:
        as_of = pd.Timestamp(datetime.today().strftime("%Y-%m-%d"))

    logger.info(f"{'='*60}")
    logger.info(f"Daily signal run  |  as_of={as_of.date()}  |  "
                f"shadow={args.shadow}")

    # Load data
    qqq, vix = load_data()

    # Compute regime signal (T-1 hard guard inside)
    sig = compute_regime(qqq, vix, as_of)

    # Load previous regime and pct_vs_sma for action determination
    prev = load_prev_signal()
    prev_regime    = prev.get("regime", sig["regime"])
    prev_pct_raw   = prev.get("pct_vs_sma", 0.0)
    prev_pct_vs_sma = float(prev_pct_raw) if prev_pct_raw not in ("", None) else 0.0

    # Resolve action (pct_vs_sma drives dynamic uncertain allocation)
    action = resolve_action(sig["regime"], prev_regime,
                            pct_vs_sma=sig["pct_vs_sma"] if not np.isnan(sig["pct_vs_sma"]) else 0.0,
                            prev_pct_vs_sma=prev_pct_vs_sma)

    # ── Gap guard check ────────────────────────────────────────────────────────
    # Run for today's live signal (not for historical --date back-calculations,
    # since we can't reconstruct intraday open prices for past dates).
    gap_triggered = False
    gap_pct_val   = float("nan")
    if not args.date:
        try:
            from ibkr.gap_guard import GapGuard
            gap_result    = GapGuard().check()
            gap_triggered = gap_result.triggered
            gap_pct_val   = gap_result.gap_pct
            if gap_triggered:
                logger.warning(
                    f"⚠  GAP GUARD: TQQQ opened {gap_pct_val*100:+.1f}% — "
                    "would block BUY orders at execution time"
                )
            else:
                logger.info(
                    f"Gap guard: clear  (TQQQ open gap {gap_pct_val*100:+.2f}%)"
                )
        except Exception as exc:
            logger.warning(f"Gap guard check skipped in signal run: {exc}")

    # Display
    print_signal(sig, action, shadow=args.shadow,
                 gap_triggered=gap_triggered, gap_pct=gap_pct_val)

    # Log signal
    log_row = {
        "as_of_date":       sig["as_of_date"].strftime("%Y-%m-%d"),
        "signal_date":      sig["signal_date"].strftime("%Y-%m-%d"),
        "regime":           sig["regime"],
        "action":           action["action"],
        "weight_a":         action["target_alloc"][0],
        "weight_b":         action["target_alloc"][1],
        "rebalance":        action["rebalance_needed"],
        "drift_pct":        action["drift_pct"],
        "qqq_price":        round(sig["price_signal"], 4),
        "sma_val":          round(sig["sma_val"], 4) if not np.isnan(sig["sma_val"]) else "",
        "pct_vs_sma":       round(sig["pct_vs_sma"], 2) if not np.isnan(sig["pct_vs_sma"]) else "",
        "vix_signal":       round(sig["vix_signal"], 2),
        "vix_raw":          round(sig["vix_raw"], 2) if not np.isnan(sig["vix_raw"]) else "",
        "shadow":           args.shadow,
        "gap_guard":        gap_triggered,
        "gap_pct":          round(gap_pct_val * 100, 2) if not np.isnan(gap_pct_val) else "",
    }
    append_signal_log(log_row)
    logger.info(f"Signal logged: regime={sig['regime']}  "
                f"action={action['action']}  "
                f"alloc={int(action['target_alloc'][0]*100)}/{int(action['target_alloc'][1]*100)}")

    # Reconciliation check
    drift = action["drift_pct"]
    if drift > RISK_CONFIG["alloc_drift_rebalance"] * 100:
        logger.warning(
            f"RECONCILIATION: Large allocation drift {drift:.1f}% — "
            f"verify actual portfolio positions match target."
        )
    elif drift > RISK_CONFIG["alloc_drift_warn"] * 100:
        logger.info(
            f"RECONCILIATION: Allocation drift {drift:.1f}% — rebalance scheduled."
        )

    return log_row


if __name__ == "__main__":
    main()
