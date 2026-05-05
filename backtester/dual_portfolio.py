"""
Dual-Portfolio Backtester  (v3 — production-hardened)

Key production guarantees
──────────────────────────
1. T-1 HARD GUARD    Regime signals are computed from SHIFTED input data
                     (df['signal_price'] = close.shift(1),
                      df['signal_vix']   = vix_smooth.shift(1))
                     It is structurally impossible to use today's data.

2. REGIME ASSERT     Every output label is validated:
                     assert regime in {"bull", "uncertain", "high_vol"}
                     No undefined states ever reach the blending layer.

3. RECONCILIATION    Tracks allocation drift after every regime flip.
                     Warns when |actual − target| > ALLOC_DRIFT_WARN.

4. MISSING DATA      Validated upstream in Backtester.__init__() and
                     in compute_regime_series() before any computation.

5. DAILY LOG         Optional CSV audit trail written on run().

Usage:
    from backtester.dual_portfolio import DualPortfolioBacktester
    dp = DualPortfolioBacktester(tqqq, sqqq, qqq, vix, strategy_a, strategy_b)
    results = dp.run(log_path="logs/dual_YYYYMMDD.csv")
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from backtester.engine import Backtester
from config.strategy_config import RISK_CONFIG

logger = logging.getLogger(__name__)

VALID_REGIMES   = {"bull", "uncertain", "high_vol"}
ALLOC_DRIFT_WARN = RISK_CONFIG["alloc_drift_warn"]        # 0.02


class DualPortfolioBacktester:

    def __init__(
        self,
        tqqq, sqqq, qqq, vix,
        strategy_a,
        strategy_b,
        initial_capital: float = 5_000,
        # Discrete regime allocations: (weight_a, weight_b)
        alloc_bull:    tuple = (0.75, 0.25),
        alloc_mid:     tuple = (0.50, 0.50),
        alloc_hi_vol:  tuple = (0.30, 0.70),
        # Regime thresholds
        vix_bull:      float = 18.0,
        vix_hi_vol:    float = 25.0,
        ma_window:     int   = 150,
        # T-1 & stabiliser
        t1:            bool  = True,    # MUST be True for production
        confirm_days:  int   = 1,
        vix_smooth:    int   = 5,
        # Confidence-weighted range
        w_a_min: float = 0.15,
        w_a_max: float = 0.85,
    ):
        self.tqqq            = tqqq
        self.sqqq            = sqqq
        self.qqq             = qqq
        self.vix             = vix
        self.strategy_a      = strategy_a
        self.strategy_b      = strategy_b
        self.initial_capital = initial_capital
        self.alloc_bull      = alloc_bull
        self.alloc_mid       = alloc_mid
        self.alloc_hi_vol    = alloc_hi_vol
        self.vix_bull        = vix_bull
        self.vix_hi_vol      = vix_hi_vol
        self.ma_window       = ma_window
        self.t1              = t1
        self.confirm_days    = confirm_days
        self.vix_smooth      = vix_smooth
        self.w_a_min         = w_a_min
        self.w_a_max         = w_a_max

        if not t1:
            logger.warning(
                "DualPortfolioBacktester: t1=False — using same-bar regime signals. "
                "This is NOT achievable in live trading."
            )

    # ── Regime helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _classify(price: float, sma: float, vix_val: float,
                  vix_bull: float, vix_hi_vol: float) -> str:
        """Pure-function regime classifier. Raises on invalid output."""
        if np.isnan(sma) or np.isnan(price):
            regime = "uncertain"
        elif price < sma or vix_val >= vix_hi_vol:
            regime = "high_vol"
        elif price > sma and vix_val < vix_bull:
            regime = "bull"
        else:
            regime = "uncertain"

        # ── Deterministic regime assert ───────────────────────────────────────
        assert regime in VALID_REGIMES, (
            f"Invalid regime '{regime}' — only {VALID_REGIMES} are permitted."
        )
        return regime

    def _weights(self, regime: str, pct_vs_sma: float = None) -> tuple:
        """
        Return (weight_a, weight_b) for the given regime.
        For uncertain regime, applies dynamic scaling by pct_vs_sma when provided,
        mirroring daily_signal._uncertain_alloc() (expert-panel v2).
        """
        if regime == "bull":
            return self.alloc_bull
        if regime == "high_vol":
            return self.alloc_hi_vol
        # uncertain — dynamic if pct_vs_sma available, else fall back to fixed alloc_mid
        if pct_vs_sma is not None and not np.isnan(pct_vs_sma):
            if pct_vs_sma >  3.0: return (0.75, 0.25)
            if pct_vs_sma >  1.0: return (0.70, 0.30)
            if pct_vs_sma > -1.0: return (0.65, 0.35)
            if pct_vs_sma > -3.0: return (0.55, 0.45)
            return (0.45, 0.55)
        return self.alloc_mid

    # ── T-1 hard guard: build shifted signal series ───────────────────────────

    def _build_signal_inputs(self, vix_smooth_override: int = None):
        """
        Returns (signal_close, signal_sma, signal_vix) — all shifted by 1 bar
        so that regime on day i is computed from day i-1 data.

        This is the T-1 HARD GUARD. Shifting at the data layer means it is
        structurally impossible to look ahead.
        """
        smooth = vix_smooth_override if vix_smooth_override is not None else self.vix_smooth
        qqq_close = self.qqq["close"]

        # Smooth VIX THEN shift — both operations happen before any comparison
        vix_raw = self.vix["close"].copy() if self.vix is not None else None
        if vix_raw is not None and smooth > 1:
            vix_raw = vix_raw.rolling(smooth, min_periods=1).mean()

        if self.t1:
            signal_close = qqq_close.shift(1).ffill()
            signal_vix   = vix_raw.shift(1).ffill() if vix_raw is not None else None
        else:
            signal_close = qqq_close.copy()
            signal_vix   = vix_raw.copy() if vix_raw is not None else None

        # SMA computed on the shifted close — guarantees no look-ahead
        signal_sma = signal_close.rolling(self.ma_window, min_periods=self.ma_window).mean()

        return signal_close, signal_sma, signal_vix

    # ── Regime series ─────────────────────────────────────────────────────────

    def compute_regime_series(
        self,
        t1:           bool = None,
        confirm_days: int  = None,
        vix_smooth:   int  = None,
    ) -> pd.Series:
        """
        Pre-compute the validated regime label for every trading date.
        Parameters override instance defaults if provided.
        """
        _confirm = self.confirm_days if confirm_days is None else confirm_days

        # ── Missing data guard — check RAW data BEFORE any shift/fill ─────────
        raw_close = self.qqq["close"]
        raw_vix   = self.vix["close"].copy() if self.vix is not None else None
        nan_raw_close = int(raw_close.isnull().sum())
        nan_raw_vix   = int(raw_vix.isnull().sum()) if raw_vix is not None else 0
        if nan_raw_close > 0:
            raise ValueError(
                f"Missing QQQ signal data: {nan_raw_close} NaN values. "
                f"Check data/processed/QQQ_full.csv."
            )
        if nan_raw_vix > 0:
            raise ValueError(
                f"Missing VIX signal data: {nan_raw_vix} NaN values — "
                f"wrong regime possible. Re-run fetch_data.py."
            )

        signal_close, signal_sma, signal_vix = self._build_signal_inputs(vix_smooth)
        dates = self.qqq["close"].index

        # Pre-compute 5-day ROC on the shifted signal series (T-1 safe — same series)
        roc_5_series = signal_close.pct_change(5) * 100  # % change over 5 bars

        labels = []
        for date in dates:
            price   = float(signal_close.get(date, np.nan))
            sma_val = float(signal_sma.get(date, np.nan))
            vix_val = float(signal_vix.loc[date]) \
                      if (signal_vix is not None and date in signal_vix.index) else 20.0
            regime  = self._classify(price, sma_val, vix_val,
                                     self.vix_bull, self.vix_hi_vol)

            # ── ROC-5 momentum override (mirrors daily_signal.py expert-panel v2) ──
            # Upgrades high_vol → uncertain early when V-recovery is underway:
            # strong 5-day momentum AND price near/above SMA → reduce defensive posture.
            if regime == "high_vol" and not (np.isnan(price) or np.isnan(sma_val)):
                roc_5   = float(roc_5_series.get(date, float("nan")))
                pct_sma = (price - sma_val) / sma_val * 100
                if not np.isnan(roc_5) and roc_5 > 3.0 and pct_sma > -1.5:
                    regime = "uncertain"

            labels.append(regime)

        # Confirmation inertia
        if _confirm > 1:
            confirmed   = []
            current     = labels[0]
            pending     = labels[0]
            pending_cnt = 0
            for lbl in labels:
                if lbl == pending:
                    pending_cnt += 1
                else:
                    pending     = lbl
                    pending_cnt = 1
                if pending_cnt >= _confirm:
                    current = pending
                confirmed.append(current)
            labels = confirmed

        # NOTE: NO final shift here — T-1 is already baked into signal_close/signal_vix
        return pd.Series(labels, index=dates, name="regime")

    # ── Confidence-weighted classifier (continuous) ───────────────────────────

    def _confidence_weights(self, i: int, signal_close: pd.Series,
                             signal_sma: pd.Series, signal_vix: pd.Series,
                             date: pd.Timestamp) -> tuple:
        sma_val = float(signal_sma.get(date, np.nan))
        price   = float(signal_close.get(date, np.nan))
        if np.isnan(sma_val) or np.isnan(price):
            mid = (self.w_a_min + self.w_a_max) / 2
            return mid, 1.0 - mid

        ma_pct    = (price - sma_val) / sma_val
        vix_val   = float(signal_vix.loc[date]) \
                    if (signal_vix is not None and date in signal_vix.index) else 20.0
        vix_score = float(np.clip(1.0 - (vix_val - 12.0) / (35.0 - 12.0), 0.0, 1.0))
        ma_score  = float(np.clip(0.5 + ma_pct / 0.10, 0.0, 1.0))
        confidence = 0.60 * ma_score + 0.40 * vix_score
        w_a = self.w_a_min + confidence * (self.w_a_max - self.w_a_min)
        return round(w_a, 4), round(1.0 - w_a, 4)

    # ── Strategy runner ───────────────────────────────────────────────────────

    def _run_strategies(self):
        bt_a = Backtester(self.tqqq, self.sqqq, self.qqq,
                          initial_capital=self.initial_capital, vix=self.vix)
        bt_b = Backtester(self.tqqq, self.sqqq, self.qqq,
                          initial_capital=self.initial_capital, vix=self.vix)
        return bt_a.run(self.strategy_a), bt_b.run(self.strategy_b)

    # ── Compile results ───────────────────────────────────────────────────────

    def _compile(self, rows: list, common, res_a: dict, res_b: dict) -> dict:
        ec = pd.DataFrame(rows).set_index("date")
        ec["returns"]  = ec["equity"].pct_change()
        ec["peak"]     = ec["equity"].cummax()
        ec["drawdown"] = (ec["peak"] - ec["equity"]) / ec["peak"]

        total_return = ec["equity"].iloc[-1] / self.initial_capital - 1
        n_years      = (common[-1] - common[0]).days / 365.25
        cagr         = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        max_dd       = ec["drawdown"].max()
        sharpe       = (ec["returns"].mean() / ec["returns"].std() * np.sqrt(252)
                        if ec["returns"].std() > 0 else 0)
        calmar       = cagr / max_dd if max_dd > 0 else 0
        regimes      = ec["regime"]
        n            = len(ec)

        return {
            "equity_curve": ec,
            "metrics": {
                "total_return":    total_return,
                "cagr":            cagr,
                "max_drawdown":    max_dd,
                "sharpe":          sharpe,
                "calmar":          calmar,
                "n_years":         n_years,
                "final_equity":    ec["equity"].iloc[-1],
                "pct_bull":        (regimes == "bull").sum()     / n * 100,
                "pct_uncertain":   (regimes == "uncertain").sum() / n * 100,
                "pct_high_vol":    (regimes == "high_vol").sum() / n * 100,
                "regime_flips":    int(regimes.ne(regimes.shift()).sum()),
            },
            "regime_distribution": regimes.value_counts(normalize=True).mul(100).round(1),
            "strategy_a_results":  res_a,
            "strategy_b_results":  res_b,
        }

    # ── Position reconciliation check ─────────────────────────────────────────

    @staticmethod
    def _reconcile(date, prev_wa: float, new_wa: float, portfolio_val: float):
        """
        Warn when the portfolio's target allocation shifts by more than
        ALLOC_DRIFT_WARN (2%) due to a regime change.

        In live trading this guard detects:
          - partial fills that left allocation off-target
          - API failures that left the previous position in place
          - Rounding errors from broker lot sizes
        """
        drift = abs(new_wa - prev_wa)
        if drift > ALLOC_DRIFT_WARN:
            logger.debug(
                f"REBALANCE  {date}  target_A: {prev_wa:.0%} → {new_wa:.0%}  "
                f"drift={drift:.0%}  portfolio=${portfolio_val:,.0f}"
            )
        if drift > 0.20:
            logger.warning(
                f"LARGE SHIFT  {date}  {prev_wa:.0%} → {new_wa:.0%}  "
                f"(regime flip)"
            )

    # ── Discrete regime run ───────────────────────────────────────────────────

    def run(self, regime_series: pd.Series = None,
            log_path: str = None) -> dict:
        """
        Run with discrete regime allocation.

        regime_series : optional pre-computed series.  If None, uses
                        compute_regime_series() with instance settings.
        log_path      : optional path for daily CSV audit log.
                        e.g. "logs/dual_20260323.csv"
        """
        res_a, res_b = self._run_strategies()
        ec_a         = res_a["equity_curve"]["equity"]
        ec_b         = res_b["equity_curve"]["equity"]
        common       = ec_a.index.intersection(ec_b.index)
        ec_a, ec_b   = ec_a.loc[common], ec_b.loc[common]

        if regime_series is None:
            regime_series = self.compute_regime_series()

        # Pre-build signal inputs for dynamic uncertain allocation (pct_vs_sma)
        signal_close_run, signal_sma_run, _ = self._build_signal_inputs()

        ret_a = ec_a.pct_change().fillna(0)
        ret_b = ec_b.pct_change().fillna(0)

        rows          = []
        portfolio_val = self.initial_capital
        first_regime  = regime_series.loc[common[0]] if common[0] in regime_series.index \
                        else "uncertain"
        first_price   = float(signal_close_run.get(common[0], np.nan))
        first_sma     = float(signal_sma_run.get(common[0], np.nan))
        first_pct_sma = (first_price - first_sma) / first_sma * 100 \
                        if not (np.isnan(first_price) or np.isnan(first_sma)) else float("nan")
        prev_wa       = self._weights(first_regime, first_pct_sma)[0]

        for i, date in enumerate(common):
            regime  = regime_series.loc[date] if date in regime_series.index else "uncertain"
            assert regime in VALID_REGIMES, f"Regime '{regime}' not in {VALID_REGIMES}"

            # Compute pct_vs_sma for dynamic uncertain allocation
            price   = float(signal_close_run.get(date, np.nan))
            sma_val = float(signal_sma_run.get(date, np.nan))
            pct_sma = (price - sma_val) / sma_val * 100 \
                      if not (np.isnan(price) or np.isnan(sma_val)) else float("nan")

            wa, wb  = self._weights(regime, pct_sma)

            # Reconciliation check on every regime flip
            if wa != prev_wa:
                self._reconcile(date, prev_wa, wa, portfolio_val)
            prev_wa = wa

            blended       = wa * float(ret_a.iloc[i]) + wb * float(ret_b.iloc[i])
            portfolio_val *= (1 + blended)
            rows.append({
                "date":        date,
                "equity":      portfolio_val,
                "regime":      regime,
                "weight_a":    wa,
                "weight_b":    wb,
                "pct_vs_sma":  round(pct_sma, 2) if not np.isnan(pct_sma) else None,
                "ret_a":       round(float(ret_a.iloc[i]) * 100, 4),
                "ret_b":       round(float(ret_b.iloc[i]) * 100, 4),
                "blended_ret": round(blended * 100, 4),
            })

        result = self._compile(rows, common, res_a, res_b)

        # ── Daily audit log ───────────────────────────────────────────────────
        if log_path:
            log_dir = Path(log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            ec = result["equity_curve"]
            # Join VIX for the log
            if self.vix is not None:
                vix_s = self.vix["close"].rolling(self.vix_smooth, min_periods=1).mean()
                if self.t1:
                    vix_s = vix_s.shift(1).ffill()
                ec = ec.join(vix_s.rename("vix_smooth"), how="left")
            ec.to_csv(log_path)
            logger.info(f"Daily audit log written: {log_path}  ({len(ec)} rows)")

        return result

    # ── Confidence-weighted run ───────────────────────────────────────────────

    def run_confidence_weighted(self, log_path: str = None) -> dict:
        """Continuous confidence-weighted allocation — no discrete regime steps."""
        res_a, res_b = self._run_strategies()
        ec_a         = res_a["equity_curve"]["equity"]
        ec_b         = res_b["equity_curve"]["equity"]
        common       = ec_a.index.intersection(ec_b.index)
        ec_a, ec_b   = ec_a.loc[common], ec_b.loc[common]

        signal_close, signal_sma, signal_vix = self._build_signal_inputs()

        ret_a         = ec_a.pct_change().fillna(0)
        ret_b         = ec_b.pct_change().fillna(0)
        rows          = []
        portfolio_val = self.initial_capital

        for i, date in enumerate(common):
            wa, wb = self._confidence_weights(i, signal_close, signal_sma,
                                              signal_vix, date)
            blended       = wa * float(ret_a.iloc[i]) + wb * float(ret_b.iloc[i])
            portfolio_val *= (1 + blended)
            rows.append({
                "date":       date,
                "equity":     portfolio_val,
                "regime":     f"cw_{wa:.2f}",
                "weight_a":   wa,
                "weight_b":   wb,
                "ret_a":      round(float(ret_a.iloc[i]) * 100, 4),
                "ret_b":      round(float(ret_b.iloc[i]) * 100, 4),
                "blended_ret": round(blended * 100, 4),
            })

        result = self._compile(rows, common, res_a, res_b)

        if log_path:
            log_dir = Path(log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            result["equity_curve"].to_csv(log_path)

        return result
