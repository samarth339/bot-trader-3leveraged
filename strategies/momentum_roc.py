"""
Strategy 1 — Momentum / Rate-of-Change  (crossover-event version)

Logic (Mallik-style):
  - Use QQQ as the signal source (cleaner than 3x ETF for trend detection)
  - REGIME changes (not daily level checks) drive entries:
      bull = price crosses ABOVE both 50-day AND 250-day MA  → enter TQQQ
      bear = price crosses BELOW both 50-day AND 250-day MA  → enter SQQQ
  - ROC confirms direction on entry only
  - Min hold period (min_hold_days) prevents rapid whipsaw re-entries
  - Exit only when the OPPOSITE regime is confirmed (not just neutral)

Key fix vs v1: track previous day's regime state so we only act on TRANSITIONS,
not on every bar where the level condition happens to be true.

Parameters:
    ma_short:      int   default 50
    ma_long:       int   default 250
    roc_window:    int   default 20
    roc_thresh:    float default 0.0   (minimum |ROC| to enter)
    min_hold_days: int   default 5     (lock-in period after entry)
    size_pct:      float default 1.0
"""

import pandas as pd
from strategies.base import BaseStrategy


class MomentumROC(BaseStrategy):

    def __init__(
        self,
        ma_short:      int   = 50,
        ma_long:       int   = 250,
        roc_window:    int   = 20,
        roc_thresh:    float = 0.0,
        min_hold_days: int   = 5,
        size_pct:      float = 1.0,
    ):
        self.ma_short      = ma_short
        self.ma_long       = ma_long
        self.roc_window    = roc_window
        self.roc_thresh    = roc_thresh
        self.min_hold_days = min_hold_days
        self.size_pct      = size_pct
        self._prev_regime  = "neutral"   # tracks last bar's regime
        self._days_held    = 0

    def reset(self):
        self._prev_regime = "neutral"
        self._days_held   = 0

    def _regime(self, price: float, sma_s: float, sma_l: float) -> str:
        if price > sma_s and price > sma_l:
            return "bull"
        if price < sma_s and price < sma_l:
            return "bear"
        return "neutral"

    def generate_signal(self, ctx: dict) -> dict:
        i   = ctx["index"]
        qqq = ctx["qqq"]["close"]
        pos = ctx["position"]

        if i < self.ma_long:
            return {"action": "hold"}

        close_series = qqq.iloc[: i + 1]
        price = float(close_series.iloc[-1])
        sma_s = self.sma(close_series, self.ma_short)
        sma_l = self.sma(close_series, self.ma_long)
        roc   = self.roc(close_series, self.roc_window)

        curr_regime = self._regime(price, sma_s, sma_l)

        # ── Track hold duration ───────────────────────────────────────────────
        if pos:
            self._days_held += 1

        # ── Exit: only when regime flips to opposite (not just neutral) ───────
        if pos and self._days_held >= self.min_hold_days:
            if pos.ticker == "TQQQ" and curr_regime == "bear":
                self._prev_regime = curr_regime
                self._days_held   = 0
                return {"action": "sell"}
            if pos.ticker == "SQQQ" and curr_regime == "bull":
                self._prev_regime = curr_regime
                self._days_held   = 0
                return {"action": "sell"}

        # ── Entry: only on regime TRANSITION (crossover event) ────────────────
        if not pos:
            regime_changed = curr_regime != self._prev_regime
            if regime_changed and curr_regime == "bull" and roc >= self.roc_thresh:
                self._prev_regime = curr_regime
                return {"action": "buy_tqqq", "size_pct": self.size_pct}
            if regime_changed and curr_regime == "bear" and roc <= -self.roc_thresh:
                self._prev_regime = curr_regime
                return {"action": "buy_sqqq", "size_pct": self.size_pct}

        self._prev_regime = curr_regime
        return {"action": "hold"}
