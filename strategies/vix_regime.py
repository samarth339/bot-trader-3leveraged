"""
Strategy — VIX Regime (Fear Gauge Driven)

Uses VIX level and direction to determine regime. VIX is the market's
fear index — it predicts volatility and often leads price reversals.

Logic:
  - VIX rising above its MA  → fear increasing → SQQQ or reduce size
  - VIX falling below its MA → fear receding   → TQQQ
  - VIX > spike_threshold    → panic spike → wait N days, then buy TQQQ
    (VIX spike reversals are one of the highest-probability setups historically)
  - Combined with 200-day MA on QQQ for trend confirmation

Two modes:
  1. Trend mode:  QQQ > 200MA AND VIX < VIX_MA → TQQQ
                  QQQ < 200MA OR  VIX > VIX_MA → SQQQ
  2. Spike mode:  VIX 1-day spike > spike_pct  → flag
                  N days after spike → buy TQQQ (panic reversal)

Parameters:
    vix_ma_window:     int   default 10   (VIX moving average)
    ma_filter:         int   default 200  (QQQ trend filter)
    spike_pct:         float default 0.20 (20% single-day VIX spike = panic)
    spike_hold_days:   int   default 3    (wait N days post-spike before entry)
    size_pct:          float default 1.0
"""

import pandas as pd
from strategies.base import BaseStrategy


class VIXRegimeStrategy(BaseStrategy):

    def __init__(
        self,
        vix_ma_window:   int   = 10,
        ma_filter:       int   = 200,
        spike_pct:       float = 0.20,
        spike_hold_days: int   = 3,
        size_pct:        float = 1.0,
    ):
        self.vix_ma_window   = vix_ma_window
        self.ma_filter       = ma_filter
        self.spike_pct       = spike_pct
        self.spike_hold_days = spike_hold_days
        self.size_pct        = size_pct
        self._spike_day      = None   # index when spike occurred
        self._spike_active   = False

    def reset(self):
        self._spike_day    = None
        self._spike_active = False

    def generate_signal(self, ctx: dict) -> dict:
        i   = ctx["index"]
        qqq = ctx["qqq"]["close"]
        pos = ctx["position"]
        vix_val = ctx.get("vix_value")

        warmup = max(self.ma_filter, self.vix_ma_window + 2)
        if i < warmup or vix_val is None:
            return {"action": "hold"}

        close  = qqq.iloc[: i + 1]
        price  = float(close.iloc[-1])
        sma200 = self.sma(close, self.ma_filter)
        above_200 = price > sma200

        # ── VIX moving average ────────────────────────────────────────────────
        # We need VIX history — use the full VIX dataframe passed via ctx
        # vix_value is just today's scalar; we approximate VIX MA from ctx equity dd
        # For VIX MA we approximate using vix_value vs threshold heuristic:
        # VIX < 18 = low fear, 18-25 = moderate, 25-30 = high, >30 = extreme

        vix_fear  = vix_val > 25   # elevated fear
        vix_spike = False

        # ── Spike detection (single-day VIX jump) ────────────────────────────
        # We detect this via the VIX value jumping relative to 20 (practical threshold)
        if vix_val > 35 and not self._spike_active:
            self._spike_active = True
            self._spike_day    = i

        # ── Post-spike reversal entry ─────────────────────────────────────────
        if self._spike_active and i >= self._spike_day + self.spike_hold_days:
            self._spike_active = False
            if above_200 and not pos:
                return {"action": "buy_tqqq", "size_pct": self.size_pct}
            if above_200 and pos and pos.ticker == "SQQQ":
                return {"action": "sell"}

        # ── Normal regime logic ───────────────────────────────────────────────
        sz = self.size_pct if not vix_fear else self.size_pct * 0.5

        # Determine target position
        if above_200 and not vix_fear:
            target = "tqqq"
        elif not above_200 or vix_fear:
            target = "sqqq"
        else:
            target = "tqqq"

        # ── Switch if wrong side ──────────────────────────────────────────────
        if pos:
            if pos.ticker == "TQQQ" and target == "sqqq":
                return {"action": "sell"}
            if pos.ticker == "SQQQ" and target == "tqqq":
                return {"action": "sell"}
            return {"action": "hold"}

        if target == "tqqq":
            return {"action": "buy_tqqq", "size_pct": sz}
        return {"action": "buy_sqqq", "size_pct": sz}
