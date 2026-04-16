"""
Strategy 2 — Mean Reversion (Rubber-Band)

Logic:
  - QQQ drops X% below its N-day SMA → oversold → buy TQQQ (snap-back)
  - QQQ rises X% above its N-day SMA → overbought → buy SQQQ (fade)
  - Exit when price reverts to SMA, or hit stop

Parameters:
    ma_window:   int   default 20
    band_pct:    float default 0.04  (4% deviation triggers entry)
    exit_at_ma:  bool  default True  (exit when price crosses back to MA)
    size_pct:    float default 0.5   (only deploy half capital — more volatile entries)
"""

import pandas as pd
from strategies.base import BaseStrategy


class MeanReversion(BaseStrategy):

    def __init__(
        self,
        ma_window:  int   = 20,
        band_pct:   float = 0.04,
        exit_at_ma: bool  = True,
        size_pct:   float = 0.5,
    ):
        self.ma_window  = ma_window
        self.band_pct   = band_pct
        self.exit_at_ma = exit_at_ma
        self.size_pct   = size_pct

    def reset(self):
        pass

    def generate_signal(self, ctx: dict) -> dict:
        i   = ctx["index"]
        qqq = ctx["qqq"]["close"]
        pos = ctx["position"]

        if i < self.ma_window:
            return {"action": "hold"}

        close_series = qqq.iloc[: i + 1]
        price = float(close_series.iloc[-1])
        ma    = self.sma(close_series, self.ma_window)

        deviation = (price - ma) / ma

        # ── Exit ──────────────────────────────────────────────────────────────
        if pos and self.exit_at_ma:
            if pos.ticker == "TQQQ" and price >= ma:
                return {"action": "sell"}
            if pos.ticker == "SQQQ" and price <= ma:
                return {"action": "sell"}
            return {"action": "hold"}

        if pos:
            return {"action": "hold"}

        # ── Entry ─────────────────────────────────────────────────────────────
        if deviation <= -self.band_pct:
            return {"action": "buy_tqqq", "size_pct": self.size_pct}
        if deviation >= self.band_pct:
            return {"action": "buy_sqqq", "size_pct": self.size_pct}

        return {"action": "hold"}
