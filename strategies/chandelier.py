"""
Strategy — Chandelier Exit (proper crossover, cooldown enforced)

Trails from highest close over N days. Only flips on a confirmed break,
with a cooldown period to prevent re-entry thrashing.

Parameters:
    chandelier_period: int   default 22
    atr_period:        int   default 22
    multiplier:        float default 3.0
    ma_filter:         int   default 200
    cooldown_days:     int   default 5   (bars to wait after stop hit)
    size_pct:          float default 1.0
"""

import pandas as pd
from strategies.base import BaseStrategy


class ChandelierStrategy(BaseStrategy):

    def __init__(
        self,
        chandelier_period: int   = 22,
        atr_period:        int   = 22,
        multiplier:        float = 3.0,
        ma_filter:         int   = 200,
        cooldown_days:     int   = 5,
        size_pct:          float = 1.0,
    ):
        self.chandelier_period = chandelier_period
        self.atr_period        = atr_period
        self.multiplier        = multiplier
        self.ma_filter         = ma_filter
        self.cooldown_days     = cooldown_days
        self.size_pct          = size_pct
        self._regime           = None
        self._cooldown         = 0

    def reset(self):
        self._regime   = None
        self._cooldown = 0

    def _stop(self, close: pd.Series, qqq_df: pd.DataFrame) -> float:
        n = self.chandelier_period
        if len(close) < n:
            return 0.0
        highest = float(close.iloc[-n:].max())
        a = self.atr(qqq_df.iloc[-(self.atr_period + 1):], self.atr_period)
        if pd.isna(a):
            return 0.0
        return highest - self.multiplier * a

    def _vix_size(self, ctx) -> float:
        vix = ctx.get("vix_value")
        if vix is not None and vix > 30:
            return self.size_pct * 0.5
        return self.size_pct

    def generate_signal(self, ctx: dict) -> dict:
        i      = ctx["index"]
        qqq_df = ctx["qqq"].iloc[: i + 1]
        pos    = ctx["position"]

        warmup = max(self.ma_filter, self.chandelier_period + self.atr_period + 5)
        if i < warmup:
            return {"action": "hold"}

        close  = qqq_df["close"]
        price  = float(close.iloc[-1])
        sma200 = self.sma(close, self.ma_filter)
        stop   = self._stop(close, qqq_df)

        new_regime = "bull" if price > sma200 else "bear"

        if self._cooldown > 0:
            self._cooldown -= 1

        # ── Chandelier stop hit ───────────────────────────────────────────────
        if pos and price < stop:
            self._regime   = None
            self._cooldown = self.cooldown_days
            return {"action": "sell"}

        # ── Regime flip ───────────────────────────────────────────────────────
        if new_regime != self._regime and self._cooldown == 0:
            self._regime = new_regime
            if pos:
                return {"action": "sell"}

        # ── Enter if flat and no cooldown ─────────────────────────────────────
        if not pos and self._cooldown == 0 and self._regime is not None:
            sz = self._vix_size(ctx)
            if self._regime == "bull":
                return {"action": "buy_tqqq", "size_pct": sz}
            return {"action": "buy_sqqq", "size_pct": sz}

        if not pos and self._regime is None and self._cooldown == 0:
            self._regime = new_regime
            sz = self._vix_size(ctx)
            if new_regime == "bull":
                return {"action": "buy_tqqq", "size_pct": sz}
            return {"action": "buy_sqqq", "size_pct": sz}

        return {"action": "hold"}
