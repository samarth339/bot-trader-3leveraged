"""
Strategy — SuperTrend (crossover-event, min hold enforced)

ATR-based dynamic trend line. Only acts on trend FLIPS, not daily levels.
Minimum hold period prevents thrashing on volatile days.

Parameters:
    atr_period:    int   default 10
    multiplier:    float default 3.0
    min_hold_days: int   default 10
    ma_filter:     int   default 200  (0 = disabled)
    size_pct:      float default 1.0
"""

import pandas as pd
from strategies.base import BaseStrategy


class SuperTrendStrategy(BaseStrategy):

    def __init__(
        self,
        atr_period:    int   = 10,
        multiplier:    float = 3.0,
        min_hold_days: int   = 10,
        ma_filter:     int   = 200,
        size_pct:      float = 1.0,
    ):
        self.atr_period    = atr_period
        self.multiplier    = multiplier
        self.min_hold_days = min_hold_days
        self.ma_filter     = ma_filter
        self.size_pct      = size_pct
        self._prev_trend   = None
        self._days_held    = 0

    def reset(self):
        self._prev_trend = None
        self._days_held  = 0

    def _compute_supertrend(self, qqq_df: pd.DataFrame) -> str:
        n = self.atr_period
        if len(qqq_df) < n + 10:
            return "bull"

        df = qqq_df.iloc[-(n + 30):].copy().reset_index(drop=True)
        hl2 = (df["high"] + df["low"]) / 2

        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(n).mean()

        upper_basic = hl2 + self.multiplier * atr
        lower_basic = hl2 - self.multiplier * atr

        upper = upper_basic.copy()
        lower = lower_basic.copy()
        trend = pd.Series("bull", index=df.index)

        for i in range(1, len(df)):
            lower.iloc[i] = (
                max(lower_basic.iloc[i], lower.iloc[i - 1])
                if df["close"].iloc[i - 1] > lower.iloc[i - 1]
                else lower_basic.iloc[i]
            )
            upper.iloc[i] = (
                min(upper_basic.iloc[i], upper.iloc[i - 1])
                if df["close"].iloc[i - 1] < upper.iloc[i - 1]
                else upper_basic.iloc[i]
            )
            prev = trend.iloc[i - 1]
            close = df["close"].iloc[i]
            if prev == "bear" and close > upper.iloc[i]:
                trend.iloc[i] = "bull"
            elif prev == "bull" and close < lower.iloc[i]:
                trend.iloc[i] = "bear"
            else:
                trend.iloc[i] = prev

        return str(trend.iloc[-1])

    def generate_signal(self, ctx: dict) -> dict:
        i      = ctx["index"]
        qqq_df = ctx["qqq"].iloc[: i + 1]
        pos    = ctx["position"]

        warmup = max(self.atr_period + 30, self.ma_filter or 0) + 5
        if i < warmup:
            return {"action": "hold"}

        trend = self._compute_supertrend(qqq_df)

        # Optional 200MA confirmation filter
        if self.ma_filter:
            price  = float(qqq_df["close"].iloc[-1])
            sma200 = self.sma(qqq_df["close"], self.ma_filter)
            if trend == "bull" and price < sma200:
                trend = "neutral"
            elif trend == "bear" and price > sma200:
                trend = "neutral"

        if pos:
            self._days_held += 1

        # ── Only flip on crossover after min hold ─────────────────────────────
        if (trend != "neutral"
                and trend != self._prev_trend
                and self._days_held >= self.min_hold_days):
            self._prev_trend = trend
            self._days_held  = 0
            if pos:
                return {"action": "sell"}

        # ── Enter if flat ─────────────────────────────────────────────────────
        if not pos and trend != "neutral":
            if self._prev_trend is None:
                self._prev_trend = trend
            self._days_held = 0
            if trend == "bull":
                return {"action": "buy_tqqq", "size_pct": self.size_pct}
            return {"action": "buy_sqqq", "size_pct": self.size_pct}

        return {"action": "hold"}
