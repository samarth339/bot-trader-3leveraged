"""
Strategy — MACD + 200-day MA Filter

Classic 12/26/9 MACD on QQQ with trend filter. Always deployed.

Logic:
  - Compute MACD line (EMA12 - EMA26) and Signal line (EMA9 of MACD)
  - Bull: MACD crosses above signal AND QQQ > 200-day MA → TQQQ
  - Bear: MACD crosses below signal AND QQQ < 200-day MA → SQQQ
  - Mixed: hold current position (no whipsaw trading)
  - Histogram expanding in direction confirms entry

Why useful:
  - More responsive than 50/200 MA crossover (less lag)
  - MACD detects momentum changes before price fully confirms
  - Histogram expansion filter eliminates weak crossovers

Parameters:
    fast:       int  default 12
    slow:       int  default 26
    signal:     int  default 9
    ma_filter:  int  default 200
    size_pct:   float default 1.0
"""

import pandas as pd
from strategies.base import BaseStrategy


class MACDStrategy(BaseStrategy):

    def __init__(
        self,
        fast:      int   = 12,
        slow:      int   = 26,
        signal:    int   = 9,
        ma_filter: int   = 200,
        size_pct:  float = 1.0,
    ):
        self.fast      = fast
        self.slow      = slow
        self.signal    = signal
        self.ma_filter = ma_filter
        self.size_pct  = size_pct

    def reset(self):
        pass

    def _ema(self, series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def _macd_signal(self, close: pd.Series):
        """Returns (macd_line, signal_line, histogram) as floats for last 2 bars."""
        ema_fast = self._ema(close, self.fast)
        ema_slow = self._ema(close, self.slow)
        macd     = ema_fast - ema_slow
        sig      = self._ema(macd, self.signal)
        hist     = macd - sig
        return (
            macd.iloc[-2],  sig.iloc[-2],  hist.iloc[-2],   # prev bar
            macd.iloc[-1],  sig.iloc[-1],  hist.iloc[-1],   # current bar
        )

    def _vix_size(self, ctx: dict) -> float:
        vix = ctx.get("vix_value")
        if vix is None:
            return self.size_pct
        if vix < 25:
            return self.size_pct
        if vix < 35:
            return self.size_pct * 0.75
        return self.size_pct * 0.5

    def generate_signal(self, ctx: dict) -> dict:
        i   = ctx["index"]
        qqq = ctx["qqq"]["close"]
        pos = ctx["position"]

        warmup = self.slow + self.signal + self.ma_filter + 5
        if i < warmup:
            return {"action": "hold"}

        close = qqq.iloc[: i + 1]
        price  = float(close.iloc[-1])
        sma200 = self.sma(close, self.ma_filter)

        m_prev, s_prev, h_prev, m_curr, s_curr, h_curr = self._macd_signal(close)

        bull_cross = m_prev <= s_prev and m_curr > s_curr   # crossed above
        bear_cross = m_prev >= s_prev and m_curr < s_curr   # crossed below
        hist_expanding_up   = h_curr > h_prev and h_curr > 0
        hist_expanding_down = h_curr < h_prev and h_curr < 0

        above_200 = price > sma200
        below_200 = price < sma200

        sz = self._vix_size(ctx)

        # ── Exit on opposite cross ────────────────────────────────────────────
        if pos:
            if pos.ticker == "TQQQ" and bear_cross:
                return {"action": "sell"}
            if pos.ticker == "SQQQ" and bull_cross:
                return {"action": "sell"}
            return {"action": "hold"}

        # ── Entry: crossover + trend filter + histogram confirmation ──────────
        if bull_cross and above_200 and hist_expanding_up:
            return {"action": "buy_tqqq", "size_pct": sz}
        if bear_cross and below_200 and hist_expanding_down:
            return {"action": "buy_sqqq", "size_pct": sz}

        # ── Re-enter if strongly trending without recent cross ─────────────────
        if not pos and above_200 and m_curr > s_curr and h_curr > 0:
            return {"action": "buy_tqqq", "size_pct": sz}
        if not pos and below_200 and m_curr < s_curr and h_curr < 0:
            return {"action": "buy_sqqq", "size_pct": sz}

        return {"action": "hold"}
