"""
Strategy 4 — RSI Regime

Logic:
  - QQQ RSI(14) + 200-day MA defines the entry filter
  - LONG  side: RSI < rsi_oversold (35)  AND price > 200-day MA  → TQQQ
  - SHORT side: RSI > rsi_overbought (65) AND price < 200-day MA  → SQQQ
  - Exit: RSI crosses back through rsi_exit (50) OR regime flips

Why:
  - RSI<35 in a bull market (above 200MA) is one of the highest-probability
    mean-reversion setups on QQQ. Adding leverage via TQQQ amplifies the snap-back.
  - RSI>65 in a bear market (below 200MA) catches overbought bounces for SQQQ.
  - 200-day MA prevents fighting the primary trend.
  - Only 1–3 signals per month, eliminating overtrading.

Parameters:
    ma_long:        int   default 200   (trend filter MA)
    rsi_window:     int   default 14
    rsi_oversold:   float default 35    (buy TQQQ trigger)
    rsi_overbought: float default 65    (buy SQQQ trigger)
    rsi_exit:       float default 50    (exit midpoint)
    atr_stop_mult:  float default 2.5   (ATR-based trailing stop multiplier)
    size_pct:       float default 1.0
"""

import pandas as pd
from strategies.base import BaseStrategy


class RSIRegimeStrategy(BaseStrategy):

    def __init__(
        self,
        ma_long:        int   = 200,
        rsi_window:     int   = 14,
        rsi_oversold:   float = 35.0,
        rsi_overbought: float = 65.0,
        rsi_exit:       float = 50.0,
        atr_stop_mult:  float = 2.5,
        size_pct:       float = 1.0,
    ):
        self.ma_long        = ma_long
        self.rsi_window     = rsi_window
        self.rsi_oversold   = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_exit       = rsi_exit
        self.atr_stop_mult  = atr_stop_mult
        self.size_pct       = size_pct
        self._stop_price    = None

    def reset(self):
        self._stop_price = None

    def _vix_size(self, ctx: dict) -> float:
        vix = ctx.get("vix_value")
        if vix is None:
            return self.size_pct
        if vix < 18:
            return self.size_pct
        if vix < 25:
            return self.size_pct * 0.75
        if vix < 30:
            return self.size_pct * 0.50
        return self.size_pct * 0.25

    def generate_signal(self, ctx: dict) -> dict:
        i   = ctx["index"]
        qqq = ctx["qqq"]["close"]
        pos = ctx["position"]

        warmup = max(self.ma_long, self.rsi_window + 2)
        if i < warmup:
            return {"action": "hold"}

        close_series = qqq.iloc[: i + 1]
        price   = float(close_series.iloc[-1])
        sma200  = self.sma(close_series, self.ma_long)
        rsi_val = self.rsi(close_series, self.rsi_window)

        bull_regime = price > sma200
        bear_regime = price < sma200

        # ── ATR stop check ────────────────────────────────────────────────────
        if pos and self._stop_price is not None:
            cur_etf_price = float(
                ctx["tqqq"]["close"].iloc[i] if pos.ticker == "TQQQ"
                else ctx["sqqq"]["close"].iloc[i]
            )
            if cur_etf_price <= self._stop_price:
                self._stop_price = None
                return {"action": "sell"}

        # ── Exit: RSI reverts to midpoint OR regime flips ─────────────────────
        if pos:
            if pos.ticker == "TQQQ" and (rsi_val >= self.rsi_exit or bear_regime):
                self._stop_price = None
                return {"action": "sell"}
            if pos.ticker == "SQQQ" and (rsi_val <= self.rsi_exit or bull_regime):
                self._stop_price = None
                return {"action": "sell"}
            return {"action": "hold"}

        # ── Entry ─────────────────────────────────────────────────────────────
        sz = self._vix_size(ctx)

        if bull_regime and rsi_val < self.rsi_oversold:
            # Set ATR-based stop on TQQQ
            tqqq_slice = ctx["tqqq"].iloc[: i + 1]
            a = self.atr(tqqq_slice, 14)
            entry_price = float(tqqq_slice["close"].iloc[-1])
            if not pd.isna(a):
                self._stop_price = entry_price - self.atr_stop_mult * a
            return {"action": "buy_tqqq", "size_pct": sz}

        if bear_regime and rsi_val > self.rsi_overbought:
            sqqq_slice = ctx["sqqq"].iloc[: i + 1]
            a = self.atr(sqqq_slice, 14)
            entry_price = float(sqqq_slice["close"].iloc[-1])
            if not pd.isna(a):
                self._stop_price = entry_price - self.atr_stop_mult * a
            return {"action": "buy_sqqq", "size_pct": sz}

        return {"action": "hold"}
