"""
Strategy 3 — Combined Momentum + Mean Reversion  (v2: ATR stop + VIX sizing)

Logic:
  - Use MomentumROC to determine REGIME (bull / bear / neutral)
  - Within regime, use MeanReversion for precise entry timing
    * Bull regime + oversold dip → buy TQQQ (trend-following entry on dip)
    * Bear regime + overbought snap → buy SQQQ
  - ATR trailing stop cuts losses before regime exit (A1 improvement)
  - VIX-adjusted position sizing reduces size in fear spikes (B3 improvement)

Parameters:
    ma_short:      int   default 50
    ma_long:       int   default 250
    roc_window:    int   default 20
    mr_window:     int   default 10
    mr_band:       float default 0.02
    atr_stop_mult: float default 2.5   (stop = entry - mult * ATR)
    size_pct:      float default 1.0
"""

import pandas as pd
from strategies.base import BaseStrategy


class CombinedStrategy(BaseStrategy):

    def __init__(
        self,
        ma_short:      int   = 50,
        ma_long:       int   = 250,
        roc_window:    int   = 20,
        mr_window:     int   = 10,
        mr_band:       float = 0.02,
        atr_stop_mult: float = 2.5,
        size_pct:      float = 1.0,
    ):
        self.ma_short      = ma_short
        self.ma_long       = ma_long
        self.roc_window    = roc_window
        self.mr_window     = mr_window
        self.mr_band       = mr_band
        self.atr_stop_mult = atr_stop_mult
        self.size_pct      = size_pct
        self._stop_price   = None

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

        if i < self.ma_long:
            return {"action": "hold"}

        close_series = qqq.iloc[: i + 1]
        price = float(close_series.iloc[-1])

        sma_short = self.sma(close_series, self.ma_short)
        sma_long  = self.sma(close_series, self.ma_long)
        roc       = self.roc(close_series, self.roc_window)
        sma_mr    = self.sma(close_series, self.mr_window)
        deviation = (price - sma_mr) / sma_mr

        bull = price > sma_short and price > sma_long
        bear = price < sma_short and price < sma_long

        # ── ATR trailing stop check ───────────────────────────────────────────
        if pos and self._stop_price is not None:
            cur_etf_price = float(
                ctx["tqqq"]["close"].iloc[i] if pos.ticker == "TQQQ"
                else ctx["sqqq"]["close"].iloc[i]
            )
            if cur_etf_price <= self._stop_price:
                self._stop_price = None
                return {"action": "sell"}

        # ── Exit: regime flip OR mean-reversion target hit ────────────────────
        if pos:
            if pos.ticker == "TQQQ" and (bear or price >= sma_mr):
                self._stop_price = None
                return {"action": "sell"}
            if pos.ticker == "SQQQ" and (bull or price <= sma_mr):
                self._stop_price = None
                return {"action": "sell"}
            return {"action": "hold"}

        # ── Entry: regime filter + mean-reversion timing ──────────────────────
        sz = self._vix_size(ctx)

        if bull and roc > 0 and deviation <= -self.mr_band:
            tqqq_slice = ctx["tqqq"].iloc[: i + 1]
            a = self.atr(tqqq_slice, 14)
            ep = float(tqqq_slice["close"].iloc[-1])
            if not pd.isna(a):
                self._stop_price = ep - self.atr_stop_mult * a
            return {"action": "buy_tqqq", "size_pct": sz}

        if bear and roc < 0 and deviation >= self.mr_band:
            sqqq_slice = ctx["sqqq"].iloc[: i + 1]
            a = self.atr(sqqq_slice, 14)
            ep = float(sqqq_slice["close"].iloc[-1])
            if not pd.isna(a):
                self._stop_price = ep - self.atr_stop_mult * a
            return {"action": "buy_sqqq", "size_pct": sz}

        return {"action": "hold"}
