"""
Strategy 5 — Dual Momentum (Antonacci-style, TQQQ/SQQQ variant)

Logic:
  - Monthly rebalance (last trading day of each calendar month)
  - Absolute Momentum: is QQQ positive over the lookback? (vs cash/T-bills ≈ 0%)
  - Relative Momentum: does QQQ outperform SPY over the lookback?
  - TQQQ  : QQQ > SPY (relative) AND QQQ > 0% (absolute) over `lookback` days
  - SQQQ  : QQQ < 0 AND SPY < 0 (both negative — true bear) over `lookback` days
  - Cash  : everything else (mixed signals)

Why:
  - Gary Antonacci's Dual Momentum is one of the few systematic strategies
    with real 40-year out-of-sample performance.
  - Adding TQQQ/SQQQ leverage on top of a well-validated signal amplifies the edge.
  - Monthly rebalancing avoids overtrading and noise.
  - Absolute momentum filter (vs 0%) keeps you in cash during sustained downtrends.

Parameters:
    lookback_days:  int   default 63   (≈ 3 months of trading days)
    atr_stop_mult:  float default 3.0  (wider stop — monthly strategy)
    size_pct:       float default 1.0
"""

import pandas as pd
from strategies.base import BaseStrategy


class DualMomentumStrategy(BaseStrategy):

    def __init__(
        self,
        lookback_days: int   = 63,
        atr_stop_mult: float = 3.0,
        size_pct:      float = 1.0,
    ):
        self.lookback_days = lookback_days
        self.atr_stop_mult = atr_stop_mult
        self.size_pct      = size_pct
        self._last_month   = None
        self._target       = "cash"   # "tqqq" | "sqqq" | "cash"
        self._stop_price   = None

    def reset(self):
        self._last_month = None
        self._target     = "cash"
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

    def _is_month_end(self, date: pd.Timestamp, dates) -> bool:
        idx = dates.get_loc(date)
        if idx >= len(dates) - 1:
            return True
        return dates[idx].month != dates[idx + 1].month

    def generate_signal(self, ctx: dict) -> dict:
        i     = ctx["index"]
        date  = ctx["date"]
        dates = ctx["dates"]
        qqq   = ctx["qqq"]["close"]
        pos   = ctx["position"]

        warmup = self.lookback_days + 5
        if i < warmup:
            return {"action": "hold"}

        qqq_series = qqq.iloc[: i + 1]

        # ── ATR stop ──────────────────────────────────────────────────────────
        if pos and self._stop_price is not None:
            cur_price = float(
                ctx["tqqq"]["close"].iloc[i] if pos.ticker == "TQQQ"
                else ctx["sqqq"]["close"].iloc[i]
            )
            if cur_price <= self._stop_price:
                self._stop_price = None
                self._target     = "cash"
                return {"action": "sell"}

        # ── Monthly rebalance only ────────────────────────────────────────────
        if not self._is_month_end(date, dates):
            return {"action": "hold"}

        # ── Compute momentum scores ───────────────────────────────────────────
        qqq_ret = self.roc(qqq_series, self.lookback_days)

        spy_series = ctx.get("spy", {}).get("close")
        if spy_series is not None and len(spy_series) > self.lookback_days:
            spy_slice = spy_series.iloc[: i + 1] if len(spy_series) > i else spy_series
            spy_ret = self.roc(spy_slice, self.lookback_days)
        else:
            spy_ret = 0.0   # No SPY data → use 0 as benchmark

        qqq_beats_spy = qqq_ret > spy_ret
        qqq_positive  = qqq_ret > 0.0
        both_negative = qqq_ret < 0.0 and spy_ret < 0.0

        if qqq_beats_spy and qqq_positive:
            new_target = "tqqq"
        elif both_negative:
            new_target = "sqqq"
        else:
            new_target = "cash"

        sz = self._vix_size(ctx)

        # ── Exit if target changed ─────────────────────────────────────────────
        if pos and new_target != self._target:
            self._stop_price = None
            self._target     = new_target
            return {"action": "sell"}

        self._target = new_target

        # ── Enter if not in position and target is not cash ───────────────────
        if not pos:
            if new_target == "tqqq":
                tqqq_slice = ctx["tqqq"].iloc[: i + 1]
                a = self.atr(tqqq_slice, 14)
                ep = float(tqqq_slice["close"].iloc[-1])
                if not pd.isna(a):
                    self._stop_price = ep - self.atr_stop_mult * a
                return {"action": "buy_tqqq", "size_pct": sz}
            if new_target == "sqqq":
                sqqq_slice = ctx["sqqq"].iloc[: i + 1]
                a = self.atr(sqqq_slice, 14)
                ep = float(sqqq_slice["close"].iloc[-1])
                if not pd.isna(a):
                    self._stop_price = ep - self.atr_stop_mult * a
                return {"action": "buy_sqqq", "size_pct": sz}

        return {"action": "hold"}
