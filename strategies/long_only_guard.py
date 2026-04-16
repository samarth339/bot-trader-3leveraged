"""
Strategy — TQQQ Long-Only with Macro Crash Guard

Philosophy: TQQQ has ~20% CAGR buy-and-hold. The only thing that destroys it
are rare, severe bear markets (-73% COVID, -85% 2022). Guard against those
specifically. Everything else: stay in TQQQ and let it compound.

Logic:
  DEFAULT STATE: Long TQQQ. Always.

  EXIT to cash when BOTH are true (dual confirmation required):
    1. QQQ closes below ma_long-day MA  (trend broken)
    2. VIX > vix_exit                   (fear confirmed — not just a dip)
    Both must hold for `confirm_bars` consecutive days before exiting.

  RE-ENTRY to TQQQ when BOTH are true:
    1. QQQ closes above ma_long-day MA  (trend restored)
    2. VIX < vix_reentry                (fear receded)
    Both must hold for `confirm_bars` consecutive days before re-entering.

  NEVER use SQQQ. Cash only during guard periods.

Why this beats other strategies:
  - Capital is deployed ~90–95% of the time (vs 15% for Combined v3)
  - TQQQ's positive drift compounds relentlessly during bull markets
  - The dual-condition guard (MA + VIX) avoids false exits on minor pullbacks
  - Cash during crashes preserves capital; no short-side decay risk

Parameters:
    ma_long:       int   default 200
    vix_exit:      float default 28    (exit TQQQ trigger — elevated fear)
    vix_reentry:   float default 22    (re-enter TQQQ trigger — fear normalized)
    confirm_bars:  int   default 3     (days both conditions must hold)
    size_pct:      float default 1.0
"""

from strategies.base import BaseStrategy


class LongOnlyGuardStrategy(BaseStrategy):

    def __init__(
        self,
        ma_long:      int   = 200,
        vix_exit:     float = 28.0,
        vix_reentry:  float = 22.0,
        confirm_bars: int   = 3,
        size_pct:     float = 1.0,
    ):
        self.ma_long      = ma_long
        self.vix_exit     = vix_exit
        self.vix_reentry  = vix_reentry
        self.confirm_bars = confirm_bars
        self.size_pct     = size_pct

        # Consecutive-day counters for confirmation
        self._exit_days    = 0   # days both exit conditions are true
        self._reentry_days = 0   # days both re-entry conditions are true
        self._guarded      = False   # True = in cash (guard active)

    def reset(self):
        self._exit_days    = 0
        self._reentry_days = 0
        self._guarded      = False

    def generate_signal(self, ctx: dict) -> dict:
        i       = ctx["index"]
        qqq     = ctx["qqq"]["close"]
        pos     = ctx["position"]
        vix_val = ctx.get("vix_value")

        if i < self.ma_long:
            return {"action": "hold"}

        close  = qqq.iloc[: i + 1]
        price  = float(close.iloc[-1])
        sma    = self.sma(close, self.ma_long)

        below_ma = price < sma
        above_ma = price > sma

        # VIX conditions (if no VIX data, use MA signal alone)
        if vix_val is not None:
            fear_high = vix_val > self.vix_exit
            fear_low  = vix_val < self.vix_reentry
        else:
            fear_high = below_ma   # fallback: use MA as proxy
            fear_low  = above_ma

        # ── Exit condition counter ────────────────────────────────────────────
        if below_ma and fear_high:
            self._exit_days   += 1
            self._reentry_days = 0
        elif above_ma and fear_low:
            self._reentry_days += 1
            self._exit_days    = 0
        else:
            # Mixed signal — don't progress either counter
            self._exit_days    = max(0, self._exit_days - 1)
            self._reentry_days = max(0, self._reentry_days - 1)

        # ── State machine ─────────────────────────────────────────────────────
        if not self._guarded:
            # Actively in TQQQ — check for exit trigger
            if self._exit_days >= self.confirm_bars:
                self._guarded   = True
                self._exit_days = 0
                return {"action": "sell"}

            # Ensure we're always in TQQQ when not guarded
            if not pos:
                return {"action": "buy_tqqq", "size_pct": self.size_pct}

        else:
            # In cash (guard active) — check for re-entry trigger
            if self._reentry_days >= self.confirm_bars:
                self._guarded      = False
                self._reentry_days = 0
                return {"action": "buy_tqqq", "size_pct": self.size_pct}

        return {"action": "hold"}
