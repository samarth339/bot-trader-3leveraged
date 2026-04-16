"""
Strategy — LongOnly Guard V2  (drawdown-targeted, 30–35% DD goal)

Builds on the Phase 2 winner (150MA, VIX>22/20, confirm=3 → 29% CAGR, 62% DD)
and applies 4 mechanisms to compress drawdown to 30–35% while targeting >30% CAGR.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mechanism 1 — Partial De-risking  (max_position_pct)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Never go 100% into TQQQ. Cap at max_position_pct (e.g. 0.70).
  The remaining fraction sits in cash, cushioning every drawdown
  by (1 - max_position_pct) automatically.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mechanism 2 — Volatility Scaling  (vol_scale)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Continuously adjust the TQQQ allocation as VIX moves:
    VIX < vix_low          → 1.0 × max_position_pct
    VIX in [vix_low, vix_mid) → 0.65 × max_position_pct
    VIX in [vix_mid, vix_exit)→ 0.35 × max_position_pct
    VIX >= vix_exit        → 0.0  (full exit)

  On each bar: compute target_pct. If current allocation >
  target, fire a sell_partial to trim to target. Reductions
  happen before the guard confirms — this is the "early warning"
  layer that prevents large single-bar drops.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mechanism 3 — Staggered Exit  (stagger_exit)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Instead of waiting for full confirm_bars confirmation before
  acting, split the exit into two stages:
    Stage 1 (1 day signal): sell 50% of remaining position
    Stage 2 (confirm_bars days): sell remaining 50%

  Effect: first bad day → half out. Confirmed bad → fully out.
  Reduces the cost of FALSE bear signals (we re-enter sooner
  with less money at risk) and softens true bear entries.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mechanism 4 — Crash Brake  (crash_brake_pct)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Hard floor: if portfolio drawdown from peak exceeds
  crash_brake_pct (e.g. 0.25), exit immediately regardless
  of MA/VIX signals. Prevents tail-risk scenarios where the
  guard fires late (e.g. a flash-crash).

  After crash brake fires, requires full re-entry conditions
  (above MA + VIX below reentry) before getting back in.

Parameters:
    ma_long:           int   default 150
    vix_exit:          float default 22   (exit threshold)
    vix_mid:           float default 19   (reduce to 35%)
    vix_low:           float default 16   (reduce to 65%)
    vix_reentry:       float default 20
    confirm_bars:      int   default 3
    max_position_pct:  float default 0.80  (Mech 1)
    vol_scale:         bool  default True  (Mech 2)
    stagger_exit:      bool  default True  (Mech 3)
    crash_brake_pct:   float default 0.25  (Mech 4, 0 = disabled)
"""

from strategies.base import BaseStrategy


class LongOnlyGuardV2(BaseStrategy):

    def __init__(
        self,
        ma_long:          int   = 150,
        vix_exit:         float = 22.0,
        vix_mid:          float = 19.0,
        vix_low:          float = 16.0,
        vix_reentry:      float = 20.0,
        confirm_bars:     int   = 3,
        max_position_pct: float = 0.80,
        vol_scale:        bool  = True,
        stagger_exit:     bool  = True,
        crash_brake_pct:  float = 0.25,
    ):
        self.ma_long          = ma_long
        self.vix_exit         = vix_exit
        self.vix_mid          = vix_mid
        self.vix_low          = vix_low
        self.vix_reentry      = vix_reentry
        self.confirm_bars     = confirm_bars
        self.max_position_pct = max_position_pct
        self.vol_scale        = vol_scale
        self.stagger_exit     = stagger_exit
        self.crash_brake_pct  = crash_brake_pct

        # State
        self._exit_days       = 0
        self._reentry_days    = 0
        self._guarded         = False   # True = in cash (guard / brake active)
        self._stage1_done     = False   # stagger: first 50% already sold
        self._current_alloc   = 0.0     # fraction of total equity in TQQQ

    def reset(self):
        self._exit_days     = 0
        self._reentry_days  = 0
        self._guarded       = False
        self._stage1_done   = False
        self._current_alloc = 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _vol_target(self, vix: float) -> float:
        """Target TQQQ fraction of total equity based on VIX level."""
        if vix >= self.vix_exit:
            return 0.0
        if vix >= self.vix_mid:
            return self.max_position_pct * 0.35
        if vix >= self.vix_low:
            return self.max_position_pct * 0.65
        return self.max_position_pct

    def _position_alloc(self, pos, equity: float) -> float:
        """Current fraction of equity in TQQQ position."""
        if not pos or equity <= 0:
            return 0.0
        pos_value = pos.shares * pos.entry_price   # approximate, uses entry price
        return pos_value / equity

    # ── Main signal ───────────────────────────────────────────────────────────

    def generate_signal(self, ctx: dict) -> dict:
        i       = ctx["index"]
        qqq     = ctx["qqq"]["close"]
        pos     = ctx["position"]
        equity  = ctx["equity"]
        dd      = ctx["drawdown"]
        vix_val = ctx.get("vix_value") or 15.0   # default to calm if no VIX data

        if i < self.ma_long:
            return {"action": "hold"}

        close    = qqq.iloc[: i + 1]
        price    = float(close.iloc[-1])
        sma      = self.sma(close, self.ma_long)
        below_ma = price < sma
        above_ma = price > sma

        # ── Mechanism 4: Crash Brake ──────────────────────────────────────────
        if self.crash_brake_pct > 0 and dd >= self.crash_brake_pct and pos:
            self._guarded      = True
            self._stage1_done  = False
            self._exit_days    = 0
            self._current_alloc = 0.0
            return {"action": "sell"}

        # ── Guard condition counters ──────────────────────────────────────────
        fear_exit    = vix_val >= self.vix_exit
        fear_low     = vix_val < self.vix_reentry

        if below_ma and fear_exit:
            self._exit_days   += 1
            self._reentry_days = 0
        elif above_ma and fear_low:
            self._reentry_days += 1
            self._exit_days    = 0
        else:
            self._exit_days    = max(0, self._exit_days - 1)
            self._reentry_days = max(0, self._reentry_days - 1)

        # ── In-position state: vol scaling + staggered exit ───────────────────
        if not self._guarded and pos:

            # Mechanism 3: Staggered exit — stage 1 (first bad day)
            if self.stagger_exit and not self._stage1_done:
                if below_ma and fear_exit:
                    self._stage1_done = True
                    return {"action": "sell_partial", "sell_pct": 0.50}

            # Full guard exit after confirm_bars
            if self._exit_days >= self.confirm_bars:
                self._guarded      = True
                self._stage1_done  = False
                self._exit_days    = 0
                self._current_alloc = 0.0
                return {"action": "sell"}

            # Mechanism 2: Vol scaling — trim if VIX has risen
            if self.vol_scale:
                target = self._vol_target(vix_val)
                current = self._position_alloc(pos, equity)
                # If target is meaningfully less than current allocation → trim
                if target < current - 0.10:
                    # Compute what fraction of current position to sell
                    sell_frac = 1.0 - (target / current) if current > 0 else 1.0
                    sell_frac = min(max(sell_frac, 0.0), 1.0)
                    if sell_frac >= 0.95:
                        self._guarded      = True
                        self._stage1_done  = False
                        self._current_alloc = 0.0
                        return {"action": "sell"}
                    if sell_frac > 0.05:
                        return {"action": "sell_partial", "sell_pct": sell_frac}

        # ── Guard state: wait for re-entry ────────────────────────────────────
        if self._guarded:
            if self._reentry_days >= self.confirm_bars:
                self._guarded      = False
                self._stage1_done  = False
                self._reentry_days = 0
                target = self._vol_target(vix_val) if self.vol_scale else self.max_position_pct
                return {"action": "buy_tqqq", "size_pct": target}
            return {"action": "hold"}

        # ── Not guarded and no position → enter ───────────────────────────────
        if not pos:
            self._stage1_done = False
            target = self._vol_target(vix_val) if self.vol_scale else self.max_position_pct
            return {"action": "buy_tqqq", "size_pct": target}

        return {"action": "hold"}
