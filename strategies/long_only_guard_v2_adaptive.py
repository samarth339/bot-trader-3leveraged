"""
Strategy — LongOnly Guard V2 Adaptive  (Phase 2 architecture enhancement)

Extends LongOnlyGuardV2 with adaptive VIX thresholds that adjust based on
VIX historical volatility context. When VIX is historically elevated, exit
thresholds are raised (less reactive). When VIX is historically calm,
thresholds are lowered (more reactive).

Theory:
- Fixed thresholds don't account for regime volatility
- COVID crash: VIX=89, but signal is "clear" (real crisis)
- Post-recovery: VIX=15, but signal is "noisy" (minor ups/downs)
- Adaptive thresholds match sensitivity to actual signal clarity

Parameters (new):
    vix_percentile_adapt:  bool  default False (enable/disable this feature)
    vix_adapt_window:      int   default 252 (1-year rolling percentile)
    vix_adapt_strength:    float default 1.0 (0–2: scaling factor for adaptivity)
"""

from strategies.base import BaseStrategy
import numpy as np


class LongOnlyGuardV2Adaptive(BaseStrategy):

    def __init__(
        self,
        ma_long:           int   = 150,
        vix_exit:          float = 22.0,
        vix_mid:           float = 19.0,
        vix_low:           float = 16.0,
        vix_reentry:       float = 20.0,
        confirm_bars:      int   = 3,
        max_position_pct:  float = 0.80,
        vol_scale:         bool  = True,
        stagger_exit:      bool  = True,
        crash_brake_pct:   float = 0.25,
        vix_percentile_adapt: bool = False,  # NEW
        vix_adapt_window:  int   = 252,      # NEW
        vix_adapt_strength: float = 1.0,     # NEW
    ):
        self.ma_long           = ma_long
        self.vix_exit          = vix_exit
        self.vix_mid           = vix_mid
        self.vix_low           = vix_low
        self.vix_reentry       = vix_reentry
        self.confirm_bars      = confirm_bars
        self.max_position_pct  = max_position_pct
        self.vol_scale         = vol_scale
        self.stagger_exit      = stagger_exit
        self.crash_brake_pct   = crash_brake_pct
        self.vix_percentile_adapt = vix_percentile_adapt  # NEW
        self.vix_adapt_window  = vix_adapt_window          # NEW
        self.vix_adapt_strength = vix_adapt_strength       # NEW

        # State
        self._exit_days       = 0
        self._reentry_days    = 0
        self._guarded         = False
        self._stage1_done     = False
        self._current_alloc   = 0.0

    def reset(self):
        self._exit_days     = 0
        self._reentry_days  = 0
        self._guarded       = False
        self._stage1_done   = False
        self._current_alloc = 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_vix_percentile(self, vix_series, current_idx: int) -> float:
        """
        Compute the percentile of current VIX within historical context.
        Returns 0–1 where 0 = all-time low seen, 1 = all-time high seen.
        """
        if current_idx < self.vix_adapt_window:
            window = vix_series.iloc[: current_idx + 1]
        else:
            window = vix_series.iloc[current_idx - self.vix_adapt_window : current_idx + 1]

        if len(window) < 2:
            return 0.5  # Default to neutral

        sorted_vals = np.sort(window.values)
        current_val = window.iloc[-1]
        # Percentile: what fraction of the window is <= current_val
        percentile = np.mean(sorted_vals <= current_val)
        return np.clip(percentile, 0.0, 1.0)

    def _adapt_thresholds(self, vix_percentile: float) -> tuple:
        """
        Adjust vix_exit, vix_mid, vix_low, vix_reentry based on percentile.

        When VIX percentile is high (70th+), thresholds are raised (less reactive).
        When VIX percentile is low (30th-), thresholds are lowered (more reactive).

        Returns: (vix_exit, vix_mid, vix_low, vix_reentry)
        """
        # Delta from neutral (0.5)
        delta = (vix_percentile - 0.5) * 2  # scales to [-1, 1]
        # Scale by strength parameter
        delta *= self.vix_adapt_strength

        # Adjust thresholds
        vix_exit_adj = self.vix_exit + delta
        vix_mid_adj = self.vix_mid + delta
        vix_low_adj = self.vix_low + delta
        vix_reentry_adj = self.vix_reentry + delta

        return vix_exit_adj, vix_mid_adj, vix_low_adj, vix_reentry_adj

    def _vol_target(self, vix: float, vix_exit: float, vix_mid: float, vix_low: float, max_pos: float) -> float:
        """Target TQQQ fraction based on VIX level and (possibly) adapted thresholds."""
        if vix >= vix_exit:
            return 0.0
        if vix >= vix_mid:
            return max_pos * 0.35
        if vix >= vix_low:
            return max_pos * 0.65
        return max_pos

    def _position_alloc(self, pos, equity: float) -> float:
        """Current fraction of equity in TQQQ position."""
        if not pos or equity <= 0:
            return 0.0
        pos_value = pos.shares * pos.entry_price
        return pos_value / equity

    # ── Main signal ───────────────────────────────────────────────────────────

    def generate_signal(self, ctx: dict) -> dict:
        i        = ctx["index"]
        qqq      = ctx["qqq"]["close"]
        vix_data = ctx.get("vix_close")  # Full VIX series
        pos      = ctx["position"]
        equity   = ctx["equity"]
        dd       = ctx["drawdown"]
        vix_val  = ctx.get("vix_value") or 15.0

        if i < self.ma_long:
            return {"action": "hold"}

        close    = qqq.iloc[: i + 1]
        price    = float(close.iloc[-1])
        sma      = self.sma(close, self.ma_long)
        below_ma = price < sma
        above_ma = price > sma

        # ── NEW: Compute adapted thresholds if enabled ──────────────────────────
        if self.vix_percentile_adapt and vix_data is not None:
            vix_pct = self._compute_vix_percentile(vix_data, i)
            vix_exit_adj, vix_mid_adj, vix_low_adj, vix_reentry_adj = self._adapt_thresholds(vix_pct)
        else:
            # Fall back to fixed thresholds
            vix_exit_adj = self.vix_exit
            vix_mid_adj = self.vix_mid
            vix_low_adj = self.vix_low
            vix_reentry_adj = self.vix_reentry

        # ── Mechanism 4: Crash Brake ──────────────────────────────────────────
        if self.crash_brake_pct > 0 and dd >= self.crash_brake_pct and pos:
            self._guarded      = True
            self._stage1_done  = False
            self._exit_days    = 0
            self._current_alloc = 0.0
            return {"action": "sell"}

        # ── Guard condition counters (using adapted thresholds) ────────────────
        fear_exit    = vix_val >= vix_exit_adj
        fear_low     = vix_val < vix_reentry_adj

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

            # Mechanism 3: Staggered exit — stage 1
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

            # Mechanism 2: Vol scaling
            if self.vol_scale:
                target = self._vol_target(vix_val, vix_exit_adj, vix_mid_adj, vix_low_adj, self.max_position_pct)
                current = self._position_alloc(pos, equity)
                if target < current - 0.10:
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
                target = self._vol_target(vix_val, vix_exit_adj, vix_mid_adj, vix_low_adj, self.max_position_pct) if self.vol_scale else self.max_position_pct
                return {"action": "buy_tqqq", "size_pct": target}
            return {"action": "hold"}

        # ── Not guarded and no position → enter ───────────────────────────────
        if not pos:
            self._stage1_done = False
            target = self._vol_target(vix_val, vix_exit_adj, vix_mid_adj, vix_low_adj, self.max_position_pct) if self.vol_scale else self.max_position_pct
            return {"action": "buy_tqqq", "size_pct": target}

        return {"action": "hold"}
