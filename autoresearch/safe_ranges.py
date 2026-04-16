"""
Safe Ranges Validator
======================
Before spending a Claude API call on a backtest, check that the proposed
parameter value is within the bounds defined in program.md.

All ranges are hard-coded here as the single source of truth.
If you want to expand a range, change it here AND in program.md.

Usage:
    from autoresearch.safe_ranges import validate, SAFE_RANGES, DEAD_ENDS

    ok, reason = validate("STRATEGY_A_CONFIG.vix_exit", 23)
    # ok=True, reason=""

    ok, reason = validate("REGIME_CONFIG.t1_execution", False)
    # ok=False, reason="t1_execution must always be True"
"""

from typing import Any, Tuple, Optional

# ── Safe ranges per parameter (inclusive) ─────────────────────────────────────
# Format: "DICT.key": (min, max) or (set_of_valid_values,) for discrete
# Tuples for ALLOC_CONFIG: range applies to the A weight (first element)

SAFE_RANGES = {
    # REGIME_CONFIG
    "REGIME_CONFIG.ma_window":      (100,  250),
    "REGIME_CONFIG.vix_bull":       (14.0, 22.0),
    "REGIME_CONFIG.vix_hi_vol":     (20.0, 32.0),
    "REGIME_CONFIG.vix_smooth":     (3,    10),
    "REGIME_CONFIG.confirm_days":   (1,    3),
    "REGIME_CONFIG.t1_execution":   "LOCKED_TRUE",   # never change

    # ALLOC_CONFIG — A weight range (B = 1 - A)
    "ALLOC_CONFIG.bull":            (0.60, 0.90),
    "ALLOC_CONFIG.uncertain":       (0.40, 0.65),
    "ALLOC_CONFIG.high_vol":        (0.15, 0.50),

    # STRATEGY_A_CONFIG
    "STRATEGY_A_CONFIG.ma_long":           (150, 300),
    "STRATEGY_A_CONFIG.vix_exit":          (20,  32),
    "STRATEGY_A_CONFIG.vix_reentry":       (18,  28),
    "STRATEGY_A_CONFIG.confirm_bars":      (1,   5),
    "STRATEGY_A_CONFIG.max_position_pct":  (0.70, 1.00),
    "STRATEGY_A_CONFIG.stagger_exit":      "LOCKED_TRUE",
    "STRATEGY_A_CONFIG.crash_brake_pct":   (0.0, 0.0),   # must stay 0 for A
    "STRATEGY_A_CONFIG.vol_scale":         "BOOL",

    # STRATEGY_B_CONFIG
    "STRATEGY_B_CONFIG.ma_long":           (100, 200),
    "STRATEGY_B_CONFIG.vix_exit":          (22,  35),
    "STRATEGY_B_CONFIG.vix_reentry":       (18,  28),
    "STRATEGY_B_CONFIG.confirm_bars":      (1,   8),
    "STRATEGY_B_CONFIG.max_position_pct":  (0.50, 0.80),
    "STRATEGY_B_CONFIG.stagger_exit":      "LOCKED_TRUE",
    "STRATEGY_B_CONFIG.crash_brake_pct":   (0.10, 0.40),
    "STRATEGY_B_CONFIG.vol_scale":         "BOOL",
}

# ── Constraints between parameters ────────────────────────────────────────────
# Checked AFTER individual range validation.
# Format: (description, lambda current_config: True_if_valid)

CROSS_CONSTRAINTS = [
    (
        "vix_bull must be < vix_hi_vol (regime ordering)",
        lambda c: c.get("REGIME_CONFIG.vix_bull", 18.0) < c.get("REGIME_CONFIG.vix_hi_vol", 25.0),
    ),
    (
        "Strategy A vix_reentry must be < vix_exit (re-enter before exit triggers again)",
        lambda c: c.get("STRATEGY_A_CONFIG.vix_reentry", 22) < c.get("STRATEGY_A_CONFIG.vix_exit", 25),
    ),
    (
        "Strategy B vix_reentry must be < vix_exit",
        lambda c: c.get("STRATEGY_B_CONFIG.vix_reentry", 22) < c.get("STRATEGY_B_CONFIG.vix_exit", 28),
    ),
    (
        "Strategy A ma_long must be >= REGIME ma_window (A is more trend-following)",
        lambda c: c.get("STRATEGY_A_CONFIG.ma_long", 200) >= c.get("REGIME_CONFIG.ma_window", 150),
    ),
]

# ── Known dead ends (skip without running backtest) ────────────────────────────
# Each entry: (param, value_or_None) where None means "any value for this param"
DEAD_ENDS = [
    # Confidence-weighted allocation: tested and worse (Calmar 0.607 < baseline 0.681)
    # Not a direct parameter but agent might propose vol_scale=True on both
    # which effectively creates confidence-weighted behaviour
    ("REGIME_CONFIG.confirm_days", 2),   # confirm_days>1 raises DD — tested
    ("REGIME_CONFIG.confirm_days", 3),
    ("REGIME_CONFIG.vix_hi_vol", 18.0),  # Too frequent, churn
    ("REGIME_CONFIG.vix_hi_vol", 19.0),
    ("REGIME_CONFIG.ma_window", 80),
    ("REGIME_CONFIG.ma_window", 90),

    # ── Confirmed failures from autoresearch v1 (388 experiments) ─────────────
    # Strategy B vix_exit: all directions tested and failed (29 skips)
    ("STRATEGY_B_CONFIG.vix_exit", 26),
    ("STRATEGY_B_CONFIG.vix_exit", 27),
    ("STRATEGY_B_CONFIG.vix_exit", 29),
    ("STRATEGY_B_CONFIG.vix_exit", 30),
    ("STRATEGY_B_CONFIG.vix_exit", 31),

    # Strategy B max_position_pct: all directions failed
    ("STRATEGY_B_CONFIG.max_position_pct", 0.60),
    ("STRATEGY_B_CONFIG.max_position_pct", 0.65),
    ("STRATEGY_B_CONFIG.max_position_pct", 0.68),
    ("STRATEGY_B_CONFIG.max_position_pct", 0.72),
    ("STRATEGY_B_CONFIG.max_position_pct", 0.75),

    # Strategy B ma_long: going higher failed consistently
    ("STRATEGY_B_CONFIG.ma_long", 155),
    ("STRATEGY_B_CONFIG.ma_long", 160),
    ("STRATEGY_B_CONFIG.ma_long", 165),
    ("STRATEGY_B_CONFIG.ma_long", 170),
    ("STRATEGY_B_CONFIG.ma_long", 175),
    ("STRATEGY_B_CONFIG.ma_long", 180),

    # Strategy B vix_reentry: tightening failed, widening failed
    ("STRATEGY_B_CONFIG.vix_reentry", 20),
    ("STRATEGY_B_CONFIG.vix_reentry", 21),
    ("STRATEGY_B_CONFIG.vix_reentry", 23),
    ("STRATEGY_B_CONFIG.vix_reentry", 24),  # was tested and reverted

    # Allocation: both bull and uncertain are at their safe range ceilings
    # — agent should not propose higher values (they're already maxed)
    ("ALLOC_CONFIG.bull", (0.95, 0.05)),
    ("ALLOC_CONFIG.uncertain", (0.70, 0.30)),
    ("ALLOC_CONFIG.uncertain", (0.75, 0.25)),
]


def validate(param: str, new_value: Any) -> Tuple[bool, str]:
    """
    Check that (param, new_value) is safe to apply.

    Returns (True, "") if OK.
    Returns (False, reason_string) if not.

    param format: "DICT_NAME.key_name"
    """
    # ── Locked parameters ──────────────────────────────────────────────────────
    if param == "REGIME_CONFIG.t1_execution":
        if new_value is not True:
            return False, "t1_execution must always be True — NON-NEGOTIABLE"
        return True, ""

    if param in ("STRATEGY_A_CONFIG.stagger_exit", "STRATEGY_B_CONFIG.stagger_exit"):
        if new_value is not True:
            return False, f"{param} must stay True — stagger_exit is a core protection mechanism"
        return True, ""

    # ── ALLOC_CONFIG tuple handling ────────────────────────────────────────────
    if param.startswith("ALLOC_CONFIG."):
        if isinstance(new_value, (list, tuple)) and len(new_value) == 2:
            a_weight = float(new_value[0])
            b_weight = float(new_value[1])
            if abs(a_weight + b_weight - 1.0) > 0.001:
                return False, f"Allocation weights must sum to 1.0 (got {a_weight}+{b_weight}={a_weight+b_weight})"
            rng = SAFE_RANGES.get(param)
            if rng and isinstance(rng, tuple):
                lo, hi = rng
                if not (lo <= a_weight <= hi):
                    return False, f"{param} A-weight {a_weight} outside range [{lo}, {hi}]"
            return True, ""
        return False, f"ALLOC_CONFIG value must be a 2-element list/tuple, got {type(new_value)}"

    # ── Standard range check ───────────────────────────────────────────────────
    spec = SAFE_RANGES.get(param)
    if spec is None:
        return False, f"Parameter '{param}' is not in the safe ranges list — do not modify"

    if spec == "BOOL":
        if not isinstance(new_value, bool):
            return False, f"{param} must be a boolean"
        return True, ""

    if isinstance(spec, tuple) and len(spec) == 2:
        lo, hi = spec
        try:
            v = float(new_value) if isinstance(lo, float) else int(new_value)
        except (ValueError, TypeError):
            return False, f"{param} value {new_value!r} is not numeric"

        if not (lo <= v <= hi):
            return False, f"{param} value {v} outside safe range [{lo}, {hi}]"

        # Special: crash_brake_pct for Strategy A must stay 0
        if param == "STRATEGY_A_CONFIG.crash_brake_pct" and v != 0.0:
            return False, "Strategy A crash_brake_pct must stay 0.0 (standalone brake = death loop)"

        return True, ""

    return True, ""


def is_dead_end(param: str, new_value: Any) -> Tuple[bool, str]:
    """
    Check if this exact (param, value) pair is a known dead end.
    Returns (True, reason) if it's a known failure, (False, "") otherwise.
    """
    for de_param, de_value in DEAD_ENDS:
        if de_param == param:
            if de_value is None or de_value == new_value:
                return True, f"{param}={new_value} is a known dead end — skip"
    return False, ""


def check_cross_constraints(current_params: dict) -> Tuple[bool, str]:
    """
    Check constraints between parameters after a proposed change.

    current_params: dict of "DICT.key" → current value for all relevant params.
    Returns (True, "") if all constraints pass.
    """
    for description, constraint_fn in CROSS_CONSTRAINTS:
        try:
            if not constraint_fn(current_params):
                return False, description
        except Exception:
            pass
    return True, ""


def describe_ranges() -> str:
    """Return a human-readable summary of all safe ranges (for prompts)."""
    lines = []
    for param, spec in SAFE_RANGES.items():
        if spec == "LOCKED_TRUE":
            lines.append(f"  {param}: LOCKED=True")
        elif spec == "BOOL":
            lines.append(f"  {param}: True or False")
        elif isinstance(spec, tuple):
            lo, hi = spec
            lines.append(f"  {param}: [{lo}, {hi}]")
    return "\n".join(lines)
