"""
pricing.py — single source of truth for "is this a usable price?"

Every external price read (yfinance, CSV) must pass through is_valid_price()
before it drives sizing or a guard decision. A NaN / inf / ≤0 price must NEVER
reach the sizing math — the 2026-09-28 incident (NaN close → phantom
"sell all shares @ $nan") is exactly what this prevents. Fail-safe: an invalid
price is treated as "no data", so the caller aborts rather than trades on garbage.
"""
from __future__ import annotations

import math


def is_valid_price(x) -> bool:
    """True only for a finite, strictly-positive number."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0.0
