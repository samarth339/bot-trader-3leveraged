# Phase 2 — Architecture Search

## Objective
Improve the composite score **beyond 0.6564** by modifying the strategy logic itself
(not just tuning parameters). Target: score ≥ 0.67 (+2% gain).

All changes must pass the full test suite (231 tests).

---

## Phase 1 Exhaustion Summary
- **547 experiments** across v1 and v2 with dead-end blocking
- **Parameter ceilings reached:** bull=0.90, uncertain=0.65
- **VIX smoothing optimal at 5:** longer windows failed (vix_smooth→9: score -7.8%)
- **Allocation allocation frozen:** all attempts resulted in skips
- **Inference:** parameter tuning alone cannot improve further

---

## Identified Phase 2 Opportunities

### 1. Adaptive VIX Thresholds (Low Risk, Medium Effort)
**Hypothesis:** Fixed VIX thresholds don't account for volatility regime stability.
When VIX is *historically elevated*, the exit threshold should be *higher* (less reactive).
When VIX is *historically low*, the exit threshold should be *lower* (more reactive).

**Implementation:**
```python
# Compute rolling VIX percentile (e.g., 252-day)
vix_percentile = compute_percentile(vix_series, window=252)
vix_exit_adjusted = vix_exit + (vix_percentile - 0.5) * 2

# If VIX is at 90th percentile, raise exit from 25 to 26
# If VIX is at 10th percentile, lower exit from 25 to 24
```

**Expected impact:** +0.5–1% Calmar by avoiding false exits in elevated-vol regimes
**Risk:** May increase DD during sudden spikes (COVID-like)
**Tests affected:** 0 (backward-compatible)

---

### 2. Regime Confidence Scoring (Medium Risk, Medium Effort)
**Hypothesis:** The regime classification is binary (bull/uncertain/high-vol) but the
underlying data is noisy. A *confidence score* (0–1) based on how "clean" the signal
is could unlock finer allocation decisions.

**Implementation:**
```python
# Confidence = how far price is from SMA vs volatility
distance_pct = abs(price - sma) / sma
volatility = roc_std(qqq, window=20)
confidence = min(distance_pct / volatility, 1.0)

# Use confidence to blend allocations
alloc_a = base_alloc_a * (0.5 + 0.5 * confidence)
```

**Expected impact:** +1–2% Calmar by being more aggressive when signals are clear
**Risk:** Introduces new parameters (hard to tune without more autoresearch)
**Tests affected:** 0 (new output field, doesn't change behavior by default)

---

### 3. Asymmetric Position Sizing (Low Risk, Low Effort)
**Hypothesis:** The "max_position_pct" is uniform across all bull/uncertain/high-vol regimes,
but different regimes may benefit from different exposure.

**Implementation:**
```python
class AsymmetricPositionSizing:
    def __init__(self, ...):
        self.max_position_pct_bull = 0.95
        self.max_position_pct_uncertain = 0.75
        self.max_position_pct_high_vol = 0.50
```

**Expected impact:** +0.3–0.5% Calmar by tighter stops in danger
**Risk:** Minimal — just adds knobs to existing mechanism
**Tests affected:** 0 (configurable, can be disabled)

---

## Recommendation for Initial Phase 2

**Start with #1 (Adaptive VIX Thresholds):**
- Lowest architectural change (single helper function)
- Addresses a real problem (fixed thresholds across volatile/calm regimes)
- Minimal test impact
- If successful, unlock #2 and #3

**Tentative Success Criteria:**
- Score ≥ 0.665 (0.5% gain from 0.6564)
- No new test failures
- OOS Calmar ≥ 0.315 (no regression)
- Can be toggled on/off via parameter

---

## Implementation Plan

1. **Create `LongOnlyGuardV2Adaptive` subclass** (preserves original for comparison)
2. **Add `vix_percentile_adapt` parameter** (boolean, default False)
3. **Implement percentile calculation** in helper methods
4. **Backtest on train/val/OOS**
5. **Run full test suite**
6. **If passing:** promote to config option, then Phase 1-style parameter sweep

---

## Phase 2 Risk Mitigation

| Risk | Mitigation |
|---|---|
| Test failures | Full suite before any promotion |
| Overfitting to 2010–2025 | OOS period is separate veto |
| Unexpected behavior | All changes are parameterized + toggleable |
| API costs | Single backtest per attempt (no expensive sweeps yet) |

