# Phase 2 — Architecture Search Results

## Overview
Tested Phase 2 Opportunity #1: **Adaptive VIX Thresholds** by implementing two variants:
1. **LongOnlyGuardV2Adaptive** - Standard approach: raise thresholds in high-VIX regimes
2. **LongOnlyGuardV2AdaptiveInverted** - Inverse approach: lower thresholds in high-VIX regimes

Both compared against baseline (fixed thresholds from Phase 1).

---

## Backtest Results (Full History: 2010-2026)

### TRAIN Period (2010-02-11 → 2018-12-31, 2237 bars)
| Approach | CAGR | Max DD | Sharpe | Calmar | Final $ |
|----------|------|--------|--------|--------|---------|
| **BASELINE (fixed)** | 28.7% | 35.3% | 0.90 | **0.8133** | **$47,037** |
| Adaptive (standard) | 24.0% | 41.6% | 0.80 | 0.5769 | $44,030 |
| Adaptive (inverted) | 19.8% | 39.3% | 0.74 | 0.5034 | $41,145 |

**Result: Baseline wins by 41% on Calmar**

### VAL Period (2019-01-01 → 2021-12-31, 757 bars, includes COVID)
| Approach | CAGR | Max DD | Sharpe | Calmar | Final $ |
|----------|------|--------|--------|--------|---------|
| **BASELINE (fixed)** | 30.3% | 57.1% | 0.92 | **0.5298** | **$11,040** |
| Adaptive (standard) | 29.4% | 57.1% | 0.90 | 0.5138 | $10,852 |
| Adaptive (inverted) | 30.1% | 57.1% | 0.92 | 0.5263 | $11,000 |

**Result: Baseline still ahead (by 0.7% on Calmar)**

### OOS Period (2022-01-01 → 2025-12-31, 1003 bars)
| Approach | CAGR | Max DD | Sharpe | Calmar | Final $ |
|----------|------|--------|--------|--------|---------|
| **BASELINE (fixed)** | 22.9% | 38.5% | 0.74 | **0.5941** | **$11,387** |
| Adaptive (standard) | 19.9% | 36.0% | 0.70 | 0.5543 | $11,078 |
| Adaptive (inverted) | 16.1% | 32.4% | 0.65 | 0.4962 | $10,784 |

**Result: Baseline wins by 7% on Calmar**

---

## Analysis

### Why Adaptive VIX Thresholds Don't Work

The hypothesis was:
> "Fixed VIX thresholds don't account for volatility regime stability. When VIX is historically elevated, exit thresholds should be raised (less reactive). When VIX is historically calm, thresholds should be lowered (more reactive)."

**Reality checks:**
1. **VIX distribution is highly skewed** (2010-2018):
   - Min: 9.1
   - Mean: 17.0
   - 50th %ile: 15.5
   - Max: 48.0
   - Most days: VIX < 20 (well below thresholds)

2. **Fixed thresholds are already optimized by Phase 1**:
   - 547 experiments tuned parameters extensively
   - Thresholds (vix_exit=25 for Strategy A) were found to be optimal
   - Parameter ceilings reached on bull/uncertain allocations

3. **Adaptive adjustments are too subtle**:
   - Even with strength=2.0, deltas range from ±0.2 to ±2.0
   - Base threshold of 25 rarely active in normal markets
   - During crises (VIX>40), small adjustments don't matter

4. **The real issue is elsewhere**:
   - Baseline already includes 4 drawdown mechanisms:
     - Vol scaling (mechanism 2)
     - Staggered exits (mechanism 3)
     - Crash brake (mechanism 4)
     - MA crossover (mechanism 1)
   - These mechanisms capture most of the risk management

---

## Conclusion

### ❌ Phase 2 Opportunity #1 (Adaptive VIX Thresholds) is **NOT VIABLE**

**Evidence:**
- All three approaches tested (baseline, adaptive standard, adaptive inverted)
- Baseline wins on ALL periods (train, val, OOS)
- Adaptive adds complexity without benefit
- Theory-inspired but empirically defeated

### 🔒 Baseline Configuration is LOCKED

The Phase 1 baseline configuration is provably optimal (vs tested alternatives):
```
STRATEGY_A_CONFIG:
  ma_long: 190
  vix_exit: 25
  vix_reentry: 24
  confirm_bars: 2
  max_position_pct: 0.95
  stagger_exit: True

STRATEGY_B_CONFIG:
  ma_long: 150
  vix_exit: 28
  vix_reentry: 22
  confirm_bars: 4
  max_position_pct: 0.70
  crash_brake_pct: 0.30
  stagger_exit: True

ALLOCATIONS:
  bull: 90% A / 10% B
  uncertain: 65% A / 35% B
  high_vol: 25% A / 75% B
```

**Composite Score: 0.6564** (target: 0.67) — Phase 1 is exhausted

---

## Next Steps

### Option A: Phase 2 Opportunity #2 (Regime Confidence Scoring)
- Requires implementing confidence-weighted allocation blending
- Expected impact: +1–2% Calmar
- Risk: Introduces new parameters, no autosearch budget left

### Option B: Phase 2 Opportunity #3 (Asymmetric Position Sizing)
- Vary max_position_pct by regime (bull: 0.95, uncertain: 0.75, high_vol: 0.50)
- Expected impact: +0.3–0.5% Calmar
- Risk: Minimal
- Budget: Could explore with remaining API credits

### Option C: Declare Optimization Complete
- Phase 1 + 2 have not found improvements beyond baseline
- Baseline score (0.6564) is robust across train/val/OOS
- Confidence in production deployment is high
- Shadow mode operating normally

---

## Recommendation

**DECLARE PHASE 2 EXPLORATION INCONCLUSIVE**
- Adaptive VIX Thresholds hypothesis was wrong
- No other Phase 2 opportunities tested yet
- Autoresearch exhausted on Phase 1 parameters
- API budget nearly depleted

**ACTION:** Lock baseline, move to production (Phase 3 shadow mode + Phase 4 paper trading)

If future market conditions warrant further optimization:
- Revisit Phase 2 #2 and #3 with fresh funding
- Consider different markets (QQQ alternatives, other leveraged ETFs)
- Explore machine learning approaches for regime detection

---

## Supporting Files

- `/Volumes/SAM/bot-test/strategies/long_only_guard_v2_adaptive.py` - Standard adaptive (archived)
- `/Volumes/SAM/bot-test/strategies/long_only_guard_v2_adaptive_inverted.py` - Inverted adaptive (archived)
- `/Volumes/SAM/bot-test/test_phase2_adaptive.py` - Full backtest comparison script
