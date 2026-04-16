# Trading Strategy Optimization Program

## Objective
Maximize `composite_score` = geometric mean of train-period Calmar × val-period Calmar.
A config is only accepted if it improves on BOTH periods simultaneously.

Current baseline (as of 2026-03-27 — post autoresearch v1, 388 experiments, 14 wins):
  Train (2010–2018): CAGR 28.7%, MaxDD 35.3%, Calmar 0.813
  Val   (2019–2021): CAGR 30.3%, MaxDD 57.1%, Calmar 0.530  ← COVID crash included
  OOS   (2022–2025): CAGR 23.7%, MaxDD 38.5%, Calmar 0.614  ← reported only
  Composite score  : 0.6564  ← sqrt(0.813 × 0.530)

  NOTE: Val period covers COVID (March 2020, VIX=89). MaxDD is naturally high.
  Veto thresholds are calibrated BELOW baseline so the agent has room to improve.
  GOAL: Improve Calmar on val period without increasing MaxDD. Focus on defensive
  parameter tuning (crash_brake, vix_smooth, high_vol allocation, ma_window).

## Evaluation Periods (NEVER CHANGE THESE)
  Train : 2010-02-11 → 2018-12-31
  Val   : 2019-01-01 → 2021-12-31  ← includes COVID crash (expected high DD)
  OOS   : 2022-01-01 → 2025-12-31  ← reported only, never used to accept/reject

## Hard Veto Constraints (score = 0.0 if ANY violated)
  Train CAGR     ≥ 24%     (baseline 29.6%)
  Train MaxDD    ≤ 42%     (baseline 39.1%)
  Train Sharpe   ≥ 0.80    (baseline 0.918)
  Val CAGR       ≥ 12%     (baseline 18.0%)
  Val MaxDD      ≤ 65%     (baseline 55.5% — COVID inflated)
  Val Calmar     ≥ 0.20    (baseline 0.325)

## Phase 1 — Parameter Search (CURRENT PHASE)
Only modify: `config/strategy_config.py`
No other file may be touched.

### Safe ranges for each parameter (CURRENT values as of 2026-03-27):
```
REGIME_CONFIG:
  ma_window:    100–250   (currently 130)  ← was 150, shortened to 130
  vix_bull:     14.0–22.0 (currently 18.0)
  vix_hi_vol:   20.0–32.0 (currently 25.0)
  vix_smooth:   3–10      (currently 5)

STRATEGY_A (BestCalmar):
  ma_long:         150–300  (currently 190)  ← was 200, shortened to 190
  vix_exit:         20–32   (currently 25)
  vix_reentry:      18–28   (currently 24)   ← was 22, raised to 24
  confirm_bars:      1–5    (currently 2)
  max_position_pct: 0.70–1.00 (currently 0.95)  ← was 0.90, raised to 0.95

STRATEGY_B (NearMiss):
  ma_long:          100–200  (currently 150)
  vix_exit:          22–35   (currently 28)
  vix_reentry:       18–28   (currently 22)   ← raised to 24 was tested and reverted
  confirm_bars:       1–8    (currently 4)
  max_position_pct:  0.50–0.80 (currently 0.70)
  crash_brake_pct:   0.10–0.40 (currently 0.30)  ← UNEXPLORED UPSIDE: try 0.35–0.40

ALLOC_CONFIG (A weight shown; B = 1 - A):
  bull:      A in 0.60–0.90  (currently 0.90)  ← AT CEILING — do not push further
  uncertain: A in 0.40–0.65  (currently 0.65)  ← AT CEILING — do not push further
  high_vol:  A in 0.15–0.50  (currently 0.25)  ← UNEXPLORED: try 0.15–0.20 for less DD

PRIORITY TARGETS (unexplored, low downside risk):
  1. STRATEGY_B_CONFIG.crash_brake_pct  → try 0.35, 0.40
  2. ALLOC_CONFIG.high_vol              → try (0.20, 0.80) or (0.15, 0.85)
  3. REGIME_CONFIG.vix_smooth           → try 3 or 4 (faster VIX reaction)
  4. REGIME_CONFIG.ma_window            → try 115–125
  5. STRATEGY_A_CONFIG.ma_long          → try 200–210
```

## Phase 2 — Architecture Search (FUTURE, do not start yet)
Mutable: `strategies/long_only_guard_v2.py`
Requirements: all 231 pytest tests must pass after any architectural change.

## Rules (non-negotiable)
1. `t1_execution` must always be True. Never disable it.
2. Never touch: backtester/, tests/, data/, ibkr/, daily_signal.py, shadow_mode.py
3. Never re-test a config already in results.tsv (check before proposing)
4. If two configs score equally, prefer the simpler one (fewer parameter changes from baseline)
5. Read git log of last 10 commits before proposing — avoid cycling back to failed configs

## Known Dead Ends (do not re-test)
- confidence_weighted_allocation: Calmar 0.607 < baseline. Disabled permanently.
- confirm_days > 2 on VIX: Raises MaxDD. Avoid.
- crash_brake_pct standalone (without stagger_exit): Creates exit/re-enter loops. Do not test.
- vix_hi_vol < 20: Triggers too frequently, excessive regime churn.
- ma_window < 80: Regime too reactive to short-term noise.

### Confirmed failures from autoresearch v1 (388 experiments):
- Strategy B vix_exit: 26, 27, 29, 30, 31 — all failed, current 28 is optimal
- Strategy B max_position_pct: 0.60, 0.65, 0.68, 0.72, 0.75 — all failed, current 0.70 is optimal
- Strategy B ma_long > 150: 155–180 all failed — do not go higher
- Strategy B vix_reentry: 20, 21, 23, 24 — all failed, current 22 is optimal
- Allocation bull > 0.90: ceiling reached, more A in bull increases DD without improving Calmar
- Allocation uncertain > 0.65: ceiling reached, same reason
