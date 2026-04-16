# CLAUDE.md — Trading Bot Context
## Project: TQQQ/SQQQ Algorithmic Trading System

---

## What This Project Is
A systematic trading bot trading **TQQQ** (3x long NASDAQ) and **SQQQ** (3x short NASDAQ),
using **QQQ** as the clean signal source. Currently in **Phase 3 shadow mode** (30-day
live observation, no real trades). Goal: $775K+ from $5K seed over ~10 years.

---

## Critical Rules — Read Before Touching Anything

1. **T-1 is non-negotiable.** All regime signals use `close.shift(1)` and
   `vix.rolling(5).mean().shift(1)`. Never use same-bar data. The hard guard
   is in `backtester/dual_portfolio.py → _build_signal_inputs()`.

2. **Never add frequent-trading strategies for TQQQ/SQQQ.** 3x decay + slippage
   destroys capital at >100 trades/year. Any strategy generating daily signals
   on leveraged ETFs will eventually go to zero. Lesson learned the hard way.

3. **Single source of truth:** `config/strategy_config.py`. All thresholds
   (VIX levels, MA windows, allocations) live there. Never hardcode in strategies.

4. **Shadow mode = no trades.** `shadow_mode.py` and `daily_signal.py --shadow`
   log signals only. Do not add execution code without explicit approval.

---

## Locked Production Configuration

| Parameter | Value |
|---|---|
| Strategy A (Aggressive) | LongOnlyGuardV2: ma=190, vix_exit=25, vix_reentry=23, max_pos=85%, stagger=True |
| Strategy B (Defensive) | LongOnlyGuardV2: ma=150, vix_exit=28, vix_reentry=24, max_pos=60%, vol_scale=True, crash_brake=30% |
| Bull allocation | 90% A / 10% B |
| Mid (uncertain) allocation | 65% A / 35% B |
| High-vol allocation | 25% A / 75% B |
| Regime MA window | 130-day SMA on QQQ |
| VIX smoothing | 5-day rolling average |
| VIX bull threshold | < 18.0 |
| VIX high-vol threshold | ≥ 25.0 |
| T-1 execution | Always ON |

**Target performance:** CAGR >30% | Max DD 30–35%

**Backtest performance (2010–2025, honest T-1) — post-autoresearch v2:**
- Train (2010–2022): CAGR 28.7% | Max DD 35.3% | Calmar 0.813
- Val (2022–2025): CAGR 30.3% | Max DD 57.1% | Calmar 0.530
- OOS (2019–2025): CAGR 17.2% | Max DD 55.4% | Calmar 0.310
- Composite score: 0.6564 (vs 0.4964 baseline, +32.2%)
- Note: OOS Calmar (0.310) below 0.40 target — 2019–2025 includes COVID crash + 2022 rate-hike bear

**Post expert-panel v2 (2026-04-16) — full period 2010–2026:**
- CAGR 20.6% | Max DD 37.7% | Calmar 0.55 | Sharpe 0.76 | Final $103K (from $5K)
- Trade-off: −6pp CAGR vs old sizing, but Max DD compressed from ~55% → 37.7%
- Changes: A max_pos 0.95→0.85, B max_pos 0.70→0.60 + vol_scale=True, ROC-5 momentum override, dynamic uncertain allocation

**Key changes from autoresearch (388 experiments, 14 wins):**
- ma_long: 200 → 190 (faster signal, less lag)
- vix_reentry A: 22 → 23 | B: 22 → 24 (wider hysteresis, less whipsaw)
- Regime MA: 150 → 130 (faster regime detection)
- Bull/uncertain allocations shifted heavily toward Strategy A
- Strategy A max_position_pct: 0.90 → 0.95

**Key changes from expert-panel v2 (2026-04-16):**
- Strategy A max_position_pct: 0.95 → 0.85 (gap-risk cash buffer)
- Strategy B max_position_pct: 0.70 → 0.60 (gap-risk cash buffer)
- Strategy B vol_scale: False → True (gradual VIX-tier de-risking during spikes)
- ROC-5 momentum override: high_vol → uncertain when 5-day QQQ momentum > +3% AND price within 1.5% of SMA (faster re-entry from V-recoveries)
- Dynamic uncertain allocation: Strategy A weight scales 45–75% based on pct_vs_sma (vs fixed 65%)
- Tests: +21 new tests (252 total, 1 skipped)

---

## File Structure

```
/Volumes/SAM/bot-test/
├── config/
│   ├── settings.py              # Global constants (tickers, paths, capital) — BACKTEST_END is dynamic
│   └── strategy_config.py       # ALL strategy parameters — single source of truth
├── strategies/
│   ├── base.py                  # BaseStrategy + rsi(), atr() helpers
│   ├── long_only_guard.py       # V1: simple TQQQ hold + dual MA/VIX exit
│   └── long_only_guard_v2.py    # V2: adds max_position_pct, vol_scale, stagger_exit, crash_brake
│   └── [10 other retired strategy files — kept for reference]
├── backtester/
│   ├── engine.py                # Backtester + Portfolio.sell_partial()
│   └── dual_portfolio.py        # DualPortfolioBacktester (v3 production-hardened)
├── ibkr/                        # Phase 4/5 live trading (not yet active)
│   ├── kill_switch.py           # Emergency stop
│   ├── position_reconciler.py   # Post-execution verification
│   ├── safety_guard.py          # Pre-trade checks
│   └── [executor, client, account, order_manager, state]
├── analysis/
│   ├── metrics.py               # Print metrics, stress-test reports, comparison tables
│   └── plots.py                 # Equity curves, trade distribution charts
├── autoresearch/                # Claude agent hyperparameter optimizer
│   ├── agent_loop.py            # Main optimization loop
│   ├── run_overnight.py         # Overnight runner (--resume flag supported)
│   ├── best_config.py           # Best config found (promoted to config/ after review)
│   ├── results.tsv              # All experiment results
│   └── snapshots/               # strategy_config.py snapshots for each best score
├── data/
│   ├── fetch_data.py            # Download + synthesize all market data
│   ├── processed/               # Full-history CSVs (current to yesterday via dynamic BACKTEST_END)
│   └── raw/                     # Raw yfinance downloads
├── scripts/dev/                 # Phase 1–2 development scripts (NOT production — archived)
│   ├── strategy_runner.py       # Compare all candidate strategies
│   ├── grid_search*.py          # Hyperparameter grid searches (v1/v2/v3/guard)
│   └── fine_tune.py             # Blending-layer micro-optimization
├── archive/grid_search/         # Grid search result CSVs from Phase 2
├── tests/                       # 231 passing pytest tests (1 skipped)
│   ├── conftest.py              # Fixtures (real + synthetic data)
│   ├── test_regime_classifier.py
│   ├── test_signal_pipeline.py
│   ├── test_data_quality.py
│   ├── test_edge_cases.py
│   ├── test_performance.py
│   ├── test_ibkr_safety.py      # IBKR order safety (mocked, no live connection)
│   ├── test_execution_stress.py # Slippage, fill timing, partial fill stress tests
│   └── [4 other test modules]
├── daily_signal.py              # Daily signal generator (T-1, regime → action → log)
├── shadow_mode.py               # 30-day shadow mode runner (auto data refresh + alerts)
├── send_email.py                # Gmail SMTP alert sender (used by shadow_mode + ibkr)
├── optimize_dual.py             # Step 1–4 optimization runner
├── dual_portfolio_runner.py     # Full backtest runner for current locked config
├── stress_test_robustness.py    # 6 robustness tests (lag, noise, shock, bias, alloc, OOS)
└── logs/
    ├── signal_history.csv       # Daily signal log (T-1 signals)
    ├── shadow_state.json        # Shadow mode day counter + regime stats
    ├── shadow_mode.log          # Full audit trail
    ├── daily_signal.log         # Per-run signal generation log
    └── pending_email.json       # Queued email alerts
```

---

## Key Commands

```bash
# Run tests
python3 -m pytest tests/ -v

# Today's signal (shadow mode — no trades)
python3 daily_signal.py --shadow

# Daily shadow runner (also refreshes data + queues email)
python3 shadow_mode.py

# Shadow mode status
python3 shadow_mode.py --status

# Pre-populate shadow portfolio with historical data
python3 shadow_mode.py --backfill 30

# Full backtest with current locked config
python3 dual_portfolio_runner.py --no-chart

# Optimization (Steps 1–4: T-1, stabiliser, bull sweep, confidence-weighted)
python3 optimize_dual.py --no-chart

# Robustness / stress tests
python3 stress_test_robustness.py --no-chart
```

---

## Current Status (as of 2026-03-28 — Phase 1 & 2 Complete, Phase 3 Active)

| Item | Status |
|---|---|
| Tests | 231/231 passing (1 skipped) |
| **Phase 1: Parameter Optimization** | ✅ **COMPLETE** — 547 experiments, 0.4964 → 0.6564 (+32.2%) |
| — v1 (388 exps) | 14 wins, parameter ceiling reached on allocations |
| — v2 (158 exps, defensive focus) | 0 wins, confirmed all major levers exhausted |
| **Phase 2: Architecture Search** | ✅ **COMPLETE** — Adaptive VIX Thresholds tested, **not viable** |
| — Opportunity #1: Adaptive VIX | ❌ Hypothesis disproven — baseline optimal on all periods |
| Robustness tests | ✅ 4/6 pass — OOS Calmar 0.310 (below 0.40 target, known issue) |
| Config locked | ✅ config/strategy_config.py is immutable baseline (score 0.6564) |
| **Phase 3: Shadow Mode** | 🔄 **ACTIVE** — Day 15/30 (as of 2026-04-16) |
| Scheduled tasks | ✅ Both enabled (bot-daily-shadow-run, bot-weekly-tests) |
| Shadow regime summary | 10 days high_vol (Mar 27–Apr 9) → 5 days uncertain (Apr 10–16, VIX declining) |
| **Next action** | Monitor to ~2026-04-28 (day 30) → then authorize Phase 4 (paper trading) |

---

## Scheduled Tasks (Claude Code Scheduled panel)

| Task ID | Schedule | What it does |
|---|---|---|
| `bot-daily-shadow-run` | Weekdays 4:30 PM | Runs shadow_mode.py + emails anomalies/regime changes |
| `bot-weekly-tests` | Monday 9:00 AM | Runs full pytest suite + emails results |

**Status:** ✅ Both tasks are **active and running**.
- `bot-daily-shadow-run` last ran: 2026-03-27 (weekdays 4:31 PM ET), next: 2026-03-30
- `bot-weekly-tests` last ran: 2026-03-24 (Mondays 9:05 AM ET), next: 2026-03-30
Email destination: **samarth339@gmail.com**

---

## Architecture — How It Works

```
QQQ close (T-1) ──► 130-day SMA comparison ──┐
                                               ├──► Regime: bull / uncertain / high_vol
VIX 5-day avg (T-1) ──► threshold check ──────┘
                                               │
                                               ▼
                     Allocation weights (90/10 bull, 65/35 uncertain, 25/75 high_vol)
                                               │
              ┌────────────────────────────────┴────────────────────────────────┐
              ▼                                                                   ▼
   Strategy A (BestCalmar)                                          Strategy B (NearMiss)
   LongOnlyGuardV2                                                  LongOnlyGuardV2
   - Stays in TQQQ by default                                       - Same but more conservative
   - Exits on MA breach + VIX > 25                                  - Exits on MA breach + VIX > 28
   - vix_reentry=23, ma=190, max_pos=95%                            - vix_reentry=24, max_pos=70%, crash_brake=30%
   - Staggered exit (50% then rest)
              │                                                                   │
              └──────────────────── Blended returns ──────────────────────────────┘
                                               │
                                               ▼
                                    Combined equity curve
```

---

## Phase Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1a | ✅ Done | Data pipeline, backtester engine, base strategies |
| Phase 1b | ✅ Done | Strategy dev, grid search, LongOnlyGuard winner, drawdown reduction |
| Phase 1c | ✅ Done | **Hyperparameter optimization** — 547 experiments, parameter space exhausted |
| Phase 2 | ✅ Done | **Architecture search** — Tested Adaptive VIX Thresholds, hypothesis disproven |
| Phase 3 | 🔄 Active | **Shadow mode** — Daily signals, regime tracking, email alerts |
| Phase 4 | ⏳ Pending | Paper trading (mock fills, slippage tracking) |
| Phase 5 | ⏳ Pending | Live trading, small size (10% of capital), kill switch |

---

## Known Issues / Gotchas

- **Do not use `--no-chart` with strategy_runner** when running from notebook — charts are needed there
- **VIX data**: fetched as `^VIX` from yfinance. Sometimes has weekend gaps — ffill handles it
- **TQQQ inception**: 2010-02-11. Pre-2010 data is synthetic (QQQ × 3x proxy)
- **crash_brake standalone = death**: Mech4 alone triggers → exits → re-enters → loops → wipeout. Only use combined with other mechanisms
- **Confidence-weighted allocation**: tested and WORSE than discrete (Calmar 0.607 vs 0.681). Do not re-enable
- **VIX confirm_days**: adding confirmation HURTS (raises DD). VIX smooth=5 is better than confirm=2
- **OOS Calmar warning**: OOS (2019–2025) Calmar is 0.310, below 0.40 target. COVID crash + 2022 bear are in OOS period making it unusually hard. Monitor carefully in shadow mode.
- **Parameter ceiling**: Bull allocation (0.90) and uncertain allocation (0.65) are at safe range maximums.
- **Phase 1 exhaustion**: 547 experiments across v1 and v2 with dead-end blocking. Parameter space is fully explored; no further gains from tuning alone.
- **Phase 2 findings**: Tested adaptive VIX thresholds (percentile-based adjustment). Hypothesis: "Fixed thresholds don't account for volatility regimes." Reality: Fixed thresholds (from Phase 1) are optimal. Adaptive approaches (both standard and inverted) underperform on all periods (train, val, OOS). See `/autoresearch/PHASE2_RESULTS.md` for details.

---

## Data Files

```
data/processed/
├── TQQQ_full.csv    # 2010-02-11 → present (OHLCV)
├── SQQQ_full.csv    # 2010-02-11 → present (OHLCV)
├── QQQ_full.csv     # 1985-10-01 → present (OHLCV)
├── VIX_full.csv     # 1990-01-02 → present (close only)
└── SPY_full.csv     # 1993-01-29 → present (OHLCV)
```

Refresh with: `python3 data/fetch_data.py`
