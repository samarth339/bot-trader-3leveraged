# Project Context — Trading Bot
### Paste this at the start of every new Claude session on this project

---

## Who I Am & What I'm Building
I'm building a systematic algorithmic trading bot for **TQQQ** (3x long NASDAQ) and **SQQQ**
(3x short NASDAQ). The project lives at `/Volumes/SAM/bot-test` on an external drive (SAM volume).
Goal: grow $5,000 → $775K+ over ~10 years using a systematic, rules-based strategy.

I'm now working on this project from my **personal Claude account (samarth339@gmail.com)**.
Previous development was done on a work account — all code and context is in the project directory.

---

## Current Status: Phase 3 — Shadow Mode (30-day live observation)

### What's been built and locked:
The system is **production-ready**. Do NOT restructure, re-architect, or change locked parameters
without very explicit discussion. The performance below was hard-won through weeks of iteration.

### Locked Production Strategy:

**Dual Portfolio — LongOnlyGuard V2 system:**

| Component | Value |
|---|---|
| Strategy A (Aggressive — "BestCalmar") | `LongOnlyGuardV2(ma_long=200, vix_exit=25, vix_reentry=22, confirm_bars=2, max_position_pct=0.90, stagger_exit=True)` |
| Strategy B (Defensive — "NearMiss") | `LongOnlyGuardV2(ma_long=150, vix_exit=28, vix_reentry=22, confirm_bars=4, max_position_pct=0.70, crash_brake_pct=0.30, stagger_exit=True)` |
| Bull regime allocation | 75% A / 25% B (QQQ > SMA-150 AND VIX < 18) |
| Mid regime allocation | 50% A / 50% B |
| High-vol regime allocation | 30% A / 70% B (QQQ < SMA-150 OR VIX ≥ 25) |
| T-1 execution | Always ON — regime computed from PREVIOUS day's close |
| VIX smoothing | 5-day rolling average before thresholding |

**Verified backtest performance (2010–2025, T-1 honest):**
- CAGR: **26.6%** | Max Drawdown: **39.1%** | Calmar: **0.681** | Sharpe: 0.848
- Final equity: **$211,997** from $5,000 seed
- COVID 2020 drawdown: 38.8% | 2022 rates drawdown: 30.9%

---

## What Has Already Been Tested and Decided

### ✅ Tested and PASSED:
1. **Out-of-sample** (2010–2018 train / 2019–2025 test) — holds up
2. **Regime lag test** (T-1, T-2, T-3 delay) — T-1 is optimal
3. **Regime misclassification** (10–20% random label flips) — robust
4. **Transition shock** (sudden VIX spikes, whipsaws) — handles correctly
5. **Overlap bias** (independent regime classifier) — confirmed
6. **Allocation sensitivity** (70/30 → 90/10 bull sweep) — 75/25 is optimal by Calmar
7. **VIX smoothing sweep** (3–7 days) — 5-day is optimal
8. **Confidence-weighted allocation** — tested and WORSE. Do not re-enable.

### ❌ Already tried and failed (do not revisit):
- Confidence-weighted (continuous) allocation: Calmar 0.607 vs 0.681 discrete
- `confirm_days=2` stabiliser: raises DD, hurts Calmar
- crash_brake as standalone mechanism: causes infinite loop → wipeout
- Any strategy with >100 trades/year on TQQQ/SQQQ: 3x decay destroys returns

---

## The Core Architecture (Read This)

```
QQQ close (T-1) → 150-day SMA comparison ──┐
                                             ├→ Regime: bull / uncertain / high_vol
VIX 5-day avg (T-1) → threshold check ──────┘
                                             │
                                             ▼
                          Portfolio weights (75/25, 50/50, 30/70)
                                             │
    ┌────────────────────────────────────────┴────────────────────────────┐
    ▼                                                                       ▼
Strategy A (BestCalmar)                                      Strategy B (NearMiss)
- Stays long TQQQ by default                                 - Same but more conservative
- Exits to cash on MA breach + high VIX                      - Smaller position, crash brake
- Staggered exit: 50% first, rest on confirmation            - Confirmed regime change needed
    │                                                                       │
    └──────────────────── Blended daily returns ────────────────────────────┘
                                             │
                                             ▼
                                Combined portfolio equity
```

**T-1 hard guard** is enforced at the data layer in `backtester/dual_portfolio.py`:
```python
signal_close = qqq_close.shift(1).ffill()
signal_vix   = vix_smooth.shift(1).ffill()
# SMA computed on shifted data — structurally impossible to look ahead
```

---

## Key Files

```
/Volumes/SAM/bot-test/
├── CLAUDE.md                    ← Full project memory for Claude (READ THIS FIRST)
├── config/strategy_config.py    ← SINGLE SOURCE OF TRUTH for all parameters
├── config/settings.py           ← Paths, tickers, capital constants
├── strategies/long_only_guard_v2.py  ← The core strategy class
├── backtester/engine.py         ← Backtester + Portfolio.sell_partial()
├── backtester/dual_portfolio.py ← DualPortfolioBacktester (v3 production)
├── daily_signal.py              ← Run this daily for today's signal
├── shadow_mode.py               ← 30-day automated shadow runner
├── optimize_dual.py             ← Steps 1–4 fine-tuning script
├── stress_test_robustness.py    ← 6 robustness/destruction tests
└── tests/                       ← 88 passing pytest tests
```

---

## Key Commands

```bash
cd /Volumes/SAM/bot-test

# Check today's signal
python3 daily_signal.py --shadow

# Run shadow mode (refreshes data, logs signal, queues email)
python3 shadow_mode.py

# Shadow mode status / 30-day report
python3 shadow_mode.py --status

# Backfill 30 days of historical shadow data
python3 shadow_mode.py --backfill 30

# Run all tests
python3 -m pytest tests/ -v

# Run full backtest with locked config
python3 dual_portfolio_runner.py --no-chart
```

---

## Scheduled Automation

Two Claude Code scheduled tasks are set up (Scheduled panel in sidebar):

| Task | Schedule | Action |
|---|---|---|
| `bot-daily-shadow-run` | Weekdays 4:30 PM | Runs shadow_mode.py, emails regime changes |
| `bot-weekly-tests` | Mondays 9:00 AM | Runs pytest suite, emails results |

Emails go to: **samarth339@gmail.com**

---

## Phase Roadmap

| Phase | Status | Notes |
|---|---|---|
| 1 — Data + Engine | ✅ Complete | yfinance data, backtester, base strategies |
| 2 — Strategy Dev | ✅ Complete | LongOnlyGuard winner, grid search, DD reduction |
| 3 — Shadow Mode | 🔄 **Active now** | 30 days, daily signals, automated monitoring |
| 4 — Paper Trading | ⏳ After 30 days | Mock fills, slippage tracking, position drift |
| 5 — Live (small) | ⏳ After paper | 10% of capital, kill switch at 30% DD |

---

## Important Constraints

1. **No new strategies** — system is locked. Fine-tuning only if Calmar improvement is meaningful
2. **Always use T-1** — same-bar signals are not achievable in live trading
3. **shadow_mode.py = no trades** — observation only until Phase 4
4. **All parameters in `config/strategy_config.py`** — never hardcode values in strategy files
5. **Run tests after any code change** — `python3 -m pytest tests/ -q` must show 88 passed

---

## Data

Data lives in `data/processed/` — OHLCV CSVs for TQQQ, SQQQ, QQQ, VIX, SPY.
TQQQ inception: 2010-02-11 (pre-2010 is QQQ × 3x synthetic proxy).
Refresh: `python3 data/fetch_data.py`
