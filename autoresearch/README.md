# Autoresearch Experiment — Trading Strategy Optimizer

Adapts Karpathy's autoresearch loop to optimize the TQQQ/SQQQ dual-portfolio
strategy. The agent proposes parameter changes, runs backtests, and keeps or
reverts based on a composite score (geometric mean of train Calmar × val Calmar).

**This branch is isolated from `main`. Production code is untouched.**

---

## Quick Start

```bash
# 1. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Check the baseline score first
python autoresearch/run_backtest.py --oos

# 3. Run a small test (10 experiments, ~5 min)
python autoresearch/agent_loop.py --experiments 10 --dry-run   # preview proposals
python autoresearch/agent_loop.py --experiments 10             # actually run

# 4. Overnight run (~600 experiments, ~8 hrs)
python autoresearch/agent_loop.py --experiments 600

# 5. Review results
python autoresearch/review_results.py
python autoresearch/review_results.py --wins
```

---

## Files

| File | Purpose |
|---|---|
| `program.md` | Research direction + constraints (edit this to guide the agent) |
| `run_backtest.py` | Evaluation harness — train/val split + composite score |
| `agent_loop.py` | The main loop — calls Claude API + applies/reverts changes |
| `review_results.py` | Read results.tsv, show wins, guide promotion to main |
| `results.tsv` | Auto-generated audit trail (every experiment logged) |

**Only file the agent modifies**: `config/strategy_config.py`

---

## How the Score Works

```
composite_score = sqrt(train_calmar × val_calmar)

Hard vetos (score = 0 if violated):
  train: CAGR ≥ 24%, MaxDD ≤ 42%, Sharpe ≥ 0.80
  val:   CAGR ≥ 18%, MaxDD ≤ 46%, Calmar ≥ 0.50

Current baseline: ~0.689
Target:            > 0.850
```

---

## Promoting a Winner to Main

When you find a config you're happy with:

```bash
# 1. Verify OOS + tests + robustness
python autoresearch/run_backtest.py --oos
python -m pytest tests/ -q
python stress_test_robustness.py --no-chart

# 2. Merge config to main
git checkout main
git checkout autoresearch-experiment -- config/strategy_config.py
git commit -m "Promote autoresearch winner: Calmar X.XXX → X.XXX"

# 3. Update CLAUDE.md locked config table
# 4. Run shadow_mode.py --backfill 30 with new config
```

---

## Phases

| Phase | Mutable file | Status |
|---|---|---|
| Phase 1 | `config/strategy_config.py` only | **Active** |
| Phase 2 | `strategies/long_only_guard_v2.py` | Not started |
| Phase 3 | New `signals/augmented.py` | Not started |

Phase 2 and 3 require manually updating `program.md` before starting.
