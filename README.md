# TQQQ/SQQQ Algorithmic Trading Bot

A systematic, fully automated trading system that trades **TQQQ** (3× long NASDAQ) using **QQQ** as the clean signal source. Uses a two-strategy blended portfolio with regime detection, VIX-based de-risking, and 10 layered safety guards.

**Current phase:** Phase 4 — Paper trading via GitHub Actions (no local machine required)  
**Goal:** $775K+ from $10K seed over ~10 years

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Current Status](#current-status)
3. [Local Setup](#local-setup)
4. [Daily Signal — Run Locally](#daily-signal--run-locally)
5. [Paper Trading Commands](#paper-trading-commands)
6. [Run the Dashboard Locally](#run-the-dashboard-locally)
7. [GitHub Actions — Remote Automation](#github-actions--remote-automation)
8. [Run the Test Suite](#run-the-test-suite)
9. [Configuration Reference](#configuration-reference)
10. [File Structure](#file-structure)
11. [Phase Roadmap](#phase-roadmap)
12. [Troubleshooting](#troubleshooting)

---

## How It Works

```
Every weekday at 3:30 PM ET (GitHub Actions):
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Fetch latest QQQ / TQQQ / VIX data from yfinance     │
  │ 2. Compute T-1 regime signal (yesterday's close + VIX)  │
  │    ├── QQQ vs 130-day SMA  ──► bull / uncertain / high_vol
  │    └── VIX 5-day average   ──► threshold classification │
  │ 3. Resolve allocation weights                           │
  │    ├── bull:      90% Strategy-A / 10% Strategy-B       │
  │    ├── uncertain: 45–75% A (dynamic, based on momentum) │
  │    └── high_vol:  25% A / 75% B                         │
  │ 4. Compute target TQQQ %                                │
  │    └── weight_a × 85% + weight_b × 60%                 │
  │ 5. Log signal to logs/signal_history.csv                │
  └─────────────────────────────────────────────────────────┘

Every weekday at 4:00 PM ET (GitHub Actions, after market close):
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Read today's signal                                  │
  │ 2. Fetch actual TQQQ closing price                      │
  │ 3. Run 7 safety guards (kill switch, drawdown, VIX, …)  │
  │ 4. Compute delta shares (drift gate: skip if < 5%)      │
  │ 5. Simulate MOC fill at close ± 10 bps slippage         │
  │ 6. Update paper_portfolio.json + paper_trades.csv       │
  │ 7. Email execution summary → samarth339@gmail.com       │
  └─────────────────────────────────────────────────────────┘
```

**Trade trigger:** A trade only fires when the actual TQQQ allocation drifts more than **5%** from the target. HOLD is the most common daily outcome.

---

## Current Status

| | |
|---|---|
| Tests | ✅ 458 passing, 1 skipped |
| Phase 3 shadow mode | ✅ Complete — 42 days observed |
| Phase 4 paper trading | 🔄 Active — **SIMULATION only** (`paper_trade.py` on GitHub Actions, NOT connected to IBKR) |
| Sim account | $10,000 seed · state in `logs/paper_portfolio.json` · reset 2026-07-10 |
| Real IBKR paper acct | `DUP540674` — untouched by the bot (Phase 5 / `ibkr/executor.py` only) |

---

## Local Setup

### Prerequisites

- Python 3.11 or 3.12
- Git

### 1. Clone and install

```bash
git clone https://github.com/samarth339/bot-trader-3leveraged.git
cd bot-trader-3leveraged
pip install -r requirements.txt
```

### 2. Fetch market data

Downloads QQQ, TQQQ, SQQQ, VIX, SPY — full history since 1985:

```bash
python3 data/fetch_data.py
```

Data is saved to `data/processed/` (gitignored — must be fetched locally or cached in CI).

### 3. Set up email alerts (optional)

Email is used by paper trading and CI test failures.

```bash
# Create a .env file (gitignored — never committed)
echo "GMAIL_USER=your@gmail.com" > .env
echo "GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx" >> .env
```

To generate a Gmail App Password: Google Account → Security → 2-Step Verification → App Passwords.

---

## Daily Signal — Run Locally

### Generate today's signal

```bash
python3 daily_signal.py
```

Reads yesterday's QQQ close and VIX, classifies the regime, resolves the allocation action, checks the gap guard, and appends a row to `logs/signal_history.csv`.

Output example:
```
╔══════════════════════════════════════════════════════╗
║  DAILY SIGNAL  —  2026-05-29                         ║
╠══════════════════════════════════════════════════════╣
║  Regime         : BULL                               ║
║  Reason         : price above SMA AND VIX 17.1 < 18  ║
║  Signal source  : 2026-05-28 (T-1 close)             ║
║  QQQ close (T-1): $717.54  (SMA-130: $624.07  +14.98%)
║  VIX (smoothed) : 17.1  (raw: 16.2) [bull<18 danger≥25]
╠══════════════════════════════════════════════════════╣
║  Target alloc   : 90% Strategy-A  /  10% Strategy-B  ║
║  Previous alloc : 90% / 10%  (unchanged)             ║
╠══════════════════════════════════════════════════════╣
║  ✅  ACTION : HOLD  (allocation in target range)      ║
╠══════════════════════════════════════════════════════╣
║  Execution      : CLOSE + 10 bps slippage            ║
║  ✅ Gap guard    : clear  (+1.12% open)               ║
╚══════════════════════════════════════════════════════╝
```

### Back-calculate for a specific date

```bash
python3 daily_signal.py --date 2026-01-15
```

Note: historical `--date` runs skip the gap guard (intraday open prices cannot be reconstructed for past dates).

### View signal history

```bash
python3 -c "import pandas as pd; print(pd.read_csv('logs/signal_history.csv').tail(10).to_string())"
```

---

## Paper Trading Commands

### Run a full demo (safe — nothing changes)

```bash
python3 paper_trade.py --demo
```

Runs the **complete simulation** using real live prices and today's actual signal, but writes everything to a temp directory that is deleted on exit. Production files are never touched.

Shows:
- Today's signal (regime, VIX, gap guard status)
- All 7 safety guard results
- The exact trade that would execute (shares, fill price, slippage cost)
- Portfolio before → after comparison
- The exact trade record that would be written to CSV
- The exact email subject and body that would be sent

Use this to preview what the real first execution will look like before GitHub Actions fires.

### Check current portfolio state

```bash
python3 paper_trade.py --status
```

```
═══════════════════════════════════════════════════════
  Paper Trade — Portfolio Status
═══════════════════════════════════════════════════════
  Kill switch:        OFF ✓
  TQQQ shares:        97
  Cash:               $1,789.47
  NLV:                $10,004.21
  Seed capital:       $10,000.00
  Return:             +0.04%
  Peak equity:        $10,004.21
  Drawdown:           0.0%
  Trades YTD:         1
═══════════════════════════════════════════════════════
```

### Compute the plan without executing

```bash
python3 paper_trade.py --dry-run
```

Runs the full guard chain and computes the rebalancing plan, but does **not** update `paper_portfolio.json` or `paper_trades.csv`.

### Reset to clean Day-1 state

```bash
python3 paper_trade.py --reset-portfolio
```

Wipes `paper_portfolio.json` back to `$10,000 · 0 shares · 0 trades` and clears `paper_trades.csv` to header-only. Use this after running `--demo` to confirm production files are ready before the first real GitHub Actions execution.

### Clear today's execution flag (recovery)

```bash
python3 paper_trade.py --reset
```

Clears `last_trade_date` so the double-submission guard allows re-execution today. Use if a run failed partway through.

### View trade history

```bash
python3 -c "import pandas as pd; print(pd.read_csv('logs/paper_trades.csv').to_string())"
```

---

## Run the Dashboard Locally

The dashboard is a Dash (Python + Plotly) web app showing the full backtest equity curve, live signal history, portfolio analytics, and risk metrics.

### Launch

```bash
bash scripts/dashboard_run.sh
```

Then open **http://127.0.0.1:8050** in your browser.

**Reload the page** to refresh all data — the dashboard reads files on each page load, no polling.

### Launch with data refresh first

```bash
REFRESH=1 bash scripts/dashboard_run.sh
```

Runs `data/fetch_data.py` before starting the server so all charts show today's data.

### Custom port

```bash
bash scripts/dashboard_run.sh --port 8080
```

### Direct Python launch

```bash
PYTHONPATH=. python3 -m dashboard.app
```

### What each panel shows

| Panel | Description |
|---|---|
| **KPI strip** | Portfolio value · Total return · Max drawdown · Sharpe 30d · Current regime/action |
| **Equity curve** | Backtest (blue) + shadow period (purple) + paper trades (green) from $10K seed |
| **Drawdown chart** | Historical drawdown series beneath the equity curve |
| **Signal history** | Scrollable table of last 60 signals — Date, Regime, Action, A%, B%, QQQ, VIX, **Gap**, 5d Fwd, Outcome |
| **Rolling Sharpe** | 7-day and 30-day rolling Sharpe ratio |
| **Returns histogram** | Daily return distribution — green positive, red negative |
| **Risk panel** | VaR 95/99 · Allocation donut chart · Regime distribution bar |
| **Historical context** | MTD/YTD chips + 1-year QQQ and VIX chart with bull/danger thresholds |

The **Gap** column in the signal table shows:
- 🚫 **red** — gap guard triggered that day (BUY was blocked)
- **green %** — gap guard was clear
- **—** — historical row before gap guard was added

---

## GitHub Actions — Remote Automation

All automation runs on **GitHub-hosted runners** — no local machine required for Phase 4.

### Workflow schedule

| Workflow | Cron (UTC) | Fires ~ET | What it does |
|---|---|---|---|
| `daily-signal.yml` | `30 19 * * 1-5` | 3:30 PM EDT | Data refresh → `daily_signal.py` → commits `signal_history.csv` |
| `paper-trade.yml` | `0 20 * * 1-5` | 4:00 PM EDT | `paper_trade.py` → simulates fill at actual close → commits portfolio state → sends email |
| `ci.yml` | on push + Mon 2 PM UTC | — | 432 tests across 7 parallel jobs |

> **DST note:** Crons are in UTC. When clocks fall back in November (EDT → EST), update `19`/`20` → `20`/`21` in both workflow files.

### Required GitHub Secrets

Go to **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Value |
|---|---|
| `GMAIL_USER` | `samarth339@gmail.com` |
| `GMAIL_APP_PASSWORD` | Your 16-character Gmail App Password (spaces OK) |

Without these, trades still execute and state is committed — only email delivery fails.

### Monitor a run

1. Go to **Actions tab** on GitHub
2. Click any workflow run
3. Open the job to see the **Job Summary** — a formatted report with today's signal, guard results, portfolio state, and trade history table

### Manual trigger

Any workflow can be triggered manually: Actions tab → select workflow → **Run workflow**.

### Disable automation temporarily

In the workflow file, remove the `schedule:` block and keep only `workflow_dispatch:`:
```yaml
on:
  workflow_dispatch:   # manual-only until re-enabled
```

---

## Run the Test Suite

### Full suite

```bash
python3 -m pytest tests/ -v
```

### Fast subset (no real data, ~5 seconds)

```bash
python3 -m pytest tests/ -m "not slow and not real_data" -v
```

### Specific modules

```bash
# IBKR safety guards (all 10, fully mocked)
python3 -m pytest tests/test_ibkr_safety.py -v

# Config contract — catches accidental parameter changes
python3 -m pytest tests/test_config_schema.py -v

# Gap guard boundary cases + fail-open behavior
python3 -m pytest tests/test_gap_guard_extended.py -v

# yfinance MultiIndex compatibility (yfinance 1.3.0 regression)
python3 -m pytest tests/test_yfinance_compat.py -v

# Dashboard gap column rendering
python3 -m pytest tests/test_dashboard_signal_panel.py -v
```

### Test coverage by module

| Module | Tests | What it guards |
|---|---|---|
| `test_config_schema.py` | 58 | Locked parameter values — any accidental config change fails here |
| `test_ibkr_safety.py` | 54 | All 10 IBKR safety guards (mocked, no live connection) |
| `test_signal_pipeline.py` | 38 | Regime classification, dynamic allocation, momentum override |
| `test_gap_guard_extended.py` | 19 | Gap guard boundaries, IEEE 754 edge cases, fail-open behavior |
| `test_dashboard_signal_panel.py` | 21 | Gap column rendering for triggered/clear/missing data |
| `test_yfinance_compat.py` | 21 | MultiIndex fix — yfinance 1.3.0 and legacy flat-column format |
| `test_lookahead_bias.py` | 13 | T-1 enforcement — future data must never contaminate signals |
| `test_guard_coverage.py` | 14 | Guard 7 (daily stop mutation), guard priority/short-circuit, P3 removal |
| `test_signal_csv_schema.py` | 19 | CSV column integrity, idempotency, backward compat |
| `test_performance.py` | 13 | Backtest performance regression — CAGR/drawdown bounds |

---

## Configuration Reference

All strategy parameters live in **`config/strategy_config.py`** — the single source of truth. Never hardcode these elsewhere.

### Locked values (post expert-panel v2, 2026-04-16)

```python
REGIME_CONFIG = {
    "ma_window":    130,   # QQQ SMA window for bull/bear detection
    "vix_smooth":     5,   # VIX rolling average (days)
    "vix_bull":    18.0,   # VIX below this → BULL
    "vix_hi_vol":  25.0,   # VIX at/above this → HIGH_VOL
    "t1_execution": True,  # ALWAYS use T-1 — non-negotiable
}

ALLOC_CONFIG = {
    "bull":      (0.90, 0.10),   # 90% A, 10% B
    "uncertain": (0.65, 0.35),   # base; scaled by pct_vs_sma (0.45–0.75)
    "high_vol":  (0.25, 0.75),   # 25% A, 75% B
}

STRATEGY_A_CONFIG = {           # BestCalmar — aggressive
    "ma_long": 190, "vix_exit": 25, "vix_reentry": 24,
    "max_position_pct": 0.85,   # max 85% of NLV in TQQQ
    "vol_scale": False, "stagger_exit": True,
}

STRATEGY_B_CONFIG = {           # NearMiss — defensive
    "ma_long": 150, "vix_exit": 28, "vix_reentry": 22,
    "max_position_pct": 0.60,   # max 60% of NLV in TQQQ
    "vol_scale": True, "crash_brake_pct": 0.30,
}

RISK_CONFIG = {
    "max_drawdown_halt":     0.35,  # halt + activate kill switch at 35% DD
    "daily_stop_loss":       0.07,  # 7% intraday loss → force full exit
    "alloc_drift_rebalance": 0.05,  # rebalance only when drift > 5%
}
```

**Target TQQQ allocation:**
- BULL:     `0.90 × 85% + 0.10 × 60%` = **82.5%** of NLV
- HIGH_VOL: `0.25 × 85% + 0.75 × 60%` = **66.25%** of NLV

---

## File Structure

```
bot-trader-3leveraged/
│
├── config/
│   ├── settings.py              # Paths, tickers, seed capital
│   └── strategy_config.py       # ALL strategy parameters — single source of truth
│
├── strategies/
│   ├── long_only_guard_v2.py    # Production strategy (stagger exit, vol_scale, crash_brake)
│   └── base.py                  # BaseStrategy helpers
│
├── backtester/
│   ├── engine.py                # Core backtesting engine
│   └── dual_portfolio.py        # DualPortfolioBacktester (T-1 hardened)
│
├── ibkr/                        # IB Gateway execution layer (Phase 5)
│   ├── executor.py              # Main orchestrator — reads signal → submits MOC order
│   ├── safety_guard.py          # 10 pre-flight safety checks
│   ├── position_reconciler.py   # Target allocation math
│   ├── gap_guard.py             # Overnight gap-down protection
│   ├── order_manager.py         # MOC / limit-close order construction
│   ├── account.py               # IB account state (NLV, positions, buying power)
│   ├── client.py                # IB Gateway connection with retry logic
│   ├── state.py                 # Persistent execution state (JSON)
│   └── kill_switch.py           # Emergency stop (file-based flag)
│
├── dashboard/
│   ├── app.py                   # Dash entry point
│   ├── data_loader.py           # Loads backtest + signal + paper portfolio data
│   ├── assets/custom.css        # Dark theme (GitHub-dark palette)
│   └── components/
│       ├── kpi_cards.py         # 5-card KPI strip
│       ├── equity_chart.py      # Equity curve + drawdown
│       ├── signal_panel.py      # Signal history table (with Gap column)
│       ├── analytics.py         # Rolling Sharpe + returns histogram
│       ├── risk_panel.py        # VaR + allocation donut + regime bars
│       └── historical.py        # Period chips + VIX/QQQ context chart
│
├── data/
│   ├── fetch_data.py            # Download QQQ/TQQQ/SQQQ/VIX/SPY
│   └── processed/               # GITIGNORED — regenerated by fetch_data.py
│
├── tests/                       # 432 tests across 19 modules
│   ├── conftest.py              # Shared fixtures (real + synthetic data)
│   └── test_*.py                # See test coverage table above
│
├── .github/workflows/
│   ├── daily-signal.yml         # 3:30 PM ET weekdays — signal generation
│   ├── paper-trade.yml          # 4:00 PM ET weekdays — paper trade simulation
│   ├── ci.yml                   # Tests on push + weekly
│   └── daily-shadow.yml         # ARCHIVED — Phase 3 complete
│
├── logs/
│   ├── signal_history.csv       # TRACKED — every signal ever generated
│   ├── paper_portfolio.json     # TRACKED — current paper portfolio state
│   ├── paper_trades.csv         # TRACKED — full trade history with fills
│   └── shadow_state.json        # TRACKED — Phase 3 shadow mode audit trail
│
├── scripts/
│   └── dashboard_run.sh         # Dashboard launcher (optional REFRESH=1 flag)
│
├── daily_signal.py              # Signal generator
├── paper_trade.py               # Paper trading simulator (demo/dry-run/status/reset)
├── shadow_mode.py               # Phase 3 shadow runner (archived)
├── send_email.py                # Gmail SMTP alert sender
├── requirements.txt             # Full dependencies (local dev)
└── requirements-ci.txt          # Lean CI subset (GitHub Actions)
```

---

## Phase Roadmap

| Phase | Status | Description |
|---|---|---|
| Phase 1 | ✅ Complete | Parameter optimization — 547 experiments, score 0.4964 → 0.6564 (+32%) |
| Phase 2 | ✅ Complete | Architecture search — Adaptive VIX hypothesis tested and disproven |
| Phase 3 | ✅ Complete | Shadow mode — 42 days, all signals validated, no look-ahead confirmed |
| **Phase 4** | 🔄 **Active** | **Paper trading** — GitHub Actions daily execution, $10K seed |
| Phase 5 | ⏳ Pending | Live trading — 10% of real capital, IB Gateway, self-hosted runner |

### Switching to Phase 5 (live trading)

IB Gateway is a desktop application — it cannot run on GitHub's servers. For fully automated live execution with no local machine dependency, the solution is **ibeam on a VPS**.

#### Why you can't use GitHub-hosted runners for live execution

GitHub Actions runners are ephemeral cloud VMs. IB Gateway requires:
- A persistent machine that's always on
- Active IBKR authentication (session re-auth ~monthly)
- Network port 4001/4002 accessible to the executor

#### Recommended setup: ibeam + VPS (~$6/month)

[ibeam](https://github.com/Voyz/ibeam) is a Docker wrapper around IB Gateway that handles headless authentication. Run it on a $4–6/month cloud server (DigitalOcean, Vultr, or Linode).

```
VPS (always on, ~$6/mo)
├── Docker: ibeam ──► IB Gateway session, auto-restores after crashes
└── cron 3:45 PM ET (weekdays):
       cd ~/bot-trader && git pull
       python3 -m ibkr.executor --live    # or --paper for paper trading
```

**Setup steps:**

```bash
# 1. Provision a $6/mo Droplet (Ubuntu 24.04, 1 GB RAM is enough)

# 2. Install Docker
curl -fsSL https://get.docker.com | sh

# 3. Clone and configure ibeam
git clone https://github.com/Voyz/ibeam.git
cd ibeam
cp env.list.template env.list
# Edit env.list: set IBEAM_ACCOUNT, IBEAM_PASSWORD

# 4. Start ibeam (keeps IB Gateway alive)
docker-compose up -d

# 5. Clone the trading bot
cd ~ && git clone https://github.com/samarth339/bot-trader-3leveraged.git
pip install -r bot-trader-3leveraged/requirements.txt

# 6. Add a crontab entry (3:30 PM signal, 3:45 PM execute)
crontab -e
# Add:
# 30 19 * * 1-5  cd ~/bot-trader-3leveraged && git pull && python3 daily_signal.py
# 45 19 * * 1-5  cd ~/bot-trader-3leveraged && python3 -m ibkr.executor --live
```

**Code change needed in `ibkr/executor.py`** — the executor host is currently hardcoded to `127.0.0.1`. On a VPS running ibeam locally, this is already correct (ibeam and executor run on the same machine).

Everything else — signal generation logic, allocation math, safety guards, email alerts — is identical and already validated through Phase 3 and Phase 4.

---

## Troubleshooting

### `No QQQ data on or before <date>`
```bash
python3 data/fetch_data.py
```
Data files are gitignored and must be regenerated locally.

### `Signal is stale: as_of_date=..., today=...`
```bash
python3 daily_signal.py
```
The signal in `logs/signal_history.csv` is from a previous day.

### `GMAIL_APP_PASSWORD not set`
Create `.env` in the project root:
```
GMAIL_USER=your@gmail.com
GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
```
For GitHub Actions: Settings → Secrets → Actions → add `GMAIL_USER` and `GMAIL_APP_PASSWORD`.

### Dashboard shows a blank page
```bash
PYTHONPATH=. python3 -m dashboard.app
```
Make sure `PYTHONPATH` is set to the project root, or use the provided script.

### `float() argument must be a string or a real number, not 'Series'`
This is the yfinance 1.3.0 MultiIndex bug. All calls in this codebase already apply `_flatten()`. If you add a new `yf.download()` call, flatten the result first:
```python
data = yf.download("TQQQ", period="1d", interval="1m", progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data = data.droplevel(1, axis=1)
```

### Paper trade runs at the wrong time (1 hour off)
The GitHub Actions crons are in UTC and don't adjust for DST. Update when clocks change:
- EDT (Mar–Nov): `19` UTC = 3 PM ET, `20` UTC = 4 PM ET
- EST (Nov–Mar): `20` UTC = 3 PM ET, `21` UTC = 4 PM ET

### Kill switch is active
```bash
python3 -c "from ibkr import kill_switch; kill_switch.deactivate()"
# or delete the file directly:
rm logs/ibkr_kill.flag
```
Read the kill reason first: `cat logs/ibkr_kill.flag`

### `Trading through this Gateway is not allowed` (IB error 321)
This is a market data subscription error — it does not block order submission. If account NLV and positions load correctly, this error is harmless.
