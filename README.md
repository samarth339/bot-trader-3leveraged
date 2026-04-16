# Phase 2 — Strategy Development & Backtesting

## Quick Start

```bash
cd /Volumes/SAM/bot-test
pip install -r requirements.txt

# 1. Download & synthesize 42 years of data
python data/fetch_data.py

# 2. Run all strategies & compare
python strategy_runner.py
```

## Project Structure

```
bot-test/
├── config/
│   └── settings.py          # All global parameters (MAs, ROC, capital, etc.)
├── data/
│   ├── fetch_data.py        # Download QQQ/TQQQ/SQQQ, synthesize pre-inception data
│   ├── raw/                 # Raw CSV downloads
│   └── processed/           # Stitched full-history CSVs
├── backtester/
│   └── engine.py            # Core event-driven backtesting engine
├── strategies/
│   ├── base.py              # BaseStrategy (helpers: roc, sma, above_ma)
│   ├── momentum_roc.py      # MA + Rate-of-Change trend following
│   ├── mean_reversion.py    # Rubber-band mean reversion
│   └── combined.py          # Regime filter (momentum) + MR entry timing
├── analysis/
│   ├── metrics.py           # Metrics printer, stress-test report, comparison table
│   └── plots.py             # Equity curve charts, trade distribution
└── strategy_runner.py       # Main script — runs all strategies, prints results
```

## Key Settings (`config/settings.py`)

| Setting              | Default      | Notes                          |
| -------------------- | ------------ | ------------------------------ |
| `MA_SHORT`           | 50           | Mallik's rule short MA         |
| `MA_LONG`            | 250          | Mallik's rule long MA          |
| `ROC_WINDOWS`        | [5,10,20,63] | Rate-of-change lookback        |
| `MAX_DRAWDOWN_LIMIT` | 50%          | Circuit breaker — stop trading |
| `DAILY_STOP_LOSS`    | 7%           | Per-trade daily stop           |
| `SIGNAL_TIME_EST`    | 15:45        | Calculate signal (Phase 3)     |
| `EXECUTE_TIME_EST`   | 15:50        | Execute trades (Phase 3)       |

## Stress Periods Tested

- 2000 Dot-com crash: 2000-03-10 → 2002-10-09
- 2008 GFC: 2007-10-09 → 2009-03-09
- 2020 COVID crash: 2020-02-19 → 2020-03-23
- 2022 Rate hike bear: 2021-11-19 → 2022-10-13

## Adding a New Strategy

1. Create `strategies/my_strategy.py` inheriting `BaseStrategy`
2. Implement `generate_signal(context) → dict` returning one of:
   - `{"action": "buy_tqqq", "size_pct": 0.0–1.0}`
   - `{"action": "buy_sqqq", "size_pct": 0.0–1.0}`
   - `{"action": "sell"}`
   - `{"action": "hold"}`
3. Add to `strategy_runner.py` strategies dict
