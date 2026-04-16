"""
Global configuration for the backtesting system.
"""
import datetime as _dt

# ── Tickers ──────────────────────────────────────────────────────────────────
TICKERS = ["TQQQ", "SQQQ", "QQQ"]  # QQQ used as benchmark / signal proxy

# TQQQ inception: Feb 11, 2010. Before that we synthesize from QQQ * 3x.
TQQQ_INCEPTION = "2010-02-11"

# Synthetic 3x data can be sourced from QQQ back to 1985-01-01
BACKTEST_START = "1985-01-01"
BACKTEST_END   = _dt.date.today().strftime("%Y-%m-%d")   # always fetch up to today

# ── Capital & Sizing ──────────────────────────────────────────────────────────
INITIAL_CAPITAL   = 5_000        # Phase 1 seed — scale up in live trading
MAX_POSITION_PCT  = 1.0          # 100 % in one side (TQQQ or SQQQ)
POSITION_STEP_PCT = 0.25         # Size in 25 % increments

# ── Risk ──────────────────────────────────────────────────────────────────────
MAX_DRAWDOWN_LIMIT = 0.50        # Stop trading if equity drops 50 % from peak
DAILY_STOP_LOSS    = 0.07        # Exit if daily loss > 7 %

# ── Execution window ──────────────────────────────────────────────────────────
SIGNAL_TIME_EST  = "15:45"       # Calculate signal 15 min before close
EXECUTE_TIME_EST = "15:50"       # Execute in last 10 min of session

# ── Moving averages (Mallik's rule) ───────────────────────────────────────────
MA_SHORT = 50
MA_LONG  = 250

# ── Rate-of-Change windows ────────────────────────────────────────────────────
ROC_WINDOWS = [5, 10, 20, 63]    # 1-wk, 2-wk, 1-mo, 1-qtr

# ── Stress-test periods ───────────────────────────────────────────────────────
STRESS_PERIODS = {
    "dot_com_crash":   ("2000-03-10", "2002-10-09"),
    "gfc_2008":        ("2007-10-09", "2009-03-09"),
    "covid_crash":     ("2020-02-19", "2020-03-23"),
    "rate_hike_2022":  ("2021-11-19", "2022-10-13"),
}

# ── Output paths ──────────────────────────────────────────────────────────────
DATA_RAW_DIR       = "data/raw"
DATA_PROCESSED_DIR = "data/processed"
LOGS_DIR           = "logs"
ANALYSIS_DIR       = "analysis"
