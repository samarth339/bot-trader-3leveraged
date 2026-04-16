"""
Shared pytest fixtures for the trading bot test suite.
Real data is used when available; synthetic data is the fallback.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
REAL_DATA_AVAILABLE = (DATA_DIR / "QQQ_full.csv").exists()


# ── Synthetic data builders ───────────────────────────────────────────────────

def _make_prices(n=300, start_price=100.0, drift=0.0003, seed=42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, 0.015, size=n)
    prices = start_price * np.cumprod(1 + returns)
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(prices, index=dates, name="close")


def _make_vix(n=300, base=16.0, seed=99) -> pd.Series:
    rng = np.random.default_rng(seed)
    vix = base + rng.normal(0, 3, size=n)
    vix = np.clip(vix, 9.0, 80.0)
    dates = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(vix, index=dates, name="close")


def _to_df(close_series: pd.Series) -> pd.DataFrame:
    """Wrap a close series into a proper OHLCV DataFrame."""
    df = pd.DataFrame({"close": close_series})
    df["open"]   = df["close"] * 0.999
    df["high"]   = df["close"] * 1.005
    df["low"]    = df["close"] * 0.995
    df["volume"] = 1_000_000
    return df


# ── Real-data fixtures (skip if files missing) ────────────────────────────────

@pytest.fixture(scope="session")
def real_qqq():
    if not REAL_DATA_AVAILABLE:
        pytest.skip("Real data not available — run fetch_data.py first")
    return pd.read_csv(DATA_DIR / "QQQ_full.csv", index_col=0, parse_dates=True)


@pytest.fixture(scope="session")
def real_vix():
    if not REAL_DATA_AVAILABLE:
        pytest.skip("Real data not available — run fetch_data.py first")
    return pd.read_csv(DATA_DIR / "VIX_full.csv", index_col=0, parse_dates=True)


@pytest.fixture(scope="session")
def real_tqqq():
    if not REAL_DATA_AVAILABLE:
        pytest.skip("Real data not available — run fetch_data.py first")
    return pd.read_csv(DATA_DIR / "TQQQ_full.csv", index_col=0, parse_dates=True)


@pytest.fixture(scope="session")
def real_sqqq():
    if not REAL_DATA_AVAILABLE:
        pytest.skip("Real data not available — run fetch_data.py first")
    return pd.read_csv(DATA_DIR / "SQQQ_full.csv", index_col=0, parse_dates=True)


# ── Synthetic fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def syn_qqq_df():
    """300-bar QQQ-like DataFrame with mild upward drift."""
    return _to_df(_make_prices(300, drift=0.0003))


@pytest.fixture
def syn_vix_df():
    """300-bar VIX-like DataFrame, calm (avg ~16)."""
    return _to_df(_make_vix(300, base=16.0))


@pytest.fixture
def bull_qqq_df():
    """200 bars trending strongly above its SMA — should produce BULL regime."""
    prices = _make_prices(200, start_price=200.0, drift=0.001, seed=1)
    return _to_df(prices)


@pytest.fixture
def bear_qqq_df():
    """200 bars trending below SMA — should produce HIGH_VOL regime."""
    prices = _make_prices(200, start_price=200.0, drift=-0.002, seed=2)
    return _to_df(prices)


@pytest.fixture
def high_vix_df():
    """VIX DataFrame with values consistently above 25 (danger zone)."""
    return _to_df(_make_vix(300, base=30.0, seed=5))


@pytest.fixture
def calm_vix_df():
    """VIX DataFrame with values consistently below 18 (calm zone)."""
    return _to_df(_make_vix(300, base=13.0, seed=6))


@pytest.fixture
def spike_vix_df():
    """VIX that spikes suddenly at bar 150 from 14 → 40, then returns to 14."""
    vix = _make_vix(300, base=14.0, seed=7).copy()
    vix.iloc[148:158] = 40.0
    return _to_df(vix)
