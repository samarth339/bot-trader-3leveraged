"""
Data Quality Tests

Validates the processed CSV files used by the trading system:
  • No missing values
  • Prices positive
  • VIX in valid range
  • Date index monotonic
  • Required date range coverage
"""
import pandas as pd
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


class TestQQQData:

    def test_no_missing_close(self, real_qqq):
        nulls = real_qqq["close"].isnull().sum()
        assert nulls == 0, f"QQQ has {nulls} missing close values"

    def test_prices_positive(self, real_qqq):
        assert (real_qqq["close"] > 0).all(), "QQQ close contains non-positive values"

    def test_ohlc_columns_present(self, real_qqq):
        for col in ["open", "high", "low", "close"]:
            assert col in real_qqq.columns, f"QQQ missing column: {col}"

    def test_high_gte_low(self, real_qqq):
        violations = (real_qqq["high"] < real_qqq["low"]).sum()
        assert violations == 0, f"QQQ: high < low on {violations} bars"

    def test_close_within_high_low(self, real_qqq):
        bad = ((real_qqq["close"] > real_qqq["high"] * 1.001) |
               (real_qqq["close"] < real_qqq["low"] * 0.999)).sum()
        assert bad == 0, f"QQQ: close outside high/low on {bad} bars"

    def test_date_index_monotonic(self, real_qqq):
        assert real_qqq.index.is_monotonic_increasing, "QQQ dates not monotonically increasing"

    def test_covers_tqqq_inception(self, real_qqq):
        assert real_qqq.index[0] <= pd.Timestamp("2010-02-11"), \
            "QQQ data must start on or before TQQQ inception (2010-02-11)"

    def test_no_duplicate_dates(self, real_qqq):
        dupes = real_qqq.index.duplicated().sum()
        assert dupes == 0, f"QQQ has {dupes} duplicate dates"

    def test_no_extreme_daily_move(self, real_qqq):
        """No single-day QQQ move > 25% — data error detector."""
        pct = real_qqq["close"].pct_change().abs()
        extreme = (pct > 0.25).sum()
        assert extreme == 0, f"QQQ has {extreme} days with >25% move (likely data error)"


class TestVIXData:

    def test_no_missing_close(self, real_vix):
        nulls = real_vix["close"].isnull().sum()
        assert nulls == 0, f"VIX has {nulls} missing close values"

    def test_vix_in_valid_range(self, real_vix):
        low  = (real_vix["close"] < 5.0).sum()
        high = (real_vix["close"] > 90.0).sum()
        assert low  == 0, f"VIX has {low} values below 5 (suspicious)"
        assert high == 0, f"VIX has {high} values above 90 (suspicious)"

    def test_vix_positive(self, real_vix):
        assert (real_vix["close"] > 0).all(), "VIX contains non-positive values"

    def test_date_index_monotonic(self, real_vix):
        assert real_vix.index.is_monotonic_increasing, "VIX dates not monotonically increasing"

    def test_no_duplicate_dates(self, real_vix):
        dupes = real_vix.index.duplicated().sum()
        assert dupes == 0, f"VIX has {dupes} duplicate dates"

    def test_covid_vix_spike_present(self, real_vix):
        """VIX should have spiked above 60 during COVID crash (March 2020)."""
        mar2020 = real_vix.loc["2020-03-01":"2020-03-31", "close"]
        assert mar2020.max() > 60, \
            f"VIX max in March 2020 = {mar2020.max():.1f} — COVID spike not found"


class TestTQQQData:

    def test_no_missing_close(self, real_tqqq):
        # Filter to post-inception dates only
        post = real_tqqq.loc["2010-02-11":]
        nulls = post["close"].isnull().sum()
        assert nulls == 0, f"TQQQ has {nulls} missing close values after inception"

    def test_prices_positive(self, real_tqqq):
        post = real_tqqq.loc["2010-02-11":]
        assert (post["close"] > 0).all(), "TQQQ has non-positive prices"

    def test_date_index_monotonic(self, real_tqqq):
        assert real_tqqq.index.is_monotonic_increasing

    def test_tqqq_starts_at_inception(self, real_tqqq):
        assert real_tqqq.index[0] <= pd.Timestamp("2010-02-15"), \
            "TQQQ data should start around Feb 11, 2010"


class TestDataAlignment:

    def test_qqq_vix_dates_overlap(self, real_qqq, real_vix):
        """QQQ and VIX must share a large common date range."""
        common = real_qqq.index.intersection(real_vix.index)
        assert len(common) > 3000, \
            f"QQQ-VIX overlap only {len(common)} bars — expected >3000"

    def test_qqq_tqqq_dates_overlap(self, real_qqq, real_tqqq):
        common = real_qqq.loc["2010-02-11":].index.intersection(
            real_tqqq.loc["2010-02-11":].index)
        assert len(common) > 3000
