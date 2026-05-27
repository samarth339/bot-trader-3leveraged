"""
test_gap_guard_extended.py — GapGuard Extended Boundary & Field Tests
=======================================================================
Supplements the 4 basic GapGuard tests in test_ibkr_safety.py with:
  - Exact boundary at GAP_THRESHOLD (-5.0%)
  - Sub-threshold cases (barely above and barely below)
  - Positive gap (market opens UP — must never trigger)
  - Zero gap (flat open)
  - GapGuardResult field population (open_price, prev_close, reason)
  - reason string content when triggered vs not triggered
  - Multiple consecutive calls are independent (no shared state)
  - Large-gap edge (10%+, e.g. COVID crash opening)
  - Missing 'close' column in CSV
  - Empty past-dates slice in CSV

All network calls are monkeypatched — no real yfinance calls are made.

Run with:
    pytest tests/test_gap_guard_extended.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_csv(tmp_path: Path, prev_close: float) -> Path:
    """Write a minimal TQQQ_full.csv with one past row."""
    today     = pd.Timestamp.today().normalize()
    yesterday = today - pd.Timedelta(days=1)
    df = pd.DataFrame({"close": [prev_close]}, index=[yesterday])
    csv = tmp_path / "TQQQ_full.csv"
    df.to_csv(csv)
    return csv


def _make_intraday(open_price: float) -> pd.DataFrame:
    """Minimal 1-minute bar DataFrame mirroring yfinance output shape."""
    idx = pd.date_range("2026-01-15 09:30", periods=3, freq="1min", tz="America/New_York")
    return pd.DataFrame(
        {"Open": [open_price, open_price + 0.1, open_price + 0.2],
         "Close": [open_price + 0.05] * 3},
        index=idx,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Boundary at exactly GAP_THRESHOLD
# ══════════════════════════════════════════════════════════════════════════════

class TestGapBoundary:
    """Strict-inequality boundary: triggered iff gap_pct < -0.05 (not ≤)."""

    def test_gap_at_4_99pct_does_NOT_trigger(self, tmp_path, monkeypatch):
        """
        IEEE 754 note: (95.0/100.0) - 1.0 == -0.050000000000000044 (barely under -0.05),
        so using open=95.0 against prev=100.0 actually triggers.  There is no representable
        double that produces gap_pct EXACTLY equal to -0.05 via this formula.

        This test uses open=95.01 (gap = -4.99%) which is unambiguously above the threshold
        and must NOT trigger.
        """
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        open_price = 95.01   # gap = -4.99%  (above threshold, should NOT trigger)
        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(open_price)):
            result = GapGuard().check()

        assert not result.triggered, (
            f"open=95.01 / prev=100 → gap≈-4.99% should NOT trigger. "
            f"gap_pct={result.gap_pct:.6f}"
        )

    def test_one_cent_below_threshold_TRIGGERS(self, tmp_path, monkeypatch):
        """
        prev=100, open=94.99 → gap = -0.0501  →  triggered.
        This is the smallest gap that should fire the guard.
        """
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        open_price = 94.99   # gap = -0.0501
        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(open_price)):
            result = GapGuard().check()

        assert result.triggered, (
            f"open=94.99 / prev=100 → gap=-5.01% should trigger. "
            f"gap_pct={result.gap_pct:.6f}"
        )

    def test_one_cent_above_threshold_does_NOT_trigger(self, tmp_path, monkeypatch):
        """
        prev=100, open=95.01 → gap = -0.0499  →  NOT triggered.
        """
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        open_price = 95.01
        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(open_price)):
            result = GapGuard().check()

        assert not result.triggered, (
            f"open=95.01 / prev=100 → gap=-4.99% should NOT trigger."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Non-triggering cases
# ══════════════════════════════════════════════════════════════════════════════

class TestNonTriggeringCases:
    """Cases that must never fire the guard."""

    def test_positive_gap_up_does_not_trigger(self, tmp_path, monkeypatch):
        """Market gaps UP — this is fine, do not block BUY orders."""
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(110.0)):
            result = GapGuard().check()

        assert not result.triggered
        assert result.gap_pct > 0.0, "Gap-up should yield positive gap_pct"

    def test_zero_gap_flat_open_does_not_trigger(self, tmp_path, monkeypatch):
        """Open exactly equals prior close — flat open should not trigger."""
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(100.0)):
            result = GapGuard().check()

        assert not result.triggered
        assert result.gap_pct == pytest.approx(0.0)

    def test_small_gap_down_2pct_does_not_trigger(self, tmp_path, monkeypatch):
        """A 2% gap-down is within normal noise — must not fire."""
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(98.0)):
            result = GapGuard().check()

        assert not result.triggered


# ══════════════════════════════════════════════════════════════════════════════
#  Result field population
# ══════════════════════════════════════════════════════════════════════════════

class TestResultFields:
    """GapGuardResult must populate all fields correctly."""

    def test_prev_close_matches_csv_value(self, tmp_path, monkeypatch):
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=87.54)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(90.0)):
            result = GapGuard().check()

        assert result.prev_close == pytest.approx(87.54)

    def test_open_price_matches_first_bar(self, tmp_path, monkeypatch):
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(93.75)):
            result = GapGuard().check()

        assert result.open_price == pytest.approx(93.75)

    def test_gap_pct_formula_correct(self, tmp_path, monkeypatch):
        """gap_pct = (open / prev_close) - 1."""
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=200.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(184.0)):
            result = GapGuard().check()

        expected_gap = (184.0 / 200.0) - 1.0   # = -0.08
        assert result.gap_pct == pytest.approx(expected_gap, rel=1e-6)

    def test_reason_populated_when_triggered(self, tmp_path, monkeypatch):
        """When triggered, reason must contain actionable info (prices + threshold)."""
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(90.0)):
            result = GapGuard().check()

        assert result.triggered
        assert len(result.reason) > 0, "reason must not be empty when triggered"
        assert "BUY" in result.reason, "reason should mention 'BUY orders blocked'"
        assert "%" in result.reason, "reason should include gap percentage"

    def test_reason_empty_when_not_triggered(self, tmp_path, monkeypatch):
        """No trigger → reason must be empty string (nothing to alert on)."""
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(102.0)):
            result = GapGuard().check()

        assert not result.triggered
        assert result.reason == "", f"reason should be empty when not triggered, got: '{result.reason}'"


# ══════════════════════════════════════════════════════════════════════════════
#  Large gap scenarios  (extreme events)
# ══════════════════════════════════════════════════════════════════════════════

class TestExtremeGaps:
    """Verify correct behavior at COVID-crash-level gaps."""

    def test_10pct_gap_down_triggers(self, tmp_path, monkeypatch):
        """10% gap-down (e.g. March 2020) must clearly trigger."""
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=50.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(45.0)):
            result = GapGuard().check()

        assert result.triggered
        assert result.gap_pct == pytest.approx(-0.10, rel=1e-6)

    def test_20pct_gap_down_triggers(self, tmp_path, monkeypatch):
        """20% gap-down (extreme crash) must trigger and return correct gap_pct."""
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(80.0)):
            result = GapGuard().check()

        assert result.triggered
        assert result.gap_pct == pytest.approx(-0.20, rel=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
#  Statelessness
# ══════════════════════════════════════════════════════════════════════════════

class TestStateIndependence:
    """Multiple GapGuard instances must be fully independent — no shared state."""

    def test_consecutive_calls_are_independent(self, tmp_path, monkeypatch):
        """
        First call: triggered (large gap).
        Second call: not triggered (small gap).
        Result must reflect the current call, not the previous one.
        """
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(90.0)):
            r1 = GapGuard().check()
        assert r1.triggered

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(101.0)):
            r2 = GapGuard().check()
        assert not r2.triggered, "Second call result must not be affected by first call"

    def test_two_instances_same_call_same_result(self, tmp_path, monkeypatch):
        """Two GapGuard instances called with same inputs must give same result."""
        from ibkr.gap_guard import GapGuard

        csv = _make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(93.0)):
            r1 = GapGuard().check()
            r2 = GapGuard().check()

        assert r1.triggered == r2.triggered
        assert r1.gap_pct == pytest.approx(r2.gap_pct)


# ══════════════════════════════════════════════════════════════════════════════
#  Malformed CSV handling
# ══════════════════════════════════════════════════════════════════════════════

class TestMalformedCSV:
    """GapGuard must fail-open (not block) when CSV data is bad."""

    def test_missing_close_column_fails_open(self, tmp_path, monkeypatch):
        """CSV without 'close' column must not crash — guard skipped gracefully."""
        from ibkr.gap_guard import GapGuard

        today     = pd.Timestamp.today().normalize()
        yesterday = today - pd.Timedelta(days=1)
        bad_df = pd.DataFrame({"volume": [1_000_000]}, index=[yesterday])
        csv = tmp_path / "TQQQ_full.csv"
        bad_df.to_csv(csv)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(95.0)):
            result = GapGuard().check()

        assert not result.triggered, "Guard must fail-open on malformed CSV"

    def test_empty_csv_fails_open(self, tmp_path, monkeypatch):
        """Completely empty CSV must not crash — guard skipped gracefully."""
        from ibkr.gap_guard import GapGuard

        csv = tmp_path / "TQQQ_full.csv"
        pd.DataFrame({"close": []}).to_csv(csv)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(95.0)):
            result = GapGuard().check()

        assert not result.triggered, "Guard must fail-open on empty CSV"

    def test_only_today_rows_in_csv_fails_open(self, tmp_path, monkeypatch):
        """If CSV only has today's partial row, no prev_close is available."""
        from ibkr.gap_guard import GapGuard

        today = pd.Timestamp.today().normalize()
        df    = pd.DataFrame({"close": [100.0]}, index=[today])
        csv   = tmp_path / "TQQQ_full.csv"
        df.to_csv(csv)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv)

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(90.0)):
            result = GapGuard().check()

        assert not result.triggered, (
            "With only today's row in CSV, prev_close is unavailable — guard must fail-open"
        )

    def test_nonexistent_csv_fails_open(self, tmp_path, monkeypatch):
        """Missing CSV file must not raise — guard should log warning and skip."""
        from ibkr.gap_guard import GapGuard

        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", tmp_path / "does_not_exist.csv")

        with patch("ibkr.gap_guard.yf.download", return_value=_make_intraday(90.0)):
            result = GapGuard().check()

        assert not result.triggered, "Guard must fail-open if CSV does not exist"
