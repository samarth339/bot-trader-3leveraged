"""
tests/test_dashboard_data_loader.py
====================================
Tests for dashboard/data_loader._load_signals() gap column behaviour.

Covers:
  - CSV WITH gap_guard / gap_pct columns → both columns present in result
  - CSV WITHOUT those columns           → both columns still present (filled NaN/empty)
  - Values in gap columns are preserved exactly as written in the CSV
  - Existing non-gap columns are not disturbed

No real network calls; TQQQ price series is provided as a minimal in-memory
pd.Series, and LOGS is monkeypatched to a tmp_path directory.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import numpy as np
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_tqqq_close(dates: list[str], prices: list[float]) -> pd.Series:
    """Build a minimal TQQQ close price Series indexed by Timestamp."""
    idx = pd.to_datetime(dates)
    return pd.Series(prices, index=idx, name="close")


def _write_signal_csv(path: Path, rows: list[dict]) -> None:
    """Write a minimal signal_history.csv to *path*."""
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _call_load_signals(tmp_path: Path, csv_rows: list[dict]) -> pd.DataFrame:
    """
    Write csv_rows to tmp_path/signal_history.csv, monkeypatch LOGS,
    and call _load_signals() with a minimal tqqq_close series.
    Returns the resulting DataFrame.
    """
    import dashboard.data_loader as dl

    csv_path = tmp_path / "signal_history.csv"
    _write_signal_csv(csv_path, csv_rows)

    # Build a tqqq_close series that spans all as_of_dates in the CSV.
    dates  = [r["as_of_date"] for r in csv_rows]
    prices = [50.0 + i for i in range(len(dates))]
    tqqq_close = _make_tqqq_close(dates, prices)

    original_logs = dl.LOGS
    dl.LOGS = tmp_path
    try:
        return dl._load_signals(tqqq_close)
    finally:
        dl.LOGS = original_logs


# ── Minimal valid signal row factory ──────────────────────────────────────────

def _base_row(as_of_date: str = "2026-05-01") -> dict:
    """Return a minimal row matching the signal_history.csv schema."""
    return {
        "as_of_date":  as_of_date,
        "signal_date": as_of_date,
        "regime":      "bull",
        "action":      "HOLD",
        "weight_a":    0.9,
        "weight_b":    0.1,
        "rebalance":   False,
        "drift_pct":   0.0,
        "qqq_price":   480.0,
        "sma_val":     460.0,
        "pct_vs_sma":  4.3,
        "vix_signal":  15.0,
        "vix_raw":     15.5,
        "shadow":      True,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Test suite
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadSignalsGapColumns:
    """_load_signals() gap column preservation."""

    # ── Test 1: CSV WITH gap columns → columns present and values correct ──────

    def test_gap_columns_present_when_csv_has_them(self, tmp_path):
        """
        When signal_history.csv contains gap_guard and gap_pct columns,
        the returned DataFrame must include them with their original values.
        """
        row = {**_base_row("2026-05-05"), "gap_guard": True, "gap_pct": -0.072}
        result = _call_load_signals(tmp_path, [row])

        assert "gap_guard" in result.columns, (
            "gap_guard column must be present when the CSV contains it"
        )
        assert "gap_pct" in result.columns, (
            "gap_pct column must be present when the CSV contains it"
        )

    def test_gap_values_preserved_correctly(self, tmp_path):
        """
        gap_pct and gap_guard values written to the CSV must survive the
        read → merge → return pipeline without modification.
        """
        row = {**_base_row("2026-05-06"), "gap_guard": True, "gap_pct": -0.083}
        result = _call_load_signals(tmp_path, [row])

        assert result.iloc[0]["gap_guard"] == True
        assert abs(float(result.iloc[0]["gap_pct"]) - (-0.083)) < 1e-9

    def test_non_triggered_gap_values_preserved(self, tmp_path):
        """
        A row with gap_guard=False and a small positive gap_pct must be
        preserved exactly — the value is not forced to NaN or empty.
        """
        row = {**_base_row("2026-05-07"), "gap_guard": False, "gap_pct": 0.012}
        result = _call_load_signals(tmp_path, [row])

        assert result.iloc[0]["gap_guard"] == False
        assert abs(float(result.iloc[0]["gap_pct"]) - 0.012) < 1e-9

    # ── Test 2: CSV WITHOUT gap columns → columns still present (NaN/empty) ───

    def test_gap_columns_added_when_csv_lacks_them(self, tmp_path):
        """
        When signal_history.csv does NOT contain gap_guard / gap_pct
        (historical rows before the gap guard feature was added),
        _load_signals() must still return a DataFrame that has both columns.
        signal_panel.py relies on them always being present.
        """
        row = _base_row("2026-04-01")   # no gap columns
        result = _call_load_signals(tmp_path, [row])

        assert "gap_guard" in result.columns, (
            "gap_guard column must be added even when absent from the CSV"
        )
        assert "gap_pct" in result.columns, (
            "gap_pct column must be added even when absent from the CSV"
        )

    def test_missing_gap_guard_filled_with_empty_string(self, tmp_path):
        """
        When gap_guard is absent from the CSV, the back-filled value should be
        '' (empty string) so that signal_panel's truthy check
        str(...).lower() in ('true','1','yes') evaluates to False.
        """
        row = _base_row("2026-04-02")
        result = _call_load_signals(tmp_path, [row])

        val = result.iloc[0]["gap_guard"]
        assert str(val).lower() not in ("true", "1", "yes"), (
            f"Back-filled gap_guard '{val}' must not evaluate as triggered"
        )

    def test_missing_gap_pct_filled_with_nan(self, tmp_path):
        """
        When gap_pct is absent from the CSV, the back-filled value should be
        NaN (float) so that signal_panel's pd.notna(gap_raw) check renders
        the cell as '—'.
        """
        row = _base_row("2026-04-03")
        result = _call_load_signals(tmp_path, [row])

        val = result.iloc[0]["gap_pct"]
        # Accept either NaN or empty string as the fill value — both cause
        # signal_panel to render '—'
        is_nan_or_empty = (
            (isinstance(val, float) and math.isnan(val))
            or str(val) in ("", "nan")
        )
        assert is_nan_or_empty, (
            f"Back-filled gap_pct '{val}' should be NaN or empty so the "
            "dashboard renders '—' for historic rows"
        )

    # ── Test 3: Mixed CSV (some rows have gap data, some don't) ───────────────

    def test_mixed_rows_gap_columns_coexist(self, tmp_path):
        """
        A CSV with some rows containing gap data and some without (e.g. a file
        that was started before the gap guard feature) must produce a DataFrame
        where the gap columns exist and older rows have NaN/empty values while
        newer rows retain their logged values.
        """
        rows = [
            _base_row("2026-03-27"),   # pre-feature, no gap columns
            {**_base_row("2026-05-20"), "gap_guard": False, "gap_pct": 0.005},
            {**_base_row("2026-05-21"), "gap_guard": True,  "gap_pct": -0.062},
        ]
        result = _call_load_signals(tmp_path, rows)

        assert "gap_guard" in result.columns
        assert "gap_pct"   in result.columns
        assert len(result) == 3

        # Sort ascending to get predictable row order
        result = result.sort_values("as_of_date").reset_index(drop=True)

        # Row 0 (pre-feature) — gap_pct should be NaN or empty
        old_gap = result.iloc[0]["gap_pct"]
        is_missing = (
            (isinstance(old_gap, float) and math.isnan(old_gap))
            or str(old_gap) in ("", "nan")
        )
        assert is_missing, (
            f"Pre-feature row gap_pct '{old_gap}' should be NaN or empty"
        )

        # Row 1 — gap not triggered, small positive gap
        assert result.iloc[1]["gap_guard"] == False
        assert abs(float(result.iloc[1]["gap_pct"]) - 0.005) < 1e-9

        # Row 2 — gap triggered, negative gap
        assert result.iloc[2]["gap_guard"] == True
        assert float(result.iloc[2]["gap_pct"]) < -0.05

    # ── Test 4: Other columns are unaffected ──────────────────────────────────

    def test_existing_columns_not_disturbed(self, tmp_path):
        """
        The gap column fix must not remove or rename any pre-existing columns.
        Core signal columns must all survive.
        """
        row = {**_base_row("2026-05-10"), "gap_guard": False, "gap_pct": 0.001}
        result = _call_load_signals(tmp_path, [row])

        required = [
            "as_of_date", "signal_date", "regime", "action",
            "weight_a", "weight_b", "rebalance",
            "qqq_price", "pct_vs_sma", "vix_signal", "vix_raw", "shadow",
            "tqqq_5d_fwd", "outcome",
            "gap_guard", "gap_pct",
        ]
        missing = [c for c in required if c not in result.columns]
        assert not missing, f"Columns missing from _load_signals() result: {missing}"

    # ── Test 5: Multiple rows with gap data — all values preserved ────────────

    def test_multiple_gap_rows_all_values_preserved(self, tmp_path):
        """
        When every row in the CSV has gap data, all values must be preserved
        (no accidental NaN injection from the fix).
        """
        rows = [
            {**_base_row("2026-05-12"), "gap_guard": False, "gap_pct":  0.009},
            {**_base_row("2026-05-13"), "gap_guard": True,  "gap_pct": -0.055},
            {**_base_row("2026-05-14"), "gap_guard": False, "gap_pct":  0.003},
        ]
        result = _call_load_signals(tmp_path, rows)
        result = result.sort_values("as_of_date").reset_index(drop=True)

        expected_pcts = [0.009, -0.055, 0.003]
        for i, expected in enumerate(expected_pcts):
            assert abs(float(result.iloc[i]["gap_pct"]) - expected) < 1e-9, (
                f"Row {i}: expected gap_pct {expected}, "
                f"got {result.iloc[i]['gap_pct']}"
            )
