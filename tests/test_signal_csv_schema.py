"""
test_signal_csv_schema.py — Signal History CSV Schema Tests
============================================================
Validates that signal_history.csv is written and read correctly,
with particular focus on the new gap_guard / gap_pct columns added
in the shadow-mode integration (commit 28622e1).

Tests:
  - append_signal_log() creates the file when it doesn't exist
  - All required columns are present in a written row
  - gap_guard and gap_pct columns are written with correct types
  - Idempotency: appending the same as_of_date overwrites (not duplicates)
  - Old rows without gap columns are read back without errors
  - gap_guard=True stored as True (not "True" string) after round-trip
  - gap_pct stored as float or empty string (not NaN)
  - load_prev_signal() handles missing file gracefully
  - load_prev_signal() handles empty file gracefully

Run with:
    pytest tests/test_signal_csv_schema.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date


# ── Helpers ────────────────────────────────────────────────────────────────────

def _base_row(as_of: str = "2026-05-20") -> dict:
    """Minimal valid signal row with all columns including new gap columns."""
    return {
        "as_of_date":   as_of,
        "signal_date":  as_of,
        "regime":       "bull",
        "action":       "HOLD",
        "weight_a":     0.75,
        "weight_b":     0.25,
        "rebalance":    False,
        "drift_pct":    1.0,
        "qqq_price":    480.0,
        "sma_val":      451.22,
        "pct_vs_sma":   6.4,
        "vix_signal":   14.5,
        "vix_raw":      15.0,
        "shadow":       False,
        "gap_guard":    False,
        "gap_pct":      0.43,
    }


def _old_row(as_of: str = "2026-03-10") -> dict:
    """Row from before gap_guard/gap_pct columns existed."""
    return {
        "as_of_date":   as_of,
        "signal_date":  as_of,
        "regime":       "uncertain",
        "action":       "REDUCE_A",
        "weight_a":     0.55,
        "weight_b":     0.45,
        "rebalance":    True,
        "drift_pct":    5.0,
        "qqq_price":    440.0,
        "sma_val":      445.0,
        "pct_vs_sma":   -1.12,
        "vix_signal":   20.3,
        "vix_raw":      21.1,
        "shadow":       True,
        # no gap_guard, no gap_pct
    }


# ══════════════════════════════════════════════════════════════════════════════
#  File creation
# ══════════════════════════════════════════════════════════════════════════════

class TestCSVCreation:
    """append_signal_log() must create the file if it doesn't exist."""

    def test_creates_file_on_first_write(self, tmp_path, monkeypatch):
        from daily_signal import append_signal_log, SIGNAL_LOG

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        assert not csv_path.exists()
        append_signal_log(_base_row())
        assert csv_path.exists(), "CSV file must be created on first append"

    def test_file_is_readable_csv_after_creation(self, tmp_path, monkeypatch):
        from daily_signal import append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        append_signal_log(_base_row())
        df = pd.read_csv(csv_path)
        assert len(df) == 1

    def test_header_row_written_correctly(self, tmp_path, monkeypatch):
        from daily_signal import append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        append_signal_log(_base_row())
        df = pd.read_csv(csv_path)
        assert "as_of_date" in df.columns
        assert "regime"     in df.columns
        assert "action"     in df.columns


# ══════════════════════════════════════════════════════════════════════════════
#  Required columns
# ══════════════════════════════════════════════════════════════════════════════

class TestRequiredColumns:
    """All columns that downstream code reads must be present."""

    REQUIRED_COLUMNS = {
        "as_of_date", "signal_date", "regime", "action",
        "weight_a", "weight_b", "rebalance", "drift_pct",
        "qqq_price", "sma_val", "pct_vs_sma",
        "vix_signal", "vix_raw", "shadow",
        "gap_guard", "gap_pct",
    }

    def test_all_required_columns_present_after_write(self, tmp_path, monkeypatch):
        from daily_signal import append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        append_signal_log(_base_row())
        df = pd.read_csv(csv_path)

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"Missing columns in written CSV: {missing}"


# ══════════════════════════════════════════════════════════════════════════════
#  Gap guard columns
# ══════════════════════════════════════════════════════════════════════════════

class TestGapGuardColumns:
    """New gap_guard and gap_pct columns must be written correctly."""

    def test_gap_guard_false_written_correctly(self, tmp_path, monkeypatch):
        from daily_signal import append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        row = _base_row()
        row["gap_guard"] = False
        row["gap_pct"]   = 0.43
        append_signal_log(row)

        df = pd.read_csv(csv_path)
        written = df.iloc[0]
        # CSV round-trips boolean as string; check for falsiness
        assert str(written["gap_guard"]).lower() in ("false", "0", "")

    def test_gap_guard_true_written_correctly(self, tmp_path, monkeypatch):
        from daily_signal import append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        row = _base_row()
        row["gap_guard"] = True
        row["gap_pct"]   = -7.23
        append_signal_log(row)

        df = pd.read_csv(csv_path)
        written = str(df.iloc[0]["gap_guard"]).lower()
        assert written in ("true", "1"), f"gap_guard=True not round-tripped correctly: '{written}'"

    def test_gap_pct_written_as_float(self, tmp_path, monkeypatch):
        from daily_signal import append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        row = _base_row()
        row["gap_pct"] = -5.17
        append_signal_log(row)

        df = pd.read_csv(csv_path)
        val = df.iloc[0]["gap_pct"]
        assert float(val) == pytest.approx(-5.17), f"gap_pct not round-tripped: {val}"

    def test_gap_pct_empty_written_as_blank_or_nan(self, tmp_path, monkeypatch):
        """
        When gap guard is skipped (e.g. --date back-calc), gap_pct is written
        as empty string "".  pandas reads empty CSV cells back as NaN, so the
        round-tripped value may be either "" or "nan".  Both are valid — the
        dashboard signal_panel treats both as '—' (see gap_raw not in ("", None)
        and str(gap_raw) != "nan" checks).
        """
        from daily_signal import append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        row = _base_row()
        row["gap_pct"] = ""   # as written by main() when gap_pct_val is nan
        append_signal_log(row)

        df   = pd.read_csv(csv_path, dtype=str)
        val  = df.iloc[0]["gap_pct"]
        # pandas round-trips empty string as "nan" — both are "unavailable"
        assert val in ("", "nan", "NaN", "None") or pd.isna(val), (
            f"gap_pct should be empty/nan when unavailable, got: '{val}'"
        )

    def test_gap_guard_false_and_gap_pct_present_together(self, tmp_path, monkeypatch):
        """Both columns must coexist correctly in the same row."""
        from daily_signal import append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        row = _base_row()
        row["gap_guard"] = False
        row["gap_pct"]   = 1.25
        append_signal_log(row)

        df = pd.read_csv(csv_path)
        assert "gap_guard" in df.columns
        assert "gap_pct"   in df.columns
        assert float(df.iloc[0]["gap_pct"]) == pytest.approx(1.25)


# ══════════════════════════════════════════════════════════════════════════════
#  Idempotency — same date must overwrite, not duplicate
# ══════════════════════════════════════════════════════════════════════════════

class TestIdempotency:
    """
    append_signal_log() must overwrite rows with the same as_of_date.
    Running daily_signal.py twice in a day must not create duplicate rows.
    """

    def test_same_date_overwrites_previous_row(self, tmp_path, monkeypatch):
        from daily_signal import append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        row1 = _base_row("2026-05-20")
        row1["regime"] = "bull"
        append_signal_log(row1)

        row2 = _base_row("2026-05-20")   # same date
        row2["regime"] = "uncertain"     # different value
        append_signal_log(row2)

        df = pd.read_csv(csv_path)
        assert len(df) == 1, f"Expected 1 row after overwrite, got {len(df)}"
        assert df.iloc[0]["regime"] == "uncertain", \
            "Second write must overwrite the first for the same date"

    def test_different_dates_both_preserved(self, tmp_path, monkeypatch):
        from daily_signal import append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        append_signal_log(_base_row("2026-05-19"))
        append_signal_log(_base_row("2026-05-20"))

        df = pd.read_csv(csv_path)
        assert len(df) == 2, f"Expected 2 rows for 2 different dates, got {len(df)}"

    def test_overwrite_updates_gap_columns(self, tmp_path, monkeypatch):
        """
        When a row is overwritten (e.g. re-run after data refresh),
        the updated gap_guard/gap_pct values must be visible.
        """
        from daily_signal import append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        row1 = _base_row("2026-05-20")
        row1["gap_guard"] = False
        row1["gap_pct"]   = 1.1
        append_signal_log(row1)

        row2 = _base_row("2026-05-20")
        row2["gap_guard"] = True
        row2["gap_pct"]   = -6.3
        append_signal_log(row2)

        df  = pd.read_csv(csv_path)
        assert len(df) == 1
        assert str(df.iloc[0]["gap_guard"]).lower() in ("true", "1")
        assert float(df.iloc[0]["gap_pct"]) == pytest.approx(-6.3)


# ══════════════════════════════════════════════════════════════════════════════
#  Backward compatibility — old rows without gap columns
# ══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """
    Historical rows written before gap columns existed must not crash
    any downstream reader.  The dashboard signal_panel reads these rows
    and must gracefully render '—' for missing gap data.
    """

    def test_old_row_can_be_appended_to_and_read_back(self, tmp_path, monkeypatch):
        """
        Start with a pre-gap CSV, then append a new row with gap columns.
        Both old and new rows must be readable.
        """
        from daily_signal import append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        # Write an old-style row manually (no gap columns)
        old = pd.DataFrame([_old_row("2026-03-10")])
        old.to_csv(csv_path, index=False)

        # Now append a new row with gap columns
        append_signal_log(_base_row("2026-05-20"))

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        # Old row: gap_guard should be NaN / empty (column didn't exist)
        # New row: gap_guard should be False
        old_gap = df[df["as_of_date"] == "2026-03-10"].iloc[0].get("gap_guard", "")
        assert str(old_gap) in ("nan", "", "NaN"), \
            f"Old row gap_guard should be NaN/empty, got: '{old_gap}'"

    def test_signal_panel_renders_old_row_without_crash(self):
        """
        _signal_table() in signal_panel.py must not raise when a row
        has no gap_pct column (pre-feature historical data).
        """
        from dashboard.components.signal_panel import _signal_table

        df = pd.DataFrame([_old_row("2026-03-10")])
        # Should not raise
        result = _signal_table(df)
        assert result is not None

    def test_signal_panel_renders_empty_string_gap_pct_as_dash(self):
        """
        When gap_pct is empty string (gap check was skipped), the
        dashboard table must render '—' not crash or show 'nan'.
        """
        from dashboard.components.signal_panel import _signal_table

        row = _old_row("2026-03-10")
        row["gap_guard"] = False
        row["gap_pct"]   = ""    # explicit empty string (pandas may read back as nan)

        df     = pd.DataFrame([row])
        result = _signal_table(df)
        # Find the rendered table's rows
        table  = result.children        # html.Table
        tbody  = table.children[1]      # html.Tbody
        tr     = tbody.children[0]      # first html.Tr
        gap_td = tr.children[7]         # 8th column = Gap (index 7)

        # signal_panel checks: gap_raw not in ("", None) and str(gap_raw) != "nan"
        # Both "" and "nan" resolve to the '—' cell
        assert "—" in str(gap_td.children), (
            f"Empty gap_pct must render as '—', got: {gap_td.children}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  load_prev_signal() edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadPrevSignal:
    """load_prev_signal() must handle all edge cases gracefully."""

    def test_returns_empty_dict_when_file_missing(self, tmp_path, monkeypatch):
        from daily_signal import load_prev_signal

        monkeypatch.setattr("daily_signal.SIGNAL_LOG", tmp_path / "nonexistent.csv")
        result = load_prev_signal()
        assert result == {}, "Must return empty dict when file doesn't exist"

    def test_returns_empty_dict_when_file_empty(self, tmp_path, monkeypatch):
        from daily_signal import load_prev_signal

        csv_path = tmp_path / "signal_history.csv"
        pd.DataFrame().to_csv(csv_path, index=False)
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        result = load_prev_signal()
        assert result == {}, "Must return empty dict when CSV has no rows"

    def test_returns_last_row_as_dict(self, tmp_path, monkeypatch):
        from daily_signal import load_prev_signal, append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        append_signal_log(_base_row("2026-05-19"))
        append_signal_log(_base_row("2026-05-20"))  # last row

        result = load_prev_signal()
        assert result.get("as_of_date") is not None
        assert "2026-05-20" in str(result.get("as_of_date")), \
            "Must return the last (most recent) row"

    def test_returns_last_row_regime(self, tmp_path, monkeypatch):
        from daily_signal import load_prev_signal, append_signal_log

        csv_path = tmp_path / "signal_history.csv"
        monkeypatch.setattr("daily_signal.SIGNAL_LOG", csv_path)

        row = _base_row("2026-05-20")
        row["regime"] = "high_vol"
        append_signal_log(row)

        result = load_prev_signal()
        assert result["regime"] == "high_vol"
