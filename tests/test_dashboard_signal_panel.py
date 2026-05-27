"""
test_dashboard_signal_panel.py — Signal Panel Gap Column Tests
===============================================================
Validates the dashboard/components/signal_panel.py gap guard column
added in commit 28622e1.

Tests:
  - Old rows (no gap_pct column) render "—" in the gap cell
  - gap_guard=True + valid gap_pct renders 🚫 in red
  - gap_guard=False + valid gap_pct renders green percentage
  - gap_pct="" (empty string from skipped guard) renders "—"
  - gap_pct NaN renders "—"
  - Negative gap percentage shows minus sign
  - Positive gap percentage shows plus sign
  - Table header contains "Gap" column
  - Column ordering: Gap appears between VIX and 5d Fwd
  - build() renders without crash for any valid DashboardData

Run with:
    pytest tests/test_dashboard_signal_panel.py -v
"""

import pytest
import pandas as pd
import numpy as np
from dash import html


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_signal_row(**overrides) -> dict:
    """Build a minimal signal row that signal_panel can render."""
    row = {
        "as_of_date":   "2026-05-20",
        "regime":       "bull",
        "action":       "HOLD",
        "weight_a":     0.75,
        "weight_b":     0.25,
        "qqq_price":    480.0,
        "vix_raw":      15.0,
        "tqqq_5d_fwd":  float("nan"),
        "outcome":      "—",
        "gap_guard":    False,
        "gap_pct":      0.43,
    }
    row.update(overrides)
    return row


def _make_df(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


def _get_gap_td(df: pd.DataFrame) -> html.Td:
    """
    Render the signal table and extract the gap cell (index 7) from the
    first data row.
    """
    from dashboard.components.signal_panel import _signal_table

    table_div = _signal_table(df)
    table     = table_div.children             # html.Table
    tbody     = table.children[1]              # html.Tbody
    first_tr  = tbody.children[0]             # first data row
    return first_tr.children[7]               # 8th column = Gap (0-indexed: Date[0]..VIX[6]..Gap[7])


def _get_header_texts(df: pd.DataFrame) -> list:
    """Extract all header <th> text values."""
    from dashboard.components.signal_panel import _signal_table

    table_div = _signal_table(df)
    table     = table_div.children
    thead     = table.children[0]
    header_tr = thead.children
    return [th.children for th in header_tr.children]


# ══════════════════════════════════════════════════════════════════════════════
#  Table header
# ══════════════════════════════════════════════════════════════════════════════

class TestSignalTableHeader:
    """The table header must include the Gap column at the right position."""

    def test_gap_column_in_header(self):
        df = _make_df(_make_signal_row())
        headers = _get_header_texts(df)
        assert "Gap" in headers, f"'Gap' column header not found. Headers: {headers}"

    def test_gap_column_position(self):
        """Gap must appear after VIX and before 5d Fwd."""
        df      = _make_df(_make_signal_row())
        headers = _get_header_texts(df)
        gap_idx = headers.index("Gap")
        vix_idx = headers.index("VIX")
        fwd_idx = headers.index("5d Fwd")
        assert vix_idx < gap_idx < fwd_idx, (
            f"Gap column must be between VIX and 5d Fwd. "
            f"Actual order: VIX={vix_idx}, Gap={gap_idx}, 5d Fwd={fwd_idx}"
        )

    def test_header_has_10_columns(self):
        """Header must have exactly 10 columns: Date Regime Action A% B% QQQ VIX Gap 5dFwd ✓"""
        df      = _make_df(_make_signal_row())
        headers = _get_header_texts(df)
        assert len(headers) == 10, (
            f"Expected 10 header columns, got {len(headers)}: {headers}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Gap cell rendering: triggered
# ══════════════════════════════════════════════════════════════════════════════

class TestGapCellTriggered:
    """When gap_guard=True, the cell must show 🚫 with red styling."""

    def test_triggered_cell_contains_stop_emoji(self):
        df  = _make_df(_make_signal_row(gap_guard=True, gap_pct=-7.23))
        td  = _get_gap_td(df)
        txt = str(td.children)
        assert "🚫" in txt, f"Triggered gap cell must contain 🚫. Got: '{txt}'"

    def test_triggered_cell_shows_percentage(self):
        df  = _make_df(_make_signal_row(gap_guard=True, gap_pct=-7.23))
        td  = _get_gap_td(df)
        txt = str(td.children)
        assert "7.2" in txt, f"Triggered cell must show percentage. Got: '{txt}'"

    def test_triggered_cell_has_red_color(self):
        df  = _make_df(_make_signal_row(gap_guard=True, gap_pct=-7.23))
        td  = _get_gap_td(df)
        style = td.style or {}
        color = style.get("color", "")
        assert "#f85149" in color or "red" in color.lower(), (
            f"Triggered gap cell must have red color. Got style: {style}"
        )

    def test_triggered_cell_has_bold_font(self):
        df    = _make_df(_make_signal_row(gap_guard=True, gap_pct=-7.23))
        td    = _get_gap_td(df)
        style = td.style or {}
        fw    = style.get("fontWeight", "")
        assert fw == "700", f"Triggered gap cell must be bold (fontWeight=700). Got: '{fw}'"

    def test_triggered_cell_negative_sign(self):
        """Triggered gap (gap-down) must show negative percentage."""
        df  = _make_df(_make_signal_row(gap_guard=True, gap_pct=-5.5))
        td  = _get_gap_td(df)
        txt = str(td.children)
        assert "-" in txt, f"Triggered gap must show negative %. Got: '{txt}'"


# ══════════════════════════════════════════════════════════════════════════════
#  Gap cell rendering: clear (not triggered)
# ══════════════════════════════════════════════════════════════════════════════

class TestGapCellClear:
    """When gap_guard=False with valid gap_pct, the cell shows green percentage."""

    def test_clear_cell_shows_percentage(self):
        df  = _make_df(_make_signal_row(gap_guard=False, gap_pct=1.25))
        td  = _get_gap_td(df)
        txt = str(td.children)
        assert "1.2" in txt or "1.25" in txt, \
            f"Clear gap cell must show percentage. Got: '{txt}'"

    def test_clear_cell_has_green_color(self):
        df    = _make_df(_make_signal_row(gap_guard=False, gap_pct=0.43))
        td    = _get_gap_td(df)
        style = td.style or {}
        color = style.get("color", "")
        assert "#3fb950" in color or "green" in color.lower(), (
            f"Clear gap cell must have green color. Got style: {style}"
        )

    def test_clear_cell_positive_shows_plus(self):
        """A positive gap-open should display with a '+' sign."""
        df  = _make_df(_make_signal_row(gap_guard=False, gap_pct=2.34))
        td  = _get_gap_td(df)
        txt = str(td.children)
        assert "+" in txt, f"Positive gap must show '+' sign. Got: '{txt}'"

    def test_clear_cell_no_stop_emoji(self):
        """Clear gap cell must not contain the stop emoji."""
        df  = _make_df(_make_signal_row(gap_guard=False, gap_pct=0.87))
        td  = _get_gap_td(df)
        txt = str(td.children)
        assert "🚫" not in txt, "Clear gap cell must not contain 🚫"


# ══════════════════════════════════════════════════════════════════════════════
#  Gap cell rendering: missing / unavailable data
# ══════════════════════════════════════════════════════════════════════════════

class TestGapCellMissingData:
    """Old rows or skipped checks must render gracefully as '—'."""

    def test_empty_string_gap_pct_renders_dash(self):
        """gap_pct='' (gap check skipped for historical row) → show '—'."""
        df  = _make_df(_make_signal_row(gap_guard=False, gap_pct=""))
        td  = _get_gap_td(df)
        assert "—" in str(td.children), \
            f"Empty gap_pct must render as '—'. Got: '{td.children}'"

    def test_nan_gap_pct_renders_dash(self):
        """gap_pct=NaN → show '—'."""
        df  = _make_df(_make_signal_row(gap_guard=False, gap_pct=float("nan")))
        td  = _get_gap_td(df)
        assert "—" in str(td.children), \
            f"NaN gap_pct must render as '—'. Got: '{td.children}'"

    def test_none_gap_pct_renders_dash(self):
        """gap_pct=None (missing column) → show '—'."""
        df  = _make_df(_make_signal_row(gap_guard=False, gap_pct=None))
        td  = _get_gap_td(df)
        assert "—" in str(td.children), \
            f"None gap_pct must render as '—'. Got: '{td.children}'"

    def test_missing_gap_columns_entirely_renders_dash(self):
        """Old rows with no gap_pct column at all → show '—'."""
        row = {
            "as_of_date":   "2026-03-10",
            "regime":       "uncertain",
            "action":       "REDUCE_A",
            "weight_a":     0.55,
            "weight_b":     0.45,
            "qqq_price":    440.0,
            "vix_raw":      21.1,
            "tqqq_5d_fwd":  float("nan"),
            "outcome":      "—",
            # no gap_guard, no gap_pct
        }
        df = pd.DataFrame([row])
        td = _get_gap_td(df)
        assert "—" in str(td.children), \
            f"Row without gap columns must render '—'. Got: '{td.children}'"

    def test_missing_gap_columns_cell_has_muted_color(self):
        """'—' cells must use muted color (not red or green)."""
        row = {
            "as_of_date":   "2026-03-10",
            "regime":       "uncertain",
            "action":       "HOLD",
            "weight_a":     0.65,
            "weight_b":     0.35,
            "qqq_price":    440.0,
            "vix_raw":      20.0,
            "tqqq_5d_fwd":  float("nan"),
            "outcome":      "—",
        }
        df    = pd.DataFrame([row])
        td    = _get_gap_td(df)
        style = td.style or {}
        color = style.get("color", "")
        assert "#484f58" in color, (
            f"Missing-data gap cell must use muted color #484f58. Got: '{color}'"
        )

    def test_string_nan_gap_pct_renders_dash(self):
        """gap_pct='nan' (pandas NaN as string) → show '—'."""
        df  = _make_df(_make_signal_row(gap_guard=False, gap_pct="nan"))
        td  = _get_gap_td(df)
        assert "—" in str(td.children), \
            f"String 'nan' gap_pct must render as '—'. Got: '{td.children}'"


# ══════════════════════════════════════════════════════════════════════════════
#  Table structure
# ══════════════════════════════════════════════════════════════════════════════

class TestTableStructure:
    """Overall structure of the rendered signal table."""

    def test_table_renders_without_crash_for_mixed_rows(self):
        """Table with mixed old/new rows must not raise."""
        rows = [
            _make_signal_row(as_of_date="2026-05-20", gap_guard=True,  gap_pct=-6.1),
            _make_signal_row(as_of_date="2026-05-19", gap_guard=False, gap_pct=0.9),
            {
                "as_of_date": "2026-03-10", "regime": "bull", "action": "HOLD",
                "weight_a": 0.75, "weight_b": 0.25, "qqq_price": 450.0,
                "vix_raw": 16.0, "tqqq_5d_fwd": float("nan"), "outcome": "—",
            },
        ]
        df = pd.DataFrame(rows)
        from dashboard.components.signal_panel import _signal_table
        result = _signal_table(df)
        assert result is not None

    def test_each_row_has_10_cells(self):
        """Every data row must have exactly 10 cells (matching the 10-column header)."""
        df = _make_df(
            _make_signal_row(gap_guard=True,  gap_pct=-7.0),
            _make_signal_row(gap_guard=False, gap_pct=1.2),
        )
        from dashboard.components.signal_panel import _signal_table

        table_div = _signal_table(df)
        table     = table_div.children
        tbody     = table.children[1]

        for i, tr in enumerate(tbody.children):
            assert len(tr.children) == 10, (
                f"Row {i} has {len(tr.children)} cells, expected 10"
            )

    def test_empty_dataframe_returns_placeholder(self):
        """Empty signals DataFrame must return a 'no signal history' placeholder."""
        from dashboard.components.signal_panel import _signal_table

        result = _signal_table(pd.DataFrame())
        # Should be a Div, not a Table
        assert isinstance(result, html.Div)
        # Should not be a table-wrap (no table in children)
        children = result.children
        if isinstance(children, str):
            assert "No signal" in children or len(children) > 0
