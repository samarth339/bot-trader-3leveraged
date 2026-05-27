"""
app.py — TQQQ Trading Bot Dashboard
=====================================
Launch:  python3 -m dashboard.app
         python3 dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import dash
import dash_bootstrap_components as dbc
from dash import html

from dashboard.data_loader import load
from dashboard.components import (
    analytics,
    equity_chart,
    historical,
    kpi_cards,
    risk_panel,
    signal_panel,
)

# ── Build app ─────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="TQQQ Bot Dashboard",
    suppress_callback_exceptions=True,
    assets_folder=str(Path(__file__).parent / "assets"),
)


def _layout() -> html.Div:
    """Build full layout from a fresh data load."""
    d = load()

    last_date = str(d.current_signal.get("as_of_date", ""))[:10]
    phase_label = "Phase 4 — Paper Trading" if d.paper_equity is not None else "Phase 3 — Shadow Mode"

    header = html.Div([
        html.Div([
            html.Div("TQQQ / SQQQ Trading Bot", className="header-title"),
            html.Div(phase_label, className="header-subtitle"),
        ]),
        html.Div([
            html.Div(f"Signal date: {last_date}", style={"marginBottom": "2px"}),
            html.Div("Static snapshot — reload to refresh", style={"color": "#484f58"}),
        ], className="header-meta"),
    ], className="header-bar")

    # ── Row 1: KPI strip ──────────────────────────────────────────────────────
    kpi_row = kpi_cards.build(d)

    # ── Row 2: Equity chart (wide) + Signal panel ─────────────────────────────
    row2 = html.Div([
        html.Div(equity_chart.build(d), style={"flex": "2"}),
        html.Div(signal_panel.build(d), style={"flex": "1", "minWidth": "0"}),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "16px", "alignItems": "flex-start"})

    # ── Row 3: Rolling Sharpe + Return distribution ───────────────────────────
    row3 = analytics.build(d)

    # ── Row 4: Performance/VIX + Risk metrics + Allocation + Regime ──────────
    row4 = html.Div([
        html.Div(historical.build(d), style={"flex": "2"}),
        html.Div([
            risk_panel.build_risk_stats(d),
            risk_panel.build_allocation(d),
        ], style={"flex": "1", "minWidth": "0"}),
        html.Div(risk_panel.build_regime_breakdown(d), style={"flex": "1", "minWidth": "0"}),
    ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"})

    return html.Div([
        header,
        kpi_row,
        row2,
        row3,
        row4,
    ], className="dash-page")


app.layout = _layout


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  TQQQ Bot Dashboard")
    print("  Open: http://127.0.0.1:8050\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
