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
from dash import dcc, html, Input, Output, ctx, no_update

from dashboard.data_loader import load
from dashboard.components import (
    analytics,
    equity_chart,
    historical,
    kpi_cards,
    paper_panel,
    risk_panel,
    signal_panel,
)

REFRESH_SECONDS = 60   # auto-refresh cadence

# ── Build app ─────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="TQQQ Bot Dashboard",
    suppress_callback_exceptions=True,
    assets_folder=str(Path(__file__).parent / "assets"),
)


_FLEX = {"display": "flex", "gap": "16px", "marginBottom": "16px", "alignItems": "flex-start"}


def _dashboard_body() -> html.Div:
    """Build the full dashboard body from a fresh data load (called on refresh)."""
    d = load()
    last_date = str(d.current_signal.get("as_of_date", ""))[:10]

    header = html.Div([
        html.Div([
            html.Div("TQQQ / SQQQ Trading Bot", className="header-title"),
            html.Div("Phase 4 — Paper Trading (simulation)", className="header-subtitle"),
        ]),
        html.Div([
            html.Div(f"Signal date: {last_date}", style={"marginBottom": "2px"}),
            html.Div(f"Auto-refresh every {REFRESH_SECONDS}s", style={"color": "#484f58"}),
        ], className="header-meta"),
    ], className="header-bar")

    # ── Live monitoring (REAL simulation account) ─────────────────────────────
    status_banner = paper_panel.build_status(d)
    kpi_row       = kpi_cards.build(d)          # now the real paper account

    monitor_row1 = html.Div([
        html.Div(paper_panel.build_positions(d), style={"flex": "2"}),
        html.Div(paper_panel.build_perf_stats(d), style={"flex": "1", "minWidth": "0"}),
        html.Div(paper_panel.build_health(d), style={"flex": "1.4", "minWidth": "0"}),
    ], style=_FLEX)

    monitor_row2 = html.Div([
        html.Div(paper_panel.build_trade_history(d), style={"flex": "1.3"}),
        html.Div(paper_panel.build_logs(d), style={"flex": "1.7", "minWidth": "0"}),
    ], style=_FLEX)

    # ── Strategy backtest & signals (clearly labelled as NOT the live account) ─
    section_label = html.Div("Strategy Backtest & Signals  ·  (16-yr model context — not the live account)",
                             className="section-label")

    row2 = html.Div([
        html.Div(equity_chart.build(d), style={"flex": "2"}),
        html.Div(signal_panel.build(d), style={"flex": "1", "minWidth": "0"}),
    ], style=_FLEX)
    row3 = analytics.build(d)
    row4 = html.Div([
        html.Div(historical.build(d), style={"flex": "2"}),
        html.Div([
            risk_panel.build_risk_stats(d),
            risk_panel.build_allocation(d),
        ], style={"flex": "1", "minWidth": "0"}),
        html.Div(risk_panel.build_regime_breakdown(d), style={"flex": "1", "minWidth": "0"}),
    ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"})

    return html.Div([
        header, status_banner, kpi_row,
        monitor_row1, monitor_row2,
        section_label, row2, row3, row4,
    ], className="dash-page")


from dashboard import admin


def _layout() -> html.Div:
    return html.Div([
        dcc.Interval(id="refresh", interval=REFRESH_SECONDS * 1000, n_intervals=0),
        dcc.Tabs(id="tabs", value="monitor", className="top-tabs", children=[
            dcc.Tab(label="📊  Monitor", value="monitor", className="top-tab",
                    selected_className="top-tab-sel"),
            dcc.Tab(label="⚙  Admin", value="admin", className="top-tab",
                    selected_className="top-tab-sel"),
        ]),
        html.Div(_dashboard_body(), id="page"),
    ])


app.layout = _layout


@app.callback(
    Output("page", "children"),
    Input("tabs", "value"),
    Input("refresh", "n_intervals"),
)
def _render_page(tab, _n):
    # Auto-refresh only re-renders the Monitor; the Admin page is static (forms).
    if tab == "admin":
        # don't rebuild admin on the 60s interval (would wipe in-progress edits)
        if ctx.triggered_id == "refresh":
            return no_update
        return admin.layout()
    return _dashboard_body()


admin.register_callbacks(app)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  TQQQ Bot Dashboard")
    print("  Warming data cache (first backtest run)…", flush=True)
    load()   # pre-compute so the first browser request is instant, not a cold backtest
    print("  Ready.  Open: http://127.0.0.1:8050\n", flush=True)
    app.run(debug=False, host="127.0.0.1", port=8050)
