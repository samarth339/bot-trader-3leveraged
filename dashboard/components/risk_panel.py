"""risk_panel.py — VaR, realised vol, allocation donut, regime breakdown."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

from ..data_loader import DashboardData

_BG    = "#161b22"
_TEXT  = "#8b949e"
_GRID  = "#21262d"
_GREEN = "#3fb950"
_RED   = "#f85149"
_AMBER = "#d29922"
_BLUE  = "#388bfd"


# ── VaR + Vol stats ────────────────────────────────────────────────────────────

def _var_vol_block(d: DashboardData) -> html.Div:
    rows = [
        ("1-day VaR 95%",  f"−${d.var_95:,.0f}",  "negative"),
        ("1-day VaR 99%",  f"−${d.var_99:,.0f}",  "negative"),
        ("Realised Vol 14d", f"{d.realized_vol_14*100:.1f}%", ""),
        ("Realised Vol 30d", f"{d.realized_vol_30*100:.1f}%", ""),
        ("Max Drawdown",   f"{d.max_dd_pct*100:.1f}%", "negative" if d.max_dd_pct < -0.01 else ""),
        ("Current Alloc",  f"{d.current_alloc*100:.0f}% TQQQ", ""),
    ]

    items = [
        html.Div([
            html.Div(label, className="stat-label"),
            html.Div(val,   className=f"stat-value {cls}".strip()),
        ], className="stat-item")
        for label, val, cls in rows
    ]

    return html.Div(items, className="stat-grid")


# ── Allocation donut ───────────────────────────────────────────────────────────

def _alloc_donut(d: DashboardData) -> dcc.Graph:
    alloc_tqqq  = d.current_alloc * 100
    alloc_cash  = max(0.0, 100.0 - alloc_tqqq)

    labels = ["TQQQ", "Cash"]
    values = [alloc_tqqq, alloc_cash]
    colors = [_GREEN, _GRID]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color=_BG, width=2)),
        textinfo="none",
        hovertemplate="%{label}: %{value:.0f}%<extra></extra>",
    ))

    fig.add_annotation(
        text=f"<b>{alloc_tqqq:.0f}%</b><br>TQQQ",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color="#e6edf3", size=14),
        align="center",
    )

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        showlegend=True,
        legend=dict(
            orientation="h", x=0.1, y=-0.05,
            font=dict(color=_TEXT, size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=0, b=30),
    )

    return dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "160px"})


# ── Regime breakdown bar ───────────────────────────────────────────────────────

def _regime_bar(d: DashboardData) -> html.Div:
    bull  = d.regime_days.get("bull", 0)
    unc   = d.regime_days.get("uncertain", 0)
    hv    = d.regime_days.get("high_vol", 0)
    total = bull + unc + hv or 1

    def pct(n):
        return f"{n/total*100:.0f}%"

    segments = [
        html.Div(style={"flex": f"{bull/total}", "minWidth": "2px"}, className="regime-bar-bull"),
        html.Div(style={"flex": f"{unc/total}",  "minWidth": "2px"}, className="regime-bar-uncertain"),
        html.Div(style={"flex": f"{hv/total}",   "minWidth": "2px"}, className="regime-bar-high-vol"),
    ]

    legend = html.Div([
        html.Div([
            html.Div(className="regime-dot", style={"background": _GREEN}),
            html.Span(f"Bull {bull}d ({pct(bull)})", className=""),
        ], className="regime-legend-item"),
        html.Div([
            html.Div(className="regime-dot", style={"background": _AMBER}),
            html.Span(f"Uncertain {unc}d ({pct(unc)})", className=""),
        ], className="regime-legend-item"),
        html.Div([
            html.Div(className="regime-dot", style={"background": _RED}),
            html.Span(f"High-vol {hv}d ({pct(hv)})", className=""),
        ], className="regime-legend-item"),
    ], className="regime-legend")

    return html.Div([
        html.Div(f"Regime distribution  ({total} days shadow)", className="stat-label",
                 style={"marginBottom": "6px"}),
        html.Div(segments, className="regime-bar"),
        legend,
    ], className="regime-bar-wrap")


# ── Public build ───────────────────────────────────────────────────────────────

def build_risk_stats(d: DashboardData) -> html.Div:
    return html.Div([
        html.Div("Risk Metrics", className="panel-title"),
        _var_vol_block(d),
    ], className="panel-card")


def build_allocation(d: DashboardData) -> html.Div:
    return html.Div([
        html.Div("Current Allocation", className="panel-title"),
        _alloc_donut(d),
    ], className="panel-card")


def build_regime_breakdown(d: DashboardData) -> html.Div:
    return html.Div([
        html.Div("Regime Breakdown", className="panel-title"),
        _regime_bar(d),
    ], className="panel-card")


def build(d: DashboardData) -> html.Div:
    return html.Div([
        build_risk_stats(d),
        build_allocation(d),
        build_regime_breakdown(d),
    ])
