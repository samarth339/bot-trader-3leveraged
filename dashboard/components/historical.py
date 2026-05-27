"""historical.py — MTD / YTD chips and VIX + price context chart."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html

from ..data_loader import DashboardData

_BG    = "#161b22"
_GRID  = "#21262d"
_TEXT  = "#8b949e"
_BLUE  = "#388bfd"
_GREEN = "#3fb950"
_RED   = "#f85149"
_AMBER = "#d29922"


def _pct_class(v: float) -> str:
    if v > 0:
        return "positive"
    if v < 0:
        return "negative"
    return ""


def _fmt_pct(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.1f}%"


def build_period_chips(d: DashboardData) -> html.Div:
    chips = [
        ("MTD",   d.mtd_return),
        ("YTD",   d.ytd_return),
        ("Total", d.total_return_pct),
    ]
    els = []
    for label, val in chips:
        cls = _pct_class(val)
        els.append(html.Div([
            html.Div(label, className="period-chip-label"),
            html.Div(_fmt_pct(val), className=f"period-chip-value {cls}".strip()),
        ], className="period-chip"))
    return html.Div(els, className="period-chips")


def build_vix_chart(d: DashboardData) -> dcc.Graph:
    """VIX 1-year with bull/high-vol threshold lines, and QQQ overlay."""
    if d.vix.empty:
        return dcc.Graph(figure=go.Figure(), style={"height": "200px"})

    vix_1y  = d.vix.iloc[-252:]
    qqq_1y  = d.qqq["close"].iloc[-252:] if not d.qqq.empty else None

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # VIX
    fig.add_trace(go.Scatter(
        x=vix_1y.index,
        y=vix_1y.values,
        mode="lines",
        name="VIX",
        line=dict(color=_RED, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}  VIX %{y:.1f}<extra></extra>",
    ), secondary_y=False)

    # Threshold lines
    fig.add_hline(y=18.0, line=dict(color=_GREEN, width=1, dash="dot"),
                  annotation_text="Bull<18", annotation_font=dict(color=_GREEN, size=9),
                  secondary_y=False)
    fig.add_hline(y=25.0, line=dict(color=_AMBER, width=1, dash="dot"),
                  annotation_text="HighVol≥25", annotation_font=dict(color=_AMBER, size=9),
                  secondary_y=False)

    # QQQ on secondary axis
    if qqq_1y is not None and not qqq_1y.empty:
        fig.add_trace(go.Scatter(
            x=qqq_1y.index,
            y=qqq_1y.values,
            mode="lines",
            name="QQQ",
            line=dict(color=_BLUE, width=1.2),
            hovertemplate="%{x|%Y-%m-%d}  QQQ $%{y:,.1f}<extra></extra>",
        ), secondary_y=True)

    fig.update_layout(
        title=dict(text="VIX & QQQ  (1 year)", font=dict(size=12, color=_TEXT), x=0.01),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        margin=dict(l=50, r=50, t=32, b=36),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1c2230", font=dict(color="#e6edf3", size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT, size=10),
                    orientation="h", y=1.04, x=0),
    )
    fig.update_yaxes(gridcolor=_GRID, tickfont=dict(color=_TEXT, size=10), secondary_y=False)
    fig.update_yaxes(gridcolor=_GRID, tickfont=dict(color=_TEXT, size=10),
                     tickprefix="$", secondary_y=True)
    fig.update_xaxes(gridcolor=_GRID, tickfont=dict(color=_TEXT, size=10))

    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": False},
        style={"height": "220px"},
    )


def build(d: DashboardData) -> html.Div:
    return html.Div([
        html.Div("Performance", className="panel-title"),
        build_period_chips(d),
        html.Hr(className="section-divider"),
        html.Div("Market Context", className="panel-title"),
        build_vix_chart(d),
    ], className="panel-card")
