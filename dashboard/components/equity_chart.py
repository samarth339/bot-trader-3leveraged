"""equity_chart.py — Main equity curve with regime shading and shadow overlay."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

from ..data_loader import DashboardData

_BG        = "#161b22"
_GRID      = "#21262d"
_TEXT      = "#8b949e"
_BULL      = "#3fb950"
_UNCERTAIN = "#d29922"
_HIGH_VOL  = "#f85149"
_BLUE      = "#388bfd"
_PURPLE    = "#bc8cff"

_REGIME_COLORS = {
    "bull":      (_BULL,      "rgba(63,185,80,0.08)"),
    "uncertain": (_UNCERTAIN, "rgba(210,153,34,0.08)"),
    "high_vol":  (_HIGH_VOL,  "rgba(248,81,73,0.08)"),
}


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13, color=_TEXT), x=0.01, xanchor="left"),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        margin=dict(l=50, r=20, t=36, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=_TEXT, size=11),
            orientation="h",
            y=1.06,
            x=0,
        ),
        xaxis=dict(
            gridcolor=_GRID, linecolor=_GRID,
            tickfont=dict(color=_TEXT, size=10),
            showspikes=True,
            spikecolor=_TEXT, spikethickness=1, spikedash="dot",
        ),
        yaxis=dict(
            gridcolor=_GRID, linecolor=_GRID,
            tickfont=dict(color=_TEXT, size=10),
            tickprefix="$",
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1c2230", font=dict(color="#e6edf3", size=11)),
    )


def _add_regime_shading(fig: go.Figure, regime_blocks: list, y_ref: str = "y") -> None:
    for start, end, regime in regime_blocks:
        _, fill = _REGIME_COLORS.get(regime, (_TEXT, "rgba(255,255,255,0.04)"))
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=fill,
            layer="below",
            line_width=0,
        )


def build_equity_chart(d: DashboardData) -> dcc.Graph:
    fig = go.Figure()

    # Regime shading
    _add_regime_shading(fig, d.regime_blocks)

    # Full backtest
    fig.add_trace(go.Scatter(
        x=d.backtest_equity.index,
        y=d.backtest_equity.values,
        mode="lines",
        name="Backtest (2010–)",
        line=dict(color=_BLUE, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}  $%{y:,.0f}<extra></extra>",
    ))

    # Shadow window highlight
    if not d.shadow_equity.empty:
        fig.add_trace(go.Scatter(
            x=d.shadow_equity.index,
            y=d.shadow_equity.values,
            mode="lines",
            name="Shadow (live)",
            line=dict(color=_PURPLE, width=2.5),
            hovertemplate="%{x|%Y-%m-%d}  $%{y:,.0f}<extra></extra>",
        ))

    # Paper overlay
    if d.paper_equity is not None and not d.paper_equity.empty:
        fig.add_trace(go.Scatter(
            x=d.paper_equity.index,
            y=d.paper_equity.values,
            mode="lines",
            name="Paper fills",
            line=dict(color=_BULL, width=2, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}  $%{y:,.0f}<extra></extra>",
        ))

    fig.update_layout(**_base_layout("Equity Curve  (normalised to $5,000)"))
    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": True, "displaylogo": False,
                "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
        style={"height": "360px"},
    )


def build_drawdown_chart(d: DashboardData) -> dcc.Graph:
    fig = go.Figure()

    _add_regime_shading(fig, d.regime_blocks)

    fig.add_trace(go.Scatter(
        x=d.drawdown.index,
        y=(d.drawdown * 100).values,
        mode="lines",
        name="Drawdown",
        line=dict(color=_HIGH_VOL, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(248,81,73,0.12)",
        hovertemplate="%{x|%Y-%m-%d}  %{y:.1f}%<extra></extra>",
    ))

    layout = _base_layout("Drawdown  (%  from peak)")
    layout["yaxis"]["ticksuffix"] = "%"
    layout["yaxis"].pop("tickprefix", None)
    fig.update_layout(**layout)

    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": False},
        style={"height": "200px"},
    )


def build(d: DashboardData) -> html.Div:
    return html.Div([
        html.Div("Equity Curve", className="panel-title"),
        build_equity_chart(d),
        html.Div(style={"height": "10px"}),
        build_drawdown_chart(d),
    ], className="equity-chart-wrap")
