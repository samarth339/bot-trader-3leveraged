"""analytics.py — Rolling Sharpe and daily returns distribution."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from ..data_loader import DashboardData

_BG    = "#161b22"
_GRID  = "#21262d"
_TEXT  = "#8b949e"
_BLUE  = "#388bfd"
_GREEN = "#3fb950"
_RED   = "#f85149"
_AMBER = "#d29922"


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=12, color=_TEXT), x=0.01, xanchor="left"),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        margin=dict(l=50, r=16, t=32, b=36),
        xaxis=dict(gridcolor=_GRID, tickfont=dict(color=_TEXT, size=10)),
        yaxis=dict(gridcolor=_GRID, tickfont=dict(color=_TEXT, size=10)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1c2230", font=dict(color="#e6edf3", size=11)),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT, size=10),
                    orientation="h", y=1.04, x=0),
    )


def build_rolling_sharpe(d: DashboardData) -> dcc.Graph:
    fig = go.Figure()

    # Zero line
    if not d.rolling_sharpe_7.empty:
        fig.add_hline(y=0, line=dict(color=_TEXT, width=1, dash="dot"))

    if not d.rolling_sharpe_7.dropna().empty:
        fig.add_trace(go.Scatter(
            x=d.rolling_sharpe_7.index,
            y=d.rolling_sharpe_7.values,
            mode="lines",
            name="7d Sharpe",
            line=dict(color=_AMBER, width=1.2),
            hovertemplate="%{x|%Y-%m-%d}  %{y:.2f}<extra></extra>",
        ))

    if not d.rolling_sharpe_30.dropna().empty:
        fig.add_trace(go.Scatter(
            x=d.rolling_sharpe_30.index,
            y=d.rolling_sharpe_30.values,
            mode="lines",
            name="30d Sharpe",
            line=dict(color=_BLUE, width=1.8),
            hovertemplate="%{x|%Y-%m-%d}  %{y:.2f}<extra></extra>",
        ))

    layout = _base_layout("Rolling Sharpe (annualised)")
    layout["yaxis"]["tickformat"] = ".1f"
    fig.update_layout(**layout)

    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": False},
        style={"height": "220px"},
    )


def build_returns_dist(d: DashboardData) -> dcc.Graph:
    ret = d.daily_returns.dropna() * 100  # convert to %

    if ret.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout("Daily Returns Distribution"))
        return dcc.Graph(figure=fig, style={"height": "220px"})

    # Colour each bar green/red
    hist_vals, bin_edges = np.histogram(ret, bins=60)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    colors = [_GREEN if x >= 0 else _RED for x in bin_mids]

    fig = go.Figure(go.Bar(
        x=bin_mids,
        y=hist_vals,
        marker_color=colors,
        hovertemplate="%{x:.1f}%  ×%{y}<extra></extra>",
        showlegend=False,
    ))

    # Vertical lines at 0 and mean
    mean_ret = float(ret.mean())
    fig.add_vline(x=0,       line=dict(color=_TEXT,  width=1, dash="dot"))
    fig.add_vline(x=mean_ret, line=dict(color=_AMBER, width=1, dash="dash"),
                  annotation_text=f"μ {mean_ret:+.2f}%",
                  annotation_font=dict(color=_AMBER, size=10))

    layout = _base_layout("Daily Returns  (%)")
    layout["yaxis"].pop("tickformat", None)
    layout["showlegend"] = False
    layout["xaxis"]["ticksuffix"] = "%"
    fig.update_layout(**layout)

    return dcc.Graph(
        figure=fig,
        config={"displayModeBar": False},
        style={"height": "220px"},
    )


def build(d: DashboardData) -> html.Div:
    return html.Div([
        html.Div([
            html.Div([
                html.Div("Rolling Sharpe", className="panel-title"),
                build_rolling_sharpe(d),
            ], className="panel-card", style={"flex": "1"}),
            html.Div([
                html.Div("Return Distribution", className="panel-title"),
                build_returns_dist(d),
            ], className="panel-card", style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px"}),
    ])
