"""signal_panel.py — Current signal block + scrollable signal history table."""

from __future__ import annotations

import pandas as pd
from dash import html

from ..data_loader import DashboardData

_REGIME_LABELS = {"bull": "BULL", "uncertain": "UNCERTAIN", "high_vol": "HIGH VOL"}


def _regime_badge(regime: str) -> html.Div:
    cls = regime.replace("_", "-")
    label = _REGIME_LABELS.get(regime, regime.upper())
    return html.Div([
        html.Div(label, style={"fontSize": "13px"}),
        html.Div("regime", style={"fontSize": "9px", "marginTop": "3px", "opacity": "0.7"}),
    ], className=f"signal-regime-badge {cls}")


def _signal_row(key: str, val: str) -> html.Div:
    return html.Div([
        html.Span(key, className="signal-key"),
        html.Span(val, className="signal-val"),
    ], className="signal-row")


def _current_signal_block(d: DashboardData) -> html.Div:
    s = d.current_signal
    if not s:
        return html.Div("No signal data.", style={"color": "#8b949e", "fontSize": "12px"})

    regime  = str(s.get("regime", "—"))
    action  = str(s.get("action", "—"))
    wa      = float(s.get("weight_a", 0))
    wb      = float(s.get("weight_b", 0))
    qqq     = s.get("qqq_price", "—")
    sma     = s.get("sma_val", "—")
    pct_sma = s.get("pct_vs_sma", 0)
    vix_sig = s.get("vix_signal", "—")
    vix_raw = s.get("vix_raw", "—")
    date    = str(s.get("as_of_date", s.get("signal_date", "—")))[:10]

    pct_sma_str = f"{float(pct_sma)*100:+.2f}%" if pct_sma not in ("—", None) else "—"
    qqq_str = f"${float(qqq):,.2f}" if qqq not in ("—", None) else "—"
    sma_str = f"${float(sma):,.2f}" if sma not in ("—", None) else "—"
    vix_str = f"{float(vix_raw):.2f}" if vix_raw not in ("—", None) else "—"
    alloc_str = f"{(wa*0.85 + wb*0.60)*100:.0f}% TQQQ"

    return html.Div([
        _regime_badge(regime),
        html.Div([
            _signal_row("Date",         date),
            _signal_row("Action",       action.upper()),
            _signal_row("Allocation",   alloc_str),
            _signal_row("A / B weight", f"{wa:.0%} / {wb:.0%}"),
            _signal_row("QQQ",          f"{qqq_str}  (SMA: {sma_str}  {pct_sma_str})"),
            _signal_row("VIX 5d avg",   f"{vix_str}  ({vix_sig})"),
        ], className="signal-details"),
    ], className="current-signal-block")


def _signal_table(signals: pd.DataFrame) -> html.Div:
    if signals.empty:
        return html.Div("No signal history.", style={"color": "#8b949e", "fontSize": "12px"})

    rows_df = signals.sort_values("as_of_date", ascending=False).head(60)

    header = html.Tr([
        html.Th(c) for c in ["Date", "Regime", "Action", "A%", "B%", "QQQ", "VIX", "5d Fwd", "✓"]
    ])

    rows = []
    for _, r in rows_df.iterrows():
        regime  = str(r.get("regime", ""))
        outcome = str(r.get("outcome", "—"))
        fwd     = r.get("tqqq_5d_fwd", float("nan"))
        regime_badge = html.Span(
            _REGIME_LABELS.get(regime, regime.upper()),
            className=f"badge badge-{regime.replace('_','-')}",
        )
        outcome_cls = (
            "outcome-win"  if outcome == "✓" else
            "outcome-loss" if outcome == "✗" else
            "outcome-na"
        )
        fwd_str = f"{fwd*100:+.1f}%" if pd.notna(fwd) else "—"
        qqq_p   = r.get("qqq_price", "")
        qqq_str = f"${float(qqq_p):,.1f}" if qqq_p not in ("", None) else "—"
        vix_r   = r.get("vix_raw", "")
        vix_str = f"{float(vix_r):.1f}" if vix_r not in ("", None) else "—"

        date_str = str(r.get("as_of_date", ""))[:10]

        rows.append(html.Tr([
            html.Td(date_str),
            html.Td(regime_badge),
            html.Td(str(r.get("action", "—")).upper()),
            html.Td(f"{float(r.get('weight_a',0)):.0%}"),
            html.Td(f"{float(r.get('weight_b',0)):.0%}"),
            html.Td(qqq_str),
            html.Td(vix_str),
            html.Td(fwd_str),
            html.Td(outcome, className=outcome_cls),
        ]))

    return html.Div(
        html.Table([html.Thead(header), html.Tbody(rows)], className="signal-table"),
        className="signal-table-wrap signal-log-scroll",
    )


def build(d: DashboardData) -> html.Div:
    return html.Div([
        html.Div("Current Signal", className="panel-title"),
        _current_signal_block(d),
        html.Hr(className="section-divider"),
        html.Div("Signal History (last 60 days)", className="panel-title"),
        _signal_table(d.signals),
    ], className="panel-card")
