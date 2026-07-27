"""
paper_panel.py — REAL paper-SIMULATION account panels.
Status header, P/L cards, open position, trade history, system health, logs.
All data from d.paper / d.system_health / d.recent_logs (the correct files).
"""
from __future__ import annotations

from dash import html
from ..data_loader import DashboardData

_GREEN = "#3fb950"
_RED   = "#f85149"
_AMBER = "#d29922"
_MUTE  = "#8b949e"

_STATUS_COLOR = {"RUNNING": _GREEN, "PAUSED": _AMBER, "HALTED": _RED}


def _money(v, sign=False):
    s = "+" if (sign and v > 0) else ""
    return f"{s}${v:,.2f}" if abs(v) < 100000 else f"{s}${v:,.0f}"


def _pct(v, d=2):
    if v is None:
        return "—"
    return f"{'+' if v > 0 else ''}{v*100:.{d}f}%"


def _cls(v):
    return "positive" if (v or 0) > 0 else ("negative" if (v or 0) < 0 else "")


# ── Status banner ───────────────────────────────────────────────────────────────

def build_status(d: DashboardData) -> html.Div:
    p = d.paper
    status = p.get("status", "UNKNOWN")
    color = _STATUS_COLOR.get(status, _MUTE)
    return html.Div([
        html.Div([
            html.Span(className="status-dot", style={"background": color}),
            html.Span(status, className="status-text", style={"color": color}),
            html.Span(p.get("status_reason", ""), className="status-reason"),
        ], className="status-left"),
        html.Div([
            html.Span("PAPER SIMULATION", className="sim-badge"),
            html.Span("not connected to IBKR", className="status-reason"),
        ], className="status-right"),
    ], className="status-banner")


# ── Performance (D/W/M) + trade stats ───────────────────────────────────────────

def build_perf_stats(d: DashboardData) -> html.Div:
    p = d.paper
    ndays = p.get("ret_n_days", 0)
    win = p.get("win_rate")
    pf = p.get("profit_factor")
    rows = [
        ("Daily", _pct(p.get("ret_daily")), _cls(p.get("ret_daily"))),
        ("Weekly", _pct(p.get("ret_weekly")), _cls(p.get("ret_weekly"))),
        ("Monthly", _pct(p.get("ret_monthly")), _cls(p.get("ret_monthly"))),
        ("Trades YTD", str(p.get("trades_ytd", 0)), ""),
        ("Win rate", (f"{win*100:.0f}%" if win is not None else "n/a"), ""),
        ("Profit factor", (f"{pf:.2f}" if pf not in (None, float('inf')) else ("∞" if pf else "n/a")), ""),
        ("Avg win", _money(p.get("avg_win", 0), sign=True), _cls(p.get("avg_win"))),
        ("Avg loss", _money(p.get("avg_loss", 0), sign=True), _cls(p.get("avg_loss"))),
    ]
    note = html.Div(
        f"⚠ win/profit metrics from {p.get('n_closes',0)} closed trades — not yet statistically meaningful",
        className="stat-note") if (p.get("n_closes", 0) < 20) else None
    items = [html.Div([html.Div(l, className="stat-label"),
                       html.Div(v, className=f"stat-value {c}".strip())], className="stat-item")
             for l, v, c in rows]
    body = [html.Div("Performance & Trade Stats", className="panel-title"),
            html.Div(items, className="stat-grid")]
    if note:
        body.append(note)
    return html.Div(body, className="panel-card")


# ── Open position ───────────────────────────────────────────────────────────────

def build_positions(d: DashboardData) -> html.Div:
    p = d.paper
    shares = p.get("shares", 0)
    if shares:
        rows = [html.Tr([
            html.Td("TQQQ"), html.Td(f"{shares}"),
            html.Td(f"${p.get('avg_cost',0):.2f}"), html.Td(f"${p.get('cur_price',0):.2f}"),
            html.Td(_money(p.get("unrealized_pl", 0), sign=True),
                    className=_cls(p.get("unrealized_pl"))),
            html.Td(f"{p.get('invested_pct',0)*100:.0f}%"),
        ])]
    else:
        rows = [html.Tr([html.Td("— flat (all cash) —", colSpan=6,
                                 style={"textAlign": "center", "color": _MUTE})])]
    table = html.Table([
        html.Thead(html.Tr([html.Th(h) for h in
                            ["Ticker", "Shares", "Avg cost", "Last", "Unreal. P/L", "% NLV"]])),
        html.Tbody(rows),
    ], className="data-table")
    return html.Div([html.Div("Open Position", className="panel-title"), table],
                    className="panel-card")


# ── Trade history ───────────────────────────────────────────────────────────────

def build_trade_history(d: DashboardData) -> html.Div:
    trades = d.paper.get("trades")
    body_rows = []
    if trades is not None and not trades.empty:
        for _, r in trades.tail(12).iloc[::-1].iterrows():
            delta = r.get("delta_shares", 0)
            act = ("BUY" if delta > 0 else "SELL" if delta < 0 else "—")
            act_c = _GREEN if delta > 0 else (_RED if delta < 0 else _MUTE)
            fill = r.get("fill_price", "")
            fill_s = f"${float(fill):.2f}" if str(fill) not in ("", "nan") else "—"
            body_rows.append(html.Tr([
                html.Td(str(r.get("date"))[:10]),
                html.Td(str(r.get("regime", ""))),
                html.Td(act, style={"color": act_c, "fontWeight": "600"}),
                html.Td(f"{int(delta):+d}" if delta else "0"),
                html.Td(fill_s),
                html.Td(str(r.get("status", ""))),
                html.Td(f"${float(r.get('nlv_after', 0)):,.0f}"),
            ]))
    else:
        body_rows = [html.Tr([html.Td("no trades yet", colSpan=7,
                              style={"textAlign": "center", "color": _MUTE})])]
    table = html.Table([
        html.Thead(html.Tr([html.Th(h) for h in
                            ["Date", "Regime", "Action", "Δ sh", "Fill", "Status", "NLV"]])),
        html.Tbody(body_rows),
    ], className="data-table")
    return html.Div([html.Div("Trade History (recent)", className="panel-title"), table],
                    className="panel-card")


# ── System health ───────────────────────────────────────────────────────────────

def build_health(d: DashboardData) -> html.Div:
    h = d.system_health or {}
    rows = []
    for name, state, detail, ok in h.get("checks", []):
        color = _GREEN if ok else (_AMBER if state in ("STALE", "FALLBACK", "SIMULATION") else _RED)
        rows.append(html.Div([
            html.Span(className="health-dot", style={"background": color}),
            html.Span(name, className="health-name"),
            html.Span(state, className="health-state", style={"color": color}),
            html.Span(detail, className="health-detail"),
        ], className="health-row"))
    return html.Div([html.Div("System Health", className="panel-title"),
                     html.Div(rows, className="health-list")], className="panel-card")


# ── Recent logs ─────────────────────────────────────────────────────────────────

def build_logs(d: DashboardData) -> html.Div:
    lc = {"ERROR": _RED, "WARNING": _AMBER, "INFO": _MUTE}
    lines = [html.Div([
        html.Span(r["level"], className="log-level", style={"color": lc.get(r["level"], _MUTE)}),
        html.Span(r["text"], className="log-text"),
    ], className="log-line") for r in (d.recent_logs or [])[-18:]]
    if not lines:
        lines = [html.Div("no recent logs", className="log-line", style={"color": _MUTE})]
    return html.Div([html.Div("Recent Logs, Errors & Warnings", className="panel-title"),
                     html.Div(lines, className="log-box")], className="panel-card")
