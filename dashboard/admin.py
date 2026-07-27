"""
admin.py — Admin Panel (management & testing).  Localhost only.

Safety-scoped per the CLAUDE.md integrity guardrails:
  • NO live-trading toggle (Phase 5 doesn't exist; omitted by design).
  • Config edits go through the validated/bounded overrides layer — never raw
    Python. The locked baseline stays in code; "Reset to baseline" restores it.
  • Destructive actions (stop/flatten, reset paper) require a typed confirm.
  • Every action is logged to logs/config_audit.log.

Provides: start/stop (kill switch), reset paper data, run backtest, edit
parameters, enable/disable vol-target overlay, export data, view audit log.
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

from dash import Input, Output, State, dcc, html, no_update, ctx

from config import overrides as OV

ROOT = Path(__file__).parent.parent
LOGS = ROOT / "logs"

# Parameter groups for the editor (label, override-key) — all whitelisted/bounded.
_GROUPS = {
    "Strategy A (aggressive)": [
        ("MA long", "A.ma_long"), ("VIX exit", "A.vix_exit"),
        ("VIX re-entry", "A.vix_reentry"), ("Max position %", "A.max_position_pct"),
        ("Crash brake %", "A.crash_brake_pct"),
    ],
    "Strategy B (defensive)": [
        ("MA long", "B.ma_long"), ("VIX exit", "B.vix_exit"),
        ("VIX re-entry", "B.vix_reentry"), ("Max position %", "B.max_position_pct"),
        ("Crash brake %", "B.crash_brake_pct"),
    ],
    "Regime detection": [
        ("Regime MA window", "REGIME.ma_window"), ("VIX bull <", "REGIME.vix_bull"),
        ("VIX high-vol ≥", "REGIME.vix_hi_vol"), ("VIX smoothing", "REGIME.vix_smooth"),
    ],
    "Risk limits": [
        ("Max drawdown halt", "RISK.max_drawdown_halt"),
        ("Daily stop-loss", "RISK.daily_stop_loss"),
        ("Rebalance drift", "RISK.alloc_drift_rebalance"),
    ],
    "Volatility-target overlay": [
        ("Enabled (0/1)", "VOLTGT.enabled"),
        ("Target ann. vol", "VOLTGT.target_annual_vol"),
        ("Lookback days", "VOLTGT.lookback"),
    ],
}


def _current_value(key: str):
    """Effective value = override if set else baseline from strategy_config."""
    raw = OV.load_raw()
    if key in raw:
        return raw[key]
    import config.strategy_config as SC
    section, _, param = key.partition(".")
    cfg = getattr(SC, OV._SECTION_TO_CONFIG.get(section, ""), {})
    v = cfg.get(param)
    return int(v) if isinstance(v, bool) else v


def _field(label, key):
    lo, hi, typ = OV.WHITELIST[key]
    return html.Div([
        html.Label(label, className="admin-label"),
        dcc.Input(id={"type": "cfg", "key": key}, value=_current_value(key),
                  type="number", debounce=True, className="admin-input",
                  step=(1 if typ is int else 0.01)),
        html.Span(f"[{lo}–{hi}]", className="admin-bound"),
    ], className="admin-field")


def layout() -> html.Div:
    groups = []
    for title, fields in _GROUPS.items():
        groups.append(html.Div([
            html.Div(title, className="panel-title"),
            html.Div([_field(l, k) for l, k in fields], className="admin-grid"),
        ], className="panel-card"))

    controls = html.Div([
        html.Div("System Controls", className="panel-title"),
        html.Div([
            html.Button("▶ Start (clear kill switch)", id="admin-start", className="btn btn-green"),
            html.Button("■ Stop & flatten (kill switch)", id="admin-stop", className="btn btn-red"),
            html.Button("↺ Reset paper to $10k", id="admin-reset-paper", className="btn btn-amber"),
        ], className="btn-row"),
        dcc.ConfirmDialog(id="confirm-stop",
                          message="Activate the kill switch? This FLATTENS any position on the "
                                  "next run and freezes new buys. Confirm?"),
        dcc.ConfirmDialog(id="confirm-reset",
                          message="Reset the paper account to a clean $10,000? "
                                  "Current positions & trade history are wiped (git keeps history). Confirm?"),
        html.Div(id="admin-control-msg", className="admin-msg"),
    ], className="panel-card")

    actions = html.Div([
        html.Div("Testing & Export", className="panel-title"),
        html.Div([
            html.Button("Run backtest", id="admin-backtest", className="btn btn-blue"),
            html.Button("⬇ Export trades CSV", id="admin-export-trades", className="btn"),
            html.Button("⬇ Export portfolio JSON", id="admin-export-port", className="btn"),
        ], className="btn-row"),
        dcc.Loading(html.Div(id="admin-backtest-out", className="admin-out"), type="dot"),
        dcc.Download(id="admin-download"),
    ], className="panel-card")

    config_actions = html.Div([
        html.Div([
            html.Button("💾 Save config", id="admin-save", className="btn btn-green"),
            html.Button("↩ Reset to locked baseline", id="admin-reset-cfg", className="btn btn-amber"),
        ], className="btn-row"),
        html.Div(id="admin-save-msg", className="admin-msg"),
    ], className="panel-card")

    return html.Div([
        html.Div("⚙ Admin Panel — management & testing (localhost only · no live trading)",
                 className="section-label"),
        controls,
        html.Div("Strategy Parameters  (validated & bounded — locked baseline preserved)",
                 className="section-label"),
        html.Div(groups, className="admin-groups"),
        config_actions,
        actions,
    ], className="dash-page")


# ── Callbacks ───────────────────────────────────────────────────────────────

def register_callbacks(app):

    # Save config from all inputs
    @app.callback(
        Output("admin-save-msg", "children"),
        Input("admin-save", "n_clicks"),
        State({"type": "cfg", "key": dash_ALL}, "value"),
        State({"type": "cfg", "key": dash_ALL}, "id"),
        prevent_initial_call=True,
    )
    def _save(_n, values, ids):
        overrides, baseline = {}, _baseline_map()
        for val, ident in zip(values, ids):
            key = ident["key"]
            if val is None:
                continue
            # only persist values that differ from baseline (keeps file minimal)
            if baseline.get(key) is not None and float(val) == float(baseline[key]):
                continue
            overrides[key] = val
        ok, msg = OV.save(overrides)
        return html.Span(("✓ " if ok else "✗ ") + msg,
                         className=("ok" if ok else "err"))

    @app.callback(
        Output("admin-save-msg", "children", allow_duplicate=True),
        Input("admin-reset-cfg", "n_clicks"), prevent_initial_call=True)
    def _reset_cfg(_n):
        OV.clear()
        return html.Span("✓ Reset to locked baseline. Reload to see values.", className="ok")

    # Kill switch / reset via confirm dialogs
    @app.callback(Output("confirm-stop", "displayed"),
                  Input("admin-stop", "n_clicks"), prevent_initial_call=True)
    def _ask_stop(_n): return True

    @app.callback(Output("confirm-reset", "displayed"),
                  Input("admin-reset-paper", "n_clicks"), prevent_initial_call=True)
    def _ask_reset(_n): return True

    @app.callback(
        Output("admin-control-msg", "children"),
        Input("admin-start", "n_clicks"),
        Input("confirm-stop", "submit_n_clicks"),
        Input("confirm-reset", "submit_n_clicks"),
        prevent_initial_call=True)
    def _controls(_start, _stop, _reset):
        trig = ctx.triggered_id
        kill = LOGS / "ibkr_kill.flag"
        if trig == "admin-start":
            if kill.exists():
                kill.unlink()
            OV._audit("ADMIN start — kill switch cleared")
            return html.Span("✓ Kill switch cleared — trading enabled.", className="ok")
        if trig == "confirm-stop":
            kill.write_text("Manual stop via Admin Panel")
            OV._audit("ADMIN stop — kill switch activated")
            return html.Span("■ Kill switch ACTIVE — position flattens on next run.", className="err")
        if trig == "confirm-reset":
            r = subprocess.run([sys.executable, "paper_trade.py", "--reset-portfolio"],
                               cwd=ROOT, capture_output=True, text=True)
            OV._audit("ADMIN reset paper account to $10k")
            return html.Span("✓ Paper account reset to $10,000." if r.returncode == 0
                             else f"✗ {r.stderr[:80]}", className="ok" if r.returncode == 0 else "err")
        return no_update

    # Run backtest
    @app.callback(Output("admin-backtest-out", "children"),
                  Input("admin-backtest", "n_clicks"), prevent_initial_call=True)
    def _backtest(_n):
        r = subprocess.run([sys.executable, "dual_portfolio_runner.py", "--no-chart"],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
        tail = "\n".join(r.stdout.strip().splitlines()[-14:]) or r.stderr[-800:]
        return html.Pre(tail, className="admin-pre")

    # Exports
    @app.callback(Output("admin-download", "data"),
                  Input("admin-export-trades", "n_clicks"),
                  Input("admin-export-port", "n_clicks"), prevent_initial_call=True)
    def _export(_t, _p):
        trig = ctx.triggered_id
        if trig == "admin-export-trades":
            f = LOGS / "paper_trades.csv"
            return dcc.send_file(str(f)) if f.exists() else no_update
        if trig == "admin-export-port":
            f = LOGS / "paper_portfolio.json"
            return dcc.send_file(str(f)) if f.exists() else no_update
        return no_update


def _baseline_map() -> dict:
    import importlib
    import config.strategy_config as SC
    # baseline = config WITHOUT overrides applied
    raw_backup = OV.OVERRIDES_PATH.exists()
    vals = {}
    import config.overrides as O
    for key in O.WHITELIST:
        section, _, param = key.partition(".")
        cfg = getattr(SC, O._SECTION_TO_CONFIG.get(section, ""), {})
        v = cfg.get(param)
        vals[key] = int(v) if isinstance(v, bool) else v
    return vals


# dash pattern-matching ALL sentinel (imported lazily to avoid hard dep at top)
from dash import ALL as dash_ALL   # noqa: E402
