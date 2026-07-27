"""kpi_cards.py — Top-row KPI strip (5 cards)."""

from dash import html
from ..data_loader import DashboardData


def _fmt_pct(v: float, decimals: int = 1) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.{decimals}f}%"


def _pct_class(v: float) -> str:
    if v > 0:
        return "positive"
    if v < 0:
        return "negative"
    return ""


def _regime_class(regime: str) -> str:
    mapping = {"bull": "regime-bull", "uncertain": "regime-uncertain", "high_vol": "regime-high-vol"}
    return mapping.get(regime, "")


def _regime_label(regime: str) -> str:
    mapping = {"bull": "BULL", "uncertain": "UNCERTAIN", "high_vol": "HIGH VOL"}
    return mapping.get(regime, regime.upper())


def build(d: DashboardData) -> html.Div:
    """Headline KPI strip — the REAL paper-SIMULATION account (paper_portfolio.json).
    Fixes the prior bug where this read stale ibkr_state.json at the $5K seed."""
    p = d.paper or {}
    seed  = p.get("seed", 10_000.0)
    nlv   = p.get("nlv", seed)
    tot   = p.get("total_return", 0.0)
    tpl   = p.get("total_pl", 0.0)
    upl   = p.get("unrealized_pl", 0.0)
    rpl   = p.get("realized_pl", 0.0)
    maxdd = p.get("max_dd", 0.0)

    cards = [
        _card("Account NLV", f"${nlv:,.2f}", f"seed ${seed:,.0f} · sim", ""),
        _card("Total P/L", f"{'+' if tpl>0 else ''}${tpl:,.2f}", _fmt_pct(tot), _pct_class(tpl)),
        _card("Unrealized P/L", f"{'+' if upl>0 else ''}${upl:,.2f}",
              f"{p.get('shares',0)} sh @ ${p.get('avg_cost',0):.2f}", _pct_class(upl)),
        _card("Realized P/L", f"{'+' if rpl>0 else ''}${rpl:,.2f}",
              f"{p.get('n_closes',0)} closed trades", _pct_class(rpl)),
        _card("Max Drawdown", _fmt_pct(-maxdd),
              "peak-to-trough (sim)", "negative" if maxdd > 0.01 else ""),
    ]
    return html.Div(cards, className="kpi-row")


def _card(label: str, value: str, delta: str, value_class: str) -> html.Div:
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className=f"kpi-value {value_class}".strip()),
        html.Div(delta, className="kpi-delta"),
    ], className="kpi-card")
