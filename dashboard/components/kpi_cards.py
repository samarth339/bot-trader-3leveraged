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
    regime = d.current_signal.get("regime", "—")
    action = d.current_signal.get("action", "—")

    cards = [
        _card(
            "Portfolio Value",
            f"${d.portfolio_value:,.0f}",
            f"Seed $5,000  ·  shadow start",
            ""
        ),
        _card(
            "Total Return",
            _fmt_pct(d.total_return_pct),
            "Since shadow start (2026-03-27)",
            _pct_class(d.total_return_pct),
        ),
        _card(
            "Max Drawdown",
            _fmt_pct(d.max_dd_pct),
            "Shadow window peak-to-trough",
            "negative" if d.max_dd_pct < -0.01 else "",
        ),
        _card(
            "Sharpe (30d)",
            f"{d.sharpe_30:.2f}",
            "Annualised, ex-ante 5% RF",
            "positive" if d.sharpe_30 > 0 else "negative",
        ),
        _card(
            "Regime · Action",
            _regime_label(regime),
            action,
            _regime_class(regime),
        ),
    ]

    return html.Div(cards, className="kpi-row")


def _card(label: str, value: str, delta: str, value_class: str) -> html.Div:
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className=f"kpi-value {value_class}".strip()),
        html.Div(delta, className="kpi-delta"),
    ], className="kpi-card")
