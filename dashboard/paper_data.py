"""
paper_data.py — Real paper-SIMULATION account state for the dashboard.

Reads the files the live simulation actually writes:
    logs/paper_portfolio.json   — current NLV, shares, cash, peak, inception
    logs/paper_trades.csv       — the executed/no-action ledger
    logs/signal_history.csv      — latest signal (regime, exposures, vol_scalar)

NOT logs/ibkr_state.json / logs/ibkr_orders.csv — those are stale Phase-5
IBKR-executor files and do NOT reflect the running simulation.

Everything here is labelled SIMULATION. The bot is not connected to IBKR.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

STALE_DAYS = 4   # a signal/trade older than this ⇒ pipeline considered stale


# ── low-level loads ────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def _load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ── average-cost P/L accounting ────────────────────────────────────────────────

def _walk_pl(trades: pd.DataFrame, cur_price: float, seed: float) -> dict:
    """
    Average-cost accounting over the executed ledger.
    Returns realized P/L, unrealized P/L, avg cost, and per-close (sell) results
    for win-rate / profit-factor (labelled small-n upstream).
    """
    shares = 0.0
    avg_cost = 0.0
    realized = 0.0
    closes = []   # realized $ per sell (a "closed trade" for win/loss stats)

    ex = trades[trades.get("status") == "executed"] if not trades.empty else trades
    for _, r in ex.iterrows():
        delta = float(r.get("delta_shares", 0) or 0)
        fill = float(r.get("fill_price", 0) or 0)
        if delta > 0:                       # buy — raise average cost
            avg_cost = (avg_cost * shares + fill * delta) / (shares + delta) if (shares + delta) else 0.0
            shares += delta
        elif delta < 0 and shares > 0:      # sell — realize vs avg cost
            qty = min(-delta, shares)
            pnl = (fill - avg_cost) * qty
            realized += pnl
            closes.append(pnl)
            shares -= qty

    unrealized = shares * (cur_price - avg_cost) if (shares and cur_price) else 0.0
    wins = [c for c in closes if c > 0]
    losses = [c for c in closes if c <= 0]
    gross_w = sum(wins)
    gross_l = abs(sum(losses))
    return {
        "realized_pl":   realized,
        "unrealized_pl": unrealized,
        "avg_cost":      avg_cost,
        "n_closes":      len(closes),
        "win_rate":      (len(wins) / len(closes)) if closes else None,
        "profit_factor": (gross_w / gross_l) if gross_l > 0 else (float("inf") if gross_w > 0 else None),
        "avg_win":       (np.mean(wins) if wins else 0.0),
        "avg_loss":      (np.mean(losses) if losses else 0.0),
    }


def _period_returns(equity: pd.Series) -> dict:
    """Daily / weekly / monthly returns from the paper NLV series (labelled
    'since inception' upstream when the window is shorter than the period)."""
    if equity is None or len(equity) < 2:
        return {"daily": None, "weekly": None, "monthly": None, "n_days": len(equity) if equity is not None else 0}
    def back(n):
        if len(equity) > n:
            return float(equity.iloc[-1] / equity.iloc[-1 - n] - 1)
        return float(equity.iloc[-1] / equity.iloc[0] - 1)   # since inception
    return {"daily": back(1), "weekly": back(5), "monthly": back(21), "n_days": len(equity)}


# ── public entry point ──────────────────────────────────────────────────────────

def load_paper_account(logs: Path, cur_tqqq_price: float | None) -> dict:
    """Assemble the real simulation account snapshot for the dashboard."""
    port = _load_json(logs / "paper_portfolio.json")
    trades = _load_trades(logs / "paper_trades.csv")

    seed = float(port.get("seed_capital", 10_000.0))
    nlv = float(port.get("nlv", seed))
    shares = int(port.get("tqqq_shares", 0))
    cash = float(port.get("cash", seed))
    peak = float(port.get("peak_equity", seed))
    cur_price = float(cur_tqqq_price) if cur_tqqq_price else (
        float(port.get("last_fill_price", 0)) or 0.0)

    # paper equity series from the ledger (one nlv_after per trading day)
    equity = pd.Series(dtype=float)
    if not trades.empty and "nlv_after" in trades.columns:
        eq = trades.dropna(subset=["nlv_after"]).set_index("date")["nlv_after"].astype(float)
        equity = eq[~eq.index.duplicated(keep="last")]

    pl = _walk_pl(trades, cur_price, seed)
    dd = ((equity.cummax() - equity) / equity.cummax()).max() if len(equity) else 0.0
    per = _period_returns(equity)

    # ── status ─────────────────────────────────────────────────────────────
    # "Running" = the pipeline ran recently. The ledger has a row per run day
    # (incl. no_action), so its last date is the true last-run marker — NOT
    # last_trade_date, which only advances on executed trades and would make a
    # normal no-trade stretch look paused.
    kill = (logs / "ibkr_kill.flag").exists()
    last_trade = str(port.get("last_trade_date") or "")
    last_run = None
    if not trades.empty:
        last_run = trades["date"].max().date()
    elif last_trade:
        try:
            last_run = datetime.strptime(last_trade[:10], "%Y-%m-%d").date()
        except Exception:
            last_run = None
    last_age = (date.today() - last_run).days if last_run else None
    if kill:
        status, reason = "HALTED", "Kill switch active — trading frozen"
    elif last_age is not None and last_age <= STALE_DAYS:
        status, reason = "RUNNING", (
            f"Last run {last_age}d ago" if last_age else "Ran today")
    else:
        status, reason = "PAUSED", (
            f"No run for {last_age}d — check workflows" if last_age is not None else "No runs yet")

    return {
        "connected_to_broker": False,     # SIMULATION — never IBKR
        "seed": seed, "nlv": nlv, "shares": shares, "cash": cash, "peak": peak,
        "cur_price": cur_price,
        "inception": port.get("inception_date", ""),
        "last_trade_date": last_trade or "—",
        "trades_ytd": int(port.get("total_trades_ytd", 0)),
        "total_return": (nlv / seed - 1.0) if seed else 0.0,
        "total_pl": nlv - seed,
        "max_dd": float(dd) if dd == dd else 0.0,
        "invested_pct": (shares * cur_price / nlv) if (nlv and cur_price) else 0.0,
        **pl,
        **{f"ret_{k}": v for k, v in per.items()},
        "equity": equity,
        "trades": trades,
        "status": status, "status_reason": reason, "kill_switch": kill,
    }
