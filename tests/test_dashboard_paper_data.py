"""Tests for dashboard/paper_data.py — real paper-account accounting."""
import json
import pandas as pd
import pytest


def _write(tmp_path, portfolio, trades_rows):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "paper_portfolio.json").write_text(json.dumps(portfolio))
    cols = ["date", "regime", "action", "status", "delta_shares", "fill_price", "nlv_after"]
    pd.DataFrame(trades_rows, columns=cols).to_csv(logs / "paper_trades.csv", index=False)
    return logs


def test_average_cost_pl_and_realized(tmp_path):
    from dashboard.paper_data import load_paper_account
    # BUY 100 @ 10, BUY 100 @ 20 (avg 15), SELL 100 @ 25 → realized (25-15)*100 = 1000
    logs = _write(tmp_path,
        {"seed_capital": 10000, "nlv": 12000, "tqqq_shares": 100, "cash": 500,
         "peak_equity": 12000, "last_trade_date": "2026-07-10", "total_trades_ytd": 3,
         "inception_date": "2026-07-01"},
        [["2026-07-02", "bull", "HOLD", "executed", 100, 10.0, 10000],
         ["2026-07-05", "bull", "HOLD", "executed", 100, 20.0, 11000],
         ["2026-07-08", "bull", "HOLD", "executed", -100, 25.0, 12000]])
    p = load_paper_account(logs, cur_tqqq_price=30.0)
    assert p["realized_pl"] == pytest.approx(1000.0)       # (25-15)*100
    assert p["avg_cost"] == pytest.approx(15.0)
    assert p["unrealized_pl"] == pytest.approx((30.0 - 15.0) * 100)  # 100 sh left @ avg 15
    assert p["n_closes"] == 1
    assert p["win_rate"] == 1.0


def test_status_running_on_recent_noaction(tmp_path, monkeypatch):
    """A recent no_action run (no executed trade) must still read RUNNING."""
    from datetime import date, timedelta
    import dashboard.paper_data as pd_mod
    recent = (date.today() - timedelta(days=1)).isoformat()
    logs = _write(tmp_path,
        {"seed_capital": 10000, "nlv": 9800, "tqqq_shares": 100, "cash": 100,
         "peak_equity": 10000, "last_trade_date": "2026-01-01", "total_trades_ytd": 1,
         "inception_date": "2026-01-01"},
        [[recent, "bull", "HOLD", "no_action", 0, "", 9800]])
    p = pd_mod.load_paper_account(logs, cur_tqqq_price=98.0)
    assert p["status"] == "RUNNING", p["status_reason"]


def test_halted_when_kill_switch(tmp_path):
    from dashboard.paper_data import load_paper_account
    logs = _write(tmp_path,
        {"seed_capital": 10000, "nlv": 6000, "tqqq_shares": 0, "cash": 6000,
         "peak_equity": 10000, "last_trade_date": "2026-07-10", "total_trades_ytd": 5,
         "inception_date": "2026-07-01"},
        [["2026-07-10", "high_vol", "HOLD", "executed", -50, 60.0, 6000]])
    (logs / "ibkr_kill.flag").write_text("35% DD halt")
    p = load_paper_account(logs, cur_tqqq_price=60.0)
    assert p["status"] == "HALTED"
    assert p["kill_switch"] is True


def test_broker_always_simulation(tmp_path):
    from dashboard.paper_data import load_paper_account
    logs = _write(tmp_path,
        {"seed_capital": 10000, "nlv": 10000, "tqqq_shares": 0, "cash": 10000,
         "peak_equity": 10000, "last_trade_date": None, "total_trades_ytd": 0,
         "inception_date": "2026-07-01"}, [])
    p = load_paper_account(logs, cur_tqqq_price=77.0)
    assert p["connected_to_broker"] is False
