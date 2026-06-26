"""
test_executor_fixes.py — Regression tests for the 2026-06 audit fixes
=====================================================================
Locks in the behavior changes that reconcile the live executor with the
validated backtest:

  1. Exposure replay  — per-strategy exposure state is deterministic, bounded,
     and de-risks in stress (not floored at the max-position cap).
  2. Exposure-state targeting — paper_trade.compute_target_pct and
     PositionReconciler.compute_blended_target_pct use weight×exposure when
     present, fall back to max-position caps (loudly) when absent.
  3. Flatten-then-freeze — _force_flatten forces a full exit, and the paper
     plan sells to zero even inside the drift band.
  4. Crash-day buy-block — a ≥7% down day blocks BUY plans only; sells pass.
  5. Trade-frequency cap — blocks BUYs only and resets on a new calendar year.
  6. Runner uses the locked config (no hardcoded drift).
"""

import json
import math
from datetime import date

import pytest


# ══════════════════════════════════════════════════════════════════════════════
#  1. Exposure replay
# ══════════════════════════════════════════════════════════════════════════════

class TestExposureReplay:
    def test_exposures_present_and_bounded(self):
        from backtester.exposure_replay import compute_exposures
        exp = compute_exposures()
        assert 0.0 <= exp["exposure_a"] <= 1.0
        assert 0.0 <= exp["exposure_b"] <= 1.0
        assert exp["state_date"]

    def test_deterministic(self):
        from backtester.exposure_replay import compute_exposures
        a = compute_exposures()
        b = compute_exposures()
        assert a["exposure_a"] == b["exposure_a"]
        assert a["exposure_b"] == b["exposure_b"]

    def test_derisks_in_2022_bear(self):
        """Both strategies should be out of (or light on) TQQQ deep in the 2022 bear."""
        from backtester.exposure_replay import compute_exposures
        exp = compute_exposures(as_of="2022-06-15")
        # Strategy A (190MA + VIX exit) must be fully out; the whole point of
        # the fix is that live exposure is NOT floored at the max-position cap.
        assert exp["exposure_a"] == 0.0, exp
        assert exp["exposure_b"] < 0.30, exp

    def test_engine_close_at_end_false_preserves_position_state(self):
        from backtester.engine import Backtester
        from backtester.exposure_replay import load_processed_data, build_strategy
        from config.strategy_config import STRATEGY_A_CONFIG
        tqqq, sqqq, qqq, vix = load_processed_data()
        bt = Backtester(tqqq, sqqq, qqq, initial_capital=10_000, vix=vix)
        res = bt.run(build_strategy(STRATEGY_A_CONFIG), close_at_end=False)
        assert "final_state" in res
        fs = res["final_state"]
        assert 0.0 <= fs["exposure_pct"] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  2. Exposure-state targeting (both executors agree)
# ══════════════════════════════════════════════════════════════════════════════

class TestExposureTargeting:
    def test_paper_uses_exposure_when_present(self):
        from paper_trade import compute_target_pct
        sig = {"weight_a": 0.9, "weight_b": 0.1,
               "exposure_a": 0.0, "exposure_b": 0.0}
        assert compute_target_pct(sig) == pytest.approx(0.0)

    def test_paper_blends_exposure(self):
        from paper_trade import compute_target_pct
        sig = {"weight_a": 0.9, "weight_b": 0.1,
               "exposure_a": 0.8, "exposure_b": 0.5}
        assert compute_target_pct(sig) == pytest.approx(0.9 * 0.8 + 0.1 * 0.5)

    def test_paper_fallback_to_max_caps_when_missing(self):
        from paper_trade import compute_target_pct
        from config.strategy_config import STRATEGY_A_CONFIG, STRATEGY_B_CONFIG
        sig = {"weight_a": 0.9, "weight_b": 0.1}   # no exposure fields
        expected = (0.9 * STRATEGY_A_CONFIG["max_position_pct"]
                    + 0.1 * STRATEGY_B_CONFIG["max_position_pct"])
        assert compute_target_pct(sig) == pytest.approx(expected)

    def test_paper_fallback_on_blank_strings(self):
        from paper_trade import compute_target_pct
        from config.strategy_config import STRATEGY_A_CONFIG, STRATEGY_B_CONFIG
        sig = {"weight_a": 0.9, "weight_b": 0.1, "exposure_a": "", "exposure_b": ""}
        expected = (0.9 * STRATEGY_A_CONFIG["max_position_pct"]
                    + 0.1 * STRATEGY_B_CONFIG["max_position_pct"])
        assert compute_target_pct(sig) == pytest.approx(expected)

    def test_reconciler_and_paper_agree_on_exposure(self):
        from paper_trade import compute_target_pct
        from ibkr.position_reconciler import PositionReconciler
        sig = {"weight_a": 0.65, "weight_b": 0.35,
               "exposure_a": 0.42, "exposure_b": 0.18}
        assert compute_target_pct(sig) == pytest.approx(
            PositionReconciler.compute_blended_target_pct(sig)
        )

    def test_exposure_floor_lower_than_maxcap_in_stress(self):
        """The whole fix: exposure-state target < max-cap target when de-risked."""
        from paper_trade import compute_target_pct
        from config.strategy_config import STRATEGY_A_CONFIG, STRATEGY_B_CONFIG
        # high_vol weights, strategies fully in cash
        sig_exp = {"weight_a": 0.25, "weight_b": 0.75,
                   "exposure_a": 0.0, "exposure_b": 0.0}
        sig_cap = {"weight_a": 0.25, "weight_b": 0.75}
        assert compute_target_pct(sig_exp) == pytest.approx(0.0)
        # legacy fallback floors at ~66%
        assert compute_target_pct(sig_cap) > 0.6


# ══════════════════════════════════════════════════════════════════════════════
#  3. Flatten-then-freeze
# ══════════════════════════════════════════════════════════════════════════════

class TestForceFlatten:
    def test_force_flatten_zeros_target_paper(self):
        from paper_trade import compute_target_pct
        sig = {"weight_a": 0.9, "weight_b": 0.1,
               "exposure_a": 0.9, "exposure_b": 0.9, "_force_flatten": True}
        assert compute_target_pct(sig) == pytest.approx(0.0)

    def test_force_flatten_sells_inside_drift_band(self):
        """A force-flatten must override the drift gate and sell to zero."""
        from paper_trade import compute_plan
        portfolio = {"tqqq_shares": 100, "cash": 1000.0}
        sig = {"weight_a": 0.9, "weight_b": 0.1,
               "exposure_a": 0.82, "exposure_b": 0.6, "_force_flatten": True}
        plan = compute_plan(sig, portfolio, tqqq_price=80.0)
        assert plan["proceed"] is True
        assert plan["target_shares"] == 0
        assert plan["delta_shares"] == -100

    def test_paper_guard_kill_switch_flattens_when_holding(self, tmp_path, monkeypatch):
        import paper_trade
        monkeypatch.setattr(paper_trade, "KILL_SWITCH_PATH", tmp_path / "kill.flag")
        (tmp_path / "kill.flag").write_text("test halt")
        sig = {"weight_a": 0.9, "weight_b": 0.1, "action": "HOLD"}
        portfolio = {"tqqq_shares": 50, "nlv": 5000.0, "peak_equity": 6000.0,
                     "cash": 1000.0, "total_trades_ytd": 0, "last_trade_date": None}
        guard = paper_trade.PaperSafetyGuard(sig, portfolio, tqqq_closes=[80.0, 79.0])
        blocked, _ = guard.run_all_checks()
        assert not blocked, "holding shares → flatten, not hard block"
        assert sig.get("_force_flatten") is True

    def test_paper_guard_kill_switch_hard_blocks_when_flat(self, tmp_path, monkeypatch):
        import paper_trade
        monkeypatch.setattr(paper_trade, "KILL_SWITCH_PATH", tmp_path / "kill.flag")
        (tmp_path / "kill.flag").write_text("test halt")
        sig = {"weight_a": 0.9, "weight_b": 0.1, "action": "HOLD"}
        portfolio = {"tqqq_shares": 0, "nlv": 5000.0, "peak_equity": 6000.0,
                     "cash": 5000.0, "total_trades_ytd": 0, "last_trade_date": None}
        guard = paper_trade.PaperSafetyGuard(sig, portfolio, tqqq_closes=[80.0, 79.0])
        blocked, reason = guard.run_all_checks()
        assert blocked, "flat account → hard block (nothing to flatten)"
        assert not sig.get("_force_flatten", False)

    def test_paper_dd_halt_flattens_and_writes_killswitch(self, tmp_path, monkeypatch):
        import paper_trade
        monkeypatch.setattr(paper_trade, "KILL_SWITCH_PATH", tmp_path / "kill.flag")
        sig = {"weight_a": 0.9, "weight_b": 0.1, "action": "HOLD"}
        # 40% drawdown vs peak → beyond the 35% halt
        portfolio = {"tqqq_shares": 50, "nlv": 6000.0, "peak_equity": 10000.0,
                     "cash": 100.0, "total_trades_ytd": 0, "last_trade_date": None}
        guard = paper_trade.PaperSafetyGuard(sig, portfolio, tqqq_closes=[80.0, 79.0])
        blocked, _ = guard.run_all_checks()
        assert not blocked
        assert sig.get("_force_flatten") is True
        assert (tmp_path / "kill.flag").exists(), "DD halt must persist the kill switch"


# ══════════════════════════════════════════════════════════════════════════════
#  4. Crash-day buy-block (paper)
# ══════════════════════════════════════════════════════════════════════════════

class TestPaperCrashDay:
    def _portfolio(self):
        return {"tqqq_shares": 0, "nlv": 10000.0, "peak_equity": 10000.0,
                "cash": 10000.0, "total_trades_ytd": 0, "last_trade_date": None}

    def test_crash_day_records_buyblock(self, tmp_path, monkeypatch):
        import paper_trade
        monkeypatch.setattr(paper_trade, "KILL_SWITCH_PATH", tmp_path / "kill.flag")
        sig = {"weight_a": 0.9, "weight_b": 0.1, "action": "HOLD"}
        guard = paper_trade.PaperSafetyGuard(sig, self._portfolio(),
                                             tqqq_closes=[100.0, 90.0])  # −10%
        blocked, _ = guard.run_all_checks()
        assert not blocked
        assert guard.buy_block_reasons
        assert not sig.get("_force_flatten", False)

    def test_normal_day_no_buyblock(self, tmp_path, monkeypatch):
        import paper_trade
        monkeypatch.setattr(paper_trade, "KILL_SWITCH_PATH", tmp_path / "kill.flag")
        sig = {"weight_a": 0.9, "weight_b": 0.1, "action": "HOLD"}
        guard = paper_trade.PaperSafetyGuard(sig, self._portfolio(),
                                             tqqq_closes=[100.0, 98.0])  # −2%
        with monkeypatch.context() as m:
            m.setattr(paper_trade, "fetch_close", lambda t: 15.0)  # calm VIX
            guard.run_all_checks()
        assert not guard.buy_block_reasons


# ══════════════════════════════════════════════════════════════════════════════
#  5. Trade-frequency cap
# ══════════════════════════════════════════════════════════════════════════════

class TestTradeFrequencyCap:
    def test_cap_buyblocks_not_hardblocks(self, tmp_path, monkeypatch):
        import paper_trade
        monkeypatch.setattr(paper_trade, "KILL_SWITCH_PATH", tmp_path / "kill.flag")
        sig = {"weight_a": 0.9, "weight_b": 0.1, "action": "HOLD"}
        portfolio = {"tqqq_shares": 0, "nlv": 10000.0, "peak_equity": 10000.0,
                     "cash": 10000.0, "total_trades_ytd": 100, "last_trade_date": None}
        guard = paper_trade.PaperSafetyGuard(sig, portfolio, tqqq_closes=[100.0, 99.0])
        with monkeypatch.context() as m:
            m.setattr(paper_trade, "fetch_close", lambda t: 15.0)
            blocked, _ = guard.run_all_checks()
        assert not blocked, "frequency cap must not hard-block (sells must pass)"
        assert any("limit" in r.lower() for r in guard.buy_block_reasons)


# ══════════════════════════════════════════════════════════════════════════════
#  6. Runner uses the locked config
# ══════════════════════════════════════════════════════════════════════════════

class TestRunnerLockedConfig:
    def test_runner_components_match_locked_config(self):
        from dual_portfolio_runner import build_locked_components
        from config.strategy_config import (
            PORTFOLIO_DEFAULTS, STRATEGY_A_CONFIG, STRATEGY_B_CONFIG,
        )
        dp = build_locked_components(initial_capital=5000,
                                     data=_tiny_data())
        assert dp.alloc_bull == PORTFOLIO_DEFAULTS["alloc_bull"]
        assert dp.alloc_mid == PORTFOLIO_DEFAULTS["alloc_mid"]
        assert dp.alloc_hi_vol == PORTFOLIO_DEFAULTS["alloc_hi_vol"]
        assert dp.ma_window == PORTFOLIO_DEFAULTS["ma_window"]
        assert dp.vix_bull == PORTFOLIO_DEFAULTS["vix_bull"]
        assert dp.vix_hi_vol == PORTFOLIO_DEFAULTS["vix_hi_vol"]
        assert dp.strategy_a.ma_long == STRATEGY_A_CONFIG["ma_long"]
        assert dp.strategy_a.max_position_pct == STRATEGY_A_CONFIG["max_position_pct"]
        assert dp.strategy_b.ma_long == STRATEGY_B_CONFIG["ma_long"]
        assert dp.strategy_b.crash_brake_pct == STRATEGY_B_CONFIG["crash_brake_pct"]


# ══════════════════════════════════════════════════════════════════════════════
#  7. Exit codes — blocked/no-trade days are exit 0, errors are exit 1
# ══════════════════════════════════════════════════════════════════════════════

class TestPaperExitCodes:
    """
    A gap-down/crash day blocks BUYs — that is the system working, not a crash.
    run() must return True so the workflow stays green and still commits the
    audit record. Genuine errors (no signal, no price) return False (exit 1).
    """

    def _seed(self, tmp_path, monkeypatch, signal_row, portfolio):
        import paper_trade
        monkeypatch.setattr(paper_trade, "KILL_SWITCH_PATH", tmp_path / "kill.flag")
        monkeypatch.setattr(paper_trade, "PORTFOLIO_FILE", tmp_path / "pf.json")
        monkeypatch.setattr(paper_trade, "TRADES_CSV", tmp_path / "trades.csv")
        monkeypatch.setattr(paper_trade, "SIGNAL_LOG", tmp_path / "sig.csv")
        import pandas as pd
        pd.DataFrame([signal_row]).to_csv(tmp_path / "sig.csv", index=False)
        paper_trade.save_portfolio(portfolio)
        return paper_trade

    def _signal_row(self, **over):
        from datetime import date
        row = {
            "as_of_date": date.today().isoformat(),
            "regime": "bull", "action": "HOLD",
            "weight_a": 0.9, "weight_b": 0.1,
            "exposure_a": 0.9, "exposure_b": 0.2,
            "shadow": False, "gap_guard": True, "gap_pct": -9.0,
        }
        row.update(over)
        return row

    def _portfolio(self):
        from paper_trade import DEFAULT_PORTFOLIO
        p = DEFAULT_PORTFOLIO.copy()
        p["last_trade_date"] = None
        return p

    def test_buy_block_returns_true_and_records(self, tmp_path, monkeypatch):
        pt = self._seed(tmp_path, monkeypatch, self._signal_row(), self._portfolio())
        # gap_guard=True in the row → BUY plan blocked; calm VIX, normal prices
        monkeypatch.setattr(pt, "fetch_recent_closes", lambda t, n=2: [100.0, 99.0])
        monkeypatch.setattr(pt, "fetch_close", lambda t: 15.0)
        monkeypatch.setattr(pt, "send_alert", lambda *a, **k: None)
        ok = pt.run(dry_run=False)
        assert ok is True, "a buy-block must NOT fail the workflow (exit 0)"
        import pandas as pd
        trades = pd.read_csv(tmp_path / "trades.csv")
        assert (trades["status"] == "buy_blocked").any()

    def test_missing_signal_returns_false(self, tmp_path, monkeypatch):
        import paper_trade
        monkeypatch.setattr(paper_trade, "SIGNAL_LOG", tmp_path / "nope.csv")
        monkeypatch.setattr(paper_trade, "send_alert", lambda *a, **k: None)
        assert paper_trade.run(dry_run=False) is False, "missing signal → exit 1"

    def test_price_unavailable_returns_false(self, tmp_path, monkeypatch):
        pt = self._seed(tmp_path, monkeypatch, self._signal_row(gap_guard=False),
                        self._portfolio())
        monkeypatch.setattr(pt, "fetch_recent_closes", lambda t, n=2: [])
        monkeypatch.setattr(pt, "send_alert", lambda *a, **k: None)
        assert pt.run(dry_run=False) is False, "no price feed → exit 1"


def _tiny_data():
    """Minimal valid OHLCV frames so the constructor's data guard passes."""
    import pandas as pd
    idx = pd.date_range("2020-01-01", periods=10, freq="1D")
    ohlcv = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                          "close": 100.0, "volume": 1_000}, index=idx)
    vix = pd.DataFrame({"close": 16.0}, index=idx)
    return ohlcv.copy(), ohlcv.copy(), ohlcv.copy(), vix
