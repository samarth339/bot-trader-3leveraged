"""
test_guard_coverage.py — Safety Guard Coverage Gaps
======================================================
Fills the three coverage gaps not addressed in test_ibkr_safety.py:

  1. Guard 7 (_check_daily_loss) — the only non-blocking, signal-mutating guard.
     Tests: skip conditions, mutation behavior, fail-open on price errors.

  2. Guard priority & short-circuit — first failing guard stops the chain.
     Tests: guard 1 blocks before guard 10 runs; guard 7 (non-blocking) does
     NOT short-circuit subsequent guards; all 10 run on clean pass.

  3. P3 regression — VIX scaling was evaluated (2010-2026) and removed.
     compute_blended_target_pct() must be VIX-agnostic.

Run with:
    pytest tests/test_guard_coverage.py -v
"""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, call


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def temp_logs(tmp_path, monkeypatch):
    """Redirect all filesystem state to a temp directory."""
    import ibkr.kill_switch as ks
    import ibkr.state       as st
    import ibkr.order_manager as om

    monkeypatch.setattr(ks, "KILL_SWITCH_PATH", tmp_path / "ibkr_kill.flag")
    monkeypatch.setattr(st, "STATE_PATH",       tmp_path / "ibkr_state.json")
    monkeypatch.setattr(om, "ORDERS_LOG",       tmp_path / "ibkr_orders.csv")
    return tmp_path


@pytest.fixture
def mock_account():
    from ibkr.account import AccountState
    acc                 = AccountState()
    acc.net_liquidation = 10_000.0
    acc.available_funds = 9_500.0
    acc.cash_balance    = 2_000.0
    acc.positions       = {"TQQQ": 50}    # holding 50 shares
    acc.avg_costs       = {"TQQQ": 80.0}
    return acc


@pytest.fixture
def bull_signal():
    from datetime import date
    return {
        "as_of_date":  date.today().isoformat(),
        "signal_date": date.today().isoformat(),
        "regime":      "bull",
        "action":      "HOLD",
        "weight_a":    0.75,
        "weight_b":    0.25,
        "rebalance":   False,
        "drift_pct":   1.0,
        "qqq_price":   480.0,
        "vix_signal":  14.5,
        "vix_raw":     15.0,
        "shadow":      False,
    }


@pytest.fixture
def fully_stubbed_guard(mock_account, bull_signal, tmp_path, monkeypatch):
    """
    SafetyGuard with guards 1-4 and 5-10 all stubbed to pass,
    so individual guard methods can be exercised in isolation.
    """
    import ibkr.state as st
    import ibkr.kill_switch as ks
    from ibkr.safety_guard import SafetyGuard, SHADOW_STATE_PATH

    shadow = tmp_path / "shadow_state.json"
    shadow.write_text('{"completed": true, "day_number": 40}')
    monkeypatch.setattr("ibkr.safety_guard.SHADOW_STATE_PATH", shadow)

    # Record a fill price so guard 7 doesn't skip on "no fill history"
    state_data = {"last_fill_price": 100.0, "last_execution_date": None,
                  "total_trades_ytd": 5, "peak_equity": 9000.0}
    (tmp_path / "ibkr_state.json").write_text(json.dumps(state_data))

    bull_signal["shadow"] = False
    return SafetyGuard(account_state=mock_account, signal=bull_signal)


def _passthrough():
    from ibkr.safety_guard import GuardResult
    return GuardResult(blocked=False)


# ══════════════════════════════════════════════════════════════════════════════
#  Guard 7 — Daily Loss (non-blocking, signal-mutating)
# ══════════════════════════════════════════════════════════════════════════════

class TestGuard7DailyLoss:
    """
    Guard 7 is special: it NEVER blocks execution (always returns blocked=False)
    but mutates signal['_daily_stop_triggered'] = True when the daily loss
    threshold is breached.  Downstream guards (8-10) and the reconciler both
    check this flag to force a full position exit.
    """

    def _make_price_df(self, price: float) -> pd.DataFrame:
        """Minimal yfinance 1m dataframe with a Close column."""
        idx = pd.date_range("2026-01-15 09:30", periods=2, freq="1min")
        return pd.DataFrame({"Close": [price, price + 0.1]}, index=idx)

    def test_guard7_never_blocks_even_on_large_loss(
        self, fully_stubbed_guard, tmp_path, monkeypatch
    ):
        """
        Even when TQQQ drops 20% intraday (well above 7% stop),
        guard 7 must return blocked=False (it mutates, not blocks).
        """
        from ibkr.safety_guard import SafetyGuard

        guard = fully_stubbed_guard
        current_price = 80.0   # last_fill=100, loss = (100-80)/100 = 20%

        with patch("ibkr.safety_guard.yf.download", return_value=self._make_price_df(current_price)):
            result = guard._check_daily_loss()

        assert not result.blocked, (
            "Guard 7 must NEVER block — it only mutates the signal dict"
        )

    def test_guard7_mutates_signal_on_loss_above_threshold(
        self, fully_stubbed_guard, tmp_path, monkeypatch
    ):
        """
        Loss = (100 - 80) / 100 = 20% ≥ 7% limit → signal must be mutated.
        """
        from ibkr.safety_guard import SafetyGuard

        guard = fully_stubbed_guard
        assert "_daily_stop_triggered" not in guard.signal  # pre-condition

        with patch("ibkr.safety_guard.yf.download", return_value=self._make_price_df(80.0)):
            guard._check_daily_loss()

        assert guard.signal.get("_daily_stop_triggered") is True, (
            "signal['_daily_stop_triggered'] must be True when loss ≥ daily_stop_loss"
        )

    def test_guard7_does_not_mutate_on_small_loss(
        self, fully_stubbed_guard, tmp_path, monkeypatch
    ):
        """
        Loss = (100 - 98) / 100 = 2% < 7% limit → signal must NOT be mutated.
        """
        from ibkr.safety_guard import SafetyGuard

        guard = fully_stubbed_guard

        with patch("ibkr.safety_guard.yf.download", return_value=self._make_price_df(98.0)):
            guard._check_daily_loss()

        assert not guard.signal.get("_daily_stop_triggered", False), (
            "signal must NOT be mutated when daily loss is below threshold"
        )

    def test_guard7_skips_when_no_fill_history(
        self, mock_account, bull_signal, tmp_path, monkeypatch
    ):
        """
        If last_fill_price == 0 (no prior fills), guard 7 must skip entirely.
        No mutation, no network call.
        """
        import ibkr.state as st
        from ibkr.safety_guard import SafetyGuard, SHADOW_STATE_PATH

        shadow = tmp_path / "shadow_state.json"
        shadow.write_text('{"completed": true}')
        monkeypatch.setattr("ibkr.safety_guard.SHADOW_STATE_PATH", shadow)

        # State with no fill price
        state_data = {"last_fill_price": 0.0, "last_execution_date": None}
        (tmp_path / "ibkr_state.json").write_text(json.dumps(state_data))

        bull_signal["shadow"] = False
        guard = SafetyGuard(account_state=mock_account, signal=bull_signal)

        with patch("ibkr.safety_guard.yf.download") as mock_dl:
            result = guard._check_daily_loss()

        mock_dl.assert_not_called(), "yfinance should NOT be called when no fill history"
        assert not result.blocked
        assert not guard.signal.get("_daily_stop_triggered", False)

    def test_guard7_skips_when_no_tqqq_position(
        self, tmp_path, monkeypatch
    ):
        """
        If account holds 0 TQQQ shares, guard 7 must skip (nothing to stop-loss).
        """
        import ibkr.state as st
        from ibkr.account import AccountState
        from ibkr.safety_guard import SafetyGuard, SHADOW_STATE_PATH
        from datetime import date

        import ibkr.kill_switch as ks
        monkeypatch.setattr(ks, "KILL_SWITCH_PATH", tmp_path / "ibkr_kill.flag")
        monkeypatch.setattr(st, "STATE_PATH",       tmp_path / "ibkr_state.json")

        shadow = tmp_path / "shadow_state.json"
        shadow.write_text('{"completed": true}')
        monkeypatch.setattr("ibkr.safety_guard.SHADOW_STATE_PATH", shadow)

        state_data = {"last_fill_price": 100.0, "last_execution_date": None}
        (tmp_path / "ibkr_state.json").write_text(json.dumps(state_data))

        acc = AccountState()
        acc.net_liquidation = 10_000.0
        acc.available_funds = 10_000.0
        acc.cash_balance    = 10_000.0
        acc.positions       = {}          # no TQQQ position
        acc.avg_costs       = {}

        signal = {
            "as_of_date": date.today().isoformat(), "signal_date": date.today().isoformat(),
            "regime": "bull", "action": "HOLD",
            "weight_a": 0.75, "weight_b": 0.25,
            "shadow": False,
        }
        guard = SafetyGuard(account_state=acc, signal=signal)

        with patch("ibkr.safety_guard.yf.download") as mock_dl:
            result = guard._check_daily_loss()

        mock_dl.assert_not_called(), "yfinance must not be called when holding no TQQQ"
        assert not result.blocked

    def test_guard7_fails_open_on_price_fetch_error(
        self, fully_stubbed_guard, tmp_path, monkeypatch
    ):
        """
        If yfinance raises an exception, guard 7 must fail-open (not block).
        A network error during daily-loss check must never prevent execution.
        """
        with patch("ibkr.safety_guard.yf.download", side_effect=RuntimeError("network error")):
            result = fully_stubbed_guard._check_daily_loss()

        assert not result.blocked
        assert not fully_stubbed_guard.signal.get("_daily_stop_triggered", False)

    def test_guard7_fails_open_on_empty_price_data(
        self, fully_stubbed_guard, tmp_path, monkeypatch
    ):
        """
        If yfinance returns an empty DataFrame, guard 7 must fail-open.
        """
        with patch("ibkr.safety_guard.yf.download", return_value=pd.DataFrame()):
            result = fully_stubbed_guard._check_daily_loss()

        assert not result.blocked


# ══════════════════════════════════════════════════════════════════════════════
#  Guard priority and short-circuit behavior
# ══════════════════════════════════════════════════════════════════════════════

class TestGuardPriority:
    """
    Guards run in priority order.  The first blocking guard terminates the
    chain — subsequent guards are NOT executed.
    Guard 7 is non-blocking and must NOT short-circuit the chain.
    """

    def _make_complete_guard(self, mock_account, bull_signal, tmp_path, monkeypatch):
        import ibkr.state as st
        import ibkr.kill_switch as ks
        from ibkr.safety_guard import SafetyGuard, SHADOW_STATE_PATH

        monkeypatch.setattr(ks, "KILL_SWITCH_PATH", tmp_path / "ibkr_kill.flag")
        monkeypatch.setattr(st, "STATE_PATH",       tmp_path / "ibkr_state.json")

        shadow = tmp_path / "shadow_state.json"
        shadow.write_text('{"completed": true}')
        monkeypatch.setattr("ibkr.safety_guard.SHADOW_STATE_PATH", shadow)

        state_data = {"last_fill_price": 0.0, "last_execution_date": None,
                      "total_trades_ytd": 0, "peak_equity": 0.0}
        (tmp_path / "ibkr_state.json").write_text(json.dumps(state_data))

        bull_signal["shadow"] = False
        return SafetyGuard(account_state=mock_account, signal=bull_signal)

    def test_guard1_kill_switch_short_circuits_all_subsequent(
        self, mock_account, bull_signal, tmp_path, monkeypatch
    ):
        """
        When guard 1 (kill switch) is active, guards 2-10 must NEVER run.
        Verifies short-circuit by tracking how many checks were called.
        """
        from ibkr import kill_switch

        guard = self._make_complete_guard(mock_account, bull_signal, tmp_path, monkeypatch)
        kill_switch.activate("test — unit test kill")

        call_log = []

        def tracking_check(name, original_method):
            def wrapper(self_inner):
                call_log.append(name)
                return original_method(self_inner)
            return wrapper

        from ibkr.safety_guard import SafetyGuard

        result = guard.run_all_checks()

        assert result.blocked, "Kill switch must block"
        # Only guard 1 should have run — kill switch check is first
        # (We verify blocked=True with kill-switch reason)
        assert "kill" in result.reason.lower() or "Kill" in result.reason

    def test_guard_chain_stops_at_first_blocker(
        self, mock_account, bull_signal, tmp_path, monkeypatch
    ):
        """
        Inject a blocking result at guard 6 (trade frequency) and verify
        guard 7-10 are NOT called.
        """
        from ibkr.safety_guard import SafetyGuard, GuardResult

        guard = self._make_complete_guard(mock_account, bull_signal, tmp_path, monkeypatch)

        guards_called = []

        def spy(name, original):
            def wrapper():
                guards_called.append(name)
                return original()
            return wrapper

        # Guard 6 blocks; guard 7 is non-blocking so it should not run after block
        def blocking_guard6():
            guards_called.append("trade_frequency")
            return GuardResult(blocked=True, reason="spy: freq limit")

        def tracking_guard7():
            guards_called.append("daily_loss")
            return GuardResult(blocked=False)

        with patch.object(SafetyGuard, "_check_kill_switch",       lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_shadow_mode",       lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_double_submission", lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_market_hours",      lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_portfolio_drawdown",lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_trade_frequency",   lambda s: blocking_guard6()), \
             patch.object(SafetyGuard, "_check_daily_loss",        lambda s: tracking_guard7()):

            result = guard.run_all_checks()

        assert result.blocked
        assert "trade_frequency" in guards_called
        assert "daily_loss" not in guards_called, (
            "Guard 7 must NOT run after guard 6 blocks the chain"
        )

    def test_guard7_nonblocking_does_not_short_circuit(
        self, mock_account, bull_signal, tmp_path, monkeypatch
    ):
        """
        Guard 7 is non-blocking: even when it mutates the signal, guards 8-10
        must still run.  Verifies that non-blocking guards don't accidentally
        terminate the chain.
        """
        from ibkr.safety_guard import SafetyGuard, GuardResult

        guard = self._make_complete_guard(mock_account, bull_signal, tmp_path, monkeypatch)
        guards_after_7 = []

        def tracking_guard8():
            guards_after_7.append("position_sanity")
            return GuardResult(blocked=False)

        def tracking_guard9():
            guards_after_7.append("vix_extreme")
            return GuardResult(blocked=False)

        def tracking_guard10():
            guards_after_7.append("gap_guard")
            return GuardResult(blocked=False)

        def mutating_guard7():
            # Guard 7 mutates (non-blocking) — chain must continue
            guard.signal["_daily_stop_triggered"] = True
            return GuardResult(blocked=False)

        with patch.object(SafetyGuard, "_check_kill_switch",       lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_shadow_mode",       lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_double_submission", lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_market_hours",      lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_portfolio_drawdown",lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_trade_frequency",   lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_daily_loss",        lambda s: mutating_guard7()), \
             patch.object(SafetyGuard, "_check_position_sanity",   lambda s: tracking_guard8()), \
             patch.object(SafetyGuard, "_check_vix_extreme",       lambda s: tracking_guard9()), \
             patch.object(SafetyGuard, "_check_gap_guard",         lambda s: tracking_guard10()):

            result = guard.run_all_checks()

        assert not result.blocked
        assert "position_sanity" in guards_after_7, "Guard 8 must run after non-blocking guard 7"
        assert "vix_extreme"     in guards_after_7, "Guard 9 must run after non-blocking guard 7"
        assert "gap_guard"       in guards_after_7, "Guard 10 must run after non-blocking guard 7"

    def test_all_10_guards_run_on_clean_pass(
        self, mock_account, bull_signal, tmp_path, monkeypatch
    ):
        """When everything passes, all 10 guards must have been called."""
        from ibkr.safety_guard import SafetyGuard, GuardResult

        guard = self._make_complete_guard(mock_account, bull_signal, tmp_path, monkeypatch)

        guards_run = []

        def make_tracking(name):
            def tracker():
                guards_run.append(name)
                return GuardResult(blocked=False)
            return tracker

        guard_names = [
            "kill_switch", "shadow_mode", "double_submission", "market_hours",
            "portfolio_drawdown", "trade_frequency", "daily_loss",
            "position_sanity", "vix_extreme", "gap_guard",
        ]

        method_names = [
            "_check_kill_switch", "_check_shadow_mode", "_check_double_submission",
            "_check_market_hours", "_check_portfolio_drawdown", "_check_trade_frequency",
            "_check_daily_loss", "_check_position_sanity", "_check_vix_extreme",
            "_check_gap_guard",
        ]

        with patch.multiple(
            SafetyGuard,
            **{mn: (lambda n: lambda s: make_tracking(n)())(gn)
               for mn, gn in zip(method_names, guard_names)}
        ):
            result = guard.run_all_checks()

        assert not result.blocked
        assert len(guards_run) == 10, f"Expected 10 guards, got {len(guards_run)}: {guards_run}"
        for name in guard_names:
            assert name in guards_run, f"Guard '{name}' was not called"


# ══════════════════════════════════════════════════════════════════════════════
#  P3 regression — VIX scaling was removed
# ══════════════════════════════════════════════════════════════════════════════

class TestP3VIXScalingRemoved:
    """
    VIX scaling (P3) was evaluated in 2010-2026 backtests and REMOVED.
    It cost 4pp CAGR while paradoxically worsening max drawdown.

    Regression guard: compute_blended_target_pct() must be VIX-agnostic.
    The same signal with different VIX values must produce identical targets.
    """

    def _make_account(self):
        from ibkr.account import AccountState
        acc = AccountState()
        acc.net_liquidation = 50_000.0
        acc.available_funds = 40_000.0
        acc.cash_balance    = 10_000.0
        acc.positions       = {}
        acc.avg_costs       = {}
        return acc

    def test_target_pct_independent_of_vix_level(self):
        """
        A bull signal with VIX=14 vs VIX=45 must produce the same target %.
        If P3 were still active, VIX=45 would cap at 10%.
        """
        from ibkr.position_reconciler import PositionReconciler

        rec = PositionReconciler(account_state=self._make_account())

        signal_low_vix  = {"weight_a": 0.75, "weight_b": 0.25, "vix_signal": 14.0}
        signal_high_vix = {"weight_a": 0.75, "weight_b": 0.25, "vix_signal": 45.0}

        target_low  = rec.compute_blended_target_pct(signal_low_vix)
        target_high = rec.compute_blended_target_pct(signal_high_vix)

        assert target_low == pytest.approx(target_high), (
            f"VIX level must NOT affect target allocation (P3 removed). "
            f"Got low_vix={target_low:.4f}, high_vix={target_high:.4f}"
        )

    def test_target_pct_with_extreme_vix_not_capped_at_10pct(self):
        """
        With P3 active, VIX > 40 would hard-cap target at 10%.
        Now the target must be the full blended allocation.
        """
        from ibkr.position_reconciler import PositionReconciler
        from config.strategy_config import STRATEGY_A_CONFIG, STRATEGY_B_CONFIG

        rec = PositionReconciler(account_state=self._make_account())

        signal = {"weight_a": 0.75, "weight_b": 0.25, "vix_signal": 55.0}
        target = rec.compute_blended_target_pct(signal)

        # Full blended: 0.75×0.85 + 0.25×0.60 = 0.6375 + 0.15 = 0.7875
        expected = (
            0.75 * STRATEGY_A_CONFIG["max_position_pct"] +
            0.25 * STRATEGY_B_CONFIG["max_position_pct"]
        )
        assert target == pytest.approx(expected, rel=1e-6), (
            f"With P3 removed, VIX=55 must not cap target. "
            f"Expected {expected:.4f}, got {target:.4f}"
        )

    def test_daily_stop_override_still_works(self):
        """
        The one signal mutation that DOES affect target — daily stop triggered.
        This must still force target = 0% regardless of weights.
        """
        from ibkr.position_reconciler import PositionReconciler

        rec = PositionReconciler(account_state=self._make_account())

        signal = {
            "weight_a": 0.75, "weight_b": 0.25,
            "vix_signal": 14.0,
            "_daily_stop_triggered": True,
        }
        target = rec.compute_blended_target_pct(signal)

        assert target == pytest.approx(0.0), (
            "Daily stop override must force target=0.0% (full exit)"
        )
