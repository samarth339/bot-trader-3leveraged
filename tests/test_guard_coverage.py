"""
test_guard_coverage.py — Safety Guard Coverage Gaps
======================================================
Fills the three coverage gaps not addressed in test_ibkr_safety.py:

  1. Crash-day guard (_check_crash_day) — BUY-only block when TQQQ fell ≥7%
     vs the previous close. Never hard-blocks; accumulates a buy_block_reason.
     (Replaces the old last-fill daily-stop, which whipsawed −20% in week 1.)

  2. Guard priority & short-circuit — first HARD-blocking guard stops the chain.
     Tests: a hard block short-circuits; buy-only guards do NOT short-circuit
     and accumulate; kill switch flattens (not blocks) when holding a position.

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
#  Crash-day guard — BUY-only block (never hard-blocks, never force-sells)
# ══════════════════════════════════════════════════════════════════════════════

class TestCrashDayBuyBlock:
    """
    _check_crash_day compares TQQQ's last close to the PREVIOUS close. When the
    drop is ≥ daily_stop_loss it appends a buy_block_reason (BUYs blocked today)
    but NEVER hard-blocks and NEVER force-sells. Exits are owned by the replayed
    strategy state — this only stops buying into a crash.
    """

    def _make_daily_df(self, prev_close: float, last_close: float) -> pd.DataFrame:
        """Minimal yfinance daily dataframe with two closes."""
        idx = pd.date_range("2026-01-14", periods=2, freq="1D")
        return pd.DataFrame({"Close": [prev_close, last_close]}, index=idx)

    def test_crash_day_blocks_buys_on_large_drop(self, fully_stubbed_guard):
        """TQQQ −10% vs prev close (≥7%) → buy_block_reason recorded, not blocked."""
        guard = fully_stubbed_guard
        df = self._make_daily_df(100.0, 90.0)   # −10%
        with patch("ibkr.safety_guard.yf.download", return_value=df):
            result = guard._check_crash_day()
        assert not result.blocked, "crash-day guard must never HARD block"
        assert guard.buy_block_reasons, "a buy-block reason must be recorded on a crash"
        assert not guard.signal.get("_force_flatten", False), (
            "crash-day guard must NOT force-sell (no whipsaw exit)"
        )

    def test_crash_day_no_block_on_small_drop(self, fully_stubbed_guard):
        """TQQQ −2% (below 7%) → no buy-block."""
        guard = fully_stubbed_guard
        df = self._make_daily_df(100.0, 98.0)   # −2%
        with patch("ibkr.safety_guard.yf.download", return_value=df):
            result = guard._check_crash_day()
        assert not result.blocked
        assert not guard.buy_block_reasons

    def test_crash_day_no_block_on_gain(self, fully_stubbed_guard):
        """A green day records no buy-block."""
        guard = fully_stubbed_guard
        df = self._make_daily_df(100.0, 105.0)
        with patch("ibkr.safety_guard.yf.download", return_value=df):
            guard._check_crash_day()
        assert not guard.buy_block_reasons

    def test_crash_day_fails_open_on_fetch_error(self, fully_stubbed_guard):
        """A network error must never block execution."""
        with patch("ibkr.safety_guard.yf.download", side_effect=RuntimeError("net")):
            result = fully_stubbed_guard._check_crash_day()
        assert not result.blocked
        assert not fully_stubbed_guard.buy_block_reasons

    def test_crash_day_fails_open_on_insufficient_data(self, fully_stubbed_guard):
        """A single-bar (or empty) response can't compute a day change → skip."""
        one_bar = pd.DataFrame({"Close": [90.0]},
                               index=pd.date_range("2026-01-15", periods=1))
        with patch("ibkr.safety_guard.yf.download", return_value=one_bar):
            result = fully_stubbed_guard._check_crash_day()
        assert not result.blocked
        assert not fully_stubbed_guard.buy_block_reasons


# ══════════════════════════════════════════════════════════════════════════════
#  Kill switch / DD halt — flatten when holding, hard-block when flat
# ══════════════════════════════════════════════════════════════════════════════

class TestFlattenThenFreeze:
    """
    A risk halt must REDUCE leveraged exposure, never freeze a TQQQ position.
    Guards 1 (kill switch) and 5 (DD halt) set signal['_force_flatten'] when
    shares are held (so the reconciler computes a full exit) and only HARD-block
    when the account is already flat (nothing to sell — freeze buys).
    """

    def test_kill_switch_flattens_when_holding(
        self, mock_account, bull_signal, tmp_path, monkeypatch
    ):
        from ibkr import kill_switch
        from ibkr.safety_guard import SafetyGuard
        monkeypatch.setattr("ibkr.safety_guard.SHADOW_STATE_PATH",
                            tmp_path / "shadow_state.json")
        (tmp_path / "shadow_state.json").write_text('{"completed": true}')
        kill_switch.activate("test — flatten path")

        mock_account.positions = {"TQQQ": 50}   # holding
        bull_signal["shadow"] = False
        guard  = SafetyGuard(account_state=mock_account, signal=bull_signal)
        result = guard._check_kill_switch()

        assert not result.blocked, "kill switch must NOT hard-block while holding"
        assert guard.signal.get("_force_flatten") is True, (
            "kill switch must force a full exit while holding a position"
        )

    def test_kill_switch_hard_blocks_when_flat(
        self, mock_account, bull_signal, tmp_path, monkeypatch
    ):
        from ibkr import kill_switch
        from ibkr.safety_guard import SafetyGuard
        kill_switch.activate("test — freeze path")

        mock_account.positions = {}   # flat
        bull_signal["shadow"] = False
        guard  = SafetyGuard(account_state=mock_account, signal=bull_signal)
        result = guard._check_kill_switch()

        assert result.blocked, "kill switch must hard-block when there is nothing to sell"
        assert not guard.signal.get("_force_flatten", False)

    def test_drawdown_halt_flattens_when_holding(
        self, mock_account, bull_signal, tmp_path, monkeypatch
    ):
        from ibkr.safety_guard import SafetyGuard
        # peak well above current NLV → DD beyond the halt threshold
        state_data = {"peak_equity": 100_000.0, "total_trades_ytd": 0,
                      "last_fill_price": 80.0}
        (tmp_path / "ibkr_state.json").write_text(json.dumps(state_data))

        mock_account.positions = {"TQQQ": 50}
        bull_signal["shadow"] = False
        guard  = SafetyGuard(account_state=mock_account, signal=bull_signal)
        result = guard._check_portfolio_drawdown()

        assert not result.blocked
        assert guard.signal.get("_force_flatten") is True


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

    def test_double_submission_short_circuits_all_subsequent(
        self, mock_account, bull_signal, tmp_path, monkeypatch
    ):
        """
        A hard block (double submission) terminates the chain — later guards
        do not run. Double-submission is now the first hard guard.
        """
        from ibkr.safety_guard import SafetyGuard, GuardResult
        from datetime import date

        guard = self._make_complete_guard(mock_account, bull_signal, tmp_path, monkeypatch)

        with patch.object(SafetyGuard, "_check_shadow_mode", lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_double_submission",
                          lambda s: GuardResult(blocked=True, reason="already executed")):
            result = guard.run_all_checks()

        assert result.blocked
        assert "already executed" in result.reason

    def test_chain_stops_at_first_hard_blocker(
        self, mock_account, bull_signal, tmp_path, monkeypatch
    ):
        """
        Inject a hard block at market_hours and verify later guards
        (kill_switch, etc.) are NOT called.
        """
        from ibkr.safety_guard import SafetyGuard, GuardResult

        guard = self._make_complete_guard(mock_account, bull_signal, tmp_path, monkeypatch)
        guards_called = []

        def blocking_market_hours(s):
            guards_called.append("market_hours")
            return GuardResult(blocked=True, reason="spy: closed")

        def tracking_kill(s):
            guards_called.append("kill_switch")
            return GuardResult(blocked=False)

        with patch.object(SafetyGuard, "_check_shadow_mode",       lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_double_submission", lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_market_hours",      blocking_market_hours), \
             patch.object(SafetyGuard, "_check_kill_switch",       tracking_kill):
            result = guard.run_all_checks()

        assert result.blocked
        assert "market_hours" in guards_called
        assert "kill_switch" not in guards_called, (
            "Guards after the first hard block must NOT run"
        )

    def test_buy_only_guards_do_not_short_circuit_and_accumulate(
        self, mock_account, bull_signal, tmp_path, monkeypatch
    ):
        """
        Buy-only guards (trade_frequency, crash_day, vix_extreme, gap_guard)
        never block the chain — they accumulate into buy_block_reasons and the
        run still returns not-blocked, with later guards still executing.
        """
        from ibkr.safety_guard import SafetyGuard, GuardResult

        guard = self._make_complete_guard(mock_account, bull_signal, tmp_path, monkeypatch)
        ran_after = []

        def freq_records_buyblock(s):
            s.buy_block_reasons.append("freq cap")
            return GuardResult(blocked=False)

        def crash_records_buyblock(s):
            s.buy_block_reasons.append("crash day")
            return GuardResult(blocked=False)

        def tracking_gap(s):
            ran_after.append("gap_guard")
            return GuardResult(blocked=False)

        with patch.object(SafetyGuard, "_check_shadow_mode",       lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_double_submission", lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_market_hours",      lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_kill_switch",       lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_portfolio_drawdown",lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_trade_frequency",   freq_records_buyblock), \
             patch.object(SafetyGuard, "_check_crash_day",         crash_records_buyblock), \
             patch.object(SafetyGuard, "_check_position_sanity",   lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_vix_extreme",       lambda s: GuardResult(blocked=False)), \
             patch.object(SafetyGuard, "_check_gap_guard",         tracking_gap):
            result = guard.run_all_checks()

        assert not result.blocked, "buy-only guards must not hard-block"
        assert len(guard.buy_block_reasons) == 2, guard.buy_block_reasons
        assert "gap_guard" in ran_after, "later guards must still run after buy-only guards"

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
            "shadow_mode", "double_submission", "market_hours", "kill_switch",
            "portfolio_drawdown", "trade_frequency", "crash_day",
            "position_sanity", "vix_extreme", "gap_guard",
        ]

        method_names = [
            "_check_shadow_mode", "_check_double_submission", "_check_market_hours",
            "_check_kill_switch", "_check_portfolio_drawdown", "_check_trade_frequency",
            "_check_crash_day", "_check_position_sanity", "_check_vix_extreme",
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

    def test_force_flatten_override_still_works(self):
        """
        The one signal mutation that DOES affect target — force-flatten (set by
        the kill switch / DD halt). Must force target = 0% regardless of weights.
        """
        from ibkr.position_reconciler import PositionReconciler

        rec = PositionReconciler(account_state=self._make_account())

        signal = {
            "weight_a": 0.75, "weight_b": 0.25,
            "vix_signal": 14.0,
            "_force_flatten": True,
        }
        target = rec.compute_blended_target_pct(signal)

        assert target == pytest.approx(0.0), (
            "Force-flatten override must force target=0.0% (full exit)"
        )
