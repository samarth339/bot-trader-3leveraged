"""
IBKR Safety Module Tests
==========================
Unit tests for all components of the IBKR live execution layer.
Uses mocks throughout — no real IB Gateway connection required.

Modules tested:
  ibkr/kill_switch.py         — file-based hard stop
  ibkr/state.py               — persistent execution state
  ibkr/safety_guard.py        — 10 pre-flight safety checks
  ibkr/position_reconciler.py — allocation math
  ibkr/order_manager.py       — order construction logic

Run with:
    pytest tests/test_ibkr_safety.py -v
"""

import json
import math
import tempfile
import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def temp_logs(tmp_path, monkeypatch):
    """
    Redirect all log file paths to a temp directory for test isolation.
    Prevents test pollution of real log files.
    """
    import ibkr.kill_switch as ks
    import ibkr.state       as st
    import ibkr.order_manager as om

    monkeypatch.setattr(ks, "KILL_SWITCH_PATH", tmp_path / "ibkr_kill.flag")
    monkeypatch.setattr(st, "STATE_PATH",       tmp_path / "ibkr_state.json")
    monkeypatch.setattr(om, "ORDERS_LOG",       tmp_path / "ibkr_orders.csv")
    return tmp_path


@pytest.fixture
def clean_state(temp_logs):
    """Return a fresh state dict (no previous execution)."""
    from ibkr import state
    return state.DEFAULT_STATE.copy()


@pytest.fixture
def mock_account():
    """Minimal AccountState for injection into SafetyGuard."""
    from ibkr.account import AccountState
    acc                   = AccountState()
    acc.net_liquidation   = 10_000.0
    acc.available_funds   = 9_500.0
    acc.cash_balance      = 2_000.0
    acc.positions         = {}
    acc.avg_costs         = {}
    return acc


@pytest.fixture
def bull_signal():
    """Valid bull-regime signal row dict."""
    return {
        "as_of_date":   date.today().isoformat(),
        "signal_date":  date.today().isoformat(),
        "regime":       "bull",
        "action":       "HOLD",
        "weight_a":     0.75,
        "weight_b":     0.25,
        "rebalance":    False,
        "drift_pct":    1.0,
        "qqq_price":    480.0,
        "sma_150":      450.0,
        "pct_vs_sma":   6.7,
        "vix_signal":   14.5,
        "vix_raw":      15.0,
        "shadow":       False,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Kill Switch Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestKillSwitch:

    def test_initially_inactive(self):
        """Kill switch should be OFF at start of test (clean temp dir)."""
        from ibkr import kill_switch
        assert not kill_switch.is_active()

    def test_activate_creates_flag_file(self, temp_logs):
        """activate() must create the flag file."""
        from ibkr import kill_switch
        assert not kill_switch.is_active()
        kill_switch.activate("test reason")
        assert kill_switch.is_active()
        assert (temp_logs / "ibkr_kill.flag").exists()

    def test_activate_writes_reason(self):
        """The reason string must be readable back from the flag file."""
        from ibkr import kill_switch
        kill_switch.activate("drawdown exceeded 50%")
        reason = kill_switch.read_reason()
        assert "drawdown exceeded 50%" in reason

    def test_deactivate_removes_file(self):
        """deactivate() must remove the flag file."""
        from ibkr import kill_switch
        kill_switch.activate("test")
        assert kill_switch.is_active()
        kill_switch.deactivate()
        assert not kill_switch.is_active()

    def test_deactivate_when_not_active_is_safe(self):
        """deactivate() on an already-inactive switch must not raise."""
        from ibkr import kill_switch
        kill_switch.deactivate()   # should not raise
        assert not kill_switch.is_active()

    def test_read_reason_when_inactive_returns_empty(self):
        """read_reason() when switch is off must return empty string."""
        from ibkr import kill_switch
        assert kill_switch.read_reason() == ""

    def test_activate_twice_overwrites(self):
        """Second activate() should overwrite the first reason."""
        from ibkr import kill_switch
        kill_switch.activate("first reason")
        kill_switch.activate("second reason")
        reason = kill_switch.read_reason()
        assert "second reason" in reason
        assert "first reason" not in reason

    def test_status_returns_correct_dict(self):
        """status() must return a dict with active, path, reason keys."""
        from ibkr import kill_switch
        kill_switch.activate("status test")
        s = kill_switch.status()
        assert s["active"] is True
        assert "status test" in s["reason"]
        assert "path" in s


# ══════════════════════════════════════════════════════════════════════════════
#  State Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestState:

    def test_load_returns_defaults_when_no_file(self):
        """load() on a missing state file must return DEFAULT_STATE."""
        from ibkr import state
        s = state.load()
        assert s["total_trades_ytd"] == 0
        assert s["last_execution_date"] is None
        assert s["peak_equity"] == 0.0

    def test_save_and_reload(self, temp_logs):
        """save() then load() must round-trip perfectly."""
        from ibkr import state
        original = state.load()
        original["total_trades_ytd"] = 42
        original["last_regime"] = "bull"
        original["peak_equity"] = 12_500.0
        state.save(original)

        reloaded = state.load()
        assert reloaded["total_trades_ytd"] == 42
        assert reloaded["last_regime"] == "bull"
        assert reloaded["peak_equity"] == 12_500.0

    def test_already_executed_today_false_initially(self):
        """already_executed_today() must return False with no prior execution."""
        from ibkr import state
        assert not state.already_executed_today()

    def test_already_executed_today_true_after_record(self):
        """After record_execution(), already_executed_today() must return True."""
        from ibkr import state
        state.record_execution(
            regime="bull", target_pct=0.85, shares_tqqq=130,
            fill_price=65.0, net_liquidation=10_000.0, order_id="test-123"
        )
        assert state.already_executed_today()

    def test_record_execution_increments_trade_count(self):
        """Each record_execution() call increments total_trades_ytd."""
        from ibkr import state
        from datetime import date

        # Use current-year dates to avoid triggering the YTD reset
        current_year = date.today().year
        fake_dates   = [f"{current_year}-01-01", f"{current_year}-01-02",
                        f"{current_year}-01-03"]

        for fake_date in fake_dates:
            # Manually set last_execution_date so record_execution sees a new day
            s = state.load()
            s["last_execution_date"] = fake_date
            state.save(s)
            state.record_execution(
                regime="bull", target_pct=0.85, shares_tqqq=100,
                fill_price=65.0, net_liquidation=10_000.0,
            )

        s = state.load()
        assert s["total_trades_ytd"] == 3

    def test_peak_equity_updates_on_new_high(self):
        """record_execution() must update peak_equity when NLV is a new high."""
        from ibkr import state
        state.record_execution("bull", 0.85, 100, 65.0, 10_000.0)

        s = state.load()
        s["last_execution_date"] = "1999-01-01"
        state.save(s)

        state.record_execution("bull", 0.85, 110, 70.0, 12_000.0)
        s = state.load()
        assert s["peak_equity"] == 12_000.0

    def test_peak_equity_does_not_decrease(self):
        """peak_equity must not decrease on a drawdown (it's a high-water mark)."""
        from ibkr import state
        state.record_execution("bull", 0.85, 100, 65.0, 15_000.0)

        s = state.load()
        s["last_execution_date"] = "1999-01-01"
        state.save(s)

        state.record_execution("high_vol", 0.30, 30, 50.0, 9_000.0)
        s = state.load()
        assert s["peak_equity"] == 15_000.0

    def test_missing_keys_backfilled_on_load(self, temp_logs):
        """Older state files missing new keys must be backfilled with defaults."""
        from ibkr import state
        # Write a minimal old-format state
        old_state = {"last_regime": "bull", "last_fill_price": 65.0}
        (temp_logs / "ibkr_state.json").write_text(json.dumps(old_state))

        s = state.load()
        assert "consecutive_reduce_days" in s, (
            "Missing key 'consecutive_reduce_days' was not backfilled from DEFAULT_STATE"
        )
        assert s["last_regime"] == "bull"   # existing values preserved


# ══════════════════════════════════════════════════════════════════════════════
#  Safety Guard Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSafetyGuard:
    """
    Test each of the 9 safety guards in isolation using a mocked account state
    and controlled clock (to avoid time-of-day dependencies).
    """

    def _make_guard(self, mock_account, bull_signal, **state_overrides):
        """Helper: build a SafetyGuard with controlled state."""
        from ibkr.safety_guard import SafetyGuard
        from ibkr import state as state_module

        if state_overrides:
            s = state_module.load()
            s.update(state_overrides)
            state_module.save(s)

        return SafetyGuard(account_state=mock_account, signal=bull_signal.copy())

    # ── Guard 1: Kill switch ────────────────────────────────────────────────

    def test_guard1_kill_switch_blocks(self, mock_account, bull_signal):
        from ibkr import kill_switch
        from ibkr.safety_guard import SafetyGuard

        kill_switch.activate("test block")
        guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
        result = guard._check_kill_switch()

        assert result.blocked, "Kill switch guard should block when flag is set"
        assert "kill switch" in result.reason.lower()

    def test_guard1_kill_switch_passes_when_inactive(self, mock_account, bull_signal):
        from ibkr.safety_guard import SafetyGuard
        guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
        result = guard._check_kill_switch()
        assert not result.blocked

    # ── Guard 2: Shadow mode ────────────────────────────────────────────────

    def test_guard2_blocks_when_shadow_true_in_signal(
        self, mock_account, bull_signal, temp_logs
    ):
        from ibkr.safety_guard import SafetyGuard, SHADOW_STATE_PATH
        # Write completed shadow state
        (temp_logs / "shadow_state.json").write_text(
            json.dumps({"completed": True, "day": 30})
        )
        shadow_signal = {**bull_signal, "shadow": True}

        with patch.object(
            __import__("ibkr.safety_guard", fromlist=["SHADOW_STATE_PATH"]),
            "SHADOW_STATE_PATH", temp_logs / "shadow_state.json"
        ):
            guard  = SafetyGuard(account_state=mock_account, signal=shadow_signal)
            result = guard._check_shadow_mode()
            assert result.blocked, "shadow=True in signal should block execution"

    def test_guard2_blocks_when_shadow_not_completed(
        self, mock_account, bull_signal, temp_logs
    ):
        from ibkr.safety_guard import SafetyGuard
        (temp_logs / "shadow_state.json").write_text(
            json.dumps({"completed": False, "day": 15})
        )
        with patch("ibkr.safety_guard.SHADOW_STATE_PATH",
                   temp_logs / "shadow_state.json"):
            guard  = SafetyGuard(account_state=mock_account,
                                 signal={**bull_signal, "shadow": False})
            result = guard._check_shadow_mode()
            assert result.blocked, "Shadow mode day 15/30 should block execution"

    # ── Guard 3: Double submission ──────────────────────────────────────────

    def test_guard3_blocks_on_same_day_execution(self, mock_account, bull_signal):
        from ibkr import state as state_module
        from ibkr.safety_guard import SafetyGuard

        s = state_module.load()
        s["last_execution_date"] = date.today().isoformat()
        state_module.save(s)

        guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
        result = guard._check_double_submission()
        assert result.blocked, "Already executed today — should block"

    def test_guard3_passes_on_new_day(self, mock_account, bull_signal):
        from ibkr import state as state_module
        from ibkr.safety_guard import SafetyGuard

        s = state_module.load()
        s["last_execution_date"] = "2020-01-01"   # yesterday
        state_module.save(s)

        guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
        result = guard._check_double_submission()
        assert not result.blocked, "Fresh day — should not block"

    # ── Guard 4: Market hours ───────────────────────────────────────────────

    def test_guard4_blocks_before_window(self, mock_account, bull_signal):
        from ibkr.safety_guard import SafetyGuard
        from datetime import time as dtime
        import pytz

        # Mock time to 10:00 AM EST (too early)
        early_dt = datetime(2024, 3, 6, 10, 0, 0,
                            tzinfo=pytz.timezone("America/New_York"))
        with patch("ibkr.safety_guard.datetime") as mock_dt:
            mock_dt.now.return_value = early_dt
            mock_dt.strptime = datetime.strptime
            guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
            result = guard._check_market_hours()

        assert result.blocked, "10:00 AM is outside execution window — should block"

    def test_guard4_blocks_after_window(self, mock_account, bull_signal):
        from ibkr.safety_guard import SafetyGuard
        import pytz

        late_dt = datetime(2024, 3, 6, 16, 5, 0,
                           tzinfo=pytz.timezone("America/New_York"))
        with patch("ibkr.safety_guard.datetime") as mock_dt:
            mock_dt.now.return_value = late_dt
            mock_dt.strptime = datetime.strptime
            guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
            result = guard._check_market_hours()

        assert result.blocked, "16:05 EST is past execution window — should block"

    def test_guard4_blocks_on_weekend(self, mock_account, bull_signal):
        from ibkr.safety_guard import SafetyGuard
        import pytz

        saturday = datetime(2024, 3, 9, 15, 50, 0,
                            tzinfo=pytz.timezone("America/New_York"))
        with patch("ibkr.safety_guard.datetime") as mock_dt:
            mock_dt.now.return_value = saturday
            mock_dt.strptime = datetime.strptime
            guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
            result = guard._check_market_hours()

        assert result.blocked, "Saturday should be blocked"

    def test_guard4_passes_during_window(self, mock_account, bull_signal):
        from ibkr.safety_guard import SafetyGuard
        import pytz

        window_dt = datetime(2024, 3, 6, 15, 50, 0,   # Wednesday 15:50 EST
                             tzinfo=pytz.timezone("America/New_York"))
        with patch("ibkr.safety_guard.datetime") as mock_dt:
            mock_dt.now.return_value = window_dt
            mock_dt.strptime = datetime.strptime
            guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
            result = guard._check_market_hours()

        assert not result.blocked, "15:50 EST on a Wednesday should pass"

    # ── Guard 5: Portfolio drawdown ─────────────────────────────────────────

    def test_guard5_blocks_on_50pct_drawdown(self, mock_account, bull_signal):
        from ibkr import state as state_module
        from ibkr.safety_guard import SafetyGuard

        # Set peak at 20_000, current NLV at 9_000 → 55% drawdown
        s = state_module.load()
        s["peak_equity"] = 20_000.0
        state_module.save(s)
        mock_account.net_liquidation = 9_000.0

        guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
        result = guard._check_portfolio_drawdown()

        assert result.blocked, "55% portfolio drawdown should block and activate kill switch"

    def test_guard5_activates_kill_switch_on_halt(self, mock_account, bull_signal):
        from ibkr import state as state_module, kill_switch
        from ibkr.safety_guard import SafetyGuard

        s = state_module.load()
        s["peak_equity"] = 20_000.0
        state_module.save(s)
        mock_account.net_liquidation = 8_000.0   # 60% drawdown

        guard = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
        guard._check_portfolio_drawdown()

        assert kill_switch.is_active(), "Kill switch must be activated on halt"

    def test_guard5_passes_on_small_drawdown(self, mock_account, bull_signal):
        from ibkr import state as state_module
        from ibkr.safety_guard import SafetyGuard

        s = state_module.load()
        s["peak_equity"] = 10_500.0  # 5% above current NLV
        state_module.save(s)
        mock_account.net_liquidation = 10_000.0

        guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
        result = guard._check_portfolio_drawdown()
        assert not result.blocked, "5% drawdown should not block"

    # ── Guard 6: Trade frequency ────────────────────────────────────────────

    def test_guard6_blocks_at_100_trades(self, mock_account, bull_signal):
        from ibkr import state as state_module
        from ibkr.safety_guard import SafetyGuard

        s = state_module.load()
        s["total_trades_ytd"] = 100
        state_module.save(s)

        guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
        result = guard._check_trade_frequency()
        assert result.blocked, "100 trades YTD should block (leveraged ETF decay rule)"

    def test_guard6_passes_at_50_trades(self, mock_account, bull_signal):
        from ibkr import state as state_module
        from ibkr.safety_guard import SafetyGuard

        s = state_module.load()
        s["total_trades_ytd"] = 50
        state_module.save(s)

        guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
        result = guard._check_trade_frequency()
        assert not result.blocked

    # ── Guard 8: Position size sanity ───────────────────────────────────────

    def test_guard8_blocks_on_zero_nlv(self, mock_account, bull_signal):
        """
        NLV = 0 (or negative) is a data error from the broker API.
        The position sanity guard must catch and block this condition.
        """
        from ibkr.safety_guard import SafetyGuard

        mock_account.net_liquidation = 0.0   # Zero NLV → account API error

        with patch("ibkr.safety_guard.yf.download") as mock_dl:
            mock_dl.return_value = MagicMock(empty=True)
            guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
            result = guard._check_position_sanity()

        assert result.blocked, "NLV=0 should be blocked as account API error"

    def test_guard8_passes_on_valid_nlv(self, mock_account, bull_signal):
        """
        Normal NLV (e.g. $10,000) with standard allocations (85% blended)
        must pass the sanity check.
        """
        from ibkr.safety_guard import SafetyGuard
        import pandas as pd

        mock_account.net_liquidation = 10_000.0

        normal_tqqq = pd.DataFrame({"Close": [65.0]})
        with patch("ibkr.safety_guard.yf.download", return_value=normal_tqqq):
            guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
            result = guard._check_position_sanity()

        assert not result.blocked, "Valid NLV with normal blended allocation should pass"

    # ── Guard 9: VIX extreme ────────────────────────────────────────────────

    def test_guard9_blocks_buy_on_vix_45(self, mock_account, bull_signal):
        """Live VIX >= 45 should block BUY orders."""
        from ibkr.safety_guard import SafetyGuard
        import pandas as pd

        mock_vix = pd.DataFrame({"Close": [45.5]})
        with patch("ibkr.safety_guard.yf.download", return_value=mock_vix):
            signal = {**bull_signal, "action": "INCREASE_A", "weight_a": 0.75}
            guard  = SafetyGuard(account_state=mock_account, signal=signal)
            result = guard._check_vix_extreme()

        assert result.blocked, "VIX=45.5 should block BUY orders"

    def test_guard9_does_not_block_sell_on_vix_45(self, mock_account, bull_signal):
        """Live VIX >= 45 should NOT block SELL/exit orders."""
        from ibkr.safety_guard import SafetyGuard
        import pandas as pd

        mock_vix = pd.DataFrame({"Close": [45.5]})
        with patch("ibkr.safety_guard.yf.download", return_value=mock_vix):
            signal = {**bull_signal, "action": "REDUCE_A",
                      "_daily_stop_triggered": True}
            guard  = SafetyGuard(account_state=mock_account, signal=signal)
            result = guard._check_vix_extreme()

        assert not result.blocked, "VIX extreme guard must NOT block SELL orders"

    def test_guard9_passes_on_normal_vix(self, mock_account, bull_signal):
        """Normal VIX (< 45) must not block anything."""
        from ibkr.safety_guard import SafetyGuard
        import pandas as pd

        normal_vix = pd.DataFrame({"Close": [17.5]})
        with patch("ibkr.safety_guard.yf.download", return_value=normal_vix):
            guard  = SafetyGuard(account_state=mock_account, signal=bull_signal.copy())
            result = guard._check_vix_extreme()

        assert not result.blocked, "Normal VIX should not block"


# ══════════════════════════════════════════════════════════════════════════════
#  Position Reconciler Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPositionReconciler:
    """Math verification for allocation → share count computation."""

    def _make_reconciler(self, mock_account, tqqq_price: float = 65.0):
        from ibkr.position_reconciler import PositionReconciler
        reconciler = PositionReconciler(account_state=mock_account)
        with patch("ibkr.position_reconciler.yf.download") as mock_dl:
            import pandas as pd
            mock_dl.return_value = pd.DataFrame({"Close": [tqqq_price]})
            return reconciler, mock_dl

    def test_blended_target_pct_bull_regime(self, mock_account, bull_signal):
        """
        Bull: weight_a=0.75 × max_a=0.90 + weight_b=0.25 × max_b=0.70 = 0.85
        """
        from ibkr.position_reconciler import PositionReconciler
        from config.strategy_config import STRATEGY_A_CONFIG, STRATEGY_B_CONFIG

        r = PositionReconciler(mock_account)
        pct = r.compute_blended_target_pct(bull_signal)

        expected = (0.75 * STRATEGY_A_CONFIG["max_position_pct"] +
                    0.25 * STRATEGY_B_CONFIG["max_position_pct"])
        assert abs(pct - expected) < 1e-9, (
            f"Blended target {pct:.4f} != expected {expected:.4f}"
        )

    def test_daily_stop_forces_zero_target(self, mock_account, bull_signal):
        """_daily_stop_triggered flag must override target to 0% (full exit)."""
        from ibkr.position_reconciler import PositionReconciler

        r = PositionReconciler(mock_account)
        signal = {**bull_signal, "_daily_stop_triggered": True}
        pct = r.compute_blended_target_pct(signal)
        assert pct == 0.0, "Daily stop must force target allocation to 0"

    def test_target_shares_math_correct(self, mock_account, bull_signal):
        """
        NLV=10_000, TQQQ@65 → target_shares = floor(NLV * blended_target / price)
        Blended target derived from bull_signal weights × strategy max_position_pct.
        """
        from ibkr.position_reconciler import PositionReconciler
        from config.strategy_config import STRATEGY_A_CONFIG, STRATEGY_B_CONFIG
        import pandas as pd

        mock_account.net_liquidation  = 10_000.0
        mock_account.available_funds  = 9_500.0
        mock_account.positions        = {}

        r = PositionReconciler(mock_account)
        with patch("ibkr.position_reconciler.yf.download") as m:
            m.return_value = pd.DataFrame({"Close": [65.0]})
            plan = r.compute_plan(bull_signal)

        target_pct = (bull_signal["weight_a"] * STRATEGY_A_CONFIG["max_position_pct"] +
                      bull_signal["weight_b"] * STRATEGY_B_CONFIG["max_position_pct"])
        expected_shares = math.floor(10_000 * target_pct / 65.0)
        assert plan.target_shares == expected_shares, (
            f"Expected {expected_shares} shares, got {plan.target_shares}"
        )
        assert plan.delta_shares == expected_shares, (
            f"Starting from 0 shares, delta should be {expected_shares}"
        )

    def test_plan_does_not_proceed_within_drift_tolerance(
        self, mock_account, bull_signal
    ):
        """
        If current allocation is already within 5% of target, proceed=False.
        """
        from ibkr.position_reconciler import PositionReconciler
        import pandas as pd
        from config.strategy_config import STRATEGY_A_CONFIG, STRATEGY_B_CONFIG

        # Set current position close to target (within 5% drift)
        tqqq_price = 65.0
        nlv        = 10_000.0
        target_pct = (0.75 * STRATEGY_A_CONFIG["max_position_pct"] +
                      0.25 * STRATEGY_B_CONFIG["max_position_pct"])
        # Place current 2% below target — always within 5% drift tolerance
        current_shares = math.floor(nlv * (target_pct - 0.02) / tqqq_price)

        mock_account.net_liquidation = nlv
        mock_account.positions       = {"TQQQ": float(current_shares)}

        r = PositionReconciler(mock_account)
        with patch("ibkr.position_reconciler.yf.download") as m:
            m.return_value = pd.DataFrame({"Close": [tqqq_price]})
            plan = r.compute_plan(bull_signal)

        assert not plan.proceed, (
            f"Current {plan.current_pct:.1%} is within 5% of target {plan.target_pct:.1%} "
            "— plan should not proceed"
        )

    def test_buying_power_check_reduces_shares(self, mock_account, bull_signal):
        """If buying power is insufficient, check_buying_power() reduces the delta."""
        from ibkr.position_reconciler import PositionReconciler, RebalancePlan

        plan = RebalancePlan(
            delta_shares=200, target_shares=200, current_shares=0,
            target_pct=0.85, current_pct=0.0, drift_pct=0.85,
            tqqq_price=65.0, regime="bull", action="HOLD",
            reason="BUY 200", proceed=True,
        )
        mock_account.available_funds = 5_000.0  # can only afford ~73 shares
        r = PositionReconciler(mock_account)
        adjusted = r.check_buying_power(plan)

        expected_max = math.floor(5_000.0 / (65.0 * 1.05))
        assert adjusted <= expected_max, (
            f"Buying power check failed: {adjusted} shares > affordable {expected_max}"
        )
        assert adjusted >= 0, "Buying power check must not return negative shares"

    def test_sell_orders_skip_buying_power_check(self, mock_account, bull_signal):
        """Selling shares does not require buying power — should pass unchanged."""
        from ibkr.position_reconciler import PositionReconciler, RebalancePlan

        plan = RebalancePlan(
            delta_shares=-100, target_shares=30, current_shares=130,
            target_pct=0.20, current_pct=0.85, drift_pct=0.65,
            tqqq_price=65.0, regime="high_vol", action="REDUCE_A",
            reason="SELL 100", proceed=True,
        )
        mock_account.available_funds = 0.0   # no cash — but it's a SELL
        r = PositionReconciler(mock_account)
        adjusted = r.check_buying_power(plan)
        assert adjusted == -100, "Sell orders must not be modified by buying power check"


# ══════════════════════════════════════════════════════════════════════════════
#  Order Manager Tests (dry-run only — no real IB connection)
# ══════════════════════════════════════════════════════════════════════════════

class TestOrderManagerDryRun:
    """Verify order construction logic without submitting to IB Gateway."""

    def _make_plan(self, delta: int, tqqq_price: float = 65.0,
                   regime: str = "bull") -> "RebalancePlan":
        from ibkr.position_reconciler import RebalancePlan
        return RebalancePlan(
            delta_shares=delta,
            target_shares=max(0, delta),
            current_shares=0,
            target_pct=0.85,
            current_pct=0.0,
            drift_pct=0.85,
            tqqq_price=tqqq_price,
            regime=regime,
            action="HOLD",
            reason="test",
            proceed=delta != 0,
        )

    def test_dry_run_does_not_submit(self, temp_logs):
        """dry_run=True must not call ib.placeOrder()."""
        from ibkr.order_manager import OrderManager

        mock_ib = MagicMock()
        mgr     = OrderManager(ib=mock_ib)
        plan    = self._make_plan(delta=100)
        result  = mgr.submit_order(plan, dry_run=True)

        mock_ib.placeOrder.assert_not_called()
        assert result.status == "dry_run"

    def test_no_action_on_zero_delta(self, temp_logs):
        """delta_shares=0 must return status='no_action' without touching IB."""
        from ibkr.order_manager import OrderManager

        mock_ib = MagicMock()
        mgr     = OrderManager(ib=mock_ib)
        plan    = self._make_plan(delta=0)
        result  = mgr.submit_order(plan, dry_run=False)

        mock_ib.placeOrder.assert_not_called()
        assert result.status == "no_action"

    def test_dry_run_logs_to_csv(self, temp_logs):
        """Dry run orders must be logged to ibkr_orders.csv."""
        from ibkr.order_manager import OrderManager
        import pandas as pd

        mock_ib = MagicMock()
        mgr     = OrderManager(ib=mock_ib)
        plan    = self._make_plan(delta=100)
        mgr.submit_order(plan, dry_run=True)

        assert (temp_logs / "ibkr_orders.csv").exists(), (
            "ibkr_orders.csv should be created even for dry-run orders"
        )
        df = pd.read_csv(temp_logs / "ibkr_orders.csv")
        assert len(df) == 1
        assert df.iloc[0]["dry_run"] == True

    def test_sell_order_has_correct_direction(self, temp_logs):
        """Negative delta must produce a SELL order."""
        from ibkr.order_manager import OrderManager
        import pandas as pd

        mock_ib = MagicMock()
        mgr     = OrderManager(ib=mock_ib)
        plan    = self._make_plan(delta=-50)

        plan.proceed = True   # force proceed for a sell
        mgr.submit_order(plan, dry_run=True)

        df = pd.read_csv(temp_logs / "ibkr_orders.csv")
        assert df.iloc[0]["direction"] == "SELL"

    def test_buy_order_has_correct_direction(self, temp_logs):
        """Positive delta must produce a BUY order."""
        from ibkr.order_manager import OrderManager
        import pandas as pd

        mock_ib = MagicMock()
        mgr     = OrderManager(ib=mock_ib)
        plan    = self._make_plan(delta=100)
        mgr.submit_order(plan, dry_run=True)

        df = pd.read_csv(temp_logs / "ibkr_orders.csv")
        assert df.iloc[0]["direction"] == "BUY"

    def test_multiple_orders_append_to_csv(self, temp_logs):
        """Multiple submit_order() calls must append rows without overwriting."""
        from ibkr.order_manager import OrderManager
        import pandas as pd

        mock_ib = MagicMock()
        mgr     = OrderManager(ib=mock_ib)

        for delta in [100, -50, 75]:
            plan = self._make_plan(delta=delta)
            plan.proceed = abs(delta) > 0
            mgr.submit_order(plan, dry_run=True)

        df = pd.read_csv(temp_logs / "ibkr_orders.csv")
        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"


# ══════════════════════════════════════════════════════════════════════════════
#  Gap Guard Tests (Guard 10)
# ══════════════════════════════════════════════════════════════════════════════

class TestGapGuard:
    """Unit tests for ibkr/gap_guard.py — no real network calls."""

    def _make_intraday_df(self, open_price: float) -> "pd.DataFrame":
        import pandas as pd
        idx = pd.date_range("2026-05-22 09:30", periods=1, freq="1min")
        return pd.DataFrame({"Open": [open_price], "Close": [open_price]}, index=idx)

    def test_no_trigger_on_small_gap(self, tmp_path, monkeypatch):
        """Gap of -3% is below the 5% threshold — guard should not trigger."""
        import pandas as pd
        from ibkr.gap_guard import GapGuard, TQQQ_CSV

        prev_close = 100.0
        open_price = 97.0   # -3% gap

        csv_df = pd.DataFrame(
            {"close": [prev_close]},
            index=pd.to_datetime(["2026-05-21"]),
        )
        csv_path = tmp_path / "TQQQ_full.csv"
        csv_df.to_csv(csv_path)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv_path)

        with patch("ibkr.gap_guard.yf.download") as mock_dl:
            mock_dl.return_value = self._make_intraday_df(open_price)
            result = GapGuard().check()

        assert not result.triggered
        assert abs(result.gap_pct - (-0.03)) < 1e-9

    def test_triggers_on_large_gap(self, tmp_path, monkeypatch):
        """Gap of -10% exceeds the 5% threshold — guard must trigger."""
        import pandas as pd
        from ibkr.gap_guard import GapGuard

        prev_close = 100.0
        open_price = 90.0   # -10% gap

        csv_df = pd.DataFrame(
            {"close": [prev_close]},
            index=pd.to_datetime(["2026-05-21"]),
        )
        csv_path = tmp_path / "TQQQ_full.csv"
        csv_df.to_csv(csv_path)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv_path)

        with patch("ibkr.gap_guard.yf.download") as mock_dl:
            mock_dl.return_value = self._make_intraday_df(open_price)
            result = GapGuard().check()

        assert result.triggered
        assert result.gap_pct < -0.05
        assert result.open_price == pytest.approx(open_price)
        assert result.prev_close == pytest.approx(prev_close)
        assert "BUY orders blocked" in result.reason

    def test_skipped_gracefully_on_price_fetch_failure(self, tmp_path, monkeypatch):
        """If yfinance fails, guard must not trigger (fail-open, not fail-closed)."""
        import pandas as pd
        from ibkr.gap_guard import GapGuard

        csv_df = pd.DataFrame(
            {"close": [100.0]},
            index=pd.to_datetime(["2026-05-21"]),
        )
        csv_path = tmp_path / "TQQQ_full.csv"
        csv_df.to_csv(csv_path)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv_path)

        with patch("ibkr.gap_guard.yf.download", side_effect=RuntimeError("network error")):
            result = GapGuard().check()

        assert not result.triggered, "Guard must fail-open if price is unavailable"

    def test_today_rows_excluded_from_prev_close(self, tmp_path, monkeypatch):
        """Rows dated today must not be used as prev_close."""
        import pandas as pd
        from ibkr.gap_guard import GapGuard

        today = pd.Timestamp.today().normalize()
        yesterday = today - pd.Timedelta(days=1)

        csv_df = pd.DataFrame(
            {"close": [80.0, 95.0]},            # 80 = yesterday, 95 = today partial
            index=[yesterday, today],
        )
        csv_path = tmp_path / "TQQQ_full.csv"
        csv_df.to_csv(csv_path)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv_path)

        with patch("ibkr.gap_guard.yf.download") as mock_dl:
            # open = 76 → gap vs 80 = -5%, vs 95 = -20%
            mock_dl.return_value = self._make_intraday_df(76.0)
            result = GapGuard().check()

        # Should use 80 (yesterday), not 95 (today)
        assert result.prev_close == pytest.approx(80.0)
        assert result.gap_pct == pytest.approx((76.0 / 80.0) - 1.0)


class TestGapGuardInSafetyGuard:
    """Integration tests: gap_guard wired into SafetyGuard.run_all_checks()."""

    def _make_guard(self, mock_account, signal, tmp_path, monkeypatch, shadow_done=True):
        """Build a SafetyGuard with all filesystem deps redirected."""
        import ibkr.kill_switch as ks
        import ibkr.state as st
        from ibkr.safety_guard import SHADOW_STATE_PATH, SafetyGuard

        monkeypatch.setattr(ks, "KILL_SWITCH_PATH", tmp_path / "ibkr_kill.flag")
        monkeypatch.setattr(st, "STATE_PATH", tmp_path / "ibkr_state.json")

        shadow = tmp_path / "shadow_state.json"
        shadow.write_text(
            '{"completed": true, "day_number": 40}' if shadow_done
            else '{"completed": false, "day": 15}'
        )
        monkeypatch.setattr("ibkr.safety_guard.SHADOW_STATE_PATH", shadow)

        signal["shadow"] = False
        return SafetyGuard(account_state=mock_account, signal=signal)

    def _passthrough(self) -> "GuardResult":
        from ibkr.safety_guard import GuardResult
        return GuardResult(blocked=False)

    def test_gap_guard_blocks_buy_on_large_gap(
        self, tmp_path, monkeypatch, mock_account, bull_signal
    ):
        """Guard 10 must block a HOLD/BUY signal when TQQQ gapped down >5%."""
        from ibkr.gap_guard import GapGuardResult
        from ibkr.safety_guard import SafetyGuard

        guard = self._make_guard(mock_account, bull_signal, tmp_path, monkeypatch)

        # Stub out guards 4, 5, 7, 9 which make real external calls or check
        # live time/prices — we only want to exercise guard 10 here.
        passthrough = self._passthrough
        with patch.object(SafetyGuard, "_check_market_hours",    passthrough), \
             patch.object(SafetyGuard, "_check_portfolio_drawdown", passthrough), \
             patch.object(SafetyGuard, "_check_daily_loss",      passthrough), \
             patch.object(SafetyGuard, "_check_vix_extreme",     passthrough), \
             patch("ibkr.safety_guard.GapGuard") as mock_gg_cls:

            mock_gg = MagicMock()
            mock_gg.check.return_value = GapGuardResult(
                triggered=True, gap_pct=-0.08, open_price=92.0, prev_close=100.0,
                reason="TQQQ opened -8.0% — BUY orders blocked.",
            )
            mock_gg_cls.return_value = mock_gg

            result = guard.run_all_checks()

        assert result.blocked
        assert "BUY orders blocked" in result.reason

    def test_gap_guard_allows_sell_on_large_gap(
        self, tmp_path, monkeypatch, mock_account, bull_signal
    ):
        """Guard 10 must NOT block SELL/REDUCE actions on a gap-down day."""
        from ibkr.gap_guard import GapGuardResult
        from ibkr.safety_guard import SafetyGuard

        sell_signal = {**bull_signal, "action": "REDUCE_A", "weight_a": 0.25, "weight_b": 0.75}
        guard = self._make_guard(mock_account, sell_signal, tmp_path, monkeypatch)

        passthrough = self._passthrough
        with patch.object(SafetyGuard, "_check_market_hours",    passthrough), \
             patch.object(SafetyGuard, "_check_portfolio_drawdown", passthrough), \
             patch.object(SafetyGuard, "_check_daily_loss",      passthrough), \
             patch.object(SafetyGuard, "_check_vix_extreme",     passthrough), \
             patch("ibkr.safety_guard.GapGuard") as mock_gg_cls:

            mock_gg = MagicMock()
            mock_gg.check.return_value = GapGuardResult(
                triggered=True, gap_pct=-0.08, open_price=92.0, prev_close=100.0,
                reason="TQQQ opened -8.0% — BUY orders blocked.",
            )
            mock_gg_cls.return_value = mock_gg

            result = guard.run_all_checks()

        assert not result.blocked, "SELL actions must pass through even on gap-down days"


# ══════════════════════════════════════════════════════════════════════════════
#  VIX Scaling (P3) — evaluated and removed
#  Backtest showed P3 costs CAGR 42%→38% while worsening max DD (-32.5%→-35.3%)
#  when combined with P4 (gap guard).  Tests removed alongside the feature.
# ══════════════════════════════════════════════════════════════════════════════
