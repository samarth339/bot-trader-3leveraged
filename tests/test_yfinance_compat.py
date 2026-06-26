"""
test_yfinance_compat.py — yfinance MultiIndex Compatibility Tests
==================================================================
Verifies that all four yf.download() call sites handle the MultiIndex
column format introduced in yfinance 1.3.0, while remaining backwards-
compatible with the older flat-column format.

Affected call sites:
  1. ibkr/gap_guard.py      — _get_today_open()       uses data["Open"]
  2. ibkr/safety_guard.py   — _check_daily_loss()     uses data["Close"]
  3. ibkr/safety_guard.py   — _check_vix_extreme()    uses data["Close"]
  4. ibkr/position_reconciler.py — get_live_tqqq_price() uses data["Close"]

Run with:
    pytest tests/test_yfinance_compat.py -v
"""

import math
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_multiindex_df(ticker: str, close: float, open_: float = None) -> pd.DataFrame:
    """
    Mimics yfinance 1.3.0 MultiIndex output for a single ticker.

    Returns a DataFrame whose columns are a MultiIndex:
      [('Close', 'TQQQ'), ('High', 'TQQQ'), ('Low', 'TQQQ'), ('Open', 'TQQQ'), ('Volume', 'TQQQ')]
    so that data["Close"] is a Series-of-Series (not a plain float Series)
    before our droplevel fix is applied.

    Args:
        ticker: ticker symbol (e.g. "TQQQ", "^VIX")
        close:  the close price to use for all rows
        open_:  the open price (defaults to same as close)
    """
    if open_ is None:
        open_ = close
    idx = pd.date_range("2026-01-15 09:30", periods=3, freq="1min")
    cols = pd.MultiIndex.from_tuples(
        [
            ("Close",  ticker),
            ("High",   ticker),
            ("Low",    ticker),
            ("Open",   ticker),
            ("Volume", ticker),
        ],
        names=["Price", "Ticker"],
    )
    row = [close, close * 1.001, close * 0.999, open_, 1_000_000]
    return pd.DataFrame([row] * 3, index=idx, columns=cols)


def _mock_flat_df(price: float, open_: float = None) -> pd.DataFrame:
    """
    Mimics old yfinance flat-column output (pre-1.3.0).
    data["Close"] and data["Open"] return plain numeric Series.
    """
    if open_ is None:
        open_ = price
    idx = pd.date_range("2026-01-15 09:30", periods=3, freq="1min")
    return pd.DataFrame(
        {
            "Close":  [price] * 3,
            "High":   [price * 1.001] * 3,
            "Low":    [price * 0.999] * 3,
            "Open":   [open_] * 3,
            "Volume": [1_000_000] * 3,
        },
        index=idx,
    )


def _empty_df() -> pd.DataFrame:
    """Mimics yfinance returning an empty DataFrame (market not open yet, etc.)."""
    return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
#  Call site 1: ibkr/gap_guard.py — _get_today_open()
# ══════════════════════════════════════════════════════════════════════════════

class TestGapGuardYFinanceCompat:
    """Tests for the yfinance MultiIndex fix in ibkr/gap_guard.py."""

    def _make_csv(self, tmp_path: Path, prev_close: float = 100.0) -> Path:
        csv_df = pd.DataFrame(
            {"close": [prev_close]},
            index=pd.to_datetime(["2026-01-14"]),
        )
        csv_path = tmp_path / "TQQQ_full.csv"
        csv_df.to_csv(csv_path)
        return csv_path

    def test_multiindex_open_price_parsed_correctly(self, tmp_path, monkeypatch):
        """
        yfinance 1.3.0 MultiIndex: data["Open"] previously returned a Series-of-
        Series, causing float() to raise TypeError.  After fix, must return a float.
        """
        from ibkr.gap_guard import GapGuard

        csv_path = self._make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv_path)

        multiindex_df = _mock_multiindex_df("TQQQ", close=97.0, open_=97.0)
        with patch("ibkr.gap_guard.yf.download", return_value=multiindex_df):
            result = GapGuard().check()

        # -3% gap — below the 5% threshold — guard must NOT trigger
        assert not result.triggered
        assert result.open_price == pytest.approx(97.0)
        assert abs(result.gap_pct - (-0.03)) < 1e-6

    def test_flat_columns_still_work(self, tmp_path, monkeypatch):
        """
        Old flat-column format must continue to work without errors.
        """
        from ibkr.gap_guard import GapGuard

        csv_path = self._make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv_path)

        flat_df = _mock_flat_df(price=97.0, open_=97.0)
        with patch("ibkr.gap_guard.yf.download", return_value=flat_df):
            result = GapGuard().check()

        assert not result.triggered
        assert result.open_price == pytest.approx(97.0)

    def test_empty_download_fails_open(self, tmp_path, monkeypatch):
        """
        Empty DataFrame from yfinance must cause _get_today_open() to return None
        and guard to fail-open (triggered=False, not raised exception).
        """
        from ibkr.gap_guard import GapGuard

        csv_path = self._make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv_path)

        with patch("ibkr.gap_guard.yf.download", return_value=_empty_df()):
            result = GapGuard().check()

        assert not result.triggered
        assert "price unavailable" in result.reason

    def test_runtime_error_fails_open(self, tmp_path, monkeypatch):
        """
        RuntimeError during yfinance download must be caught; guard must fail-open.
        """
        from ibkr.gap_guard import GapGuard

        csv_path = self._make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv_path)

        with patch("ibkr.gap_guard.yf.download", side_effect=RuntimeError("net err")):
            result = GapGuard().check()

        assert not result.triggered
        assert "price unavailable" in result.reason

    def test_multiindex_large_gap_triggers(self, tmp_path, monkeypatch):
        """
        MultiIndex format with a -10% gap must still trigger the guard.
        """
        from ibkr.gap_guard import GapGuard

        csv_path = self._make_csv(tmp_path, prev_close=100.0)
        monkeypatch.setattr("ibkr.gap_guard.TQQQ_CSV", csv_path)

        multiindex_df = _mock_multiindex_df("TQQQ", close=90.0, open_=90.0)
        with patch("ibkr.gap_guard.yf.download", return_value=multiindex_df):
            result = GapGuard().check()

        assert result.triggered
        assert result.gap_pct < -0.05
        assert "BUY orders blocked" in result.reason


# ══════════════════════════════════════════════════════════════════════════════
#  Call site 2: ibkr/safety_guard.py — _check_daily_loss()
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def redirect_ibkr_paths(tmp_path, monkeypatch):
    """Redirect kill switch and state paths to tmp_path for test isolation."""
    import ibkr.kill_switch as ks
    import ibkr.state as st
    monkeypatch.setattr(ks, "KILL_SWITCH_PATH", tmp_path / "ibkr_kill.flag")
    monkeypatch.setattr(st, "STATE_PATH", tmp_path / "ibkr_state.json")


@pytest.fixture
def mock_account_with_tqqq():
    """AccountState holding TQQQ shares with a prior fill price."""
    from ibkr.account import AccountState
    from ibkr import state as state_module

    acc = AccountState()
    acc.net_liquidation = 10_000.0
    acc.available_funds = 9_500.0
    acc.cash_balance = 2_000.0
    acc.positions = {"TQQQ": 100.0}
    acc.avg_costs = {"TQQQ": 65.0}

    # Store last_fill_price so guard 7 has something to compare against
    s = state_module.load()
    s["last_fill_price"] = 65.0
    state_module.save(s)

    return acc


@pytest.fixture
def bull_signal_live():
    """Signal dict representing a live (non-shadow) bull regime."""
    from datetime import date
    return {
        "as_of_date": date.today().isoformat(),
        "signal_date": date.today().isoformat(),
        "regime": "bull",
        "action": "HOLD",
        "weight_a": 0.75,
        "weight_b": 0.25,
        "rebalance": False,
        "drift_pct": 1.0,
        "qqq_price": 480.0,
        "sma_150": 450.0,
        "pct_vs_sma": 6.7,
        "vix_signal": 14.5,
        "vix_raw": 15.0,
        "shadow": False,
    }


def _mock_multiindex_2day(ticker: str, prev_close: float, last_close: float) -> pd.DataFrame:
    """Two-bar MultiIndex daily df (prev close then last close) for crash-day tests."""
    idx = pd.date_range("2026-01-14", periods=2, freq="1D")
    cols = pd.MultiIndex.from_tuples(
        [("Close", ticker), ("High", ticker), ("Low", ticker),
         ("Open", ticker), ("Volume", ticker)],
        names=["Price", "Ticker"],
    )
    rows = [
        [prev_close, prev_close, prev_close, prev_close, 1_000_000],
        [last_close, last_close, last_close, last_close, 1_000_000],
    ]
    return pd.DataFrame(rows, index=idx, columns=cols)


def _mock_flat_2day(prev_close: float, last_close: float) -> pd.DataFrame:
    """Two-bar flat-column daily df for crash-day tests."""
    idx = pd.date_range("2026-01-14", periods=2, freq="1D")
    return pd.DataFrame(
        {"Close": [prev_close, last_close], "High": [prev_close, last_close],
         "Low": [prev_close, last_close], "Open": [prev_close, last_close],
         "Volume": [1_000_000, 1_000_000]},
        index=idx,
    )


class TestSafetyGuardCrashDayCompat:
    """Tests for the yfinance MultiIndex fix in safety_guard._check_crash_day()."""

    def _make_guard(self, mock_account_with_tqqq, bull_signal_live):
        from ibkr.safety_guard import SafetyGuard
        return SafetyGuard(
            account_state=mock_account_with_tqqq,
            signal=bull_signal_live.copy(),
        )

    def test_multiindex_parsed_no_buyblock_on_small_drop(
        self, mock_account_with_tqqq, bull_signal_live
    ):
        """MultiIndex must parse cleanly; −3% day records no buy-block."""
        df = _mock_multiindex_2day("TQQQ", prev_close=65.0, last_close=63.0)
        guard = self._make_guard(mock_account_with_tqqq, bull_signal_live)

        with patch("ibkr.safety_guard.yf.download", return_value=df):
            result = guard._check_crash_day()

        assert not result.blocked
        assert not guard.buy_block_reasons

    def test_flat_columns_crash_day_still_work(
        self, mock_account_with_tqqq, bull_signal_live
    ):
        """Old flat-column format must still work in _check_crash_day."""
        df = _mock_flat_2day(prev_close=65.0, last_close=63.0)
        guard = self._make_guard(mock_account_with_tqqq, bull_signal_live)

        with patch("ibkr.safety_guard.yf.download", return_value=df):
            result = guard._check_crash_day()

        assert not result.blocked

    def test_multiindex_buyblocks_on_large_drop(
        self, mock_account_with_tqqq, bull_signal_live
    ):
        """−9% day (≥7%) records a buy-block but never hard-blocks/force-sells."""
        df = _mock_multiindex_2day("TQQQ", prev_close=65.0, last_close=59.0)
        guard = self._make_guard(mock_account_with_tqqq, bull_signal_live)

        with patch("ibkr.safety_guard.yf.download", return_value=df):
            result = guard._check_crash_day()

        assert not result.blocked
        assert guard.buy_block_reasons, "≥7% drop must record a BUY block"
        assert not guard.signal.get("_force_flatten", False)

    def test_empty_download_fails_open_crash_day(
        self, mock_account_with_tqqq, bull_signal_live
    ):
        """Empty DataFrame from yfinance must not block — fails-open."""
        guard = self._make_guard(mock_account_with_tqqq, bull_signal_live)

        with patch("ibkr.safety_guard.yf.download", return_value=_empty_df()):
            result = guard._check_crash_day()

        assert not result.blocked
        assert not guard.buy_block_reasons

    def test_runtime_error_fails_open_crash_day(
        self, mock_account_with_tqqq, bull_signal_live
    ):
        """RuntimeError during download must be swallowed; guard fails-open."""
        guard = self._make_guard(mock_account_with_tqqq, bull_signal_live)

        with patch(
            "ibkr.safety_guard.yf.download",
            side_effect=RuntimeError("network error"),
        ):
            result = guard._check_crash_day()

        assert not result.blocked


# ══════════════════════════════════════════════════════════════════════════════
#  Call site 3: ibkr/safety_guard.py — _check_vix_extreme()
# ══════════════════════════════════════════════════════════════════════════════

class TestSafetyGuardVixExtremeCompat:
    """Tests for the yfinance MultiIndex fix in safety_guard._check_vix_extreme()."""

    def _make_guard(self, account, signal):
        from ibkr.safety_guard import SafetyGuard
        return SafetyGuard(account_state=account, signal=signal.copy())

    def test_multiindex_vix_normal_does_not_block(
        self, mock_account_with_tqqq, bull_signal_live
    ):
        """MultiIndex VIX data at 17.5 (below 45) must not block BUY orders."""
        guard = self._make_guard(mock_account_with_tqqq, bull_signal_live)
        multiindex_df = _mock_multiindex_df("^VIX", close=17.5)

        with patch("ibkr.safety_guard.yf.download", return_value=multiindex_df):
            result = guard._check_vix_extreme()

        assert not result.blocked

    def test_flat_columns_vix_still_work(
        self, mock_account_with_tqqq, bull_signal_live
    ):
        """Old flat-column VIX data must still be handled correctly."""
        guard = self._make_guard(mock_account_with_tqqq, bull_signal_live)
        flat_df = _mock_flat_df(price=17.5)

        with patch("ibkr.safety_guard.yf.download", return_value=flat_df):
            result = guard._check_vix_extreme()

        assert not result.blocked

    def test_multiindex_vix_extreme_records_buyblock(
        self, mock_account_with_tqqq, bull_signal_live
    ):
        """MultiIndex VIX at 46 records a BUY block (never hard-blocks)."""
        signal = {**bull_signal_live, "action": "INCREASE_A", "weight_a": 0.75}
        guard = self._make_guard(mock_account_with_tqqq, signal)
        multiindex_df = _mock_multiindex_df("^VIX", close=46.0)

        with patch("ibkr.safety_guard.yf.download", return_value=multiindex_df):
            result = guard._check_vix_extreme()

        assert not result.blocked, "VIX extreme must not hard-block (sells must pass)"
        assert guard.buy_block_reasons
        assert any("46" in r or "VIX" in r for r in guard.buy_block_reasons)

    def test_multiindex_vix_extreme_allows_sell(
        self, mock_account_with_tqqq, bull_signal_live
    ):
        """VIX extreme must NOT hard-block — SELL plans always proceed."""
        signal = {**bull_signal_live, "action": "REDUCE_A"}
        guard = self._make_guard(mock_account_with_tqqq, signal)
        multiindex_df = _mock_multiindex_df("^VIX", close=46.0)

        with patch("ibkr.safety_guard.yf.download", return_value=multiindex_df):
            result = guard._check_vix_extreme()

        assert not result.blocked, "Extreme VIX must not block the guard chain"

    def test_empty_download_fails_open_vix(
        self, mock_account_with_tqqq, bull_signal_live
    ):
        """Empty VIX DataFrame → fails-open (guard skipped, not blocked)."""
        guard = self._make_guard(mock_account_with_tqqq, bull_signal_live)

        with patch("ibkr.safety_guard.yf.download", return_value=_empty_df()):
            result = guard._check_vix_extreme()

        assert not result.blocked

    def test_runtime_error_fails_open_vix(
        self, mock_account_with_tqqq, bull_signal_live
    ):
        """RuntimeError during VIX fetch → swallowed, guard fails-open."""
        guard = self._make_guard(mock_account_with_tqqq, bull_signal_live)

        with patch(
            "ibkr.safety_guard.yf.download",
            side_effect=RuntimeError("timeout"),
        ):
            result = guard._check_vix_extreme()

        assert not result.blocked


# ══════════════════════════════════════════════════════════════════════════════
#  Call site 4: ibkr/position_reconciler.py — get_live_tqqq_price()
# ══════════════════════════════════════════════════════════════════════════════

class TestPositionReconcilerYFinanceCompat:
    """Tests for the yfinance MultiIndex fix in position_reconciler.get_live_tqqq_price()."""

    def _make_reconciler(self, mock_account_with_tqqq):
        from ibkr.position_reconciler import PositionReconciler
        return PositionReconciler(account_state=mock_account_with_tqqq)

    def test_multiindex_close_price_parsed_correctly(self, mock_account_with_tqqq):
        """
        MultiIndex DataFrame must not cause TypeError; get_live_tqqq_price()
        must return the correct float price.
        """
        reconciler = self._make_reconciler(mock_account_with_tqqq)
        multiindex_df = _mock_multiindex_df("TQQQ", close=67.50)

        with patch(
            "ibkr.position_reconciler.yf.download", return_value=multiindex_df
        ):
            price = reconciler.get_live_tqqq_price()

        assert price == pytest.approx(67.50)

    def test_flat_columns_reconciler_still_work(self, mock_account_with_tqqq):
        """Old flat-column format must still return a valid price."""
        reconciler = self._make_reconciler(mock_account_with_tqqq)
        flat_df = _mock_flat_df(price=67.50)

        with patch(
            "ibkr.position_reconciler.yf.download", return_value=flat_df
        ):
            price = reconciler.get_live_tqqq_price()

        assert price == pytest.approx(67.50)

    def test_empty_download_falls_back_to_fill_price(self, mock_account_with_tqqq):
        """
        Empty DataFrame → yfinance path skipped → falls back to last_fill_price
        from state (set to 65.0 by mock_account_with_tqqq fixture).
        """
        reconciler = self._make_reconciler(mock_account_with_tqqq)

        with patch(
            "ibkr.position_reconciler.yf.download", return_value=_empty_df()
        ):
            price = reconciler.get_live_tqqq_price()

        # Fallback to state last_fill_price = 65.0 (set in mock_account_with_tqqq)
        assert price == pytest.approx(65.0)

    def test_runtime_error_falls_back_to_fill_price(self, mock_account_with_tqqq):
        """RuntimeError → yfinance path skipped → fallback to last_fill_price."""
        reconciler = self._make_reconciler(mock_account_with_tqqq)

        with patch(
            "ibkr.position_reconciler.yf.download",
            side_effect=RuntimeError("network error"),
        ):
            price = reconciler.get_live_tqqq_price()

        assert price == pytest.approx(65.0)

    def test_multiindex_used_in_compute_plan(
        self, mock_account_with_tqqq, bull_signal_live
    ):
        """
        End-to-end: compute_plan() with a MultiIndex mock must produce a
        valid RebalancePlan without errors.
        """
        from ibkr.position_reconciler import PositionReconciler
        from config.strategy_config import STRATEGY_A_CONFIG, STRATEGY_B_CONFIG

        mock_account_with_tqqq.net_liquidation = 10_000.0
        mock_account_with_tqqq.positions = {}

        reconciler = PositionReconciler(account_state=mock_account_with_tqqq)
        multiindex_df = _mock_multiindex_df("TQQQ", close=65.0)

        with patch(
            "ibkr.position_reconciler.yf.download", return_value=multiindex_df
        ):
            plan = reconciler.compute_plan(bull_signal_live)

        assert plan.tqqq_price == pytest.approx(65.0)
        target_pct = (
            bull_signal_live["weight_a"] * STRATEGY_A_CONFIG["max_position_pct"]
            + bull_signal_live["weight_b"] * STRATEGY_B_CONFIG["max_position_pct"]
        )
        expected_shares = math.floor(10_000.0 * target_pct / 65.0)
        assert plan.target_shares == expected_shares
