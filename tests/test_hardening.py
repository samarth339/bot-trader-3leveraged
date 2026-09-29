"""Tests for the Phase-A hardening (2026-09-28): price validation, plan sanity
gate, and the missed-run watchdog."""
import math
import pytest


class TestPriceValidation:
    def test_is_valid_price(self):
        from ibkr.pricing import is_valid_price
        for bad in (float("nan"), float("inf"), -float("inf"), 0, -1, "", None, "abc"):
            assert not is_valid_price(bad), bad
        for ok in (0.01, 84.5, 100000):
            assert is_valid_price(ok)


class TestPlanSanityGate:
    def _pf(self):
        return {"tqqq_shares": 100, "cash": 1000.0, "nlv": 8000.0}

    def test_nan_price_refuses(self):
        from paper_trade import compute_plan
        plan = compute_plan({"weight_a": 0.9, "weight_b": 0.1}, self._pf(), float("nan"))
        assert plan["proceed"] is False and plan["delta_shares"] == 0

    def test_zero_price_refuses(self):
        from paper_trade import compute_plan
        plan = compute_plan({"weight_a": 0.9, "weight_b": 0.1}, self._pf(), 0.0)
        assert plan["proceed"] is False

    def test_out_of_bounds_target_refuses(self, monkeypatch):
        import paper_trade
        monkeypatch.setattr(paper_trade, "compute_target_pct", lambda s: 5.0)  # 500% — impossible
        plan = paper_trade.compute_plan({"weight_a": 0.9, "weight_b": 0.1}, self._pf(), 80.0)
        assert plan["proceed"] is False and "out of bounds" in plan["reason"]

    def test_valid_price_proceeds_normally(self):
        from paper_trade import compute_plan
        # flat account, valid price, bull exposure → should produce a BUY plan
        plan = compute_plan({"weight_a": 0.9, "weight_b": 0.1,
                             "exposure_a": 0.9, "exposure_b": 0.2},
                            {"tqqq_shares": 0, "cash": 10000.0, "nlv": 10000.0}, 80.0)
        assert plan["proceed"] is True and plan["delta_shares"] > 0


class TestWatchdog:
    def test_stale_run_flagged(self, monkeypatch):
        import scripts.watchdog as w
        monkeypatch.setattr(w, "_last_date", lambda csv, col: "2020-01-01")
        # force a weekday "now"
        import datetime as _dt
        class _Now(_dt.datetime):
            @classmethod
            def now(cls, tz=None): return cls(2026, 9, 28, 17, 0, tzinfo=tz)  # Monday
        monkeypatch.setattr(w, "datetime", _Now)
        healthy, msg = w.check(grace_days=1)
        assert healthy is False and "stale" in msg.lower()

    def test_current_run_ok(self, monkeypatch):
        import scripts.watchdog as w, datetime as _dt
        monkeypatch.setattr(w, "_last_date", lambda csv, col: "2026-09-28")
        class _Now(_dt.datetime):
            @classmethod
            def now(cls, tz=None): return cls(2026, 9, 28, 17, 0, tzinfo=tz)
        monkeypatch.setattr(w, "datetime", _Now)
        healthy, _ = w.check(grace_days=0)
        assert healthy is True
