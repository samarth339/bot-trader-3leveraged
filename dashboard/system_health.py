"""
system_health.py — data-feed / pipeline / config health + recent log tail.

All checks are FILE-BASED (no network) so the dashboard stays fast and offline-
safe. "Broker connection" is intentionally reported as simulation — the bot does
not connect to IBKR.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd


def _age_days(datestr: str) -> int | None:
    try:
        return (date.today() - datetime.strptime(str(datestr)[:10], "%Y-%m-%d").date()).days
    except Exception:
        return None


def check_health(root: Path) -> dict:
    logs = root / "logs"
    data = root / "data" / "processed"
    checks = []

    # ── Data feed freshness ──────────────────────────────────────────────────
    try:
        qqq = pd.read_csv(data / "QQQ_full.csv", index_col=0, parse_dates=True)
        age = _age_days(str(qqq.index[-1].date()))
        ok = age is not None and age <= 5
        checks.append(("Market data (QQQ)", "OK" if ok else "STALE",
                       f"last bar {qqq.index[-1].date()} ({age}d ago)", ok))
    except Exception as e:
        checks.append(("Market data (QQQ)", "ERROR", str(e)[:40], False))

    # ── Signal freshness + exposure-fallback tripwire ────────────────────────
    try:
        sig = pd.read_csv(logs / "signal_history.csv")
        last = sig.iloc[-1]
        age = _age_days(last.get("as_of_date"))
        ok = age is not None and age <= 4
        checks.append(("Daily signal", "OK" if ok else "STALE",
                       f"{last.get('as_of_date')} · {last.get('regime')} ({age}d)", ok))
        # exposure fallback: blank exposure_a/b ⇒ executor over-allocates
        exp_a = str(last.get("exposure_a", "")).strip()
        has_exp = exp_a not in ("", "nan", "None")
        checks.append(("Exposure sizing", "OK" if has_exp else "FALLBACK",
                       "exposure-state active" if has_exp
                       else "BLANK → max-cap fallback (over-allocates!)", has_exp))
    except Exception as e:
        checks.append(("Daily signal", "ERROR", str(e)[:40], False))

    # ── Kill switch ──────────────────────────────────────────────────────────
    kill = (logs / "ibkr_kill.flag").exists()
    checks.append(("Kill switch", "ACTIVE" if kill else "OFF",
                   (logs / "ibkr_kill.flag").read_text()[:40] if kill else "trading enabled",
                   not kill))

    # ── Broker connection (always simulation) ────────────────────────────────
    checks.append(("Broker", "SIMULATION", "not connected to IBKR (Phase 4 sim)", True))

    # ── Vol-target overlay state ─────────────────────────────────────────────
    try:
        from config.strategy_config import VOL_TARGET_CONFIG
        on = VOL_TARGET_CONFIG.get("enabled")
        checks.append(("Vol-target overlay", "ON" if on else "OFF",
                       f"target {VOL_TARGET_CONFIG['target_annual_vol']:.0%}" if on
                       else "disabled (baseline)", True))
    except Exception:
        pass

    healthy = all(c[3] for c in checks)
    return {"checks": checks, "healthy": healthy}


def tail_logs(root: Path, n: int = 40) -> list[dict]:
    """Return the most recent log lines across the pipeline logs, newest last,
    tagged with level for colour."""
    logs = root / "logs"
    out = []
    for fname in ("paper_trade.log", "daily_signal.log"):
        p = logs / fname
        if not p.exists():
            continue
        try:
            lines = p.read_text(errors="ignore").splitlines()[-n:]
            for ln in lines:
                lvl = ("ERROR" if "ERROR" in ln else
                       "WARNING" if "WARNING" in ln else "INFO")
                out.append({"file": fname, "level": lvl, "text": ln})
        except Exception:
            continue
    return out[-n:]
