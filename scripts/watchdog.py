"""
watchdog.py — alert if a scheduled trading run is missing or stale.

Detects the failure mode that today's late/failed run represents: a day where
the signal or paper-trade run did not commit its result. Compares the freshest
committed run date to the expected trading day (ET).

Run it AFTER the expected execution window:
  • On the VPS (reliable timing): fire ~4:30 PM ET → same-day detection.
  • On GitHub Actions (best-effort, may itself be delayed): fire late evening;
    it tolerates lateness by only alerting when a run is a full day stale.

Exit 0 = healthy, 1 = alerted. Emails via send_email if configured.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

ET   = pytz.timezone("America/New_York")
LOGS = Path(__file__).resolve().parent.parent / "logs"


def _last_date(csv: str, col: str):
    try:
        df = pd.read_csv(LOGS / csv)
        vals = df[col].dropna().astype(str)
        return vals.iloc[-1][:10] if len(vals) else None
    except Exception:
        return None


def check(grace_days: int = 1) -> tuple[bool, str]:
    """Return (healthy, message). grace_days=1 tolerates a same-day late run
    (only alerts once a run is a full trading day behind); set 0 on the VPS
    for strict same-day detection."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return True, "weekend — no run expected"

    today = now.date()
    threshold = today - timedelta(days=grace_days)
    problems = []
    for csv, col, label in [("signal_history.csv", "as_of_date", "daily-signal"),
                            ("paper_trades.csv",   "date",       "paper-trade")]:
        last = _last_date(csv, col)
        if last is None:
            problems.append(f"{label}: no data found")
        else:
            last_d = datetime.strptime(last, "%Y-%m-%d").date()
            if last_d < threshold:
                problems.append(f"{label}: last run {last} (≥{grace_days}d stale, today {today})")

    if problems:
        return False, "WATCHDOG — a scheduled run appears MISSING/stale:\n  " + "\n  ".join(problems)
    return True, f"watchdog OK — runs current as of {today}"


def main() -> int:
    grace = 0 if "--strict" in sys.argv else 1
    healthy, msg = check(grace_days=grace)
    print(msg)
    if not healthy:
        try:
            import send_email
            send_email.send_email(subject="[BOT] ⚠ Missed/stale scheduled run", body=msg)
        except Exception as exc:
            print(f"(watchdog email failed: {exc})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
