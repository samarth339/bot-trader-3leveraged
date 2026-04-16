"""
kill_switch.py — Hard Stop for Live Trading
=============================================
The kill switch is a plain file flag: logs/ibkr_kill.flag

  - Presence of the file = ALL order submission is BLOCKED
  - It is NEVER removed automatically — human must delete it manually
  - It can be activated by:
      - safety_guard.py  (drawdown breach, critical error)
      - executor.py      (repeated order failures)
      - shadow_mode.py   (VIX > 45 extreme event)
      - User manually    (touch logs/ibkr_kill.flag)

To resume trading after fixing the underlying issue:
    python -m ibkr.kill_switch deactivate   (or delete the file manually)
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("ibkr.kill_switch")

KILL_SWITCH_PATH = Path("logs/ibkr_kill.flag")


# ── Core API ───────────────────────────────────────────────────────────────────

def is_active() -> bool:
    """Return True if kill switch is currently active."""
    return KILL_SWITCH_PATH.exists()


def activate(reason: str = "manually activated"):
    """
    Activate the kill switch. Writes reason + timestamp to the flag file.
    Logs CRITICAL — this will show up in all log aggregators.
    """
    KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    content   = f"ACTIVATED: {timestamp}\nReason: {reason}\n"
    KILL_SWITCH_PATH.write_text(content)
    logger.critical(f"KILL SWITCH ACTIVATED — {reason}")


def deactivate():
    """
    Deactivate the kill switch. Call explicitly after investigating the issue.
    Never call this automatically from reconnect or retry logic.
    """
    if KILL_SWITCH_PATH.exists():
        KILL_SWITCH_PATH.unlink()
        logger.warning(
            "Kill switch DEACTIVATED by operator. "
            "Ensure root cause is resolved before next execution."
        )
    else:
        logger.info("Kill switch was not active")


def read_reason() -> str:
    """Return the reason written when the switch was activated, or empty string."""
    if KILL_SWITCH_PATH.exists():
        return KILL_SWITCH_PATH.read_text().strip()
    return ""


def status() -> dict:
    return {
        "active": is_active(),
        "path":   str(KILL_SWITCH_PATH.resolve()),
        "reason": read_reason() if is_active() else "",
    }


# ── CLI interface ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        s = status()
        print(f"Kill switch: {'ACTIVE ⛔' if s['active'] else 'OFF ✓'}")
        if s["reason"]:
            print(f"Reason:\n{s['reason']}")

    elif cmd == "activate":
        reason = " ".join(sys.argv[2:]) or "manual activation via CLI"
        activate(reason)
        print(f"Kill switch activated: {reason}")

    elif cmd == "deactivate":
        deactivate()
        print("Kill switch deactivated")

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python -m ibkr.kill_switch [status|activate|deactivate]")
        sys.exit(1)
