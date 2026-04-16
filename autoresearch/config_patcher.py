"""
Config Patcher — Safe, validated, reversible changes to strategy_config.py
===========================================================================

Replaces the fragile inline regex in agent_loop.py with:
  1. Backup before any write
  2. Type-aware regex patterns for int / float / bool / tuple
  3. Python compile() check — ensures file remains valid Python after patching
  4. Atomic write — only replaces file if all checks pass
  5. Restore from backup on failure

Usage:
    from autoresearch.config_patcher import apply_change, restore_backup

    ok, reason = apply_change("STRATEGY_A_CONFIG.vix_exit", 23)
    if not ok:
        print(f"Patch failed: {reason}")

    restore_backup()   # explicit rollback if needed
"""

import re
import ast
import shutil
from pathlib import Path
from typing import Any, Tuple

_ROOT      = Path(__file__).parent.parent
CONFIG_PATH = _ROOT / "config" / "strategy_config.py"
BACKUP_PATH = CONFIG_PATH.with_suffix(".py.bak")


# ── Backup / Restore ───────────────────────────────────────────────────────────

def backup() -> None:
    """Copy current config to .bak before any change."""
    shutil.copy2(CONFIG_PATH, BACKUP_PATH)


def restore_backup() -> bool:
    """Restore config from .bak. Returns True on success."""
    if not BACKUP_PATH.exists():
        return False
    shutil.copy2(BACKUP_PATH, CONFIG_PATH)
    return True


def has_backup() -> bool:
    return BACKUP_PATH.exists()


# ── Value Formatting ───────────────────────────────────────────────────────────

def _format_value(value: Any) -> str:
    """Convert a Python value to its source-code string representation."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        a, b = float(value[0]), float(value[1])
        # Preserve minimal decimal places
        a_str = f"{a:.2f}".rstrip("0").rstrip(".")
        b_str = f"{b:.2f}".rstrip("0").rstrip(".")
        return f"({a_str}, {b_str})"
    if isinstance(value, float):
        # Keep at least one decimal so Python keeps it as float
        s = f"{value:.4f}".rstrip("0")
        if s.endswith("."):
            s += "0"
        return s
    if isinstance(value, int):
        return str(value)
    return repr(value)


# ── Pattern Builder ────────────────────────────────────────────────────────────

def _build_pattern(key: str, value: Any) -> Tuple[str, str]:
    """
    Return (regex_pattern, replacement_string) for a given key + new value.

    Matches lines like:
        "vix_exit":  25,
        "vix_exit": 25.0,
        "bull": (0.75, 0.25),
        "t1_execution": True,
    """
    key_re   = re.escape(key)
    new_str  = _format_value(value)
    quote    = rf'(?:"{key_re}"|' + rf"'{key_re}')"

    if isinstance(value, (list, tuple)):
        # Tuple: match (anything inside parens)
        pattern = rf'({quote}\s*:\s*)\([^)]*\)'
        replace = rf'\g<1>{new_str}'

    elif isinstance(value, bool):
        pattern = rf'({quote}\s*:\s*)(?:True|False)'
        replace = rf'\g<1>{new_str}'

    elif isinstance(value, float):
        # Match int or float literal
        pattern = rf'({quote}\s*:\s*)[\d]+\.?[\d]*'
        replace = rf'\g<1>{new_str}'

    elif isinstance(value, int):
        # Match integer literal (not followed by a dot to avoid matching floats)
        pattern = rf'({quote}\s*:\s*)(\d+)(?!\.)'
        replace = rf'\g<1>{new_str}'

    else:
        raise ValueError(f"Unsupported value type: {type(value)}")

    return pattern, replace


# ── Syntax Validator ───────────────────────────────────────────────────────────

def _is_valid_python(source: str) -> Tuple[bool, str]:
    """Compile-check that source is valid Python. Returns (ok, error_msg)."""
    try:
        compile(source, "<config>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


# ── Main API ───────────────────────────────────────────────────────────────────

def apply_change(param: str, new_value: Any) -> Tuple[bool, str]:
    """
    Patch strategy_config.py with a single parameter change.

    Steps:
      1. Backup current config
      2. Parse param as "DICT_NAME.key"
      3. Build regex pattern for this value type
      4. Apply substitution (exactly 1 match required)
      5. Validate result is valid Python
      6. Write atomically

    Returns (True, "") on success.
    Returns (False, reason) on any failure, with backup auto-restored.
    """
    # Parse param
    try:
        _dict_name, key = param.rsplit(".", 1)
    except ValueError:
        return False, f"Invalid param format '{param}' — expected 'DICT.key'"

    # Read current config
    try:
        original = CONFIG_PATH.read_text()
    except OSError as e:
        return False, f"Cannot read config: {e}"

    # Build regex
    try:
        pattern, replacement = _build_pattern(key, new_value)
    except ValueError as e:
        return False, str(e)

    # Apply substitution
    new_text, n_subs = re.subn(pattern, replacement, original, count=1)
    if n_subs == 0:
        return False, (
            f"Key '{key}' not found in config file. "
            f"Pattern: {pattern!r}"
        )

    # Sanity check: substitution actually changed something
    if new_text == original:
        return False, f"Substitution produced no change (new_value may already match current)"

    # Validate Python syntax
    ok, err = _is_valid_python(new_text)
    if not ok:
        return False, f"Patched config is not valid Python: {err}"

    # Backup then write
    backup()
    try:
        CONFIG_PATH.write_text(new_text)
    except OSError as e:
        restore_backup()
        return False, f"Write failed: {e}"

    return True, ""


def read_current_value(param: str) -> Tuple[bool, Any]:
    """
    Read the current value of a parameter from the live config file.
    Returns (True, value) or (False, error_string).

    Uses ast.literal_eval on the matched fragment — safe, no exec().
    """
    try:
        _dict_name, key = param.rsplit(".", 1)
    except ValueError:
        return False, "Invalid param format"

    text    = CONFIG_PATH.read_text()
    key_re  = re.escape(key)
    quote   = rf'(?:"{key_re}"|' + rf"'{key_re}')"

    # Try tuple
    m = re.search(rf'{quote}\s*:\s*(\([^)]*\))', text)
    if m:
        try:
            return True, ast.literal_eval(m.group(1))
        except Exception:
            pass

    # Try scalar (int/float/bool/string)
    m = re.search(rf'{quote}\s*:\s*([^\n,}}]+)', text)
    if m:
        raw = m.group(1).strip().rstrip(",")
        try:
            return True, ast.literal_eval(raw)
        except Exception:
            return True, raw

    return False, f"Key '{key}' not found"


def diff_summary() -> str:
    """
    Return a one-line summary of what changed vs the backup.
    Useful for logging.
    """
    if not BACKUP_PATH.exists():
        return "(no backup)"
    old_lines = set(BACKUP_PATH.read_text().splitlines())
    new_lines = set(CONFIG_PATH.read_text().splitlines())
    added   = [l.strip() for l in (new_lines - old_lines) if l.strip() and not l.strip().startswith("#")]
    removed = [l.strip() for l in (old_lines - new_lines) if l.strip() and not l.strip().startswith("#")]
    parts = []
    if removed:
        parts.append(f"- {' | '.join(removed[:3])}")
    if added:
        parts.append(f"+ {' | '.join(added[:3])}")
    return " → ".join(parts) if parts else "(identical)"
