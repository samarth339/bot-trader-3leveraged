#!/usr/bin/env bash
# phase4_setup_launchd.sh — Install a macOS LaunchAgent that runs phase4_run.sh
# at 3:45 PM Eastern time (Mon–Fri).
#
# 3:45 PM ET is the executor's submission window open time. The script pipeline
# (git pull + data fetch + IB sync) takes ~25s, so the MOC order hits at ~3:45:25 —
# well inside the 3:45–3:50 MOC window.
#
# Detects your machine's timezone and converts 3:45 PM ET automatically,
# so the plist fires at the correct local time regardless of where you are.
#
# Usage:  bash scripts/phase4_setup_launchd.sh
#         bash scripts/phase4_setup_launchd.sh --uninstall
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_LABEL="com.tradingbot.phase4"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"
RUN_SCRIPT="$REPO_ROOT/scripts/phase4_run.sh"

# ── Uninstall ──────────────────────────────────────────────────────────────────
if [ "${1:-}" = "--uninstall" ]; then
  launchctl unload "$PLIST_PATH" 2>/dev/null && echo "LaunchAgent unloaded." || true
  rm -f "$PLIST_PATH" && echo "Plist removed: $PLIST_PATH" || true
  echo "Phase 4 LaunchAgent uninstalled."
  exit 0
fi

# ── Compute local execution time (ET 15:43 → local) ───────────────────────────
# Uses macOS date command to resolve timezone offsets without needing pytz.
ET_TARGET_HOUR=15
ET_TARGET_MIN=45

ET_OFFSET_STR=$(TZ=America/New_York date +%z)    # e.g. -0400 or -0500
LOCAL_OFFSET_STR=$(date +%z)                     # e.g. -0700

# Parse offset strings into total minutes (e.g. -0400 → -240)
offset_to_mins() {
  local s="$1"
  local sign=1
  [[ "${s:0:1}" == "-" ]] && sign=-1
  local h="${s:1:2}"; local m="${s:3:2}"
  echo $(( sign * (10#$h * 60 + 10#$m) ))
}

ET_MINS=$(offset_to_mins "$ET_OFFSET_STR")
LOCAL_MINS=$(offset_to_mins "$LOCAL_OFFSET_STR")
DIFF=$(( LOCAL_MINS - ET_MINS ))

TARGET_ET=$(( ET_TARGET_HOUR * 60 + ET_TARGET_MIN ))
TARGET_LOCAL=$(( (TARGET_ET + DIFF + 1440) % 1440 ))
LOCAL_HOUR=$(( TARGET_LOCAL / 60 ))
LOCAL_MIN=$(( TARGET_LOCAL % 60 ))

LOCAL_TZ=$(date +%Z)

echo ""
echo "══════════════════════════════════════════════"
echo "  Phase 4 LaunchAgent Setup"
echo "══════════════════════════════════════════════"
echo ""
echo "  Execution target: 3:45 PM ET (NYSE MOC window: 3:45–3:50 PM)"
echo "  Your timezone:    $LOCAL_TZ  ($LOCAL_OFFSET_STR)"
echo "  Scheduled at:     $(printf '%02d:%02d' $LOCAL_HOUR $LOCAL_MIN) local"
echo ""

# ── Write plist ────────────────────────────────────────────────────────────────
mkdir -p "$HOME/Library/LaunchAgents"

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
    "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>

    <key>Label</key>
    <string>${PLIST_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${RUN_SCRIPT}</string>
    </array>

    <!-- Mon–Fri at ${LOCAL_HOUR}:$(printf '%02d' $LOCAL_MIN) local (= 3:45 PM ET) -->
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>${LOCAL_HOUR}</integer><key>Minute</key><integer>${LOCAL_MIN}</integer></dict>
        <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>${LOCAL_HOUR}</integer><key>Minute</key><integer>${LOCAL_MIN}</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>${LOCAL_HOUR}</integer><key>Minute</key><integer>${LOCAL_MIN}</integer></dict>
        <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>${LOCAL_HOUR}</integer><key>Minute</key><integer>${LOCAL_MIN}</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>${LOCAL_HOUR}</integer><key>Minute</key><integer>${LOCAL_MIN}</integer></dict>
    </array>

    <key>WorkingDirectory</key>
    <string>${REPO_ROOT}</string>

    <key>StandardOutPath</key>
    <string>${REPO_ROOT}/logs/launchd_phase4.log</string>

    <key>StandardErrorPath</key>
    <string>${REPO_ROOT}/logs/launchd_phase4_err.log</string>

    <!-- Inherit PATH from pyenv so python3 resolves correctly -->
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/Users/$(whoami)/.pyenv/shims:/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
        <key>PYTHONPATH</key>
        <string>${REPO_ROOT}</string>
    </dict>

    <!-- Do not run immediately on load — wait for the scheduled time -->
    <key>RunAtLoad</key>
    <false/>

</dict>
</plist>
PLIST

# ── Load (or reload) the agent ─────────────────────────────────────────────────
launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl load  "$PLIST_PATH"

echo "  LaunchAgent installed and loaded."
echo "  Plist: $PLIST_PATH"
echo ""
echo "  Useful commands:"
echo "    Check status:  launchctl list $PLIST_LABEL"
echo "    Trigger now:   launchctl start $PLIST_LABEL"
echo "    View log:      tail -f $REPO_ROOT/logs/phase4_run.log"
echo "    Uninstall:     bash scripts/phase4_setup_launchd.sh --uninstall"
echo ""
echo "  Note: your Mac must be awake and IB Gateway running at $(printf '%02d:%02d' $LOCAL_HOUR $LOCAL_MIN) $LOCAL_TZ."
echo "  launchd will NOT catch up on missed fires if the machine was asleep."
echo ""
