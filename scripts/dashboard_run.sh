#!/usr/bin/env bash
# dashboard_run.sh — Launch the TQQQ trading dashboard.
# Usage:  bash scripts/dashboard_run.sh [--port 8050]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

PORT="${2:-8050}"
if [ "${1:-}" = "--port" ]; then
  PORT="${2:-8050}"
fi

echo ""
echo "══════════════════════════════════════════"
echo "  TQQQ Bot Dashboard"
echo "══════════════════════════════════════════"
echo ""
echo "  URL:  http://127.0.0.1:${PORT}"
echo "  Stop: Ctrl+C"
echo ""

# Optional: refresh data before launching
if [ "${REFRESH:-0}" = "1" ]; then
  echo "  Refreshing market data first..."
  python3 data/fetch_data.py
  echo ""
fi

exec python3 -m dashboard.app
