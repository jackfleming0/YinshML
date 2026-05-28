#!/bin/bash
# Pull YINSH game logs from the Mac mini to this laptop.
#
# Intended to be invoked by ~/Library/LaunchAgents/com.jackfleming.yinsh-log-pull.plist
# on a daily timer. Silent on connection failures — the laptop is often
# off-network at the scheduled hour, and we don't want launchd marking it
# as crashed when it's just "Mac mini wasn't reachable today."
#
# Requires SSH key auth from this machine to the Mac mini. Test with:
#   ssh jackfleming@mac-mini.local true
# Should return without prompting for a password.

set -uo pipefail

# Defaults — override via environment if your setup differs.
MAC_MINI_HOST="${YNS_MAC_MINI_HOST:-mac-mini.local}"
MAC_MINI_USER="${YNS_MAC_MINI_USER:-jackfleming}"
REMOTE_DIR="${YNS_MAC_MINI_LOGS_DIR:-/Users/jackfleming/PycharmProjects/YinshML/analysis_board/multiplayer/deploy/games/}"
LOCAL_DIR="${YNS_LOCAL_LOG_DIR:-$HOME/PycharmProjects/YinshML/game_logs/}"

mkdir -p "$LOCAL_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
echo "[$(ts)] pulling ${MAC_MINI_USER}@${MAC_MINI_HOST}:${REMOTE_DIR} → ${LOCAL_DIR}"

# BatchMode=yes makes SSH fail fast if the key isn't loaded (no interactive
# prompt), and ConnectTimeout caps the off-network wait.
rsync -avz --partial \
  -e "ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
  "${MAC_MINI_USER}@${MAC_MINI_HOST}:${REMOTE_DIR}" \
  "$LOCAL_DIR"
RC=$?

case $RC in
  0)
    echo "[$(ts)] sync complete."
    exit 0
    ;;
  12|30|255)
    # 255 = SSH connection failure (off-network / mini off), 30 = timeout,
    # 12 = rsync protocol error. All "expected sometimes" failure modes.
    echo "[$(ts)] Mac mini unreachable (rsync rc=$RC) — skipping today."
    exit 0
    ;;
  *)
    echo "[$(ts)] rsync failed unexpectedly with rc=$RC"
    exit $RC
    ;;
esac
