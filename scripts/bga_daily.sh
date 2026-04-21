#!/bin/bash
# Daily BGA scrape + parse. Invoked by ~/Library/LaunchAgents/com.jack.yinsh-bga-daily.plist
# Exits non-zero on setup errors (bad cwd, missing venv, missing cookies); exits 0 on
# cap-hit / network glitches since those are expected and resume on the next run.
#
# To disable: launchctl unload ~/Library/LaunchAgents/com.jack.yinsh-bga-daily.plist
# To remove:  add `rm ~/Library/LaunchAgents/com.jack.yinsh-bga-daily.plist` after unload.

set -u

REPO="/Users/jackfleming/PycharmProjects/YinshML"
LOG_DIR="$REPO/expert_games/bga"
LOG_FILE="$LOG_DIR/daily.log"

cd "$REPO" || { echo "$(date '+%Y-%m-%d %H:%M:%S') [FATAL] cannot cd to $REPO" >&2; exit 2; }
mkdir -p "$LOG_DIR"

{
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') daily run start ====="

    if [[ ! -f "$REPO/venv/bin/python" ]]; then
        echo "[FATAL] venv python missing at $REPO/venv/bin/python"
        exit 2
    fi

    if [[ ! -f "$REPO/.bga_cookies.json" ]]; then
        echo "[FATAL] cookies missing at $REPO/.bga_cookies.json — refresh from browser"
        exit 2
    fi

    "$REPO/venv/bin/python" "$REPO/scripts/bga_fetch.py" --cookies "$REPO/.bga_cookies.json"
    fetch_exit=$?
    echo "[info] bga_fetch.py exit=$fetch_exit"

    "$REPO/venv/bin/python" "$REPO/scripts/bga_parse.py"
    parse_exit=$?
    echo "[info] bga_parse.py exit=$parse_exit"

    raw_count=$(find "$LOG_DIR/raw" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    parsed_count=$(find "$LOG_DIR/parsed" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
    echo "[info] raw=$raw_count parsed=$parsed_count"
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') daily run end ====="
    echo
} >> "$LOG_FILE" 2>&1
