#!/bin/bash
# Quick-start redeploy for the YINSH analysis board on the Mac mini.
#
# Pulls latest code, installs the LaunchAgent plists from this repo into
# ~/Library/LaunchAgents/ (overwriting whatever's there), kills any rogue
# port-5173 squatter that would block launchd, restarts both services,
# and verifies the server is actually responding.
#
# Idempotent — safe to run whenever you don't know the current state.
#
# Usage:
#   bash /Users/jackfleming/PycharmProjects/YinshML/analysis_board/multiplayer/deploy/redeploy.sh
#
# Or set up an alias once:
#   alias yinsh-redeploy='bash /Users/jackfleming/PycharmProjects/YinshML/analysis_board/multiplayer/deploy/redeploy.sh'
# Then just:  yinsh-redeploy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LAUNCH_DIR="$HOME/Library/LaunchAgents"
SERVER_PLIST="com.jackfleming.yinsh-server.plist"
TUNNEL_PLIST="com.jackfleming.yinsh-tunnel.plist"
PORT=5173

ts() { date +"%H:%M:%S"; }

echo "==> [1/6] [$(ts)] Pulling latest code from origin..."
cd "$REPO_ROOT"
git pull --ff-only
echo

echo "==> [2/6] [$(ts)] Installing plists into $LAUNCH_DIR/..."
cp "$SCRIPT_DIR/$SERVER_PLIST" "$LAUNCH_DIR/"
cp "$SCRIPT_DIR/$TUNNEL_PLIST" "$LAUNCH_DIR/"
echo "    server + tunnel plists copied"

# Overlay secrets from ~/.yinsh.env into the installed server plist. The
# repo plist has empty <string></string> placeholders for any secret env
# vars; this step writes the real values after the copy, so secrets never
# live in git. ~/.yinsh.env format: one `export KEY=value` per line, file
# permissions tight (chmod 600). Add new secret names to the SECRET_VARS
# list below when /api/* endpoints need them.
ENV_FILE="$HOME/.yinsh.env"
SECRET_VARS=()
if [ -f "$ENV_FILE" ]; then
  echo "    overlaying secrets from $ENV_FILE"
  # shellcheck source=/dev/null
  set -a; source "$ENV_FILE"; set +a
  for VAR in "${SECRET_VARS[@]}"; do
    VAL="${!VAR-}"
    if [ -n "$VAL" ]; then
      if /usr/libexec/PlistBuddy \
          -c "Set :EnvironmentVariables:$VAR $VAL" \
          "$LAUNCH_DIR/$SERVER_PLIST" 2>/dev/null; then
        echo "    injected $VAR"
      else
        # Placeholder didn't exist — Add it instead. Lets new secrets
        # land even if the repo plist hasn't shipped the placeholder yet.
        /usr/libexec/PlistBuddy \
          -c "Add :EnvironmentVariables:$VAR string $VAL" \
          "$LAUNCH_DIR/$SERVER_PLIST" \
          && echo "    added $VAR (no placeholder in repo plist)"
      fi
    fi
  done
else
  echo "    no $ENV_FILE found — no secrets to overlay (SECRET_VARS is empty by default)"
fi
echo

echo "==> [3/6] [$(ts)] Unloading LaunchAgents (if loaded)..."
if launchctl unload "$LAUNCH_DIR/$SERVER_PLIST" 2>/dev/null; then
  echo "    server: unloaded"
else
  echo "    server: was not loaded"
fi
if launchctl unload "$LAUNCH_DIR/$TUNNEL_PLIST" 2>/dev/null; then
  echo "    tunnel: unloaded"
else
  echo "    tunnel: was not loaded"
fi
echo

echo "==> [4/6] [$(ts)] Clearing port $PORT (killing any squatters)..."
ROGUE_PIDS="$(lsof -ti :$PORT 2>/dev/null || true)"
if [ -n "$ROGUE_PIDS" ]; then
  echo "    SIGTERM to: $ROGUE_PIDS"
  kill $ROGUE_PIDS 2>/dev/null || true
  sleep 2
  STILL="$(lsof -ti :$PORT 2>/dev/null || true)"
  if [ -n "$STILL" ]; then
    echo "    SIGTERM ignored, SIGKILL to: $STILL"
    kill -9 $STILL 2>/dev/null || true
    sleep 1
  fi
  echo "    port $PORT clear"
else
  echo "    port $PORT already clear"
fi
echo

echo "==> [5/6] [$(ts)] Loading LaunchAgents..."
launchctl load "$LAUNCH_DIR/$SERVER_PLIST"
launchctl load "$LAUNCH_DIR/$TUNNEL_PLIST"
echo "    both loaded"
echo

echo "==> [6/6] [$(ts)] Verifying (polling HTTP up to 60s for model load + bind)..."
SERVER_LINE="$(launchctl list | grep yinsh-server || echo 'MISSING')"
TUNNEL_LINE="$(launchctl list | grep yinsh-tunnel || echo 'MISSING')"

CAP="$(grep -A 1 YNS_MAX_NUM_SIMS "$LAUNCH_DIR/$SERVER_PLIST" 2>/dev/null | grep '<string>' | sed -E 's/.*<string>([0-9]+)<\/string>.*/\1/' || echo '?')"

echo "    server agent: $SERVER_LINE"
echo "    tunnel agent: $TUNNEL_LINE"

# Poll for up to ~60s — PyTorch model load into MPS routinely takes 5-15s
# from cold, and a full reboot cold start (Torch + coremltools import +
# multiple checkpoints) can push past 30s. Exits early as soon as HTTP responds.
HTTP_OK=0
for i in $(seq 1 30); do
  if curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/api/models"; then
    HTTP_OK=1
    echo "    HTTP OK after ~${i}*2s"
    break
  fi
  sleep 2
done

echo
echo "============================================"
echo "Summary"
echo "============================================"
if [ "$HTTP_OK" = "1" ]; then
  echo "  [OK] Server responding on http://127.0.0.1:$PORT"
else
  echo "  [FAIL] Server NOT responding on http://127.0.0.1:$PORT"
  echo "         Check: tail $SCRIPT_DIR/yinsh-server.err.log"
fi
echo "  MCTS sim cap = $CAP"
echo
echo "  Public URL: https://yinsh.jackflemingux.com"
echo "  Browser:    hard-refresh (Cmd+Shift+R) to pick up new static assets"
echo "============================================"
