# Deploy runbook — Mac mini host

Operations docs for the YINSH analysis board running on the Mac mini, fronted
by Cloudflare Tunnel at `https://yinsh.jackflemingux.com`. Public access — no
auth layer.

## What's in here

| File | Purpose |
|---|---|
| `cloudflared.yml` | Tunnel ingress config (hostname → localhost:5173) |
| `com.jackfleming.yinsh-server.plist` | LaunchAgent for the Flask server |
| `com.jackfleming.yinsh-tunnel.plist` | LaunchAgent for the cloudflared daemon |
| `*.log` / `*.err.log` | Runtime output (gitignored) |

## One-time setup

Run on the Mac mini. Assumes repo is at `/Users/jackfleming/PycharmProjects/YinshML`
and venv is at `venv/` under it; edit the plists if either differs.

### 1. Install cloudflared

```bash
brew install cloudflared
which cloudflared   # should be /opt/homebrew/bin/cloudflared (Apple Silicon)
```

If `which` returns something other than `/opt/homebrew/bin/cloudflared`, update
the `ProgramArguments` path in `com.jackfleming.yinsh-tunnel.plist`.

### 2. Authenticate with Cloudflare

```bash
cloudflared tunnel login
```

Opens a browser. Sign in to Cloudflare and authorize the `jackflemingux.com`
zone. Writes `~/.cloudflared/cert.pem`.

### 3. Create the tunnel

```bash
cloudflared tunnel create yinsh
```

Prints a UUID and writes credentials to `~/.cloudflared/<UUID>.json`. Copy the
UUID — you'll need it next.

### 4. Wire the UUID into `cloudflared.yml`

```bash
cd /Users/jackfleming/PycharmProjects/YinshML/analysis_board/multiplayer/deploy
# Replace both occurrences of <TUNNEL_UUID> with the UUID from step 3.
# (Or use sed: sed -i '' "s/<TUNNEL_UUID>/<UUID>/g" cloudflared.yml)
```

### 5. Point DNS at the tunnel

```bash
cloudflared tunnel route dns yinsh yinsh.jackflemingux.com
```

Creates a CNAME at `yinsh.jackflemingux.com → <UUID>.cfargotunnel.com`. Verify
in the Cloudflare dashboard → jackflemingux.com → DNS that the record exists
and is "proxied" (orange cloud).

### 6. Test the tunnel in the foreground

In one terminal start the Flask server:

```bash
cd /Users/jackfleming/PycharmProjects/YinshML
source venv/bin/activate
YNS_MAX_NUM_SIMS=1600 python analysis_board/server.py
```

In a second terminal start the tunnel:

```bash
cloudflared tunnel --config /Users/jackfleming/PycharmProjects/YinshML/analysis_board/multiplayer/deploy/cloudflared.yml run
```

From a phone on cell data (or any off-network device), visit
`https://yinsh.jackflemingux.com` — should load the analysis board. Play one
move to confirm `/api/move` round-trips through the tunnel.

Ctrl+C both processes when you've confirmed it works.

### 7. Install the LaunchAgents

```bash
cp /Users/jackfleming/PycharmProjects/YinshML/analysis_board/multiplayer/deploy/com.jackfleming.yinsh-server.plist ~/Library/LaunchAgents/
cp /Users/jackfleming/PycharmProjects/YinshML/analysis_board/multiplayer/deploy/com.jackfleming.yinsh-tunnel.plist ~/Library/LaunchAgents/

launchctl load ~/Library/LaunchAgents/com.jackfleming.yinsh-server.plist
launchctl load ~/Library/LaunchAgents/com.jackfleming.yinsh-tunnel.plist
```

Both processes start immediately and restart automatically on crash
(`KeepAlive=true` in each plist). For them to also come back after a Mac
mini reboot — see [Reboot survival](#reboot-survival) below.

### 8. Verify end-to-end

```bash
launchctl list | grep yinsh
# com.jackfleming.yinsh-server   <PID>  0
# com.jackfleming.yinsh-tunnel   <PID>  0
```

Exit code `0` = healthy. A nonzero exit code means the process crashed
recently — check the `.err.log` next to the plist.

Then: load `https://yinsh.jackflemingux.com` from a phone again. Should
respond just like the foreground test.

## Reboot survival

LaunchAgents load when a **user logs in**, not when the Mac boots. So
without these two settings, a power outage or OS-update reboot leaves
the Mac mini sitting at the login screen with the site offline until
someone physically logs in.

### Auto-login

System Settings → Users & Groups → click the **(i)** next to your user
→ enable **Automatically log in as**. Enter your password to confirm.

After any reboot the mini logs you in automatically, the LaunchAgents
fire, and the site comes back in ~90 seconds.

Caveat: if FileVault is enabled, this option is grayed out (FileVault
needs the disk-unlock password before any user can log in). For a
headless server, the usual tradeoff is FileVault off. If you want
FileVault, the only "hands-off" path is to give up on auto-login and
plan for manual recovery after every reboot.

### Disable idle sleep

```bash
sudo pmset -a sleep 0
```

Sets the idle-system-sleep timer to 0 (never). Display sleep is
independent — your screen can still go dark, the OS keeps running.

Verify with:

```bash
pmset -g | grep -E "^ *(sleep|displaysleep)"
```

`sleep` should report `0`. `displaysleep` can be whatever.

### What this covers

- Power outage → mini boots → auto-login → LaunchAgents load → site
  is back. ~90s downtime.
- macOS auto-update reboot → same path.
- Idle desktop → no sleep, site stays reachable indefinitely.

### What still requires you

- Pulling new code: see "Pull and redeploy" below.
- Replacing model checkpoints: drop new `.pt` into `models/<name>/`
  and reload the server LaunchAgent.

## BGA cookies for Review mode (optional)

The "Review game" mode on the analysis board lets visitors paste a
[boardgamearena.com](https://boardgamearena.com) YINSH replay URL and step
through it ply by ply with engine commentary. This requires a BGA session
on the **server** to authenticate the replay fetch — there is no public BGA
API, and BGA's replay endpoints reject anonymous requests. Cookies stay
server-side and are never exposed to the frontend.

Skipping this section is fine: Review mode still loads, but pasting a URL
will return a friendly "BGA cookies missing or expired" error. Setup, Play,
and Play-vs-Engine all work unaffected.

### One-time, on the Mac mini

1. **Log into boardgamearena.com in a browser** (any account: must be
   verified, >24h old, and have played at least 2 games — BGA's own
   requirements for replay access).
2. **Export the session cookies** to `~/.bga_cookies.json` using a browser
   extension like
   [Cookie-Editor](https://chrome.google.com/webstore/detail/cookie-editor/hlkenndednhfkekhgcdicdfddnkalmdm)
   or [EditThisCookie](https://www.editthiscookie.com). The file must be a
   JSON object mapping cookie name → value. **Required cookies** (export
   all of these — fewer will cause `/archive` requests to fail with "not
   logged in"):
   ```
   PHPSESSID
   TournoiEnLigne_sso_id
   TournoiEnLigne_sso_user
   TournoiEnLigneid
   TournoiEnLigneidt
   TournoiEnLignetk
   TournoiEnLignetkt
   ```
3. **Tighten permissions** — this is an active session credential:
   ```bash
   chmod 600 ~/.bga_cookies.json
   ```
4. **Restart the server** so it picks up the cookies on next import:
   ```bash
   yinsh-redeploy
   ```

### Path override

The server defaults to `~/.bga_cookies.json`. Override via the
`YNS_BGA_COOKIES` env var if your setup needs a different location:

```xml
<!-- in com.jackfleming.yinsh-server.plist, inside <dict>EnvironmentVariables</dict> -->
<key>YNS_BGA_COOKIES</key>
<string>/path/to/cookies.json</string>
```

The default path matches what `scripts/bga_fetch.py` (the bulk crawler)
uses, so a single cookie export works for both tools.

### When the session expires

BGA rotates session credentials periodically. When the server starts
returning "BGA cookies missing or expired" on every import, re-export from
the browser and repeat step 2. There is no automatic refresh — BGA's login
flow is JS-rendered and resists scripting.

### Rate limits

The endpoint caches every successful import under
`analysis_board/multiplayer/bga_imports/` (last 1000 games, LRU) so repeat
visits to the same game cost nothing. Novel imports are rate-limited to 5
per IP per hour. Both bounds protect BGA's 200/day per-account cap from
being exhausted by a single visitor.

## Laptop-side log pull (separate from Mac mini setup)

The Mac mini server writes one JSONL file per UTC day to `games/`
(gitignored — private). To pull them to your laptop for offline analysis
without thinking about it, install the third LaunchAgent on your
**laptop** (NOT the Mac mini):

```bash
# On the laptop — first verify SSH key auth to the Mac mini works
ssh jackfleming@mac-mini.local true   # should return silently, no password

# If the above prompts for a password, set up key auth first:
ssh-keygen -t ed25519   # if you don't have a key yet
ssh-copy-id jackfleming@mac-mini.local

# Install the pull agent
cp /Users/jackfleming/PycharmProjects/YinshML/analysis_board/multiplayer/deploy/com.jackfleming.yinsh-log-pull.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.jackfleming.yinsh-log-pull.plist

# Verify (fires immediately on load via RunAtLoad=true)
launchctl list | grep yinsh-log-pull
tail /Users/jackfleming/PycharmProjects/YinshML/analysis_board/multiplayer/deploy/log-pull.log
```

The script (`laptop_pull_logs.sh`) `rsync`s `games/` from the mini into
`~/PycharmProjects/YinshML/game_logs/` (gitignored at the repo root) every
night at 02:00 local time. If the laptop is off-network when the timer
fires, the script exits silently — no launchd error spam, just "tomorrow
will catch it." If the laptop was asleep at 02:00, macOS fires the
LaunchAgent on next wake.

Overrides via environment if your setup differs:

| Variable | Default | What |
|---|---|---|
| `YNS_MAC_MINI_HOST` | `mac-mini.local` | mDNS / hostname / IP of the mini |
| `YNS_MAC_MINI_USER` | `jackfleming` | SSH login on the mini |
| `YNS_MAC_MINI_LOGS_DIR` | `…/deploy/games/` | Where logs live on the mini |
| `YNS_LOCAL_LOG_DIR` | `~/PycharmProjects/YinshML/game_logs/` | Local destination |

To trigger a pull manually (without waiting for the timer):

```bash
launchctl kickstart -k gui/$(id -u)/com.jackfleming.yinsh-log-pull
```

## Day-to-day operations

### Redeploy everything (one command, idempotent)

After any code or plist change you've pushed from your laptop, on the Mac mini run:

```bash
bash /Users/jackfleming/PycharmProjects/YinshML/analysis_board/multiplayer/deploy/redeploy.sh
```

Or set up an alias once (e.g., in `~/.zshrc`) and use that going forward:

```bash
alias yinsh-redeploy='bash /Users/jackfleming/PycharmProjects/YinshML/analysis_board/multiplayer/deploy/redeploy.sh'
```

The script does the whole loop in order: `git pull` → copy plists to `~/Library/LaunchAgents/` → unload services → kill anything squatting on port 5173 → load services → wait 3s → verify the server responds + report the active `YNS_MAX_NUM_SIMS` cap. Safe to run whenever you're unsure of current state.

The sections below are for partial / manual operations when you don't want the full sweep.

### Restart the server only (no pull, no plist update)

```bash
launchctl unload ~/Library/LaunchAgents/com.jackfleming.yinsh-server.plist
launchctl load ~/Library/LaunchAgents/com.jackfleming.yinsh-server.plist
```

### Restart the tunnel

```bash
launchctl unload ~/Library/LaunchAgents/com.jackfleming.yinsh-tunnel.plist
launchctl load ~/Library/LaunchAgents/com.jackfleming.yinsh-tunnel.plist
```

### Stop everything

```bash
launchctl unload ~/Library/LaunchAgents/com.jackfleming.yinsh-server.plist
launchctl unload ~/Library/LaunchAgents/com.jackfleming.yinsh-tunnel.plist
```

### View logs

```bash
cd /Users/jackfleming/PycharmProjects/YinshML/analysis_board/multiplayer/deploy
tail -f yinsh-server.log       # Flask request log + MCTS info
tail -f yinsh-server.err.log   # Tracebacks
tail -f yinsh-tunnel.log       # cloudflared connection log
```

Logs don't rotate automatically. Truncate manually if they grow:

```bash
: > yinsh-server.log
: > yinsh-server.err.log
```

### Pull and redeploy

Use the one-shot `redeploy.sh` at the top of this section. The
manual-equivalent steps are documented there.

Note: the tunnel doesn't need restart for code-only changes — only run
the full sweep when plists have changed too. `redeploy.sh` cycles both
unconditionally because it's cheap and removes the "did I forget the
plist" failure mode.

### Owner bypass for the sims cap

`YNS_MAX_NUM_SIMS` (3200 in the plist) protects public visitors from one
client queuing a multi-minute, lock-serialized search that blocks
everyone else. But the owner sometimes wants to go far deeper (256000
sims on a quiet position). `YNS_OWNER_TOKEN` grants exactly that: an
*analysis* request carrying the matching token skips the cap; everyone
else stays capped. (Play-mode engine moves never carry the token and are
additionally hard-clamped to 3200 client-side, so a game against the
engine always stays snappy.)

The committed token is the simple shared secret `jackfleming` — enough to
stop friends from *accidentally* queuing a giant run, not a real
credential. If you ever need actual protection, swap in
`openssl rand -hex 24` and update the plist + the value you paste into the
board.

#### 1. (Already done) token lives in the plist

`com.jackfleming.yinsh-server.plist` ships with:

```xml
<key>YNS_OWNER_TOKEN</key>
<string>jackfleming</string>
```

If you change it, run `yinsh-redeploy` so launchd picks up the new env
var. Empty = bypass disabled.

#### 2. Paste the same token into the board, once

On your own machine: Engine settings → Advanced MCTS → **Owner token**,
type `jackfleming`. It's stored in your browser's `localStorage`
(`yns_owner_token`) and sent with every analysis eval, so you only do this
once per browser. Public visitors leave it blank.

#### 3. Verify

Set MCTS sims to something above the cap (e.g. 5000) and Analyze. With a
valid token the status banner shows the full count and no "capped"
warning; clear the token and the same request reports
`⚠ capped to 3200 of 5000 requested`. Server-side, an over-cap request
*without* the token logs `capping num_sims …`; *with* it, no such line.

#### Batched search (the speed lever)

The analysis board uses **batched MCTS** (`search_batch`): it collects a batch
of leaf nodes and evaluates them in one network forward pass instead of one pass
per simulation — **10-20× faster on Apple Silicon** for `pure_neural`/`hybrid`.
With Dirichlet disabled (as the board runs it) the results are *identical* to the
old per-leaf path — same visit counts, same value — just much faster (verified by
`yinsh_ml/tests/test_mcts_serial_vs_batch_parity.py`). `pure_heuristic` mode
can't batch (heuristics are per-position), so it stays at the old speed.

Batch size defaults to 64; override deployment-wide with the `YNS_BATCH_SIZE`
env var, or per request with a `batch_size` field in the `/api/evaluate` body.
Bigger batches mean fewer network calls (faster) at the cost of slightly more
virtual-loss-guided selection; 64 is a good balance for big searches.

#### Big searches run as background jobs

Even batched, a search can exceed the Cloudflare tunnel's ~100s response
timeout, so anything above **8000 sims** is dispatched to `/api/evaluate_async`:
the search runs on a background thread, the status banner shows a live
`done/total · % · elapsed` readout, and the SPA polls `/api/evaluate_result/<id>`
twice a second until it lands — instead of the request dying with the
`Unexpected token '<'` HTML-timeout error. The server logs
`async eval job … done: N sims in X.Xs (Y sims/s)` on completion, so the real
batched rate is visible in `yinsh-server.log` — handy for deciding whether to
raise the 8000 cutoff. Searches still run one-at-a-time server-wide (the
`NetworkWrapper` tensor pool isn't thread-safe), so while a long owner job
churns, other users' analyses queue behind it.

### Lock the URL down later (add auth)

If public access becomes a problem, add a Cloudflare Access policy:

1. Dashboard → Zero Trust → Access → Applications → "Add an application"
2. "Self-hosted" application
3. Application domain: `yinsh.jackflemingux.com`
4. Add a policy: "Allow", include emails → list yours + anyone else allowed.

Takes ~5 minutes. Existing tunnel + plists keep working; only the auth check
is added.

## Troubleshooting

### `https://yinsh.jackflemingux.com` returns 502 / 530

Server isn't responding on `127.0.0.1:5173`. Check:

```bash
launchctl list | grep yinsh-server   # should show a PID
curl http://127.0.0.1:5173/api/models   # should return JSON
tail yinsh-server.err.log
```

If the plist shows exit code `78` or similar, the venv path or script path
in the plist is wrong. Re-check and reload.

### Tunnel won't start

```bash
tail yinsh-tunnel.err.log
```

Common causes:
- `<TUNNEL_UUID>` placeholder left in `cloudflared.yml` (replace with real UUID).
- Credentials file path wrong — `~/.cloudflared/<UUID>.json` doesn't exist or
  belongs to a different user.
- DNS not pointed at the tunnel yet (re-run `cloudflared tunnel route dns ...`).

### Server runs but moves don't apply

Check `yinsh-server.err.log` for tracebacks. The most common cause is a
model file mismatch — if you pulled new code that changed the policy head
size, the cached checkpoint won't load. Restart the server after replacing
the checkpoints.

### Replace the friend access (lock down later)

The free Cloudflare Access tier handles 50 users. To gate the public URL:

1. Cloudflare dashboard → Zero Trust → Access → Applications.
2. Application type: self-hosted, domain `yinsh.jackflemingux.com`.
3. Identity provider: one-time PIN (or Google) — email allowlist.
4. Save. Existing tunnel keeps working; the auth check is added at the edge.
