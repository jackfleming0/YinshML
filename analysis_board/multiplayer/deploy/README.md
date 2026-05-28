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

## Day-to-day operations

### Restart the server (e.g., after pulling new code)

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

```bash
cd /Users/jackfleming/PycharmProjects/YinshML
git pull
launchctl unload ~/Library/LaunchAgents/com.jackfleming.yinsh-server.plist
launchctl load ~/Library/LaunchAgents/com.jackfleming.yinsh-server.plist
```

The tunnel doesn't need restart for code changes — it only proxies HTTP.

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
