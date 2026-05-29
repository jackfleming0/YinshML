# Telemetry / game-log schema TODO

Backlog of improvements to the `/api/move` JSONL log schema. None are urgent;
surface when next iterating on the server, frontend, or analysis tooling.

**Motivation** (2026-05-29): first N=4 sessions analyzed via `scripts/analyze_game_logs.py`
showed the engine's "cluster" opening strategy is clearly distinguishable from
the human's "spread" strategy via the per-color spread metric — but we had to
*infer* which side was the engine. With explicit attribution + per-player
identity, future analyses could go from "we think the engine plays BLACK"
to "the engine plays the cluster regardless of color, with N% win rate as
each side, against M distinct human testers."

---

## Priority 1: side attribution

**Problem:** the JSONL doesn't record which side was the engine vs the human.
Today's analysis infers from move patterns. Works at N=4, won't scale.

**Fix:** add `engine_side: "WHITE" | "BLACK" | null` to the `/api/move`
payload from the frontend. The value already exists as
`state.game.computerSide`. Server logs it alongside the existing fields.
`null` for self-play / Setup-mode / non-game-instance requests.

**Cost:** ~5 lines (one in `currentPositionPayload()`, one in
`_log_move_event` in server.py).

**Unlocks:**
- Win rate by side for the engine
- Whether the cluster strategy is color-dependent
- Per-side capture timing distributions

## Priority 2: cross-session human identification

**Problem:** can't tell whether two completed games were played by the same
person or two different friends. Today they look identical in
`analyze_game_logs.py`.

**Fix options** (in increasing intrusiveness; pick one):

- **Salted IP hash.** Read `request.headers.get("CF-Connecting-IP")` (Cloudflare
  injects this automatically through the tunnel), salt with a static
  server-side secret in `YNS_IP_HASH_SALT` env var, SHA-256, log first 16 hex
  chars. Two games from the same household correlate. Raw IP never touches
  disk. Doesn't track across mobile-to-home (feature, not bug). Survives
  browser changes / incognito.
- **Persistent client UUID.** Generate a UUID in `localStorage` on first
  visit (separate from `play_session_id` which resets per game). Sticky across
  tabs, not across browsers or incognito. More accurate within a browser,
  more PII-adjacent.
- **User-agent string.** Cheap, useful for rough OS / browser fingerprinting
  ("all sessions are iOS Safari = probably same person"). Combine with the
  hashed IP for tighter inference without storing more.

The salted-IP-hash is the most defensible privacy-wise — the hash is
uncorrelated with anything outside this server's secret, and rotating
`YNS_IP_HASH_SALT` invalidates all prior identifiers.

## Priority 3: session-lifecycle events

**Problem:** the JSONL only captures `/api/move` events, so people who land on
the page and bounce *without playing* aren't in the log. The "3 of 4
sessions abandoned" stat is undercounted — true engagement funnel is
unknown.

**Fix:**
- Add `POST /api/session_start` called once per page load. Log: ts, IP hash,
  user-agent, referrer.
- Wire `navigator.sendBeacon("/api/session_end", {reason})` on page-close
  + on explicit "End game" click. `reason ∈ {"resigned", "page_close",
  "natural_end", "idle_timeout"}`.

Pairs with Priority 2 to compute true unique-visitor and engagement-funnel
metrics, not just move-throughput.

## Priority 4: engine configuration snapshot

**Problem:** if engine settings change (sim cap, model swap, c_puct), past
game logs don't capture the setup that produced them. Today's "cluster
strategy" analysis would be ambiguous about whether it's the 1600-sim
behavior, the 3200-sim behavior, or the iter1_ema vs supervised checkpoint.

**Fix:** the first `/api/move` per session sends the engine config; server
stamps it onto every subsequent event in that session (so analysis doesn't
need to chain by session_id to recover config). Fields: `model_id`,
`num_sims`, `c_puct`, `fpu_reduction`, `evaluation_mode`, `heuristic_weight`.

## Priority 5: per-move timing

**Problem:** can't see how long humans deliberate vs how fast the engine
plays without computing deltas from `ts`. Adding the delta explicitly makes
analysis cheaper and surfaces "engine took 22s, human took 90s" patterns
directly.

**Fix:** add `wall_clock_ms_since_prev_move` to each event (server computes
from previous event's `ts` for the same `play_session_id`). Cheap.

## Priority 6: retention + log rotation

**Problem:** no documented retention policy. JSONL files grow unbounded.

**Fix:**
- Document a retention period (suggest 90 days) in `deploy/README.md`.
- Add a launchd-scheduled cleanup that `find … -mtime +90 -delete` on the
  games directory.
- If implementing the IP hash, the salt should be in `YNS_IP_HASH_SALT` env
  var so rotating it invalidates all prior identifiers if needed.

## Priority 7: schema versioning

**Problem:** as items above land, old logs and new logs have different
shapes. `scripts/analyze_game_logs.py` will need to handle both.

**Fix:** add `schema_version: 1` to every event going forward; bump on
incompatible additions; the analyzer reads version and dispatches accordingly.

---

## Triage note

The highest-leverage single addition is **`engine_side`** (Priority 1) — one
frontend line + one server line, unlocks every per-side analysis that's
currently inference-based. The second-best is **salted IP hash** (Priority 2)
— gives a cross-game identity without compromising privacy. The rest are
incremental.

If implementing all of the above in one pass: bump `schema_version` once,
add the lifecycle events, log everything from the start.
