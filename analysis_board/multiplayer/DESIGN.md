# Player-vs-Computer + Shareable Hosting — design doc

**Status:** captured, not yet implemented. Sibling to `analysis_board/loop/` (per-position analysis) and `analysis_board/opening_map/` (opening study). Two coupled features captured together because they share the underlying server + UI; deploy is what makes the first actually useful with collaborators.

## Goals

1. **Game mode** — play a full YINSH game against the engine, alternating moves, with engine recommendations hidden during play. Optional post-game review with annotations.
2. **Shareable hosting** — a friend (and ~5 others total) can play the engine remotely without Jack's involvement, so the friend can give honest appraisals of engine strength independent of any in-the-moment hand-holding.

## Non-goals

- Durability — server can restart; in-flight games may be lost. Not building reliable persistence in v1.
- High concurrency — ~5 simultaneous users max, often only 1-2 at a time.
- ELO/ranking/leaderboards — out of scope.
- Real-time multiplayer (two humans against each other) — out of scope.
- Time controls — out of scope for v1; can add later.

## Game-mode UX

A new top-level mode toggle alongside `Setup` / `Play` / (and the existing line-walk modal). Call it `Game`.

**New-game setup screen** (replaces the board when entering Game mode with no game in progress):

- Which side do I play? (White / Black / Random)
- Opponent strength: dropdown of (Network only / MCTS 200 / 800 / 3200 / 6400). Maps directly to `num_sims` on the existing /api/evaluate.
- Model: existing dropdown.
- Spoilers during play: off (default) / on. When off, the entire right-side analysis panel (top moves, value bars, PV ▶ buttons) is hidden until the game ends or until the player opts in.
- "Start game" → enters in-game UI, alternating moves.

**In-game UI:**

- Same board renderer as Play mode.
- When it's the human's turn: click-to-select, click destination, capture sequence via top-of-board prompts (we already have the multi-row UX from Play mode — reuse).
- When it's the computer's turn: a "Thinking…" indicator on the board, then the computer move appears with a soft animation (~600ms fade-in of the move arrow + piece transition). A new history row gets added.
- Move history list visible in the sidebar (already exists).
- Score + phase indicator visible (already exists).
- Spoilers off: top-moves panel hidden, value bars hidden, MCTS sims controls hidden. Only "Resign", "Offer draw" (skip for v1), and a "Show analysis" toggle that reveals the panel if the user wants to peek.

**End of game:**

- Show "Game over — X wins, score Y-Z" prominently.
- "Review" button reveals the full analysis panel + adds a step-back UI to walk the played moves, with the engine's evaluation at each position shown alongside. (This is the part the friend uses for honest appraisal — they play through, then review with engine eyes.)
- "New game" → returns to setup screen.

**Open question:** computer auto-plays vs manual "Make computer move" button? Auto-play feels more natural (it's a game). But a button gives the human time to think between turns. Compromise: auto-play with a 1.5s delay (Jack can use that to ponder). Configurable in setup screen if needed.

## Server-side changes (small)

The existing `/api/evaluate` and `/api/move` endpoints cover everything Game mode needs. Computer-move-picking = call `/api/evaluate` from the frontend with the current position + selected num_sims, take `top_moves[0]`, send through `/api/move`. The frontend already does this for "click top-move to apply" in Play mode — Game mode is the same flow with the human deferring to the engine.

Optional additions for v2:

- `POST /api/game` — create a server-side game record (id, current position, history). Returns a `game_id` the frontend can use to resume.
- `GET /api/game/:id` — load by id. Enables URL-based sharing ("play this exact position against me").
- Simple SQLite store, ~50 LOC. Skip for v1.

The current model + MCTS caches already work multi-user (Python's GIL serializes Flask request handlers; each cached MCTS instance reuses its tree across calls but `reset_tree()` is called before each search, so cross-user contamination isn't a risk). One per-process concern: two users hitting MCTS-3200 at the same time will queue — request B waits for request A to finish search. At 5 users that's tolerable; if it becomes a problem, add a small `multiprocessing.Pool` or just enforce one MCTS at a time with a status indicator.

## Hosting architecture (Mac mini)

**Recommended path: Cloudflare Tunnel + Cloudflare Access.**

```
[Mac mini, local network]
  |
  +-- analysis_board/server.py (Flask, port 5173)
  |     |
  |     +-- Model wrappers (cached, shared across users)
  |     +-- MCTS instances (cached per (model, num_sims, ...))
  |
  +-- cloudflared (Cloudflare Tunnel daemon)
        |
        +-- Outbound tunnel to Cloudflare edge
              |
              +-- Public hostname: e.g., yinsh.<your-domain>
                    |
                    +-- Cloudflare Access in front
                          |
                          +-- Allowlist by email (yours + friend's + a few others)
                          +-- Google / GitHub / one-time-PIN login
```

Why this combo:

- **No port forwarding** — tunnel is outbound from the Mac mini. Home network stays closed.
- **TLS handled by Cloudflare** — no Let's Encrypt setup, no cert renewal.
- **Auth without writing any code** — Cloudflare Access does the OAuth dance + email allowlist. Free tier handles up to 50 users.
- **Free for the scale we need.**
- **Easy revoke** — remove an email from the Access policy and they're locked out instantly.

Alternative paths (less recommended):

- **Tailscale** — works but requires the friend to install Tailscale. Adds friction for "I want to play YINSH for 20 minutes." Better for engineering peers, worse for casual collaborators.
- **ngrok** — easy but free tier rotates URLs (bookmarkable URL requires paid plan).
- **Self-host + port forward + Let's Encrypt** — full control but a real setup project. Overkill for 5 users.
- **VPS** — defeats the "use my Mac mini" intent, and the model loading is faster on the Mac mini's MPS than on a cheap CPU VPS.

## Resource + concurrency management

Concerns:

- **MCTS-heavy concurrent requests can pile up.** Python GIL + Flask threading means each request handler runs sequentially in practice for compute-heavy work. With 5 users x 3200 sims each, worst case is 5 * 25s = 2 minutes for the last user.

Mitigations:

- **Cap `num_sims` server-side at 1600 for hosted mode.** Maybe a config flag `MAX_NUM_SIMS=1600` when running in shared mode. 25s → 11s worst-case.
- **Single-MCTS-at-a-time semaphore.** `threading.Semaphore(1)` around the MCTS path so users queue but each gets full resources. Display "Engine is thinking for another player (~Xs left)" while waiting.
- **Per-user game state lives entirely in the frontend.** Server stays stateless; no sticky-session concerns.

Memory:

- Each cached `(model, num_sims, c_puct, fpu, mode, hw)` tuple holds one MCTS instance. With ~5 users using mostly defaults, the cache won't blow up — maybe 5-10 instances total. Each is ~10-50MB.
- Models cached separately — one wrapper per checkpoint, all users share.

## Stretch features (v2+)

1. **Share-by-URL.** Save game state to SQLite, mint a short URL, friend clicks → opens the analysis board mid-game in spectator mode. Useful for "look at this position I'm stuck on."
2. **Review-with-annotations.** After a game, walk the moves with the engine's value + top moves visible at each ply. Lets the friend say "the engine missed this defensive resource here, here, here." Export the review as a markdown report.
3. **Spectator mode.** Read-only URL where someone watches a game in progress without being able to make moves. Useful for review sessions.
4. **Per-user game history.** Tied to Cloudflare Access identity, store every game played, let users browse their history. Probably overkill unless we discover real demand.

## What this enables for engine evaluation

The friend's role here is genuinely informative: a strong YINSH player gives qualitative feedback that the alignment loop can't. Specifically:

- "The engine missed a 3-move tactic on move 27" — categorically different signal from "the policy head ranks move-X as #4 instead of #1." Tactical depth is hard to measure with the alignment metrics; human evaluation surfaces it.
- "I was ahead the whole game then lost because the engine baited me into a path-flip trap" — points at evaluation-of-trap-positions as a model weakness.
- Subjective "the engine plays passively in the opening" or "the engine over-values rings on the wall" — hypotheses worth turning into measurements in the alignment loop.

So the deployment isn't just a fun toy — it's a data-collection mechanism for the kind of qualitative signal we can't get from self-play. Pair with the review mode (stretch feature 2) and ask the friend to annotate 5-10 games. Then look for patterns.

## Open questions to resolve before starting

- **Auto-play computer move or button?** Discussed above — leaning auto-play with 1.5s delay, configurable.
- **What domain?** Need a hostname. `yinsh.jackfleming.dev` or similar — Jack to provide.
- **Friend's email + auth method?** Need to know who's on the allowlist before deploying.
- **Resignation logic.** When does a game end? Player resigns explicitly, or reaches game-over via play. No timeout logic in v1.
- **Computer's identity in the move history.** "Engine (yngine_volume_15ch / MCTS 3200)" or just "Engine"? Probably the full name so reviewing games later, you know which model played.
- **Should computer plays be logged separately for retrospective analysis?** Easy yes — a per-game JSONL of every position + computer's eval would feed straight into the analysis loop.

## Scripts to write (when we pick this back up)

```
analysis_board/multiplayer/
├── DESIGN.md                  # this file
├── (no new scripts — frontend changes only for v1)
└── deploy/
    ├── cloudflared.yml        # tunnel config
    ├── launchd.plist          # macOS service definitions
    └── README.md              # one-command deploy + revocation steps
```

Most of the work is frontend UX changes in `analysis_board/static/*` + a small handful of server flags (`MAX_NUM_SIMS`, semaphore wrapper). Deploy artifacts live under `deploy/` so they're not mixed with the app code.

## Practical scope (first cut)

**v1 (local-only Game mode, no hosting):** 1 day of work.
- Game mode toggle + new-game setup screen + computer-auto-play + spoilers toggle.
- Reuse the existing /api/evaluate + /api/move endpoints unchanged.
- End-of-game screen with review toggle.

**v1.5 (hosting):** half a day.
- Install cloudflared, configure tunnel + Access policy.
- Add `MAX_NUM_SIMS` config flag and `threading.Semaphore(1)` around the MCTS code path.
- launchd script so the server + tunnel survive Mac mini reboots.

**v2 (review + share-by-URL):** another 1-2 days when there's demand.

Net: a focused 1.5-2 day project to get the friend playing. Worth picking up after the next round of analysis-loop findings, when we want qualitative feedback to corroborate or refute the quantitative patterns.

---

## Mac mini kickoff prompt

This section is the cold-start prompt for a new Claude Code session on the Mac mini that will deploy this thing. The session needs to (a) implement Game mode in the existing analysis board, (b) add the resource caps for shared use, (c) set up Cloudflare Tunnel + Access for remote access, and (d) wire launchd so it survives reboots. Paste the block below into a fresh session running in the YinshML repo on the Mac mini.

```
You're continuing work on the YinshML analysis board — a Flask + JS web tool
at `analysis_board/` for analyzing YINSH positions with a trained neural net.
It currently runs locally on my laptop. The next phase is two coupled things:

1. Add a "Game" mode so I (and a small number of friends) can play full games
   against the engine, with engine recommendations hidden during play and an
   optional post-game review pass that reveals MCTS/value annotations at
   each move.
2. Host the resulting app on this Mac mini, exposed via Cloudflare Tunnel
   + Cloudflare Access (email allowlist), so ~5 trusted users can play
   remotely without me being involved.

This Mac mini is the deployment host. We'll be doing the local code + setup
work here. Architecture is fully designed at:

  analysis_board/multiplayer/DESIGN.md

READ THAT FIRST — it spells out the UX, server changes, hosting plan, and
resource-management plan. The decisions are locked; don't re-litigate the
architecture. Specifically pre-decided:

  • Hosting: Cloudflare Tunnel + Cloudflare Access (NOT Tailscale, NOT
    ngrok, NOT self-hosted nginx). Free tier of CF Access handles email
    allowlist for ~50 users.
  • Server stays STATELESS. User game state lives in each browser. No
    SQLite or database for v1.
  • Concurrency: `MAX_NUM_SIMS=1600` cap + single-MCTS `threading.Semaphore(1)`
    around the MCTS path. Display a "Engine is thinking for another
    player" indicator on the waiting client.
  • Existing endpoints (`/api/evaluate`, `/api/move`) already cover Game
    mode unchanged. No new backend endpoints in v1.

Project context (read these too):
  • `CLAUDE.md` (root) — project overview, encoding, MCTS, model anchor.
  • `analysis_board/README.md` — analysis-board feature overview.
  • `analysis_board/server.py` — Flask backend, where MAX_NUM_SIMS +
    semaphore go.
  • `analysis_board/static/{index.html, app.js, style.css}` — frontend,
    where Game mode UI goes.

What I need from this session, in order:

PHASE 1 — Local repo setup on this machine (~30 minutes)
  • Clone the repo (or pull if already cloned) and checkout the
    `training-pipeline-fixes` branch.
  • Set up `venv`, install requirements, install Flask.
  • Copy or sync model checkpoints from my laptop into `models/`. The
    one we definitely need is `models/yngine_volume_15ch_pretrain/best_supervised.pt`
    (the 15-channel anchor). Others optional but useful for the model
    dropdown. Help me figure out the right rsync command.
  • Verify the server runs on this machine — `python analysis_board/server.py`
    + curl http://127.0.0.1:5173/api/models.

PHASE 2 — Game mode implementation (~6-8 hours of work)
  • Per the DESIGN.md "Game-mode UX" section: new mode toggle, new-game
    setup screen, alternating play, spoilers-off-by-default analysis panel,
    end-of-game with review toggle.
  • Auto-play computer move with a 1.5s delay (configurable in the setup
    screen — let me decide what default I want once I see it).
  • Reuse the existing `/api/evaluate` + `/api/move` infrastructure
    untouched. The only server change in this phase is plumbing
    `MAX_NUM_SIMS` and the MCTS semaphore.
  • Test locally end-to-end before moving to deploy.

PHASE 3 — Cloudflare Tunnel + Access setup (~1-2 hours, mostly waiting)
  • Install cloudflared via brew.
  • Walk me through `cloudflared tunnel login`, tunnel creation, DNS
    record setup, and the Cloudflare Access policy configuration in the
    CF dashboard. I have a Cloudflare account but it's been a while —
    you'll need to give me step-by-step.
  • I'll need to decide on a hostname (something like
    `yinsh.<my-domain>.com`) and tell you the friend's email for the
    allowlist. Surface these as questions when you need them.

PHASE 4 — Persistence via launchd (~30 minutes)
  • LaunchAgent plists for the Flask server + cloudflared so both
    auto-start on login and restart on crash.
  • Store config in `analysis_board/multiplayer/deploy/` so it lives
    with the rest of the multiplayer artifacts.
  • Test by killing both processes and confirming they come back up.

PHASE 5 — End-to-end verification
  • Open the public URL from an off-network device (phone on cell, or
    a friend's machine). Confirm the CF Access prompt appears, login
    works, the board loads, a game can be played start to finish, and
    the analysis panel correctly hides during play in spoilers-off mode.

Out of scope for this session (explicitly v2 — do not build):
  • SQLite game persistence
  • Share-by-URL
  • Review-mode with engine annotations at each ply
  • Spectator mode
  • Per-user game history

Don't add features beyond v1. The friend wants to play games and give
feedback; that's the success criterion.

Once setup is complete: commit the work to the `training-pipeline-fixes`
branch, push, and produce a short README at
`analysis_board/multiplayer/deploy/README.md` covering: how to start/stop
locally, how to add/remove an allowlisted email in the CF Access policy,
where the launchd plists live, and how to view server logs.

Start by reading `analysis_board/multiplayer/DESIGN.md` and confirming
back to me what you've understood, then begin Phase 1. Ask me about
hostname + emails when you get to Phase 3.
```

If you (current session, doing the architecture work) need to revise the prompt as the design evolves, update it here. Keep it the single source of truth for the deploy session.
