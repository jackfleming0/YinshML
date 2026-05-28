# Continuation prompt — analysis_board UI polish + hosting deploy

**Status:** drafted at the end of the 2026-05-28 session, after PR [#18](https://github.com/jackfleming0/YinshML/pull/18) was opened to merge `training-pipeline-fixes` into `main`. Paste the block below into a fresh Claude Code session (laptop *or* Mac mini, depending on whether you're polishing UI or deploying) to continue the work.

---

```
You're continuing work on the YINSH analysis board — a Flask + three.js
web tool at `analysis_board/` for playing YINSH against a trained
network and analyzing positions.

REPO + BRANCH
  • Repo: YinshML (this one)
  • Branch: `training-pipeline-fixes` (PR #18 against `main`) — or `main`
    if that PR has merged by now. Check `git branch --show-current`.
  • Author: Jack Fleming. Solo contributor.

READ FIRST (in this order)
  1. `analysis_board/multiplayer/DESIGN.md` — the original architecture
     spec. Read sections "Goals", "Hosting architecture", "Resource +
     concurrency management". The Phase 1–5 kickoff at the bottom is
     LARGELY done already (see "What's done" below) but Phases 3–5
     (Cloudflare deploy + launchd) are still pending.
  2. `analysis_board/README.md` — feature overview, run instructions.
  3. `analysis_board/server.py` — Flask backend. Key entry points:
     `/api/models`, `/api/evaluate`, `/api/move`. MCTS cache + model
     wrapper cache live here.
  4. `analysis_board/static/board3d.js` — three.js scene module. Owns
     the 3D rendering: scene, camera, OrbitControls, raycasting,
     piece geometries with terrazzo-speckle textures and teal rim,
     coordinate labels, dwell tooltip, hover/selection/arrow meshes.
  5. `analysis_board/static/app.js` — app logic. State, mode +
     Opponent toggles, /api/* calls, drag-drop, click handlers, game
     lifecycle (enterGameSetup / startGame / computerMakeMove /
     endGameSession), line-mode (PV walking), evaluate + renderResult.
  6. `analysis_board/static/index.html` + `style.css` — structure and
     dark theme palette. Two top-level modes: `Set up position` and
     `Play`. Inside Play, an `Opponent: Play both sides / vs Engine`
     pill switches the panel between self-play and engine-session UI.
  7. `CLAUDE.md` (repo root) — project conventions, encoding details,
     MCTS engine notes.

GLOBAL CONVENTIONS (don't drift on these)
  • NO `Co-Authored-By: Claude` (or any Claude/Anthropic attribution)
    in commit messages. Write commits as if Jack authored them.
  • Match Jack's visual idiom for the board: cool icy-blue board surface
    (NOT parchment cream — see memory entry
    `feedback-yinsh-board-visual-reference`), terrazzo-speckled cream/
    obsidian discs, teal rim, dark charcoal page, 3D perspective tilt.
    Reference is BoardGameArena / Rio Grande, NOT the physical
    tabletop board.
  • Jack prefers a warm-start interview (~20 focused questions + bring
    existing knowledge) over blank templates. He wants peer-level
    collaboration — bold and opinionated, frequently wrong is fine.
    Don't hedge.
  • Plan first, then execute in bulk. Don't ask permission on
    low-stakes decisions you can just make.

WHAT'S DONE (don't redo any of this)
  Three.js 3D renderer (board3d.js):
    - BGA-style cool blue-glass slab with vertex-color vignette
    - LineSegments hex grid + × hash marks at intersections
    - Cylinder markers + LatheGeometry rings, both with procedural
      CanvasTexture speckle and teal rim
    - OrbitControls constrained: pitch 0–~50°, no pan, dolly 12–28
    - Reset-view button (overlaid top-right of board) with eased tween
    - A-K + 1-11 coordinate labels printed on board (paired-back: 0.22
      board-units, 0.42 alpha, sans, weight 400)
    - 350ms dwell tooltip near cursor with position name
    - Raycaster `pixelToBoardPos(clientX, clientY)` returns {col, row}
      or null
    - Suggestion arrow rendering (source disc + shaft + cone head)

  Mode collapse (3 modes → 2 modes):
    - Top toggle: `Set up position` / `Play`
    - Inside Play: `Opponent` pill = `Play both sides` (auto-evaluate
      after each move, top-moves visible, click-to-apply) OR
      `vs Engine` (engine config pre-game, engine session during play,
      spoilers-off blind mode)
    - State adds `state.opponent: "self"|"engine"`; mode-game CSS
      rules ported to `body.opponent-engine`

  Sidebar IA restructure (sections with identity):
    - Opponent toggle (Play only)
    - STATUS (compact: Phase | To move side-by-side + score/rings)
    - Pieces palette (Set up only)
    - Setup actions: Analyze position / Play from here → / Clear board
    - Engine setup / Engine status (Play+Engine only)
    - NETWORK ANALYSIS (block-title header, value bars, top-moves)
    - MOVE HISTORY
    - ENGINE SETTINGS — collapsed <details> with Model + MCTS sims +
      Advanced MCTS

  vs-Engine bug cluster (all rooted in stale `state.lastResult` when
  spoilers off — no auto-evaluate runs between turns):
    - computerMakeMove was reading pre-move side from lastResult;
      now uses currentSide() (the radio, kept fresh by
      applyNewPosition). Engine wasn't taking turns.
    - handlePlayClick had the same stale check in 4 places (mode-
      engine guard, side-ring inference, RING_PLACEMENT phase gate,
      RING_REMOVAL phase gate). Switched to currentSide() /
      phaseSel.value.
    - Capture phases (ROW_COMPLETION / RING_REMOVAL) now trigger an
      evaluate regardless of spoilers — the analysis panel is the
      ONLY UI for picking which row/ring to remove.
    - Inline "Show engine analysis" toggle now refreshes the
      analysis instead of rendering stale pre-game results.

  Capture-phase blind mode (spoilers off):
    - Engine numerics hidden via CSS (percentages, visit counts, value
      bars, best-move chip, on-board top-move arrow)
    - Options re-sorted by canonical board position so engine's
      preferred row is no longer at position 1

  Defensive guards:
    - Click block when Opponent=Engine but no active game (with hint
      to click Start)
    - Camera damping bumped 0.05 → 0.15; rotateSpeed 0.6 → 0.40

WHAT'S PENDING

  Hosting (Phases 3–5 of the original DESIGN.md kickoff):
    The Mac mini is the chosen host. Cloudflare Tunnel + Access in
    front (free tier handles email allowlist). Jack already owns
    `jackflemingux.com`; the plan is to use a subdomain like
    `yinsh.jackflemingux.com` so the existing site at the apex stays
    untouched. Steps:
      a. Install cloudflared via brew (Mac mini)
      b. `cloudflared tunnel login` → tunnel create → DNS CNAME →
         Cloudflare Access policy with Jack's email + friend's email
         allowlist
      c. Add `MAX_NUM_SIMS=1600` cap server-side (env var or config
         constant) so a friend on the public URL can't accidentally
         queue a 38-second 3200-sim eval
      d. `threading.Semaphore(1)` around the MCTS path so concurrent
         users queue instead of contending
      e. launchd plists (analysis_board/multiplayer/deploy/) for the
         Flask server + cloudflared so both auto-start on login and
         survive crashes
      f. End-to-end verification: open the public URL from a phone on
         cell data, confirm CF Access prompt, login works, play a
         game start-to-finish

    Need from Jack when you get to (b): final hostname choice + the
    friend's email for the allowlist.

  UI polish items Jack flagged (not blockers, but the next iteration):
    - Status grid density — Phase + To move in 2 columns may feel
      cramped at this width; could go single-column stack
    - "W / B" shorthand on side radio — too cryptic? Full words?
    - Engine Settings disclosure default state — currently closed;
      maybe open in Set up mode (you tune sims often there)
    - Coordinate label tuning — already paired back, but verify
      they're "indicators not dominant"
    - Tooltip dwell time — 350ms; maybe 500 if it feels fidgety
    - Hero score treatment — could promote the score line into a
      larger visual element (Jack said "the start of the show is the
      score")
    - Section header text in capture-blind mode — currently "Network
      analysis" stays; could swap to "Pick a row" / "Pick a ring"
    - "Play from scratch" affordance — currently you go Set up →
      Clear → Play from here →. Could be a one-click button in Play.

  Animation polish (was task #8, never started):
    - Smooth tween for ring movement (slide src → dst)
    - Marker flip when ring passes (rotate 180° around Z so speckled
      face shows new color)
    - Gentle ease-in on engine-move appearance
    - Suggestion arrow tween on hover

  Wishlist items (memory entries — won't evaporate):
    - `project_analysis_board_game_import_wishlist.md` — FEN/PGN-like
      import from BGA. Snapshot (paste a position) + game-stream
      (paste a BGA log → step through with engine analysis at each
      ply). The bga.py parser already exists at
      `yinsh_ml/data/scrapers/bga.py`; this is mostly a UI build.

LOCAL DEV LOOP
  • Start server: `python analysis_board/server.py` from repo root
    (after `source venv/bin/activate`)
  • URL: http://127.0.0.1:5173/
  • Models live at `models/<name>/<checkpoint>.pt`. Active dropdown
    entries on this laptop: `iter1_ema_2026-05-27` and
    `supervised_2026-05-27`. Older models parked in `models/_archive/`
    are hidden from the dropdown by a `_`/`.` prefix filter in
    server.py.
  • Iteration cycle on the deployed Mac mini: edit → git pull →
    ctrl+c Flask, restart → refresh browser. ~30s.
  • Browser cache can sting after CSS / JS edits — Cmd+Shift+R
    (hard refresh).

START BY:
  1. Reading the files in the "READ FIRST" list above.
  2. Confirming back to Jack what you understand the current state to
     be and which thread he wants to pick up first — UI polish, or
     deploy.
  3. If UI polish: ask which of the pending items he wants to land
     this session (don't try to do all of them — pick 2-3 high-value
     ones).
  4. If deploy: confirm hostname + friend's email, then walk through
     the Cloudflare setup step by step (Jack hasn't done CF Tunnel
     before; give clear UI directions for the dashboard parts).
```

---

## Notes for updating this prompt

Keep this file as the single source of truth for the cross-session handoff. When something on the "pending" list ships, move it to "what's done." When new items emerge, add them to "pending." Don't let it drift — the next session leans on it as the warm-start brief.
