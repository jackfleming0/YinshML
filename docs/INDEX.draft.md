# YinshML — Index

> **This is a DRAFT** of the proposed post-cleanup `docs/INDEX.md`. The cleanup
> pass (Task #12) hasn't happened yet — file paths shown here assume the moves
> are done. Review, edit, then rename to `INDEX.md` after the cleanup session.

YinshML is an AlphaZero-inspired ML framework for the YINSH board game. This
file is the entry point: read it when you start a session, or when you need to
remember "where is X?"

The repo has three kinds of doc:

- **Reference** — always-current (game rules, architecture, quick start).
- **Plans** — forward-looking (roadmaps, the experiment backlog).
- **Experiments** — per-branch session logs, runbooks, results.

Plus a small set of root-level docs (`CLAUDE.md`, `README.md`, `TECH_DEBT.md`)
that stay at the top because they're loaded by tooling or are constant
lookups.

---

## Where am I?

- **Active branch:** `training-pipeline-fixes`
- **Current frozen anchor:** `models/branchC_volume_pretrain/best_iter_4.pt`
- **Last decisive verdict:** *(updated as runs land — see Timeline below)*
- **Next experiment in queue:** see [Forward queue](#forward-queue) below

When you finish a session, update the three lines above. The whole point of
this index is that 60 seconds here = full project context.

---

## Timeline

Most recent first. Each entry is one line in the index; full detail lives in
the linked doc.

| Date | Branch / experiment | Verdict | Doc |
|---|---|---|---|
| 2026-05-25 | **Branch D.2** — 15-ch enhanced encoding | *(SPRT pending)* | [experiments/branch-D.2/RESULTS.md](experiments/branch-D.2/RESULTS.md) |
| 2026-05-24 | **Branch D.1 v2** — GAP value head + hidden Linear | NOT_STRONGER (1-15-0, structural determinism verified) | [experiments/branch-D.1/RESULTS.md](experiments/branch-D.1/RESULTS.md) |
| 2026-05-23 | **Branch D.1 v1** — GAP value head, direct projection | NOT_STRONGER (1-15-0) | [experiments/branch-D.1/RESULTS.md](experiments/branch-D.1/RESULTS.md) |
| 2026-05-22 | **Step 2 MCTS-400** — search depth lever | INCONCLUSIVE small edge (WR 0.552, CI95 [0.504, 0.600]) | [experiments/step2-mcts400/RESULTS.md](experiments/step2-mcts400/RESULTS.md) |
| 2026-05-21 | **Yardstick built** — frozen-anchor SPRT instrument | tool shipped | [experiments/yardstick/RESULTS.md](experiments/yardstick/RESULTS.md) |
| 2026-05-20 | **Branch C MCTS-200 rerun** — volume-pretrain warm-start | HOLDS strength, no compound (Anchor 100% all iters, Elo flat) | [experiments/branch-C/RESULTS.md](experiments/branch-C/RESULTS.md) |
| 2026-05-19 | **Volume-corpus pretrain** (Option 2) | **STRONGEST checkpoint at the time** — swept HA d1/d2/d3 at 100% | [experiments/branch-C/RESULTS.md](experiments/branch-C/RESULTS.md) |
| earlier | Wave 3 / Warm-start exploration | superseded | [archive/WAVE3_EXPERIMENT_LOG.md](archive/WAVE3_EXPERIMENT_LOG.md) |
| earlier | Bitboard port + follow-up | shipped | [archive/bitboard-port/](archive/bitboard-port/) |
| earlier | 100K games statistical analysis | informed heuristic weights still in use | [archive/ANALYSIS_SUMMARY_100K_GAMES.md](archive/ANALYSIS_SUMMARY_100K_GAMES.md) |

---

## Forward queue

The ranked queue. Full table + per-experiment write-ups live in
[plans/EXPERIMENT_BACKLOG.md](plans/EXPERIMENT_BACKLOG.md); the short summary
here is enough to pick the next thing to run.

**Top 5 right now** (conditioned on D.2 SPRT outcome):

1. **A1** — Direct SPRT: D.2 pretrain vs `best_iter_4`. Free info, gates
   interpretation of D.2.
2. **F1** — Audit + fix bare `NetworkWrapper(device=...)` sites. Cheap,
   unblocks future cross-arch evals.
3. **A3** — Re-pretrain 6-ch baseline at 6 epochs. Removes "encoding vs
   training budget" confound from every comparison.
4. **A4** — Regression value head pretrain. Highest mechanism-based prior of
   any single change.
5. **B1** — Tighten Wilson gate to 0.50. Single config knob; tests whether
   the self-play loop can be made gain-preserving.

For the rest of the queue, the five-axis stack-rank, and per-experiment
write-ups → [plans/EXPERIMENT_BACKLOG.md](plans/EXPERIMENT_BACKLOG.md).

---

## Quick navigation

### Reference (always-current)

These describe how things *are*, not what you're trying to change.

- [`reference/YINSH_RULES.md`](reference/YINSH_RULES.md) — game rules, board
  geometry, win conditions.
- [`reference/QUICK_START.md`](reference/QUICK_START.md) — first session
  setup, commands, environment.
- [`reference/ARCHITECTURE.md`](reference/ARCHITECTURE.md) — system overview
  (new file; distill from `CLAUDE.md` in the cleanup pass).
- [`../CLAUDE.md`](../CLAUDE.md) — global instructions, loaded automatically
  by Claude Code. Stays at root.
- [`../TECH_DEBT.md`](../TECH_DEBT.md) — known bugs and instrument-
  correctness issues. Stays at root for fast lookup.

### Plans (forward-looking)

- [`plans/ROADMAP_TO_ALPHAZERO.md`](plans/ROADMAP_TO_ALPHAZERO.md) —
  long-horizon plan toward strong play.
- [`plans/EXPERIMENT_BACKLOG.md`](plans/EXPERIMENT_BACKLOG.md) — **the
  ranked queue**. Update on every session that adds or resolves an entry.
- [`plans/TRAINING_REFACTOR_PLAN.md`](plans/TRAINING_REFACTOR_PLAN.md) —
  pipeline overhaul plan (some items shipped, some pending — check status
  header).
- [`plans/ARCHITECTURAL_IMPROVEMENTS_PLAN.md`](plans/ARCHITECTURAL_IMPROVEMENTS_PLAN.md)
  — exploratory architecture options. Mostly superseded by Branch D work
  but kept for reference.

### Experiments (per-branch session logs + runbooks + results)

One folder per experiment, with PREP / RUNBOOK / RESULTS docs as relevant.

- [`experiments/branch-C/`](experiments/branch-C/) — volume-corpus pretrain
  + MCTS-200 rerun (2026-05-19 → 21).
- [`experiments/step2-mcts400/`](experiments/step2-mcts400/) — search-depth
  ceiling test (2026-05-22).
- [`experiments/yardstick/`](experiments/yardstick/) — frozen-anchor SPRT
  instrument build-out.
- [`experiments/branch-D.1/`](experiments/branch-D.1/) — GAP value head
  v1 + v2 (2026-05-23 → 24).
- [`experiments/branch-D.2/`](experiments/branch-D.2/) — 15-channel
  enhanced encoding (2026-05-24 → 25).

### Archive

Completed or superseded. Kept for history but not part of the active mental
model.

- [`archive/WAVE3_EXPERIMENT_LOG.md`](archive/WAVE3_EXPERIMENT_LOG.md)
- [`archive/WARMSTART_PHASE_LOG.md`](archive/WARMSTART_PHASE_LOG.md)
- [`archive/OVERNIGHT_RESULTS.md`](archive/OVERNIGHT_RESULTS.md)
- [`archive/MODEL_PLAY_OBSERVATIONS.md`](archive/MODEL_PLAY_OBSERVATIONS.md)
- [`archive/REMOTE_TRAINING_INVESTIGATION.md`](archive/REMOTE_TRAINING_INVESTIGATION.md)
- [`archive/CLOUD_TRAINING_PLAN.md`](archive/CLOUD_TRAINING_PLAN.md) — early
  cloud-training plan, superseded by the vast.ai workflow now in use.
- [`archive/RESEARCH_LOG.md`](archive/RESEARCH_LOG.md)
- [`archive/ANALYSIS_SUMMARY_100K_GAMES.md`](archive/ANALYSIS_SUMMARY_100K_GAMES.md) — the
  100K-game statistical analysis that informed the heuristic weight values
  still in use (`yinsh_ml/heuristics/weight_manager.py`). Source of truth
  for *why* those weights; not actively edited.
- [`archive/bitboard-port/`](archive/bitboard-port/) —
  `BITBOARD_PORT_PROMPT.md`, `BITBOARD_FOLLOWUP_PLAN.md`,
  `POST_BITBOARD_TRAINING_BRIEF.md`. Shipped; no active work.

### TODOs (root-level, retained for habit)

- [`../TODO_baseline.md`](../TODO_baseline.md) — baseline-level cleanups.
- [`../TODO_frontier.md`](../TODO_frontier.md) — frontier explorations.
- [`../NEXT_UP.md`](../NEXT_UP.md) — short list for the next session.

(*Possible consolidation:* fold all three into a single TODO section here.
Defer to the cleanup pass.)

---

## How to maintain this file

This is a **live index**. Three update triggers:

1. **At the end of every session:** update the "Where am I?" lines at the top
   (current anchor, last verdict, next experiment). 30 seconds.
2. **When a SPRT verdict lands:** add a row to the Timeline. 1 minute.
3. **When you add or resolve an experiment in the backlog:** the linked
   `plans/EXPERIMENT_BACKLOG.md` is the source of truth; just make sure the
   Top-5 list in [Forward queue](#forward-queue) stays current.

**What this index is NOT:**

- Not a project plan (use `plans/EXPERIMENT_BACKLOG.md` + `plans/ROADMAP_TO_ALPHAZERO.md`).
- Not a result log (use the per-experiment `RESULTS.md` files).
- Not a TODO list (use `TODO_*.md` for that).

If you find yourself writing more than a paragraph here for something, it
belongs in one of the linked docs instead. The index's job is to *point*, not
to *contain*.

---

## Status headers (proposed convention)

After the cleanup pass, every doc gets a status header so its state is
machine-greppable:

```yaml
---
status: active | archived | superseded-by: <path>
last-updated: 2026-05-25
owner: jack
---
```

Greppable:
```bash
grep -l "^status: active" docs/**/*.md
grep -l "^status: superseded-by:" docs/**/*.md
```

That's the entire convention. Don't over-engineer.
