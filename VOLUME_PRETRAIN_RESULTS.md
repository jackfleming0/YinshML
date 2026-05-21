# Volume-Corpus Pretraining + Branch C Rerun — Results & Next Steps

Run dates: 2026-05-19 → 2026-05-21. Branch: `training-pipeline-fixes`.
Predecessor context: `WAVE3_CATCHUP.md` (the deferred "option 2" this executes).

Read this first if picking up cold. The companion new-session prompt is at the
bottom (§"Prompt for the next session").

---

## TL;DR

Two experiments, run back-to-back on a rented vast.ai RTX 4090 (~$30 total):

1. **Option 2 — volume-corpus value-head pretraining.** Trained a network from
   scratch on 13.6M yngine self-play positions. The resulting checkpoint
   **swept the entire heuristic anchor ladder: 100% vs HA(d=1), d=2, and d=3**
   (40 games each, MCTS-64). Strongest checkpoint the project has produced —
   beats the prior best (Branch C original iter 1 at 82.5%).

2. **Option 1 — Branch C MCTS-200 self-play rerun, warm-started from that
   pretrained checkpoint.** 5 iterations, 16.1h. The recipe **HOLDS the
   strength but does NOT compound it**: anchor stayed 100% every iteration (no
   collapse — the original Branch C dropped 82.5%→60%), but internal Elo was
   flat (1500 → 1633 → 1477 → 1483 → 1506; the iter-1 spike is noise). iter 4
   preserved the full d1/d2/d3 = 100% sweep.

3. **Human gut-check passed.** The pretrained/best checkpoint beats a competent
   human (the author, "okay at the game") with play that feels legitimate, not
   exploity — confirming the 100% ladder isn't an artifact of gaming the
   heuristic.

**Net:** we have a genuinely strong, stable YINSH bot, built cheaply. The
pretrained init did the work; self-play treads water on top of it. The
bottleneck is now the **ceiling**, not recipe stability — and, critically, we
**can no longer measure improvement** because every yardstick we have is
saturated.

---

## What we did, in detail

### Option 2: volume-corpus pretraining

- **Corpus**: `expert_games/yngine_volume.npz` — 13.6M positions from 200K
  yngine self-play games. Stored as int32 `policy_indices` (argmax move), not
  (N, 7433) one-hot (which would've been ~400GB).
- **Code landed** (commits `2e9b248`, `e76af9e`):
  - `scripts/run_supervised_pretraining.py` now accepts the `policy_indices`
    schema (branches to `F.cross_entropy` on integer targets) and a `--data-dir`
    flag that memmaps `.npy` shards (the corpus decompresses to ~40GB; memmap
    keeps RAM flat).
  - `scripts/convert_npz_to_mmap_shards.py` — streams an npz → `.npy` dir.
- **Training decision**: trained **from scratch**, NOT warm-started from
  `supervised_seed` — the seed's quality was uncertain and yngine is a
  known-decent engine, so its 13.6M games are the better foundation. (This was
  a deliberate call; revisit if the seed is later shown stronger.)
- **Config**: 3 epochs, batch 512, LR 1e-3 cosine, val-split 0.02, 256ch×12
  blocks, 6-channel basic encoding. ~3h on the 4090.
- **Result**: final val PAcc 0.286, **VAcc 0.629** (chance = 0.333 on the
  3-class value head). VAcc climbed every epoch (0.612 → 0.619 → 0.629), no
  overfit — the volume corpus genuinely grounded the value head.
- **Validation gate** (`models/yngine_volume_pretrain/best_supervised.pt` vs
  HeuristicAgent, MCTS-64, n=40 each):
  - **d=1: 40/40 (100%)**, d=2: 40/40 (100%), d=3: 40/40 (100%). All CI95 [0.91, 1.0].

### Option 1: Branch C rerun from the pretrained init

- **Launch**: `scripts/run_training.py --config configs/wave3_branchC_mcts200.yaml
  --init-checkpoint models/yngine_volume_pretrain/best_supervised.pt`. Warm-start
  loaded 238/238 tensors clean.
- **Recipe** (one point in hyperparameter space): MCTS sims 200 (early) / 100
  (late), 100 games/iter, 1 epoch/iter, 5 iters, base LR 1e-4 cosine warmup=1,
  EMA decay 0.999, promotion gate Wilson 0.20 (loose), anchor skips iter 1.
- **Run dir**: `runs_wave3_branchC/20260520_044946/` (5 iters, promotions 5/5, 16.1h).

#### Anchor (vs HA d=1) — STABLE

| iter | anchor MCTS WR | note |
|---|---|---|
| 1 | (skipped by config) | `skip_first_n_iterations: 1` |
| 2 | 100% (40/40) | |
| 3 | 100% (40/40) | |
| 4 | 100% (40/40) | best iter |
| 5 | 100% (40/40) | |

No collapse. The original Branch C's central failure (82.5% → 60% by iter 2) is
fixed by the value-grounded init.

**Parsing gotcha**: anchor WRs are NOT in `metrics.json` (`anchor_eval={}` stays
empty) — they're only in the run log (`logs/branchc_yngine_init.log`), lines
`Anchor vs HeuristicAgent(d=1):` and `ANCHOR (prev,raw/mcts):`.

#### Internal Elo — FLAT

| iter | tournament_rating | tournament_win_rate |
|---|---|---|
| 0 | 1500.0 | 0.00 |
| 1 | 1633.3 | 0.70 |
| 2 | 1477.4 | 0.45 |
| 3 | 1483.1 | 0.46 |
| 4 | 1505.6 | 0.51 |

Settled at ~1500 = the warm-started baseline. The iter-1 spike is within the
noise band (20 games/match, window 3 → ±150 swings). **Self-play produced ~zero
net Elo gain.**

#### Best-iter ladder (iter 4, MCTS-64, n=40)

d=1: 100%, d=2: 100%, d=3: 100%. **Preserved the pretrained sweep exactly** —
self-play neither improved nor eroded harder-opponent strength.

---

## Verdict + the epistemic caveat (important)

**Confident claim**: the recipe does not *collapse*. Stability is well-measured
(anchor 100% every iter, full ladder preserved). The original Branch C failure
mode is genuinely fixed by the value-grounded init.

**Weaker claim, do NOT overstate**: "plateaued / doesn't compound." This is a
statement about **one recipe configuration**, measured with **saturated external
anchors** (HA ladder pinned at 100% — can't discriminate stronger models) and a
**noisy internal Elo** (5 points clustered in the noise band). "Truly plateaued"
and "improving in ways our instruments can't see" are **observationally
identical** given the current measurement. We have NOT shown that more MCTS sims,
more epochs, or different iteration lengths wouldn't help — we've shown that *this
config, measured this way,* didn't visibly move.

In particular **MCTS-1000 self-play targets** have a strong prior: the pretrained
net is already strong, so its MCTS-200 self-play produces targets ≈ its own
policy (you can't learn from a teacher who plays like you). To improve, search has
to *outrun* the current policy — which is exactly what more sims buy. 200 may
simply not have been a big enough jump over an already-strong init.

**The real bottleneck right now is measurement.** The model has outgrown the
entire heuristic ladder AND a competent human. Before any ceiling experiment, we
need a **non-saturated yardstick** — otherwise MCTS-1000 vs MCTS-200 would just be
two flat noise-bands and we'd learn nothing.

---

## Artifacts (all local on laptop unless noted)

**Checkpoints:**
- `models/yngine_volume_pretrain/best_supervised.pt` — the pretrained checkpoint
  that swept the ladder (also `supervised_epoch_{1,2,3}.pt`, `supervised_final.pt`).
- `models/branchC_volume_pretrain/best_iter_4.pt` — Branch C best iter (the one
  that beat the human).
- `runs_wave3_branchC/20260520_044946/checkpoint_iteration_{0,1,2,3}_ema.pt` —
  intermediate Branch C iters (iter 4 = best_iter_4.pt above). iter 1 had the
  Elo spike (1633).

**Eval JSONs / logs:**
- `logs/validation_gate.json` — pretrained checkpoint d=1 eval.
- `logs/branchC_iter4_d{1,2,3}.json` — iter-4 ladder (all 100%).
- `logs/branchc_yngine_init.log` — full Branch C run log (anchor WRs live here).
- `runs_wave3_branchC/20260520_044946/{metrics.json per iter, manifest_final.json,
  suggestions_iter_5.yaml}`.

**Data / tooling:**
- `expert_games/yngine_volume.npz` — the 13.6M-position corpus.
- `scripts/play_dashboard.py` — clickable Plotly play-vs-model app (the gut-check
  tool). Run: `streamlit run scripts/play_dashboard.py`.
- `scripts/spot_check_model_games.py` — model-vs-HA → parquet for dashboard replay.
- `scripts/convert_npz_to_mmap_shards.py` — npz → memmap `.npy`.

**Repo state**: branch `training-pipeline-fixes`. Pushed through `4b46d3a`.
Local-only (not pushed): `5175ed4`, `f4654c1` (both `play_dashboard.py`). Push if
you want the tooling on the remote.

**Cloud**: vast.ai instance 37092392 — TERMINATE via web UI if not already done
(it survived a maintenance window and may still be billing). Everything is backed
up locally; nothing on the box is needed. The 40GB `yngine_volume_mmap/` and the
run dir there are redundant with local copies.

---

## Session update — 2026-05-21: yardstick built (step 1 DONE)

**Decision: frozen-checkpoint anchor, NOT yngine (yet).** The gating question for
the next move is *relative* ("does lever X beat our current best?"), not absolute
("where's the ceiling?"). A frozen `best_iter_4` anchor answers the relative
question exactly and cheaply — `~0.50` WR = no improvement, `>0.55` = genuinely
stronger, and if a candidate saturates the anchor you re-freeze to it and climb
again (the standard AlphaZero iterate-and-promote ladder). yngine is the only
*absolute* ceiling-detector, but it's not local (box torn down, no binary/runner
in repo — only the offline shard parser `scripts/yngine_corpus_to_npz.py`), and
building it = clone+build temhelk/yinsh on macOS + write the deferred V2b C++
stdin/stdout protocol driver + Python bridge. **Defer-yngine trigger:** build the
bridge only when (a) a long scaled self-play run needs a collusion tripwire, or
(b) a candidate convincingly beats frozen-best and we want the absolute number.

**What shipped:**
- `scripts/eval_vs_frozen_anchor.py` — candidate(s) vs a fixed anchor (default
  `best_iter_4.pt`), n split by color, reports WR + Wilson CI + STRONGER / WEAKER /
  inconclusive verdict. Built on the **batched MCTS** (`self_play.MCTS.search_batch`),
  the same engine `run_anchor_eval` uses — see the bug note below. Opening moves are
  sampled (`--opening-sample-plies`, default 20) then greedy, to defeat the
  deterministic-side artifact. Per-game `gc`+`torch.mps.empty_cache()` so long runs
  don't OOM. Cost on MPS: ~30 min/candidate at n=40 @ 64 sims (≈16–23 s/game).
- **⚠ Found a serious bug: the legacy MCTS `search()` is broken** (separate doc:
  `TECH_DEBT.md`). My first cut reused `eval_head_to_head_mcts.py`, which calls
  `yinsh_ml/search/mcts.py::MCTS.search()` — that method **never expands the root**
  and returns a **uniform-random, net-blind** policy (`effective_child_visits=0.0`;
  every checkpoint plays identically; verified). Its prior "white-wins pattern"
  finding is an argmax-on-uniform artifact. **Blast radius is contained:** self-play
  training and the HA-ladder anchor gates both use `search_batch` (`use_batched_mcts`
  defaults `True`), so **this doc's headline results are NOT contaminated.** The
  broken path only fed `eval_head_to_head_mcts.py` (diagnostic) and a test-only
  `MCTSPolicy`. The bug surfaced only because the yardstick build included a positive
  control — see below.
- **Validation — the yardstick discriminates** (batched MCTS, opening-sampled):
  - *Negative control* — `best_supervised` vs `best_iter_4`: inconclusive (CI spans
    0.5; point est. 0.55 @ n=40/64sims, 0.70 @ n=20/32sims — both CIs include 0.5).
    Correct: both sweep the HA ladder, ≈ equal.
  - *Positive control* — abandoned `supervised_seed` and `supervised_seed_humans_only`
    vs `best_iter_4`: **0/20 = 0.000, WEAKER** (CI95 ≤ 0.161), **balanced across
    colors** (0 as White, 0 as Black) — genuine strength gap, not the color artifact.
  - **Caveat — gradient is coarse for now.** It cleanly separates "comparable" from
    "much weaker," but n=20 CIs are ±0.37 wide, so resolving a *small* real gain
    (e.g. 0.55 vs 0.50) needs higher n — and at ~30 min/candidate on MPS that means
    the GPU box for the real step-2 measurement.
  - JSONs: `logs/frozen_anchor_disc/*.json`.
- **Internal Elo beefed:** `configs/wave3_branchC_mcts200.yaml` `arena.games_per_match`
  20 → 100 (tournament is raw-policy, cheap; shrinks the ±150 band ~2.2×).
- **SPRT mode added + validated** (`--sprt`). Sequential test (H0 p0=0.5 vs H1 p1,
  α=β configurable, Bernoulli LLR over decisive games, colors alternate per game)
  that stops the moment the result is decisive instead of paying fixed-n for every
  candidate — the fishtest approach, the load-bearing fix for the coarseness. Both
  boundaries validated with p1=0.65: `best_iter_4` vs seed → STRONGER in 12 games;
  seed vs `best_iter_4` → NOT_STRONGER in 9 games (LLR increments match theory).
  JSONs: `logs/sprt_{upper,lower}.json`. (Deferred, theory-free option for further
  variance reduction: a shared on-distribution opening book / common-random-numbers
  pairing — only add if SPRT alone still burns too many games.)

**Now unblocked → the actual experiment.** Run MCTS-1000 (or 400) self-play
targets (strongest prior for raising the ceiling) and/or Branch D architecture,
then screen each new candidate vs frozen `best_iter_4` with
`eval_vs_frozen_anchor.py --sprt` (set `--sprt-p1` to the smallest edge worth
promoting; clear cases stop fast, borderline ones run to `--sprt-max-games`).
Likely on GPU — borderline cases can still need many games, and MPS is ~16–50 s/game.

---

## Next steps — raising the ceiling

The ordered plan. Step 1 is a prerequisite for everything after it.

### 1. Build a non-saturated yardstick — ✅ DONE 2026-05-21 (frozen anchor; see session update above)

We've outgrown the heuristic ladder. Without a measurement that can distinguish
"stronger" from "this strong," every experiment below produces flat noise.
Options (pick one or combine):
- **yngine head-to-head** — the engine the corpus came from; plausibly stronger
  than HA(d=3). A `checkpoint vs yngine` eval harness, n≥40, both colors. This is
  the most informative and reuses an engine we already trust.
- **Frozen-checkpoint anchor** — freeze `best_iter_4.pt` (or the pretrained) as a
  fixed opponent; future runs measure WR against it. Cross-run comparable.
- **Beefed-up internal Elo** — many more games/match (e.g. 100+) so the ±150 noise
  band shrinks enough to read a real trend.

### 2. Ceiling-raising experiments (now measurable)

Once step 1 gives a gradient we can see:
- **MCTS-1000 (or 400) self-play targets** — strongest prior. Test whether a
  stronger teacher than the already-strong init's own policy produces targets the
  net can learn from. Costly (~5× MCTS-200 per-move), but the direct test of the
  "stronger teacher" thesis.
- **Branch D — architecture** (audit Gap 1): SE blocks + global-average-pool value
  head. Raise the representational ceiling. See `ARCHITECTURAL_IMPROVEMENTS_PLAN.md`.
- **Secondary levers**: more epochs/iter, longer iters (more games → less
  overfit), tighter promotion gate (Wilson 0.20 is too loose for a real run).

### 3. If scaling the AlphaZero loop

Branch C *is* an AlphaZero self-play loop; the question was always whether to
*scale* it (20-50+ iters). Only worth it if step 1 shows the model is still
climbing. If you do:
- Internal Elo = dense per-iter progress signal.
- Harder external anchor (yngine / frozen checkpoint) = absolute-strength tripwire,
  run every N iters. Guards against self-play collusion (the population drifting
  into a shared blind spot while internal Elo rises — invisible to internal Elo by
  construction).

### Deferred / not recommended
- **Naive scale-up** of the exact Branch C recipe — flat Elo says more iters of
  the same won't compound. Don't, until a lever (sims/arch) changes the picture.

---

## Prompt for the next session

> I'm continuing the YINSH ceiling-raising work. Read `VOLUME_PRETRAIN_RESULTS.md`
> first — it has the full results of the volume-corpus pretraining + Branch C
> rerun and the next-step plan.
>
> Short version: we have a strong, stable bot
> (`models/branchC_volume_pretrain/best_iter_4.pt` — sweeps the HA d1/d2/d3 ladder
> at 100% and beats a competent human) but it's plateaued *as far as we can
> measure* — every yardstick is saturated. So before any ceiling experiment
> (MCTS-1000 targets, Branch D architecture), the immediate task is **step 1 from
> that doc: build a non-saturated yardstick.**
>
> Specifically: write an eval harness that plays a checkpoint head-to-head against
> **yngine** (the engine our corpus came from), n≥40 split by color, and reports
> WR + Wilson CI — analogous to `scripts/eval_vs_heuristic.py` but with yngine as
> the opponent instead of HeuristicAgent. Check whether a yngine-play harness or
> binary already exists in the repo (the corpus was generated from yngine
> self-play, so there's a translator/runner somewhere) before writing from scratch.
> Then run `best_iter_4.pt` and the pretrained `best_supervised.pt` against it to
> get our first non-saturated strength numbers.
>
> Operational notes: the old vast.ai box is torn down — spin a fresh instance only
> if a real run is needed (the yngine eval may be cheap enough to run locally on
> MPS). All checkpoints + corpus are local (see the doc's Artifacts section).
> Branch `training-pipeline-fixes`; two play_dashboard commits are local-only if
> you want them pushed. Don't re-litigate: training from scratch (not seed
> warm-start) was deliberate; the recipe-doesn't-compound finding is measurement-
> limited, not proven — see the doc's epistemic caveat.
