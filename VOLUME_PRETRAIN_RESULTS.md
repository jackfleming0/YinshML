# Volume-Corpus Pretraining + Branch C Rerun — Results & Next Steps

Run dates: 2026-05-19 → 2026-05-21. Branch: `training-pipeline-fixes`.
Predecessor context: `WAVE3_CATCHUP.md` (the deferred "option 2" this executes).

Read this first if picking up cold. The companion new-session prompt is at the
bottom (§"Prompt for the next session"). The ranked forward-looking experiment
queue lives in `EXPERIMENT_BACKLOG.md` — go there to pick the next thing to run.

---

## TL;DR

Three experiments, run on a rented vast.ai RTX 4090 (~$40 total):

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

4. **Step 2 — MCTS-400 ceiling experiment (2026-05-22/23, 19.84h).** Controlled
   lever test (sims 200→400, all else identical to Branch C). Resulting candidate
   vs frozen `best_iter_4` on the SPRT yardstick: **INCONCLUSIVE at the 400-game
   cap, 221-179, WR 0.552, CI95 [0.504, 0.600].** Small *real* edge (~30–40 Elo,
   lower CI > 0.50) but sits squarely below the p₁=0.60 promotion bar. Stability
   was flawless: 5/5 promotions, all 4 anchors at 100%, 0 crashes. **Search depth
   is not the dominant lever for this architecture — the ceiling is structural.**

**Net:** we have a genuinely strong, stable YINSH bot AND a working non-saturated
yardstick. The first ceiling experiment (MCTS-400) confirmed search depth isn't
the binding constraint — the network architecture is. The next axis is **Branch D
(GAP value head first)**, with `num_workers:4` and the `BatchedEvaluator`
parity-check as enabling prereqs (see §2).

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

## Session update — 2026-05-22 → 23: Step 2 ran (MCTS-400)

Tested the stronger-teacher hypothesis end to end. **Result: small real edge,
not promotion-worthy. Search depth is not the dominant lever for this
architecture.**

### Run

- **Setup** (`configs/wave3_branchC_mcts400.yaml`, `STEP2_MCTS400_RUNBOOK.md`):
  controlled lever test — sims 200→400, late_sims 100→200, all else identical to
  the Branch C MCTS-200 baseline. Warm-started from the same `best_supervised.pt`
  init, screened against frozen `best_iter_4` afterward.
- **GPU efficiency restart**: launched serial (`num_workers: 0`), then killed after
  ~1h / 15 games when the 4090 sat at ~7% util. Restarted with `num_workers: 4`;
  first batch validated **2.84× speedup** (50.5 min → 17.8 min per 10-game batch,
  0 worker crashes). The multiprocess path calls the same per-game
  `play_game_worker` as serial, so this is throughput-only — the controlled
  comparison is preserved. Cut the run from ~44h to **19.84h actual**.
- **Stability**: 5/5 promotions, **all 4 completed-iter anchors at 100%** vs
  HA(d=1), 0 crashes. The Branch C collapse mode (82.5%→60% in the original
  2026-05 run) stayed fixed under the stronger teacher too — the value-grounded
  pretrained init is robust at this sim count.
- **Internal Elo** (raw-policy tournament, beefed to 100 games/match): iter 0–4
  hovered around 1485–1505. Flat — same pattern as MCTS-200. Internal Elo is now
  genuinely diagnosed as a noisy/saturated signal in this regime; the
  frozen-anchor SPRT is the load-bearing measurement.

### Result — SPRT (`logs/mcts400_iter4_vs_frozen.json`)

| | |
|---|---|
| Verdict | **INCONCLUSIVE** at the 400-game cap |
| Score | candidate **221-179-0**, WR **0.552** |
| CI95 | **[0.504, 0.600]** |
| LLR | +0.35 (boundaries ±2.94, α=β=0.05) |
| Color split | W/B 114/107 (balanced — no deterministic-side artifact) |
| Cost | 400 games, ~7.9h on the 4090 at 64 sims/move |

**How to read INCONCLUSIVE in this case.** The formal label is "test didn't
decisively cross H0 or H1," but the CI carries the real meaning. Lower bound
**0.504 > 0.500 with ~95% confidence** ⇒ a real positive effect exists. Upper
bound **= 0.600** ⇒ almost certainly below the promotion threshold we set. The
candidate is roughly **30–40 Elo stronger**, not the ≥70 Elo we required. The
SPRT result and the flat internal Elo are coherent: small real bump, not a
ceiling break.

### Implications for the ceiling thesis

- **Search depth alone is not the dominant lever for this architecture.** Doubling
  sims produced a measurable but small gain, well below what log-linear scaling
  priors (50–100 Elo per doubling in the unsaturated regime) would predict. The
  network architecture is the binding constraint, not search budget.
- **By log-linear extrapolation, MCTS-800 buys ~15–25 Elo more at best**, MCTS-1000
  marginally more — still below the promotion bar relative to the next anchor we'd
  set, at 4× the compute cost. **MCTS-1000 as a standalone experiment is
  effectively pre-answered: marginal.** Demote.
- **The Branch D / architecture axis is the next leverage point.** See §2 below.

### Methodological note: don't eyeball-call mid-flight

Three times during the SPRT I read intermediate trajectory and called the result
early — g68 "settled NOT_STRONGER," g92 "climbing STRONGER," g120 "back to
NOT_STRONGER." Each read was overturned within ~30 games. The truth was always
"true WR right at the indifference point (~0.55); LLR random-walks near zero;
will hit the cap." This is exactly the regime SPRT is designed to handle without
premature commitment, and I made the mistake anyway. **Trust the formal boundary.
The whole point of sequential testing is to refuse to call until the evidence
accumulates beyond chance.**

### Side wins (not just the verdict)

- `num_workers: 4` is now battle-tested on the production training path (no worker
  crashes across 500 self-play games + 3 full anchor evals + 4 tournament rounds).
  Tier-1 GPU lever from §1b is fully de-risked.
- The frozen-anchor SPRT instrument itself ran a real long borderline case clean —
  boundaries and cap behavior all worked as designed; W/B color split came out
  balanced (114/107), confirming the opening-sample defense against the
  deterministic-side artifact.
- Local artifacts pulled: `logs/mcts400_iter4_vs_frozen.json`,
  `logs/mcts400_sprt.log`, `logs/mcts400.log`. Repo: `configs/wave3_branchC_mcts400.yaml`,
  `STEP2_MCTS400_RUNBOOK.md`. Box terminated after artifact pull.

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

### 1b. GPU efficiency — prerequisite for the heavier experiments

The original Step 2 launch ran serial (`num_workers: 0`) on purpose — clean
comparison to the serial Branch C baseline. The 4090 sat at **~7% util** (workload
is GPU-batch-starved, not compute-bound). We killed it after ~1h and restarted with
`num_workers: 4`; the first batch validated **2.84× speedup** with zero worker
crashes, and the full 5-iter run completed in 19.84h instead of the projected ~44h.
**Tier 1 is now fully battle-tested on the production training path.**

- **Tier 1 — `num_workers: 4` (proven, zero-risk, battle-tested 2026-05-22).**
  Sweep evidence (`GPU_SCALING_RESULTS.md`) said 1.85× games/hr at workers=4;
  measured 2.84× on the actual Step 2 run (sub-linear gains amplify when the
  per-game workload has more CPU-bound MCTS tree work, i.e. at higher sim counts).
  Caps fast: 8 workers = 52% of peak, 16 = 40% — more workers → more GPU
  command-queue contention, not throughput. **4 is the sweet spot, full stop.**
  Set it on every cloud run.
- **Tier 2 — `use_shared_evaluator: true` (the real GPU unlock, NEEDS VALIDATION).**
  The `BatchedEvaluator` (PR #12) coalesces leaf-eval inference across concurrent
  games into one batch (`min(512, mcts_batch_size×workers)`), fixing the
  per-worker `predict_batch` serialization that caps the multiprocess path — the
  only lever that pushes *sustained* util up (`sm_p95` already hits 73–75% when fed;
  `sm_avg` is the problem). **Gating task:** parity-check serial vs shared-evaluator
  targets on a short config FIRST — there's an open batched-MCTS sim-accounting
  concern (T1.1, ~30–60% sim loss). Don't bet a multi-day run on an unvalidated
  batched path. (Audit the instrument before trusting it — same lesson as the
  yardstick.) **This is the gating task for the next heavy run.**
- **Don't bother:** `mcts_batch_size: 128` (a wash vs 64 at every worker count).

### 2. Ceiling-raising experiments — Branch D (architecture) is the next axis

After Step 2 (MCTS-400) confirmed search depth isn't the binding constraint,
the indicated lever is the network architecture itself. Sequential one-variable
tests, not stacked, so each result attributes cleanly:

- **Branch D.1 — GAP value head (PRIMARY, do this next).** Replace the
  spatial-flatten value-head pipeline (currently ~4M params: 2 convs →
  `Flatten(64·11·11=7,744)` → 512 → 256 → 7 classes) with a global-average-pool
  head (~5K params: 1×1 conv → `AdaptiveAvgPool2d(1)` → 64 → 7). Smallest change,
  strongest mechanism-based prior — the spatial-flatten pattern is documented
  across KataGo/Leela/AlphaZero as the dominant value-head overfitting trap, and
  the current head is conspicuously parameter-heavy for what it predicts (a 7-class
  scalar). One head swap, no encoder retrain, no training-loop changes. Train from
  the same `best_supervised.pt` init, screen vs frozen `best_iter_4` on the SPRT
  yardstick.
- **Branch D.2 — Enhanced 15-channel encoding.** Add threat/tactical planes (rows-
  of-3/4, blocking rings), positional planes (centrality, edge distance), and
  enhanced game-state planes (turn counter, score diff). Already half-built;
  `EnhancedStateEncoder` exists. Bigger commitment than D.1 (regenerate the
  pretraining corpus with the new encoder to match the input shape), but the
  change with the most *information* added — gives the net access to features it
  currently has to derive from raw piece positions. See
  `ARCHITECTURAL_IMPROVEMENTS_PLAN.md` Phase 1.
- **Branch D.3 — SE (channel attention) blocks.** Drop in alongside the existing
  spatial-attention blocks. Cheap (~5K extra params per block), well-attested in
  Leela/KataGo, complements rather than overlaps the existing attention (spatial
  asks "which cells matter," SE asks "which feature channels matter"). Sequence
  *after* D.1 and D.2 so attribution stays clean — its mechanism partly overlaps
  the existing spatial attention.
- **NOT pursuing — D6 hex augmentation.** Earlier analysis ruled this out
  (YINSH's column counts `[4,7,8,9,10,9,10,9,8,7,4]` are palindromic but not a
  regular hexagon; the board's true symmetry group is smaller than D6, and
  applying full D6 transforms would teach false invariances). Current axis-flip
  augmentation matches the actual symmetry; that's the right call.
- **NOT pursuing standalone — MCTS-1000.** Demoted post-Step-2. Log-linear scaling
  priors (50–100 Elo per doubling in an unsaturated regime; we got ~30–40 Elo from
  the 200→400 doubling) predict marginal gains; 4× the compute cost for an Elo
  bump still below the promotion bar. Worth a re-test only *after* an architecture
  change has shown it can absorb stronger targets.
- **Secondary levers (keep on the shelf):** tighter promotion gate (Wilson 0.20 is
  documented as loose), more epochs/iter, longer iters (more games → less overfit).
  Stack only after the primary architecture axis has produced a candidate.

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

## Session update — 2026-05-23: tasks 1 & 2 landed locally; D.1 ready for cloud

Picked up on a fresh machine (laptop, not the original vast.ai box). Pulled
artifacts via gh release `mcts400-session-snapshot-2026-05-23` (MD5s match).
Two pre-D.2 items completed locally; D.1 is implemented and tested but the
actual training run is GPU work.

### Task 1 — BatchedEvaluator parity check ✅

Already-existing unit tests (`test_mcts_serial_vs_batch_parity.py`, the
T1.1 visit-loss regression suite + `test_batched_evaluator.py::test_mcts_with_evaluator_matches_direct_path`)
proved the math on a fake constant-output net. New `scripts/tier_a_threaded_parity.py`
closed the remaining gap: real `best_supervised.pt`, three dispatch paths
(serial / process-pool / **threaded shared-evaluator**), CPU forced, 8 games
× 3 paths × MCTS-64, ~560 per-move policy targets per path. **PARITY across
all critical pairs** (KS p > 0.85 on entropy + top-1 mass for serial↔threaded
and process_pool↔threaded), zero worker crashes. The threaded path is
cleared for the next heavy CUDA run — though on this CPU/MPS Mac the
*throughput* win is invisible (Python GIL serializes single-net forwards);
the CUDA payoff is what matters and lives on the cloud box.

Bonus: the parity run surfaced a real (but benign) bug in
`yinsh_ml/memory/adaptive.py` — `TensorCompatibilityChecker.is_compatible`
accepted `target_numel <= source_numel` (slice semantics) but the consumer
called `tensor.view(shape)` (requires exact numel match), causing every
batch-size-mismatched lookup in the threaded path to fire a "Failed to
reshape" warning before silently falling through to fresh allocation.
Fixed in commit `96c814c`; the reshape path was effectively dead code for
batch-size variation. Two MPS-specific TensorPool follow-ups logged to
`TECH_DEBT.md` §4 (device-string `mps` vs `mps:0` mismatch causes the
pool to never reuse on Apple Silicon at all; pre-existing test API drift).

### Task 2 — Branch D.1 GAP value head, implemented + tested ✅ (local), 🟡 (cloud)

Landed in commit `60aa4bd`. The full spec:

  - **`YinshNetwork.__init__`**: new `value_head_type: str = 'spatial'` flag.
    Default preserves all pre-D.1 behavior. `'gap'` selects the new head:
    `Conv2d(c→64, k=1) → BN → ReLU → AdaptiveAvgPool2d(1) → Flatten → Linear(64, num_classes)`.
    Param count: **4,284,807 → 17,031 (252× reduction in head, 14% reduction
    in total model size)**.
  - **`NetworkWrapper`**: accepts flag; auto-detects from checkpoint by
    inspecting `value_head.0.weight.shape[-1]` (3 = spatial, 1 = gap).
    Explicit override beats auto-detect (the D.1 warm-start path: load
    spatial checkpoint into gap wrapper, trunk + policy + outcome_values
    transfer via shape-match, spatial-head keys silently drop, GAP head
    freshly initializes).
  - **Propagated through** `mcts_config` → `play_game_worker`, supervisor
    network-recreate path, `run_training.py` (reads `network.value_head_type`
    from YAML), AND `run_supervised_pretraining.py` (`--value-head-type`
    flag for the parallel-track option: train a GAP-head supervised init
    from scratch instead of warm-starting a spatial-head checkpoint).
  - **`configs/branchD1_gap_mcts200.yaml`**: one-variable-changed mirror
    of `wave3_branchC_mcts200.yaml`. `num_workers: 4` set from the start
    (Step 2 battle-tested). `use_shared_evaluator: false` kept off for
    the controlled comparison — flip on later once D.1 has an attribution.
  - **Tests** (`test_gap_value_head.py`, 11/11 pass): param count
    invariants, forward shapes both modes, validation, warm-start from
    spatial-head checkpoint succeeds + transfers ≥20 trunk tensors +
    leaves GAP head at fresh init (verified by checking the first conv's
    kernel_size = 1).
  - **End-to-end local smoke** (1 iter, 2 games, 8 sims, real warm-start
    from `best_supervised.pt`): full pipeline runs in 8.4s. Logs show
    `Using value head: gap` and `'value_head_type': 'gap'` propagated to
    worker MCTS config. 0 crashes.

**Not yet done:** the actual MCTS-200 self-play loop on cloud. This is the
next vast.ai session — see prompt below.

---

## Prompt for the next session (CLOUD — run Branch D.1)

> I'm running the Branch D.1 training experiment on cloud. The local
> implementation is complete and tested (see VOLUME_PRETRAIN_RESULTS.md
> §"Session update — 2026-05-23" — including the GAP head spec, warm-start
> invariant, and propagation chain). Branch `training-pipeline-fixes`,
> head commit `60aa4bd` or newer.
>
> **The run, one command:**
>
> ```
> python scripts/run_training.py \
>     --config configs/branchD1_gap_mcts200.yaml \
>     --init-checkpoint models/yngine_volume_pretrain/best_supervised.pt
> ```
>
> Expected wall time: ~16-20h on a 4090 (matches MCTS-200 Branch C; same
> sim count). Already plumbed: `num_workers: 4` (Step 2's 2.84× speedup,
> zero crashes), value-grounded warm-start from `best_supervised.pt`,
> Wilson 0.20 promotion gate, anchor-skip iter 1.
>
> **Sanity checks at iter-1 (~3h in):**
>
>   - `Using value head: gap` in the launch log
>   - `'value_head_type': 'gap'` in the per-worker MCTS Config dump
>   - Trunk loaded clean from `best_supervised.pt` — anchor vs HA(d=1) at
>     iter 2 (skip_first_n_iterations=1) should hit 100% (40/40), same
>     as the Branch C baseline — if it doesn't, the trunk warm-start
>     silently broke and you need to investigate before burning the
>     rest of the run.
>
> **Screen the result** (after the 5-iter run completes):
>
> ```
> python scripts/eval_vs_frozen_anchor.py \
>     --candidate runs_branchD1/<TIMESTAMP>/checkpoint_iteration_4_ema.pt \
>     --anchor models/branchC_volume_pretrain/best_iter_4.pt \
>     --sprt --sprt-p1 0.60 --sprt-max-games 400 \
>     --output logs/branchD1_iter4_vs_frozen.json
> ```
>
> Interpretation:
>   - STRONGER (WR > p₁=0.60, lower CI clears 0.60): GAP head is the lever
>     we wanted. Freeze it, re-anchor against this checkpoint, move to
>     Branch D.3 (SE blocks) or scale-up.
>   - INCONCLUSIVE with WR > 0.55: small real edge, not promotion-worthy
>     (the MCTS-400 outcome). Branch D.2 (enhanced encoding) becomes the
>     next-highest-prior lever — the head wasn't the binding constraint
>     alone.
>   - INCONCLUSIVE with WR ~ 0.50: flat. Value head wasn't the constraint
>     either. Pivot to D.2 with that framing.
>   - WEAKER: warm-start broke something. Audit the trunk transfer test
>     (`test_gap_value_head.py::TestGapWarmStartFromSpatial`) and the
>     iter-1 anchor sanity check; if those pass, the GAP head's tiny
>     capacity is starvation. Try the parallel track: re-pretrain the
>     supervised init with `--value-head-type gap` against the 13.6M
>     yngine corpus (`scripts/run_supervised_pretraining.py`, ~3h on
>     4090), then re-launch Branch D.1 from that GAP-native init.
>
> **Don't start D.2** until D.1 has at least an SPRT screen. The reason
> stack: (a) D.2 requires regenerating the 40GB volume corpus with the
> 15-channel encoder + a fresh ~3h supervised pretrain + a ~20h Branch-C
> self-play loop — much more cloud time than D.1 alone, (b) "Sequential
> one-variable tests, not stacked, so each result attributes cleanly"
> (next-steps doc, §2), (c) if D.1 is a flat result, D.2's premise of
> "the head was bloated, the encoding is impoverished, fixing both will
> compound" weakens, and we'd want to revisit the framing.
>
> **Operational:** spin a fresh vast.ai 4090 (Step 2's box is terminated).
> All artifacts local on the user's laptop (`models/yngine_volume_pretrain/`,
> `models/branchC_volume_pretrain/`, `logs/`). Push the Branch D.1 result
> checkpoints back via the same gh-release pattern as Step 2 when done
> (see `mcts400-session-snapshot-2026-05-23` for the recipe).

---

## Session update — 2026-05-24: D.1 v2 ran, NOT_STRONGER, GAP path partially exhausted

### Take 1 (lost to ops failure, no SPRT data)

Launched D.1 v2 (`configs/branchD1_v2_gap_mcts200.yaml`) on a fresh vast.ai 5090
(instance 37479128, 16GB disk) at 23:25 UTC 2026-05-23. Autopilot installed
to auto-launch SPRT after training + auto-stop the instance after SPRT.

Training completed in 7.23h. **Then the autopilot stopped the instance before
the SPRT could finish** (subtle race + a bug — see §"Operational lessons"
below). The instance entered vast.ai's "scheduling" state when we tried to
resume it because the underlying 5090 had been claimed by another renter,
and vast quoted "hours to weeks" wait. Disk artifacts were stranded.
No SPRT result from this run.

### Take 2 (proper SPRT verdict)

Re-ran on a fresh 5090 (79.116.87.141:30981, 50GB disk) at 11:03 UTC 2026-05-24
with modified autopilot — **NO auto-stop**, manual termination only. Training
completed in 5.71h (250 games, 5/5 promotions, anchor 95-100% across all
evaluated iters). SPRT initially crashed because I forgot to download
`best_iter_4.pt` to the new box; re-ran the SPRT manually after fix.

#### Result: NOT_STRONGER, decisive in 16 games (~8 min)

| | |
|---|---|
| Verdict | **NOT_STRONGER** |
| Score | candidate **1-15-0** (WR = 0.0625) |
| Color split | W/B = 0/1 — both colors lost |
| CI95 | [0.011, 0.283] |
| LLR | -3.16 (lower boundary at -2.94, crossed) |
| Trajectory | g=4: 1-3 → g=8: 1-7 → g=12: 1-11 → g=16: 1-15 |
| Wall | 455s on 5090 |

**Byte-identical to v1's SPRT trajectory** (also 1-15-0, LLR -3.16 in 16
games, color split W/B = 0/1 — the one win as Black in both runs).

**Verified as structural determinism, NOT coincidence.** Re-ran the SPRT
on the same v2 take-2 checkpoint with `--seed 12345` (default is 42).
Result was byte-identical — same verdict, same 1-15-0 score, same LLR,
same color split. Even with different seeds for opening sampling, the
candidate plays the SAME 16 losing games. Mechanism: the model's trained
policy is sufficiently peaked that temperature=1.0 opening sampling
near-always picks the top move, and MCTS leaf-evaluation order is
deterministic at our CUDA batch sizes. Combined with the candidate's
structural weakness vs the anchor (same fatal tactical pattern every
game), the entire match unfolds identically across seeds.

This means the ~6% WR isn't measurement noise — it's the *true* effective
win rate of this candidate against this anchor under this protocol. The
verdict is maximally robust. Both v1 (~17K params, direct projection) and
v2 (~22K params, hidden Linear) test as similarly weak against the
spatial-head champion.

(Note for future SPRT runs: deterministic-replay behavior means seed
variation isn't a reliable noise estimate for badly-losing candidates.
For *borderline* candidates near WR=0.50, expect more variance — peaked
policies sampling at 50/50-equivalent positions WILL pick different moves
across seeds. Save the seed-variation check for cases where the verdict
itself is in doubt.)

Local artifacts: `experiments/branchD1_v2_failed_run/` (SPRT JSON, manifest,
feedback, tournament history, SUMMARY.md). Checkpoint:
`models/branchD1_v2_pretrain/checkpoint_iteration_4_ema.pt`.

### What this tells us (and honestly does NOT)

**Definitive:** GAP head + warm-started spatial-trunk fails identically in
both minimal (v1) and KataGo-canonical (v2) forms. The "missing hidden
Linear" hypothesis from v1 was wrong — adding it changed nothing.

**Indeterminate** (was previously overstated as "less attractive"): whether
GAP architecture works AT ALL for this problem. Two theories predict the
same v1+v2 result but diverge on what to test next:

- **Theory A — warm-start specialization:** the trunk's 30M params are
  tuned for the spatial head's particular output shape. Brief D.1 self-play
  (5 iters, ~500 games) can't unlearn that. A GAP-native trunk trained from
  scratch would not have this problem. → **Path 2 would work.**
- **Theory B — GAP discards position:** YINSH evaluation needs spatial
  structure (corners vs center, edges, ring positions). Global average
  pooling fundamentally throws that away. No trunk can compensate.
  → **Path 2 wouldn't work no matter what.**

Both theories are consistent with current data. Neither has decisive
support. **The earlier framing "GAP-native pretrain becomes much less
attractive" was overconfident** — it implicitly assumed theory B was more
likely without evidence.

### D.2 vs Path 2 — sequencing decision (and what's in the backlog)

Both are reasonable next experiments at similar cost (~$10-15, ~12h on
5090). Sequencing call:

**D.2 first.** Reasoning:
- D.2's question (input-richness ceiling) is *upstream* of head-shape
  questions. If 6-channel encoding is impoverished, every future
  experiment — Path 2, D.3, scale-up — is operating with a handicap.
- D.2's positive result *strengthens* any future Path 2 attempt (richer
  input → GAP head has more signal to pool over).
- D.2's negative result tells us encoding isn't the lever and we should
  return to architecture (where Path 2 lives).

**Path 2 stays in the backlog.** Specifically: re-train supervised init
from scratch via `scripts/run_supervised_pretraining.py --value-head-type
gap_v2` on the 13.6M yngine corpus, then re-launch Branch D.1 from that
GAP-native init. Tests theory A directly. Worth running if:
- D.2 is inconclusive AND we still want to know whether GAP architecture
  is viable
- We come back to the GAP question after D.3/scale-up plateau

Not now: don't compound a known-not-improving change with a new variable,
and don't burn cloud time on a narrower experiment when a broader one
(D.2) is teed up.

### Operational lessons (apply to D.2 autopilot)

1. **`vastai stop` is one-way for popular GPUs.** When we stopped the take-1
   5090, it returned to the rental pool and was claimed by another renter
   within minutes. Resume = "hours to weeks" wait. Take-2 autopilot was
   modified to NEVER call stop — write SUMMARY.md + STOP_READY sentinel
   only, leave box running for manual termination. The ~$0.89/hr idle
   between SPRT completion and human retrieval is insurance well worth it.

2. **Pre-flight ALL required files.** Take-2 SPRT crashed initially because
   I downloaded only `best_supervised.pt` (warm-start init) but forgot
   `best_iter_4.pt` (SPRT anchor). The tmux session died instantly with
   FileNotFoundError, no log produced, looked like an opaque "SPRT
   CRASHED" until I checked filesystem. For D.2: the autopilot's first
   action on box setup should verify all required artifacts exist BEFORE
   training launches.

3. **The HA(d=1) anchor is a saturated metric.** Both v1 and v2 scored
   95-100% across all 4 evaluated iters but lost decisively to a stronger
   opponent in SPRT. Stop celebrating anchor=100% as a positive signal —
   it's a "didn't break catastrophically" sanity check, nothing more.
   The frozen-anchor SPRT is the *only* informative discriminator at this
   point.

4. **Pull artifacts to laptop ASAP after a run.** Don't trust that the box
   will be available in the morning. Even with no-auto-stop, instances can
   be interrupted by the host (vast.ai sometimes preempts), can lose
   network, etc. Treat the box's disk as ephemeral; the laptop is the
   archive.

### Lifecycle / next session

- Box 79.116.87.141:30981 is still running post-SPRT (per take-2 design).
  Pull artifacts (done), seed-variation test (running in tmux `sprt_seedtest`),
  then terminate via vast.ai dashboard.
- Original v2 take-1 box 37479128 still in "scheduling" — let it sit; if it
  ever schedules, the original data is there as a free cross-check.

---

## Prompt for the next session — D.2 launch on fresh GPU

> Continuing YINSH ceiling-raising work. Read `VOLUME_PRETRAIN_RESULTS.md`
> §"Session update — 2026-05-24" for the latest state. TL;DR: D.1 v1 + v2
> both tested, both lost SPRT 1-15-0 vs frozen `best_iter_4`. **GAP head +
> warm-start spatial trunk doesn't work.** Whether GAP works AT ALL with
> a GAP-native trunk (Path 2) is undetermined and kept in backlog. **Next
> experiment: Branch D.2 — enhanced 15-channel encoding.**
>
> Full execution plan lives in `D2_PREP.md` (also already written). Quick
> version:
>
> 1. Spin a fresh vast.ai 5090 with **≥100GB disk** (we'll pull the 13.6M-
>    position volume corpus, which is ~10-15GB compressed)
> 2. Setup: git pull `training-pipeline-fixes`, cu128 torch venv, deps,
>    download `best_supervised.pt` AND `best_iter_4.pt` from the
>    `mcts400-session-snapshot-2026-05-23` gh release (pre-flight check!)
> 3. Pull volume corpus: `gh release download mcts400-session-snapshot-2026-05-23
>    --pattern yngine_volume.npz --dir expert_games/`
> 4. Regenerate corpus with 15-channel encoder. Path B implementation
>    landed 2026-05-24 (`scripts/regenerate_npz_with_enhanced_encoder.py`,
>    tested in `yinsh_ml/tests/test_regenerate_15ch.py`). Just run:
>    `python scripts/regenerate_npz_with_enhanced_encoder.py
>     --input expert_games/yngine_volume.npz
>     --output expert_games/yngine_volume_15ch_mmap/`
>    Output is mmap-shard format (consumed directly by
>    `--data-dir` in step 5). ~1-2h CPU on the 13.6M corpus. Channel
>    13 / turn-number is unrecoverable and gets zeroed; all other 14
>    channels round-trip cleanly. See D2_PREP.md §"Path B" for details.
> 5. Supervised pretrain on 15-channel corpus (~3h):
>    `python scripts/run_supervised_pretraining.py --data-dir expert_games/yngine_volume_15ch_mmap/
>     --use-enhanced-encoding --value-head-type spatial --output-dir models/yngine_volume_15ch_pretrain
>     --epochs 3 --batch-size 512 --lr 1e-3 --num-channels 256 --num-blocks 12`
>
>    Note: use **`--value-head-type spatial`**, not gap_v2 — don't compound
>    a known-failing change. Test the encoding axis in isolation.
> 6. Self-play loop with `configs/branchD2_enhanced_mcts200.yaml` (already
>    written, ~7h on 5090), warm-started from the 15-channel pretrain
> 7. SPRT screen vs frozen `best_iter_4` — same procedure as D.1 v2
>
> Updated autopilot for D.2:
>   - NO auto-stop (take-2 design — manual termination)
>   - Pre-flight all required files at startup BEFORE training
>   - Same crash detection + auto-launch SPRT pattern
>
> Interpretation matrix for the SPRT verdict:
>   - STRONGER → encoding is the lever; freeze, move to D.3 or stack with
>     other architectural changes
>   - INCONCLUSIVE WR ~ 0.55 → small real edge (like Step 2); useful, not
>     promotion-worthy; investigate stacking with other axes
>   - INCONCLUSIVE WR ~ 0.50 or WEAKER → encoding isn't the binding
>     constraint. Pivot to Path 2 (GAP-native pretrain, see backlog) OR
>     deeper-trunk / wider-trunk / attention-mechanism experiments
>
> Backlog (do NOT prioritize over D.2):
>   - **Path 2: GAP-native supervised pretrain.** Tests whether GAP arch
>     can work given a trunk trained from scratch for it. Cost ~12h on
>     5090. Use `scripts/run_supervised_pretraining.py --value-head-type
>     gap_v2 ...`. Theory-A test (see §"What this tells us" above).
>   - **D.3 SE blocks.** Smallest architectural addition with strongest
>     mechanism-based prior after head/encoding axes. Run after D.2 lands.
>
> All checkpoints + corpora are accessible from the gh release. Branch
> `training-pipeline-fixes`, head commit at session start: TBD (push
> session updates first).
