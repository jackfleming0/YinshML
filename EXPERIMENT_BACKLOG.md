# Experiment Backlog

The forward-looking queue: ranked list of candidate experiments with rationale,
counterarguments, and cost estimates. This is the **source of truth for "what
should we run next."**

Complements `VOLUME_PRETRAIN_RESULTS.md` (the chronological session log) and
`D2_PREP.md` (a specific experiment's scoping doc). When a session ends without
a clear next step, this file is the first thing to read.

## How to read this doc

- **Stack-rank table** (next section) is a five-axis comparison — scan for
  totals, but the per-axis scores matter more for the situation you're actually
  in (e.g. if compute is tight, weight Cost; if you're protecting a hypothesis,
  weight Implementation risk).
- **Detailed write-ups** (later) explain each candidate. Read the section for
  any experiment you're considering before committing to it — especially the
  *"Reasons to not believe"* part, which is where the cheapest correction lives.

## How to maintain it

- **When an experiment runs to completion** (positive, negative, or
  inconclusive), move its writeup to a `## Done` section at the bottom with a
  one-paragraph result. Don't delete — the rationale that *led* to running it
  is durable knowledge for future calibration.
- **When a new hypothesis surfaces**, draft an entry here BEFORE running
  anything. Forces explicit thinking about cost vs. info gain.
- **Re-rank** when the situation changes — a new finding can promote or demote
  prior entries.

---

## Status snapshot (as of 2026-05-31 ~14:00 UTC) — recovery + Tasks 1 & 2 landed

The 2026-05-29/30 work was recovered from a stash (it was never committed; the
`policy-dropout-fix` branch was empty) onto **`policy-symmetry-fixes`**. Then:

- **L1 (Dropout 0.3→0)** — recovered into `model.py` (commit f8bdbcb).
- **L3a (symmetric MCTS at inference)** — shipped to the analysis board:
  `server.py::_symmetric_search_batch` (sync + async paths), default-on via
  `YNS_SYMMETRIC_MCTS`, UI toggle, effective-4×-budget async routing. Validated
  on deployed iter1_ema: opening D6 concentration 0.857→0.214 (commit 09a6d86).
  **Code-complete, not yet deployed** (needs `git push` + `yinsh-redeploy`).
- **L2 (label smoothing ε=0.1)** — wired into **`scripts/run_supervised_pretraining.py`**
  (`--label-smoothing`, default 0.1, applied to the hard-target CE). NB: the
  handoff said "trainer.py", but trainer.py's policy loss is a *soft*-target CE
  (MCTS visit distributions) where classic label smoothing doesn't apply. The
  validated dry-run (`dry_run_dropout_plus_ls.py`) used the *supervised* hard-
  target CE — that's where L2 belongs.
- **E16 (symmetric-weight regularizer)** — wired into `trainer.py`
  (`enable_symmetric_reg`, α=`symmetric_reg_weight` 0.1, K=`symmetric_reg_every_k_steps`
  10, `symmetric_reg_value_weight` 0.5), threaded through the supervisor from
  `mode_settings`, off by default. Full faithful policy-KL + value-asymmetry
  form. The full 7433-move-index permutation is precomputed once and **verified
  to exactly match the validated per-state augmenter permutation**; policy KL is
  masked to the `target_probs>0` valid-move support (the unmasked full-softmax KL
  was ~100× inflated by never-trained invalid-move logits). Tests:
  `yinsh_ml/tests/test_symmetric_regularizer.py`. **`symmetric_reg_value_weight`
  default set to 10.0** (was a guessed 0.5) from a measured gradient-pressure
  analysis (`scripts/investigate_e16_value_weight.py`): value_asym penalizes a
  scalar value in ~[-1,1] while policy-KL lives on a simplex, so at 0.5 the value
  term exerted only ~1/20th the gradient pressure of the policy term — backwards,
  since E11 named the value head the dominant asymmetry. ~10 equalizes ‖grad‖ on
  the shared trunk; raise toward 15-20 to prioritize value. Both magnitudes are
  logged every K steps — tune live (a CPU dynamic probe is impractical at ~50s/
  step on this net; confirm on the GPU run instead).

**Remaining: Task 3** — next cloud run stacking L1+L2+E16 (supervised pretrain
with `--label-smoothing 0.1` on a Dropout=0 net, then self-play with
`enable_symmetric_reg: true`). Symmetric MCTS (L3a) stays as a deploy-time
noise-reducer even once the network is symmetric.

---

## Status snapshot (as of 2026-05-29 ~13:30 UTC)

**Active investigation:** placement pathology + plateau diagnostics.
Full diagnosis in
[`analysis_board/multiplayer/EXPERIMENT_opening_theory.md`](analysis_board/multiplayer/EXPERIMENT_opening_theory.md).

- **Friend-tester feedback (2026-05-28):** engine plays anomalous
  4-of-5 ring cluster opening. Investigation traced root cause:
  yngine pretrain corpus has uniform-random placement targets (F6 at
  1.3% = exactly uniform); supervised model faithfully learned uniform
  policy; MCTS + FPU + uniform policy → "first-visited child wins all
  visits" stall; modal opening (A5 for iter1_ema, D6 for supervised)
  is path-dependence, not strategic preference.
- **E6 dry-run (continued pretrain on H-vs-H full-game data) — VERDICT:
  policy fix works, strength regresses.** F6 modal achieved (27% on
  policy head, 80% in self-play), but H2H vs iter1_ema = **8-22 (27%
  WR, Wilson 95% CI [0.14, 0.44]).** Continued pretrain on H-vs-H
  main-game forgot yngine-learned tactics. **Retired as a production
  path.**
- **E7 (split-corpus pretrain) is the leading next candidate.**
  Per-game placement sampled from human-replay + random + BGA-marginal;
  main-game phase plays out with iter1_ema for both sides. Cleanly
  separates the two fixes — human placement targets where we want
  them, iter1-quality tactics where we want them. Cost ~$200 / ~16h
  cloud for v1.
- **Three plateau-diagnostic experiments (P1/P2/P3) gate E7
  commitment.** Each ~1-4h, no cloud:
  - P1 — MCTS amplification quality (does deeper search find moves the
    raw policy didn't already know?)
  - P2 — Value-head calibration on held-out H-vs-H positions
  - P3 — Self-play policy-target signal (do MCTS visit dists differ
    meaningfully from raw policy in recent replay buffer?)
  Outcomes inform whether E7's mechanism actually breaks the post-iter1
  plateau or just rearranges it.
- **E7b (cross-teacher: iter1 + supervised + heuristic on different
  sides) on standby.** If P3 shows self-play signal is near zero,
  cross-teacher is the only version that can produce genuinely
  disagreeing training targets.
- **Earlier 2026-05-28 yngine benchmark still stands:** iter1_ema
  17-0-0 SPRT vs yngine at both MCTS-200 and MCTS-800. Confirms
  iter1 > yngine teacher quality, which is the load-bearing claim
  for E7.

### Architecture finding (2026-05-29) — Dropout(0.3) in policy head is the plateau cause

P1/P2/P3 diagnostics ran 2026-05-29 afternoon:

- **P1 — MCTS amplification**: ✓ works. 10 main-game positions, mean
  KL(visits_800 || visits_96) = 3.46 nats, top-1 disagrees 50% of the
  time between 96-sim and 800-sim search. Deep search finds different,
  sharper distributions than shallow search.
- **P2 — Value-head calibration**: ✓ calibrated. 5K H-vs-H positions,
  linear-fit slope 0.98 (1.0 = perfect), Brier score 0.66 vs baseline
  0.78 (15% improvement). Value predictions track actual outcomes
  monotonically across the [-1,+1] range.
- **P3 — Policy-target signal in self-play**: ✗ policy head can't
  reproduce MCTS targets. KL(MCTS_visits || policy_pred) median 9.0
  nats (max possible ~9 for 7,433-way classification), top-1 match
  rate 0%, mean policy peak 0.000329 (uniform = 0.000134, so model is
  ~2.5× uniform).

Root cause located in `yinsh_ml/network/model.py` policy head:

```python
self.policy_head = nn.Sequential(
    nn.Conv2d(num_channels, 64, 1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(64 * 11 * 11, 1024), nn.ReLU(),
    nn.Dropout(0.3),                       # ← culprit
    nn.Linear(1024, self.total_moves)
)
```

Dropout(0.3) on the 1024-feature bottleneck before a 1024→7,433 layer
forces ensemble-averaging over dropout subnetworks during training.
For 7,433-way classification with sharp MCTS targets, the minimum-loss
solution becomes near-uniform output. Even at eval time (dropout off),
the trained weights produce flat output because they were calibrated
to the averaged-subnetwork minimum-loss point during training.

Consistent with all measured policy outputs across all training
lineages (supervised, iter1, iter5) — they all converge to ~2-5×
uniform peak, regardless of training data quality.

### What this means for E6 / E7

- **E6 dry-run negative result is partially confounded.** The
  weak-policy issue at empty board WAS fixed by E6 (F6 27% policy
  output) but the underlying architectural limit means the policy
  head can still only produce ~25% peak — not the sharp 60-90%
  distributions humans actually play. Some of E6's main-game
  strength regression may also be the dropout limiting how well the
  model can sharpen on new training signal.
- **E7's claimed value is fully eclipsed by the dropout fix.** No
  amount of better corpus quality matters if the policy head
  architecturally cannot learn from it. E7 stays on the backlog as a
  *future* improvement after the architecture fix lands, not as the
  next priority.
- **The fix is one line of code.** `Dropout(0.3)` → `Dropout(0.0)`
  in `model.py:140`. AlphaZero-style policy heads typically don't
  use dropout; regularization can come from weight decay (already
  1e-4 on policy optimizer) or label smoothing if needed.

### Branch `policy-dropout-fix` (2026-05-29)

Created from main. Single change: `Dropout(0.3) → Dropout(0.0)` in
`yinsh_ml/network/model.py` policy head with diagnostic comment.

**Local validation in progress:** continued pretrain from
`supervised_2026-05-27/best_supervised.pt` on `hvh_full_game_15ch.npz`
(identical setup to E6 dry-run for direct A/B), 3 epochs,
LR 5e-5, weight decay 1e-3. Output:
`models/supervised_2026-05-27/dropout_patch.pt`.

Test criteria:
- Empty-board policy: peak > 0.05 (vs E6 dry-run's 0.27 was the
  ceiling under dropout; if patched is similar or higher, no penalty;
  if dramatically higher, dropout was the cap).
- Policy entropy at empty board: should DROP (sharper distribution)
  vs E6's value.
- H2H vs iter1_ema: same Wilson-LB ≥ 0.50 non-regression bar as E6.

If the patch validates, the next training run is the **architecturally
patched recipe**: this dropout fix + E6's H-vs-H placement data + the
existing self-play loop. E7 stays queued for later as a quality lift
on top of the corrected architecture.

### Validation results (2026-05-29, branch `policy-dropout-fix`) — patch works at the policy-head level, over-confidence is a new failure mode

After patching `Dropout(0.3) → Dropout(0.0)`, ran the same continued-pretrain
recipe as E6 dry-run (107K H-vs-H positions, 3 epochs, LR 5e-5, weight decay
1e-3, 4× placement oversample). Output:
`models/supervised_2026-05-27/dropout_patch.pt`.

**Policy head learns dramatically faster:**

| Metric | Before training | E6 (5 epochs) | Dropout-patched (3 epochs) |
|---|---|---|---|
| Empty-board peak | 1.39% | 27.2% | **31.3%** |
| Empty-board entropy | 4.44 | ~3.5 | **3.39** |
| Multiplier vs uniform | 1.04× | 20.3× | **22.4×** |

The policy head sharpened past E6's epoch-5 ceiling in just epoch 2 (F6 28.9%
at epoch 2 of the patched run vs F6 27.2% at epoch 5 of E6). Decisively
confirms the diagnosis: Dropout(0.3) was the architectural cap on policy
sharpness.

**But self-play exposed a new failure mode — over-confidence collapse:**

50-game deployed_sampled self-play on the patched model:
- Slot-1 modal: **F6 100%** (every single game)
- Aggregate cluster: F6/G6/G7/F7/G8 (mean_pw 1.26)
- Tight-cluster rate: **100%** (all 50 games)
- **White WR: 6% (3 of 50)** — catastrophic regression

The over-confidence cause: cross-entropy on one-hot human-move targets with
no label smoothing lets the policy head drive loss arbitrarily close to 0 by
over-concentrating. AlphaZero-style training uses MCTS visit distributions
(naturally entropic) as policy targets, which we don't have in the H-vs-H
corpus. Label smoothing is the cheap approximation.

**Label-smoothing variant (in progress, 2026-05-29 ~17:30 UTC):**
Same setup but `F.cross_entropy(..., label_smoothing=0.1)`. Output target:
`models/supervised_2026-05-27/dropout_plus_ls01.pt`. Epoch 1 result:
- F6 = 12.9% (vs dropout-only 21.7% at same epoch)
- Entropy = 4.69 (vs dropout-only 3.70)
Smoothing is working as designed — preventing premature over-concentration.
Two more epochs to land final result.

### Reframing (Jack, 2026-05-29 ~17:50 UTC) — don't treat humans as ground truth

**Key insight:** YINSH lacks anything like chess opening theory. BGA "top-10"
is a few hundred enthusiasts, not centuries of accumulated wisdom. F6 might
be the right opening, or it might just be cargo-culted as "the obvious first
move." Treating the H-vs-H placement distribution as the *target* the model
must converge to is an unsupported assumption.

**Decisive evidence the model's current convergence is path-dependence, not
strategy** — board symmetry argument:

YINSH has D2 board symmetry (Klein 4-group: identity + 180° rotation + 2
reflections). iter1_ema's slot-1 distribution:
- A5: 72.0%
- A2: 2.5%

A2 is the horizontal-reflection partner of A5. In a D2-symmetric game, if
A5 is genuinely strategically optimal, the model should play A2 (and the
other two symmetry partners K7, K10) at ~equal rates — roughly 18% each.
The observed 72%/2.5% asymmetry is *physically impossible* if the model
had learned any real strategic principle. It can only happen via the FPU
+ uniform-policy MCTS-stall mechanism we diagnosed in P3. The symmetry
breakage is the smoking gun for path-dependence.

**Implication for the production recipe:**

The right framing is "set the model up to *explore well*, then let it
converge to whatever wins under symmetry-respecting search" — NOT "force
it to match human play."

| Old framing | New framing |
|---|---|
| Fix placement so it matches human play | Fix architecture so the model can express what it learns, then let exploration discover |
| F6 modal is the target | Symmetric exploration is the target; whatever wins under self-play is the answer |
| Use H-vs-H data to teach placement | Use H-vs-H to initialize away from uniform; use random + symmetric variants for exploration; let self-play decide |

### Production recipe (2026-05-29 final) — what the next big training run should include

Three layers, each individually validated this session:

1. **Architecture**: `Dropout(0)` in policy head (this branch).
   - Validated: policy head sharpens 22.4× from uniform in 3 epochs.
   - Without this: policy head architecturally cannot represent sharp
     distributions, regardless of training data quality.

2. **Loss**: Cross-entropy with **label smoothing ε ≈ 0.05-0.1**, OR train
   on MCTS visit distributions where available.
   - Validated negatively: without smoothing, policy collapses to F6 100%
     and white WR drops to 6%.
   - Smoothing puts a floor on the loss, prevents over-confidence collapse.

3. **Data + exploration**: split corpus (H-vs-H placement + iter1 main game)
   AND aggressive exploration knobs (phase-aware Dirichlet + temperature +
   FPU at placement) AND symmetric-MCTS at inference time.
   - The H-vs-H placement is an *initialization* signal, not a target.
   - iter1 main-game is the strongest tactical teacher we have (per yngine
     benchmark, see [`YNGINE_BENCHMARK_RESULTS.md`](YNGINE_BENCHMARK_RESULTS.md)).
   - Symmetric MCTS forces the model's output to respect game symmetry
     regardless of any asymmetric weights — guarantees we don't see another
     A5-style asymmetric collapse.
   - Random + symmetric placement starts in corpus generation give the
     model broad state coverage.

### New experiment entries spun out from this session

**E8 — Symmetric MCTS at inference time** (new, 2026-05-29)

At each MCTS root expansion, evaluate the network on all 4 symmetric
variants of the position (D2 Klein 4-group), then average policy + value
across the variants before MCTS uses them. Guarantees output respects
board symmetry regardless of policy-head weight asymmetries.

- **Why this matters**: iter1_ema's A5 72% / A2 2.5% asymmetry is the
  smoking gun for MCTS path-dependence. Even after fixing the policy head
  (dropout patch), any residual asymmetry in the trained weights could
  still produce symmetry-breaking convergence. Symmetric MCTS makes the
  inference-time output mathematically symmetric.
- **Cost**: 4× network forward at each MCTS leaf. Could be optimized by
  batching the 4 variants. ~50 LOC + ~1 hour wall.
- **Standard in AlphaZero implementations** (Leela, KataGo) — not a novel
  invention, just not wired in this codebase.

**E9 — Phase-aware exploration knobs** (already proposed as E1 in opening_theory
doc — relabeled here for clarity)

Per-phase Dirichlet + temperature + FPU settings:
- `placement_dirichlet_alpha: 1.0` (vs 0.3 globally)
- `placement_epsilon_mix: 0.5` (vs tapering 0.25 → 0.14)
- `placement_temperature: 1.0` (vs tapering 1.0 → 0.55)
- `placement_fpu_reduction: 0.0` (vs 0.25 — defensive against re-emergence
  of FPU stall if policy ever drifts uniform)
- Implementation: gate the existing knobs in
  `yinsh_ml/training/self_play.py::MCTS` on `state.phase ==
  GamePhase.RING_PLACEMENT`. ~30 LOC.

**E10 — Random + symmetric placement injection in corpus generation**

For corpus generation (whether E7-style iter1-vs-iter1 or future
engine-corpus iterations):
- 40% placement from H-vs-H human replay (initialization signal)
- 20% from BGA marginal sampling (variety on the human distribution)
- 20% uniform random (state-coverage diversity)
- 20% **symmetric augmentation of human placements** (explicit symmetry
  signal during training)
- Each placement contributes the full D2 augmentation (4× data per
  position) to ensure symmetric coverage.

### Provenance / thread map (so the timeline is preserved)

For future reference — what was discovered when, and how everything
connects:

1. **2026-05-28 early afternoon**: friend-tester feedback flagged the
   weird opening in the deployed model. Triggered the entire
   investigation in
   [`analysis_board/multiplayer/EXPERIMENT_opening_theory.md`](analysis_board/multiplayer/EXPERIMENT_opening_theory.md).

2. **2026-05-28 evening**: traced root cause to yngine corpus having
   uniform random placement targets → supervised model learned uniform
   policy → MCTS FPU + uniform policy creates path-dependent stall →
   modal opening (A5 for iter1, D6 for supervised) is a tiny-difference
   amplification, not a strategic choice.

3. **2026-05-28 late evening**: E6 dry-run kicked off — continued pretrain
   on H-vs-H to fix the data side of the problem.

4. **2026-05-29 morning**: E6 dry-run completed. F6 modal recovered at
   policy head (27%), but white WR regressed to 38% in self-play AND
   E6 lost H2H vs iter1_ema 22-8 (Wilson LB [0.14, 0.44]). Data fix
   alone wasn't enough.

5. **2026-05-29 early afternoon**: ran P1/P2/P3 plateau diagnostics.
   - P1 (MCTS amplification): ✓ works (mean KL 3.46 nats between 96
     and 800 sims, top-1 changes 50% of the time)
   - P2 (value-head calibration): ✓ calibrated (slope 0.98, Brier 15%
     better than baseline)
   - **P3 (policy-target signal)**: ✗ broken — KL(MCTS visits, policy
     pred) median 9 nats, near-uniform policy outputs.

6. **2026-05-29 mid-afternoon**: traced P3 result to architecture —
   `Dropout(0.3)` on the policy head's 1024-feature bottleneck. Ensemble-
   averaging during training caps the policy head's achievable sharpness
   at training time, weights stay calibrated to that minimum-loss point
   even after dropout is disabled at eval.

7. **2026-05-29 late afternoon**: created branch `policy-dropout-fix`,
   patched to `Dropout(0.0)`, ran continued pretrain. Validated: policy
   sharpens past E6's ceiling in 3 epochs.

8. **2026-05-29 evening**: tested patched model in self-play.
   Over-confidence collapse — F6 100% modal, white WR 6%. Diagnosed as
   one-hot CE without label smoothing letting the now-flexible policy
   head drive loss to 0 by collapsing.

9. **2026-05-29 evening (current)**: label smoothing variant in progress,
   expected to demonstrate the production recipe.

10. **Jack's reframing during step 9**: symmetry argument (A5 72% / A2
    2.5% is unphysical for D2-symmetric game) — proves path-dependence,
    drives the "don't force humans, set up for exploration" recipe.

11. **Overnight 2026-05-29 → 2026-05-30 (~8h wall)**: built symmetric
    MCTS at inference (averaged policy + value across 4 D2 transforms
    per move). 50 games each of iter1_ema and dropout+LS.
    Implementation: `scripts/measure_symmetric_openings.py`.

12. **Symmetric MCTS results (2026-05-30 morning)** — decisive
    validation of the symmetry hypothesis, plus unexpected main-game
    strength bonus:

    iter1_ema A5 orbit ({A5, K7, E1, G11}):

    | Position | Vanilla iter1 (n=200) | Symmetric iter1 (n=50) |
    |---|---|---|
    | A5 | 72.0% | 40.0% |
    | K7 | 0% | 8.0% |
    | E1 | 0% | 6.0% |
    | G11 | 1.0% | 0% |
    | A5 share of orbit | **99%** | **74%** |

    ~75% of path-dependence broken; A5 dropped from 72% → 40%.
    White WR 48% → 54%.

    Dropout+LS model: white WR **24% → 46%** under symmetric MCTS — a
    massive +22 pp bonus from noise reduction in MCTS leaf evaluations
    even though the policy is over-concentrated. Wasn't predicted.

### E8 (VALIDATED) — symmetric MCTS at inference

**Status:** validated overnight 2026-05-29 → 2026-05-30. Two clean
benefits:
1. Breaks ~75% of MCTS path-dependence (orbit partners activate)
2. Improves main-game WR via noise reduction in leaf evaluations
   (+6 pp for iter1, +22 pp for dropout+LS)

**SHIP IMMEDIATELY**: 4× evaluation cost per move is well within the
deployed_sampled budget. Wire into `analysis_board/multiplayer/deploy/`
inference path. Even without retraining, gives users a less weird
opening AND a stronger main game. The
`symmetric_search()` function in `scripts/measure_symmetric_openings.py`
is the reference implementation; ~50 LOC adapter for the deploy path.

### Verifying the residual 25% — diagnostic experiments

The orbit-internal asymmetry remaining after symmetric MCTS (A5 still
~3× over orbit average — 20 vs 4/3/0 for K7/E1/G11) has four candidate
causes. Each is testable with a small experiment, ordered by
cost-to-information ratio.

**Hypotheses for the residual 25%:**
- **H_W: network weights are asymmetric** — D2 augmentation during
  training didn't fully enforce weight symmetry. *(Highest prior — the
  augmenter is opt-in per-sample and can skip transforms; even if
  applied perfectly, gradient noise across batches might not converge
  to D2-symmetric weights.)*
- **H_M: MCTS noise** — at 96 sims with 85 valid moves, the 4
  transformed searches each have stochastic visit counts that don't
  fully cancel under averaging.
- **H_N: sampling noise** — 50 games × temp 0.5 isn't enough to
  resolve the true orbit distribution; with larger n it may converge
  to uniform.
- **H_E: encoding pipeline lossiness** — the augmenter's
  encode→transform→decode roundtrip might introduce small artifacts.

**E11 — Direct weight symmetry check** (5 min, ~30 LOC) — the prime
diagnostic.

For a single position s (e.g., empty board, then a few mid-game
positions) and each transform tid:
1. Compute s_tid = augmenter.transform_state(s, tid)
2. Run network forward on s_tid → raw policy_tid, value_tid
3. Inverse-transform policy_tid back to original action space
4. Compare the 4 inverse-transformed policies against the original
   policy bytewise (and against each other)

If all 4 versions are equal (modulo numerical precision): network
weights produce D2-symmetric outputs, residual asymmetry is purely
MCTS-side. Eliminates H_W; focus on H_M/H_N.

If they differ measurably: H_W is real. Magnitude tells us how much.

**This is the single most informative test for the residual 25%.**
Run E11 first.

**E12 — MCTS sim-budget sweep** (~6h MPS) — diagnostic for H_M vs
H_N.

If E11 shows symmetric weights, the residual asymmetry is MCTS noise.
Re-run symmetric MCTS at 200 and 400 sims (50 games each, ~3h + 6h).
If orbit-internal asymmetry shrinks with sim budget → MCTS noise
(H_M) is the cause. If it stays the same → sampling noise (H_N)
dominates and we need more games.

**E13 — Increase game count** (~12h MPS) — only if E11 says symmetric
weights AND E12 says MCTS isn't the cause.

Run 200 games of symmetric MCTS. With more samples the orbit
distribution should converge toward symmetric if path-dependence is
fully broken.

**E14 — Augmenter pipeline integrity check** (15 min, ~20 LOC) —
rule out H_E.

For 100 random states from a replay buffer:
- For each tid in {1, 2, 3}: state → transform tid → transform tid
  (involution) → compare to original bytewise
- For each tid: policy at state → forward transform → back transform
  → compare bytewise
- All round-trips must be exact

If any state fails, the augmenter has a bug that explains some
fraction of the residual asymmetry mechanically.

**E15 — Training-time augmentation coverage audit** (30 min) —
diagnostic for *why* H_W happens, if E11 confirms it.

Pull saved `AugmentationStats` from a recent training run. Check:
- `total_augmentations / num_train_samples` — should be ~4 with
  reflections enabled
- `invalid_transforms_skipped` — should be 0 or near-zero
- Per-position augmentation count distribution — does any subset of
  positions get systematically fewer augmentations?

If significant skips, that's a candidate cause for non-symmetric
weights.

**E16 — Symmetric weight regularizer in training loss** — the fix
if H_W is confirmed.

Add a loss term that explicitly penalizes asymmetric output. For each
training batch, compute output on D2-transformed versions, inverse-
transform back, penalize KL divergence between the original and the
average of the 4 inverse-transformed versions.

~30 LOC change in `yinsh_ml/training/trainer.py`. Cost during
training: 4× forward per step + a small KL term. Acts as continuous
pressure to keep weights in the D2-symmetric subspace.

**E17 — Explicit D2 weight tying via equivariant network** (more
invasive, save for last).

If E16 doesn't fully eliminate asymmetry, modify the network
architecture so weights are explicitly shared between symmetric
channels (e.g., D2-equivariant Conv2d kernels — Cohen & Welling
style). Guarantees weight symmetry by construction.

Requires retraining from scratch + architecture verification work.
Defer unless E16 proves insufficient.

### Recommended sequence for the residual-25% investigation

1. **E11 (5 min)**: weight symmetry check. Quickest possible
   disambiguation between H_W and H_M/H_N. *Do this today.*
2. **E14 (15 min)**: pipeline integrity. Rules out a stupid bug
   before we spend more compute.
3. Conditional on E11 result:
   - **E11 says symmetric weights** → run E12 (6h MPS) for sim sweep,
     then E13 (12h MPS) if needed. Likely E13 + larger n resolves it.
   - **E11 says asymmetric weights** → run E15 (30 min) to find why
     training didn't symmetrize. Then E16 (30 LOC + retrain) to fix.
4. **E17 only if E16 falls short.**

### E11 + E14 results (2026-05-30 morning) — H_W confirmed, H_E ruled out

**E14 — Augmenter pipeline integrity: PASSED.** State round-trip
(transform→transform = identity since D2 transforms are involutions)
bytewise-exact. Policy round-trip also bytewise-exact. **H_E ruled
out** — no pipeline bug.

**E11 — Weight symmetry check across 6 states (empty + 5 mid-game):**

For each state, ran the network on all 4 D2-transformed versions,
inverse-transformed the policy back to original action space, and
compared against the identity prediction.

| State | Value-head: 4 transforms | Value symmetric? | Policy top-1 stable across transforms? |
|---|---|---|---|
| Empty board | -0.0086 ×4 | **YES (exact)** | No (top-1 differs across 3 transforms) |
| After move 2 | -0.016, -0.013, -0.016, -0.013 | 2-way pairing | No |
| After move 4 | -0.027, -0.015, -0.027, -0.014 | Near-paired | 2 of 3 differ |
| After move 6 | -0.045, -0.026, -0.037, -0.031 | All differ (1.7× range) | 2 of 3 differ |
| After move 8 | -0.034, -0.012, -0.025, -0.027 | All differ (2.8× range) | 3 of 3 differ |

**Two strong signals confirming H_W:**

1. **Value head is exactly symmetric on the empty board** (all 4
   transforms produce -0.008561 to 6-digit precision) but **becomes
   increasingly asymmetric as the position fills up**. By move 8,
   value varies 2.8× across orientations for *the same position*.
2. **Policy top-1 changes between transforms** in 5 of 6 states. KL
   divergences are small in absolute terms (~0.003-0.006 nats) but
   enough to flip the argmax on a near-uniform distribution.

**Mechanism:** the empty board is symmetric because the input has no
spatial information — all rings/markers channels are zero; only
uniform-broadcast channels (phase, sentinel) are nonzero, so
asymmetric weights have nothing to act on differently. Once there's
real game state, asymmetric weights produce different outputs for
different D2 orientations. The asymmetry magnitude grows with the
amount of spatial information.

**Why training didn't symmetrize the weights:** D2 augmentation
ensures the *training data* is symmetric but does not *constrain* the
learned weights to be symmetric. Gradient noise across batches +
non-symmetric initialization + per-orientation BatchNorm running
stats can produce weights that nearly-but-not-quite respect D2
symmetry.

**Verdict on the residual 25%:** H_W (asymmetric weights) is the
primary cause. E12/E13 (sim sweeps, larger n) are deferred — they
were diagnostics for if H_W was false. E15 (training augmentation
coverage audit) could be useful supporting evidence but isn't
blocking. The fix is E16.

### E16 prototype — symmetric weight regularizer for training loss

Prototype in `scripts/prototype_e16_symmetric_loss.py`. NOT wired into
trainer.py yet — runnable standalone for verification.

**Loss formulation:**
```
For each state s in batch:
    Forward on s → policy_0, value_0
    For tid in [1, 2, 3]:
        s_tid = T_tid(s)
        Forward → policy_tid_raw, value_tid_raw
        policy_tid = T_tid(policy_tid_raw)   # inverse via involution
    policy_sym = mean(policy_0, policy_1, policy_2, policy_3)
    value_sym = mean(value_0, value_1, value_2, value_3)
    loss_sym = KL(policy_0 || policy_sym) + α * mean((v_i - value_sym)^2)
```

**Prototype results on iter1_ema (validates math + measures penalty
magnitude):**

| State | KL policy asym | Value asym MSE | Value range | Total loss |
|---|---|---|---|---|
| Empty | 0.0020 | 0.000000 | (constant) | 0.0020 |
| After move 2 | 0.0062 | 0.000000 | 0.003 | 0.0062 |
| After move 4 | 0.0116 | 0.00004 | 0.013 | 0.0116 |
| After move 6 | 0.0173 | 0.00005 | 0.019 | 0.0173 |
| After move 8 | 0.0231 | 0.00006 | 0.022 | 0.0231 |

Penalty grows ~12× from empty to move 8, mirroring the E11
finding that asymmetry scales with game complexity. The regularizer
is nonzero and well-behaved — meaningful gradient signal to push
weights toward symmetric subspace.

**Trainer integration sketch (~30 LOC change, NOT yet applied to
trainer.py):**

```python
# In yinsh_ml/training/trainer.py train_step / train_epoch:
# Every K=10 batches, compute the symmetric regularizer.
# For each sample in batch (small loop — only every K steps):
#     sym_loss = symmetric_loss_term(self.network.network, state, ...)
# Add to total loss with small weight (alpha=0.1):
#     total_loss = policy_loss + value_loss + alpha * sym_loss_batch.mean()
#     total_loss.backward()
```

**Compute cost:** 4× forward per regularized step. At K=10, that's
+0.3× total training time. Manageable on cloud.

**Expected effect:** after a few thousand training steps with the
regularizer active, the network's per-position output asymmetry
(measured by re-running E11) should drop below the noise floor.
Symmetric MCTS at inference should then produce orbit-symmetric
visit distributions — fixing the residual 25%.

### Updated production recipe (2026-05-30, post-E11/E14)

| Layer | Status | Function |
|---|---|---|
| L1: Dropout(0) in policy head | ✓ Validated 2026-05-29 | Unblocks policy-head sharpening |
| L2: Label smoothing ε=0.1 | ✓ Validated 2026-05-29 | Prevents overconfidence collapse |
| **L3a: Symmetric MCTS at inference** | **✓ Validated 2026-05-30, SHIP NOW** | Breaks 75% of path-dependence; +6 to +22 pp WR bonus |
| **L3d: Symmetric weight regularizer (E16)** | **Prototyped 2026-05-30, ready to wire** | Fix residual 25% by pulling weights into D2-symmetric subspace |
| L3b: iter1-corpus pretrain (E7) | Not yet tested | Better teacher quality for main-game |
| L3c: Phase-aware exploration (E9) | Not yet tested | More diversity at placement |

### Action items emerging from 2026-05-30

1. **Ship symmetric MCTS at inference** (E8 reference implementation
   in `scripts/measure_symmetric_openings.py::symmetric_search`).
   ~50 LOC adapter for `analysis_board/multiplayer/deploy/` —
   pure win, no retraining.
2. **Wire E16 prototype into trainer.py** for the next cloud training
   run. ~30 LOC, prototype validated. Apply with alpha=0.1 and K=10.
3. **De-prioritized**: E12/E13 (sim/sample sweeps) — H_W confirmed
   means these would measure noise, not signal. Skip.
4. **Optional supporting evidence (30 min)**: E15 audit of training
   augmentation stats from a recent run, to confirm "D2 augmentation
   was applied but didn't enforce weight symmetry" rather than "D2
   augmentation was skipped a lot." Either result still points to
   E16 as the fix; useful diagnostic, not blocking.

### Updated production recipe (2026-05-30, with symmetric MCTS)

| Layer | Status | Function |
|---|---|---|
| L1: Dropout(0) in policy head | ✓ Validated | Unblocks policy-head sharpening |
| L2: Label smoothing ε=0.1 | ✓ Validated | Prevents overconfidence collapse |
| **L3a: Symmetric MCTS at inference** | **✓ Validated, SHIP NOW** | Breaks 75% of path-dependence; +6 to +22 pp WR bonus |
| L3b: iter1-corpus pretrain (E7) | Not yet tested | Better teacher quality for main-game |
| L3c: Phase-aware exploration (E9) | Not yet tested | More diversity at placement |
| L3d: Symmetric weight regularizer (E16) | Pending E11 result | Fix residual 25% if weights asymmetric |

**Ship-immediately item:** wire E8 (symmetric MCTS) into deployed
inference path. Pure win, no retraining. Users see better play +
varied openings starting today.

**Next training run recipe:** L1 + L2 architecturally + symmetric
MCTS at search time + (depending on E11) E16 symmetric weight
regularizer. L3b/L3c stack on top once foundation is solid.

This thread should connect cleanly with the older B1B2B3 / D.2 / value-head
investigation if the next training run pursues the production recipe above
and finally breaks past the post-iter_1 plateau.

## Status snapshot (as of 2026-05-28 ~23:30 UTC)

- **First measured win rate vs yngine landed — verdict STRONGER at both
  MCTS-200 and MCTS-800 (deployed `iter1_ema` 17-0-0 each, SPRT minimum
  termination)**. Closes the V2b bridge gap from the 2026-05-21 session.
  Full detail: [`YNGINE_BENCHMARK_RESULTS.md`](YNGINE_BENCHMARK_RESULTS.md);
  PR [#20](https://github.com/jackfleming0/YinshML/pull/20); Done entry
  below. The natural follow-up — yngine-MCTS-10K to find where the model
  breaks — is the new top-priority external-strength experiment.

## Earlier status snapshot (as of 2026-05-27 ~18:00 UTC)

- **B1+B2+B3 RE-RUN #2 complete — verdict NOT_STRONGER** (WR 0.476,
  CI95 [0.382, 0.571], decided at game 103). Phase-weight fix verified
  empirically (buffer phase mix 76.6% MAIN_GAME) and the in-loop +5 WR
  jump at iter 2 reproduces across two independent re-runs (51.7%,
  52.0%) — but the resulting iter_1 model is statistically
  indistinguishable from the warm-start at the SPRT 0.60 bar.
  **B1+B2+B3 is a closed experimental branch.**
- **Earlier B1+B2+B3 (2026-05-26) is INVALIDATED** by the phase-weight
  bug, alongside D.2 and D.1 v2 (all 15-channel runs trained under the
  bug). Phase fix is in HEAD; future 15-ch runs are clean.
- **Next experiment:** A4 + D1-partial combined. Pretrain with
  regression value head AND the saved replay buffer's MCTS targets.
  Lowest cost, highest mechanism prior remaining. Code prepped
  (commits `343aab6` + `2070650`); launch-ready.
- **Cloud box can be released.** Artifacts pulled locally:
  - iter_1_ema (the only promoted candidate): `models/yngine_volume_15ch_pretrain_b1b2b3_rerun2_iter1/iter1_ema.pt`
  - iter_4_ema (final reverted candidate): `models/yngine_volume_15ch_pretrain_b1b2b3_rerun2_iter4/iter4_ema.pt`
  - SPRT JSON: `logs/branchB1B2B3_rerun2_iter1_ema_vs_anchor.json`
  - Replay buffer + manifest + tournament_history in
    `experiments/branchB1B2B3_rerun2_2026-05-27/full_run_dir/20260527_001626/`

### Gate-override path is permissive — fix queued for post-run

Observed in iter 2 of the B1+B2+B3 RE-RUN (2026-05-26 19:26 UTC):

```
Wilson Gate Check: wins=207/400 (win_rate=0.517, SE=0.025,
                   CI95=[0.469, 0.566], threshold=0.5) -> REJECT
✅ NEW BEST: Copied checkpoint_iteration_1.pt to best_model.pt
Decision: ✅ NEW BEST (promoted to best model)
```

Wilson said REJECT (lower bound 0.469 < 0.50 threshold) — but
`supervisor.py:1731-1736` ran a third branch that promoted on
`candidate_elo > best_model_elo` regardless. Comment in code calls
this intentional ("Wilson failed but Elo improved"), but it means
**the Wilson gate is effectively advisory** — any candidate with
higher Glicko Elo gets promoted even when Wilson explicitly rejects.

The `wilson_attempted_no_data` fail-closed defense added 2026-05-26
only catches the "Wilson couldn't run" case, not the "Wilson ran and
said no" case. That's a second branch that wasn't on our radar at
the time.

**Why we're not fixing mid-run:** the user (2026-05-26 ~19:45 UTC)
chose to let the run continue rather than kill and patch — there's
real signal in the iter 2 promotion (WR climbed from 46.5% → 51.7%
under the corrected training) and the gate's permissive behavior is
known, not catastrophic. Future iter regressions self-correct given
the gate works correctly elsewhere (Wilson-rejects + Elo-also-down
= REVERT, which we observed in the invalidated run's iter 1-3).

**Patch shape for next session** (~30 min, before any other run):
Tighten the elif branch at supervisor.py:1731 so Elo override is
only allowed when `perform_wilson_check == False` (Wilson wasn't run
because of equality with best, no prior best, etc.). When Wilson WAS
run and said REJECT, the loop should fall through to "kept current
best" instead of overriding via Elo.

```python
# Proposed:
elif candidate_elo > self.best_model_elo and not perform_wilson_check:
    # Elo override is only valid when Wilson didn't speak.
    promote = True
    self.logger.info(f"... Elo improved ({candidate_elo:.1f} > {self.best_model_elo:.1f})")
```

Add a regression test in `test_supervisor_gate_fail_closed.py` that
asserts: when Wilson ran, returned False, and Elo is up → outcome is
"kept" not "promoted."

This is the right time to land this: we now have *two* concrete
failure modes documented (iter_4 in invalidated run, iter 2 in
re-run) that both promoted via the Elo override when the gate
intended to reject. The patch is justified.

### Post-B1B2B3-rerun-#2 investigation queue — "what would change my mind about the drift story"

Captured 2026-05-27 mid-run, during a discussion about whether iters 3-4's
declining WR (52.0 → 50.5 → 45.0 vs iter_1) is "degenerating spiral"
vs "luck of the draw." With 3 data points and SE ~2.5% per measurement,
the spiral framing was overconfident — the null hypothesis (random walk
around iter_1's true level) explains the data fine. **But several
legitimate alternative mechanisms COULD produce real drift, and each is
testable.** Logging the full investigation list here so it survives the
end of the active run.

**Working priors** (~13:00 UTC 2026-05-27, my honest read):

- ~50% pure noise / random walk around iter_1's true strength
- ~30% heuristic-weight annealing exposing iter_1's noisy value head
- ~15% optimizer state surviving revert and accumulating bad-direction momentum
- ~5% buffer mode collapse / something I haven't thought of

**Updated priors after diagnostics** (2026-05-27 evening, post-SPRT —
see `experiments/branchB1B2B3_rerun2_2026-05-27/POST_RUN_DIAGNOSTICS.md`
for the full report):

- ~60-70% pure noise (now the leading hypothesis)
- ~30-40% heuristic-weight annealing (still untested; requires GPU re-run)
- **0% optimizer state surviving revert** — ❌ ELIMINATED by code
  inspection. `supervisor.py`'s revert path explicitly calls
  `_reinitialize_optimizers()` (default-on via `reset_optimizer_on_revert`),
  which creates fresh `optim.Adam(...)` and `optim.SGD(...)` instances.
  No carried state across reverts.
- **0% buffer composition** — ❌ ELIMINATED by 3-way buffer comparison
  (D.2 buggy vs B1B2B3 original buggy vs B1B2B3 RE-RUN #2 fixed). All
  three buffers have ~76% MAIN_GAME (true distribution), stable
  ~64-move game length, similar policy sparsity. One small shift in
  the fixed run: 14.8% extreme-value samples vs ~11% in buggy runs —
  consistent with "MAIN_GAME-emphasized training produces more
  decisive games," not a drift cause.

The "3 data points = noise" discipline rule is now the leading
explanation. If the next experiment (A4 + D1-partial) also lands
NOT_STRONGER and we still care about the drift question, Mechanism 1
(heuristic_weight_end=0.3 re-run) is the only remaining structural
alternative to test.

**Discipline (carry forward):** when observing a "trend" across 2-3
data points each with ±5% CI, default to noise. Require 5+ data points
or a mechanism-level argument before claiming structural drift.

#### Mechanism 1 — Heuristic-weight annealing exposes iter_1's noisy value head

The config anneals `heuristic_weight: 0.5 → 0.0` over 5 iters:

| Iter | heuristic_weight | MCTS eval mix |
|---|---|---|
| 1 | 0.5 | 50% net + 50% heuristic |
| 2 | 0.4 | 60% net + 40% heuristic |
| 3 | 0.3 | 70% net + 30% heuristic |
| 4 | 0.2 | 80% net + 20% heuristic |
| 5 | 0.1 | 90% net + 10% heuristic |

If iter_1 has noisy value estimates (plausible — iter 2's +5 WR could
be a policy-head gain that doesn't transfer to the value head), MCTS
targets get progressively less reliable as heuristic regularization
falls away. By iter 4 the search is mostly trusting a possibly-bad
value head.

**Testable:** re-run with `heuristic_weight_end: 0.3` (vs 0.0). If
drift disappears or weakens, this mechanism is dominant. Cheap variant
of the next B1B2B3 run.

**Cost:** Same as B1B2B3 (~12-15h, 1 config knob change).

#### Mechanism 2 — Optimizer state survives revert and accumulates bad momentum

When the gate reverts iter N, the **weights** go back to the prior best.
But does the Adam optimizer's `m` (first moment) and `v` (second moment)
state get reset? If not, the optimizer carries momentum from the
rejected iter's training direction into the next iter's training.
Rejected candidates would then "prime" the next iter to drift in the
same (rejected) direction.

**Testable (no GPU needed):** read `yinsh_ml/training/supervisor.py`
around the revert path. Specifically look for:
- Where `best_model.pt` is reloaded (search for `_load_best_model` or
  similar around supervisor.py:1740-1780).
- Whether the trainer's `optimizer.state_dict()` is reset alongside
  the weight reload.
- The path through `TrainingSupervisor.train_iteration` when
  `decision_kind == 'reverted'` — does optimizer state carry over?

If optimizer state survives revert, that's a **bug-shaped mechanism**.
Fix: reset optimizer state to zeros (or to a saved-at-promotion
snapshot) on revert. Add regression test that asserts adam.state is
empty / matches snapshot after a reverted iter.

**Cost:** 30-60 min to inspect code + write the test if a bug is
confirmed.

#### Mechanism 3 — Buffer composition / mode collapse

Original "buffer contamination" hand-wave, re-examined: with FIFO
eviction and `max_buffer_size: 100000`, the buffer transitions from
"warm-start self-play games" → "iter_1 self-play games" over iters 2-3.
Whether that's *worse* depends on whether iter_1's games are
qualitatively different (e.g., more concentrated visit distributions
→ less exploration; or value targets clustering at extreme ±1.0 →
shorter or more decisive games).

**Testable (no extra training):** post-run, load the final buffer and
compare to the iter 1 buffer. Look for:
- **Visit-distribution sparsity:** per-row nonzero count in
  `move_probs`. If iter 4's buffer has rows with significantly fewer
  nonzero moves than iter 1's, that's visit-distribution mode collapse.
  Concretely: `np.array([(p > 0).sum() for p in buf['move_probs']])`.
- **Value-target distribution shift:** histogram of `buf['values']`
  at iter 1 vs iter 4. If iter 4 clusters at the extremes (-1.0, +1.0)
  more than iter 1, games are getting more decisive — could indicate
  one side dominating.
- **Game-length distribution:** infer from `buf['move_numbers']`
  resets. Shorter games would indicate teacher-side dominance or
  shorter exploration paths.

If any of these show a clean shift between iter 1 and iter 4 buffers,
the buffer composition story has legs.

**Cost:** ~30 min of analysis post-run, all data already on disk.

#### Mechanism 4 — Pure noise

Three iters, each with SE ~2.5% on WR. Observing 52.0 → 50.5 → 45.0
as draws from a stable ~50%-mean distribution is uncommon but not
crazy (45.0 is ~2σ below mean).

**Testable:** run more iters (relax the 5-iter cap to 10-15) and watch
the distribution. If WRs continue jumping all over with no consistent
trend, it's noise. If they keep falling, it's not.

**Cost:** extending B1B2B3 to 10 iters ≈ +30h compute. Probably not
worth doing as a standalone experiment, but worth noting that ANY
future B-family run should consider running more iters to get past
the 3-data-point ambiguity.

#### Cross-mechanism diagnostic — buffer/value-head inspection post-run

Regardless of which mechanism turns out to be dominant, **after this
run ends** (whether iter 5 promotes, reverts, or the run completes
normally), do this analysis pass:

1. Download `replay_buffer.pkl.gz` from the run dir.
2. Run buffer diagnostics (sparsity, value distribution, game length).
3. Inspect optimizer state code path in supervisor.py.
4. Write up findings as an addendum to the B1B2B3 RE-RUN #2 Done entry.

The B1B2B3 RE-RUN #2 Done entry should NOT claim "loop is
non-additive past iter 2" without first ruling out at least
Mechanism 2 (optimizer state) and Mechanism 3 (buffer composition).
Both are answerable from data we already have on disk.

#### Discipline for future write-ups

The "spiral / degenerating" framing was over-confident given 3 data
points. New rule: when the eye sees a "trend" but n < 5 and each
measurement has ±5% CI, the writeup MUST either (a) state the null
(noise) and the alternatives explicitly and assign probabilities, or
(b) be deferred until more data is in. The temptation to narrate a
single observed trajectory as a mechanism is strong — guard against it.

- **Current frozen anchor:** `models/yngine_volume_15ch_pretrain/best_supervised.pt`
  (the D.2 15-ch pretrained warm-start; re-frozen 2026-05-25 after A1 SPRT
  showed it STRONGER than the prior `best_iter_4` anchor at WR 0.905, CI95
  [0.711, 0.973]). Prior anchor: `models/branchC_volume_pretrain/best_iter_4.pt`
  (Branch C, 6-ch).
- **Last decisive SPRT verdicts:**
  - 2026-05-27: B1+B2+B3 RE-RUN #2 (phase-fix verified) —
    **NOT_STRONGER** (49-54-0, WR 0.476, CI95 [0.382, 0.571], 103 games,
    LLR -3.116). Loop produced one promotion (iter_1 via Elo override on
    Wilson REJECT) but the promoted model is statistically
    indistinguishable from the warm-start at the SPRT bar.
    Phase-weight fix WORKS empirically (buffer phase mix correct); the
    in-loop +5 WR jump at iter 2 is REPRODUCIBLE (52.0% here, 51.7%
    in prior re-run); but the gain doesn't survive independent
    re-evaluation. **B1+B2+B3 is a closed experimental branch.** Next:
    A4 + D1-partial combined.
  - 2026-05-26: B1+B2+B3 (original) — **NOT_STRONGER (44-50-0,
    WR 0.468, CI95 [0.370, 0.568]), but INTERPRETATION INVALIDATED.**
    Training ran under the phase-weight bug; superseded by RE-RUN #2.
  - 2026-05-25: A1 (D.2 iter_0 / best_supervised vs best_iter_4) —
    **STRONGER** (19-2-0, WR 0.905, CI95 [0.711, 0.973]). New anchor.
    *Unaffected by the phase-weight bug — A1 is a pretrain checkpoint
    comparison, not a self-play loop test.*
  - 2026-05-25: D.2 — **NOT_STRONGER** (160-144-0, WR 0.526, CI95
    [0.470, 0.582] vs OLD anchor `best_iter_4`). **Same phase-weight
    bug exposure as B1+B2+B3.** "Loop destroyed value" framing may
    have over-called what was really "loop trained with wrong phase
    weights." The verdict against the OLD anchor still holds as a
    measurement; the interpretation about loop behavior doesn't.
  - 2026-05-24: D.1 v2 (GAP value head) — **NOT_STRONGER** (1-15-0).
    Same exposure as D.2 / B1+B2+B3 (15-ch run). The structural
    determinism verification is independent of the bug; the
    interpretation about GAP-vs-spatial is contaminated.
  - 2026-05-23: Step 2 (MCTS-400 self-play) — **INCONCLUSIVE** at the
    400-game cap. **6-channel run — UNAFFECTED by the phase-weight
    bug.** Search-depth-isn't-the-lever conclusion still stands.

---

## Findings driving this backlog

These are the conclusions from sessions through 2026-05-25 that shape the
ranking below. If any of these turn out to be wrong, the ranking changes.

1. **Pretrain is where the strength lives now.** D.2's iter_0 (the 15-ch
   pretrained init) appears to be stronger than any later self-play iter. The
   self-play loop at MCTS-200 dilutes the warm-start rather than improving it
   (iter_1 was -107 Glicko from iter_0; iter_2/3 oscillate ~50-100 Elo below
   iter_0). The "best D.2 model" is plausibly `best_supervised.pt` itself.

2. **15-ch encoding moved pretrain metrics only marginally.** Val PAcc 0.300
   vs 0.286 (+1.4 pts), Val VAcc 0.636 vs 0.629 (+0.7 pts) over 6-ch baseline.
   But the 6-ch baseline was undertrained (3 epochs, still climbing) — fair
   comparison is unresolved.

3. **Value head VAcc plateaued early.** 0.628 at epoch 1, only 0.636 at epoch
   6 — narrow band despite extra training. Val P-loss kept decreasing
   monotonically, so the model IS still learning; argmax-VAcc may be the wrong
   metric. Plateau could be: head capacity saturation, 3-class discretization
   ceiling, yngine corpus noise floor, or "argmax doesn't capture confidence
   gains." We can't separate these yet.

4. **Wilson 0.20 promotion gate is too loose** when the warm-start is strong.
   It promotes candidates at 42% WR vs the prior best (clearly worse) just
   because they're not catastrophic. Each promotion then propagates the
   weakness forward.

5. **Search depth isn't the dominant lever** (Step 2 result). MCTS-400 vs
   MCTS-200 produced ~30-40 Elo of improvement — real but well below the
   ~70-100 Elo log-linear scaling priors would predict. MCTS-1000 is
   effectively pre-answered: marginal.

6. **The cross-architecture eval infrastructure was fragile.** The
   `use_enhanced_encoding` flag had to be plumbed through the tournament
   manager + SPRT script after a costly crash. There are several other scripts
   with the same bare-`NetworkWrapper(device=...)` pattern that would break on
   cross-encoding evaluations.

---

## Stack-rank table

Five axes, each 1-5 (higher = better). Total is a tie-breaker only — read
the axes individually for the decision you're making.

| ID | Experiment | Likely + | Unblocks | Info gain | Cost ($, h) | Impl risk | **Sum** |
|---|---|---|---|---|---|---|---|
| **A1** | Direct SPRT: D.2 pretrain vs `best_iter_4` | 4 | 5 | **5** | **5** (~30 min, $0.50) | 5 | **24** |
| **F1** | Audit + fix bare-NetworkWrapper construction sites | 5 | 5 | 3 | 5 (~1h coding, $0) | 5 | **23** |
| **A3** | Re-pretrain 6-ch baseline at 6 epochs | 3 | 5 | 5 | 4 (~3h, $5) | 5 | **22** |
| **A4** | Regression value head pretrain | 4 | 4 | 5 | 4 (~3h pretrain + 6h self-play, $15) | 4 | **21** |
| **B1** | Tighten Wilson gate to 0.50 | 4 | 3 | 4 | 4 (1 config knob; needs new self-play run, $10) | 5 | **20** |
| **A2** | Extend D.2 pretrain to 9-12 epochs via `--resume` | 2 | 2 | 3 | 5 (~3h, $5) | 5 | **17** |
| **B2** | Lower self-play LR by 5-10× | 3 | 2 | 3 | 4 ($10) | 5 | **17** |
| **B4** | Disable promotion gate entirely | 2 | 3 | 4 | 4 ($10) | 4 | **17** |
| **D1** | Self-play data corpus for pretrain | 3 | 4 | 4 | 2 (~15h, $25) | 3 | **16** |
| **E1** | GAP-native pretrain from scratch (Path 2) | 2 | 4 | 4 | 2 (~12h, $20) | 3 | **15** |
| **C1** | Branch D.3 — SE blocks | 2 | 3 | 3 | 2 (~12h, $20) | 4 | **14** |
| **B3** | 200-400 games/iter (vs 100) | 2 | 2 | 3 | 2 (2-4× cost, $20-40) | 5 | **14** |
| **C2** | Deeper trunk (more blocks) | 2 | 2 | 3 | 2 (re-pretrain needed, $30) | 4 | **13** |
| **C3** | Skip-connection value head | 2 | 2 | 3 | 2 ($30) | 3 | **12** |
| **D2** | Yngine + self-play data hybrid pretrain | 2 | 2 | 3 | 2 (~20h, $30) | 3 | **12** |
| **D3** | Filter yngine corpus by decisiveness | 2 | 2 | 3 | 3 (~4h re-pretrain, $7) | 4 | **14** |

**Scoring rubrics:**
- *Likely +*: 1 = expected to flatline; 3 = real positive plausible; 5 = strong
  mechanism-based prior.
- *Unblocks*: 1 = one-off; 3 = makes ~one downstream choice cleaner; 5 = gates
  multiple downstream experiments.
- *Info gain*: 1 = result is ambiguous regardless; 3 = narrows the field; 5 =
  decisively settles a current open question.
- *Cost*: 1 = >$50 / >20h; 3 = ~$15 / ~8h; 5 = <$2 / <2h.
- *Impl risk*: 1 = high chance of bugs / wasted compute; 5 = config-knob,
  battle-tested code paths.

---

## Detailed write-ups

### A1 — Direct SPRT: D.2 pretrain vs `best_iter_4`

**Goal:** measure `models/yngine_volume_15ch_pretrain/best_supervised.pt`
directly against the frozen 6-ch anchor `best_iter_4.pt`, with no D.2 self-play
loop in between.

**Mechanism:** the D.2 self-play loop appears to be diluting the strength of
the pretrained warm-start (iter_1-3 all 50-100 Glicko below iter_0). If true,
the strongest checkpoint D.2 produced is iter_0 itself. SPRT'ing it directly
separates "is the 15-ch pretrain stronger than 6-ch + self-play?" from "is the
full D.2 loop stronger?" — currently entangled.

**Supporting evidence:**
- D.2 tournament data: iter_0 beat iter_1 116-84 (CI95 [0.511, 0.646], real
  edge of ~107 Glicko Elo).
- iter_2 vs iter_1 was 99-101 (effectively tied, candidate slightly worse).
- iter_3 dropped back to Glicko 1467 vs iter_2's 1478.
- The pattern is noisy oscillation in a band well below iter_0, not climbing
  recovery.

**Reasons to not believe:**
- **Glicko within a small population is noisy.** With 3-4 models and 200 games
  per pair, the ratings carry meaningful CI. But iter_0's edge over iter_1 had
  CI95 lower bound 0.511 — strictly above 0.5 — so it's not pure noise.
- **EMA weights might tell a different story.** Tournament uses
  `use_ema_for_eval=True` — both iter_0 and iter_1 played with EMA-smoothed
  weights. The non-EMA iter_4 (last self-play state, what the SPRT will
  actually consume via `*_ema.pt` siblings) could differ.
- **iter_0 was the warm-start before any self-play; its eval used the EMA**
  which on a fresh init starts identical to the raw weights. Later iters had
  more EMA-vs-raw divergence. Possible artifact.

**Methodology:**
```bash
python scripts/eval_vs_frozen_anchor.py \
    --candidate models/yngine_volume_15ch_pretrain/best_supervised.pt \
    --anchor    models/branchC_volume_pretrain/best_iter_4.pt \
    --sprt --sprt-p1 0.60 --sprt-max-games 400 \
    --device cuda --quiet-mcts \
    --output-json logs/d2_pretrain_iter0_vs_frozen.json
```

Same protocol as the D.1 / Step 2 SPRT — opening-sample-plies=20, n=400 cap,
WR-0.60 promotion bar.

**Cost:** ~30 min if decisive (which both D.1 v2 outcomes were); up to ~4h on
cap if borderline. $0.50 - $4 of cloud compute.

**Dependencies:**
- Blocked on: D.2 self-play run completing (so the cloud box is available for
  the SPRT call). Alternative: run on a separate fresh box; checkpoint is on
  laptop.
- Unblocks: interpretation of the whole D.2 result. Without A1, we can't
  separate the encoding axis from the self-play loop's behavior.

**Definition of done:** JSON written to `logs/d2_pretrain_iter0_vs_frozen.json`
with a verdict (STRONGER / NOT_STRONGER / INCONCLUSIVE). Result logged in the
"Done" section below.

**Open questions:**
- If iter_0 IS STRONGER but iter_4 ends up INCONCLUSIVE, that's near-proof the
  self-play loop is hurting. Then B1 (tighten gate) becomes the highest-prior
  follow-up.
- If iter_0 is also INCONCLUSIVE, the encoding lift alone is small. A4 (value
  head loss change) becomes the higher-prior next test.

---

### A3 — Re-pretrain 6-ch baseline at 6 epochs

**Goal:** train a fresh 6-channel supervised checkpoint to 6 epochs (same
schedule as D.2's 15-ch pretrain) so we have an apples-to-apples comparison
where only the *encoding* differs.

**Mechanism:** the current 6-ch baseline (`models/yngine_volume_pretrain/best_supervised.pt`)
was trained 3 epochs. The doc notes it "climbed every epoch (VAcc 0.612 →
0.619 → 0.629), no overfit" — i.e. it was undertrained, not converged. Every
D.2 comparison against this baseline carries a "more epochs OR more channels?"
confound.

**Supporting evidence:**
- 6-ch baseline at epoch 3: VAcc 0.629, PAcc 0.286.
- 15-ch D.2 at epoch 3 (with T_max=6 schedule, so different LR curve): VAcc
  0.631, PAcc 0.277. Slightly different despite "same epoch count" — the LR
  schedule shape differs.
- 15-ch D.2 at epoch 6: VAcc 0.636, PAcc 0.300. Of the +0.7 VAcc improvement
  vs 6-ch baseline, *some non-zero fraction* is purely the extra training
  budget, not the encoding.

**Reasons to not believe:**
- **Encoding might still be the dominant factor.** Even if a 6-ch-6-epoch
  beats 6-ch-3-epoch, it may still fall short of 15-ch-6-epoch. The
  comparison just lets us *quantify* the encoding contribution.
- **Reproducibility of the original 6-ch pretrain:** if we can't fully
  reproduce the 6-ch-3-epoch result (different random seed, different
  hardware), the baseline is shifting under our feet. Worth checking.

**Methodology:**
```bash
python scripts/run_supervised_pretraining.py \
    --data-dir expert_games/yngine_volume_mmap/ \
    --output-dir models/yngine_volume_6ch_pretrain_v2 \
    --epochs 6 --batch-size 512 --lr 1e-3 \
    --num-channels 256 --num-blocks 12 \
    --value-head-type spatial
```

(Note: do NOT pass `--use-enhanced-encoding` — that's the whole point.)

Then evaluate:
- Val PAcc / VAcc per epoch (compare directly to D.2 15-ch trajectory)
- SPRT vs `best_iter_4` (does 6-ch-6-epoch alone already beat the 6-ch-3-epoch-
  plus-self-play champion?)

**Cost:** ~3h on 5090, ~$5. SPRT after: 30 min - 4h.

**Dependencies:**
- Need the original 6-ch yngine_volume mmap corpus locally or pull from gh
  release. The 6-ch mmap was created on the original training box and torn
  down; the *npz* survives in the gh release. So step 0 is regenerating the
  6-ch mmap (use `scripts/convert_npz_to_mmap_shards.py` — ~30 min).
- Blocks: any future "is encoding the lever?" question.
- Unblocks: A1's interpretation (if 6-ch-6-epoch beats `best_iter_4`, our
  whole baseline assumption was off).

**Definition of done:** new checkpoint at `models/yngine_volume_6ch_pretrain_v2/best_supervised.pt`,
plus a comparison-table entry in `VOLUME_PRETRAIN_RESULTS.md` showing per-
epoch PAcc/VAcc trajectory next to 15-ch.

**Open questions:**
- Do we ALSO want a 9-12 epoch run of the 6-ch baseline, in case it keeps
  climbing past 6? Cheap to extend via `--resume`. Defer until we see the
  6-epoch result.

---

### A4 — Regression value head pretrain

**Goal:** replace the 3-class cross-entropy value head with a scalar regression
head (`tanh` output, MSE loss against `value ∈ [-1, 1]`), then re-run the D.2
pipeline. Tests whether the value-head plateau is caused by the discretization,
not the architecture.

**Mechanism:** the current value head predicts {-1, 0, +1} discretized into 3
classes. YINSH outcomes have a real "decisiveness" gradient (2 captures ahead
≠ tied with momentum ≠ 1 capture behind) that the 3-class collapse throws
away. A scalar head can encode that gradient. If the value-head plateau is
caused by representation loss in the target structure, this fixes it directly.

**Supporting evidence:**
- Val P-loss decreased monotonically through 6 epochs of D.2 pretrain (3.16 →
  2.84) while VAcc was nearly flat (0.628 → 0.636). The argmax stayed similar
  but the *distribution* kept tightening — exactly the symptom of a model
  that's still learning but on a metric the loss-discretization can't
  resolve.
- The original AlphaZero paper used a scalar `tanh` value head; the 3-class
  variant here is an inherited convention, not a justified design choice.
- The codebase already supports a `value_mode='regression'` path in
  `NetworkWrapper.__init__` (see `wrapper.py` line 28). Not exercised in
  current configs but the plumbing exists.

**Reasons to not believe:**
- **Value-head plateau could be the data ceiling (theory III).** If yngine
  outcomes are noise-limited (many positions could go either way), no loss
  function gets you above the Bayes error from raw outcomes. Regression
  doesn't help in that case.
- **MSE on raw outcomes can be *worse* than CE on classes.** With only
  {-1, 0, +1} as raw targets, MSE pushes the network toward 0 (the mean),
  which is uninformative. We may need to use MCTS-rolled-out value estimates
  as targets, not raw outcomes — and those aren't in the yngine corpus. This
  is closely related to D1.
- **Self-play loss surface differs:** the self-play trainer uses CE on a
  discretized MCTS rollout value. If we pretrain with regression but
  self-play with classification, the warm-start washes out on iter 1.
  Probably need to *also* switch self-play to regression to be consistent.
  Doubles the code change.

**Methodology:**

Phase 1 — Add regression head support to `run_supervised_pretraining.py`
**— prepped 2026-05-25 during B1B2B3 run:**
- ✅ `--value-mode {classification, regression}` CLI flag added (default
  classification; back-compat preserved).
- ✅ `--num-value-classes` override added (classification mode only).
- ✅ Training + eval loops branch on `model.value_mode`: regression uses
  `F.mse_loss(value_pred, values)` against the scalar tanh output;
  classification keeps the CE-on-discretized-classes path.
- ✅ Value-accuracy metric branches: regression logs *sign accuracy*
  (proxy — does the model predict the right side of zero?); classification
  keeps argmax-class accuracy.
- ✅ Smoke-tested both paths construct and forward correctly (regression
  returns `(B,)` value tensor; MSE loss computes cleanly).
- Save under a new path when actually run: `models/yngine_volume_15ch_pretrain_regr/`.

Phase 1.5 — Self-play side wiring (DEFERRED until we actually run A4):
- `scripts/run_training.py:452` constructs `NetworkWrapper` without
  passing `value_mode`. A regression-trained checkpoint loaded here will
  hard-fail at `load_model` because the value head's final-layer output
  dim is 1 (regression) vs `num_value_classes` (classification). The
  hard-fail is *desired* — louder than silent misload.
- When A4 actually runs, fix by either (a) adding `value_mode` as a
  config knob in the training YAML, or (b) auto-detecting from the
  checkpoint's last value-head layer output dim (parallel to existing
  encoding / capacity auto-detect in `wrapper.py:160`). Prefer (b).

Phase 2 — Decide on self-play matching:
- Option A: also switch self-play to regression. Requires changes in
  `yinsh_ml/training/trainer.py`'s value loss path. Bigger lift.
- Option B: pretrain regression, but cast back to classification at
  warm-start time (probably won't work cleanly — the network's last layer
  shape differs).
- Recommended: bite the bullet on Option A, since the whole point of A4 is
  to test the value-head structure.

Phase 3 — Run the same D.2 self-play loop with the regression-head pretrain
init.

Phase 4 — SPRT vs `best_iter_4` (the 3-class champion).

**Cost:**
- Code changes: ~2-3h (single dev-day).
- Pretrain: ~3h, ~$5.
- Self-play: ~6h, ~$10.
- SPRT: 30 min - 4h, ~$1-5.
- **Total: ~12h on 5090, ~$20**, plus dev time.

**Dependencies:**
- Blocks: nothing critical; A4 is a parallel-track experiment.
- Unblocks: if positive, it changes the recommended architecture for ALL
  future Branch D experiments. If negative, narrows the value-head plateau
  hypothesis to "head capacity" or "data ceiling" rather than "target
  discretization."

**Definition of done:** SPRT verdict in
`logs/d2_regr_iter4_vs_frozen.json`, regression-head supervised pretrain
checkpoint saved, code changes committed (gated behind a flag so the
classification path stays default).

**Open questions:**
- How do we handle the self-play replay buffer's pre-existing
  classification-target structure during transition? Probably need to
  invalidate (delete) any prior buffer when switching modes.
- Does the EMA path need adjusting? Same network, same parameters — should
  be fine, but worth a smoke test.

---

### B1+B2+B3 — in progress (launched 2026-05-25 23:35 UTC)

Bundled stop-the-leak run. All three knobs changed in a single config
(`configs/branchB1B2B3_mcts200.yaml`) and launched together with warm-start
from `best_supervised.pt` (the refrozen anchor). Run dir
`runs_branchB1B2B3/20260525_233508` on RTX 5090.

| Knob | Value (was → now) |
|---|---|
| B1 `arena.promotion_threshold` | 0.20 → 0.50 |
| B1 `arena.games_per_match` | 100 → 200 |
| B2 `trainer.lr` | 1e-4 → 1e-5 |
| B3 `self_play.games_per_iteration` | 100 → 200 |
| (unchanged) encoding | enhanced (15-ch) |
| (unchanged) num_iterations | 5 |

**Attribution caveat (logged before result is known):** because three knobs
move together, a STRONGER verdict cannot attribute the gain to a specific
knob. We are deliberately bundling because (a) the priors on each knob being
individually load-bearing are similar — they all target the same dilution
mechanism — and (b) attribution ablations would each cost 4-6h, eating
budget that's better spent compounding gains. **Policy decided 2026-05-25
during planning:** if STRONGER, skip attribution; proceed directly to A4 to
compound. If NOT_STRONGER, use the decision matrix below to pick the next
experiment.

**Decision matrix — outcome → next experiment**

Three independent axes of evidence the run will produce: (1) SPRT verdict
vs `best_supervised.pt`, (2) gate behavior across 5 iters, (3) Glicko
trajectory shape.

|  | All 5 promote | Mixed promote/revert | 0 promote (stuck at iter_0) |
|---|---|---|---|
| **STRONGER** (CI95 lower > 0.50) | Loop unambiguously works → **A4** to compound | Same → **A4** | (contradictory — gate wouldn't reject in a STRONGER world) |
| **NOT_STRONGER, preserved** (CI hugs 0.50) | (rare) | Leak fixed, no gain → **A4** *or* **D1** | Gate froze us at iter_0 → **B4** (disable gate as negative control) |
| **NOT_STRONGER, leaks** (CI clearly < 0.50) | Gate ineffective → **B4** + escalate | Tuning insufficient → **A4** mandatory, **D1** parallel | Worst case (gate too tight AND no signal) → **A4** + **D1** in series |

**Prior probabilities** (recorded so we can calibrate later): p≈0.55
preserved-no-gain, p≈0.25 STRONGER, p≈0.20 still-leaks. Reasoning: B1+B2
should at minimum *prevent* destruction (gate now rejects when iter_N+1 is
worse than iter_N), so the "leak" outcome requires both the gate AND the
LR fix to be insufficient — unlikely. B3 adds signal volume but may not
add signal *quality*, so STRONGER requires the volume to be load-bearing,
which is a weaker prior.

**Why bundled now and not staged:** A1 showed the warm-start is decisively
strong. Every additional D.2-style run on top of it costs ~12h of
value-destruction. The fastest way to learn whether the loop is
fundamentally salvageable in the fine-tuning regime is to throw all three
plausibly-helpful knobs at it. If salvageable, downstream attribution can
wait. If not, we need a structural change (A4/D1), not a knob tweak.

**While the run goes** (CPU-only work during GPU-bound self-play): F1
(bare-NetworkWrapper cleanup), A4 code changes (so the next experiment is
launch-ready), D1 sketch (verify MCTS-target dump path exists).

**Replay buffer kept (2026-05-26 12:38 UTC):** 12 MB compressed
(~2.2 GB uncompressed, ~50K MCTS-target samples across 5 self-play
iters with the warm-start teacher) saved to
`experiments/branchB1B2B3_run_2026-05-26/full_run_dir/20260525_233508/replay_buffer.pkl.gz`.
Useful for D1 as a partial corpus from a strong teacher — already
generated via the supervisor's proper batched MCTS path with subtree
reuse. Lets D1 test "does pretrain on MCTS targets beat pretrain on
yngine outcomes?" without the 10-15h generation cost up front.

**Definition of done:** SPRT JSON at `logs/branchB1B2B3_iter4_vs_anchor.json`,
plus Done entry below following the standard template, plus the next
experiment kicked off (or queued with config ready) per the matrix above.

---

### B1 — Tighten Wilson gate to 0.50

**Goal:** change the promotion gate from `promotion_threshold: 0.20` to
`0.50` (or higher), so candidates only promote if they're at-least-as-strong
as the current best. Re-run D.2 self-play with this change.

**Mechanism:** the current gate at 0.20 promotes any candidate that wins
≥20% of head-to-head games. That bar is so low it effectively *just* filters
out total catastrophes. With a strong warm-start (D.2 iter_0), every later
iter was visibly worse but still passed the gate at 42-49% WR. A 0.50 gate
would have rejected iter_1 (42% WR) and reverted to iter_0; subsequent iters
would have re-trained from iter_0, possibly preserving the warm-start.

**Supporting evidence:**
- D.2 iter_1 promoted at 42% WR despite CI95 [0.354, 0.489] putting the
  candidate's true WR confidently below 0.50.
- The doc (`VOLUME_PRETRAIN_RESULTS.md`) already flagged Wilson 0.20 as
  "loose" but it was OK in Branch C because the warm-start there was weaker
  than the first self-play iter would naturally produce.
- The pattern of "promoted but worse" repeated in iter_2 (49.5% WR, CI95
  [0.426, 0.564]).

**Reasons to not believe:**
- **Tighter gate might stall the loop.** If iter_1 is genuinely *near-equal*
  to iter_0 (just noise), Wilson 0.50 might never accept any candidate,
  leaving us stuck at iter_0 forever. That's not necessarily *bad* (if
  iter_0 is the best we've got), but it means self-play stops contributing.
- **Wilson at 0.50 with limited games is noisy.** With 200 games and
  unbiased H2H, a candidate that's truly equal to iter_0 will pass the gate
  ~50% of the time and fail ~50%. Promotion becomes a coin-flip on equal
  candidates. We'd want at least 400 games per H2H to get a tight CI around
  0.50, which roughly doubles tournament cost per iter.
- **The whole loop may be the wrong tool** — if MCTS-200 self-play can't
  improve over a strong init regardless of gate, tightening the gate just
  preserves the floor without enabling progress. We need a *different
  mechanism* (more sims, different loss, etc.) to make the loop additive
  again.

**Methodology:**
1. Modify `configs/branchD2_enhanced_mcts200.yaml`:
   ```yaml
   arena:
     promotion_threshold: 0.50  # was 0.20
     games_per_match: 200       # was 100 → tighter CI around 0.50
   ```
2. Optionally bump `tournament_games: 200` so gate-relevant H2H volume is
   higher.
3. Re-run D.2 self-play (warm-start from same `best_supervised.pt`).
4. Compare iter_4_ema vs A1's iter_0 result. If iter_4 has caught up to or
   exceeded iter_0, the gate fix worked.

**Cost:** ~6h self-play, ~$10. (Games-per-match bump may push to 7-8h.)

**Dependencies:**
- Blocked by: A1 result. If A1 says iter_0 already beats `best_iter_4`, B1
  is the natural follow-up. If A1 is INCONCLUSIVE / NOT_STRONGER, B1 won't
  help on its own (the warm-start isn't strong enough to preserve).

**Definition of done:** D.2 re-run with 0.50 gate; comparison logged. If
all 5 iters revert to iter_0, that's the result — "self-play cannot improve
the warm-start at this MCTS budget."

**Open questions:**
- Should the gate be 0.50 exactly, or something more like 0.55 (require a
  real edge)? 0.50 = "non-worse"; 0.55 = "small positive." Worth a test
  matrix if compute allows.
- Should `tournament_sliding_window` increase from 3 → 5 to give the gate
  more historical context? Marginal change; not load-bearing.

---

### B2 — Lower self-play LR by 5-10×

**Goal:** reduce the self-play training LR from 1e-4 to 1e-5 (or 2e-5), so
gradient updates are gentler on the converged pretrained weights.

**Mechanism:** the D.2 pretrain LR schedule ended at 1e-5 (cosine decay to
eta_min). The self-play then takes over at lr=1e-4 (`configs/branchD2_enhanced_mcts200.yaml`,
`trainer.lr: 0.0001`). That's a **10× jump** at the moment we transition from
a well-converged supervised state to an MCTS-target-driven update — likely
perturbing the converged weights more than necessary. Lowering matches the
pretrain end-LR and lets the self-play loop *fine-tune* rather than retrain.

**Supporting evidence:**
- D.2 iter_1 dropped 107 Glicko Elo from iter_0 after one self-play epoch —
  exactly the symptom of "LR too high for the starting state."
- The original Branch C config used the same 1e-4 LR, but Branch C's warm-
  start was significantly weaker (Branch C pretrain VAcc unknown but the
  iter_0 vs iter_1 transition in Branch C was an *improvement*, not a
  regression).
- This is a 1-character config change.

**Reasons to not believe:**
- **Lower LR might also lose information.** Self-play targets *are*
  different from pretrain targets (MCTS visit distributions vs. argmax
  policy; rolled-out values vs. raw outcomes). If self-play has new signal
  to learn, lower LR slows that learning. We'd avoid regression but also
  avoid progression.
- **LR is one of several knobs.** The right value might depend on
  batch_size, epochs_per_iteration, games_per_iter, etc. Single-knob testing
  may underestimate the interaction.

**Methodology:**
```yaml
trainer:
  lr: 0.00001  # was 0.0001
```

Or test a range: 1e-4 (control), 5e-5, 1e-5. Three short runs (3 iters each,
to save compute) would map the response curve. Then pick the best and run
the full 5-iter loop.

**Cost:** 1 run = ~6h, $10. Range test (3 runs × 3 iters each) = ~10h, $17.

**Dependencies:**
- Bundles naturally with B1 (single re-run could test both at once, but then
  attribution is muddier). Recommend B1 first (config-only), then B2 if B1
  alone doesn't fix it.

**Definition of done:** SPRT verdict for the lowered-LR run vs
`best_iter_4` (or vs A1's iter_0 result for self-play-loop-fitness).

**Open questions:**
- Should we use a *schedule* during self-play too? Currently flat LR. A
  cosine schedule across the 5 iters might be smarter than a constant.

---

### A2 — Extend D.2 pretrain to 9-12 epochs via `--resume`

**Goal:** continue training the D.2 pretrain (`yngine_volume_15ch_pretrain`)
from its 6-epoch state for another 3-6 epochs, using the resume support
shipped in commit `d5e5151`.

**Mechanism:** PAcc was still climbing at epoch 6 (0.271 → 0.300 monotonic;
+0.009 in the last epoch). The cosine LR schedule had bottomed out at eta_min
~1e-5, so the model was effectively done with the current schedule. A fresh
cosine over 3-6 more epochs might extract additional signal.

**Supporting evidence:**
- Epoch trajectory PAcc 0.271 → 0.275 → 0.277 → 0.284 → 0.291 → 0.300 — the
  delta from epoch 5→6 (+0.009) was larger than 4→5 (+0.007). Acceleration,
  not deceleration.
- Resume support is now tested (commit `d5e5151`); cosine schedule rebuilds
  fresh against the new `--epochs`, then advances to the resume point.

**Reasons to not believe:**
- **Val P-loss decrease was decelerating** (epoch deltas: -0.43, -0.03, -0.05,
  -0.06, -0.06, -0.09 — wait, that's actually accelerating again at the end).
  Mixed signal.
- **VAcc plateaued** (0.628 → 0.630 → 0.631 → 0.629 → 0.633 → 0.636 over 6
  epochs — narrow band). Extra epochs may move PAcc but not VAcc, which is
  the more important head for self-play warm-start.
- **The D.2 self-play loop's behavior was driven mostly by iter_0's
  strength, not its precise PAcc value.** A 0.305 PAcc instead of 0.300
  might not change the self-play outcome perceptibly.

**Methodology:**
```bash
python scripts/run_supervised_pretraining.py \
    --data-dir expert_games/yngine_volume_15ch_mmap/ \
    --use-enhanced-encoding \
    --value-head-type spatial \
    --output-dir models/yngine_volume_15ch_pretrain \
    --epochs 9 --batch-size 512 --lr 1e-3 \
    --num-channels 256 --num-blocks 12 \
    --resume
```

Note: `--epochs 9` (or 12) is the NEW total epoch count. The resume logic
will start from epoch 7 and run through 9. The cosine schedule rebuilds with
T_max=9 and advances 6 steps to land at the right LR for "epoch 7 of 9."

**Cost:** ~50min/epoch × 3-6 epochs = ~3-5h, ~$5-9.

**Dependencies:**
- Blocked by: nothing. The corpus + checkpoint + last_resume_state.pt are
  all on the box (or laptop).
- Unblocks: if PAcc climbs significantly (≥0.310) without VAcc moving, that
  reinforces the "value head is the bottleneck" reading and de-risks A4.

**Definition of done:** new `best_supervised.pt` (overwritten in-place since
output_dir is the same) with epoch 9+ metrics in the log. Optionally A1-style
SPRT against `best_iter_4` to see if extra pretrain matters end-to-end.

**Open questions:**
- Is the in-place overwrite of `best_supervised.pt` correct, or do we want a
  separate output_dir for the extended run? Suggest: separate dir
  (`yngine_volume_15ch_pretrain_v2`) so we can compare both checkpoints.

---

### B4 — Disable promotion gate entirely

**Goal:** run D.2 self-play with no gate — just take the latest checkpoint as
the new "best" each iter. Tests whether the gate is helping or hurting.

**Mechanism:** as a negative control. If the no-gate run produces an iter_4
that's *worse* than the gated run's iter_4, the gate was net-positive
(filtering out bad candidates). If the no-gate run produces an *equal or
better* iter_4, the gate was net-zero or net-negative on this regime.

**Supporting evidence:**
- D.2's loose gate promoted clearly-worse candidates. If a tight gate (B1)
  fixes that, no-gate would presumably be even worse — confirming the gate
  isn't useless, just mis-tuned. Useful as a calibration point.

**Reasons to not believe:**
- **Less informative than B1.** B1 directly tests the tuning hypothesis. B4
  tests "does the gate matter at all" — coarser question.
- **Could waste compute.** If the gate is genuinely helping, no-gate run
  produces a worse model and the run is throwaway info-only.

**Methodology:**
```yaml
arena:
  promotion_threshold: 0.0  # promote any candidate
```

Or modify the supervisor to short-circuit the gate entirely.

**Cost:** ~6h, ~$10.

**Dependencies:** orthogonal to others.

**Definition of done:** SPRT vs `best_iter_4` AND vs the gated D.2 result.
Two-way comparison clarifies the gate's contribution.

**Open questions:**
- Is "always-promote" the same as "promote_threshold=0.0", or does the
  Wilson math handle 0.0 weirdly (e.g. divide-by-zero)? Check before
  running.

---

### D1 — Self-play data corpus for pretrain

**Goal:** generate ~100K self-play games using the best D.2 model (probably
iter_0), capture MCTS visit distributions + rolled-out values as targets,
build a 15-channel corpus from them, re-pretrain a new init from this corpus.

**Mechanism:** the yngine corpus uses raw outcomes (W/L/D) as value targets.
Many positions have ambiguous outcomes — the result depends on subsequent
play decisions. MCTS-rolled-out values use search depth to give a
position-conditioned value estimate that's plausibly less noisy than the raw
outcome. Better targets → higher achievable VAcc → stronger warm-start.

**Supporting evidence:**
- The D.2 value head plateaued at VAcc 0.636 — possibly the Bayes error on
  yngine raw outcomes.
- AlphaZero and successors use MCTS-rolled-out values as targets precisely
  because they encode search-informed value, not just raw rollout outcomes.
- The codebase already has self-play data collection infrastructure.

**Reasons to not believe:**
- **Self-play targets aren't free of bias.** They're biased toward whatever
  the *current* policy + value head predicts. A pretrain on self-play data
  may just reinforce existing model knowledge rather than add new signal —
  the "self-play collusion" failure mode the doc flagged elsewhere.
- **Generation cost is real.** ~100K games at MCTS-200 with batched eval =
  ~10-15h on 5090. Plus the pretrain on top.
- **The strength gap matters.** Our best self-play teacher (D.2 iter_0) is
  ~50 Elo below the previous champion `best_iter_4`. Using `best_iter_4`
  itself as the teacher might be better — but it's 6-ch and would produce a
  6-ch corpus.

**Methodology:**
1. Pick teacher: `models/yngine_volume_15ch_pretrain/best_supervised.pt`
   (refrozen anchor, A1 verified STRONGER vs the prior Branch C
   champion — A1 result re-points D1 here away from `best_iter_4`).
   Genuinely 15-ch, no re-encoding problem.
2. **Generator script does NOT exist** — D1-sketch finding 2026-05-25.
   The closest is `scripts/run_selfplay_worker.py` which calls
   `SelfPlay.generate_games()` and prints the count but throws the
   results away. The data tuple from `generate_games` IS the right
   shape: per-game `(states, policies, values, history)` where
   `policies` are MCTS visit distributions and `values` are MCTS root
   values per position (Fix #1 in self_play.py:1647). Connector
   script needed (~2-3h coding):
   - Load teacher with auto-detect: `NetworkWrapper(model_path=...)`
   - Loop `SelfPlay.generate_games(batch)` in chunks of ~1000 games
   - **Scatter** per-position `policies` (over valid moves) into the
     full 7433-slot move-encoder space before saving — pretrain
     expects shape `(N, total_moves)` for soft targets, or `(N,)`
     argmax indices for hard targets. Soft preferred (more signal).
   - Save as npz with `states.npy`, `policies.npy` (soft, `(N, 7433)`)
     or `policy_indices.npy` (argmax, `(N,)`), and `values.npy` (`(N,)`).
     Schema matches `run_supervised_pretraining.py:97-114`.
   - Resume support — persist a "games-done" counter so crashes don't
     restart from zero (100K games is ~10-15h on 5090).
3. Convert npz → mmap shards (`scripts/convert_npz_to_mmap_shards.py`,
   already exists, schema-agnostic).
4. Pretrain on the mmap shards via existing
   `scripts/run_supervised_pretraining.py --data-dir <mmap_dir>
   --use-enhanced-encoding --value-head-type spatial --epochs 6`.
5. SPRT the resulting init vs `best_supervised.pt` (new anchor).

**Cost:**
- Game generation: ~10-15h, ~$20.
- Corpus conversion: ~1h, ~$1.
- Pretrain: ~5h, ~$8.
- SPRT: ~1-4h, ~$2-5.
- **Total: ~17-25h, ~$30**.

**Dependencies:**
- Unblocks: a meaningful answer on theory III (data ceiling). If self-play-
  pretrained warm-start dramatically beats yngine-pretrained, the corpus
  was the limit.

**Definition of done:** new corpus + checkpoint + SPRT verdict logged.

**Open questions:**
- Use soft policy targets (full visit distributions) or argmax? The
  policy loss code already branches on shape. Soft targets carry more info.
- Use rolled-out root values or final-game outcomes as value targets? Doc
  notes the self-play trainer uses both (`trainer.value_loss_weights:
  [0.5, 0.5]`). Worth replicating that mix.

---

### E1 — GAP-native pretrain from scratch (Path 2)

**Goal:** train a 15-ch (or 6-ch?) supervised pretrain from scratch with the
GAP value head as the architecture from epoch 0, then self-play, then SPRT.
Tests whether D.1's failure mode was *warm-start specialization* (theory A)
or *GAP-is-fundamentally-wrong* (theory B).

**Mechanism:** D.1 v1 + v2 both warm-started a GAP head from a spatial-head
checkpoint, then self-played. Both failed SPRT 1-15-0 (structural
determinism verified). Hypothesis A: the trunk's 30M params were tuned for
the spatial head's output shape and brief self-play couldn't unlearn that.
Hypothesis B: GAP is fundamentally wrong for YINSH (position-discarding via
average pooling kills value signal). E1 tests A directly — a from-scratch
GAP-native trunk has no specialization to overcome.

**Supporting evidence:**
- D.1 v1/v2 SPRTs were verified as structural determinism (same trajectory
  across seeds), so the failures aren't noise. They're real — but they
  don't distinguish theory A from theory B.
- Network code supports `--value-head-type gap_v2` end-to-end (D.1 v2 used
  it).

**Reasons to not believe:**
- **GAP-native pretrain may just confirm theory B.** If the architecture
  fundamentally can't learn YINSH value, no amount of "fresh training" fixes
  it. Then E1 is throwaway compute.
- **6-ch GAP might be more sensible than 15-ch GAP.** GAP is averaging
  spatial info; doing it with 15 input channels gives more raw signal to
  start with but may not help if the head's bottleneck is the avg-pooling.

**Methodology:**
```bash
# Option A: 15-ch corpus (existing)
python scripts/run_supervised_pretraining.py \
    --data-dir expert_games/yngine_volume_15ch_mmap/ \
    --use-enhanced-encoding \
    --value-head-type gap_v2 \
    --output-dir models/yngine_volume_15ch_gap_pretrain \
    --epochs 6 --batch-size 512 --lr 1e-3 \
    --num-channels 256 --num-blocks 12

# Option B: 6-ch corpus
# Same but without --use-enhanced-encoding and pointed at 6-ch corpus
```

Then full D.2-style self-play loop and SPRT.

**Cost:** pretrain ~3h ($5) + self-play ~6h ($10) + SPRT ~1-4h ($2-5) =
**~12h, ~$20**.

**Dependencies:**
- Sequence after A1, A3, A4 — they all narrow the search space better than
  E1.
- Unblocks: a confirmed answer on the GAP architecture question (which has
  been hanging since D.1 v2).

**Definition of done:** SPRT verdict.

**Open questions:**
- Should we test both 6-ch and 15-ch GAP-native? The 6-ch version is a
  cleaner test of "GAP itself works" (no encoding confound). Pick one and
  defer the other.

---

### C1 — Branch D.3: SE (squeeze-and-excitation) blocks

**Goal:** add channel-attention SE blocks to the trunk, alongside the
existing spatial-attention blocks. Tests whether selective per-channel
feature emphasis helps.

**Mechanism:** spatial attention asks "which board cells matter" — already
present. SE blocks ask "which feature channels matter" — orthogonal axis.
~5K extra params per block; cheap. Well-attested in Leela / KataGo.

**Supporting evidence:**
- The doc has this teed up as the next architecture experiment after D.2.
- Mechanism is sound; the question is just whether YINSH benefits enough to
  justify the run.

**Reasons to not believe:**
- **D.2's near-zero ceiling movement weakens C1's prior.** If 15-ch
  encoding (which added 9 input channels of explicit features) barely
  moved metrics, channel attention adding implicit per-channel scaling
  probably also won't.
- **More likely to compound with other changes than stand alone.** SE +
  regression value head (A4) + lowered LR (B2) bundled might be much more
  than the sum of parts.

**Methodology:** add SE block module, integrate into trunk, run pretrain +
self-play + SPRT.

**Cost:** ~12h pretrain + self-play, ~$20. Plus coding (~half-day).

**Dependencies:**
- De-prioritized in light of D.2's findings; sequence behind A4, B1, D1.

**Definition of done:** SPRT verdict.

---

### F1 — Audit + fix bare `NetworkWrapper(device=...)` construction sites

**Goal:** find every script that constructs `NetworkWrapper(device=...)` then
calls `.load_model(path)`, replace with `NetworkWrapper(model_path=path,
device=...)` so the auto-detection path engages. Add a test that ensures
cross-architecture checkpoint loads work.

**Mechanism:** the bug we hit in D.2 (encoder channel mismatch crash in
tournament + anchor eval) was due to bare-construction-then-load patterns.
The wrapper has auto-detection logic, but only when `model_path` is passed
to `__init__`. We fixed the critical path (`tournament.py`,
`eval_vs_frozen_anchor.py`); other scripts have the same pattern.

**Supporting evidence:**
- Scripts confirmed to have the pattern (from a grep we did):
  - `play_step.py:228`
  - `cross_era_tournament.py:48`
  - `eval_compare_checkpoints.py:144`
  - `gpu_probe.py:47`
  - `tier_a_threaded_parity.py:244`
  - `replay_h2h_game.py:157, 159`
  - `eval_head_to_head.py:190`
  - `play_vs_model_mcts.py:322`

**Reasons to not believe:** none. This is pure technical debt cleanup.

**Methodology:**
1. For each script, change `NetworkWrapper(device=...)` followed by
   `.load_model(path)` to `NetworkWrapper(model_path=path, device=...)`.
2. Some scripts may need additional logic if they don't know the path
   up-front. Audit case-by-case.
3. Add a regression test in `yinsh_ml/tests/` that loads a 15-ch checkpoint
   into a fresh wrapper via the constructor path.

**Cost:** ~1h coding, ~$0 compute.

**Dependencies:** none.

**Definition of done:** all 8 sites fixed, regression test added,
committed.

**Open questions:**
- Should we ALSO add channel auto-detection to `NetworkWrapper.load_model`
  itself (not just `__init__`)? Would prevent this class of bug
  permanently. Tradeoff: the current hard-fail is a deliberate safety
  check; auto-detect could mask wrong-flag callers. Probably better to
  fix sites explicitly than weaken the guard.

---

## Recommended sequencing

Conditioned on the D.2 SPRT outcome (currently pending):

### If D.2 SPRT is STRONGER (unlikely given iter trajectory):
1. **A1** — confirm whether iter_0 was even stronger (free info).
2. **C1** (SE blocks) — stack with the encoding gain.
3. **B1** (tighten gate) — preserve gains in future loops.

### If D.2 SPRT is INCONCLUSIVE WR ~0.55 (Step-2-style small edge):
1. **A1** — quantify the pretrain-alone contribution.
2. **A3** — get a fair-comparison 6-ch baseline at 6 epochs.
3. **A4** — regression value head (highest mechanism-based prior).

### If D.2 SPRT is INCONCLUSIVE WR ~0.50 or WEAKER (most likely):
1. **A1** — does iter_0 beat `best_iter_4`?
2. **F1** — clean up bare-wrapper sites (cheap, removes a bug class).
3. **B1** + **B2** bundled — make the self-play loop preserve the warm-start.
4. **A4** — change value-head loss structure (theory II).
5. **D1** — self-play-data pretrain (theory III).

### Universal next steps (regardless of D.2 outcome):
- **F1** is cheap and unblocks future cross-arch evals.
- **A1** is cheap and gates interpretation.
- **A3** removes a confound that's currently in every comparison.

---

## Cross-references

- `VOLUME_PRETRAIN_RESULTS.md` — chronological log of branch D sessions.
  Each new SPRT result here gets a session-update entry there.
- `D2_PREP.md` — scoping doc for Branch D.2 (Path B re-encoding details).
- `STEP2_MCTS400_RUNBOOK.md` — Step 2 runbook, source of the "MCTS-400
  marginal" finding.
- `TRAINING_REFACTOR_PLAN.md` — original ceiling-raising roadmap.
- `ARCHITECTURAL_IMPROVEMENTS_PLAN.md` — original architecture exploration
  plan; references SE blocks, deeper trunk, etc.
- `TECH_DEBT.md` — known bugs / instrument-correctness issues.
- `YNGINE_BENCHMARK_RESULTS.md` — first measured win rates of a checkpoint
  vs the external yngine engine. Detail for the 2026-05-28 Done entry.

---

## Done

Completed experiments with their results. Don't delete the original write-ups
above (when they exist) — those are durable knowledge for future calibration.

### Entry format (follow for new entries)

Each Done entry follows this structure. The discipline: front-load the
verdict, then the **data points that change priors for future experiments**
— not just "what happened" but "what should we now believe differently."

```markdown
### <Experiment ID + name> — **<VERDICT>** (<date>)

<One paragraph: what ran, how, total compute. Keep tight.>

**SPRT verdict** (or analogous result block):
- **<DECISION>** (decision context — boundary crossed, cap hit, etc.)
- Score, WR, CI95, LLR, color split — whatever the metric set is
- Duration, $, hardware
- JSON path, run dir

**The crucial detail:** <one sentence — the single most important fact
about this result. Often a CI bound, a comparison to a prior result, or a
specific number that gates downstream interpretation.>

**<Trajectory / comparison table>:** when relevant — Glicko per iter, val
metrics per epoch, SPRT vs other SPRTs, etc.

**Confirmed/pending findings:** check each item from the "Findings driving
this backlog" section above. Use ✅ for confirmed, ❌ for falsified, 🟡 for
pending downstream evidence. Always say WHY in one line.

**Operational lessons logged:** bugs found, infrastructure surprises,
calibration shifts. Each one a 2-3 line bullet. These are the durable
knowledge most likely to be forgotten without a written record.

**Next experiments per sequencing matrix:** which branch of the recommended-
sequencing tree this result puts us on, with the actual queue of next IDs.
```

Why the structure: when picking up cold, the first question is "what's the
current state of the priors?" The verdict + crucial detail + confirmed
findings answer that in ~30 seconds. The operational lessons section
prevents the most common research-debt failure mode: rediscovering the
same bug or pattern session after session.

---

### vs-yngine harness — `iter1_ema` (deployed) sweeps yngine-MCTS-1K — **STRONGER** (2026-05-28)

First measured win rate of *any* model checkpoint against an external
reference (yngine). Closes the V2b bridge gap deferred in
`VOLUME_PRETRAIN_RESULTS.md § 2026-05-21` — we'd been measuring relative
strength (frozen anchor) for two weeks without an absolute number.
Vendored `temhelk/yngine` as a submodule, shipped a stdin/stdout C++
driver + Python bridge + SPRT-capable eval harness, and ran two SPRTs
against the deployed `iter1_ema` model. Full detail in
[`YNGINE_BENCHMARK_RESULTS.md`](YNGINE_BENCHMARK_RESULTS.md); PR
[#20](https://github.com/jackfleming0/YinshML/pull/20).

**SPRT verdicts** (both p0=0.50, p1=0.60, α=β=0.05; upper bound LLR=+2.94):

| Our sims | yngine sims | games | record | WR | CI95 | LLR | verdict |
|---:|---:|---:|---|---:|---:|---:|---|
| 200 | 1,000 | 17 | 17-0-0 | 1.000 | [0.816, 1.000] | +3.10 | **STRONGER** |
| 800 | 1,000 | 17 | 17-0-0 | 1.000 | [0.816, 1.000] | +3.10 | **STRONGER** |

Color balance is clean (W: 9 / B: 8 per run; SPRT alternates colors).
JSON: `logs/iter1_ema_vs_yngine_sims{200,800}_sprt.json`. Per-game wall:
75.8 s mean at MCTS-200 / 288 s at MCTS-800; total ~21 min + ~82 min on
Apple Silicon MPS.

**The crucial detail:** both runs **terminated at the minimum 17 games
SPRT allows** at these params. Even an opponent with WR=0.95 would
require >17 games on average to hit the upper boundary; the model
saturates yngine-1K so completely that the test can't distinguish "much
stronger" from "infinitely stronger." We have a *lower bound* on the
WR (0.816 from Wilson) but no upper-bound-side information at all.

**Confirmed/pending findings:**
- ✅ **The supervised + self-play loop produces a model that beats the
  engine that taught it**, at the corpus-generation sim level. This is
  the expected outcome but had never been measured. Sanity check on
  the entire pipeline.
- ✅ **Frozen-anchor WR has been a faithful proxy for absolute strength
  improvement at this regime.** `iter1_ema` is the iter-1 promotion
  from the B1+B2+B3 RE-RUN #2 (above), which the SPRT vs frozen anchor
  said was indistinguishable from warm-start. Both being decisively
  above yngine-1K reframes "indistinguishable at SPRT bar" as
  "indistinguishable while both far above yngine-1K" — exactly the
  saturation behavior the frozen-anchor design anticipates.
- 🟡 (open) **Where does the model break?** Not measured at higher
  yngine compute. The WAVE3 V2a fingerprint audit described yngine at
  MCTS-10K as "a serious engine, not hobby code"; 10× the level
  benchmarked here. The 17-0 result tells us nothing about MCTS-10K
  behavior.
- 🟡 (open) **Is the self-play loop's contribution to this win rate
  large or small?** A SPRT of `yngine_volume_15ch_pretrain`
  (pretraining-only baseline, no self-play) vs yngine-1K would
  partition the "supervised" vs "self-play" gain. If pretraining alone
  hits ~50%, the entire 50→100 gap visible here is the self-play
  loop's contribution.

**Operational lessons logged:**
- **yngine has two upstream bugs we worked around in the driver, not
  the submodule.** (1) `MCTS::~MCTS` unconditionally joins an
  uninitialized `search_thread`; we run a 1-iter warmup search after
  every `MCTS` construction to make it joinable. (2) yngine's MCTS
  prints `DEBUG:` lines unconditionally to `std::cout`
  (`mcts.cpp:237, 304, 406`); driver redirects stdout to `/dev/null`
  and uses a duped fd for protocol replies. Both detailed in
  `third_party/yngine_driver/yngine_driver.cpp` comments.
- **Memory pressure killed our first MCTS-800 run on game 6.**
  `ArenaAllocator` mmaps the full pool up front; 512 MB × sequential
  games triggered the macOS OOM killer mid-search (empty stderr,
  silent SIGKILL). Dropped driver default to 128 MB — still well above
  the ~50 MB MCTS-10K peak observed in the WAVE3 V2a cloud run. Crash
  recovery in the bridge is sound: yngine_driver dying mid-game logs
  `err: yngine_driver closed stdout unexpectedly`, the eval counts
  the game as a draw, and SPRT excludes draws from LLR — so the lost
  game was a no-op statistically (but ~30 min of wall clock).
- **SPRT was the right call mid-stream.** Original plan was 100
  fixed-n games at each setting (~16h wall total); the first MCTS-200
  run hit 26-0 in ~30 min and was clearly going to keep sweeping. The
  user prompted the switch to SPRT, which terminated at 17 games for
  each setting (~1.7h total wall vs ~16h). Discipline: when point
  estimate is far from the test boundary, fixed-n is wasted compute
  *and* worse statistically (it doesn't reveal the saturation).
  Default future eval runs to SPRT unless there's a reason to want a
  specific fixed n.
- **Apple Silicon support is a one-line build patch** (add `__APPLE__`
  to upstream's `__linux__`-gated `mmap` branch). Idempotent sed via
  `build.sh`; we don't fork the submodule.

**Next experiments per sequencing matrix:**
- **yngine-MCTS-10K SPRT** (top-priority follow-up). Same harness,
  `--yngine-sims 10000`. yngine at 10K is described as "serious" in
  V2a; this is where the gap should narrow or flip. ~3-5× the
  per-move yngine time of the MCTS-1K run, so budget ~5h wall for
  SPRT.
- **`yngine_volume_15ch_pretrain` vs yngine-1K SPRT.** Partition the
  supervised-vs-self-play contribution.
- **Wire yngine WR into the post-promotion gate.** Run `eval_vs_yngine`
  after every promoted iter as a cheap absolute-strength tripwire.
  SPRT at MCTS-200 vs yngine-1K is ~20 min wall — affordable per
  promotion. Flag any regression below WR=0.5 immediately.

---

### Alignment-loop analysis of B1+B2+B3 RE-RUN #2 — alignment matches WR trajectory, validates autopilot decisions — **MEASURED** (2026-05-27)

3-way comparison (`anchor` / `iter1_ema` / `iter4_ema`) on 95 stratified
MAIN_GAME-heavy positions from the post-fix replay buffer, evaluated at
`sims=[0, 400, 1600, 3200]` per model. Tooling: `analysis_board/loop/`
— stratified sampler (`sample_positions.py --stratify move_number`),
per-position MCTS measurement (`measure.py`), N-way per-metric report
(`compare.py`). Total compute ~3h on M-series MPS across the three
measurement passes. Run dir
`analysis_board/loop/runs/20260527_142250_b1b2b3_postfix/`, report at
`compare_report/report.md`. This is a *child* analysis of the
B1+B2+B3 RE-RUN #2 entry below, not an independent experiment.

**Result block (at 3200 sims, N=95 shared positions):**

| Metric | anchor | iter1 | iter4 |
|---|---:|---:|---:|
| Mean rank_of_final_best (lower better) | 7.44 | 7.49 | 7.94 |
| % misaligned (rank≥3, lower better) | 46.3% | 45.3% | 45.3% |
| Mean value_gain_over_raw (closer to zero better) | -0.005 | -0.007 | -0.012 |
| % costly (gain≥0.1, lower better) | 7.4% | **5.3%** | 10.5% |
| % opposite-sign divergence (lower better) | 2.8% | 4.2% | 7.1% |
| Mean best_move_value | +0.275 | +0.267 | +0.246 |

**The crucial detail:** **iter1 has the lowest % costly (5.3% vs anchor's
7.4%)** — a real, signed improvement that SPRT at N=103 missed (47.6%
WR, NOT_STRONGER). iter4 is materially worse than iter1 across every
alignment metric, matching the autopilot's revert decision. Alignment
appears to be a *sensitive* training-quality signal where SPRT-at-small-N
is noise-limited.

**Trajectory (anchor → iter1 → iter4) across the metric set:**
- mean rank: 7.44 → 7.49 → 7.94 (slight regression then larger regression)
- % costly: 7.4% → **5.3%** → 10.5% (real improvement at iter1, then degradation)
- % opposite-sign: 2.8% → 4.2% → 7.1% (monotonic degradation)
- best_move_value: +0.275 → +0.267 → +0.246 (monotonic degradation)

The % costly trajectory is the noisiest but most signed; opposite-sign
and best_move_value are *cleanly monotonic* through iter4 and corroborate
the WR-trajectory degradation.

**Confirmed/pending findings:**
- ✅ **The post-fix iters DID produce measurable behavioral change vs
  anchor**, not just noise. Multiple alignment metrics shift with
  consistent direction. Confirms the training pipeline is doing
  *something*, even when SPRT can't resolve it as a WR change at N=103.
- ✅ **Alignment trajectory matches WR trajectory** (anchor → iter1
  slight improvement → iter4 degradation). First independent
  corroboration of the autopilot's iter-by-iter promote/revert decisions
  from an axis other than win-rate.
- ✅ **Opposite-sign divergence is rare, not systemic.** 2.8–7.1% at 3200
  sims, NOT the 43.8% an earlier corrupted run suggested (see
  operational lessons). The row-10-counter-capture-style failure
  pattern from 2026-05-26 is real but uncommon.
- 🟡 (open) **Does alignment improvement predict SPRT improvement at
  larger N?** Iter1's +2.1pt improvement in % costly maps to NOT_STRONGER
  at N=103. Worth testing whether N=400-800 SPRT can resolve the
  alignment delta into a WR signal. If yes, alignment becomes a useful
  *faster* training-quality detector than SPRT.

**Operational lessons logged:**
- **`measure.py` was silently corrupting per-position MCTS measurements**
  for two days. Root cause: `server.get_mcts()` was updated to use
  `enable_subtree_reuse=True` (needed for the analysis-board's
  step-into-line PV extraction); the Flask `/api/evaluate` path was
  updated with `mcts.reset_tree()` before each search to keep
  evaluations independent, but `measure.py` (which calls `mcts.search()`
  directly) was not. Result: every position after the first in a
  measurement batch inherited the previous position's tree, producing
  the prior position's policy distribution mapped onto a different board.
  Smoking-gun symptom in the corrupted reports: identical
  `rank_of_final_best` across all sim budgets for every position —
  which I missed when generating the morning 158-position stratified
  run, surfacing as the alarming "43.8% opposite-sign rate" headline
  that turned out to be the bug talking. Fix landed in commit `35b3977`
  (one-line `mcts.reset_tree()` addition to `measure_one()`). All
  loop-runs since `a52e3f5` (2026-05-26 afternoon, the step-into-line
  commit) had to be re-run.
- **Discipline takeaway:** when enabling state-persistence on a shared
  instance (`enable_subtree_reuse=True`), audit every caller for paired
  resets. The Flask path was fine because I added the reset there;
  `measure.py` wasn't because I didn't think about the new invariant
  when retrofitting subtree-reuse for an unrelated PV-extraction feature.
- **Stratified sampling** (`sample_positions.py --stratify move_number`)
  made late-MAIN_GAME buckets sturdy enough for analysis. Previous
  uniform sampling had ~3 positions in move 60+ buckets; stratified has
  30+. Stratification should be the default going forward.
- **Per-row best_move_value pills + opposite-sign-divergent flag in the
  analysis-board UI** make the value-head-blind-spot pattern visible at
  a glance without stepping into each line. Right tool to surface
  *before* generating measurements — would have caught the 43.8% number
  as suspicious immediately if it had been visible.

**Next experiments per sequencing matrix:**
- **TODO_baseline.md #28** — fix `trainer.py:1442` search-consistency
  mid-capture contamination before the next training cycle. Bounded
  magnitude (~1.3% loss contamination — `0.1 SC weight × ~13%
  mid-capture buffer fraction`) but real, and gated as "do not land
  mid-run." Affects every config with `search_consistency: enabled:
  true`, which is all current B1B2B3 / wave3 configs.
- **Iter1-as-warm-start training run** — current evidence (1.7pt
  improvement in % costly, opposite-sign +1.4pt regression but absolute
  rate still low at 4.2%) is consistent with iter1 being marginally
  stronger than anchor. Whether that compounds across more iterations
  is the open question. Cheap to test: same config, swap warm-start.
- **(deferred)** Calibrate whether alignment metric can replace SPRT as
  the autopilot's promote/revert criterion in regimes where N is
  compute-limited. Needs 2-3 more training-run analyses across
  different config tweaks to characterize the alignment ↔ WR mapping.

### B1+B2+B3 RE-RUN #2 — phase-fix verified, loop neutral at SPRT bar — **NOT_STRONGER** (2026-05-27)

The actual experiment, on the post-phase-weight-fix code. Five-iteration
MCTS-200 self-play from `best_supervised.pt` warm-start with the same
config that gave us the invalidated run (B1 gate 0.50 + games_per_match
200, B2 lr 1e-5, B3 games_per_iter 200). Total: 15.81h self-play + 98 min
SPRT = **~17.5h on RTX 5090** (second new host after the first attempt
crashed on CUDA at iter 3). Run dir `runs_branchB1B2B3/20260527_001626`.

**SPRT verdict:**
- **NOT_STRONGER** (crossed -2.944 LLR boundary at game 103 of 400-cap)
- Candidate (iter_1_ema) 49-54-0, WR **0.4757**, CI95 **[0.3819, 0.5713]**, LLR **-3.116**
- Color split cand_white=30, cand_black=19 (acceptable at low n)
- Duration: 98 min on RTX 5090
- JSON: `logs/branchB1B2B3_rerun2_iter1_ema_vs_anchor.json`

**The headline finding:** the phase-weight fix **reproducibly produces a
+5 WR jump at iter 2 in the in-loop arena** (51.7% in the first re-run,
52.0% in this one — independent hosts, statistically identical), but
**that jump does NOT translate to a decisively stronger model at the
SPRT bar.** iter_1's true WR vs warm-start under independent
re-evaluation is ~0.476, within sampling noise of 0.50.

**Comparison vs the invalidated run:** these SPRTs are statistically
indistinguishable.

| Run | Candidate | Phase fix? | SPRT WR | CI95 | Verdict |
|---|---|---|---|---|---|
| Invalidated | iter_4_ema (final of 5 reverted) | NO (buggy) | 0.468 | [0.370, 0.568] | NOT_STRONGER |
| RE-RUN #2 | iter_1_ema (only promoted candidate) | YES | 0.476 | [0.382, 0.571] | NOT_STRONGER |

The phase fix changed the loop's **shape** (one real promotion at
iter_1 instead of zero promotions through five iters) but not the
**ceiling** (final best model is ~0.47-0.48 WR vs warm-start in both
cases).

**Within-run iter-by-iter** (Wilson gate WR vs current best at gate
time):

| Run-iter | Candidate | WR vs best | Gate verdict | Best after |
|---|---|---|---|---|
| 1 | iter_0 (warm-start) | n/a | NEW BEST (auto-initial) | iter_0 |
| 2 | iter_1 | 52.0% vs iter_0 | Wilson REJECT, Elo override **PROMOTED** | iter_1 (Elo 1513.3) |
| 3 | iter_2 | 50.5% vs iter_1 | Wilson REJECT, Elo not improved → REVERT | iter_1 |
| 4 | iter_3 | 45.0% vs iter_1, CI [0.402, 0.499] | REVERT (statistically below 0.50) | iter_1 |
| 5 | iter_4 | (reverted, Elo 1498.3 vs 1513.3) | REVERT | iter_1 |

Pattern: one real candidate (iter_1) clears the gate via the Elo override
path; iters 3-5 all revert.

**Confirmed findings** — see the "post-B1B2B3-rerun-#2 investigation
queue" section above for the working priors at run time. Below are
which mechanisms the run's data supports/rejects:

1. ✅ **Phase-weight fix works empirically.** iter 1 buffer phase mix
   was MAIN_GAME 76.6% / RING_PLACEMENT 15.4% / RING_REMOVAL 8.0% —
   matches the intended distribution. The MAIN_GAME=2.0 weight is now
   actually applying to ~77% of samples. The mechanism that was
   silently broken in D.2 / invalidated B1+B2+B3 is now wired correctly.
2. ✅ **The +5 WR-jump signal at iter 2 is REPRODUCIBLE** across
   independent hosts. First re-run hit 51.7%, this re-run hit 52.0%.
   Same config, same warm-start, different host hardware. Not noise.
3. ❌ **The +5 WR jump does NOT survive independent SPRT
   re-evaluation.** iter_1's true WR vs the warm-start is ~0.476.
   The in-loop arena and SPRT are statistically consistent (CIs
   overlap), but the SPRT's 0.60 STRONGER bar is not cleared. We do
   not have evidence that iter_1 is meaningfully stronger than
   `best_supervised.pt`.
4. ❌ **Iters 3-5 all revert** under the fixed phase weighting — same
   broad behavior as the invalidated run, although the WR pattern
   differs (invalidated run had iter_1/2/3 all reverting at ~47%,
   this run has iter_2 promote at 52%, then iters 3-5 revert).
5. 🟡 **Drift mechanism (iters 3-5 declining vs iter_1) is unresolved.**
   We have 3 data points (50.5%, 45.0%, iter 5 not gate-logged) with
   SE ~2.5% — the "spiral" framing was overconfident. Run the buffer
   diagnostics + optimizer-state inspection in the investigation queue
   to actually answer this.

**Operational lessons logged:**

- **Phase-weight fix is empirically necessary** for 15-channel runs to
  produce the training environment the config intends. All 15-ch runs
  prior to this one were operating under under-sampled MAIN_GAME
  positions. We should re-test D.1 v2 (GAP value head, was 1-15-0
  NOT_STRONGER) on the fixed code to see if its conclusion changes —
  the buggy training may have unfairly tarred GAP heads. Logged as
  potential D.1 v3 candidate.
- **First host's CUDA failed mid-run.** The first re-run attempt
  crashed during iter 3's anchor eval with `CUDA_ERROR_UNKNOWN`.
  Container-level reboots didn't fix; full host restart at the vast.ai
  level was required, and even that didn't fully recover. Resolution
  was to destroy + spin up on a different physical host. Logged as
  operational risk: long runs on vast.ai have a non-trivial
  failure-mid-run probability; consider checkpointing more
  aggressively or running shorter iter counts split into resumable
  segments.
- **The Elo-override path is conditional, not blanket bypass.** Iter
  3-5 all had Wilson REJECT AND Elo not improved → all REVERTED
  correctly. The override only fires when Wilson rejects AND Elo
  improves. This is closer to "intentional legacy behavior" than I'd
  characterized — the rest of the gate works as designed. The
  proposed fix (require `not perform_wilson_check` for the Elo path)
  is still right, but it's a tightening, not a bug-fix-for-broken.

**Artifacts preserved locally:**
- SPRT JSON: `logs/branchB1B2B3_rerun2_iter1_ema_vs_anchor.json`
- Promoted iter_1 checkpoint:
  `models/yngine_volume_15ch_pretrain_b1b2b3_rerun2_iter1/iter1_ema.pt`
- Replay buffer (~100K samples, correct phase mix):
  `experiments/branchB1B2B3_rerun2_2026-05-27/full_run_dir/20260527_001626/replay_buffer.pkl.gz`
- All run metadata (configs, metrics, manifest, tournament history)
  in same experiments/ subdir.

**Next experiments — REPRIORITIZED 2026-05-27 post-SPRT:**

The "is the loop additive" question is now resolved at the SPRT level:
**no, not with these knobs alone.** Path forward shifts from
"tuning the loop" to "structural change":

1. **A4 — regression value head.** Highest mechanism prior remaining.
   The +5 in-loop WR jump under correct phase weighting shows the
   loop CAN produce a real (if marginal) improvement; the value-head
   discretization is the remaining structural candidate for why the
   improvement doesn't scale to a decisive gain. Code already prepped
   (commits `343aab6` + `2070650`). Launch-ready for next session.
2. **D1-partial — pretrain on the saved buffer.** Now we have TWO
   correctly-trained buffers (this run + the first re-run) totaling
   ~100K MCTS-target samples from the warm-start teacher with correct
   phase distribution. Run `scripts/convert_replay_buffer_to_mmap.py`
   and feed to `run_supervised_pretraining.py`. Cheap probe (~3-5h,
   no new generation cost). Should answer: does pretrain on the
   MCTS-target signal beat pretrain on yngine outcomes?
3. **A4 + D1-partial combined** — pretrain with regression value head
   AND the buffer's MCTS targets simultaneously. Lowest cost, highest
   mechanism prior. This is the actual right next experiment.
4. **Post-run investigation queue** — see "what would change my mind"
   section above. Three items doable from local data:
   - Inspect optimizer state code path in `supervisor.py` revert
   - Buffer mode-collapse diagnostics (sparsity, value distribution,
     game-length shift between iter 1 and iter 5 buffers)
   - Document any findings as addendum to this entry

5. **D.1 v3 candidate (low priority)** — re-run D.1 v2 (GAP value
   head, 15-channel) on the post-phase-fix code. The original D.1 v2
   NOT_STRONGER (1-15-0) was measured under buggy phase weighting,
   so the conclusion that "GAP doesn't work" may have been confounded.
   Worth a 12h re-run someday but not urgent.

**Deeper meta-finding:** B1+B2+B3 is now a **completed experimental
branch.** The bundle of three knobs reproduces a small (+5 WR) in-loop
signal under correct phase weighting, but cannot produce an SPRT-level
improvement. The structural ceiling of MCTS-200 self-play with this
warm-start, configured this way, is "roughly the same strength as
warm-start." Any further progress requires changing either the value-
head loss structure (A4) or the corpus / target distribution (D1) —
not the gate, LR, or game count.

---

### B1+B2+B3 — stop-the-leak bundled run — **INVALIDATED** (training operated under the phase-weight bug; re-run pending) (2026-05-26)

> 🚨 **INVALIDATED 2026-05-26 (~14:30 UTC).** During post-run cleanup we
> discovered a long-standing bug in `trainer.py`'s `decode_phase` helper:
> it read `state[5]` unconditionally to label each sample's game phase,
> which is correct for the 6-channel basic encoder but wrong for the
> 15-channel enhanced encoder (where CH_GAME_PHASE=12 and channel 5
> carries sparse row-threat data). Every 15-channel sample was labelled
> `RING_PLACEMENT`, silently disabling the configured
> `phase_weights: MAIN_GAME=2.0` boost in `sample_batch`. **MAIN_GAME
> positions were under-sampled by 2× throughout B1+B2+B3 training.**
>
> This means the network B1+B2+B3 trained is NOT the network the
> configuration was supposed to produce. Every "how the loop behaved"
> conclusion derived from this run is a measurement of a different
> system than the one we thought we were testing.
>
> **What stays valid** (independent of training quality):
> - The phase-weight bug discovery itself (see Operational lessons +
>   TECH_DEBT.md).
> - The gate fail-closed defense (logic, not data).
> - The A4 phase 1.5 auto-detect wiring.
> - The buffer converter + filter logic.
> - The named channel constants on both encoders.
>
> **What's invalidated:**
> - "B1/B2/B3 confirmed working" — gate operated on H2H games between
>   buggy-trained networks; we can't infer it would behave the same on
>   correctly-trained networks.
> - The Glicko-drop comparison vs D.2 (-23 vs -107) — both runs had the
>   same bug, so the ratio compares buggy to buggy.
> - The SPRT verdict's *interpretation* — the WR 0.468 result is a real
>   measurement, but of a model trained with broken phase sampling. We
>   don't actually know what iter_4 would have looked like under correct
>   training.
> - The decision-matrix outcome (cell mapping → next experiment).
>
> **Same exposure: D.2.** D.2 was also a 15-channel run trained under
> the buggy `decode_phase`. Its conclusions about the self-play loop
> destroying value should be treated with the same skepticism — the
> "loop destroyed value" framing might have over-called what was really
> "loop trained with wrong phase weights and hit a different fixed
> point." A1's STRONGER verdict (pretrain vs old anchor) is unaffected
> because pretraining doesn't use the trainer's phase weighting.
>
> **Re-run plan:** see `## Recommended sequencing` below — B1+B2+B3 must
> be re-run on the fixed code BEFORE any other experiment, because the
> "loop neutralized vs additive" question is now genuinely open. The
> two prior 15-ch results don't answer it. Re-launch is one command on
> the cloud box once the fix is pulled.

---

Five-iteration MCTS-200 self-play from `best_supervised.pt` warm-start
with three knobs bundled: B1 (gate 0.50 + games_per_match 200), B2
(lr 1e-5), B3 (games_per_iter 200). Total: 12.39h self-play + 1.08h
SPRT = **~13.5h on RTX 5090**. Run dir `runs_branchB1B2B3/20260525_233508`.
**Trained under the phase-weight bug — see invalidation banner above.**

**SPRT verdict:**
- **NOT_STRONGER** (crossed -2.94 boundary at game 94 of 400-cap)
- Candidate 44-50-0, WR **0.468**, CI95 **[0.370, 0.568]**, LLR -3.135
- Color split cand_white=24, cand_black=20 (acceptable at low n)
- Duration: 65 min, RTX 5090
- JSON: `logs/branchB1B2B3_iter4_ema_vs_anchor.json`

**The crucial detail:** SPRT WR (0.468) is **statistically identical
to iter_1's and iter_2's in-loop arena WRs** (0.465, 0.468 — the
reverted candidates). The iter_4 in-loop arena bump to 0.516 — which
the gate promoted — was sampling noise. Under independent re-sampling,
iter_4's true WR vs iter_0 is ~0.47. **The loop did not produce a
model that meaningfully beats the warm-start.**

**Headline comparison vs D.2** (after refreezing the anchor to
`best_supervised.pt`):

| Run | Loop | Anchor | Verdict | WR | CI95 |
|---|---|---|---|---|---|
| D.2 | iter_4_ema, lr 1e-4 | OLD `best_iter_4` (weaker) | NOT_STRONGER | 0.526 | [0.470, 0.582] |
| B1+B2+B3 | iter_4_ema, lr 1e-5 | NEW `best_supervised` (strong) | NOT_STRONGER | 0.468 | [0.370, 0.568] |

Both translate to "≈ same strength as starting checkpoint." But D.2
was vs the weaker anchor and only barely held; B1B2B3 was vs the new
strong anchor and held at neutral. The much harder bar makes B1B2B3
the meaningfully more rigorous test — **and the loop is no longer
net-destructive.**

**Within-run iter-by-iter** (gate-relevant WR vs iter_0):

| Cand | WR | Elo | Wilson Gate | Decision |
|---|---|---|---|---|
| iter 1 | 46.5% | 1476.7 | wins=186/400, CI95 [0.417, 0.514] | REVERT |
| iter 2 | 46.8% | 1496.0 | wins=187/400, CI95 [0.419, 0.516] | REVERT |
| iter 3 | 48.4% | 1492.7 | (similar CI band) | REVERT |
| iter 4 | 51.6% | 1507.3 | (point estimate > 0.50) | **PROMOTED** (false positive per SPRT) |

iter 1 Glicko drop vs iter_0 was **-23** here (1500 → 1476.7) vs D.2's
**-107**. That's the **B2 (lr 1e-5)** effect: ~80 Glicko of value
preservation. The within-loop arena trend (46.5→46.8→48.4→51.6) is
weakly suggestive that buffer accumulation surfaces signal — but SPRT
rules out a *decisive* gain at 5 iters.

**Confirmed findings:** ⚠️ **INVALIDATED.** Each item below
described a behavior of the self-play loop derived from this run's
training. With the phase-weight bug now known (MAIN_GAME positions
under-sampled by 2× throughout), every conclusion about *how the
loop behaved* is measuring a different system than the one the config
was supposed to test. The full original list (B1/B2/B3 individually
"confirmed," the loop's "preservation but not addition" framing,
the "replay-buffer accumulation may help" caveat) has been struck —
those questions are genuinely open and must be re-answered on the
fixed code.

What remains in the SPRT data: the candidate trained under buggy
phase weights landed at WR 0.468 vs `best_supervised`. That's an
honest measurement of *the model that was actually produced*, not
of *the model the config aimed to produce*. The two are not the
same network.

**Operational lessons logged:**

- **🚨 PHASE-WEIGHT BUG (discovered 2026-05-26 during D1-partial prep).**
  `yinsh_ml/training/trainer.py`'s local `decode_phase` helper read
  `state[5]` unconditionally — correct for the 6-channel basic encoder
  (CH_GAME_PHASE=5) but silently wrong for the 15-channel enhanced
  encoder where channel 5 is a sparse row-threat channel and
  CH_GAME_PHASE=12. As a result, **every position in every 15-channel
  run's replay buffer was labelled RING_PLACEMENT.** The mislabelling
  flowed through `ReplayBuffer.sample_batch`'s phase-aware weighting
  (`trainer.py:446-447`), where the configured
  `phase_weights: MAIN_GAME=2.0` weight was silently disabled. Every
  sample got the RING_PLACEMENT weight (1.0) regardless of actual
  phase.

  **Real phase mix (verified post-fix, on the B1+B2+B3 buffer):**
  `MAIN_GAME=75.6%, RING_PLACEMENT=16.0%, RING_REMOVAL=8.4%`.

  **Affected runs:** D.2, B1+B2+B3 (both 15-ch). MAIN_GAME positions
  were under-sampled by 2× across both runs.

  **Hypothesis worth testing in the next experiment:** part of the
  reason the self-play loop couldn't become additive may be that
  the most-important phase (MAIN_GAME — where the actual game tactics
  happen) was under-sampled by 2× throughout training. A re-run with
  the fix applied could legitimately surface improvement that the
  buggy weighting was suppressing. This adds a **fourth knob option
  (correct sampling)** to the B-experiment family that wasn't on
  any previous list.

  **Fix (landed in this work):** added `CH_GAME_PHASE` named constant
  to `StateEncoder` (basic), plus `phase_channel_index(num_channels)`
  and `decode_phase_from_state(state)` utility in `encoding.py` as
  single sources of truth. Trainer now imports + delegates to that
  utility. Regression tests in `yinsh_ml/tests/test_decode_phase_cross_encoder.py`
  pin the contract for both encoders + reject unknown channel counts.
  See also `TECH_DEBT.md` entry.

- **Buffer is 63% zero-policy rows.** Of 100K samples, ~63% have
  all-zero `move_probs` — likely a mix of terminal-position dummies
  and early-game positions where MCTS visit distributions weren't
  captured (the supervisor's self-play appends a dummy zero policy
  at game end, plus some RING_PLACEMENT moves may bypass MCTS). The
  D1-partial corpus is effectively ~36K samples, not 100K. The
  converter at `scripts/convert_replay_buffer_to_mmap.py` filters
  these out before saving.

- **`--export-every 0` CLI override is broken.** CoreML export ran at
  iter 5 despite the flag because `run_training.py:254` does
  `export_every = args.export_every or int(...)` — Python's "0 is
  falsy" bug. Fix: `args.export_every if args.export_every is not None else ...`.
  CoreML export crashed (known 6-vs-15 ch bug) but was wrapped in
  try/except so non-fatal. Cosmetic, low priority.
- **Gate uses point-estimate, not Wilson lower bound.** iter_4 promoted
  at 51.6% (Wilson lower ~0.467 < 0.50) but SPRT re-sample shows true
  WR 0.468. Wilson-lower-vs-threshold would have correctly rejected.
  **Recommended gate-rule refinement before A4/D1:** require Wilson
  lower bound ≥ threshold (not point estimate). Trivial code change in
  TrainingSupervisor's gate path. Worth doing — otherwise every future
  run risks a "false-positive promotion → next iter trains from a
  noisy iter_N → potentially propagates the noise" failure mode.
- **Replay buffer preserved** (~50K MCTS-target samples, 12 MB gzip):
  `experiments/branchB1B2B3_run_2026-05-26/.../replay_buffer.pkl.gz`.
  Generated by the strong warm-start over 5 self-play iters with
  proper batched MCTS + subtree reuse. Lets D1 test the "MCTS-target
  pretrain beats yngine-outcome pretrain" hypothesis on a partial
  corpus without paying the 10-15h generation cost up front.
- **Promoted iter_4 checkpoint preserved** at
  `models/yngine_volume_15ch_pretrain_b1b2b3_iter4/iter4{,_ema}.pt`.
  Even though SPRT rules out decisive gain, it's the strongest
  *output* of any self-play loop we've run — keep for forensic
  comparisons / future warm-start ablations.

**Next experiments — REPRIORITIZED 2026-05-26 post-invalidation:**

1. **B1+B2+B3 RE-RUN — highest priority, must come before any other
   15-channel experiment.** Same config (`configs/branchB1B2B3_mcts200.yaml`),
   same warm-start (`best_supervised.pt`). The fix lives in
   `trainer.py` automatically — pulling the latest `training-pipeline-fixes`
   branch is sufficient. The re-run answers the actual question this
   experiment was designed to answer: does the bundle of B1/B2/B3
   knobs salvage the self-play loop on a strong warm-start? We don't
   know the answer; the prior run measured a different system. Cost:
   ~13.5h on RTX 5090 (~13.5h self-play + SPRT). See "B1+B2+B3 re-launch
   plan" subsection below for the exact command.

2. **A4 (regression value head)** — code prepped (commits `343aab6`
   + `2070650` for the auto-detect). Wait for the B1+B2+B3 re-run to
   complete first — if the re-run is value-additive, A4 stacks; if
   still neutral, A4 is the next structural change to test.

3. **D1-partial** — pretrain on the saved buffer (`replay_buffer.pkl.gz`,
   ~36K usable samples after filtering). The buffer ITSELF is unaffected
   by the phase-weight bug (its contents are MCTS self-play targets
   produced by the teacher network, not gradient-update outputs). The
   training that *consumed* the buffer was buggy, but the buffer's
   data is what it is. Still a useful D1 corpus. Wait for B1+B2+B3
   re-run for sequencing.

4. **B1+B2+B3 re-run + phase-weight fix is now a single test.** The
   "fourth knob" framing from the prior write-up (correct phase
   sampling) is no longer a separate experiment — it's table stakes
   for any 15-channel run going forward.

**B1+B2+B3 re-launch plan (cloud box, fresh clone of fixed code):**

```bash
ssh -p 32740 root@85.10.218.46
# Clean slate: drop the old run dirs to avoid resume confusion
cd /root/YinshML
rm -rf runs_branchB1B2B3
# Pull latest fix
git fetch origin && git reset --hard origin/training-pipeline-fixes
# Launch — same config, same warm-start, --export-every 0 to suppress
# the CoreML crash at the end (cosmetic).
. venv/bin/activate
RUN_LOG=logs/branchB1B2B3_rerun_$(date +%Y%m%d_%H%M%S).log
echo $RUN_LOG > /root/YinshML/.current_run_log
tmux new-session -d -s b1b2b3_rerun "cd /root/YinshML && . venv/bin/activate && python -u scripts/run_training.py --config configs/branchB1B2B3_mcts200.yaml --init-checkpoint models/yngine_volume_15ch_pretrain/best_supervised.pt --export-every 0 2>&1 | tee $RUN_LOG; echo === DONE === ; sleep 600"
```

Sanity to verify before walking away (one-liner): `grep -E "Phase mix|MAIN_GAME=" /root/YinshML/$(cat /root/YinshML/.current_run_log)` should show MAIN_GAME ~70-80% **after the first iter's data is in the buffer** — confirms the fix is active. If the buffer is still 100% RING_PLACEMENT, the fix didn't land.

**Deeper meta-finding:** ⚠️ **INVALIDATED.** The original framing
here claimed B1+B2+B3 had moved the loop from "actively destructive"
to "neutral but not additive," and that the "additive mechanism" was
the structural gap to close (favoring A4/D1). All three pieces of
that framing rested on the contaminated WR/Glicko numbers. We don't
actually know whether the loop was destructive, neutral, or additive
under the configured knobs — we only know what happened under buggy
phase weighting. The re-run on the fixed code is the next data point
that can speak to this question.

---

### A1 — D.2 pretrain (iter_0) vs best_iter_4 — **STRONGER** (2026-05-25)

Direct SPRT of `models/yngine_volume_15ch_pretrain/best_supervised.pt` (the
D.2 pretrained warm-start, never touched by self-play) against the frozen
6-ch champion `best_iter_4`. Fired immediately after D.2 SPRT landed
NOT_STRONGER, to test whether the self-play loop had diluted away a
genuinely-stronger model. Decisive in 21 games, 15 minutes of compute.

**SPRT verdict:**
- **STRONGER** (crossed +2.94 boundary at game 21 of 400-cap)
- Candidate 19-2-0, WR **0.905**, **CI95 [0.711, 0.973]**, LLR +3.02
- Color split 11/8 (acceptable at low n; would balance at higher n)
- Duration: 15 minutes 3 sec on RTX 5090
- JSON: `logs/d2_pretrain_iter0_vs_frozen.json`

**The crucial detail:** CI95 lower bound **0.711** — far above the 0.60
STRONGER bar. This isn't "marginally stronger" — it's *decisively* stronger
than the previous champion. The pretrained 15-ch init alone is the
strongest checkpoint the project has produced.

**Comparison table:** Two SPRTs run minutes apart, same anchor, same
protocol, same candidate *family* (D.2 pipeline). Different stages of the
pipeline.

| Candidate | Stage | Verdict | Games | WR | CI95 |
|---|---|---|---|---|---|
| `iteration_4_ema` | pretrain + 5 self-play iters | **NOT_STRONGER** | 304 | 0.526 | [0.470, 0.582] |
| `best_supervised` | **pretrain only, no self-play** | **STRONGER** | 21 | **0.905** | **[0.711, 0.973]** |

The self-play loop destroyed approximately **250-300 Elo** of value between
these two candidates.

**Confirmed/pending findings** (against the backlog's "Findings driving"
section above):

1. ✅ **"Pretrain is where the strength lives now"** — confirmed
   decisively. iter_0 alone beats the prior champion 19-2. The "strongest
   D.2 model = the pretrain" hypothesis was correct.
2. ✅ **"15-ch encoding moved pretrain metrics only marginally" was the
   wrong frame.** The marginal metric movement (+1.4 PAcc, +0.7 VAcc)
   understated the strength gain at the SPRT level — the network was
   evidently learning something the val-metric argmax couldn't capture.
   This is the "theory IV (argmax-VAcc is the wrong metric)" hypothesis
   from A4's write-up *getting indirect support* — val P-loss kept
   dropping every epoch, and that movement turned out to be load-bearing
   even though VAcc plateaued.
3. ✅ **"Self-play at MCTS-200 has no headroom; is a random walk on this
   warm-start."** Confirmed in the strongest possible form: not a random
   walk, an *actively destructive* drift. iter_0's 0.905 WR collapsed to
   iter_4's 0.526 WR over 5 iters of the current loop. Net-negative, not
   net-zero.
4. ✅ **"Wilson 0.20 is too loose for this regime."** Every iter promoted
   at 42-49% WR; the cumulative effect was 250-300 Elo of value
   destruction. The gate is the *mechanism* by which the dilution
   propagated forward.
5. ❌ **Search depth isn't the dominant lever** was the read after Step 2.
   That's still true at MCTS-400 vs MCTS-200. But now we know the
   *encoding* axis WAS a real lever — it just got masked by the self-play
   loop's mistuning. Step 2's "ceiling is structural" framing partially
   wrong: the ceiling at MCTS-200 wasn't network capacity, it was the
   self-play loop discarding pretrain gains.

**Operational lessons logged:** None new for A1 itself (15-min run, no
issues). But it reinforces the *D.2-level* lesson that the autopilot's
SUMMARY.md writer should print SPRT details directly to the consolidated
log so we don't need to manually pull the JSON each time. (See D.2 entry
below for the writer bug.)

**Next experiments per sequencing matrix** — A1 STRONGER materially
re-ranks the priorities:

1. **Re-freeze the anchor to `best_supervised.pt`** *before any further
   SPRT*. Otherwise every future comparison is against a stale 6-ch
   reference. This is operationally cheap (just point new SPRT calls at
   the new anchor) but matters immediately.
2. **B1 + B2 + B3 jump to the top.** The self-play loop is *actively
   destroying value*, not just failing to add it. This is no longer a
   tuning task — it's a "stop the leak" task. B1 (tighten Wilson),
   B2 (lower LR), B3 (more games/iter) are direct fixes for the dilution
   mechanism. Until the loop is either fixed or disabled, every iter on a
   strong init is net-negative.
3. **F1 (bare-NetworkWrapper cleanup) remains queued** — cheap, removes a
   class of bugs.
4. **A4 (regression value head) becomes more interesting**, not less.
   With the value-head plateau read getting indirect support from A1's
   "metric understated strength" finding, theory II (the 3-class
   discretization is leaving signal on the table) has stronger grounds.
5. **D1 (self-play data corpus pretrain)** moves up too — if the
   self-play loop is broken at this strength, generating a corpus from
   our strongest model and re-pretraining might *replace* self-play
   entirely as the iteration mechanism. Becomes the natural "instead of
   fixing the loop, replace it" path.
6. **C1 (SE blocks) drops further.** Architecture exploration is less
   urgent now that we have a clearly-stronger pretrain to anchor against
   and a known-broken loop to fix first.

The deeper meta-finding: **the self-play loop is built for a model in the
"learning" regime, not the "fine-tuning" regime.** At weak warm-start
(Branch C era), it holds value. At strong warm-start (D.2 era), it
destroys value. The transition was not gradual — it was a binary regime
shift. This is a known failure mode in RL with strong priors (LLM RLHF
uses tiny LRs + KL penalties precisely to anchor toward the supervised
init).

---

### Branch D.2 — 15-channel enhanced encoding — **NOT_STRONGER** (2026-05-25)

> ⚠️ **Caveat added 2026-05-26.** D.2 ran under the same phase-weight
> bug as B1+B2+B3 (15-channel encoder + `decode_phase` reading
> `state[5]`, which is a row-threat channel in the 15-ch layout —
> NOT the phase channel). MAIN_GAME positions were under-sampled by 2×
> throughout D.2 training. The SPRT measurement (WR 0.526 vs old anchor)
> stands as a real comparison of the trained models, but the *interpretation*
> ("self-play loop destroyed pretrain gains") may have over-called what
> was really "loop trained under wrong phase weights settled at a different
> fixed point." A1's STRONGER verdict for the pretrain checkpoint is
> unaffected (pretraining doesn't use the trainer's phase weighting).
> See B1+B2+B3 invalidation banner above for the full discussion.


The run that generated this backlog. Full pipeline: Path B 6-ch→15-ch corpus
re-encoding (~16 min, 13.6M positions fp16 mmap) → 6-epoch supervised
pretrain (~5h, val PAcc 0.300 / VAcc 0.636) → 5-iter MCTS-200 self-play loop
(~9h) → SPRT vs frozen `best_iter_4` (~4h, 304 games).

**SPRT verdict:**
- **NOT_STRONGER** (crossed -2.94 boundary at game 304 of 400-cap)
- Candidate 160-144-0, WR 0.526, **CI95 [0.470, 0.582]**, LLR -2.96
- Color split 80/80 — no deterministic-side artifact
- Duration: 4.00h on RTX 5090
- JSON: `logs/branchD2_iter4_vs_frozen.json`
- Run dir: `runs_branchD2/20260525_041120/`

**The crucial detail:** CI95 lower bound 0.470 is *below 0.50* — we can't
even confidently say "candidate is non-worse." This is **worse than Step 2
(MCTS-400)**, which landed at CI95 [0.504, 0.600] — real positive edge just
below the 0.60 promotion bar. D.2 doesn't even clear that fainter "real
edge" signal.

**Self-play iter Glicko trajectory** (relative to iter_0 warm-start ≈ 1553):

| Iter | Glicko Elo | Δ vs iter_0 | Wilson gate result |
|---|---|---|---|
| 0 | ~1553 (initial 1500 ref) | — | PROMOTED (first) |
| 1 | 1446.7 | **-107** | PROMOTED (loose 0.20, 42% WR) |
| 2 | 1478.5 | -75 | PROMOTED (49.5% WR) |
| 3 | 1467.3 | -86 | PROMOTED (43% WR) |
| 4 | **1463.9** | **-90** | PROMOTED (42% WR) |

iter_0 (the pretrained init) appears to be the strongest D.2 model. The
self-play loop's noise was dominated by the loose Wilson gate accepting
clearly-worse candidates as "best" — the dilution that's now confirmed by
the SPRT verdict.

**Confirmed findings (from the "Findings driving this backlog" section):**

1. ✅ Self-play dilutes the warm-start at this MCTS budget. iter_4 is ~90
   Glicko below iter_0; the SPRT verdict reflects that gap.
2. ✅ Wilson 0.20 is too loose when the warm-start is strong. Every iter
   promoted at 42-49% WR — clearly worse than its predecessor.
3. 🟡 *Pending A1 result:* whether iter_0 itself beats `best_iter_4`. If yes,
   the encoding lever WAS real and the self-play loop wasted it.
4. ✅ Search depth wasn't the only weak lever (Step 2). Encoding alone isn't
   either (D.2).

**Operational lessons logged:**

- **Encoder-flag propagation bug** (commit `4e984ef`): `ModelTournament`
  + `eval_vs_frozen_anchor.py` constructed bare `NetworkWrapper(device=...)`
  instances, missing the `use_enhanced_encoding` flag → hard-fail when
  loading 15-ch checkpoints. Caught at D.2 iter 1; cost ~1h of compute to
  detect + fix + restart. F1 (audit + fix all bare-construction sites)
  remains queued.
- **fp16 corpus storage**: D2_PREP.md was wrong by 10× on disk math (claimed
  9.9 GB for states.npy, actual was 99 GB float32). Mid-run patched the
  regenerator to fp16 (49.4 GB) — loader does `.float()` cast on load, so
  storage dtype is transparent to training. No precision impact observed.
- **6-epoch pretrain decision** (vs the 3 epochs D2_PREP specified): driven
  by the observation that PAcc kept climbing at ep 1 and was still climbing
  at ep 5. The extra 3 epochs added +0.029 PAcc (0.271 → 0.300) and +0.008
  VAcc (0.628 → 0.636). The longer cosine schedule also gave a higher
  effective LR throughout — different experiment from "3 epochs with
  T_max=3 schedule."
- **CoreML export failed at iter 4** ("C_in / groups = 6/1 != weight[1]
  (15)") — same root cause as the tournament bug; CoreML export has a
  hardcoded 6-ch assumption. Non-blocking; logged for cleanup.
- **Autopilot SUMMARY.md writer bug**: expected flat SPRT JSON fields;
  actual schema nests under `results[0].sprt`. SUMMARY.md printed
  "UNKNOWN / 0-0-0 / None". Trivial fix; logged.
- **`--resume` support** for pretrain shipped (commit `d5e5151`) but not
  exercised yet — future extension runs can resume cleanly via
  `last_resume_state.pt`.

**Next experiments per sequencing matrix** ("INCONCLUSIVE WR ~0.50 or WEAKER"
case):

1. **A1** — RUNNING as of 2026-05-25 17:14 UTC (iter_0 direct SPRT). Result
   pending.
2. **F1** — bare-wrapper cleanup. Cheap, unblocks future cross-arch evals.
3. **B1 + B2 bundled** — tighter Wilson gate + lower self-play LR. Tests
   whether the loop can be tuned to preserve the warm-start.
4. **A4** — regression value head (theory II).
5. **D1** — self-play data corpus pretrain (theory III).

---
