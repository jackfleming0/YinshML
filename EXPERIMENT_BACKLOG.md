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

## Status snapshot (as of 2026-05-25)

- **Active run:** B1+B2+B3 — stop-the-leak self-play (tightened gate 0.50,
  lr 1e-5, games_per_iter 200, games_per_match 200). Launched 23:35 UTC on
  RTX 5090 vast.ai box. Run dir `runs_branchB1B2B3/20260525_233508`. Budget
  17h from launch, ends ~16:35 UTC 2026-05-26. See "B1+B2+B3 — in progress"
  write-up below for decision tree mapping outcomes to next experiment.
- **Current frozen anchor:** `models/yngine_volume_15ch_pretrain/best_supervised.pt`
  (the D.2 15-ch pretrained warm-start; re-frozen 2026-05-25 after A1 SPRT
  showed it STRONGER than the prior `best_iter_4` anchor at WR 0.905, CI95
  [0.711, 0.973]). Prior anchor: `models/branchC_volume_pretrain/best_iter_4.pt`
  (Branch C, 6-ch).
- **Last decisive SPRT verdicts:**
  - 2026-05-24: D.1 v2 (GAP value head) — **NOT_STRONGER** (1-15-0, structural
    determinism verified). GAP + warm-started spatial trunk doesn't work.
  - 2026-05-23: Step 2 (MCTS-400 self-play) — **INCONCLUSIVE** at the 400-game
    cap. Small real edge (WR 0.552, CI95 [0.504, 0.600]). Search depth alone is
    not the dominant lever.

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
