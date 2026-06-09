# B1+B2+B3 — stop-the-leak bundled run

**Status:** DONE: bundle INVALIDATED (phase-weight bug) → RE-RUN #2 NOT_STRONGER (at SPRT bar) → alignment-loop analysis MEASURED. **B1+B2+B3 is a closed experimental branch.**
**Date(s):** original bundled run 2026-05-26 (launched 2026-05-25 23:35 UTC); invalidated 2026-05-26 ~14:30 UTC; RE-RUN #2 2026-05-27; alignment-loop analysis 2026-05-27.
**Cost / hardware:**
- Original: 12.39h self-play + 1.08h SPRT = ~13.5h on RTX 5090.
- RE-RUN #2: 15.81h self-play + 98 min SPRT = ~17.5h on RTX 5090 (second host after first crashed on CUDA at iter 3).
- Alignment-loop analysis: ~3h on M-series MPS.
**Branch / run dir / artifacts:**
- Config: `configs/branchB1B2B3_mcts200.yaml`
- Original run dir: `runs_branchB1B2B3/20260525_233508`; experiments dir `experiments/branchB1B2B3_run_2026-05-26/full_run_dir/20260525_233508/`
- RE-RUN #2 run dir: `runs_branchB1B2B3/20260527_001626`; experiments dir `experiments/branchB1B2B3_rerun2_2026-05-27/full_run_dir/20260527_001626/`
- RE-RUN #2 SPRT JSON: `logs/branchB1B2B3_rerun2_iter1_ema_vs_anchor.json`
- Original SPRT JSON: `logs/branchB1B2B3_iter4_ema_vs_anchor.json`
- Promoted iter_1: `models/yngine_volume_15ch_pretrain_b1b2b3_rerun2_iter1/iter1_ema.pt`; final reverted iter_4: `models/yngine_volume_15ch_pretrain_b1b2b3_rerun2_iter4/iter4_ema.pt`
- Original promoted iter_4: `models/yngine_volume_15ch_pretrain_b1b2b3_iter4/iter4{,_ema}.pt`
- Alignment-loop run dir: `analysis_board/loop/runs/20260527_142250_b1b2b3_postfix/`, report `compare_report/report.md`
- Post-run diagnostics: `experiments/branchB1B2B3_rerun2_2026-05-27/POST_RUN_DIAGNOSTICS.md`

## Description

Bundled "stop-the-leak" run. After A1 showed the warm-start (D.2 pretrain iter_0)
was decisively strong and the self-play loop was *actively destroying* value, this
experiment threw three plausibly-helpful knobs at the loop in a single config,
warm-started from `best_supervised.pt` (the refrozen anchor), 5-iteration MCTS-200
self-play. Question: does the bundle salvage the loop in the fine-tuning regime?

**The three knobs (was → now):**

| Knob | Value (was → now) |
|---|---|
| B1 `arena.promotion_threshold` | 0.20 → 0.50 |
| B1 `arena.games_per_match` | 100 → 200 |
| B2 `trainer.lr` | 1e-4 → 1e-5 |
| B3 `self_play.games_per_iteration` | 100 → 200 |
| (unchanged) encoding | enhanced (15-ch) |
| (unchanged) num_iterations | 5 |

**B1 — Tighten Wilson gate to 0.50.** The gate at 0.20 promotes any candidate
winning ≥20% of H2H games — effectively just filters total catastrophes. With a
strong warm-start, every later iter was visibly worse but still passed at 42–49%
WR. A 0.50 gate ("non-worse"; 0.55 would be "small positive") would have rejected
iter_1 (42% WR) and reverted to iter_0. Games_per_match 100→200 tightens the CI
around 0.50. Risk: a tighter gate might stall the loop if iter_1 is genuinely
near-equal (Wilson 0.50 with limited games is a coin-flip on equal candidates;
~400 games per H2H wanted for a tight CI, roughly doubling tournament cost).

**B2 — Lower self-play LR by 5–10×.** The D.2 pretrain LR schedule ended at 1e-5
(cosine decay to eta_min). Self-play then took over at lr=1e-4 — a **10× jump** at
the transition from a well-converged supervised state to MCTS-target updates,
perturbing converged weights more than necessary. Lowering to 1e-5 matches the
pretrain end-LR and lets the loop *fine-tune* rather than retrain. Evidence: D.2
iter_1 dropped 107 Glicko from iter_0 after one epoch — the symptom of "LR too high
for the starting state." Risk: lower LR also slows learning of genuinely-new
self-play signal (MCTS visit distributions vs argmax policy; rolled-out values vs
raw outcomes).

**B3 — 200–400 games/iter (vs 100).** Adds signal *volume* per iteration. May not
add signal *quality* — STRONGER requires the volume to be load-bearing, a weaker
prior than B1/B2. (Scored 14 in the backlog rubric vs B1's 20.) Set to 200 here.

**Attribution caveat (logged before result):** because three knobs move together, a
STRONGER verdict cannot attribute the gain to a specific knob. Bundled deliberately
because (a) priors on each knob being load-bearing are similar (all target the same
dilution mechanism) and (b) attribution ablations would each cost 4–6h. **Policy
decided 2026-05-25:** if STRONGER, skip attribution, proceed to A4 to compound; if
NOT_STRONGER, use the decision matrix to pick the next experiment.

**Decision matrix — outcome → next experiment** (three evidence axes: SPRT verdict
vs `best_supervised.pt`, gate behavior across 5 iters, Glicko trajectory shape):

|  | All 5 promote | Mixed promote/revert | 0 promote (stuck at iter_0) |
|---|---|---|---|
| **STRONGER** (CI95 lower > 0.50) | Loop works → **A4** | → **A4** | (contradictory) |
| **NOT_STRONGER, preserved** (CI hugs 0.50) | (rare) | Leak fixed, no gain → **A4** *or* **D1** | Gate froze us → **B4** (disable gate as negative control) |
| **NOT_STRONGER, leaks** (CI clearly < 0.50) | Gate ineffective → **B4** + escalate | Tuning insufficient → **A4** mandatory, **D1** parallel | Worst case → **A4** + **D1** in series |

**Prior probabilities** (recorded to calibrate later): p≈0.55 preserved-no-gain,
p≈0.25 STRONGER, p≈0.20 still-leaks.

---

## Outcome

### Result 1 — Original bundled run (2026-05-26) — **INVALIDATED**

> 🚨 **INVALIDATED 2026-05-26 (~14:30 UTC).** A long-standing bug in `trainer.py`'s
> `decode_phase` helper read `state[5]` unconditionally to label each sample's game
> phase — correct for the 6-channel basic encoder but wrong for the 15-channel
> enhanced encoder (where CH_GAME_PHASE=12 and channel 5 carries sparse row-threat
> data). Every 15-channel sample was labelled `RING_PLACEMENT`, silently disabling
> the configured `phase_weights: MAIN_GAME=2.0` boost in `sample_batch`. **MAIN_GAME
> positions were under-sampled by 2× throughout B1+B2+B3 training.** The network
> trained is NOT the network the config was supposed to produce.
>
> **What stays valid:** the phase-weight bug discovery itself; the gate fail-closed
> defense (logic, not data); the A4 phase 1.5 auto-detect wiring; the buffer
> converter + filter logic; the named channel constants on both encoders.
>
> **What's invalidated:** "B1/B2/B3 confirmed working"; the Glicko-drop comparison
> vs D.2 (-23 vs -107, buggy-vs-buggy); the SPRT verdict's *interpretation* (WR
> 0.468 is a real measurement, but of a model trained with broken phase sampling);
> the decision-matrix outcome. **Same exposure: D.2** (also a 15-ch run under the
> buggy `decode_phase`). A1's STRONGER verdict is unaffected (pretraining doesn't
> use the trainer's phase weighting).

SPRT verdict of the (invalidated) run:
- **NOT_STRONGER** (crossed -2.94 boundary at game 94 of 400-cap)
- Candidate 44-50-0, WR **0.468**, CI95 **[0.370, 0.568]**, LLR -3.135
- Color split cand_white=24, cand_black=20
- Duration: 65 min, RTX 5090
- JSON: `logs/branchB1B2B3_iter4_ema_vs_anchor.json`

**The crucial detail:** SPRT WR (0.468) is statistically identical to iter_1's and
iter_2's in-loop arena WRs (0.465, 0.468 — the reverted candidates); the iter_4
in-loop arena bump to 0.516 (which the gate promoted) was sampling noise — under
independent re-sampling iter_4's true WR vs iter_0 is ~0.47.

### Result 2 — RE-RUN #2 (2026-05-27) — **NOT_STRONGER**

The actual experiment, on post-phase-weight-fix code. Same config (B1 gate 0.50 +
games_per_match 200, B2 lr 1e-5, B3 games_per_iter 200), same warm-start.
- **NOT_STRONGER** (crossed -2.944 LLR boundary at game 103 of 400-cap)
- Candidate (iter_1_ema) 49-54-0, WR **0.4757**, CI95 **[0.3819, 0.5713]**, LLR **-3.116**
- Color split cand_white=30, cand_black=19 (acceptable at low n)
- Duration: 98 min on RTX 5090
- JSON: `logs/branchB1B2B3_rerun2_iter1_ema_vs_anchor.json`

**The crucial detail:** the phase-weight fix **reproducibly produces a +5 WR jump at
iter 2 in the in-loop arena** (51.7% first re-run, 52.0% this one — independent
hosts, statistically identical), but **that jump does NOT translate to a decisively
stronger model at the SPRT bar** — iter_1's true WR vs warm-start under independent
re-evaluation is ~0.476, within sampling noise of 0.50.

### Result 3 — Alignment-loop analysis (2026-05-27) — **MEASURED** (child analysis)

3-way comparison (`anchor` / `iter1_ema` / `iter4_ema`) on 95 stratified
MAIN_GAME-heavy positions from the post-fix replay buffer, at sims=[0, 400, 1600,
3200]. Tooling: `analysis_board/loop/` — stratified sampler
(`sample_positions.py --stratify move_number`), per-position MCTS measurement
(`measure.py`), N-way per-metric report (`compare.py`).

**Result block (3200 sims, N=95 shared positions):**

| Metric | anchor | iter1 | iter4 |
|---|---:|---:|---:|
| Mean rank_of_final_best (lower better) | 7.44 | 7.49 | 7.94 |
| % misaligned (rank≥3, lower better) | 46.3% | 45.3% | 45.3% |
| Mean value_gain_over_raw (closer to 0 better) | -0.005 | -0.007 | -0.012 |
| % costly (gain≥0.1, lower better) | 7.4% | **5.3%** | 10.5% |
| % opposite-sign divergence (lower better) | 2.8% | 4.2% | 7.1% |
| Mean best_move_value | +0.275 | +0.267 | +0.246 |

**The crucial detail:** **iter1 has the lowest % costly (5.3% vs anchor's 7.4%)** —
a real, signed improvement that SPRT at N=103 missed (47.6% WR, NOT_STRONGER). iter4
is materially worse than iter1 across every alignment metric, matching the
autopilot's revert decision. Alignment appears to be a *sensitive* training-quality
signal where SPRT-at-small-N is noise-limited.

---

## Details

### RE-RUN #2 — comparison vs invalidated run (statistically indistinguishable)

| Run | Candidate | Phase fix? | SPRT WR | CI95 | Verdict |
|---|---|---|---|---|---|
| Invalidated | iter_4_ema (final of 5 reverted) | NO (buggy) | 0.468 | [0.370, 0.568] | NOT_STRONGER |
| RE-RUN #2 | iter_1_ema (only promoted candidate) | YES | 0.476 | [0.382, 0.571] | NOT_STRONGER |

The phase fix changed the loop's **shape** (one real promotion at iter_1 instead of
zero through five iters) but not the **ceiling** (final best ~0.47–0.48 WR vs
warm-start in both cases).

### RE-RUN #2 — within-run iter-by-iter (Wilson gate WR vs current best)

| Run-iter | Candidate | WR vs best | Gate verdict | Best after |
|---|---|---|---|---|
| 1 | iter_0 (warm-start) | n/a | NEW BEST (auto-initial) | iter_0 |
| 2 | iter_1 | 52.0% vs iter_0 | Wilson REJECT, Elo override **PROMOTED** | iter_1 (Elo 1513.3) |
| 3 | iter_2 | 50.5% vs iter_1 | Wilson REJECT, Elo not improved → REVERT | iter_1 |
| 4 | iter_3 | 45.0% vs iter_1, CI [0.402, 0.499] | REVERT (statistically below 0.50) | iter_1 |
| 5 | iter_4 | (reverted, Elo 1498.3 vs 1513.3) | REVERT | iter_1 |

Pattern: one real candidate (iter_1) clears the gate via the Elo override path;
iters 3–5 all revert.

### RE-RUN #2 — confirmed/falsified mechanisms

1. ✅ **Phase-weight fix works empirically.** iter 1 buffer phase mix MAIN_GAME 76.6% / RING_PLACEMENT 15.4% / RING_REMOVAL 8.0% — matches intended distribution; the MAIN_GAME=2.0 weight now actually applies to ~77% of samples.
2. ✅ **The +5 WR-jump at iter 2 is REPRODUCIBLE** across independent hosts (51.7% then 52.0%, same config/warm-start, different hardware). Not noise.
3. ❌ **The +5 jump does NOT survive independent SPRT re-evaluation.** iter_1's true WR vs warm-start ~0.476; the SPRT's 0.60 bar is not cleared.
4. ❌ **Iters 3–5 all revert** under fixed phase weighting — same broad behavior as the invalidated run (WR pattern differs: invalidated had iter_1/2/3 all reverting ~47%; this run iter_2 promotes at 52% then iters 3–5 revert).
5. 🟡 **Drift mechanism (iters 3–5 declining vs iter_1) is unresolved.** 3 data points (50.5%, 45.0%, iter 5 not gate-logged), SE ~2.5% — "spiral" framing was overconfident.

### Original (invalidated) run — within-run iter-by-iter (gate-relevant WR vs iter_0)

| Cand | WR | Elo | Wilson Gate | Decision |
|---|---|---|---|---|
| iter 1 | 46.5% | 1476.7 | wins=186/400, CI95 [0.417, 0.514] | REVERT |
| iter 2 | 46.8% | 1496.0 | wins=187/400, CI95 [0.419, 0.516] | REVERT |
| iter 3 | 48.4% | 1492.7 | (similar CI band) | REVERT |
| iter 4 | 51.6% | 1507.3 | (point estimate > 0.50) | **PROMOTED** (false positive per SPRT) |

iter 1 Glicko drop vs iter_0 was **-23** here (1500 → 1476.7) vs D.2's **-107** —
the **B2 (lr 1e-5)** effect: ~80 Glicko of value preservation. (Comparison is
buggy-vs-buggy; both runs had the phase-weight bug.)

**Original headline comparison vs D.2** (after refreezing anchor to `best_supervised.pt`):

| Run | Loop | Anchor | Verdict | WR | CI95 |
|---|---|---|---|---|---|
| D.2 | iter_4_ema, lr 1e-4 | OLD `best_iter_4` (weaker) | NOT_STRONGER | 0.526 | [0.470, 0.582] |
| B1+B2+B3 | iter_4_ema, lr 1e-5 | NEW `best_supervised` (strong) | NOT_STRONGER | 0.468 | [0.370, 0.568] |

Both ≈ "same strength as starting checkpoint," but B1B2B3 was vs the harder strong
anchor — the more rigorous test, and the loop is no longer net-destructive.

### Alignment-loop analysis — trajectory & findings

Trajectory (anchor → iter1 → iter4): mean rank 7.44 → 7.49 → 7.94; % costly 7.4% →
**5.3%** → 10.5% (real improvement at iter1, then degradation); % opposite-sign
2.8% → 4.2% → 7.1% (monotonic); best_move_value +0.275 → +0.267 → +0.246
(monotonic). The % costly trajectory is noisiest but most signed; opposite-sign and
best_move_value are cleanly monotonic and corroborate WR-trajectory degradation.

- ✅ Post-fix iters DID produce measurable behavioral change vs anchor, not just noise.
- ✅ Alignment trajectory matches WR trajectory — first independent corroboration of the autopilot's promote/revert decisions from an axis other than win-rate.
- ✅ Opposite-sign divergence is rare, not systemic (2.8–7.1% at 3200 sims), NOT the 43.8% an earlier corrupted run suggested.
- 🟡 (open) Does alignment improvement predict SPRT improvement at larger N? iter1's +2.1pt % costly improvement maps to NOT_STRONGER at N=103 — worth testing whether N=400–800 SPRT resolves the alignment delta into a WR signal (→ alignment as a faster training-quality detector than SPRT).

### Gate-override path is permissive (note carried from 2026-05-26 19:26 UTC)

Observed in iter 2 of the RE-RUN:
```
Wilson Gate Check: wins=207/400 (win_rate=0.517, SE=0.025, CI95=[0.469, 0.566], threshold=0.5) -> REJECT
✅ NEW BEST: Copied checkpoint_iteration_1.pt to best_model.pt
Decision: ✅ NEW BEST (promoted to best model)
```
Wilson said REJECT (lower bound 0.469 < 0.50) but `supervisor.py:1731-1736` ran a
third branch promoting on `candidate_elo > best_model_elo` regardless — making the
Wilson gate effectively **advisory**. The `wilson_attempted_no_data` fail-closed
defense (added 2026-05-26) only catches "Wilson couldn't run," not "Wilson ran and
said no." Not fixed mid-run (user chose to let it continue; real signal in the iter
2 promotion, behavior known not catastrophic). **Patch shape (~30 min):** tighten
the elif at `supervisor.py:1731` so the Elo override is only allowed when
`perform_wilson_check == False`:
```python
elif candidate_elo > self.best_model_elo and not perform_wilson_check:
    promote = True
    self.logger.info(f"... Elo improved ({candidate_elo:.1f} > {self.best_model_elo:.1f})")
```
Add regression test in `test_supervisor_gate_fail_closed.py` asserting: Wilson ran,
returned False, Elo up → outcome "kept" not "promoted." Two concrete failure modes
documented (iter_4 in invalidated run, iter 2 in re-run) both promoted via Elo
override when the gate intended to reject. **RE-RUN #2 nuance:** the override is
*conditional, not blanket bypass* — iters 3–5 all had Wilson REJECT AND Elo not
improved → all REVERTED correctly. The proposed fix is a tightening, not a
bug-fix-for-broken.

### Operational lessons

- **🚨 PHASE-WEIGHT BUG** (discovered 2026-05-26 during D1-partial prep). `trainer.py`'s local `decode_phase` read `state[5]` unconditionally — correct for 6-ch (CH_GAME_PHASE=5), silently wrong for 15-ch (channel 5 = sparse row-threat; CH_GAME_PHASE=12). **Every position in every 15-channel replay buffer was labelled RING_PLACEMENT.** The mislabel flowed through `ReplayBuffer.sample_batch`'s phase-aware weighting (`trainer.py:446-447`), disabling `phase_weights: MAIN_GAME=2.0` — every sample got RING_PLACEMENT weight (1.0). Real phase mix (verified post-fix): MAIN_GAME=75.6%, RING_PLACEMENT=16.0%, RING_REMOVAL=8.4%. Affected runs: D.2, B1+B2+B3. **Fix:** added `CH_GAME_PHASE` named constant to `StateEncoder`, plus `phase_channel_index(num_channels)` and `decode_phase_from_state(state)` in `encoding.py` as single sources of truth; trainer delegates to that utility; regression tests in `yinsh_ml/tests/test_decode_phase_cross_encoder.py`. See `TECH_DEBT.md`. This added a **fourth knob option (correct sampling)** to the B-family that wasn't on any previous list.
- **Buffer is 63% zero-policy rows.** Of 100K samples ~63% have all-zero `move_probs` (terminal-position dummies + early-game positions where MCTS distributions weren't captured). D1-partial corpus is effectively ~36K samples, not 100K; the converter `scripts/convert_replay_buffer_to_mmap.py` filters them before saving.
- **`--export-every 0` CLI override is broken.** CoreML export ran at iter 5 despite the flag because `run_training.py:254` does `export_every = args.export_every or int(...)` ("0 is falsy" bug). Fix: `args.export_every if args.export_every is not None else ...`. Export crashed (6-vs-15 ch bug) but try/except-wrapped, non-fatal. Cosmetic.
- **Gate uses point-estimate, not Wilson lower bound.** iter_4 promoted at 51.6% (Wilson lower ~0.467 < 0.50) but SPRT re-sample shows true WR 0.468. **Recommended:** require Wilson lower bound ≥ threshold (not point estimate) — trivial code change; otherwise every future run risks "false-positive promotion → next iter trains from a noisy iter_N → propagates noise."
- **First host's CUDA failed mid-run** (RE-RUN #2). First attempt crashed during iter 3 anchor eval with `CUDA_ERROR_UNKNOWN`; container reboots + vast.ai host restart didn't recover; resolution was destroy + spin up on a different physical host. Operational risk: long vast.ai runs have non-trivial failure-mid-run probability — checkpoint more aggressively or split into resumable segments.
- **Phase-weight fix is empirically necessary** for 15-ch runs. Logged D.1 v3 candidate: re-run D.1 v2 (GAP value head, was 1-15-0 NOT_STRONGER) on fixed code — buggy training may have unfairly tarred GAP heads.
- **(alignment-loop) `measure.py` was silently corrupting per-position MCTS measurements** for two days. `server.get_mcts()` updated to `enable_subtree_reuse=True` (for step-into-line PV extraction); the Flask `/api/evaluate` path got `mcts.reset_tree()` before each search but `measure.py` (calling `mcts.search()` directly) did not — so every position after the first inherited the previous position's tree. Smoking-gun: identical `rank_of_final_best` across all sim budgets, surfacing as the alarming "43.8% opposite-sign rate" headline that was the bug talking. Fix: one-line `mcts.reset_tree()` in `measure_one()` (commit `35b3977`). All loop-runs since `a52e3f5` (2026-05-26 afternoon) had to be re-run. **Discipline:** when enabling state-persistence on a shared instance, audit every caller for paired resets.
- **(alignment-loop) Stratified sampling** (`--stratify move_number`) made late-MAIN_GAME buckets sturdy (move 60+ went from ~3 to 30+ positions). Should be default. Per-row best_move_value pills + opposite-sign-divergent flag in the analysis-board UI surface the value-head blind spot at a glance.
- **Replay buffer preserved** (original: ~50K MCTS-target samples, 12 MB gzip, `experiments/branchB1B2B3_run_2026-05-26/.../replay_buffer.pkl.gz`; RE-RUN #2: ~100K samples correct phase mix, `experiments/branchB1B2B3_rerun2_2026-05-27/full_run_dir/20260527_001626/replay_buffer.pkl.gz`). Generated by the strong warm-start with proper batched MCTS + subtree reuse — lets D1 test "MCTS-target pretrain beats yngine-outcome pretrain" without the 10–15h generation cost.

### Follow-up investigation queue — "what would change my mind about the drift story"

Captured 2026-05-27 mid-run, on whether iters 3–4's declining WR (52.0 → 50.5 →
45.0 vs iter_1) is a "degenerating spiral" vs "luck of the draw." With 3 data points
and SE ~2.5%, the spiral framing was overconfident.

**Working priors (~13:00 UTC 2026-05-27):** ~50% pure noise / random walk; ~30%
heuristic-weight annealing exposing iter_1's noisy value head; ~15% optimizer state
surviving revert; ~5% buffer mode collapse / unknown.

**Updated priors (2026-05-27 evening, post-SPRT, see `POST_RUN_DIAGNOSTICS.md`):**
~60–70% pure noise (now leading); ~30–40% heuristic-weight annealing (untested,
requires GPU re-run); **0% optimizer state** (❌ ELIMINATED by code inspection —
`supervisor.py`'s revert path calls `_reinitialize_optimizers()`, default-on via
`reset_optimizer_on_revert`, creating fresh `optim.Adam`/`optim.SGD`; no carried
state); **0% buffer composition** (❌ ELIMINATED by 3-way buffer comparison — D.2
buggy vs B1B2B3 original buggy vs RE-RUN #2 fixed all have ~76% MAIN_GAME, stable
~64-move length, similar policy sparsity; one small shift: 14.8% extreme-value
samples in the fixed run vs ~11% in buggy, consistent with "MAIN_GAME-emphasized
training → more decisive games," not a drift cause). **Discipline rule:** with a
"trend" across 2–3 data points each ±5% CI, default to noise; require 5+ data points
or a mechanism-level argument before claiming structural drift.

Named probes (each testable):

- **Mechanism 1 — Heuristic-weight annealing exposes iter_1's noisy value head.** Config anneals `heuristic_weight` 0.5 → 0.0 over 5 iters (iter1 50%net+50%heur … iter5 90%net+10%heur). If iter_1 has noisy value estimates (iter 2's +5 WR could be a policy-head gain not transferring to value head), MCTS targets get progressively less reliable as heuristic regularization falls away. **Testable:** re-run with `heuristic_weight_end: 0.3` (vs 0.0); if drift disappears/weakens, this is dominant. **Cost:** same as B1B2B3 (~12–15h, 1 config knob). *(Only remaining structural alternative still untested.)*
- **Mechanism 2 — Optimizer state survives revert and accumulates bad momentum.** On revert, weights go back but does Adam's `m`/`v` reset? If not, momentum from the rejected iter's direction carries into the next. **Testable (no GPU):** read `supervisor.py` revert path (`_load_best_model` ~1740–1780; whether `optimizer.state_dict()` resets alongside weight reload; the `decision_kind == 'reverted'` path). If state survives, bug-shaped — fix: reset optimizer state on revert + regression test. **Cost:** 30–60 min. *(❌ ELIMINATED — see updated priors.)*
- **Mechanism 3 — Buffer composition / mode collapse.** With FIFO eviction and `max_buffer_size: 100000`, buffer transitions warm-start games → iter_1 games over iters 2–3. **Testable (no training):** compare final buffer to iter-1 buffer — visit-distribution sparsity (`np.array([(p > 0).sum() for p in buf['move_probs']])`), value-target distribution shift (histogram `buf['values']` iter1 vs iter4, clustering at ±1.0), game-length distribution (from `buf['move_numbers']` resets). **Cost:** ~30 min, data on disk. *(❌ ELIMINATED — see updated priors.)*
- **Mechanism 4 — Pure noise.** 52.0 → 50.5 → 45.0 as draws from a stable ~50% distribution is uncommon but not crazy (45.0 ~2σ below mean). **Testable:** relax the 5-iter cap to 10–15 and watch. **Cost:** extending to 10 iters ≈ +30h compute — probably not standalone-worth, but any future B-family run should consider more iters to clear the 3-data-point ambiguity. *(Now the leading explanation.)*
- **Cross-mechanism diagnostic — buffer/value-head inspection post-run.** Regardless of dominant mechanism, after the run: (1) download `replay_buffer.pkl.gz`; (2) run buffer diagnostics (sparsity, value distribution, game length); (3) inspect optimizer state code path; (4) write up as addendum to the RE-RUN #2 entry. The entry should NOT claim "loop is non-additive past iter 2" without first ruling out Mechanism 2 and Mechanism 3 (both answerable from on-disk data).
- **Discipline for future write-ups.** When the eye sees a "trend" but n < 5 and each measurement has ±5% CI, the writeup MUST either (a) state the null (noise) and alternatives with explicit probabilities, or (b) defer until more data. Guard against narrating a single observed trajectory as a mechanism.

### Snapshot narrative (2026-05-27 / 28)

- **Current frozen anchor:** `models/yngine_volume_15ch_pretrain/best_supervised.pt` (the D.2 15-ch pretrained warm-start; re-frozen 2026-05-25 after A1 showed it STRONGER vs the prior `best_iter_4` anchor at WR 0.905, CI95 [0.711, 0.973]). Prior anchor: `models/branchC_volume_pretrain/best_iter_4.pt` (Branch C, 6-ch).
- **Cloud box can be released.** Artifacts pulled locally (see Artifacts above).
- **B1+B2+B3 is a closed experimental branch.** The bundle reproduces a small (+5 WR) in-loop signal under correct phase weighting but cannot produce an SPRT-level improvement. The structural ceiling of MCTS-200 self-play with this warm-start, configured this way, is "roughly the same strength as warm-start." Further progress requires changing the value-head loss structure (A4) or the corpus/target distribution (D1) — not the gate, LR, or game count.
- **Next experiment:** A4 + D1-partial combined (pretrain with regression value head AND the saved buffer's MCTS targets). Lowest cost, highest mechanism prior. Code prepped (commits `343aab6` + `2070650`); launch-ready. Two correctly-trained buffers (this run + first re-run) total ~100K MCTS-target samples with correct phase distribution. D.1 v3 (re-run GAP value head on fixed code) is a low-priority candidate.

## Provenance & links

- Original bundled run 2026-05-26 (launched 2026-05-25 23:35 UTC); invalidated 2026-05-26; RE-RUN #2 + alignment-loop analysis 2026-05-27; snapshot 2026-05-28.
- Predecessor: [[a1_d2_pretrain_vs_iter4]] (showed the warm-start strong + loop destructive → put B1+B2+B3 at the top of the queue). Shares the phase-weight bug with [[branchD2_15ch_encoding]].
- iter_1 promotion later benchmarked externally in [[vs_yngine_benchmark]] (`iter1_ema` swept yngine-1K 17-0).
- Cross-doc refs: `TECH_DEBT.md` (phase-weight bug), `POST_RUN_DIAGNOSTICS.md`, `VOLUME_PRETRAIN_RESULTS.md`, `D2_PREP.md`, `TODO_baseline.md` #28 (trainer.py:1442 search-consistency mid-capture contamination, ~1.3% loss, gated "do not land mid-run").
