# Wave 3 — Branches A / B / C

Three branches that follow from `WAVE2_STEP_2_POSTMORTEM.md`. Sequential by default — A informs the scope of B, A+B informs C.

---

## Branch A — Eval non-EMA candidates

**Goal**: settle the F4 EMA-drift hypothesis. Re-run the §8-style anchor eval against each iter's **non-EMA** `checkpoint_iteration_N.pt`, compare to the EMA results we already have.

**Cost**: $0 if local (laptop CPU/MPS), or ~30 min on the existing Vast.ai box if it's still up. Each eval is 40 games at 48-sim ≈ 20 min on the 4090.

**Pre-conditions**:
- Step 2 cloud box still alive at `cloud` alias (else re-fetch the 4 candidate checkpoints from `runs_warm_start_combined_recipe/20260514_122701/iteration_{1,2,3,4}/checkpoint_iteration_{N}.pt`).
- §8 EMA evals (currently running) finished — we want their iter 0 baseline as the comparator.

**Procedure**:
```bash
# Cloud (preferred — same hardware, less variance vs the EMA evals):
ssh cloud
cd /workspace/YinshML && . .venv/bin/activate
RUN=runs_warm_start_combined_recipe/20260514_122701
OUT=$RUN/branchA_evals
mkdir -p $OUT

for it in 0 1 2 3 4; do
    python scripts/eval_vs_heuristic.py \
        --checkpoint $RUN/iteration_$it/checkpoint_iteration_$it.pt \
        --num-games 40 --depth 1 --mcts-simulations 48 \
        --label iter${it}_noema_48 \
        --device cuda --output-json $OUT/eval_noema_iter$it.json
done
```

**Comparators**:
| Iter | EMA MCTS-48 WR (Step 2 sidecar) | non-EMA WR (this) |
|---|---|---|
| 0 | 60.0% | ? |
| 1 | 50.0% | ? |
| 2 | 35.0% | ? |
| 3 | 32.5% | ? |
| 4 | 47.5% | ? |

**Outcome paths**:

| If non-EMA shows… | Then F4 is… | Next step |
|---|---|---|
| Non-EMA holds steady ~60% across iters | **Confirmed**. EMA drift is the bottleneck. | Branch B should focus on `use_ema_for_eval: false` or much higher `ema_decay` (e.g. 0.9999). |
| Non-EMA also decays in lockstep with EMA | **Refuted**. Training itself is producing weaker candidates. | Branch B should focus on the training step: LR schedule, buffer composition, exploration knobs. |
| Mixed (iter 1 non-EMA close to iter 0; later iters drift) | **Partial**. EMA amplifies a real underlying decay. | Branch B does both — first stop the EMA bleeding, then attack training degradation. |

**Headline number to report**: the iter 3 or iter 4 non-EMA WR. If it's ≥ 55%, EMA drift is the explanation. If it's ≤ 35%, training is the explanation.

**Why this is free and high-value**: it's a one-knob measurement that bisects the hypothesis space. We don't need a full re-train; the candidate checkpoints already exist.

---

## Branch B — Recipe sensitivity passes

**Goal**: identify the single highest-leverage recipe knob, change it in isolation, re-run a 5-iter recipe, see if best-iter raw policy moves above 33%.

**Cost**: ~$10-15 per pass on the 4090. One pass = same recipe as Step 2 (`warm_start_combined_recipe.yaml`) with exactly one knob changed. Wall time ~15-20h per pass. Budget for 2-3 passes (~$30-45).

**Pre-conditions**:
- Branch A complete; its outcome decides which knobs to vary first.
- `WAVE2_STEP_2_POSTMORTEM.md` understood.

**Candidate knobs (priority order, decided after A)**:

If Branch A says EMA is the bottleneck:
1. `arena.use_ema_for_eval: false` — eval the raw candidate. Tournament uses the same setting, so promotions also change. Most-direct test.
2. `trainer.ema_decay: 0.999 → 0.9999` — slow the EMA drift by 10×. Less invasive than turning EMA off entirely.

If Branch A says training is the bottleneck:
1. `self_play.dirichlet_alpha: 0.3 → 1.0` — increase exploration noise at root. The seed's policy may be a strong local optimum; more noise lets self-play sample around it.
2. `self_play.epsilon_mix_iteration_end: 0.3 → 0.7` — keep the epsilon-mix exploration up across iters instead of tapering it down (the iteration-progress scaling currently kills exploration by iter 4).
3. `trainer.lr_schedule: cosine → constant` — stop the LR from decaying to near-zero by iter 4 (suspected interaction with iter 4's value collapse, F5).

If Branch A says it's both:
1. Run two passes in parallel: one with `use_ema_for_eval: false`, one with `dirichlet_alpha: 1.0`. Compare to Step 2's baseline.

**Per-pass success criteria** (same as the original Step 2 doc's Q1):
- Best-iter raw-policy WR > 50% with CI lower bound > 40% at n=40 → real lift.
- Best-iter raw-policy ~33% with overlapping CI → recipe knob doesn't help.
- Worse than Step 2 → that knob was load-bearing in the wrong direction.

**What NOT to do during Branch B**:
- Don't batch multiple knobs into one pass. The whole point is isolation.
- Don't reach for hyperparameter sweep frameworks yet. Single-axis sensitivity first, then a sweep is informed.
- Don't change init checkpoint. Keep `supervised_seed/best_supervised.pt` to stay apples-to-apples with Step 2.

---

## Branch C — Architectural rethink

**Goal**: decide whether the *shape* of the warm-start + self-play pipeline is right for YINSH, or whether something more invasive is warranted.

**Cost**: a day or two of design work. No cloud spend.

**Pre-conditions**: Branch A + at least one Branch B pass complete. C is informed by what A and B reveal.

**Questions to answer (in order)**:

1. **Is the supervised seed too strong, or are we discarding its quality?** F2 says the seed already MCTS-48 ≈ 60% vs heuristic d1. If self-play data is generated by a 60%-MCTS policy that exploits depth-rather-than-policy, the buffer is full of moves that look correct only with search. Training to imitate those moves with just the policy head can't produce a network that wins without search — the training target is mis-specified.
   - Test: look at the raw vs MCTS gap. Seed: raw 20%, MCTS 60%. Gap is 40 points. That gap is the "search overhead" the network is being asked to internalize. It's huge.
2. **Should we train policy head from scratch, value head warm-started?** The seed's value head provides a calibrated value estimate. The policy head's exploration is bounded by what the seed already knows. Separating them: policy head learns from scratch via MCTS visit distributions; value head bootstraps from seed and gets refined.
   - This needs `--init-checkpoint` semantics that load partial weights, or a small refactor in `NetworkWrapper.load_model`.
3. **Or: drop the seed entirely; pure AlphaZero from scratch.** Long horizon, but the supervised seed may be a local optimum that no self-play recipe can escape. From-scratch with strong exploration may find a different, ultimately better basin.
4. **Or: ditch self-play, do supervised on stronger expert games.** Per the `project_expert_data.md` memory: the Boardspace/CG/BGA expert-data pipeline is built. If self-play can't produce better-than-seed games, expert games can.
5. **What does TRAINING_REFACTOR_PLAN.md propose, and is its priority list still right given F2/F4?**

**Deliverable**: a `WAVE3_BRANCH_C_DECISION.md` proposing one architectural change, with the test that would falsify it. Not a paper — a decision doc with one bet and one experiment.

**What NOT to do**:
- Don't start coding alternative architectures before the decision doc lands.
- Don't treat C as "do everything in TRAINING_REFACTOR_PLAN.md" — pick one bet.

---

## Out-of-scope for Wave 3

- Bigger network (resnet depth, channel width) — separate variable; takes the budget out of the recipe-tuning sweet spot.
- Multi-worker self-play — orthogonal to "is the recipe right?" Save for after Wave 3.
- C++ engine integration — orthogonal performance win.

## Pre-flight before Branch A

- [ ] Step 2 §8 evals finished (the EMA-baseline numbers — pulls iter 0 EMA's 400-sim / 48-sim / raw at n=60).
- [ ] `WAVE2_STEP_2_POSTMORTEM.md` reviewed.
- [ ] Cloud box still up (cheap to re-launch if not).
- [ ] supervisor.py:2028 log bug fixed (so future runs are readable).
