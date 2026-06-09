# E19 — Plateau-break ablation LADDER from iter1_ema (dual-arm depth)

**Status:** DONE: NOT_STRONGER (depth treads water; Arm A flat)
**Date(s):** scoped 2026-06-01; Arm-A depth result came in flat by 2026-06-02
**Cost:** Rough: 2–3 iters × 800 sims × 2 arms ≈ 20–30 GPU-h sequential (~10–15 h wall parallel) → order $10–25 on vast.ai. Cost is not the constraint; cores + wall-clock are.
**Branch / artifacts:** self-play config change `num_simulations: 200 → 800`; eval via 60-game color-balanced H2H vs frozen `iter1_ema`; profiling `profiling/symmetry_run/gpu_util_iter5_selfplay.csv`, `profiling/cprof_selfplay.py`. Seeds: `iter1_ema` (Arm A), `symmetric-15ch-iter1-ema` (Arm B).

## Description
The experiment we had never run: change **ONE variable at a time** off the *known-good* base (`iter1_ema`), and **measure H2H vs iter1_ema (color-balanced, ~60 games) at every rung.** Never rebuild from scratch.

Ladder rungs (each only fires if the prior stalls):
- **Rung 1 — depth only.** Continue-self-play from iter1_ema, sims 200→800, everything else identical, 3 iters. Question: does deeper search alone beat the champion? (Highest prior.)
- **Rung 2 — + dropout-off (L1) + label-smoothing (L2)** — only if rung 1 stalls. Lets the net learn the deeper targets without the over-confidence collapse the bare dropout-fix hit.
- **Rung 3 — + `discrimination_weight` on (≈0.5) + E2** — only if rung 2 stalls. Value-head hygiene.

First rung that beats iter1_ema is the lever.

**Hypothesis (depth):** at 200 sims the MCTS-improved policy barely exceeds the raw net, so the training target ≈ the net → no gradient. P1 confirms amplification exists (KL 3.46 nats, 96→800 sims), so deeper search should manufacture a target that exceeds the student.

**Dual-arm extension — measure SLOPE, not just level (Jack, 2026-06-01).** The H2H verdict measures each model's *level* under a *broken* loop; the decision-relevant quantity is its *slope* under the fixed (deep-search) loop, and those can rank-order differently. sym15's weakness is shallow/data-induced (45%-random corpus, which self-play overwrites) and it carries learner advantages iter1_ema lacks — dropout=0, the D2 symmetry prior (≈4× sample efficiency), 15ch features — that only cash out once there's a signal to learn. So run **rung 1 as TWO parallel arms: seed A = iter1_ema, seed B = sym15-iter1-ema, IDENTICAL loop config (the only difference is the seed model).** H2H *both* arms vs a **frozen** copy of iter1_ema every iteration and compare slopes (A starts ~50% vs frozen-self, B at ~27%). If B's curve rises steeper, the symmetric/15ch architecture is the better *substrate* and the big run pivots to it.

## Outcome
**DONE — NOT_STRONGER. Depth treads water; Arm A came in flat.** Deeper search alone did not beat the champion. The post-E24 lever board (2026-06-07) lists E19 (depth) as one of four swings at the plateau, all NOT_STRONGER, all of which "poked the mirror-continuation self-play loop while keeping the same value head." The E19 read that fed downstream: depth couldn't beat iter1_ema because the binding constraint is *value-head resolution* — the value head is only ~15% better than baseline (P2: Brier 0.66 vs 0.78), so 4× search just walks a wider tree to the same blurry verdict (depth amplifies evaluation, can't exceed it). Arm B's dropout-off head **also declined** → the limiter is the value *target*, not head architecture. This verdict directly chose [[e22]] (cross-teacher) as the next swing.

## Details
**Decision gate:** a rung wins → proceed to [[e20]] to scale it; all rungs stall → the plateau is structural — bank iter1_ema+E8 as the product and consider a deeper change (equivariant net E17 / value-head redesign / MuZero-style) rather than more of the same.

**Measurement discipline:** north-star metric is H2H vs the FIXED champion iter1_ema, not the within-lineage tournament (which is blind to absolute strength — it green-checked 5 iters of an 80-20 loser). Tighten promotion gate 0.20→0.55.

**Counter-prior (mild, not strong):** depth only helps if the value head can steer the deeper search, and that's sym15's confirmed weak axis (discrimination 0.035 vs iter1_ema's calibrated slope 0.98) — a possible chicken-and-egg trap. Defeasible (the value head can re-learn under a working loop), so not strong enough to skip the measurement.

**Implementation & infra (the elevated-runbook bit):**
- **Training change is trivial:** `num_simulations: 200 → 800` in the self-play config. That is the ONLY change for rung 1; both arms share it, the controlled variable is the seed model.
- **Binding resource is CPU CORES, not the GPU.** Self-play is CPU-bound (5090 sat at ~32% mean util, 40% idle, at 200 sims). 800 sims ≈ 4× MCTS tree work → ~4× CPU load and ~4× wall-clock per iter. One arm ≈ 80 cores (16 workers × ~5). **Two parallel arms ≈ 160 cores.**
- **GPU *tier* barely matters; GPU *count* + memory do.** A 4090 handles the batched inference fine even at 800 sims — do NOT pay up for a bigger GPU, it sits idle. But each arm wants its own card: 16 worker CUDA contexts ≈ 16 GB/arm, so two arms won't co-fit on one 24 GB GPU → **one GPU per arm.**
- **Box-rental decision:** filter by **core count (≥160 for two arms) + 2 GPUs ≥16 GB each.** A **dual-4090 box only wins if it ALSO has ~160+ cores** — many consumer dual-4090 instances ship with 32-64 cores and would be CPU-STARVED, *worse* than the single 192-core 5090 we used. Two clean options:
  - **(a) Sequential on one high-core box** (e.g., the 192-core 5090): arm A (3 iters) then arm B. Simplest, one box, ~2× wall-clock. Fine for a 2-3 iter diagnostic.
  - **(b) Parallel on a ≥160-core dual-GPU box:** one arm per GPU, ~half the wall-clock, more orchestration. Pick only if impatient.
- **Eval overhead is real:** H2H every iter × 2 arms × ~45 min (60-game color-balanced on CUDA); drop to 40 games if it dominates. Frozen iter1_ema is the fixed yardstick for both arms.
- **Do NOT build E20 (shared evaluator / bitboards) for this diagnostic** — eat the 4× slowness; throughput is only worth building once a rung proves depth works (R5/R9: prove the lever before amplifying it). Next box: rent with `--cap-add SYS_PTRACE` (py-spy); interpreter `/venv/main/bin/python`.

**Strategic framing (2026-06-01, Jack + AI — volume vs fine-tuning):** volume is a *multiplier on the per-iteration improvement rate, and ours is ≈0* — adding iterations/data has only ever tread water or regressed. Do NOT scale throughput at current settings (buys a more confident plateau at the same Elo). The highest-prior plateau-break lever is **search DEPTH, not game count**. **Throughput is not an alternative to fine-tuning — it is how you afford the depth that breaks the plateau.** Sequence: prove the lever cheaply, THEN build throughput ([[e20]]) to scale it.

**Meta-finding context (2026-06-01) that motivated E19:** nothing built had beaten iter1_ema — not E6 (lost 22-8), not the dropout-fix lineage (over-confidence collapse, WR 6%), not the symmetry run (lost 48-12). A pre-dropout-fix, pre-symmetry 05-27 model is still champion across 5 interventions. The plateau is "five targeted fixes failed to produce any model stronger than iter1_ema."

**An operational incident referenced downstream:** the E19 run had a NaN-target incident, leaving behind a NaN-target guard requirement (see [[e21]], [[e22]]).

## Provenance & links
- Source snapshot: 2026-06-01 ~21:30 UTC (symmetry-run verdict + next-run strategy); flat-result note folded from the [[e21]]/[[e22]] write-ups (2026-06-02/03).
- Related: [[e08]] (symmetric MCTS / E8), [[e18]] (deploy that symmetric MCTS), [[e20]] (throughput build, gated on an E19 win), [[e21]] (ensemble-teacher, parked behind the E19 verdict), [[e22]] (cross-teacher, chosen after the E19 verdict), [[e24]] (LR sweep, listed alongside E19 as a NOT_STRONGER swing).
- Cross-doc: memory `project_e18_e19.md`, `project_symmetry_run_outcome.md`.
