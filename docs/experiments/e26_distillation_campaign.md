# E26 — High-budget-search distillation campaign (Lever A)

**Status:** RUNNING / REAIMED (2026-06-09 — policy-distill, not value; pipeline committed, not concluded)
**Date(s):** scoped 2026-06-07 (post-E24 lever board); reaimed 2026-06-09
**Cost:** real — needs the [[e20]] throughput build (high-sim self-play is CPU-bound) or patience/compute; a multi-day run.
**Branch / artifacts:** branch `e25-binding-constraint` (current); runbook [`e26_box_runbook.md`](e26_box_runbook.md) (3-stage gen/distill/H2H); tooling `gen_selfplay_labeled_corpus.py`, `e25_ablation_h2h.py` (+ MCTS `ablate_policy`/`ablate_value` flags), plus E25's `value_head_calibration.py`. Recent commits: a47d1d5 (teacher-data generator + policy-distillation reaimed per E25), 36a3453 (distill detach-warning fix + box runbook), 4a9d3be (cap each CPU worker to 1 torch thread).

> **See the operational runbook** [`e26_box_runbook.md`](e26_box_runbook.md) for the 3-stage box workflow (generate teacher data → policy-distill → H2H). This file is the **experiment rationale**.

## Description
The principled root-cause fix: **the training target must EXCEED the student.** Generate data from a deliberately *stronger* teacher — iter1_ema (or the ensemble, [[e21]]) running MCTS at **very high budget (1600–3200+ sims)** — and **supervised-distill** those (search-improved policy, search-value) targets into the net.

**Why it's different from everything that failed:** it does NOT need the value head to spontaneously improve — *search manufactures the better signal* (KL grows with sims, P1-confirmed), then distillation banks it. [[e19]] tried depth *in the loop at lower budget*; this is *dedicated high-budget teacher-gen + clean offline distillation*, a materially stronger setup that attacks the "no gradient" root cause directly.

## Outcome
Not yet concluded — RUNNING / REAIMED. **Pipeline committed, not concluded.** The decision-relevant signal: a distilled net that beats frozen iter1_ema in H2H (the only verdict, R1).

**E25 gate resolved (2026-06-09) → the reaim.** The value eval is at an intrinsic ceiling (E25 §9: **0.663 on-distribution**), and no external engine exceeds iter1 (iter1 ≫ yngine > sharkdp), so there is **no stronger teacher** and the value-distill half is undercut. BUT search *does* manufacture a better **policy** (visit counts ≫ raw prior, P1), and the [[e25]] ablation proves **policy is the binding head**. **Reaim E26: distill the search-improved policy via high-sim self-distillation (iter1 searched harder), treat value targets as secondary.** This is the one surviving target-improvement lever; pair with encoding/capacity (A4 / richer representation).

## Details
- **Cost:** real — needs the [[e20]] throughput build (high-sim self-play is CPU-bound) or patience/compute; a multi-day run.
- **Reasons to not believe:** high-budget MCTS is still *guided* by the current eval, so if the eval is genuinely blind ([[e25]] says so), the teacher may not exceed the student by enough — which is exactly why **[[e25]] gates E26**. Also overlaps D1 (self-play corpus pretrain) and E1 — fold those in.
- The binding-constraint diagnostic (2026-06-09) made the consequence explicit: aim distillation at the **search-improved policy** (strong, improvable), not value targets (saturated). Head ablation: flatpolicy (uniform prior + real value) loses **0.90**, blindvalue (real prior + 0 value) loses **1.00** — both heads necessary, **policy load-bearing** (caveat: blindvalue=const-0 overstates value; ablation tests *necessity*, not *headroom*).
- Tooling for the reaim: `gen_selfplay_labeled_corpus.py`, `e25_ablation_h2h.py` (+ MCTS `ablate_policy`/`ablate_value` flags).

**Recommended sequence (2026-06-09):** ship [[e18]] + higher play-time sims (free) → run [[e25]] (~a day; find the real binding constraint) → commit the big chips to **E26**, *configured by what E25 reveals* (value → E26/[[e21]] teacher; policy/capacity → D/A4). The discipline is unchanged: cheap diagnostics aim the expensive swing; H2H vs the fixed champion iter1_ema is the only verdict (R1).

**Lever-board context (post-E24, 2026-06-07).** Four swings at the plateau — [[e19]] (depth), [[e22]] (cross-teacher), [[e24]] (LR sweep) — all NOT_STRONGER, but every one poked the mirror-continuation self-play loop while keeping the same value head. We exhausted *one class* of lever, not the space. The crack in the "ceiling" story: (1) the value-head ceiling was measured on noisy human games (AUC 0.737) — enthusiast play has a blunder floor no evaluator can predict; the one cleaner-corpus attempt (E24's `gen_engine_labeled_corpus.py`) was OOD garbage; (2) it was only ever trained with signals that don't exceed it — mirror self-play at 200 sims, where the MCTS-improved target ≈ the raw net (no gradient). So the untested lever is **inject a stronger, cleaner signal at scale** — the canonical thing that makes these systems strong. E26 is **Lever A** on that board.

## Provenance & links
- Source snapshots: 2026-06-07 (post-E24 lever board, Lever A); 2026-06-08 (value-ceiling snapshot); 2026-06-09 (E25 gate resolved + reaim to policy-distill).
- Related: [[e25]] (gates E26; its on-distribution 0.663 + policy-load-bearing ablation drove the reaim), [[e19]] (tried in-loop depth at lower budget — the weaker precursor), [[e20]] (throughput build E26's high-sim gen needs), [[e21]] (ensemble as an alternative teacher), [[e18]] (the free test-time-compute lever shipped in parallel). Lever D / A4 (scalar regression value head / bigger trunk) is the paired encoding/capacity track.
- Runbook: [`e26_box_runbook.md`](e26_box_runbook.md).
- Cross-doc: E25 full write-up [`completed/e25_sharkdp_value_ceiling.md`](completed/e25_sharkdp_value_ceiling.md).
