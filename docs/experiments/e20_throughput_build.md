# E20 — Throughput build (shared evaluator + bitboards) IN SERVICE OF depth

**Status:** QUEUED (gated on a proven depth lever)
**Date(s):** scoped 2026-06-01
**Cost:** not separately costed; it is an engineering build, not a compute run.
**Branch / artifacts:** config `use_shared_evaluator: true`; `board.py` / `moves.py` (bitboards); profiling `profiling/symmetry_run/gpu_util_iter5_selfplay.csv`, `profiling/cprof_selfplay.py`. Box python `/venv/main/bin/python`; rent with `--cap-add SYS_PTRACE` for py-spy.

## Description
Self-play is hard CPU-bound: 5090 at mean 32% util, ~40% of wall-clock at literal 0%, bursting to ~90% (full iter-5 profile: `profiling/symmetry_run/gpu_util_iter5_selfplay.csv`). ~2-3× GPU headroom sits behind CPU troughs. Two levers, both spent on **DEEPER search, not more shallow games**:
- **Shared/centralized evaluator** (`use_shared_evaluator: true`) — consolidate 16 workers' ragged small batches into large GPU batches; decouples CPU expansion from GPU eval so they overlap. **Likely the bigger lever** for raising the 4% median.
- **Bitboards** in `board.py`/`moves.py` (currently pure-Python over 85-cell dicts/lists) — shrink the CPU troughs. ROI = the board-ops share of worker CPU; **pending cProfile** (`profiling/cprof_selfplay.py`, result TBD this session) to size it. If board-ops is a big slice → 2× plausible; if dominated by MCTS-tree/torch-dispatch → smaller.

The principle: throughput **multiplies** improvement, it does not **create** it. "Throughput is not an alternative to fine-tuning — it is how you afford the depth that breaks the plateau."

## Outcome
Not yet concluded — QUEUED, gated. **Decision gate (R5/R9 — prove the lever before amplifying it):** only build E20 once a depth rung ([[e19]]) shows a *real per-iteration learning rate*. The signal that would un-gate it: an [[e19]]-style rung (or [[e26]] high-budget distillation) demonstrating depth/search actually produces a stronger model. As of the post-E24 board, no depth lever has been proven, so E20 remains queued. [[e26]] explicitly calls out that its high-sim self-play "needs the E20 throughput build (high-sim self-play is CPU-bound) or patience/compute."

## Details
- **Infra:** launch the next vast.ai box with `--cap-add SYS_PTRACE` so py-spy live-profiling works (the symmetry-run container lacked it; cProfile was the workaround). Box python = `/venv/main/bin/python`.
- Do NOT build E20 for the [[e19]] diagnostic — eat the 4× slowness; throughput is only worth building once a rung proves depth works.
- Sizing note for bitboards is **pending cProfile** — board-ops share of worker CPU determines whether 2× is plausible or the win is smaller (MCTS-tree / torch-dispatch dominated).

## Provenance & links
- Source snapshot: 2026-06-01 ~21:30 UTC.
- Related: [[e19]] (depth ladder; E20 gated on an E19 rung winning), [[e26]] (high-budget distillation; needs E20 throughput for high-sim self-play). Lever D / A4 and ensemble levers are cross-referenced on the post-E24 board.
