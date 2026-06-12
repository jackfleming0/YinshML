# E29 — Search-path optimization (throughput + parity-gated C++)

**Status:** QUEUED · **infra; part (a) build-first, part (b) parity-gated + deferred**
**Date(s):** scoped 2026-06-12
**Cost:** bf16 flip free; behavior-preserving throughput = free labor; GPU benchmarking ~$2–6. C++/tree-op search = free to *build*, **not** free to *trust*.
**Branch / artifacts:** `yinsh_ml/training/self_play.py::MCTS`, `server.py`; parity guard `yinsh_ml/tests/test_mcts_serial_vs_batch_parity.py`; `inference_server_dtype` config; E20 server (`docs/experiments/e20_throughput_build.md`).

## Description
Cut the cost of every paid high-sim run. The dominant paid costs in the current
program are the **≥400-sim gates** and **1600-sim teacher-gen** — and per the E26
infra finding both are **CPU/tree-bound** (1600-sim tier ran at GPU 10–12% util,
one core pegged). Since agent/dev labor is a prepaid/free resource here, the
default flips from "defer infra" to "build the safe speedups first" — they're free
to build and they cut the only thing that actually costs money (GPU-hours).

Two **distinct** classes, which must not be conflated:

- **(a) Behavior-preserving** — identical search output, just faster:
  - flip `inference_server_dtype: bf16` (E20-measured ~1.3×, already banked);
  - **parallelize the gate across cores** (independent H2H games on separate
    processes — each game is tree-bound on one core, so N cores ≈ N× on the gate);
  - corpus-gen batch width / coalescing on the wide-parallel self-play path.
  These are pure wins: free to build, cut the paid bill, **zero downside.**

- **(b) Behavior-changing** — a C++/tree-op search rewrite, larger virtual-loss
  batches that change which leaves co-expand. These are **new variables**, not free
  speedups. A silent search bug (miscounted visit, transposition collision) does
  not announce itself — it quietly corrupts the corpus and the gate *verdicts*, and
  you burn **paid** GPU running experiments on a broken foundation while trusting
  wrong answers. Free to build; **not free to trust.**

## Outcome
Pending —
- **(a) ship immediately**, before/alongside [[e27]] turn 1. No gate beyond "the
  profile confirms gate + corpus-gen are the binding hot paths" (they are; the
  inference server is **not** the lever for deep single-game search).
- **(b) build in the background, parity-gated** against the Python path
  (`test_mcts_serial_vs_batch_parity` must pass bit-for-bit on visit
  distributions before any experiment trusts it), and **deploy only after [[e27]]
  confirms the loop compounds** — its payoff is making *more loop turns* cheaper,
  so a saturated loop = no payoff. Sequenced ahead of E27 it's a sunk cost on turns
  you might never take.

## Details
- **Don't optimize the non-binding path just because it's free.** The E20 inference
  server is the flashiest lever but per E26 it does *nothing* for deep single-game
  search — only wide parallel self-play. Free labor on a non-binding optimization
  is still waste *and* adds bug surface to code paid runs depend on. Profile first
  (also free); confirm the hot path before touching it.
- **The scarce resources, once labor is free, are (i) trust in the foundation and
  (ii) wall-clock to the decisive answer** — not dollars (the whole near-term
  program is ~$25–45). (b) threatens (i); building infra ahead of E27 threatens
  (ii). Hence: (a) now, E27 turn 1 now, (b) background + deferred deploy.
- E20 is already at ~55% of the bf16 roofline; the remaining server-transport
  re-architecture (~1.5×) is explicitly "only if a run is provably
  throughput-bound" — it is **not** the gate/teacher bottleneck, so it stays
  parked.

## Provenance & links
- 2026-06-09 [[e20]] throughput build (bf16, virtual loss, coalescing; the
  ~55%-roofline / transport-bound finding).
- 2026-06-11 [[e26]] infra finding (high-sim search is CPU/tree-bound; server
  doesn't extend to deep search).
- Enables cheaper turns of [[e27]]; gated *by* [[e27]]'s compounding verdict.
