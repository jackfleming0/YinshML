# vs-yngine — `iter1_ema` (deployed) sweeps yngine-MCTS-1K

**Status:** DONE: STRONGER
**Date(s):** 2026-05-28 (~23:30 UTC snapshot)
**Cost / hardware:** ~21 min (MCTS-200) + ~82 min (MCTS-800) ≈ ~1.7h total on Apple Silicon MPS. Per-game wall: 75.8 s mean at MCTS-200 / 288 s at MCTS-800.
**Branch / run dir / artifacts:**
- PR [#20](https://github.com/jackfleming0/YinshML/pull/20)
- Candidate: deployed `iter1_ema` (the iter-1 promotion from the B1+B2+B3 RE-RUN #2)
- SPRT JSON: `logs/iter1_ema_vs_yngine_sims{200,800}_sprt.json`
- Vendored `temhelk/yngine` as a submodule; driver in `third_party/yngine_driver/yngine_driver.cpp`

## Description

First measured win rate of *any* model checkpoint against an external reference
(yngine). Closes the V2b bridge gap deferred in `VOLUME_PRETRAIN_RESULTS.md
§ 2026-05-21` — relative strength (frozen anchor) had been measured for two weeks
without an absolute number. Vendored `temhelk/yngine` as a submodule, shipped a
stdin/stdout C++ driver + Python bridge + SPRT-capable eval harness, and ran two
SPRTs against the deployed `iter1_ema` model. Full detail in
`YNGINE_BENCHMARK_RESULTS.md`.

## Outcome

**STRONGER** at both sim settings. SPRT params: both p0=0.50, p1=0.60, α=β=0.05;
upper bound LLR=+2.94.

| Our sims | yngine sims | games | record | WR | CI95 | LLR | verdict |
|---:|---:|---:|---|---:|---:|---:|---|
| 200 | 1,000 | 17 | 17-0-0 | 1.000 | [0.816, 1.000] | +3.10 | **STRONGER** |
| 800 | 1,000 | 17 | 17-0-0 | 1.000 | [0.816, 1.000] | +3.10 | **STRONGER** |

Color balance clean (W: 9 / B: 8 per run; SPRT alternates colors). JSON:
`logs/iter1_ema_vs_yngine_sims{200,800}_sprt.json`.

**The crucial detail:** both runs **terminated at the minimum 17 games SPRT allows**
at these params. Even an opponent with WR=0.95 would require >17 games on average to
hit the upper boundary; the model saturates yngine-1K so completely the test can't
distinguish "much stronger" from "infinitely stronger." We have a *lower bound* on
WR (0.816 from Wilson) but no upper-bound-side information at all.

## Details

**Confirmed/pending findings:**
- ✅ **The supervised + self-play loop produces a model that beats the engine that taught it**, at the corpus-generation sim level. Expected but never measured — a sanity check on the entire pipeline.
- ✅ **Frozen-anchor WR has been a faithful proxy for absolute strength improvement at this regime.** `iter1_ema` is the iter-1 promotion from the B1+B2+B3 RE-RUN #2, which the SPRT vs frozen anchor said was indistinguishable from warm-start. Both being decisively above yngine-1K reframes "indistinguishable at SPRT bar" as "indistinguishable while both far above yngine-1K" — exactly the saturation behavior the frozen-anchor design anticipates.
- 🟡 (open) **Where does the model break?** Not measured at higher yngine compute. WAVE3 V2a fingerprint audit described yngine at MCTS-10K as "a serious engine, not hobby code" — 10× the level benchmarked here. The 17-0 result tells us nothing about MCTS-10K behavior.
- 🟡 (open) **Is the self-play loop's contribution to this WR large or small?** A SPRT of `yngine_volume_15ch_pretrain` (pretraining-only baseline, no self-play) vs yngine-1K would partition "supervised" vs "self-play" gain. If pretraining alone hits ~50%, the entire 50→100 gap here is the self-play loop's contribution.

**Operational lessons logged:**
- **yngine has two upstream bugs worked around in the driver, not the submodule.** (1) `MCTS::~MCTS` unconditionally joins an uninitialized `search_thread` — we run a 1-iter warmup search after every `MCTS` construction to make it joinable. (2) yngine's MCTS prints `DEBUG:` lines unconditionally to `std::cout` (`mcts.cpp:237, 304, 406`) — driver redirects stdout to `/dev/null` and uses a duped fd for protocol replies. Both detailed in `third_party/yngine_driver/yngine_driver.cpp` comments.
- **Memory pressure killed our first MCTS-800 run on game 6.** `ArenaAllocator` mmaps the full pool up front; 512 MB × sequential games triggered the macOS OOM killer mid-search (empty stderr, silent SIGKILL). Dropped driver default to 128 MB — still well above the ~50 MB MCTS-10K peak observed in the WAVE3 V2a cloud run. Crash recovery in the bridge is sound: yngine_driver dying mid-game logs `err: yngine_driver closed stdout unexpectedly`, the eval counts the game as a draw, and SPRT excludes draws from LLR — so the lost game was a no-op statistically (but ~30 min of wall clock).
- **SPRT was the right call mid-stream.** Original plan was 100 fixed-n games at each setting (~16h wall total); the first MCTS-200 run hit 26-0 in ~30 min and was clearly going to keep sweeping. The user prompted the switch to SPRT, which terminated at 17 games each setting (~1.7h total wall vs ~16h). Discipline: when point estimate is far from the test boundary, fixed-n is wasted compute *and* worse statistically (it doesn't reveal the saturation). Default future eval runs to SPRT unless there's a reason to want a specific fixed n.
- **Apple Silicon support is a one-line build patch** (add `__APPLE__` to upstream's `__linux__`-gated `mmap` branch). Idempotent sed via `build.sh`; we don't fork the submodule.

**Next experiments per sequencing matrix:**
- **yngine-MCTS-10K SPRT** (top-priority follow-up). Same harness, `--yngine-sims 10000`. yngine at 10K is "serious" per V2a; this is where the gap should narrow or flip. ~3–5× the per-move yngine time of the MCTS-1K run — budget ~5h wall for SPRT.
- **`yngine_volume_15ch_pretrain` vs yngine-1K SPRT.** Partition the supervised-vs-self-play contribution.
- **Wire yngine WR into the post-promotion gate.** Run `eval_vs_yngine` after every promoted iter as a cheap absolute-strength tripwire. SPRT at MCTS-200 vs yngine-1K is ~20 min wall — affordable per promotion. Flag any regression below WR=0.5 immediately.

## Provenance & links

- 2026-05-28. Closes the V2b bridge gap from the 2026-05-21 session.
- The benchmarked checkpoint `iter1_ema` is the iter-1 promotion from [[b1b2b3_stop_the_leak]] (RE-RUN #2). Pipeline ancestry: [[a1_d2_pretrain_vs_iter4]], [[branchD2_15ch_encoding]].
- Cross-doc refs: `YNGINE_BENCHMARK_RESULTS.md` (full detail), `VOLUME_PRETRAIN_RESULTS.md` (§ 2026-05-21 bridge gap), PR [#20](https://github.com/jackfleming0/YinshML/pull/20).
