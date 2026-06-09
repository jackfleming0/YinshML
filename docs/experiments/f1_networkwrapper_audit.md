# F1 — Audit + fix bare `NetworkWrapper(device=...)` construction sites

**Status:** QUEUED  (the critical path — `tournament.py`, `eval_vs_frozen_anchor.py` — was already fixed during the D.2 crash recovery; the 8 remaining sites below are still outstanding)
**Cost:** ~1h coding, ~$0 compute.
**Stack-rank:** Likely+ 5 / Unblocks 5 / Info-gain 3 / Cost 5 / Impl-risk 5 / Sum 23
**Dependencies / blocks:** None.

## Description
**Goal:** find every script that constructs `NetworkWrapper(device=...)` then calls `.load_model(path)`, replace with `NetworkWrapper(model_path=path, device=...)` so the auto-detection path engages. Add a test that ensures cross-architecture checkpoint loads work.

**Mechanism:** the bug we hit in D.2 (encoder channel mismatch crash in tournament + anchor eval) was due to bare-construction-then-load patterns. The wrapper has auto-detection logic, but only when `model_path` is passed to `__init__`. We fixed the critical path (`tournament.py`, `eval_vs_frozen_anchor.py`); other scripts have the same pattern.

## Outcome
Pending — all 8 sites fixed, regression test added, committed.

## Details

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
1. For each script, change `NetworkWrapper(device=...)` followed by `.load_model(path)` to `NetworkWrapper(model_path=path, device=...)`.
2. Some scripts may need additional logic if they don't know the path up-front. Audit case-by-case.
3. Add a regression test in `yinsh_ml/tests/` that loads a 15-ch checkpoint into a fresh wrapper via the constructor path.

**Open questions:**
- Should we ALSO add channel auto-detection to `NetworkWrapper.load_model` itself (not just `__init__`)? Would prevent this class of bug permanently. Tradeoff: the current hard-fail is a deliberate safety check; auto-detect could mask wrong-flag callers. Probably better to fix sites explicitly than weaken the guard.

## Provenance & links
- Related: A1 / D.2 cross-encoding eval crash (the failure that motivated the audit). Finding 6 in "Findings driving this backlog" ("several other scripts with the same bare-`NetworkWrapper(device=...)` pattern").
- Source: `EXPERIMENT_BACKLOG.md` "Detailed write-ups" section.
