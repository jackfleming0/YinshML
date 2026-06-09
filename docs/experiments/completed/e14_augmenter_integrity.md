# E14 — Augmenter pipeline integrity check (residual-25% diagnostic, H_E)

**Status:** DONE: H_E ruled out (no pipeline bug)
**Date(s):** proposed 2026-05-30; run 2026-05-30 morning
**Cost:** ~15 min, ~20 LOC; no cloud
**Branch / artifacts:** standalone diagnostic (~20 LOC) over 100 random states from a replay buffer, using the validated per-state `augmenter`. Branch `policy-symmetry-fixes`.

## Description

The cheap rule-out for hypothesis **H_E** (encoding pipeline lossiness) among the four candidate causes of the residual 25% asymmetry remaining after symmetric MCTS ([[e8_symmetric_mcts]]). H_E: the augmenter's encode→transform→decode roundtrip might introduce small artifacts that mechanically explain some of the residual asymmetry.

**Method** — for 100 random states from a replay buffer:
- For each tid in {1, 2, 3}: `state → transform tid → transform tid` (involution) → compare to original bytewise.
- For each tid: `policy at state → forward transform → back transform` → compare bytewise.
- All round-trips must be exact.

If any state fails, the augmenter has a bug that explains some fraction of the residual asymmetry mechanically. Run E14 second (after E11), to rule out a stupid bug before spending more compute.

## Outcome

**PASSED — H_E ruled out, no pipeline bug.** State round-trip (transform→transform = identity, since D2 transforms are involutions) was **bytewise-exact**. Policy round-trip also **bytewise-exact**.

Combined with E11's result, this cleanly localizes the residual 25% to **H_W (asymmetric network weights)** rather than a mechanical pipeline artifact. The fix is E16.

## Details

E14 is the integrity-check companion to E11. Together they were the first two steps of the residual-25% investigation sequence (E11 weight check first at ~5 min, E14 pipeline integrity second at ~15 min) — specifically ordered so a trivial augmenter bug would be caught before committing the ~6h/~12h MPS sim/sample sweeps (E12/E13).

D2 transforms being involutions is what makes the state round-trip a clean identity test: applying the same transform twice must return the original state exactly. The policy round-trip (forward transform then inverse transform) likewise must be the identity on the action-space permutation. Both being bytewise-exact confirms the augmenter's encode→transform→decode path is lossless, so none of the observed orbit-internal asymmetry is a pipeline artifact.

## Provenance & links

- Source snapshot: 2026-05-30 morning ("E11 + E14 results — H_W confirmed, H_E ruled out").
- Residual identified by [[e8_symmetric_mcts]]; the prime diagnostic is [[e11_weight_symmetry_check]] (H_W confirmed); the fix is [[e16_symmetric_weight_reg]].
