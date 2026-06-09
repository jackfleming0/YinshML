# Branch D.2 — 15-channel enhanced encoding

**Status:** DONE: NOT_STRONGER
**Date(s):** 2026-05-25 (caveat added 2026-05-26)
**Cost / hardware:** Full pipeline ~16 min re-encode + ~5h pretrain + ~9h self-play + ~4h SPRT. SPRT itself 4.00h on RTX 5090.
**Branch / run dir / artifacts:**
- Run dir: `runs_branchD2/20260525_041120/`
- SPRT JSON: `logs/branchD2_iter4_vs_frozen.json`
- Candidate: `iteration_4_ema` (pretrain + 5 self-play iters)
- Anchor: frozen 6-ch `best_iter_4`

## Description

The run that generated this backlog. Full pipeline: Path B 6-ch→15-ch corpus
re-encoding (~16 min, 13.6M positions fp16 mmap) → 6-epoch supervised pretrain
(~5h, val PAcc 0.300 / VAcc 0.636) → 5-iter MCTS-200 self-play loop (~9h) → SPRT
vs frozen `best_iter_4` (~4h, 304 games).

Hypothesis: the 15-channel enhanced encoding (row threats, partial rows, ring
mobility, center distance, ring influence, turn number, score differential) lifts
strength over the 6-ch baseline once run through the standard pretrain + self-play
pipeline.

## Outcome

**NOT_STRONGER** (crossed -2.94 boundary at game 304 of 400-cap).
- DECISION: NOT_STRONGER
- Candidate 160-144-0, WR 0.526, **CI95 [0.470, 0.582]**, LLR -2.96
- Color split 80/80 — no deterministic-side artifact
- Duration: 4.00h on RTX 5090
- JSON: `logs/branchD2_iter4_vs_frozen.json`
- Run dir: `runs_branchD2/20260525_041120/`

**The crucial detail:** CI95 lower bound 0.470 is *below 0.50* — we can't even
confidently say "candidate is non-worse." This is worse than Step 2 (MCTS-400),
which landed at CI95 [0.504, 0.600] (real positive edge just below the 0.60 bar);
D.2 doesn't even clear that fainter "real edge" signal.

## Details

> ⚠️ **Caveat added 2026-05-26.** D.2 ran under the same phase-weight bug as
> B1+B2+B3 (15-channel encoder + `decode_phase` reading `state[5]`, which is a
> row-threat channel in the 15-ch layout — NOT the phase channel). MAIN_GAME
> positions were under-sampled by 2× throughout D.2 training. The SPRT measurement
> (WR 0.526 vs old anchor) stands as a real comparison of the trained models, but
> the *interpretation* ("self-play loop destroyed pretrain gains") may have
> over-called what was really "loop trained under wrong phase weights settled at a
> different fixed point." A1's STRONGER verdict for the pretrain checkpoint is
> unaffected (pretraining doesn't use the trainer's phase weighting). See
> [[b1b2b3_stop_the_leak]] invalidation banner for the full discussion.

**Self-play iter Glicko trajectory** (relative to iter_0 warm-start ≈ 1553):

| Iter | Glicko Elo | Δ vs iter_0 | Wilson gate result |
|---|---|---|---|
| 0 | ~1553 (initial 1500 ref) | — | PROMOTED (first) |
| 1 | 1446.7 | **-107** | PROMOTED (loose 0.20, 42% WR) |
| 2 | 1478.5 | -75 | PROMOTED (49.5% WR) |
| 3 | 1467.3 | -86 | PROMOTED (43% WR) |
| 4 | **1463.9** | **-90** | PROMOTED (42% WR) |

iter_0 (the pretrained init) appears to be the strongest D.2 model. The self-play
loop's noise was dominated by the loose Wilson gate accepting clearly-worse
candidates as "best" — the dilution now confirmed by the SPRT verdict.

**Confirmed findings:**
1. ✅ Self-play dilutes the warm-start at this MCTS budget. iter_4 is ~90 Glicko below iter_0; the SPRT verdict reflects that gap. *(Interpretation later qualified by the phase-weight caveat above.)*
2. ✅ Wilson 0.20 is too loose when the warm-start is strong. Every iter promoted at 42–49% WR — clearly worse than its predecessor.
3. 🟡 *Pending A1 result:* whether iter_0 itself beats `best_iter_4`. *(A1 later landed STRONGER — encoding lever WAS real, the self-play loop wasted it.)*
4. ✅ Search depth wasn't the only weak lever (Step 2). Encoding alone isn't either (D.2).

**Operational lessons logged:**
- **Encoder-flag propagation bug** (commit `4e984ef`): `ModelTournament` + `eval_vs_frozen_anchor.py` constructed bare `NetworkWrapper(device=...)` instances, missing the `use_enhanced_encoding` flag → hard-fail loading 15-ch checkpoints. Caught at D.2 iter 1; cost ~1h compute to detect + fix + restart. F1 (audit + fix all bare-construction sites) remains queued.
- **fp16 corpus storage**: D2_PREP.md was wrong by 10× on disk math (claimed 9.9 GB for states.npy, actual 99 GB float32). Mid-run patched regenerator to fp16 (49.4 GB) — loader does `.float()` cast on load, so storage dtype is transparent to training. No precision impact observed.
- **6-epoch pretrain decision** (vs the 3 epochs D2_PREP specified): PAcc kept climbing at ep 1 and was still climbing at ep 5. The extra 3 epochs added +0.029 PAcc (0.271 → 0.300) and +0.008 VAcc (0.628 → 0.636). The longer cosine schedule also gave a higher effective LR throughout — different experiment from "3 epochs with T_max=3 schedule."
- **CoreML export failed at iter 4** ("C_in / groups = 6/1 != weight[1] (15)") — same root cause as the tournament bug; CoreML export has a hardcoded 6-ch assumption. Non-blocking; logged for cleanup.
- **Autopilot SUMMARY.md writer bug**: expected flat SPRT JSON fields; actual schema nests under `results[0].sprt`. SUMMARY.md printed "UNKNOWN / 0-0-0 / None". Trivial fix; logged.
- **`--resume` support** for pretrain shipped (commit `d5e5151`) but not exercised yet — future extension runs can resume cleanly via `last_resume_state.pt`.

**Next experiments per sequencing matrix** ("INCONCLUSIVE WR ~0.50 or WEAKER" case):
1. **A1** — RUNNING as of 2026-05-25 17:14 UTC (iter_0 direct SPRT). *(Later STRONGER.)*
2. **F1** — bare-wrapper cleanup. Cheap, unblocks future cross-arch evals.
3. **B1 + B2 bundled** — tighter Wilson gate + lower self-play LR. Tests whether the loop can be tuned to preserve the warm-start.
4. **A4** — regression value head (theory II).
5. **D1** — self-play data corpus pretrain (theory III).

## Provenance & links

- Run + result: 2026-05-25; phase-weight caveat added 2026-05-26.
- Related: [[a1_d2_pretrain_vs_iter4]] (disentangles pretrain from loop — landed STRONGER), [[b1b2b3_stop_the_leak]] (shares the phase-weight bug; the "stop the leak" follow-up).
- Cross-doc refs: `D2_PREP.md` (disk-math correction), `VOLUME_PRETRAIN_RESULTS.md`, `TECH_DEBT.md` (phase-weight bug).
