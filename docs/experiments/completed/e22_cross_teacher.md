# E22 — Cross-teacher self-play (sharpen the value head)

**Status:** DONE: FAILED (cross arm DEGRADES vs frozen iter1_ema)
**Date(s):** RAN 2026-06-03
**Cost:** ~200-sim / 5-iter × 2 arms — cheaper than [[e19]]. Box: ≥160 cores / 1×GPU≥16GB.
**Branch / artifacts:** branch `e22-cross-teacher`; configs `configs/e22_mirror.yaml`, `configs/e22_cross.yaml`; driver `scripts/e22_dualarm.sh`; tests `yinsh_ml/tests/test_cross_teacher.py`; diagnostics `scripts/value_head_calibration.py`, `scripts/value_head_finetune_probe.py`. Seeds: `iter1_ema`, `symmetric-15ch-iter1-ema` (sym15).

## Description
**Chosen after the [[e19]] verdict** (depth treads water; the limiter is the evaluator/value-head, NOT the policy head — Arm B's dropout-off head declined too). [[e19]] evidence says the value head is *calibrated but not sharp* (P2 Brier 0.66, ~15% over baseline) and Arm B's different head still declined → the limiter is the value *target*, not head architecture.

**Hypothesis:** mirror self-play between equals yields ~50/50 noisy outcome labels, so the value head can't learn discrimination; pitting iter1_ema against a *different* model makes games decisive → informative outcome signal → the value head sharpens → strength climbs.

**Dual-arm, ONE variable = the opponent** (both warm-start from iter1_ema (R2); both H2H'd vs a FROZEN iter1_ema each iteration; compare SLOPES):
- **Arm A (control):** mirror — iter1_ema vs iter1_ema (`configs/e22_mirror.yaml`).
- **Arm B (treatment):** cross — iter1_ema vs **sym15** (`symmetric-15ch-iter1-ema`, ~27% so iter1 wins ~70-75% = decisive-not-saturated + decorrelated style) (`configs/e22_cross.yaml`).
- Held constant: **200 sims** ([[e19]]: depth isn't the lever — keep cheap/constant to isolate the opponent), 5 iters, disc_weight 0, E16 off, gate 0.55, learner-only color-balanced data.

## Outcome
**DONE — FAILED.** The **cross arm DEGRADES** vs frozen iter1_ema (**−4.5 pp/iter: 51.7 → 33.3**) while **mirror treads water (+1.2)**. The decisive-game signal **corrupted the POLICY** (overfit to beating sym15); the **value head was unchanged**.

The single crucial detail: the treatment's decisive signal did not sharpen the value head as hypothesized — it overfit the *policy* to beating a specific weaker opponent, which is the "I'm beating a *weaker* opponent" distribution-shift failure the H2H gate was designed to catch. The cross-teacher hypothesis is falsified for this loop. This is one of four NOT_STRONGER swings (with [[e19]], [[e24]]) that "poked the mirror-continuation self-play loop while keeping the same value head" (post-E24 board, 2026-06-07).

## Details
**Decision gate (pre-registered):** Arm B climbs (slope up, ideally crossing >55% vs frozen iter1_ema) AND beats Arm A's slope → decisive-outcome signal is the lever → scale it / fold into the big run. Both flat → the value-head plateau isn't a *data-signal* problem → escalate to architecture (value-head redesign) or [[e21]] ensemble-teacher (a manufactured better target). **Honest risk the H2H tests (which materialized):** the value head may learn "I'm beating a *weaker* opponent" (distribution shift) rather than "good position" — wouldn't transfer to equal play.

**Follow-up value-head diagnostics** (`scripts/value_head_calibration.py`, `scripts/value_head_finetune_probe.py`) showed the value head is **FROZEN by the self-play loop AND near a data/arch ceiling** (AUC ~0.74; supervised fine-tune overfits in 1 epoch). **BUT** this only indicts the *cautious micro-loop* (lr 1e-5, ~1K games) — NOT self-play at scale, which we never ran. Next direction = a real self-play campaign (real LR + scale + staged kill gates) — became [[e24]].

**Implementation (DONE, branch `e22-cross-teacher`, validated):** no two-model support existed — `self_play.py::play_game_worker` loaded one net for both sides. Added:
- an `opponent_model_path` knob (config → `run_training.py` mode_settings whitelist → supervisor → worker),
- a second net+MCTS per worker (own GameState pool),
- per-side routing in `_run_game_loop_inner`,
- and — the validity-critical part — ONLY the learner's positions stored, color-balanced by game parity.
- Backward-compatible (opponent unset = the old mirror path, byte-identical).
- Tests: `yinsh_ml/tests/test_cross_teacher.py` (3/3 — no opponent-position leakage, correct color, mirror unchanged); existing MCTS suite 35/35; real two-net smoke green (decisive games W1-B3, color-balanced WHITE/BLACK by parity, checkpoint NaN-clean).
- **A bug the smoke caught that the unit test couldn't:** `opponent_model_path` was missing from `run_training.py`'s mode_settings whitelist, so it silently ran as mirror — fixed.
- **Aside:** `epsilon_mix_iteration_start/end` are also absent from that whitelist → ignored since forever, incl. [[e19]]; minor, out of scope.
- **Supersedes the old E7b stub.**

**Launch:** re-rent a box (≥160 cores / 1×GPU≥16GB), `git checkout e22-cross-teacher`, scp seeds, `PY=/venv/main/bin/python bash scripts/e22_dualarm.sh`.

## Provenance & links
- Source snapshot: 2026-06-01 scope; RAN/result 2026-06-03.
- Related: [[e19]] (its verdict chose E22), [[e21]] (the escalation target if E22 flat), [[e23]] (the league scale-up, gated on E22 climbing — DROPPED when E22 declined), [[e24]] ("real self-play campaign" follow-up the E22 diagnostics pointed to).
- Cross-doc: memory `project_e18_e19.md`; full detail logged in memory.
