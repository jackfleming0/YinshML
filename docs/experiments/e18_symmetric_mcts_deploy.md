# E18 — Deploy symmetric MCTS (L3a/E8) to the analysis board

**Status:** QUEUED (code-complete, validated, not yet live)
**Date(s):** scoped 2026-06-01; still queued as of 2026-06-09
**Cost:** ~1h, no training
**Branch / artifacts:** `server.py::_symmetric_search_batch` (sync+async); env flag `YNS_SYMMETRIC_MCTS` (default-on); UI toggle. Deploy = `git push` + `yinsh-redeploy` (Mac mini).

## Description
Deploy the symmetric-MCTS inference path (E8 / L3a) to the analysis board. The symmetry goal is solved at **inference** by averaging one net's policy+value over the 4 D2 transforms at each leaf — so we should **stop spending training on it**. Closes the friend-tester loop (the *original* trigger of the whole symmetry effort) with zero new training and a free main-game strength bump.

## Outcome
Not yet concluded — QUEUED. It is code-complete and validated, just not deployed. Decision gate: none beyond the deploy action itself; this is a "do first" cheap free win. Validated on the deployed `iter1_ema`: opening D6 concentration **0.857 → 0.214** (symmetrized), and iter1 win-rate **48% → 54%**. The signal that it's working post-deploy is the same opening-concentration drop and the WR bump showing up live.

## Details
- E8/L3a is **code-complete and validated, not yet live.** `server.py::_symmetric_search_batch` provides both sync and async paths, is **default-on via `YNS_SYMMETRIC_MCTS`**, and exposes a **UI toggle**.
- Validation on deployed `iter1_ema`: opening D6 concentration **0.857 → 0.214**; iter1 WR **48% → 54%**.
- Action: `git push` + `yinsh-redeploy` (Mac mini).
- Rationale: closes the friend-tester loop (the *original* trigger) with **zero new training** and a **free main-game strength bump**. The symmetry goal is solved at **INFERENCE** — stop spending training on it.
- **Residual:** A5 is still ~40% of orbit post-symmetrization — the "residual 25%" weight-symmetry question (E11/E16) is a separate, lower-priority, strength-neutral cosmetic item.
- From the post-E24 lever board (2026-06-07): E18 is **Lever B — test-time compute = ship NOW, free strength** (deploy symmetric MCTS, +6–22 pp, no training), paired with simply playing at higher MCTS budget. Raises *played* strength immediately; **do regardless** of the other experiments' verdicts.
- Recommended sequence (2026-06-09): ship **E18 + higher play-time sims** (free) → run E25 (~a day) → commit big chips to E26.

## Provenance & links
- Source snapshots: 2026-06-01 (symmetry-run verdict, E18 scope), 2026-06-07 (post-E24 lever board, Lever B), 2026-06-09 (recommended sequence).
- Related: [[e08]]/L3a (symmetric MCTS implementation it deploys), [[e19]] (the symmetry foundation run whose failure reframed symmetry as an inference-time problem), [[e26]] (the other surviving lever, gated on E25).
- Cross-doc: memory `project_e18_e19.md` (E18 LIVE+verified noted there in a later state; this backlog entry reflects the QUEUED snapshot).
