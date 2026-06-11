# E26 — High-budget-search policy distillation  ·  `DONE · STRONGER`

**Verdict:** Distilling the search-improved *policy* of iter1_ema (teacher at
800–1600 sims) into a fresh net produces an engine that **beats frozen iter1_ema
at tournament search budgets** — ~0.59–0.63 win rate at sims ≥ 400, every CI
clear of 0.50. The gain is worth roughly a **1.5× search-efficiency multiplier
(~60–70 Elo)**. This is the first plateau-break since iter1_ema: the only verdict
that counts (H2H vs the frozen champion) finally moved.

Two secondary results fell out of the closeout:
- The strength-vs-search relationship is **U-shaped**, which retroactively
  explains why E26 first looked like a null.
- **Test-time D2 averaging adds nothing on the distilled net** (refines E8/E18,
  which validated it on iter1).

Closed 2026-06-11. Candidate = `e26_lc_full.pt` (local:
`models/e26_distilled_2026-06-10/e26_distilled.pt`). Anchor = `iter1_frozen.pt`
(= iter1_ema). All result JSONs: `docs/experiments/e26_overnight_results/`.

---

## The campaign

Reaimed 2026-06-09 from value-distillation to **policy**-distillation after E25
ruled the value head out (intrinsic AUC ceiling 0.663). Teacher generation was
accelerated by the E20 inference server (`gen_distill_corpus.py
--use-inference-server --inference-dtype bf16`, ~645 pos/min @ 800 sims on the
4090). The corpus stores per-move `policy_q` (search-improved visit
distribution) and `policy_prior` (raw net prior); the distill regresses the net
onto `policy_q`.

Learning-curve subset distills (`e26_lc_150000.pt`, `e26_lc_300000.pt`,
`e26_lc_full.pt`) confirmed **data-sufficiency from ~150k positions** — test
policy-CE was flat from 150k onward, so corpus size was never the binding
constraint. `e26_lc_full.pt` is the keeper.

---

## Result 1 — distilled beats iter1 at tournament search (the U-curve)

Distilled-vs-iter1 win rate as a function of MCTS simulations/move
(color-balanced throughout; `measure_h2h` is temperature-sampled, the frozen-anchor
SPRT path is greedy-after-opening — the two agree at every well-powered point):

| sims | distilled WR | n | harness |
|------|-------------|----|---------|
| 0 (raw policy) | **0.650** | 200 | prior-only |
| 16   | 0.407 | 150 | measure_h2h |
| 32   | 0.453 | 150 | measure_h2h |
| 48   | 0.540 | 150 | measure_h2h |
| 64   | 0.520 | 150 | measure_h2h |
| 96 (old gate) | 0.508 | 120 | measure_h2h |
| 200  | 0.547 | 150 | measure_h2h |
| 400  | 0.588 / 0.589 | 250 / 192 | measure_h2h / SPRT-STRONGER |
| 800  | 0.622 | 148 | measure_h2h |
| 1600 | 0.630 | 92  | SPRT-STRONGER |

**Shape:** raw-prior peak (0.65) → **trough ~0.41–0.51 at sims 16–96** → recovery
to a **~0.60–0.63 plateau at sims ≥ 400**. It is not monotonic and it does not
keep climbing past 400 — it plateaus.

**Why E26 first looked like a null:** the original H2H gate ran at ~64–96 sims —
*exactly the bottom of the trough*, where distilled barely ties iter1 (0.508 @
96). The "plateau / no improvement" was a measurement artifact of gating in the
valley. Read at either end of the search axis the distilled net is clearly
different: sharper raw policy (wins at 0 sims), and a stronger deep-search engine
(wins at ≥400 sims).

**Mechanistic read (hypothesis):** distillation sharpened the *policy prior*
(dominates at 0 sims). A little search lets iter1's shallow-search behaviour
catch up through the trough; only with enough tree (≥400) do the distilled net's
better leaf evaluations compound back into a lead. The trough is the regime where
distilled's prior is being second-guessed by shallow search before its evaluation
edge is online.

### Caveat — the 0.82 @ 800 was early-stop bias, not signal
The 800-sim SPRT accepted STRONGER at 23-5 (n=28) → 0.821. SPRT stops *because*
it hit a favourable streak, so a 28-game accept is an upward-biased point
estimate. The well-powered 800 measurement is `measure_h2h` at **0.622 (n=148)**,
consistent with the 400/1600 plateau. **Use 0.62, not 0.82, for sims=800.**
(General rule for this repo: SPRT win-rates from early accepts are biased toward
the boundary; only the fixed-N or near-cap SPRT estimates are unbiased.)

---

## Result 2 — search-efficiency law (~1.5×)

Asymmetric iso-strength match (`--anchor-simulations`):

| match | result | n |
|-------|--------|----|
| distilled@400 vs iter1@400 | 0.589 (distilled wins) | 192 |
| distilled@400 vs iter1@**800** | **0.433 (distilled loses)** | 150 |

Elo arithmetic: distilled's edge at equal search ≈ **+63 Elo** (0.589). Doubling
iter1's search 400→800 is worth ≈ **+110 Elo** (it flips a 0.589 deficit into a
0.567 lead = 0.433 for distilled). So distillation bought ≈ 63/110 ≈ **0.57 of a
search-doubling → ~1.5× effective search**. Distilled@400 ≈ iter1@~600, *not*
iter1@800. Real and useful, but short of a clean 2×.

> Implication for how we gate future candidates: a single-sim-count H2H can land
> anywhere on a non-monotonic curve. Gate at tournament sims (≥400), and/or
> measure the iso-strength multiplier, rather than trusting one point — especially
> not a point near the trough.

The higher-budget confirmation (distilled@800 vs iter1@1600) was queued but cut
to save ~5h of box time; the 400-vs-800 result + the equal-search plateau already
triangulate the multiplier.

---

## Result 3 — test-time D2 averaging is null on the distilled net

Plugged a `SymmetrizingEvaluator` (averages each MCTS leaf over the 4 D2 board
symmetries; `yinsh_ml/network/symmetrized_evaluator.py`, equivariance-tested) into
the distilled net:

| match | result | n | verdict |
|-------|--------|----|---------|
| distilled+D2 vs distilled-plain @400 | 0.504 (126-124) | 250 | INCONCLUSIVE |
| distilled+D2 vs iter1 @400 | 0.580 (87-63) | 150 | ≈ plain's 0.589 |

**Test-time D2 averaging gives the distilled net nothing.** This does **not**
contradict E8/E18 (which validated D2 averaging on *iter1*: opening
path-dependence 0.857→0.214, +6–22 pp WR, deployed to the analysis board). E11
showed iter1's weights are markedly asymmetric (value drifts to 2.8× range by
move 8; policy top-1 flips in 5/6 states) — D2 averaging's value on iter1 is
mostly *correcting that asymmetry*. The most likely reason it's null here: the
distilled net, trained on the D2-augmented corpus, **already has the symmetry
consistency baked in**, so there's little left to correct, and at 400 sims the
leaf-noise-reduction benefit is already supplied by the tree.

**Cheap follow-up to confirm the mechanism:** run the E11 weight-symmetry
diagnostic on `e26_lc_full.pt`. Prediction: far less drift than iter1. If so,
"distillation-on-augmented-data ⇒ symmetric weights" is a free corollary worth
banking — it would mean the symmetric-MCTS inference path is redundant for
distilled-family models (cheaper inference on the analysis board).

Net on symmetry as a lever: **training regularizer = unvalidated/parked (E16);
test-time averaging = real on asymmetric nets (E8/E18, deployed), null on
already-symmetric nets (here).** It is not "dead" — it is "already captured" by
the distilled net.

---

## Infra finding — high-sim MCTS is CPU/tree-bound, not GPU-bound

The 1600-sim SPRT tier ran at GPU ~10–12% util with one core pegged at 100%; a
single 1600-sim game took tens of minutes (the tier took 4.8h for 92 games). At
high sims the batched MCTS loses batch width (tree dependency serializes leaf
expansion), so it becomes launch/tree-bound and the E20 inference-server win does
**not** extend to deep single-game search. Consequence: the inference server pays
off for *self-play throughput* (many parallel games, wide batches), not for
*deep search in one game*. Speeding up high-sim play needs a different lever
(tree-op optimization, larger virtual-loss batches, or C++ search).

---

## Methodology notes

- **Color-balancing** (alternate which model plays white) is mandatory; YINSH has
  a ~0.53–0.59 first-player edge *at these models' strength* (not asserted as an
  inherent game property). Pre-fix `measure_h2h` assigned the candidate white
  every game (`TECH_DEBT.md` §7, fixed).
- **SPRT early-stop bias** (see Result 1 caveat): accepts at small n overstate WR.
- **Two harnesses agree:** greedy frozen-anchor SPRT and temperature `measure_h2h`
  land on the same WR at every well-powered sim count (400: 0.589 vs 0.588), which
  is the cross-check that the U-curve is real and not a play-style artifact.

---

## Artifacts (all pulled local before box teardown)

- **Models:** `models/e26_distilled_2026-06-10/{e26_distilled.pt (=e26_lc_full),
  e26_distilled_1600.pt, e26_lc_150000.pt, e26_lc_300000.pt}`.
- **Result JSONs:** `docs/experiments/e26_overnight_results/` —
  `e26_sprt_s{400,800,1600}.json`, `d2_self_s400.json`,
  `d2_4_vs_iter1_s400.json`, `iso_400v800.json`, plus the full campaign log dir
  `box_logs/` (sweeps s16…s400, confirm, highsim, learning-curve h2h, yngine).
- **Code:** `yinsh_ml/network/symmetrized_evaluator.py` +
  `yinsh_ml/tests/test_symmetrized_evaluator.py`; `scripts/eval_vs_frozen_anchor.py`
  gained `--anchor-simulations` (iso-strength) and `--candidate-d2/--anchor-d2`
  (commit e0f1af2).

---

## Open threads / what's next

1. **e26_lc_full is the new strongest engine at tournament sims** — candidate to
   become the next frozen champion / next self-play seed. Promotion is a fresh
   H2H gate at ≥400 sims (NOT 96).
2. **Iterate the distillation loop:** e26_lc_full is now the strongest teacher at
   high sims → regenerate a corpus from *it* at 1600 sims → distill again. The
   standard AlphaZero compounding move, now with a teacher that's ~60 Elo
   stronger.
3. **E11 diagnostic on e26_lc_full** (cheap) to confirm the "distillation ⇒
   symmetric weights" corollary and retire symmetric-MCTS for distilled models.
4. **Untested:** `e26_distilled_1600.pt` (distill on the 1600-sim quality-tilt
   corpus) — strength never measured; pull-and-park.
