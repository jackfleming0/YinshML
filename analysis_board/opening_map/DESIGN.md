# Opening Map — design doc

**Status:** captured, not yet implemented. Sibling to `analysis_board/loop/`
(per-position inference). This work is per-game-played and only makes sense
once the position-loop work has produced a couple rounds of training findings
worth comparing across.

## Goal

Empirically characterize YINSH openings — which placement configurations
lead to which outcomes — in a way that **isolates opening quality from play
quality**. The naive "compute win rate per opening in existing self-play"
conflates these: a "bad" opening might lose at 40% only because no model has
learned to play it well, and could in principle be fine under stronger play.

## Confound to disentangle

> Wall-cluster openings might beat center-cluster openings under bad play,
> and lose to them under great play. We don't know which we're measuring
> when we look at raw self-play stats.

This isn't a small confound — it's the *primary* confound. Any "opening X is
better than opening Y" claim from a single model's self-play is suspect.

## Experimental design

The core control: **same model on both sides**, so any systematic outcome
asymmetry across openings reflects something about the opening, not about
play asymmetry.

### Three-axis sweep

1. **Opening variant** — ~20 representative openings (see *Variant selection*
   below). Each represents a "style cluster" (center-heavy, wall-hugging,
   diagonal-spread, etc.).
2. **Model strength** — at least 3 models of increasing strength. Suggested:
   - random-policy baseline (cheap, sets the floor)
   - heuristic agent at depth 3
   - `yngine_volume_15ch_pretrain/best_supervised.pt` (post-fix anchor)
   - `yngine_volume_15ch_pretrain_b1b2b3_iter4/iter4_ema.pt` (post-B1B2B3)
   - any future stronger model
3. **Game replicates** — 500-1000 games per (opening × model) cell.

The *gradient across model strength* is the most informative signal:
- Opening sits at 40% across every model → genuinely structurally worse.
- Opening climbs from 35% → 50% with stronger models → "needs skill" type.
- Opening stays even across models → robust / equivalent.

### Variant selection — cluster, don't sample

The opening space is ~85^10 ≈ 10^19. Random sampling lands mostly on
indistinguishable variants. Instead:

1. Take placement-completed positions from existing replay buffers (`states`
   field, filter to `phase == MAIN_GAME` and `move_number == 10`).
2. Extract a fixed-length feature vector per position. Two options:
   - Use the network's penultimate-layer activation (semantic embedding).
   - Hand-engineered features: centroid distance from board center,
     ring-pair distances, wall-vs-center ratio, symmetry score.
3. K-means with K=20-40 clusters.
4. Take each cluster centroid (or the closest-to-centroid real position) as
   a variant.

This gives variants that span the *observed* opening landscape, weighted by
naturalness. We could also deliberately add a few hand-picked variants that
the network *doesn't* explore (e.g., a corner-cluster opening that no model
plays) — those are the most interesting for "needs skill" detection.

### Metrics per (opening × model) cell

Don't reduce to a single win rate. Three numbers:

- **Win rate** (binary outcome, Wilson CI). Headline.
- **Mean game length** (in moves). Shorter = more decisive opening.
- **Mean value_cost of misalignment** across the resulting MAIN_GAME
  positions (from the position-loop metric). Lower = the opening leads to
  positions the network can play; higher = the opening leads to positions
  where the network's policy is wrong even though search can find the
  answer. This is the most novel signal — it measures *how hard the opening
  is to play* in addition to whether it wins.

### Critical confound: first-mover advantage

YINSH almost certainly has a first-mover advantage (placement order
1-2-1-2-…-1-2-1-2-1 — five for each side, but white goes first AND tenth).
A 50/50 opening is *not* balanced — it's "opening that neutralizes the
first-mover advantage."

Baseline: measure first-mover win rate on uniform-random openings across
each model. Then report opening results as **Δ-from-baseline**:

    opening_score = (white_winrate_in_opening) - (white_winrate_baseline_for_model)

Now positive = opening helps white, negative = opening helps black, zero =
opening is first-mover-neutral.

## Practical scope (first cut)

- 20 variants × 4 models × 500 games = 40K games.
- Heuristic-agent throughput is ~50 games/hour serial (per `yinsh_ml/viz/README.md`),
  ~400/hour with workers=8.
- Neural self-play throughput depends on sim budget per move. At 200
  sims/move on MPS, maybe 100-200 games/hour single-threaded.
- Realistic total wall time on the M-series Mac: 2-3 days for the full
  sweep. Could be faster with multiprocessing for the cheaper models.

Smaller first pass: 10 variants × 2 models (anchor + heuristic) × 200 games
= 4K games, ~12 hours overnight. Enough to validate the methodology and see
if any opening shows a striking model-strength gradient.

## Scripts to write (when we pick this back up)

```
analysis_board/opening_map/
├── DESIGN.md                  # this file
├── cluster_openings.py        # replay-buffer → 20 representative openings
├── play_games.py              # opening × model → games (parallelized)
├── analyze_outcomes.py        # game corpus → per-cell metrics + Wilson CIs
└── report.py                  # markdown summary + matplotlib charts
```

Each plays the same architectural role as the position-loop counterparts:
sample → measure → report, with run output isolated under
`opening_map/runs/<timestamp>/`.

## Where this sits in the bigger picture

The position-loop work (alignment cost, rank-of-final-best) tells us
*where the current network is wrong* — which positions, by how much. That
work directly produces training-improvement hypotheses (e.g., upweight
high-misalignment positions, adaptive sim budgeting, etc.).

This opening-map work tells us *what the opening landscape actually looks
like* — independent of any single model's preferences. It produces
different actions: opening-book design, exploration-budget allocation
during self-play, possibly a separate training procedure for placement.

Both feed back into training, but with different mechanisms and on
different time horizons. The position-loop work is "next training cycle"
actionable; the opening-map work is more like "next strategic direction."

## Open questions to resolve before starting

- **Embedding for clustering** — network penultimate layer is information-rich
  but model-dependent. Hand-engineered features are interpretable. Probably
  start with hand-engineered and use network features as a check.
- **Game outcome attribution** — when a game is captured at MAX_MOVES with
  a non-decisive score, how to score it? Treat as draw, treat as score-delta,
  exclude? Probably exclude for opening-map purposes (we want clean wins).
- **Symmetry** — YINSH has board symmetries (D2 = horizontal + vertical
  flip). Should we cluster modulo symmetry? Yes — it 4x's the effective
  sample size per cluster.
- **Are there *deliberate* hand-picked openings we should add?** E.g., the
  "obviously correct" openings the network already plays (sanity check) and
  the "obviously suspect" ones it never plays. Curated additions might be
  worth 5-10 of the 20 variant slots.
