# E25 — sharkdp benchmark + the value-head ceiling (2026-06-08)

**Standalone write-up.** Two linked investigations from one session:
1. Integrated a second external engine (**sharkdp/yinsh**) and benchmarked the
   engine field against our champion `iter1_ema`.
2. Followed the result into a corpus hypothesis, which led to a decisive
   diagnostic on **why iter1 has plateaued** — and ruled out the value head as
   the lever.

**One-sentence verdict:** our NN champion already dominates every available
classical engine, and the thing blocking further progress is **not** value
discrimination (it's at a hard ~0.70–0.74 AUC ceiling that no retraining or
spread-rich data moves) — so the lever is search/policy/encoding, not a better
teacher corpus.

---

## 0. Why this happened

Post-E24 we're stuck: continued self-play from `iter1_ema` doesn't improve it
(e24 phase1a: every LR treads water or collapses; lr3e-4 iter2 = 32% H2H vs a
frozen iter1). The E25 branch is a binding-constraint hunt. This session started
from a tangential question — "are there other competitive YINSH engines we could
benchmark/learn from besides yngine?" — and the answer cascaded into the
plateau diagnosis.

---

## 1. sharkdp/yinsh — a second benchmark engine

### What it is
[sharkdp/yinsh](https://github.com/sharkdp/yinsh) — David Peter's Rust engine
(author of `bat`/`fd`/`hyperfine`), actively maintained, clean and
benchmark-minded. Algorithm: **negamax + alpha-beta** (the `minimax` Rust crate)
over a compact **5-feature `SimpleHeuristic`**:

| feature | weight |
|---|---|
| points (rings removed) | 10000 |
| markers on board | 100 |
| controlled markers — own | 5 |
| **controlled markers — opponent** | **10** |
| accessible fields (ring mobility) | 1 |

Note the **defensive asymmetry**: controlling opponent markers is weighted 2× own —
a genuine positional/defensive term our 7 all-attack/all-differential features
lack. That's why it was worth integrating even before measuring strength.

The broader engine landscape is thin: most other GitHub YINSH repos are
abandoned IIT-Delhi course minimax projects; CodinGame has a Yinsh bot
leaderboard worth mining later; Boardspace's BestBot is closed-source.

### How it's wired (mirrors the yngine bridge pattern)
- Clone at `third_party/sharkdp_yinsh/` (embedded git repo; ~120 MB `target/`,
  not tracked by the parent repo). Rust installed via rustup.
- New driver crate `crates/yinsh_driver` → binary `yinsh-driver`, speaking the
  **exact same line wire protocol** as `third_party/yngine_driver`, so the
  Python side reuses `yinsh_ml/yngine/move_codec.py` verbatim. Build:
  `cd third_party/sharkdp_yinsh && cargo build --release -p yinsh_driver`.
- Python bridge `yinsh_ml/sharkdp/bridge.py::Sharkdp` subclasses `Yngine`;
  only swaps the binary path + the search command (`go depth N`, since sharkdp
  searches to a fixed negamax depth, not an MCTS sim budget). `SHARKDP_DRIVER`
  env overrides the binary.

### The coordinate bijection (the hard part)
sharkdp uses hex `Coord{x,y}` on −5..5 (circle test); the wire uses yngine's
`(x,y)` on 0..10. Found and verified computationally — both are the standard
85-point board, so a hex symmetry aligns them:

```
x_wire = x_shark + 5      x_shark = x_wire − 5
y_wire = 5 − y_shark      y_shark = 5 − y_wire
```

It's a line-preserving symmetry (preserves ring slides and 5-in-a-row runs).
Direction codes map per a fixed table (a shark delta `(dx,dy)` → wire delta
`(dx,−dy)`). sharkdp also splits a logical ring move into `PlaceMarker` +
structural `Wait` + `MoveRing` plies; the driver recombines them into one wire
`M`. A subtlety: sharkdp scores the point at `RemoveRun` and sets `winner()`
*before* the follow-up ring removal, so on an already-decided position the driver
falls back to first-legal-move (else `Negamax::choose_move` unwraps `None` and
panics).

### Parity validation
`scripts/smoke_sharkdp_bridge.py --mode referee` plays two sharkdp engines
against each other with **our `GameState` refereeing every move**. 3 full games
completed with every move round-tripping and both rule engines agreeing
(coordinate bijection + direction mapping + turn bridging + win handling all
correct). **PASS** — the bridge is trustworthy.

---

## 2. Benchmark results — the engine pecking order

```
iter1_ema (NN, 200 sims)  ≫  yngine (MCTS 1000)  >  sharkdp (negamax d6)  >  HeuristicAgent
```

| Match | Result | WR | Tool |
|---|---|---|---|
| iter1_ema vs sharkdp d6 | **20–0** (10W/10B) | 1.000, CI95 [0.839, 1.0] | `scripts/eval_vs_sharkdp.py` |
| yngine s1000 vs sharkdp d6 | **26–14** for yngine | sharkdp 0.350 | `scripts/eval_engine_vs_engine.py` |
| sharkdp d6 vs HeuristicAgent | **6–0** for sharkdp | — | `scripts/smoke_sharkdp_bridge.py` |
| sharkdp d4 vs HeuristicAgent | **4–0** for sharkdp | — | (longer games — grinding) |

Takeaways:
- **iter1_ema utterly dominates the classical field** at just 200 sims/move.
- **sharkdp is weaker than yngine** at typical settings (operating-point caveat:
  d6-vs-s1000 is arbitrary; sharkdp at d8/d10 might edge it). This is decisive
  for the corpus question below.
- sharkdp > our HeuristicAgent — its defensive term is real, but it doesn't
  close the gap to the NN.

Each NN game is ~3–4 min at 200 sims on MPS (model MCTS is the cost; the engines
are instant). sharkdp "depth" counts sub-plies (~2 per logical move) so direct
depth-matching is fuzzy.

---

## 3. The corpus hypothesis (and why it was the wrong joint)

The natural idea: iter1 was warm-started on a 200K-game **yngine** corpus and
then surpassed its teacher via self-play. If a *stronger* teacher gave a higher
warm-start floor, maybe we'd get a congruent lift — so build a corpus from
sharkdp (or a yngine×sharkdp blend) and pretrain a stronger base.

Why it doesn't work as framed:
1. **sharkdp is weaker than yngine** (§2). A sharkdp corpus floors *below* the
   yngine corpus we already used — a downgrade, not an upgrade.
2. **Behavioral cloning caps at the teacher.** iter1 beat yngine because of the
   self-play RL phase, not the corpus. Any corpus from an engine weaker than
   iter1 anchors a fresh model *below* iter1. The only teacher stronger than
   iter1 is **iter1 itself searched harder** (high-sim self-distillation).
3. The "relabel positions with a strong engine" variant is **already the status
   quo** — `trainer.py` Fix #1 uses MCTS root values (not raw ±1 outcomes) as
   value targets. That experiment is a no-op.
4. A prior engine-labeled-corpus attempt (`gen_engine_labeled_corpus.py`) tanked
   iter1's value AUC to **0.575 (worse than blind)** because its *positions*
   were myopic-engine OOD. Cloning engine positions repeats that.

The only salvageable corpus rationale was **diversity** (sharkdp's different
defensive style as a plateau-breaker). To test whether more/different data could
help at all, we went after the actual constraint: the value head.

---

## 4. The value-head ceiling investigation

### The setup
The value head's discrimination on a decisive corpus (the human H-vs-H corpus,
`expert_games/hvh_full_game_15ch.npz`) is stuck at **AUC 0.737**
(`value_head_calibration.py`; AUC = P(value ranks a win above a loss); 0.5 =
blind). Question: can it be improved by *training on decisive, spread-rich data*,
or is 0.737 a hard ceiling?

Probe tool: `scripts/value_ceiling_probe.py`. Splits the (game-ordered) corpus
**contiguously** train/test with a gap, to avoid same-game leakage (a random
split leaks: consecutive positions from one game would land in both sides and
inflate test AUC).

### Finding A — 0.737 is a *supervised-era* ceiling
`best_supervised` and `iter1_ema` have **identical** value AUC (0.737). The
entire self-play lineage never moved value discrimination at all. (Human
*policy*-finetuning made it worse — 0.611 — because it trained policy, not value.)

### Finding B — frozen features can't be read past ~0.70
Retraining **only the value head** on iter1's frozen features (full 85k decisive
positions):

```
zero-shot test AUC 0.696  →  train AUC 0.86→0.97 (overfits), test AUC 0.70→0.66 (DECLINES)
```

The head has plenty of capacity (memorizes train), but iter1's representation
doesn't carry **generalizable** win/loss signal past ~0.70.

### Finding C — retraining the *trunk* doesn't help either
**Full-net** retraining (trunk + head, weight decay), on both a 6k CPU subsample
and the full 85k on MPS:

```
                      train_auc       test_auc (best)
6k  (CPU):  ep1→4   0.77 → 0.83    0.704 0.699 0.701 0.701                  (flat)
85k (MPS):  ep1→8   0.84 → 0.98    0.694 0.700 0.696 0.695 0.694 0.667 0.672 0.674
                                   (best 0.700 @ep2 = zero-shot, then declines)
```

Train AUC climbs to near-perfect memorization (0.98), held-out AUC is **pinned at
~0.70** regardless of data size and *declines* in later epochs as the overfit
deepens. The best held-out AUC ever reached (0.700) is indistinguishable from the
zero-shot value (0.696).

### Summary table

| Probe | data | train AUC | held-out AUC |
|---|---|---|---|
| iter1 zero-shot | — | — | ~0.70–0.74 |
| `best_supervised` | — | — | 0.737 (identical to iter1) |
| frozen head retrain | 85k | 0.97 | declines to 0.66 |
| full-net retrain | 6k | 0.83 | flat ~0.70 |
| full-net retrain | 85k | 0.98 | best 0.700, declines to 0.67 |

**Nothing moves held-out value discrimination off ~0.70–0.74** — not head-only,
not full-trunk, not more data, not decisive spread-rich data.

---

## 5. Conclusions

1. **Value discrimination is ruled out as the lever to beat iter1.** It's at a
   hard ceiling that is **representational/intrinsic, not a data-spread problem.**
   Generating spread/diversity corpora to sharpen the value head would hit this
   exact wall — so the corpus instinct from §3, even in its strongest (diversity)
   form, is the wrong joint.
2. **~0.74 is plausibly near the irreducible (Bayes) ceiling** for static YINSH
   position value — a balanced abstract game where one tempo flips the result is
   genuinely hard to value from a single position. Under that reading the
   "frozen" value head is a **solved sub-problem**, not a bug.
3. **iter1's strength comes from search + policy, not value sharpness** — it
   sweeps every engine at only 200 sims. When the value signal is saturated,
   MCTS leans on the policy prior.

### Where the lever actually is
- **Search budget + policy quality** — more sims at play; sharpen the policy head
  (the part MCTS actually exploits when value is saturated).
- **Richer encoding / capacity** — the only way to break a *representational*
  value ceiling is a better state representation, not more/different data.

---

## 6. Caveats (honest)

- The ceiling probes used the **human corpus** (limited size, somewhat off the
  NN's own distribution). The fully clean confirmation relabels iter1's *own*
  representative positions with deep-search values (the "circular" worry is
  unfounded — search value ≠ raw value head — but it's a bigger build).
- ~0.74-as-Bayes-ceiling is a *hypothesis*; the intrinsic-ceiling check below
  is what would confirm it.
- sharkdp benchmark operating points (d6 vs s1000) are arbitrary; a depth sweep
  would calibrate sharkdp into a graded yardstick.

---

## 7. Open follow-ups (ranked)

1. **Intrinsic-ceiling check** (cheap, decisive): train a value head *hard* from
   scratch on a large decisive corpus (`yngine_volume` / `boardspace_human_*`)
   with a proper **game-split**. If even that caps ~0.74 → the ceiling is the
   **encoding**, and the lever is a richer representation. If it clears 0.80 →
   the encoding is fine and iter1's trunk just never learned it (reopens a
   narrow corpus angle).
2. **Search + policy levers** — the redirect implied by §5.
3. **sharkdp depth sweep** (d8/d10) to make it a graded benchmark; mine the
   CodinGame leaderboard for a second competitive opponent.

---

## 8. Reproduction

```bash
# Build sharkdp engine + driver
cd third_party/sharkdp_yinsh && cargo build --release -p yinsh_driver && cd -

# Bridge parity check
python scripts/smoke_sharkdp_bridge.py --mode referee --games 3 --depth 4

# Benchmarks
python scripts/eval_vs_sharkdp.py --model-path models/iter1_ema_2026-05-27/iter1_ema.pt \
    --num-sims 200 --num-games 20 --depth 6 --output logs/iter1_ema_vs_sharkdp_d6_sims200.json
python scripts/eval_engine_vs_engine.py --engine-a sharkdp --a-depth 6 \
    --engine-b yngine --b-sims 1000 --games 40

# Value-head ceiling probes
python scripts/value_head_calibration.py --data expert_games/hvh_full_game_15ch.npz --n 8000 \
    iter1_ema:models/iter1_ema_2026-05-27/iter1_ema.pt \
    best_sup:models/supervised_2026-05-27/best_supervised.pt
python scripts/value_ceiling_probe.py --ckpt models/iter1_ema_2026-05-27/iter1_ema.pt \
    --epochs 8 --lr 1e-3                                   # frozen head
python scripts/value_ceiling_probe.py --full-net --device mps \
    --epochs 8 --lr 1e-4 --wd 1e-4 --batch 1024           # full net (needs MPS free)
```

### Artifacts / files
- Engine: `third_party/sharkdp_yinsh/` (crate `crates/yinsh_driver`)
- Bridge: `yinsh_ml/sharkdp/` (`bridge.py`)
- Scripts: `scripts/eval_vs_sharkdp.py`, `scripts/eval_engine_vs_engine.py`,
  `scripts/smoke_sharkdp_bridge.py`, `scripts/value_ceiling_probe.py`
- Results: `logs/iter1_ema_vs_sharkdp_d6_sims200.json`,
  `logs/sharkdp_d6_vs_yngine_s1000.txt`, `logs/value_ceiling_fullnet*.txt`
- Related: `docs/experiments/completed/e24_phase1a_results.md` (the plateau this explains)

---

## 9. Binding-constraint diagnostic — on-distribution value-eval + head ablation (2026-06-09)

A separate E25 thread ran the two probes from the backlog's E25 entry on
`iter1_ema`. Both **corroborate §5** and close the open caveat in §6.1 — the
ceiling in §4 was measured on the *human* corpus; this measures it on the NN's
**own** positions.

### 9a. On-distribution value-eval — closes §6.1, and the ceiling is *lower* on strong play
Generated iter1's **own** representative positions — 200-sim neural-MCTS self-play,
8000 outcome-labeled decisive positions (`scripts/gen_selfplay_labeled_corpus.py`,
held-out by construction) — and re-ran `value_head_calibration.py`:

| corpus (n=8000, decisive) | AUC | sign-acc | corr |
|---|---|---|---|
| clean 200-sim self-play (NN's own dist.) | **0.663** | 0.600 | 0.316 |
| human H-vs-H (the §4 corpus) | 0.737 | 0.646 | 0.378 |

On the NN's own strong-play distribution the value head reads **0.663 — *below* the
0.737 human number, not above.** So the ceiling is **not** a human-corpus artifact;
it holds, and is *lower*, on home turf. Mechanism: stronger play → balanced games →
the loser sits in neutral-looking positions before losing → the win/loss boundary
blurs in feature space. Direct support for the **"~0.74 ≈ Bayes ceiling, one tempo
flips the game"** reading in §5.2 — strong play makes outcomes *less*
position-determined, so the relevant-distribution ceiling is nearer 0.66 than 0.74.

**Methodological caution for the §7.1 intrinsic-ceiling check:** a 50-sim corpus
(weaker play) gave an inflated **0.796** — weak play makes lopsided, easy-to-rank
games. *Corpus play-strength drives the measured AUC*, so the intrinsic-ceiling
probe must fix (and prefer) strong play, or it will over-report the ceiling.

### 9b. Head ablation — the positive leg behind "MCTS leans on policy" (§5.3)
Neutered one head at a time in MCTS (new `ablate_policy` / `ablate_value` flags on
`self_play.py::MCTS`, default off) and played color-balanced H2H vs full, 60
games/pairing @ 200 sims (`scripts/e25_ablation_h2h.py`):

| pairing | full's score | W–L–D |
|---|---|---|
| full vs flatpolicy (uniform prior + real value) | **0.90** (±0.08) | 54–6–0 |
| full vs blindvalue (real prior + 0 value) | **1.00** | 60–0–0 |

**Both heads are necessary; neither is a free sacrifice.** This is the *positive*
complement to §4's negative result: §4 showed the value head can't be *improved*;
9b shows the **policy is load-bearing** — directly confirming the §5.3 inference
that MCTS leans on the policy prior. Strength lives in policy + search, not value.

**Caveats (honest):** `blindvalue` = constant-0 is a *biased* cut ("every position
is a draw" is actively wrong in a decisive game), so 60–0 **overstates** value's
importance vs a genuinely-uninformative random-value arm (a worthwhile refinement
if this number ever turns load-bearing). And ablation measures **necessity, not
headroom** — it confirms the current value head matters, not that a better one
would help (§4 already answers that: it wouldn't).

### 9c. Net
No change to §5's direction — **reinforced from a second, independent angle.** Value
is out (now confirmed on the NN's own distribution); policy is provably the binding
head; the levers remain **search budget + policy quality** and **encoding /
capacity**. The one forward consequence for E26: aim its distillation at the
**search-improved policy** (the strong, improvable signal), not value targets
(saturated). New tooling: `scripts/gen_selfplay_labeled_corpus.py`,
`scripts/e25_ablation_h2h.py` (+ MCTS `ablate_policy`/`ablate_value` flags),
`docs/experiments/e25_ablation.json`.
