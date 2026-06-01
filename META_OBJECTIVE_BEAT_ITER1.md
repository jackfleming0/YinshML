# Meta-Objective: Beat iter1_ema

> **Hand this to a new session before it touches training.** It is the north-star and
> the operating rules — distilled from a string of runs that each failed the same way.
> Detailed experiment write-ups live in [`EXPERIMENT_BACKLOG.md`](EXPERIMENT_BACKLOG.md);
> this doc is the *why* and the *rules*, kept short on purpose.

## Why this doc exists

We keep getting bogged down in competing experiments (E6, dropout fix, symmetry reg, E10
corpus, E16, E2 …) and losing the plot. This is the anti-sprawl charter. Before proposing
or running anything, it must serve the one objective below and obey the rules below. If it
doesn't, it's a distraction — park it in the backlog and move on.

## The objective (one sentence)

**Produce a model that wins a color-balanced head-to-head against `iter1_ema_2026-05-27`.**

That's it. Not "fix the opening," not "raise value discrimination," not "make it symmetric."
Those are proxies. **iter1_ema is the undefeated champion** — every intervention to date has
*lost* to it (see scoreboard). The single bar for success is beating it in actual games.

## Operating rules (hard-won; each cost a run)

**R1 — The only metric that counts is H2H vs the fixed champion iter1_ema.**
*Why:* the within-lineage tournament is blind to absolute strength. *Evidence:* it cheerfully
promoted all 5 iterations of the symmetry run (45.5/50.0/46.0/46.5% per step), a model that
loses **80-20** to iter1_ema. Anchor-vs-heuristic saturates at ~100% and can't differentiate
either. Internal Elo is relative. Cash everything out in H2H vs iter1_ema.

**R2 — Always continue from iter1_ema. Never rebuild from scratch.**
*Why:* from-scratch throws away the strongest weights we have. *Evidence:* the symmetry run
pretrained from scratch on a diversified corpus → weaker base (empty-board policy peak 1.7%,
PAcc 0.324) that 5 iterations of self-play never recovered → 20% H2H.

**R3 — One variable per experiment.**
*Why:* we have *never run a clean ablation* — every run changed 3+ things at once, so none could
attribute cause. *Evidence:* the symmetry run simultaneously rebuilt from scratch, set
`discrimination_weight=0`, used a 45%-random corpus, and added E16. All push "flatter/weaker";
we can't say which dominated. Change exactly one thing, measure, then the next.

**R4 — Color-balance every H2H.**
*Why:* there is a strong second-player (black) advantage in-engine. *Evidence:* in the symmetry-run
H2H, sym15 scored 7% as white but 33% as black. A single-color match is misleading by ~25 pp.
Run both colors (30+30), combine.

**R5 — Throughput is a multiplier on the per-iteration improvement rate, not a cure for a zero rate.**
*Why:* scaling a process that isn't improving buys a more confident plateau at the same Elo, for
more money. *Evidence:* every run that added iterations/data tread water or regressed. Prove a
positive learning rate at *small* scale, THEN scale throughput to amplify it.

**R6 — Depth before volume.**
*Why:* the highest-prior plateau-break lever is search depth, not game count. At 200 sims the
MCTS-improved policy barely exceeds the raw net → training target ≈ the net → no gradient.
Deeper search (800–1600) makes the target meaningfully stronger → real signal. *Evidence:* P1
diagnostic confirms amplification exists (KL 3.46 nats, 96→800 sims). **Throughput's job is to
afford depth, not more shallow games.**

**R7 — Symmetry is an inference-time property, already solved (E8). Don't spend training on it.**
*Why:* test-time D2 averaging (symmetric MCTS) makes any model's opening symmetric AND adds
main-game strength. *Evidence:* validated on iter1_ema — opening D6 concentration 0.857→0.214,
white WR 48%→54%. The friend-tester's cosmetic complaint never needed a training run.

**R8 — Strength is the only judge; proxies lie.**
*Why/Evidence:* sharper policy ≠ stronger (the dropout fix sharpened policy but collapsed to
over-confidence, WR 6%). Lower asymmetry ≠ stronger (symmetry run was symmetric and lost 80-20).
Higher discrimination ≠ stronger. Never declare victory on a proxy.

**R9 — Prove the lever at small scale before amplifying it with compute.**
Sequence is fixed: **hypothesis-test → infra-invest → scale.** Don't build throughput for a
mechanism you haven't shown moves the H2H.

**R10 — Tighten the promotion gate (0.20 → 0.55).**
*Why:* a 0.20 gate lets `best_model` random-walk sideways onto non-improving models. "Best"
should only advance on a real edge.

## Scoreboard: the champion is undefeated

| Challenger | vs iter1_ema | Verdict |
|---|---|---|
| E6 (continued pretrain on H-vs-H) | 8-22 | lost |
| dropout-fix lineage | over-confidence collapse, self-play WR 6% | lost |
| symmetry run (`symmetric-15ch-iter4-ema`, 2026-06-01) | **12-48 (20%)** | lost |
| **iter1_ema_2026-05-27** | — | **CHAMPION** |

Five interventions, zero wins. The plateau is not "iter1→iter2 won't climb" — it is "nothing
we have built is stronger than a pre-dropout-fix, pre-symmetry 05-27 model."

## The current plan (detail in EXPERIMENT_BACKLOG.md)

- **E18 — Deploy symmetric MCTS (E8/L3a).** Built + validated, needs `git push` + `yinsh-redeploy`.
  Closes the friend-tester loop today, no training. *Do first.*
- **E19 — Ablation ladder from iter1_ema, H2H-gated.** Rung 1: depth only (sims 200→800). Rung 2:
  + dropout-off + label-smoothing. Rung 3: + `discrimination_weight` on + E2. One rung at a time,
  H2H vs iter1_ema each rung. First rung that wins is the lever. *The real prize.*
- **E20 — Throughput (shared evaluator + bitboards) in service of depth.** Gated on E19 showing a
  real learning rate. (Self-play is hard CPU-bound: 5090 at ~32% mean util, 40% of wall-clock idle.)

**Decision gate:** a rung beats iter1_ema → scale it (E20). All rungs stall → the plateau is
structural; bank iter1_ema + E8 as the product and consider a deeper change (equivariant net E17,
value-head redesign, MuZero-style) rather than more of the same.

## Operational facts a new session needs

- **Champion:** `models/iter1_ema_2026-05-27/iter1_ema.pt` — 15ch enhanced, policy_out 7433.
- **H2H harness:** `scripts/measure_h2h.py` — `NetworkWrapper` auto-detects 6/15-ch per side, so
  cross-encoder matches work as-is. Run two invocations with `--white`/`--black` swapped (30 each)
  and combine for color balance. Decode space is the shared 7433-move table.
- **All recent checkpoints are policy_out 7433** (post the 8390→7433 move-encoding fix) → loadable.
  Pre-fix (8390) checkpoints hard-fail to load — don't bother with them.
- **Rented box:** vast.ai 5090; interpreter `/venv/main/bin/python` (NOT `venv/bin/activate`).
  Launch the next box with `--cap-add SYS_PTRACE` so py-spy live-profiling works.
- **Model naming:** rename pulled checkpoints descriptively (`<approach>-<enc>-<stage>.pt`), log in
  the runbook Model Registry — same-name collisions are a known pain.

## What "done" looks like

A model with a **>55% color-balanced H2H vs iter1_ema**, promoted under a 0.55 gate, deployed with
symmetric MCTS on. Until then, iter1_ema is still the product.
