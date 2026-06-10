# Analysis board — product backlog

Feature / product ideas for the analysis board (`server.py` + `static/`).
Distinct from the model-training `EXPERIMENT_BACKLOG.md` — this is about the
*board as a product*, not about beating iter1 in H2H.

---

## Deploy the E26-distilled net as the FAST-tier / hint engine  `proposed 2026-06-10`

**What:** Expose `models/e26_distilled_*/…pt` as a board model and use it for
the **low-latency / few-sims** path (fast difficulty tier, instant move hints),
*not* for the strong/analysis tier.

**Why (measured, not hypothetical):** The E26 policy-distilled net has a much
better *policy prior* than iter1, but search washes the edge out. Color-balanced
H2H vs iter1 across a fixed budget sweep (2026-06-10):

| sims | distilled vs iter1 |
|---|---|
| **0 (prior-only)** | **0.65 ± 0.07** — clearly stronger |
| 48 | 0.54 |
| 64 | 0.52 |
| 96 | 0.51 |
| 200+ | ~0.50 (wash) |

So at the **instant / low-search end the distilled net is a free strength
upgrade** (better play, zero extra compute); at the board's strong/analysis
budget (≥~100 sims, deployed cap 1600) it's a wash with iter1 — *don't* swap it
in there.

**Concrete uses:**
- A **"Fast" / casual difficulty tier**: instant moves that are meaningfully
  stronger than iter1's instant moves, no added latency.
- **Move hints / suggestions** that must be instant: surface the distilled net's
  top move (no search) — it's now a better suggestion. (Pairs with the puzzle
  fields the teacher corpus now stores — see `EXPERIMENT_BACKLOG.md` "YINSH
  puzzle mode".)
- **Cheaper serving** of casual play — strength without paying for search.

**Caveat:** only an upgrade at low search. Keep iter1 (or distilled@high-sims,
equivalent) for the strong/analysis tier. The board already supports per-request
`(model_id, num_sims)`, so this is wiring + a tier label, not new infra.

**Provenance:** E26 derisk, 2026-06-10. The "distillation moves the policy, not
the ceiling" finding — see `EXPERIMENT_BACKLOG.md` E26 + `TECH_DEBT.md` §7
(the H2H-budget caveat that hid this).
