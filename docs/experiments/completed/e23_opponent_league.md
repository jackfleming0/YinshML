# E23 — Gap-controlled opponent league (E22 scale-up)

**Status:** DROPPED (2026-06-03)
**Date(s):** spun out of a 2026-06-02 discussion (Jack); dropped 2026-06-03
**Cost:** never built.
**Branch / artifacts:** none built.

## Description
Spun out of a 2026-06-02 discussion (Jack): if [[e22]] cross-teacher works, can we keep replacing the opponent with progressively stronger models to compound performance? Yes — and it maps onto a known pattern (opponent pools / leagues / play-vs-past-versions, à la AlphaZero/AlphaStar). But the right framing is **NOT "upgrade to a stronger teacher to imitate"** (cross-teacher is not distillation; the opponent is weaker *on purpose*). The value comes from **decisive** games, and decisiveness has a **sweet spot in the strength GAP**:
- opponent ≈ equal → ~50/50 → noise (the [[e22]] problem);
- opponent *much* weaker → learner wins ~100% → saturated → *also* no signal;
- opponent moderately weaker (~25-30 pp gap, e.g. iter1_ema vs sym15) → a *range* of outcomes → max signal.

So the ladder = **keep the gap in the decisive-but-not-saturated band as the learner climbs**: as the learner improves past iter1_ema, sym15 saturates, so swap in a stronger/gapped opponent (last rung's winner, or a deliberately-weakened past checkpoint). Naturally becomes a **pool/league** — a panel of opponents spanning a strength range gives a richer outcome distribution than any single one (a refinement of "iter1_ema AND the winner": mix several gaps).

## Outcome
**DROPPED 2026-06-03:** gated on [[e22]] *climbing*; E22 declined, so the league premise is moot. (Also: the broader lesson is that loop-variant tweaks don't beat iter1_ema — see the [[e22]] result.)

## Details
**Does it compound or just tread a treadmill?** Only compounds if the value head learns *transferable* position-evaluation ("good position") rather than the distribution-shift failure ("I'm beating a weakling") — and those look identical in self-play WR, diverging only under equal-strength eval. So **every rung must be gated on H2H vs the FIXED frozen iter1_ema (R1)**; a rung that beats its opponent but doesn't move the frozen-H2H is elaborate treading-water, not strength.

**Honest context:** canonical AlphaZero bootstraps fine from pure mirror self-play, so cross-teacher + league is a **crutch** that injects the decisiveness our loop fails to generate on its own (because the value head is too blurry for MCTS amplification to produce variance — the [[e19]] finding). A working crutch is still a win, but it's a symptom-treatment, which is *why* the external H2H gate is non-negotiable.

**Gate / sequence (R9):** do NOT build until [[e22]] rung 1 *climbs vs frozen iter1_ema*. E22 flat → no league; escalate to architecture (value-head redesign) or [[e21]] ensemble-teacher instead. E22 climbs → build the gap-controlled league (opponent pool + per-rung frozen-H2H gate); this is the candidate plateau-break.

**Numbering note:** a parallel session independently scoped a "real self-play campaign" as **E23** too — that one was renumbered **E24** to avoid colliding with this (now-dropped) entry.

## Provenance & links
- Source snapshot: 2026-06-02 discussion (Jack) scope; dropped 2026-06-03.
- Related: [[e22]] (the prerequisite that declined → DROPPED), [[e19]] (the value-head finding behind the "crutch" framing), [[e21]] (the escalation target instead), [[e24]] (the renumbered "real self-play campaign").
