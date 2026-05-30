"""The PI (principal-investigator) routing layer.

Turns a Tier-0 result into a disposition. It encodes the autonomy policy we
agreed on:

- **Everything** lands in the feed (visibility), including clear-cut results you
  won't review — so auto-proceed never means silent.
- A **decisive loss with no better-than-baseline evidence** is auto-rejected:
  killing a clearly-dead candidate is cheap and reversible, so agents do it.
- A **decisive win with a clean panel** is *not* auto-promoted: promotion is
  irreversible, so it's queued at the gate for your sign-off.
- **Anything ambiguous** — inconclusive SPRT, a failure-mode flag, or no baseline
  to judge against — is queued at the gate for review.

In this slice the "reasoning" is a deterministic policy + a templated writeup
(see ``journal``). A richer LLM-authored interpretation is the planned deepening;
the routing seam stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .funnel import Tier0Result
from .registry import GateKind, Status

Route = Literal["feed", "gate"]


@dataclass
class RoutingDecision:
    route: Route
    next_status: Status
    reason: str
    feed_headline: str
    feed_detail: str
    gate_kind: Optional[GateKind] = None
    gate_summary: str = ""


class PIRouter:
    def route(self, tier0: Tier0Result) -> RoutingDecision:
        o = tier0.outcome
        record = f"{o.wins}W/{o.draws}D/{o.losses}L"

        if tier0.is_clear_win:
            return RoutingDecision(
                route="gate",
                next_status="awaiting_ratification",
                reason=(
                    "Decisively beat the baseline (SPRT accept_h1) with a clean "
                    "failure-mode panel. Promotion is irreversible, so it's queued "
                    "for your sign-off rather than auto-applied."
                ),
                feed_headline=f"✅ Clear win vs baseline ({record}) — awaiting promotion sign-off",
                feed_detail=(
                    f"SPRT decisive (LLR {tier0.sprt.llr:.2f}); panel all-green. "
                    "Queued at the gate as a promotion."
                ),
                gate_kind="promotion",
                gate_summary=f"Promote? Decisive win {record}, panel green.",
            )

        if tier0.is_clear_loss:
            return RoutingDecision(
                route="feed",
                next_status="rejected",
                reason=(
                    "Decisively no better than the baseline (SPRT accept_h0). "
                    "Auto-rejected — killing a clearly-dead candidate is cheap and "
                    "reversible, so no sign-off is needed. Logged to the feed."
                ),
                feed_headline=f"❌ No better than baseline ({record}) — auto-rejected",
                feed_detail=f"SPRT decisive against the candidate (LLR {tier0.sprt.llr:.2f}).",
            )

        # Everything else is ambiguous -> review gate.
        reason_bits = []
        if not tier0.had_baseline:
            reason_bits.append("no baseline to play against (first candidate)")
        if not tier0.panel_green:
            flags = ", ".join(c.name for c in tier0.panel.failures)
            reason_bits.append(f"failure-mode flag(s): {flags}")
        if tier0.had_baseline and tier0.sprt is not None and not tier0.sprt.decisive:
            reason_bits.append("SPRT inconclusive (within the indifference region)")
        why = "; ".join(reason_bits) or "result needs judgement"

        return RoutingDecision(
            route="gate",
            next_status="awaiting_ratification",
            reason=f"Ambiguous — {why}. Queued for review.",
            feed_headline=f"🟡 Ambiguous ({record}) — queued for review",
            feed_detail=why,
            gate_kind="review",
            gate_summary=f"Review needed: {why}.",
        )
