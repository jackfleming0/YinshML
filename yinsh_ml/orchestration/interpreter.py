"""Rung 1 of the agentic ladder: the PI interpreter (an *augmented step*).

This is the first place real intelligence enters the pipeline — but it is NOT yet
an agent. It is a single Claude call inside an otherwise-deterministic pipeline:
structured facts in, a structured interpretation out. No loop, no tools, no
decisions that move the system on their own.

The deliberate boundary: **the LLM advises, the rules gate.** The routing decision
(auto-reject / promotion-gate / review) stays in ``pi.py`` as deterministic logic —
a model call must never be what decides whether a candidate gets promoted. The
interpreter only authors the *narrative*: what the result means, what to doubt,
what to try next. If there's no API key or the call fails, ``interpret`` returns
``None`` and the journal falls back to its template — the pipeline never breaks
because the LLM is unavailable.

Patterns this rung teaches: structured outputs (``messages.parse`` into a Pydantic
schema), prompt caching (a frozen system rubric cached across every call), adaptive
thinking, and graceful degradation. Model: Opus 4.8 (per the project default).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .funnel import Tier0Result
    from .pi import RoutingDecision
    from .spec import ExperimentSpec

logger = logging.getLogger(__name__)

# Frozen rubric — the PI's standing instructions. Stable across every call, so it
# rides the prompt cache (≈0.1× input cost on repeat). Keep volatile per-result
# facts out of here; they go in the user turn.
_SYSTEM_RUBRIC = """\
You are the principal investigator (PI) for a YINSH AlphaZero-style training \
program. You read the Tier-0 evaluation of a candidate model and write a short, \
honest interpretation for the human lead who oversees the program.

Domain context you must reason with:
- The candidate is judged against a baseline by playing games; an SPRT verdict of \
accept_h1 means decisively stronger, accept_h0 means no better than baseline, and \
continue means inconclusive (within the indifference region — could be noise).
- A failure-mode panel runs cheap checks: policy_entropy (low = the policy is \
collapsing onto a few moves), value_calibration (a saturated or inaccurate value \
head), draw_rate (games degenerating to draws), and offense_only_equilibrium — the \
domain's signature failure: YINSH's heuristic features are all attack-oriented and \
differential, so defense can collapse at shallow search depth. Its signature is a \
sustained one-sided growth of the completed-runs differential with no response from \
the trailing side.
- A panel check may be SKIPPED for lack of data; a skip is not a pass — flag it.

Your job:
- Say plainly what this result means, reasoning ONLY from the facts provided. Never \
invent numbers or claim evidence that isn't in the facts.
- Be skeptical. Always surface concrete reasons to doubt the conclusion (small \
samples, inconclusive SPRT, skipped checks, single-baseline comparisons, etc.).
- Suggest the single most useful next step.
- Give an honest confidence level (high / medium / low) in your own read.
Keep prose tight — this is a research log entry, not an essay."""


@dataclass
class Interpretation:
    """The PI's structured read of one Tier-0 result. Plain value object."""

    headline: str
    assessment: str
    reasons_to_doubt: List[str]
    suggested_next_step: str
    confidence: str  # "high" | "medium" | "low"


def _build_schema():
    """Lazily build the Pydantic output schema (keeps pydantic out of import path)."""
    from typing import List as TList
    from typing import Literal

    from pydantic import BaseModel, Field

    class PIInterpretationSchema(BaseModel):
        headline: str = Field(description="One-line feed summary of the result.")
        assessment: str = Field(description="2-4 sentences on what the result means.")
        reasons_to_doubt: TList[str] = Field(
            description="Concrete reasons the conclusion might be wrong."
        )
        suggested_next_step: str = Field(description="The single most useful next action.")
        confidence: Literal["high", "medium", "low"]

    return PIInterpretationSchema


class PIInterpreter:
    """Single-call Claude interpreter. Advisory only — never gates."""

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_tokens: int = 6000,
        client=None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = client  # injectable for tests

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: only needed when an interpretation is requested

            self._client = anthropic.Anthropic()
        return self._client

    def interpret(
        self,
        spec: "ExperimentSpec",
        tier0: "Tier0Result",
        decision: "RoutingDecision",
    ) -> Optional[Interpretation]:
        """Return the PI's read, or ``None`` to fall back to the template.

        Any failure — missing API key, network error, an SDK too old for the
        params — is swallowed and logged. The pipeline must survive the LLM being
        unavailable, so this never raises.
        """
        try:
            client = self._get_client()
            response = client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_RUBRIC,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": self._facts(spec, tier0, decision)}],
                output_format=_build_schema(),
            )
            p = response.parsed_output
            return Interpretation(
                headline=p.headline,
                assessment=p.assessment,
                reasons_to_doubt=list(p.reasons_to_doubt),
                suggested_next_step=p.suggested_next_step,
                confidence=p.confidence,
            )
        except Exception as exc:  # noqa: BLE001 - advisory layer must never break the pipeline
            logger.warning(
                "PI interpreter unavailable (%s); falling back to templated writeup.", exc
            )
            return None

    @staticmethod
    def _facts(spec, tier0: "Tier0Result", decision: "RoutingDecision") -> str:
        """Serialize the Tier-0 result into the volatile user turn (not cached)."""
        o = tier0.outcome
        lines = [
            "Evaluate this Tier-0 result and respond via the structured schema.",
            "",
            f"Experiment config: {spec.config_path}",
            f"Baseline: {spec.baseline_id or '(none — first candidate, no opponent)'}",
            "",
            f"Match (candidate's perspective): {o.wins}W / {o.draws}D / {o.losses}L "
            f"({o.total} games, {o.decisive} decisive)",
        ]
        if tier0.had_baseline:
            lines.append(
                f"Win-rate 95% Wilson interval (decisive games): "
                f"[{tier0.wilson_lower:.2f}, {tier0.wilson_upper:.2f}]"
            )
        if tier0.sprt is not None:
            lines.append(
                f"SPRT verdict: {tier0.sprt.verdict} "
                f"(log-likelihood ratio {tier0.sprt.llr:.2f}; "
                f"decision bounds [{tier0.sprt.lower:.2f}, {tier0.sprt.upper:.2f}])"
            )
        else:
            lines.append("SPRT: not run (no baseline to play against).")

        lines += ["", "Failure-mode panel:"]
        for c in tier0.panel.checks:
            state = "SKIPPED" if c.skipped else ("pass" if c.passed else "FAIL")
            lines.append(f"  - {c.name}: {state} — {c.detail}")

        lines += [
            "",
            f"Deterministic routing decision (already made by rules, not by you): "
            f"{decision.route.upper()} → status '{decision.next_status}'.",
            f"Routing rationale: {decision.reason}",
        ]
        return "\n".join(lines)
