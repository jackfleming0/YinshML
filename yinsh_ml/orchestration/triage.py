"""Rung 2 of the agentic ladder: triage (a *workflow* — code-controlled tool loop).

This is the first real agentic loop, but **code holds the reins**: Claude is given
a small set of bounded tools and decides which to call to gather evidence on an
*ambiguous* Tier-0 result; our code runs the loop, executes the tools, and caps the
budget. It sits between Rung 1 (one call, no tools) and Rung 3 (the model drives its
own open-ended trajectory).

Why it's safe — the gate boundary still holds. The tools only *gather evidence*
(play more games against the baseline, inspect a flagged failure). The routing that
follows is still the **deterministic SPRT recomputed over the new games**, not
Claude's opinion: more games may push the SPRT across a boundary, flipping the
result to a clear win (→ still gated for promotion) or a clear loss (→ auto-reject,
cheap and reversible). Claude decides *what evidence to collect*; the statistics
decide what it means. Ordering more local games is cheap and reversible, so the
agent does it autonomously — exactly the relief valve for the ratification
bottleneck (inconclusive results self-resolve instead of landing on the human).

Degrades gracefully: any failure returns ``None`` and the scheduler keeps its
deterministic routing. Model: Opus 4.8, adaptive thinking, cached system rubric.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..utils.stats import sprt_decision, wilson_bounds
from .funnel import Tier0Result
from .match_runner import MatchOutcome

if TYPE_CHECKING:
    from .funnel import EvaluationFunnel
    from .match_runner import MatchRunner
    from .spec import ExperimentSpec

logger = logging.getLogger(__name__)

_SYSTEM_RUBRIC = """\
You are the principal investigator (PI) triaging an AMBIGUOUS Tier-0 evaluation of \
a YINSH training candidate — the SPRT was inconclusive and/or a failure-mode check \
fired. Your job is to gather just enough evidence to resolve the ambiguity before \
escalating to the human lead, using the tools provided.

Guidance:
- If the SPRT is merely inconclusive (within the indifference region), the cheapest \
fix is usually to order more games — a clearly-stronger or clearly-weaker candidate \
will cross a boundary quickly. Ordering games is cheap and reversible; do it freely.
- If a failure-mode check fired (especially offense_only_equilibrium), inspect it to \
understand HOW the candidate is failing before concluding.
- Do not over-spend: stop as soon as the picture is clear, or when more games won't \
help (the budget is bounded for you).
- You do NOT promote or reject anything — that stays with the statistics and the \
human gate. When done, call submit_triage with your recommendation and the evidence \
you gathered. 'resolved_promote'/'resolved_reject' mean the added games made the \
result decisive; 'escalate' means it still needs a human.
Reason only from tool results and the facts given; never invent numbers."""


@dataclass
class TriageVerdict:
    recommendation: str  # "resolved_promote" | "resolved_reject" | "escalate"
    rationale: str
    evidence_summary: str


@dataclass
class TriageResult:
    tier0: Tier0Result
    """Re-evaluated Tier-0 (reflects any games the agent ordered)."""
    verdict: TriageVerdict
    games_added: int


class _TriageTools:
    """Bounded tool implementations + mutable evidence state for one triage run."""

    def __init__(self, tier0: Tier0Result, match_runner: "MatchRunner", funnel: "EvaluationFunnel"):
        self._panel = tier0.panel
        self._runner = match_runner
        self._funnel = funnel
        self._outcome = MatchOutcome(
            wins=tier0.outcome.wins, draws=tier0.outcome.draws, losses=tier0.outcome.losses
        )
        self._added = 0

    @property
    def games_added(self) -> int:
        return self._added

    def order_more_games(self, n: int) -> str:
        remaining = self._funnel.max_games - self._outcome.total
        if remaining <= 0:
            return "Budget exhausted — no more games can be ordered."
        n = max(1, min(int(n), remaining))
        batch = self._runner.play_batch(n)
        self._outcome = self._outcome + batch
        self._added += batch.total
        s = self._sprt()
        return (
            f"Played {batch.total} more games. Cumulative: "
            f"{self._outcome.wins}W/{self._outcome.draws}D/{self._outcome.losses}L. "
            f"SPRT now: {s.verdict} (LLR {s.llr:.2f}, bounds [{s.lower:.2f}, {s.upper:.2f}]). "
            f"{self._outcome.total}/{self._funnel.max_games} of the game budget used."
        )

    def inspect_failure(self, check: str) -> str:
        for c in self._panel.checks:
            if c.name == check:
                state = "SKIPPED" if c.skipped else ("passed" if c.passed else "FAILED")
                return f"{c.name}: {state}. {c.detail}" + (
                    f" (value={c.value})" if c.value is not None else ""
                )
        names = ", ".join(c.name for c in self._panel.checks)
        return f"No check named '{check}'. Available: {names}."

    def _sprt(self):
        f = self._funnel
        return sprt_decision(
            self._outcome.wins, self._outcome.losses, self._outcome.draws,
            f.p0, f.p1, f.alpha, f.beta,
        )

    def build_updated_tier0(self) -> Tier0Result:
        lower, upper = wilson_bounds(self._outcome.wins, self._outcome.decisive)
        return Tier0Result(
            panel=self._panel,
            outcome=self._outcome,
            sprt=self._sprt(),
            wilson_lower=lower,
            wilson_upper=upper,
            had_baseline=True,
        )


_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "order_more_games",
        "description": "Play N more games of the candidate vs the baseline and return "
        "the updated tally and recomputed SPRT verdict. Cheap and reversible.",
        "input_schema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "Number of games to play."}},
            "required": ["n"],
        },
    },
    {
        "name": "inspect_failure",
        "description": "Return the detailed signal behind a failure-mode panel check "
        "(e.g. 'offense_only_equilibrium', 'policy_entropy').",
        "input_schema": {
            "type": "object",
            "properties": {"check": {"type": "string", "description": "Panel check name."}},
            "required": ["check"],
        },
    },
    {
        "name": "submit_triage",
        "description": "Conclude triage with a recommendation and the evidence gathered. "
        "This does not promote or reject — the statistics and human gate decide.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recommendation": {
                    "type": "string",
                    "enum": ["resolved_promote", "resolved_reject", "escalate"],
                },
                "rationale": {"type": "string"},
                "evidence_summary": {"type": "string"},
            },
            "required": ["recommendation", "rationale", "evidence_summary"],
        },
    },
]


class TriageWorkflow:
    """Runs the manual agentic loop over the bounded triage tools."""

    def __init__(
        self,
        funnel: "EvaluationFunnel",
        model: str = "claude-opus-4-8",
        max_tokens: int = 8000,
        max_iterations: int = 4,
        client=None,
    ):
        self.funnel = funnel
        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def run(
        self, spec: "ExperimentSpec", tier0: Tier0Result, match_runner: "MatchRunner"
    ) -> Optional[TriageResult]:
        """Drive the loop; return the re-evaluated result + verdict, or None on failure."""
        try:
            client = self._get_client()
            tools = _TriageTools(tier0, match_runner, self.funnel)
            messages: List[Dict[str, Any]] = [
                {"role": "user", "content": self._initial(spec, tier0)}
            ]

            for _ in range(self.max_iterations):
                response = client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    thinking={"type": "adaptive"},
                    system=[
                        {"type": "text", "text": _SYSTEM_RUBRIC, "cache_control": {"type": "ephemeral"}}
                    ],
                    tools=_TOOL_SCHEMAS,
                    messages=messages,
                )
                messages.append({"role": "assistant", "content": response.content})
                tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
                if not tool_uses:
                    break  # ended without submitting — treat as escalate

                verdict, results = self._run_tools(tool_uses, tools)
                messages.append({"role": "user", "content": results})
                if verdict is not None:
                    return TriageResult(tools.build_updated_tier0(), verdict, tools.games_added)

            # Loop budget exhausted without an explicit verdict.
            return TriageResult(
                tools.build_updated_tier0(),
                TriageVerdict("escalate", "Triage budget exhausted without resolution.", ""),
                tools.games_added,
            )
        except Exception as exc:  # noqa: BLE001 - advisory layer must never break the pipeline
            logger.warning("Triage workflow unavailable (%s); keeping deterministic routing.", exc)
            return None

    def _run_tools(self, tool_uses, tools: _TriageTools):
        """Execute each tool call; return (verdict|None, tool_result blocks)."""
        verdict: Optional[TriageVerdict] = None
        results: List[Dict[str, Any]] = []
        for b in tool_uses:
            if b.name == "submit_triage":
                verdict = self._parse_verdict(b.input)
                content = "Triage recorded."
            elif b.name == "order_more_games":
                content = tools.order_more_games(b.input.get("n", 0))
            elif b.name == "inspect_failure":
                content = tools.inspect_failure(b.input.get("check", ""))
            else:
                content = f"Unknown tool: {b.name}"
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": content})
        return verdict, results

    @staticmethod
    def _parse_verdict(data: Dict[str, Any]) -> TriageVerdict:
        rec = data.get("recommendation", "escalate")
        if rec not in ("resolved_promote", "resolved_reject", "escalate"):
            rec = "escalate"
        return TriageVerdict(rec, data.get("rationale", ""), data.get("evidence_summary", ""))

    @staticmethod
    def _initial(spec: "ExperimentSpec", tier0: Tier0Result) -> str:
        o = tier0.outcome
        lines = [
            "This Tier-0 result is ambiguous. Gather evidence with the tools, then submit_triage.",
            "",
            f"Baseline: {spec.baseline_id or '(none)'}",
            f"Current record: {o.wins}W/{o.draws}D/{o.losses}L ({o.decisive} decisive)",
        ]
        if tier0.sprt is not None:
            lines.append(f"SPRT so far: {tier0.sprt.verdict} (LLR {tier0.sprt.llr:.2f}).")
        flags = [c.name for c in tier0.panel.failures]
        skips = [c.name for c in tier0.panel.skips]
        if flags:
            lines.append(f"Failure-mode flags: {', '.join(flags)}.")
        if skips:
            lines.append(f"Skipped checks (no data): {', '.join(skips)}.")
        return "\n".join(lines)
