"""The Tier-0 failure-mode panel.

These are the "nearly free" checks: they read signals the training run already
produced (policy entropy, value-head calibration) plus a first cut at the
domain-specific *offense-only equilibrium* detector that today only exists as a
manual eyeball pass over ``scripts/replay_heuristic_vs_heuristic.py``.

Design note: the panel consumes *already-extracted* signals (a ``PanelInput``),
not raw parquet or checkpoints. That keeps the parquet/model dependencies out of
the funnel and makes every check unit-testable with plain floats. Wiring real
runs into a ``PanelInput`` is a separate, swappable concern (``from_metrics``).

A check that can't run for lack of data returns ``passed=True`` but says so in its
detail, so missing data never silently blocks a candidate — but it *does* surface
in the PI writeup's reasons-to-doubt. Tiers 1-2 add the gauntlet on top; this
slice ships Tier 0 only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    value: Optional[float] = None
    skipped: bool = False


@dataclass
class PanelResult:
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def green(self) -> bool:
        """All checks that actually ran passed."""
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def skips(self) -> List[CheckResult]:
        return [c for c in self.checks if c.skipped]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "green": self.green,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "skipped": c.skipped,
                    "detail": c.detail,
                    "value": c.value,
                }
                for c in self.checks
            ],
        }


@dataclass
class PanelInput:
    """Extracted signals for one candidate. Any field may be ``None`` (skips it)."""

    policy_entropy: Optional[float] = None
    value_accuracy: Optional[float] = None
    value_variance: Optional[float] = None
    draw_rate: Optional[float] = None
    # Per-game signed "completed-runs differential" trajectories (one list per
    # sampled game, signed so + favors the candidate). Used by the offense-only
    # detector. Empty/None -> check skipped.
    run_diff_trajectories: Optional[Sequence[Sequence[float]]] = None

    @classmethod
    def from_metrics(cls, metrics: Dict[str, Any], **overrides) -> "PanelInput":
        """Build from an ``IterationMetrics``-style dict (final iteration).

        Pulls the fields the supervisor already logs; trajectory extraction from
        parquet is left to the caller (pass ``run_diff_trajectories=``).
        """
        data = dict(
            policy_entropy=metrics.get("policy_target_entropy_mean"),
            value_accuracy=metrics.get("value_accuracy"),
            value_variance=metrics.get("value_variance"),
        )
        data.update(overrides)
        return cls(**data)


class FailurePanel:
    """Runs the Tier-0 checks. Thresholds are constructor-configurable."""

    def __init__(
        self,
        min_policy_entropy: float = 0.5,
        min_value_accuracy: float = 0.40,
        min_value_variance: float = 1e-3,
        max_draw_rate: float = 0.85,
        offense_only_growth: float = 3.0,
        offense_only_window: int = 8,
    ):
        self.min_policy_entropy = min_policy_entropy
        self.min_value_accuracy = min_value_accuracy
        self.min_value_variance = min_value_variance
        self.max_draw_rate = max_draw_rate
        self.offense_only_growth = offense_only_growth
        self.offense_only_window = offense_only_window

    def evaluate(self, data: PanelInput) -> PanelResult:
        return PanelResult(
            checks=[
                self._policy_entropy(data.policy_entropy),
                self._value_calibration(data.value_accuracy, data.value_variance),
                self._draw_rate(data.draw_rate),
                self._offense_only(data.run_diff_trajectories),
            ]
        )

    def _policy_entropy(self, entropy: Optional[float]) -> CheckResult:
        if entropy is None:
            return CheckResult("policy_entropy", True, "skipped (no entropy logged)", skipped=True)
        ok = entropy >= self.min_policy_entropy
        return CheckResult(
            "policy_entropy", ok,
            f"entropy={entropy:.3f} (min {self.min_policy_entropy}); "
            + ("healthy" if ok else "COLLAPSE — policy concentrating on few moves"),
            value=entropy,
        )

    def _value_calibration(
        self, accuracy: Optional[float], variance: Optional[float]
    ) -> CheckResult:
        if accuracy is None and variance is None:
            return CheckResult("value_calibration", True, "skipped (no value metrics)", skipped=True)
        problems = []
        if accuracy is not None and accuracy < self.min_value_accuracy:
            problems.append(f"accuracy {accuracy:.3f} < {self.min_value_accuracy}")
        if variance is not None and variance < self.min_value_variance:
            problems.append(f"variance {variance:.2e} < {self.min_value_variance} (head saturated)")
        ok = not problems
        return CheckResult(
            "value_calibration", ok,
            "well-calibrated" if ok else "; ".join(problems),
            value=accuracy,
        )

    def _draw_rate(self, draw_rate: Optional[float]) -> CheckResult:
        if draw_rate is None:
            return CheckResult("draw_rate", True, "skipped (no draw rate)", skipped=True)
        ok = draw_rate <= self.max_draw_rate
        return CheckResult(
            "draw_rate", ok,
            f"draw_rate={draw_rate:.2f} (max {self.max_draw_rate}); "
            + ("ok" if ok else "EXPLOSION — games collapsing to draws"),
            value=draw_rate,
        )

    def _offense_only(
        self, trajectories: Optional[Sequence[Sequence[float]]]
    ) -> CheckResult:
        """First-cut offense-only-equilibrium detector.

        The failure signature (per CLAUDE.md / replay_heuristic_vs_heuristic.py):
        the completed-runs differential grows sustainedly in one direction without
        the trailing side ever responding. We flag a game when the signed
        differential climbs by >= ``offense_only_growth`` across a window of
        ``offense_only_window`` plies while never recovering toward zero — i.e.
        one-sided runaway, the negamax-depth-1/2 collapse mode. A candidate fails
        if a majority of sampled games show the signature.
        """
        if not trajectories:
            return CheckResult(
                "offense_only_equilibrium", True,
                "skipped (no run-diff trajectories sampled)", skipped=True,
            )

        flagged = sum(1 for t in trajectories if self._is_one_sided_runaway(t))
        frac = flagged / len(trajectories)
        ok = frac <= 0.5
        return CheckResult(
            "offense_only_equilibrium", ok,
            f"{flagged}/{len(trajectories)} sampled games show one-sided runaway "
            f"({frac:.0%}); " + ("balanced" if ok else "OFFENSE-ONLY equilibrium suspected"),
            value=frac,
        )

    def _is_one_sided_runaway(self, traj: Sequence[float]) -> bool:
        if len(traj) < self.offense_only_window:
            return False
        # Largest sustained monotone-ish climb over any window, in the dominant
        # sign, with no return toward zero inside the window.
        w = self.offense_only_window
        for i in range(len(traj) - w + 1):
            seg = traj[i : i + w]
            growth = seg[-1] - seg[0]
            if abs(growth) < self.offense_only_growth:
                continue
            sign = 1.0 if growth > 0 else -1.0
            # "No response" = the trailing side never claws the differential back
            # toward zero within the window (segment stays on one side of start).
            if all((v - seg[0]) * sign >= -1e-9 for v in seg):
                return True
        return False
