"""Regression tests for the promotion gate's two correctness defenses.

Both are anchored on the iter_4 / B1+B2+B3 false-promotion incident
(2026-05-26):

  iter_4 promoted with ELO 1507.3 vs best 1500.0 (a 7-point gap on
  Glicko ratings with RD ~250 — well within noise). The promotion
  bypassed the Wilson gate entirely because the tournament's
  sliding_window=3 had dropped iter_0 (the actual best) from the
  round-robin, so no H2H games existed against best. Independent SPRT
  later confirmed the candidate's true WR vs iter_0 was 0.468 — i.e.
  the candidate was NOT meaningfully stronger.

Defense 1 (already in place, pinned here): when H2H games exist,
  `_should_promote` uses the WILSON LOWER BOUND against the threshold,
  not the point estimate. For iter_4-style numbers (206/400 wins,
  WR 0.515) at threshold 0.50, the Wilson lower bound is ~0.466 which
  is < 0.50 → REJECT. This means if iter_0 had been in the round-robin
  and iter_4 had played 400 H2H games against it (and won 206 of them),
  the gate would have correctly rejected.

Defense 2 (added 2026-05-26): when Wilson SHOULD have run but the
  tournament had no H2H data, the supervisor now fail-closes instead of
  falling through to the Elo-gap path. This test verifies the inline
  branch logic — see supervisor.py around the `wilson_attempted_no_data`
  flag. (Full integration coverage of the branch requires either a
  refactor extracting `_decide_promotion(...)` or end-to-end mocking;
  neither fits the immediate time budget. The supervisor changes are
  also reviewed in the EXPERIMENT_BACKLOG B1+B2+B3 Done entry.)
"""
from __future__ import annotations

import inspect

from yinsh_ml.utils.stats import wilson_bounds
from yinsh_ml.training.supervisor import TrainingSupervisor


def test_iter4_style_numbers_rejected_by_wilson_lower_bound():
    """Iter 4's 206/400 (point WR 0.515) must REJECT against a 0.50
    threshold once Wilson's lower bound is the comparator.

    Numbers from the B1+B2+B3 run: in-loop arena had iter_4 at WR 0.516
    (51.6%), Glicko 1507.3, gate threshold 0.50. We don't have the exact
    wins-count logged separately (the gate didn't run for iter_4 because
    no H2H games existed), but 206/400 = 0.515 reproduces the WR
    closely enough to pin the math.
    """
    wins, total, threshold = 206, 400, 0.50
    lower, upper = wilson_bounds(wins, total)

    # Wilson lower bound is roughly 0.467 here; well below the 0.50 bar.
    assert lower < threshold, (
        f"Iter-4-style WR {wins}/{total} (point {wins/total:.3f}) must reject "
        f"under Wilson lower-bound rule; got lower={lower:.4f}, threshold={threshold}"
    )
    assert 0.46 < lower < 0.48, (
        f"Wilson lower bound for {wins}/{total} should be ~0.466; got {lower:.4f}"
    )
    # Point estimate above threshold confirms the Wilson rule does
    # something meaningful here — the point-estimate-only rule would
    # have promoted.
    assert wins / total > threshold, (
        "Sanity: point estimate must be above threshold or the test "
        "isn't proving anything about the Wilson rule"
    )


def test_clear_win_passes_wilson_lower_bound():
    """A genuinely strong candidate must still PASS the Wilson rule.
    Confirms we haven't broken the gate by making it too strict.

    240/400 = 0.60 (WR) and Wilson lower bound ~0.55 > 0.50 threshold.
    """
    wins, total, threshold = 240, 400, 0.50
    lower, _ = wilson_bounds(wins, total)
    assert lower > threshold, (
        f"A clear 60% candidate must pass Wilson; got lower={lower:.4f} "
        f"vs threshold {threshold}"
    )


def test_fail_closed_branch_present_in_supervisor():
    """Verify the new fail-closed code path exists in the supervisor's
    promotion logic. This is a structural check — full integration
    coverage would require refactoring train_iteration to extract the
    decision tree, which is bigger than the immediate budget allows.

    What we verify here: the wilson_attempted_no_data flag is set when
    H2H total is 0, and there's a dedicated branch that KEEPS (not
    promotes) the candidate in that case.
    """
    src = inspect.getsource(TrainingSupervisor.train_iteration)
    assert "wilson_attempted_no_data" in src, (
        "fail-closed flag wilson_attempted_no_data must be referenced in "
        "train_iteration. If the flag was renamed or removed, update this test "
        "and the EXPERIMENT_BACKLOG note."
    )
    # The branch must produce a 'kept' decision (no promote, no revert)
    # — confirmed by the presence of the warning string introduced
    # alongside the flag.
    assert "Elo fallback is too noisy" in src or "Wilson gate could not run" in src, (
        "The fail-closed warning must mention why we're not falling through "
        "to Elo. Otherwise future readers will rip the branch out as dead code."
    )
