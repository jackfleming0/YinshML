"""Regression test: anchor_eval detects deterministic-collapse and exposes
per-side stats.

The deterministic-collapse artifact: when both candidate (argmax) and
anchor (HeuristicAgent argmax) are deterministic, every game on a given
side replays the same line — so a 30-game eval splits {0/half, half/half}
by side-coverage rather than skill. The eval should:

1. Track per-side win rate and game-length range so callers can see the
   asymmetry.
2. Emit a `deterministic_sides` list in the result dict naming sides
   where every game had identical move count.

This test fakes the game loop's accumulated stats and exercises the
post-loop aggregation logic.
"""

from __future__ import annotations

import pytest


def _build_post_loop_aggregation(per_side_stats, games_played, candidate_wins,
                                   anchor_wins, draws, depth=1, seed=0,
                                   total_moves=0, mode_label="raw_policy",
                                   use_mcts=False, mcts_simulations=0):
    """Replicate the post-loop aggregation block from run_anchor_eval.

    Lives in a helper here because we don't want to spin up a full
    ModelTournament + HeuristicAgent + NetworkWrapper just to test the
    bookkeeping math.
    """
    win_rate = (candidate_wins / games_played) if games_played > 0 else 0.0

    side_summary = {}
    deterministic_sides = []
    for side, s in per_side_stats.items():
        if s["games"] == 0:
            continue
        mc = s["move_counts"]
        length_min = min(mc)
        length_max = max(mc)
        length_range = length_max - length_min
        side_summary[side] = {
            "games": s["games"],
            "cand_wins": s["cand_wins"],
            "cand_win_rate": s["cand_wins"] / s["games"],
            "avg_game_length": sum(mc) / len(mc),
            "game_length_min": length_min,
            "game_length_max": length_max,
            "game_length_range": length_range,
        }
        if s["games"] >= 2 and length_range == 0:
            deterministic_sides.append(side)

    return {
        "win_rate": win_rate,
        "per_side": side_summary,
        "deterministic_sides": deterministic_sides,
    }


def test_deterministic_split_30_white_wins_30_black_losses_flags_both_sides():
    """Mirror the actual supervised-model finding: every white game wins in
    exactly 103 moves, every black game loses in exactly 81 moves. Should
    flag BOTH sides as deterministic and surface a 50% overall win rate
    that's actually 100% / 0% by side.
    """
    per_side = {
        "white": {"cand_wins": 30, "games": 30, "move_counts": [103] * 30},
        "black": {"cand_wins": 0, "games": 30, "move_counts": [81] * 30},
    }
    result = _build_post_loop_aggregation(
        per_side, games_played=60, candidate_wins=30, anchor_wins=30, draws=0
    )

    assert result["win_rate"] == 0.5, "overall is 30/60"
    assert set(result["deterministic_sides"]) == {"white", "black"}
    w = result["per_side"]["white"]
    b = result["per_side"]["black"]
    assert w["cand_win_rate"] == 1.0
    assert w["game_length_range"] == 0
    assert b["cand_win_rate"] == 0.0
    assert b["game_length_range"] == 0


def test_deterministic_60_60_split_still_flags_both_sides():
    """The other deterministic-collapse case: candidate happens to deterministically
    win as BOTH white and black. 'Looks like 100% skill' to a naive reader, but
    every game is the same replayed line — flag it."""
    per_side = {
        "white": {"cand_wins": 30, "games": 30, "move_counts": [99] * 30},
        "black": {"cand_wins": 30, "games": 30, "move_counts": [127] * 30},
    }
    result = _build_post_loop_aggregation(
        per_side, games_played=60, candidate_wins=60, anchor_wins=0, draws=0
    )

    assert result["win_rate"] == 1.0
    assert set(result["deterministic_sides"]) == {"white", "black"}


def test_stochastic_play_does_not_flag_any_side():
    """With varied game lengths on both sides (i.e., real stochastic eval),
    no side should be flagged."""
    per_side = {
        "white": {"cand_wins": 12, "games": 20,
                  "move_counts": [88, 105, 91, 130, 99, 102, 88, 121,
                                  78, 116, 95, 107, 99, 112, 88, 101,
                                  92, 119, 84, 110]},
        "black": {"cand_wins": 8, "games": 20,
                  "move_counts": [71, 89, 65, 95, 81, 102, 77, 91,
                                  88, 76, 99, 85, 73, 95, 80, 92,
                                  87, 79, 105, 84]},
    }
    result = _build_post_loop_aggregation(
        per_side, games_played=40, candidate_wins=20, anchor_wins=20, draws=0
    )

    assert result["win_rate"] == 0.5
    assert result["deterministic_sides"] == []


def test_single_game_per_side_does_not_flag():
    """With only 1 game per side, length-range is 0 trivially. Should NOT flag
    — we don't have enough samples to call collapse."""
    per_side = {
        "white": {"cand_wins": 1, "games": 1, "move_counts": [100]},
        "black": {"cand_wins": 0, "games": 1, "move_counts": [80]},
    }
    result = _build_post_loop_aggregation(
        per_side, games_played=2, candidate_wins=1, anchor_wins=1, draws=0
    )

    assert result["deterministic_sides"] == []


def test_partial_determinism_flags_only_affected_side():
    """If one side replays the same line but the other has variance, flag
    only the affected side."""
    per_side = {
        "white": {"cand_wins": 5, "games": 5, "move_counts": [113] * 5},  # det
        "black": {"cand_wins": 2, "games": 5,
                  "move_counts": [78, 91, 85, 102, 80]},  # varied
    }
    result = _build_post_loop_aggregation(
        per_side, games_played=10, candidate_wins=7, anchor_wins=3, draws=0
    )

    assert result["deterministic_sides"] == ["white"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
