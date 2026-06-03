"""Engine regression fixture: a real strong-human BGA game must replay legally.

This drives every move of BGA table 862307561 (Black wins 3-2) through our
``GameState`` engine. It exercises ring placement, long board-crossing ring
moves, marker flipping, and the full row-completion -> marker-removal ->
ring-removal -> player-handoff cycle across both colors. If the move generator,
row detection, or removal/handoff logic regresses, this game stops reproducing.
"""

import pytest

from yinsh_ml.game.types import GamePhase
from yinsh_ml.game.constants import Player
from yinsh_ml.data.human_games import bga_862307561 as game


def test_full_game_replays_legally_and_black_wins():
    """The entire transcribed game is legal and ends exactly 3-2 Black."""
    state = game.replay()  # raises IllegalReplayMove if any move is rejected
    assert state.phase == GamePhase.GAME_OVER
    assert state.white_score == 2
    assert state.black_score == 3
    assert state.get_winner() == Player.BLACK


def test_score_progression_matches_source_log():
    """Scores update at exactly the turns the BGA log records a point.

    Expected cumulative ``(white_score, black_score)`` immediately after each
    scoring turn. NB: the BGA log prints scores *mover-first* ("mover :
    opponent"), not White:Black — so its "2 : 1" on Black's turn 56 means
    black=2, white=1, i.e. ``(white=1, black=2)`` here.
    """
    expected = {
        31: (1, 0),  # White completes the B-file row it loaded all game
        54: (1, 1),  # Black's first conversion
        56: (1, 2),  # Black's second (BGA shows "2 : 1", mover-first)
        57: (2, 2),  # White equalizes
        70: (2, 3),  # Black's winning third row
    }
    seen = {}
    for turn_no, _mover, state in game.iter_states():
        if turn_no in expected:
            seen[turn_no] = (state.white_score, state.black_score)
    assert seen == expected


def test_iter_states_yields_one_state_per_main_game_turn():
    states = list(game.iter_states())
    assert len(states) == len(game.MAIN)
    # BGA numbering: main game starts at turn 11.
    assert states[0][0] == len(game.PLACEMENTS) + 1 == 11
    assert states[-1][0] == len(game.PLACEMENTS) + len(game.MAIN)


def test_placement_count_is_five_rings_each():
    whites = [t for t in game.PLACEMENTS if t[0] == "W"]
    blacks = [t for t in game.PLACEMENTS if t[0] == "B"]
    assert len(whites) == 5
    assert len(blacks) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
