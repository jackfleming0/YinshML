"""E22 cross-teacher self-play invariants.

The validity of the cross-teacher experiment rests on one thing: ONLY the
learner's positions become training data, and the learner only ever plays its
own color. A silent break here (opponent moves leaking into the buffer, or a
broken color assignment) would *run fine* and quietly invalidate the result —
so we assert it directly with two tagged stub MCTS.
"""
import logging
import numpy as np
import pytest

from yinsh_ml.game import GameState
from yinsh_ml.game.types import Player
from yinsh_ml.utils.encoding import StateEncoder
from yinsh_ml.training.self_play import _run_game_loop_inner

LEARNER_TAG = 0.111
OPP_TAG = 0.222


class StubMCTS:
    """Returns a uniform-ish policy tagged with a recognizable constant so we
    can tell whose move produced any stored target. Records the side-to-move it
    was asked to search, to verify color assignment."""

    def __init__(self, encoder, tag, seen_players=None):
        self.encoder = encoder
        self.tag = tag
        self.seen_players = seen_players
        self.heuristic_evaluator = None  # post-game cache-stats block checks this

    def _probs(self, state):
        if self.seen_players is not None:
            self.seen_players.append(state.current_player)
        return np.full(self.encoder.total_moves, self.tag, dtype=np.float32)

    def search_batch(self, state, move_count, batch_size=None):
        return self._probs(state)

    def search(self, state, move_count):
        return self._probs(state)

    def get_temperature(self, move_count):
        return 1.0  # sample (keeps random play moving toward terminal)

    def advance_root(self, move):
        pass


def _play(opponent, learner_color, seed=0):
    np.random.seed(seed)
    enc = StateEncoder()
    state = GameState()
    learner_seen = []
    learner = StubMCTS(enc, LEARNER_TAG, learner_seen)
    opp = StubMCTS(enc, OPP_TAG) if opponent else None
    states, policies, values, temp_data, hist = _run_game_loop_inner(
        state=state, mcts=learner, state_encoder=enc, game_id=0,
        worker_logger=logging.getLogger("test_cross_teacher"),
        use_batched_mcts=True, mcts_batch_size=8,
        opponent_mcts=opp, learner_color=learner_color,
    )
    return states, policies, values, learner_seen


@pytest.mark.parametrize("learner_color", [Player.WHITE, Player.BLACK])
def test_cross_teacher_stores_only_learner_positions(learner_color):
    states, policies, values, learner_seen = _play(
        opponent=True, learner_color=learner_color)

    assert len(policies) > 0, "no positions stored"
    assert len(states) == len(policies) == len(values), "list lengths desynced"

    # No stored target may carry the OPPONENT's tag — that would mean an
    # opponent move leaked into the learner's training data.
    assert not any(abs(float(p[0]) - OPP_TAG) < 1e-4 for p in policies), \
        "opponent position leaked into training data"

    # The learner only ever searched on its own color.
    assert learner_seen, "learner never searched"
    assert all(pl == learner_color for pl in learner_seen), \
        "learner searched on the wrong color"


def test_mirror_mode_unchanged_stores_both_colors():
    # opponent=None → ordinary self-play: every move stored, both colors seen.
    states, policies, values, learner_seen = _play(opponent=None, learner_color=None)
    assert len(policies) > 0
    seen = set(learner_seen)
    assert Player.WHITE in seen and Player.BLACK in seen, \
        "mirror self-play should play (and store) both colors"
