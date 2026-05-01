"""Regression tests for the MCTS backprop perspective fix.

YINSH does not switch the player to move on every ply — capture sequences
(``MOVE_RING``→``ROW_COMPLETION``→``REMOVE_MARKERS``→``REMOVE_RING``) keep the
same player for 3+ consecutive plies. The pre-fix MCTS unconditionally negated
the running value at every step of backprop, which scrambled Q-values along
exactly the sub-trees that represent finishing tactics. These tests pin:

  1. ``_get_value`` / ``_get_terminal_value`` return values in leaf-player POV,
     not objective WHITE-POV.
  2. ``_backpropagate`` flips the running value only across edges where the
     player to move actually changes.
  3. Chess-style alternating paths still match the legacy storage convention
     (no behavior change for the canonical AlphaZero case).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from yinsh_ml.game.constants import Player
from yinsh_ml.game.game_state import GameState
from yinsh_ml.training.self_play import MCTS as SelfPlayMCTS, Node
from yinsh_ml.search.mcts import MCTS as SearchMCTS, MCTSNode, MCTSConfig
from yinsh_ml.utils.encoding import StateEncoder


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _fake_network(total_moves):
    network = MagicMock()
    network.state_encoder = StateEncoder()

    def predict_from_state(_state):
        return torch.zeros(1, total_moves), torch.zeros(1, 1)

    network.predict_from_state = predict_from_state
    return network


def _selfplay_mcts():
    encoder = StateEncoder()
    return SelfPlayMCTS(
        network=_fake_network(encoder.total_moves),
        evaluation_mode="pure_neural",
        num_simulations=4,
        late_simulations=4,
        simulation_switch_ply=20,
        c_puct=1.0,
        dirichlet_alpha=0.0,
        value_weight=1.0,
        max_depth=50,
        initial_temp=1.0,
        final_temp=1.0,
        annealing_steps=1,
    )


def _search_mcts():
    network = _fake_network(StateEncoder().total_moves)
    config = MCTSConfig(num_simulations=4)
    return SearchMCTS(network=network, config=config)


def _make_node():
    """A bare Node usable by self_play.MCTS._backpropagate."""
    return Node(state=None, c_puct=1.0)


def _make_mcts_node():
    return MCTSNode(state=None, c_puct=1.0)


# --------------------------------------------------------------------------- #
# Terminal value perspective                                                  #
# --------------------------------------------------------------------------- #


def test_self_play_get_value_leaf_pov_white_to_move_white_wins():
    """White to move at a terminal state where White scored 3 → +1 in White's POV."""
    mcts = _selfplay_mcts()
    state = MagicMock()
    state.is_terminal.return_value = True
    state.white_score = 3
    state.black_score = 0
    state.current_player = Player.WHITE
    assert mcts._get_value(state) == pytest.approx(1.0)


def test_self_play_get_value_leaf_pov_black_to_move_white_wins():
    """Black to move at a terminal where White scored 3 → -1 in Black's POV."""
    mcts = _selfplay_mcts()
    state = MagicMock()
    state.is_terminal.return_value = True
    state.white_score = 3
    state.black_score = 0
    state.current_player = Player.BLACK
    assert mcts._get_value(state) == pytest.approx(-1.0)


def test_self_play_get_value_leaf_pov_black_wins_black_to_move():
    mcts = _selfplay_mcts()
    state = MagicMock()
    state.is_terminal.return_value = True
    state.white_score = 0
    state.black_score = 3
    state.current_player = Player.BLACK
    assert mcts._get_value(state) == pytest.approx(1.0)


def test_self_play_get_value_returns_none_for_non_terminal():
    mcts = _selfplay_mcts()
    state = MagicMock()
    state.is_terminal.return_value = False
    assert mcts._get_value(state) is None


def test_search_mcts_get_terminal_value_leaf_pov():
    mcts = _search_mcts()
    state = MagicMock()
    state.is_terminal.return_value = True
    state.get_winner.return_value = Player.WHITE

    state.current_player = Player.WHITE
    assert mcts._get_terminal_value(state) == pytest.approx(1.0)

    state.current_player = Player.BLACK
    assert mcts._get_terminal_value(state) == pytest.approx(-1.0)


def test_search_mcts_get_terminal_value_draw():
    mcts = _search_mcts()
    state = MagicMock()
    state.is_terminal.return_value = True
    state.get_winner.return_value = None
    state.current_player = Player.WHITE
    assert mcts._get_terminal_value(state) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Backprop sign-flip — chess-alternating path (regression: legacy unchanged)  #
# --------------------------------------------------------------------------- #


def test_self_play_backprop_chess_alternating_unchanged():
    """[X, Y, X] with leaf v=+0.5 in X's POV → leaf=-0.5, child=+0.5, root=-0.5."""
    mcts = _selfplay_mcts()
    nodes = [_make_node() for _ in range(3)]
    players = [Player.WHITE, Player.BLACK, Player.WHITE]
    mcts._backpropagate(nodes, players, 0.5)

    assert nodes[0].value_sum == pytest.approx(-0.5)  # root
    assert nodes[1].value_sum == pytest.approx(0.5)   # child
    assert nodes[2].value_sum == pytest.approx(-0.5)  # leaf
    assert all(n.visit_count == 1 for n in nodes)


def test_search_mcts_backprop_chess_alternating_unchanged():
    """search/mcts.py uses the +value (no leading negation) convention."""
    mcts = _search_mcts()
    nodes = [_make_mcts_node() for _ in range(3)]
    players = [Player.WHITE, Player.BLACK, Player.WHITE]
    mcts._backpropagate(nodes, players, 0.5)

    assert nodes[0].value_sum == pytest.approx(0.5)   # root
    assert nodes[1].value_sum == pytest.approx(-0.5)  # child
    assert nodes[2].value_sum == pytest.approx(0.5)   # leaf


# --------------------------------------------------------------------------- #
# Backprop sign-flip — YINSH same-player capture sequences                    #
# --------------------------------------------------------------------------- #


def test_self_play_backprop_capture_sequence_same_player():
    """[X, X, X] (same player throughout — capture sub-tree).

    Pre-fix behavior: leaf=-v, mid=+v, root=-v (alternating, WRONG: PUCT at
    root would see child Q with the wrong sign for X).
    Post-fix behavior: leaf and mid both store value in *parent's POV*, which
    equals running value's POV when player matches. Storage convention
    preserves "same as parent → +running, different → -running".
    """
    mcts = _selfplay_mcts()
    nodes = [_make_node() for _ in range(3)]
    players = [Player.WHITE, Player.WHITE, Player.WHITE]
    mcts._backpropagate(nodes, players, 0.7)

    # leaf (i=2): same player as parent (i=1) → store +running = +0.7
    assert nodes[2].value_sum == pytest.approx(0.7)
    # mid (i=1): same player as parent (i=0) → store +running = +0.7
    assert nodes[1].value_sum == pytest.approx(0.7)
    # root (i=0): no parent → legacy convention, store -running = -0.7
    assert nodes[0].value_sum == pytest.approx(-0.7)


def test_self_play_backprop_mixed_path():
    """[X, Y, Y, Y] — X moves, then Y captures across 3 plies.

    After X's move, player flips to Y. Y then makes a capturing move, removes
    markers, removes ring — staying as Y throughout. Leaf value v in Y's POV.

    Walking up:
      i=3 (leaf, Y): parent at i=2 is Y → same → store +v.
      i=2 (Y): parent at i=1 is Y → same → store +v.
      i=1 (Y): parent at i=0 is X → different → store -v, flip running to -v.
      i=0 (root, X): no parent → legacy, store -running = -(-v) = +v.

    The root storage being +v means PUCT, which reads child.value() as
    "Q from root-POV" (X's POV), would see child[Y] with value -v. If v>0
    (Y winning), -v<0 → X correctly sees this child as bad. That's the whole
    point: the fix lets X actually distinguish "opponent completes a row".
    """
    mcts = _selfplay_mcts()
    nodes = [_make_node() for _ in range(4)]
    players = [Player.WHITE, Player.BLACK, Player.BLACK, Player.BLACK]
    mcts._backpropagate(nodes, players, 0.6)

    assert nodes[3].value_sum == pytest.approx(0.6)   # leaf, same as parent
    assert nodes[2].value_sum == pytest.approx(0.6)   # same as parent
    assert nodes[1].value_sum == pytest.approx(-0.6)  # different from parent
    assert nodes[0].value_sum == pytest.approx(0.6)   # root: legacy stores -running


def test_search_mcts_backprop_capture_sequence_same_player():
    """Same scenario for search/mcts.py (its convention stores +running).

    [X, X, X] all same player, leaf v in X's POV:
      Pre-fix: leaf=+v, mid=-v, root=+v — alternating regardless of player.
      Post-fix: running never flips (no transitions) → leaf=mid=root=+v.
    """
    mcts = _search_mcts()
    nodes = [_make_mcts_node() for _ in range(3)]
    players = [Player.WHITE, Player.WHITE, Player.WHITE]
    mcts._backpropagate(nodes, players, 0.4)

    assert nodes[0].value_sum == pytest.approx(0.4)
    assert nodes[1].value_sum == pytest.approx(0.4)
    assert nodes[2].value_sum == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# Cumulative behavior across multiple sims                                    #
# --------------------------------------------------------------------------- #


def test_self_play_backprop_accumulates_across_sims():
    """Running two sims should sum value_sum and bump visit_count to 2."""
    mcts = _selfplay_mcts()
    nodes = [_make_node() for _ in range(2)]
    players = [Player.WHITE, Player.BLACK]
    mcts._backpropagate(nodes, players, 1.0)
    mcts._backpropagate(nodes, players, 0.2)

    # leaf (different player from root): store -running = -1.0 then -0.2 → -1.2
    # root: legacy stores -running. running flipped at the leaf→root transition,
    #   so the root receives running=-1.0 first sim, -0.2 second sim, stored as
    #   -running = +1.0 then +0.2 → +1.2.
    assert nodes[0].visit_count == 2
    assert nodes[1].visit_count == 2
    assert nodes[0].value_sum == pytest.approx(1.2)
    assert nodes[1].value_sum == pytest.approx(-1.2)


def test_self_play_backprop_root_only_path():
    """Single-node path (root is the leaf): only the legacy "store -running"
    branch fires."""
    mcts = _selfplay_mcts()
    nodes = [_make_node()]
    mcts._backpropagate(nodes, [Player.WHITE], 0.3)
    assert nodes[0].visit_count == 1
    assert nodes[0].value_sum == pytest.approx(-0.3)


# --------------------------------------------------------------------------- #
# End-to-end: PUCT at parent reads correct sign for capture-sequence child   #
# --------------------------------------------------------------------------- #


def test_self_play_puct_q_for_capture_child_has_correct_sign():
    """End-to-end check on the convention that matters for selection.

    Build a 2-deep tree manually: parent (X to move), child (still X — capture
    move). After backprop with leaf v=+0.5 in X's POV, PUCT reads
    ``child.value()`` directly as "Q from parent's POV" (X's POV).

    Pre-fix, child.value() = +v (alternation flipped twice across 2 levels)
    but parent's POV value should match X's POV value at leaf = +v. Wait —
    in chess this would coincidentally be correct because alternation. The
    YINSH-specific bug shows up when parent and child share a player.

    Concretely: parent is X, child is X (capture continuation), leaf is X
    (further capture step). v=+0.5 in X's POV. We want child.value() = +0.5
    so that PUCT at parent (X to move) sees this child as good for X.

    Pre-fix: ``-value; value=-value`` per ply → child stores +v then root
    stores -v but ALSO any internal node stored -v. With path [X,X,X], pre-fix
    leaf=-0.5, mid=+0.5, root=-0.5 — mid.value() = +0.5 only by accident,
    because of double-flip parity. The bug bites specifically when path
    length is odd within a same-player run, which is exactly the typical
    capture sequence (3 plies: MOVE_RING + REMOVE_MARKERS + REMOVE_RING).

    Test the 3-ply same-player case directly:
    """
    mcts = _selfplay_mcts()
    parent = _make_node()
    mid = _make_node()
    leaf = _make_node()
    players = [Player.WHITE, Player.WHITE, Player.WHITE]
    mcts._backpropagate([parent, mid, leaf], players, 0.5)

    # PUCT at parent uses mid.value() as Q from parent's POV (X). Since X is
    # winning at leaf (+0.5 in X's POV), PUCT must see mid.value() > 0.
    assert mid.value() > 0, (
        "After fix: mid is in same-player chain, no sign flip; mid.value() "
        "should be +0.5 in parent's POV, matching X-winning at leaf."
    )
    # Concretely:
    assert mid.value() == pytest.approx(0.5)
