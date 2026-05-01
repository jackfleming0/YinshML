"""Parity tests for the vectorized MCTS._select_action.

The vectorized implementation (BITBOARD_FOLLOWUP_PLAN.md Candidate B')
must pick the same move as the original scalar implementation under
identical inputs. Since UCB is numerically sensitive — wrong selection
silently corrupts training data — we re-implement the legacy scalar
version inline and compare picks across many synthetic node states.

ε-noise is patched to zero throughout so the comparison is
deterministic. The noise is purely a tiebreaker and adding the same
distribution before vs. after the refactor doesn't change correctness.

Run: pytest yinsh_ml/tests/test_select_action_vector.py -v
"""

import random
import unittest
from unittest import mock

import numpy as np

from yinsh_ml.game.constants import Player, Position
from yinsh_ml.game.types import Move, MoveType
from yinsh_ml.training.self_play import MCTS, Node


def _scalar_select_action(mcts, node):
    """Verbatim re-implementation of the pre-vectorization logic.

    Kept here so the parity test exercises both paths in the same
    process. If the production _select_action ever drifts
    semantically, this function — and only this function — is the
    fixed reference.
    """
    valid_moves = list(node.children.keys())
    parent_visit_count = node.visit_count
    if parent_visit_count == 0:
        return random.choice(valid_moves) if valid_moves else None

    best_score = -float("inf")
    best_move = None
    epsilon = 1e-8

    if mcts.fpu_reduction > 0:
        visited_policy_sum = 0.0
        for _m in valid_moves:
            _c = node.children[_m]
            if _c.visit_count > 0:
                visited_policy_sum += _c.prior_prob
        q_parent_pov = -node.value()
        fpu_q = q_parent_pov - mcts.fpu_reduction * np.sqrt(visited_policy_sum)
    else:
        fpu_q = 0.0

    for move in valid_moves:
        child = node.children[move]
        if child.visit_count == 0:
            q_value = fpu_q
        else:
            q_value = child.value()
        scaled_q = mcts.value_weight * q_value

        if child.visit_count == 0:
            u_value = (child.c_puct * child.prior_prob
                       * np.sqrt(parent_visit_count + epsilon))
        else:
            u_value = (child.c_puct * child.prior_prob
                       * np.sqrt(parent_visit_count) / (1 + child.visit_count))

        score = scaled_q + u_value + np.random.uniform(0, epsilon)
        if score > best_score:
            best_score = score
            best_move = move

    return best_move


def _make_mcts_stub(c_puct=1.5, value_weight=1.0, fpu_reduction=0.25):
    """A bare object with the attributes _select_action reads. Avoids
    constructing a full MCTS (which wants a network, evaluator, etc.)."""
    stub = mock.MagicMock(spec=MCTS)
    stub.c_puct = c_puct
    stub.value_weight = value_weight
    stub.fpu_reduction = fpu_reduction
    stub.logger = mock.MagicMock()
    return stub


def _random_move(rng, idx):
    """Construct a unique, hashable Move per index. Type doesn't matter
    for UCB — only that distinct indices give distinct keys."""
    cols = "ABCDEFGHIJK"
    col = cols[idx % 11]
    row = (idx % 9) + 2
    return Move(
        type=MoveType.PLACE_RING,
        player=Player.WHITE if rng.random() < 0.5 else Player.BLACK,
        source=Position(column=col, row=row),
    )


def _build_random_node(rng, c_puct, num_children, visited_fraction):
    """Synthetic Node with `num_children` children whose stats span
    visited / unvisited / virtual-loss territory."""
    state = mock.MagicMock()
    node = Node(state, c_puct=c_puct)
    # Parent visit count must be > 0 for both implementations to reach
    # the UCB branch (the 0 case bails to random.choice).
    node.visit_count = rng.randint(1, 200)
    node.value_sum = rng.uniform(-50.0, 50.0)
    node.virtual_losses = rng.randint(0, 3)

    for i in range(num_children):
        m = _random_move(rng, i)
        child = Node(state, parent=node, c_puct=c_puct)
        if rng.random() < visited_fraction:
            child.visit_count = rng.randint(1, 50)
            child.value_sum = rng.uniform(-30.0, 30.0)
        else:
            child.visit_count = 0
            child.value_sum = 0.0
        child.virtual_losses = rng.randint(0, 2)
        child.prior_prob = rng.uniform(1e-4, 0.4)
        node.children[m] = child
    return node


class TestVectorScalarParity(unittest.TestCase):
    """Vector and scalar implementations pick the same move."""

    @classmethod
    def setUpClass(cls):
        cls.cases = []
        rng = random.Random(2026_05_01)
        # Mix child counts that span the typical YINSH branching
        # (~80 in MAIN_GAME, ~5–20 in PLACEMENT, ~2–10 in REMOVAL).
        for n_children in [2, 5, 10, 20, 50, 80]:
            for visited_fraction in [0.0, 0.3, 0.7, 1.0]:
                for _ in range(8):  # 8 trials per (n, frac) bucket
                    cls.cases.append(
                        _build_random_node(rng, c_puct=1.5,
                                           num_children=n_children,
                                           visited_fraction=visited_fraction)
                    )

    def _zero_noise(self):
        """Patch np.random.uniform to return zeros so both paths are
        deterministic. Returning a float for scalar callers and an
        array for vector callers. Both call sites pass either no-args
        or size=N — so default returns a 0.0 scalar; size= returns
        zeros of that size."""
        def _u(low=0.0, high=0.0, size=None):
            if size is None:
                return 0.0
            return np.zeros(size)
        return mock.patch("numpy.random.uniform", side_effect=_u)

    def test_vector_matches_scalar_on_random_nodes(self):
        mcts = MCTS.__new__(MCTS)  # uninitialized — we just need the method
        mcts.c_puct = 1.5
        mcts.value_weight = 1.0
        mcts.fpu_reduction = 0.25
        mcts.logger = mock.MagicMock()

        with self._zero_noise():
            for i, node in enumerate(self.cases):
                got = MCTS._select_action(mcts, node)
                want = _scalar_select_action(mcts, node)
                self.assertEqual(
                    got, want,
                    f"case {i} (n={len(node.children)}): "
                    f"vector picked {got}, scalar picked {want}",
                )

    def test_parity_with_fpu_disabled(self):
        """fpu_reduction=0 collapses to the old prior-only scoring;
        verify the vector path still tracks the scalar path."""
        mcts = MCTS.__new__(MCTS)
        mcts.c_puct = 1.5
        mcts.value_weight = 1.0
        mcts.fpu_reduction = 0.0
        mcts.logger = mock.MagicMock()

        with self._zero_noise():
            for node in self.cases[:30]:
                self.assertEqual(
                    MCTS._select_action(mcts, node),
                    _scalar_select_action(mcts, node),
                )

    def test_value_weight_zero_collapses_to_pure_exploration(self):
        mcts = MCTS.__new__(MCTS)
        mcts.c_puct = 1.5
        mcts.value_weight = 0.0
        mcts.fpu_reduction = 0.25
        mcts.logger = mock.MagicMock()

        with self._zero_noise():
            for node in self.cases[:30]:
                self.assertEqual(
                    MCTS._select_action(mcts, node),
                    _scalar_select_action(mcts, node),
                )


class TestEdgeCases(unittest.TestCase):
    """Boundary inputs the vectorization could mishandle."""

    def setUp(self):
        self.mcts = MCTS.__new__(MCTS)
        self.mcts.c_puct = 1.0
        self.mcts.value_weight = 1.0
        self.mcts.fpu_reduction = 0.25
        self.mcts.logger = mock.MagicMock()

    def test_no_children_returns_none(self):
        node = Node(mock.MagicMock(), c_puct=1.0)
        node.visit_count = 5
        # Empty children dict
        self.assertIsNone(MCTS._select_action(self.mcts, node))

    def test_all_unvisited_children(self):
        """Every child has visit_count=0; the path should still pick
        a child via FPU + U-value, not crash on the empty visited mask."""
        rng = random.Random(7)
        node = _build_random_node(rng, c_puct=1.0, num_children=10,
                                  visited_fraction=0.0)
        # Just verify it returns one of the children — actual identity
        # depends on FPU baseline math, exercised in parity tests.
        chosen = MCTS._select_action(self.mcts, node)
        self.assertIn(chosen, node.children)

    def test_single_child(self):
        rng = random.Random(8)
        node = _build_random_node(rng, c_puct=1.0, num_children=1,
                                  visited_fraction=0.5)
        chosen = MCTS._select_action(self.mcts, node)
        self.assertIn(chosen, node.children)

    def test_parent_visit_count_zero_returns_random_choice(self):
        rng = random.Random(9)
        node = _build_random_node(rng, c_puct=1.0, num_children=5,
                                  visited_fraction=0.5)
        node.visit_count = 0
        chosen = MCTS._select_action(self.mcts, node)
        self.assertIn(chosen, node.children)


if __name__ == "__main__":
    unittest.main()
