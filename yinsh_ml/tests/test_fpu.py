"""Tests for First-Play Urgency (FPU) in PUCT selection.

FPU changes what score unvisited children receive. Without FPU they implicitly
use ``q=0`` and fall back to prior-only ordering; with KataGo-style FPU they use
``q_fpu = q_parent_pov - fpu_reduction * sqrt(Σ π(c) for visited c)``.

Two behaviors we can assert deterministically:
  * When parent-POV Q is high (parent thinks their position is good), unvisited
    children's baseline goes UP relative to low-Q visited siblings — so
    exploration is encouraged when things are going well.
  * With ``fpu_reduction=0`` (disable), selection reverts to the old behavior:
    visited and unvisited children share the ``q=0`` baseline, and a visited
    child with a bad Q loses to an unvisited sibling of equal prior.

Tests also pin the perspective convention (`q_parent_pov = -node.value()`) with
a hand-built tree, so future refactors don't silently invert the sign.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.training.self_play import MCTS, Node
from yinsh_ml.utils.encoding import StateEncoder


def _fake_network():
    encoder = StateEncoder()
    network = MagicMock()
    network.state_encoder = encoder
    def predict_from_state(_state):
        return torch.zeros(1, encoder.total_moves), torch.zeros(1, 1)
    network.predict_from_state = predict_from_state
    return network


def _build_mcts(fpu_reduction=0.25):
    return MCTS(
        network=_fake_network(),
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
        temp_clamp_fraction=1.0,
        enable_subtree_reuse=False,
        fpu_reduction=fpu_reduction,
    )


def _parent_with_two_children(visited_child_value, visited_prior, unvisited_prior, parent_visit_count=5):
    """Parent with one visited child (known Q, known visit count) and one
    unvisited child. Returns (parent, visited_child, unvisited_child)."""
    parent = Node(state=None, c_puct=1.0)
    parent.is_expanded = True
    parent.visit_count = parent_visit_count
    # node.value() = value_sum / visit_count. Pick value_sum so value() is a
    # known quantity. Recall: node.value() is grandparent-POV; in the UCB we
    # negate it for parent-POV.
    parent.value_sum = 0.0  # keeps parent's `value()` at 0.0 — neutral for easy reasoning

    visited = Node(state=None, parent=parent, prior_prob=visited_prior, c_puct=1.0)
    visited.visit_count = 3
    visited.value_sum = visited_child_value * visited.visit_count  # visited.value() = visited_child_value

    unvisited = Node(state=None, parent=parent, prior_prob=unvisited_prior, c_puct=1.0)
    # visit_count = 0, value_sum = 0

    parent.children = {"visited": visited, "unvisited": unvisited}
    return parent, visited, unvisited


class TestFpuConfiguration:
    def test_default_is_katago_025(self):
        mcts = _build_mcts(fpu_reduction=0.25)
        assert mcts.fpu_reduction == pytest.approx(0.25)

    def test_disable_zero(self):
        mcts = _build_mcts(fpu_reduction=0.0)
        assert mcts.fpu_reduction == 0.0

    def test_custom_value(self):
        mcts = _build_mcts(fpu_reduction=0.5)
        assert mcts.fpu_reduction == pytest.approx(0.5)


class TestFpuSelection:
    """Behavior of _select_action under FPU."""

    def test_disabled_fpu_matches_old_behavior(self):
        """With fpu=0, a visited child with Q=-0.5 loses to an equal-prior
        unvisited sibling whose implicit Q is 0. The old code path."""
        mcts = _build_mcts(fpu_reduction=0.0)
        parent, visited, unvisited = _parent_with_two_children(
            visited_child_value=-0.5, visited_prior=0.5, unvisited_prior=0.5,
        )
        # Seed numpy to silence the tie-breaking epsilon noise (it's 1e-8 so
        # essentially irrelevant, but makes the test independent of RNG state).
        np.random.seed(0)
        chosen = mcts._select_action(parent)
        assert chosen == "unvisited"

    def test_enabled_fpu_flips_decision_vs_low_prior_unvisited(self):
        """FPU's main job: stop a tiny-prior unvisited sibling from being
        picked over a well-explored decent-Q sibling just because the U-term
        (c_puct·π·sqrt(N)/(1+visits)) favours unexplored nodes. Needs a
        regime where the U-advantage of the unvisited sibling is small (low
        prior, high N, heavily-visited sibling so 1+visits is big). At
        fpu=0, the unvisited sibling just barely wins; at fpu=0.25, its Q
        drops and the visited sibling takes the lead."""
        # Parent: N=100 visits, value()=0 (parent's-POV Q = 0).
        # Visited: prior=0.5, 50 visits, Q=0. u = 1.0 * 0.5 * sqrt(100) / 51
        #          ≈ 0.098. Total = 0.098.
        # Unvisited: prior=0.01, u = 1.0 * 0.01 * sqrt(100) / 1 = 0.10.
        #   fpu=0 → Total = 0.10 (barely beats visited).
        #   fpu=0.25 → Q = 0 - 0.25·sqrt(0.5) ≈ -0.177. Total ≈ -0.077
        #   (loses to visited).
        def _setup():
            parent = Node(state=None, c_puct=1.0)
            parent.is_expanded = True
            parent.visit_count = 100
            parent.value_sum = 0.0
            visited = Node(state=None, parent=parent, prior_prob=0.5, c_puct=1.0)
            visited.visit_count = 50
            visited.value_sum = 0.0
            unvisited = Node(state=None, parent=parent, prior_prob=0.01, c_puct=1.0)
            parent.children = {"visited": visited, "unvisited": unvisited}
            return parent

        mcts_off = _build_mcts(fpu_reduction=0.0)
        np.random.seed(0)
        assert mcts_off._select_action(_setup()) == "unvisited"

        mcts_on = _build_mcts(fpu_reduction=0.25)
        np.random.seed(0)
        assert mcts_on._select_action(_setup()) == "visited"

    def test_fpu_uses_parent_pov_sign(self):
        """When the parent thinks their own position is BAD (parent's-POV Q
        is negative), FPU should push unvisited children DOWN further. In this
        codebase, node.value() is from grandparent's POV, so parent-POV =
        -node.value(). If the sign convention flips silently, unvisited
        children will get an inflated baseline instead of a deflated one, and
        this test will catch it."""
        mcts = _build_mcts(fpu_reduction=0.25)
        parent = Node(state=None, c_puct=1.0)
        parent.is_expanded = True
        parent.visit_count = 4
        # node.value() = +0.8 (from grandparent's POV). Parent's-own-POV Q =
        # -0.8 ("parent is losing"). FPU for unvisited = -0.8 - 0.25*sqrt(Σπ).
        parent.value_sum = 0.8 * parent.visit_count

        # Visited child with actual Q = 0.0 and prior 0.3.
        visited = Node(state=None, parent=parent, prior_prob=0.3, c_puct=1.0)
        visited.visit_count = 2
        visited.value_sum = 0.0  # value = 0

        # Unvisited child with equal prior 0.3.
        unvisited = Node(state=None, parent=parent, prior_prob=0.3, c_puct=1.0)

        parent.children = {"visited": visited, "unvisited": unvisited}

        np.random.seed(0)
        chosen = mcts._select_action(parent)
        # If sign is correct: FPU = -0.8 - 0.25*sqrt(0.3) ≈ -0.937. Visited Q = 0.
        # Visited should win (its Q is higher). If sign were flipped, FPU
        # would be +0.8 - 0.25*sqrt(0.3) ≈ +0.663 and unvisited would win.
        assert chosen == "visited", (
            "FPU used the wrong perspective — unvisited child got an inflated "
            "baseline when parent's own-POV Q was negative."
        )

    def test_fpu_reduction_scales_correctly(self):
        """Larger fpu_reduction should produce strictly-lower FPU baselines
        (more pessimistic about unvisited children), so a visited sibling with
        a fixed positive Q can beat an unvisited sibling under a larger
        reduction but lose under zero reduction."""
        # Configure so that at fpu=0 the unvisited child wins (equal priors,
        # visited q=0 vs unvisited q=0 → epsilon tie; we tilt via priors).
        # Parent.value() = 0 → parent-POV = 0 → FPU = -r * sqrt(0.3).
        visited_q = 0.0
        parent, visited, unvisited = _parent_with_two_children(
            visited_child_value=visited_q, visited_prior=0.3, unvisited_prior=0.7,
        )

        # With fpu=0: both q=0, unvisited has higher prior → wins.
        mcts_off = _build_mcts(fpu_reduction=0.0)
        np.random.seed(0)
        chosen_off = mcts_off._select_action(parent)
        assert chosen_off == "unvisited"

        # With fpu=0.25: unvisited's q drops by 0.25*sqrt(0.3) ≈ 0.137.
        # Visited keeps u_value = c_puct * 0.3 * sqrt(5) / (1+3) ≈ 0.168.
        # Unvisited u_value = c_puct * 0.7 * sqrt(5) / 1 ≈ 1.565.
        # Visited total ≈ 0 + 0.168 = 0.168; Unvisited total ≈ -0.137 + 1.565
        # ≈ 1.428. Unvisited STILL wins. FPU nudges but doesn't flip here.
        mcts_on = _build_mcts(fpu_reduction=0.25)
        np.random.seed(0)
        chosen_on = mcts_on._select_action(parent)
        assert chosen_on == "unvisited"

        # With fpu=2.0 (extreme): unvisited Q = -2.0 * sqrt(0.3) ≈ -1.095.
        # Unvisited total ≈ -1.095 + 1.565 = 0.470. Visited total ≈ 0.168.
        # Unvisited still wins here too because u_value dominates at these
        # priors/visits. Bump the visited child's Q instead to flip.
        parent, visited, unvisited = _parent_with_two_children(
            visited_child_value=+0.5, visited_prior=0.3, unvisited_prior=0.7,
        )
        mcts_extreme = _build_mcts(fpu_reduction=2.0)
        np.random.seed(0)
        chosen_extreme = mcts_extreme._select_action(parent)
        assert chosen_extreme == "visited", (
            "A high visited Q combined with a heavy FPU reduction should beat "
            "an unvisited sibling, even with a lower prior."
        )


class TestFpuInEndToEndSearch:
    def test_search_runs_with_fpu_enabled(self):
        """Smoke — a real search() call with FPU on doesn't crash and produces
        a valid policy."""
        mcts = _build_mcts(fpu_reduction=0.25)
        state = GameState()
        probs = mcts.search(state, move_number=0)
        assert probs.shape == (mcts.state_encoder.total_moves,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-3)

    def test_search_runs_with_fpu_disabled(self):
        mcts = _build_mcts(fpu_reduction=0.0)
        state = GameState()
        probs = mcts.search(state, move_number=0)
        assert probs.sum() == pytest.approx(1.0, abs=1e-3)
