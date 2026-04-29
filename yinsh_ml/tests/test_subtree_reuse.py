"""Tests for MCTS subtree reuse (Track A polish item).

Covers:
  * `advance_root` on a known move promotes the child to cached root, frees siblings.
  * `advance_root` on an unknown move clears the cache.
  * `reset_tree` clears the cache and returns pooled states.
  * With reuse enabled, `_cached_root` survives across `search()` calls and
    subsequent searches accumulate visits on top of the carried tree.
  * With reuse disabled, `_cached_root` stays None and every search runs
    against a fresh root.
  * Dirichlet noise gets fresh application on a reused (already-expanded) root.
  * play-style sequence (search → advance → search) preserves tree continuity.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.training.self_play import MCTS, Node
from yinsh_ml.utils.encoding import StateEncoder


def _make_fake_network(total_moves):
    """Minimal NetworkWrapper stand-in: uniform policy logits, zero value.
    `predict_from_state` is what the training MCTS calls under `pure_neural`."""
    network = MagicMock()
    network.state_encoder = StateEncoder()
    assert network.state_encoder.total_moves == total_moves

    def predict_from_state(_state):
        policy_logits = torch.zeros(1, total_moves)  # softmax → uniform
        value = torch.zeros(1, 1)
        return policy_logits, value

    network.predict_from_state = predict_from_state
    return network


def _build_mcts(enable_subtree_reuse=True, num_simulations=8, dirichlet_alpha=0.0):
    """Build a pure-neural MCTS with a fake network. Dirichlet alpha=0 by default
    so we can assert determinism of the tree structure in most tests; override
    in the noise-specific test."""
    encoder = StateEncoder()
    network = _make_fake_network(encoder.total_moves)
    return MCTS(
        network=network,
        evaluation_mode="pure_neural",
        num_simulations=num_simulations,
        late_simulations=num_simulations,
        simulation_switch_ply=20,
        c_puct=1.0,
        dirichlet_alpha=dirichlet_alpha,
        value_weight=1.0,
        max_depth=50,
        initial_temp=1.0,
        final_temp=1.0,  # no annealing — keeps outputs comparable across calls
        annealing_steps=1,
        temp_clamp_fraction=1.0,
        enable_subtree_reuse=enable_subtree_reuse,
    )


class TestAdvanceRootMechanics:
    """Unit tests on advance_root logic, using a hand-built Node tree so we're
    not coupled to the search() code path."""

    def _build_tree_with_two_children(self):
        root = Node(state=None, c_puct=1.0)
        child_a = Node(state=None, parent=root, prior_prob=0.6, c_puct=1.0)
        child_b = Node(state=None, parent=root, prior_prob=0.4, c_puct=1.0)
        root.children = {"A": child_a, "B": child_b}
        root.is_expanded = True
        root.visit_count = 10
        child_a.visit_count = 6
        child_b.visit_count = 4
        return root, child_a, child_b

    def test_advance_on_known_move_promotes_child(self):
        mcts = _build_mcts()
        root, child_a, child_b = self._build_tree_with_two_children()
        mcts._cached_root = root

        mcts.advance_root("A")

        assert mcts._cached_root is child_a
        assert child_a.parent is None
        # Siblings got cleared (clear_tree wipes their children/parent/state).
        assert child_b.children == {}
        assert child_b.parent is None

    def test_advance_on_unknown_move_clears_cache(self):
        mcts = _build_mcts()
        root, _, _ = self._build_tree_with_two_children()
        mcts._cached_root = root

        mcts.advance_root("Z")  # Not in children

        assert mcts._cached_root is None

    def test_advance_with_reuse_disabled_is_noop(self):
        mcts = _build_mcts(enable_subtree_reuse=False)
        # Pretend a tree got into cache somehow — even then, advance_root should
        # not touch it (search() does per-call cleanup in the disabled path).
        root, child_a, _ = self._build_tree_with_two_children()
        mcts._cached_root = root

        mcts.advance_root("A")

        # No-op: cached_root untouched, tree structure intact.
        assert mcts._cached_root is root
        assert root.children["A"] is child_a

    def test_advance_with_no_cache_is_noop(self):
        mcts = _build_mcts()
        assert mcts._cached_root is None
        mcts.advance_root("anything")  # Must not raise.
        assert mcts._cached_root is None

    def test_reset_tree_clears_cache(self):
        mcts = _build_mcts()
        root, _, _ = self._build_tree_with_two_children()
        mcts._cached_root = root

        mcts.reset_tree()

        assert mcts._cached_root is None
        # Root got fully cleared (children emptied).
        assert root.children == {}

    def test_advance_root_strips_stale_dirichlet_flag(self):
        """Kept subtree must start fresh w.r.t. root-noise tracking so the next
        search applies Dirichlet again."""
        mcts = _build_mcts()
        root, child_a, _ = self._build_tree_with_two_children()
        child_a.dirichlet_applied = True  # stale flag
        mcts._cached_root = root

        mcts.advance_root("A")

        assert not hasattr(mcts._cached_root, "dirichlet_applied")


class TestSearchWithReuse:
    """Integration tests that actually drive search() end-to-end."""

    def test_first_search_caches_root(self):
        mcts = _build_mcts(enable_subtree_reuse=True, num_simulations=4)
        state = GameState()
        mcts.search(state, move_number=0)
        assert mcts._cached_root is not None
        assert mcts._cached_root.is_expanded
        # Root accumulated visits.
        assert mcts._cached_root.visit_count > 0

    def test_reuse_disabled_drops_root_every_call(self):
        mcts = _build_mcts(enable_subtree_reuse=False, num_simulations=4)
        state = GameState()
        mcts.search(state, move_number=0)
        assert mcts._cached_root is None

    def test_sequential_searches_accumulate_visits(self):
        """Second search on the same root must start from existing visit counts
        rather than rebuild from zero — this is the core correctness guarantee
        of subtree reuse."""
        mcts = _build_mcts(enable_subtree_reuse=True, num_simulations=4)
        state = GameState()

        mcts.search(state, move_number=0)
        assert mcts._cached_root is not None
        visits_after_first = mcts._cached_root.visit_count
        root_after_first = mcts._cached_root

        mcts.search(state, move_number=0)
        assert mcts._cached_root is root_after_first  # same object reused
        assert mcts._cached_root.visit_count > visits_after_first

    def test_advance_then_search_carries_forward_subtree_visits(self):
        """After playing a move, the child chosen as the new root should already
        carry its prior visits — so the next search doesn't start from zero
        under the new root."""
        mcts = _build_mcts(enable_subtree_reuse=True, num_simulations=16)
        state = GameState()

        mcts.search(state, move_number=0)
        assert mcts._cached_root is not None
        # Pick the most-visited child as the "played" move (typical of greedy
        # temp=0 play).
        children = mcts._cached_root.children
        played_move, played_child = max(children.items(), key=lambda kv: kv[1].visit_count)
        child_prior_visits = played_child.visit_count
        assert child_prior_visits >= 1, "sanity: child should have been visited at least once"

        state.make_move(played_move)
        mcts.advance_root(played_move)

        # New cached root = the promoted child, with its accumulated visits intact.
        assert mcts._cached_root is played_child
        assert mcts._cached_root.visit_count == child_prior_visits

        # Next search adds *on top of* the carried visits.
        mcts.search(state, move_number=1)
        assert mcts._cached_root.visit_count > child_prior_visits

    def test_reuse_disabled_advance_is_noop_for_cache(self):
        mcts = _build_mcts(enable_subtree_reuse=False, num_simulations=4)
        state = GameState()
        mcts.search(state, move_number=0)  # clears tree at end
        assert mcts._cached_root is None
        mcts.advance_root("whatever")  # must be safe
        assert mcts._cached_root is None

    def test_reset_tree_during_game(self):
        """reset_tree should drop the current tree so the next search starts
        fresh — e.g., for game-end cleanup in the worker."""
        mcts = _build_mcts(enable_subtree_reuse=True, num_simulations=4)
        state = GameState()
        mcts.search(state, move_number=0)
        assert mcts._cached_root is not None

        mcts.reset_tree()
        assert mcts._cached_root is None

        mcts.search(state, move_number=0)
        assert mcts._cached_root is not None  # fresh cache


class TestDirichletNoiseOnReusedRoot:
    """When the root carries over, the normal expansion path (which adds
    Dirichlet) is skipped — _apply_root_dirichlet_noise fills the gap."""

    def test_noise_reaches_reused_root(self):
        # Use a budget big enough that the most-visited child gets expanded on
        # the first search, so the promoted kept-subtree root is already
        # expanded when the second search calls _apply_root_dirichlet_noise.
        mcts = _build_mcts(enable_subtree_reuse=True, num_simulations=64, dirichlet_alpha=0.3)
        state = GameState()

        mcts.search(state, move_number=0)
        # Pick the most-visited child — guaranteed to have been expanded,
        # since MCTS expands a child on the sim that first descends into it.
        played_move, played_child = max(
            mcts._cached_root.children.items(), key=lambda kv: kv[1].visit_count
        )
        assert played_child.is_expanded, "test setup: promoted root must be pre-expanded"
        state.make_move(played_move)
        mcts.advance_root(played_move)

        reused_root = mcts._cached_root
        priors_before = {m: c.prior_prob for m, c in reused_root.children.items()}

        np.random.seed(1234)  # make noise deterministic for the assertion
        mcts.search(state, move_number=1)

        priors_after = {m: c.prior_prob for m, c in reused_root.children.items()}
        # At least one prior must have shifted — noise mixing should move them.
        changed = any(priors_before[m] != priors_after[m] for m in priors_before)
        assert changed, "Dirichlet noise should perturb at least one reused-root prior"

    def test_no_noise_when_alpha_is_zero(self):
        """Degenerate guard: alpha=0 means no exploration noise even on a
        reused root. _apply_root_dirichlet_noise must short-circuit."""
        mcts = _build_mcts(enable_subtree_reuse=True, num_simulations=16, dirichlet_alpha=0.0)
        state = GameState()
        mcts.search(state, move_number=0)
        first_move = next(iter(mcts._cached_root.children))
        state.make_move(first_move)
        mcts.advance_root(first_move)

        reused_root = mcts._cached_root
        priors_before = {m: c.prior_prob for m, c in reused_root.children.items()}
        # Monkey-patch np.random.dirichlet so a bug that calls it would raise.
        # (Extra safety net on top of the alpha<=0 short-circuit.)
        orig = np.random.dirichlet
        def _fail(*a, **kw):
            raise AssertionError("Dirichlet should not be sampled when alpha<=0")
        np.random.dirichlet = _fail
        try:
            mcts.search(state, move_number=1)
        finally:
            np.random.dirichlet = orig

        priors_after = {m: c.prior_prob for m, c in reused_root.children.items()}
        # Priors untouched by noise — whatever search did to them via visits
        # doesn't rewrite `prior_prob` (only value_sum/visit_count change).
        for m in priors_before:
            assert priors_before[m] == priors_after[m]
