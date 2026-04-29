"""Tests for the `epsilon_mix` taper on root Dirichlet noise.

Root-noise mixing fraction used to be hardcoded to 0.25 at every call site
— four copies in `training/self_play.py` + one in `search/mcts.py`. Now
parameterized as a `start → end` linear interpolation over
`epsilon_mix_taper_moves`. The taper stops injecting randomness into
late-game tactical positions (move ≥ taper_moves) while preserving early-
game exploration diversity.

Tests cover:
  * `_compute_epsilon_mix` linear interpolation math.
  * Boundary + degenerate configs (taper=0, negative move, past-taper).
  * `_apply_root_dirichlet_noise` on a reused root respects the taper:
    early moves perturb priors, late moves leave them untouched.
  * Parameter-matching across the config pipeline (MCTSConfig for
    search/mcts.py and MCTS.__init__ for training/self_play.py).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from yinsh_ml.training.self_play import MCTS, Node
from yinsh_ml.utils.encoding import StateEncoder


def _fake_network():
    encoder = StateEncoder()
    network = MagicMock()
    network.state_encoder = encoder
    network.predict_from_state = lambda _s: (
        torch.zeros(1, encoder.total_moves), torch.zeros(1, 1)
    )
    return network


def _build_mcts(
    epsilon_mix_start=0.25,
    epsilon_mix_end=0.0,
    epsilon_mix_taper_moves=20,
    dirichlet_alpha=0.3,
):
    return MCTS(
        network=_fake_network(),
        evaluation_mode="pure_neural",
        num_simulations=4,
        late_simulations=4,
        simulation_switch_ply=20,
        c_puct=1.0,
        dirichlet_alpha=dirichlet_alpha,
        value_weight=1.0,
        max_depth=50,
        initial_temp=1.0,
        final_temp=1.0,
        annealing_steps=1,
        temp_clamp_fraction=1.0,
        enable_subtree_reuse=False,
        epsilon_mix_start=epsilon_mix_start,
        epsilon_mix_end=epsilon_mix_end,
        epsilon_mix_taper_moves=epsilon_mix_taper_moves,
    )


class TestComputeEpsilonMix:
    def test_starts_at_start_value(self):
        mcts = _build_mcts(epsilon_mix_start=0.25, epsilon_mix_end=0.0, epsilon_mix_taper_moves=20)
        assert mcts._compute_epsilon_mix(0) == pytest.approx(0.25)

    def test_reaches_end_at_taper_moves(self):
        mcts = _build_mcts(epsilon_mix_start=0.25, epsilon_mix_end=0.0, epsilon_mix_taper_moves=20)
        assert mcts._compute_epsilon_mix(20) == pytest.approx(0.0)

    def test_half_way_is_linear_midpoint(self):
        mcts = _build_mcts(epsilon_mix_start=0.25, epsilon_mix_end=0.0, epsilon_mix_taper_moves=20)
        assert mcts._compute_epsilon_mix(10) == pytest.approx(0.125)

    def test_past_taper_is_clamped_at_end(self):
        mcts = _build_mcts(epsilon_mix_start=0.25, epsilon_mix_end=0.0, epsilon_mix_taper_moves=20)
        assert mcts._compute_epsilon_mix(100) == pytest.approx(0.0)
        assert mcts._compute_epsilon_mix(21) == pytest.approx(0.0)

    def test_negative_move_treated_as_zero(self):
        """Defensive — an out-of-range move_number shouldn't produce a
        value outside [start, end]."""
        mcts = _build_mcts(epsilon_mix_start=0.25, epsilon_mix_end=0.0, epsilon_mix_taper_moves=20)
        assert mcts._compute_epsilon_mix(-5) == pytest.approx(0.25)

    def test_taper_disabled_returns_start(self):
        """`epsilon_mix_taper_moves=0` means the taper is off; start value
        applies at every move_number."""
        mcts = _build_mcts(epsilon_mix_start=0.25, epsilon_mix_end=0.0, epsilon_mix_taper_moves=0)
        for mn in [0, 5, 50, 500]:
            assert mcts._compute_epsilon_mix(mn) == pytest.approx(0.25)

    def test_constant_schedule_via_equal_start_and_end(self):
        """A user that wants the old constant 0.25 behavior can set
        start == end; taper_moves becomes a no-op."""
        mcts = _build_mcts(epsilon_mix_start=0.25, epsilon_mix_end=0.25, epsilon_mix_taper_moves=20)
        for mn in [0, 10, 20, 50]:
            assert mcts._compute_epsilon_mix(mn) == pytest.approx(0.25)

    def test_increasing_taper_supported(self):
        """Nothing forces end < start — ramp *up* is also valid."""
        mcts = _build_mcts(epsilon_mix_start=0.1, epsilon_mix_end=0.5, epsilon_mix_taper_moves=10)
        assert mcts._compute_epsilon_mix(0) == pytest.approx(0.1)
        assert mcts._compute_epsilon_mix(5) == pytest.approx(0.3)
        assert mcts._compute_epsilon_mix(10) == pytest.approx(0.5)


def _parent_with_children(priors):
    parent = Node(state=None, c_puct=1.0)
    parent.is_expanded = True
    parent.visit_count = 3
    for i, p in enumerate(priors):
        child = Node(state=None, parent=parent, prior_prob=p, c_puct=1.0)
        parent.children[f"m{i}"] = child
    return parent


class TestApplyRootDirichletNoise:
    def test_early_move_perturbs_priors(self):
        """At move 0 with start=0.25, priors should shift from their
        deterministic initial values."""
        mcts = _build_mcts(epsilon_mix_start=0.25, epsilon_mix_end=0.0, epsilon_mix_taper_moves=20)
        parent = _parent_with_children([0.5, 0.5])
        before = [c.prior_prob for c in parent.children.values()]

        np.random.seed(0)
        mcts._apply_root_dirichlet_noise(parent, move_number=0)

        after = [c.prior_prob for c in parent.children.values()]
        assert before != after, "expected root noise to perturb priors at move 0"

    def test_past_taper_leaves_priors_untouched(self):
        """Once we're past the taper (move ≥ taper_moves) with end=0, noise
        should be suppressed and priors unchanged."""
        mcts = _build_mcts(epsilon_mix_start=0.25, epsilon_mix_end=0.0, epsilon_mix_taper_moves=20)
        parent = _parent_with_children([0.5, 0.5])
        before = [c.prior_prob for c in parent.children.values()]

        # Patch np.random.dirichlet so a regression that still samples the
        # noise would surface — the production code has to short-circuit
        # *before* the RNG call.
        orig = np.random.dirichlet
        np.random.dirichlet = lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("Dirichlet should not be sampled past taper with end=0")
        )
        try:
            mcts._apply_root_dirichlet_noise(parent, move_number=25)
        finally:
            np.random.dirichlet = orig

        after = [c.prior_prob for c in parent.children.values()]
        assert before == after

    def test_halfway_perturbs_less_than_start(self):
        """A sanity check on the gradient of the taper — at move=taper/2 the
        mixed prior should be *between* the original and the move=0 result."""
        parent_start = _parent_with_children([0.8, 0.2])
        parent_half = _parent_with_children([0.8, 0.2])
        originals = [0.8, 0.2]

        mcts = _build_mcts(epsilon_mix_start=0.5, epsilon_mix_end=0.0, epsilon_mix_taper_moves=10)

        np.random.seed(1234)
        mcts._apply_root_dirichlet_noise(parent_start, move_number=0)

        np.random.seed(1234)  # same noise draw
        mcts._apply_root_dirichlet_noise(parent_half, move_number=5)

        # At move 0, eps=0.5 — 50% mixing. At move 5, eps=0.25 — 25% mixing.
        # The "distance moved from original prior" should be smaller at move 5.
        dist_start = sum(abs(c.prior_prob - o) for c, o in zip(parent_start.children.values(), originals))
        dist_half = sum(abs(c.prior_prob - o) for c, o in zip(parent_half.children.values(), originals))
        assert dist_half < dist_start, (
            f"taper should shrink perturbation size (d0={dist_start}, d5={dist_half})"
        )


class TestConstructorDefaults:
    def test_defaults_match_katago_at_move_zero(self):
        mcts = _build_mcts()
        assert mcts.epsilon_mix_start == pytest.approx(0.25)
        assert mcts.epsilon_mix_end == pytest.approx(0.0)
        assert mcts.epsilon_mix_taper_moves == 20
        assert mcts._compute_epsilon_mix(0) == pytest.approx(0.25)

    def test_mctsconfig_exposes_same_params(self):
        """The secondary MCTS (search/mcts.py) via MCTSConfig must carry
        matching field names so both code paths share a config vocabulary."""
        from yinsh_ml.search.mcts import MCTSConfig
        c = MCTSConfig()
        assert c.epsilon_mix_start == pytest.approx(0.25)
        assert c.epsilon_mix_end == pytest.approx(0.0)
        assert c.epsilon_mix_taper_moves == 20

    def test_search_mcts_helper_matches_training_helper(self):
        """The taper formula lives in two places (search/mcts.py and
        training/self_play.py) because of the MCTS split. Pin that they
        compute identical values so drift between the two doesn't silently
        change arena behavior vs. training behavior."""
        from yinsh_ml.search.mcts import MCTS as SearchMCTS, MCTSConfig
        from unittest.mock import MagicMock

        cfg = MCTSConfig(
            num_simulations=4, dirichlet_alpha=0.3,
            epsilon_mix_start=0.3, epsilon_mix_end=0.05, epsilon_mix_taper_moves=15,
            use_heuristic_evaluation=False,
        )
        # search/mcts.py MCTS builds a YinshHeuristics by default — skip that
        # by disabling heuristic eval and mocking the network.
        network = MagicMock()
        network.predict = lambda _s: (
            np.ones(SearchMCTS(network=MagicMock(), config=cfg).state_encoder.total_moves)
            / SearchMCTS(network=MagicMock(), config=cfg).state_encoder.total_moves,
            0.0,
        )
        search_mcts = SearchMCTS(network=network, config=cfg)

        train_mcts = _build_mcts(
            epsilon_mix_start=0.3, epsilon_mix_end=0.05, epsilon_mix_taper_moves=15,
        )

        for mn in [0, 5, 10, 15, 30]:
            a = search_mcts._compute_epsilon_mix(mn)
            b = train_mcts._compute_epsilon_mix(mn)
            assert a == pytest.approx(b), f"diverged at move {mn}: search={a}, train={b}"
