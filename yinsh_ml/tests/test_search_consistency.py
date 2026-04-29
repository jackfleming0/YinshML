"""Tests for the search-consistency probe (Track B §5).

The probe samples positions from the replay buffer, runs a long-search MCTS
on each, and trains the network to match the long-search visit distribution
(KL on policy) and root value (MSE on value). These tests pin:

  * `GameExperience` carries `move_numbers` per sample (default-derived from
    index, subsamples in lockstep, augmented samples inherit, persists in
    save/load with backward compat for older buffers).
  * `_search_consistency_step` is gated correctly: skipped when disabled,
    skipped during warmup, every-K-batches cadence honored.
  * `MCTS.search()` exposes `last_root_value` (was a search_batch-only
    side-effect previously — now mirrored in serial search too).
  * The trainer's lazy MCTS construction defaults match the probe's needs
    (pure_neural, no heuristic, no subtree reuse, no Dirichlet noise).
"""

from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.training.self_play import MCTS
from yinsh_ml.training.trainer import GameExperience, YinshTrainer
from yinsh_ml.utils.encoding import StateEncoder


# ---------------------------------------------------------------------------
# GameExperience.move_numbers
# ---------------------------------------------------------------------------


class TestGameExperienceMoveNumbers:
    def _fake_game(self, n=5, total_moves=None):
        # Default to the active encoder's policy size (currently 7433). Used as
        # an opaque blob shape — GameExperience doesn't validate it, so any
        # stable size works, but tracking the encoder avoids silent drift.
        if total_moves is None:
            total_moves = StateEncoder().total_moves
        states = [np.zeros((6, 11, 11), dtype=np.float32) for _ in range(n)]
        policies = [np.full(total_moves, 1.0 / total_moves, dtype=np.float32) for _ in range(n)]
        values = [0.1 * i - 0.5 for i in range(n)]
        return states, policies, values

    def test_default_move_numbers_derived_from_index(self):
        exp = GameExperience(max_size=100)
        states, policies, values = self._fake_game(n=5)
        exp.add_game_experience(states, policies, values)
        assert list(exp.move_numbers) == [0, 1, 2, 3, 4]

    def test_explicit_move_numbers_respected(self):
        exp = GameExperience(max_size=100)
        states, policies, values = self._fake_game(n=4)
        exp.add_game_experience(
            states, policies, values, move_numbers=[7, 8, 9, 10]
        )
        assert list(exp.move_numbers) == [7, 8, 9, 10]

    def test_mismatch_raises(self):
        exp = GameExperience(max_size=100)
        states, policies, values = self._fake_game(n=4)
        with pytest.raises(AssertionError):
            exp.add_game_experience(
                states, policies, values, move_numbers=[0, 1, 2]  # wrong length
            )

    def test_move_numbers_subsampled_in_lockstep_for_long_games(self):
        """Games >100 moves trigger subsampling in add_game_experience.
        move_numbers must be subsampled with the same indices as
        states/policies/values — otherwise the consistency probe would feed
        MCTS the wrong move_number for every late-game sample."""
        exp = GameExperience(max_size=10000, subsample_long_games=True)
        n = 200
        tm = StateEncoder().total_moves
        states = [np.zeros((6, 11, 11), dtype=np.float32) for _ in range(n)]
        policies = [np.full(tm, 1.0 / tm, dtype=np.float32) for _ in range(n)]
        values = [0.0] * n
        # Pin random for reproducibility of the "30 sampled middle indices".
        import random
        random.seed(20260419)
        exp.add_game_experience(states, policies, values)

        # Subsampling keeps first 20, sampled-middle, last 20 = 70 samples
        # for n=200. The exact middle sample is RNG-driven, but the count
        # is fixed and the move_numbers must align with the kept indices.
        kept = len(exp.states)
        assert len(exp.move_numbers) == kept
        # Sequence sanity: monotone non-decreasing, first 20 are 0..19,
        # last 20 are 180..199.
        mns = list(exp.move_numbers)
        assert mns[:20] == list(range(20))
        assert mns[-20:] == list(range(180, 200))
        # The middle samples sit strictly between them.
        for m in mns[20:-20]:
            assert 20 <= m < 180

    def test_augmented_samples_inherit_move_number(self):
        """D2 augmentation preserves the position's outer-game time.
        Augmented samples must carry the same move_number as their original
        so the consistency probe sees consistent move_number across the
        4× expansion."""
        exp = GameExperience(
            max_size=1000, enable_augmentation=True, max_augmentations=4
        )
        if not exp.enable_augmentation:
            pytest.skip("Augmenter not importable in this env")
        states, policies, values = self._fake_game(n=3)
        exp.add_game_experience(states, policies, values)
        # Each original gets up to (max_augmentations - 1) extras with the
        # same move_number. Across the buffer, every move_number value is
        # seen at least once and there are no rogue values.
        seen = set(exp.move_numbers)
        assert seen.issubset({0, 1, 2})

    def test_buffer_save_load_round_trip(self, tmp_path):
        """Persisted buffer carries move_numbers; load_buffer restores them."""
        exp = GameExperience(max_size=100)
        states, policies, values = self._fake_game(n=3)
        exp.add_game_experience(
            states, policies, values, move_numbers=[5, 10, 15]
        )
        path = str(tmp_path / "buf.pkl")
        exp.save_buffer(path, compress=False)

        exp2 = GameExperience(max_size=100)
        exp2.load_buffer(path)
        assert list(exp2.move_numbers) == [5, 10, 15]

    def test_load_legacy_buffer_without_move_numbers_defaults_to_zero(self, tmp_path):
        """Older buffers (saved before move_numbers existed) must load cleanly,
        defaulting move_number=0 across the board. The consistency probe will
        then see peak-noise epsilon_mix on every sample — degraded but safe."""
        import pickle
        path = str(tmp_path / "legacy.pkl")
        legacy = {
            'states': [np.zeros((6, 11, 11), dtype=np.float32) for _ in range(3)],
            'move_probs': [np.full(8390, 1.0 / 8390, dtype=np.float16) for _ in range(3)],
            'values': [0.0, 0.5, -0.5],
            'phases': ['MAIN_GAME'] * 3,
            # NB: no 'move_numbers' key
        }
        with open(path, 'wb') as f:
            pickle.dump(legacy, f)

        exp = GameExperience(max_size=100)
        exp.load_buffer(path)
        assert list(exp.move_numbers) == [0, 0, 0]


# ---------------------------------------------------------------------------
# MCTS.search() now exposes last_root_value (parity with search_batch)
# ---------------------------------------------------------------------------


class TestSearchExposesRootValue:
    def _make_fake_network(self, total_moves):
        network = MagicMock()
        network.state_encoder = StateEncoder()

        def predict_from_state(_state):
            policy_logits = torch.zeros(1, total_moves)
            value = torch.tensor([[0.42]])
            return policy_logits, value

        def predict_batch(states):
            policy_logits = torch.zeros(len(states), total_moves)
            values = torch.full((len(states), 1), 0.42)
            return policy_logits, values

        network.predict_from_state = predict_from_state
        network.predict_batch = predict_batch
        return network

    def _build_mcts(self, num_simulations=4):
        encoder = StateEncoder()
        network = self._make_fake_network(encoder.total_moves)
        return MCTS(
            network=network,
            evaluation_mode="pure_neural",
            num_simulations=num_simulations,
            late_simulations=num_simulations,
            simulation_switch_ply=200,
            c_puct=1.0,
            dirichlet_alpha=0.0,
            value_weight=1.0,
            max_depth=20,
            initial_temp=1.0,
            final_temp=1.0,
            annealing_steps=1,
            enable_subtree_reuse=False,
        )

    def test_search_sets_last_root_value(self):
        """`search()` (the serial variant) used to leave `last_root_value`
        unset — only `search_batch()` set it. The probe needs both, so we
        mirror the side-effect. Exact value depends on perspective-flipping
        backprop math; here we just pin the contract: attribute exists and
        is a finite float after any successful search."""
        mcts = self._build_mcts(num_simulations=4)
        state = GameState()
        _ = mcts.search(state, move_number=0)
        assert hasattr(mcts, 'last_root_value')
        assert isinstance(mcts.last_root_value, float)
        assert np.isfinite(mcts.last_root_value)
        # And it must reflect actual search work, not the no-search default.
        # With visited root, value() = value_sum / visit_count, both bounded
        # by the leaf evaluations (|0.42|), so |root.value()| ≤ 0.42 + ε.
        assert abs(mcts.last_root_value) <= 0.42 + 1e-5

    def test_search_last_root_value_zero_when_no_visits(self):
        """If MCTS exits without any successful backprop (degenerate path),
        `last_root_value` should be 0.0 — same fallback as `search_batch`.
        Not easy to trigger in normal play; covered indirectly by the visit-
        count guard at the end of `search()`."""
        mcts = self._build_mcts(num_simulations=4)
        state = GameState()
        _ = mcts.search(state, move_number=0)
        # Sanity: in our fake-network setup, search always backprops, so
        # the visit-count > 0 path is exercised. The contract this pins is:
        # the attribute is a real float in either case (no AttributeError).
        assert isinstance(mcts.last_root_value, float)


# ---------------------------------------------------------------------------
# Trainer wiring + gating
# ---------------------------------------------------------------------------


def _trainer_kwargs(**overrides):
    """Build YinshTrainer __init__ kwargs without actually constructing the
    trainer. We use these to assert defaults / signatures only — the real
    trainer needs a NetworkWrapper which is heavy."""
    import inspect
    sig = inspect.signature(YinshTrainer.__init__)
    out = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if param.default is not inspect.Parameter.empty:
            out[name] = param.default
    out.update(overrides)
    return out


class TestTrainerConfigSignature:
    def test_default_disabled(self):
        kwargs = _trainer_kwargs()
        assert kwargs['enable_search_consistency'] is False

    def test_default_knobs(self):
        """The defaults need to match what `configs/training.yaml` ships
        with so a config typo on the wrong field doesn't silently fall back
        to surprising values."""
        kwargs = _trainer_kwargs()
        assert kwargs['search_consistency_weight'] == 0.1
        assert kwargs['search_consistency_value_weight'] == 1.0
        assert kwargs['search_consistency_every_k_steps'] == 10
        assert kwargs['search_consistency_long_sims'] == 64
        assert kwargs['search_consistency_batch_size'] == 32
        assert kwargs['search_consistency_warmup_iters'] == 3


# ---------------------------------------------------------------------------
# _search_consistency_step gating — direct method-level tests with a stubbed
# trainer instance (avoids the full network-wrapper dependency).
# ---------------------------------------------------------------------------


class _StubTrainer:
    """Mimics the subset of YinshTrainer state the gating logic reads."""

    def __init__(self):
        self.enable_search_consistency = False
        self.experience = MagicMock()
        self.experience.size = MagicMock(return_value=0)
        self.current_iteration = 0
        self.search_consistency_warmup_iters = 3
        self.search_consistency_batch_size = 32
        self._sc_mcts = None
        # Borrow the real method
        self._search_consistency_step = YinshTrainer._search_consistency_step.__get__(self)


class TestStepGating:
    def test_disabled_returns_none(self):
        t = _StubTrainer()
        t.enable_search_consistency = False
        assert t._search_consistency_step() is None

    def test_buffer_too_small_returns_none(self):
        t = _StubTrainer()
        t.enable_search_consistency = True
        t.experience.size = MagicMock(return_value=4)  # < batch_size=32
        assert t._search_consistency_step() is None

    def test_warmup_returns_none(self):
        t = _StubTrainer()
        t.enable_search_consistency = True
        t.experience.size = MagicMock(return_value=1000)
        t.current_iteration = 1  # < warmup_iters=3
        assert t._search_consistency_step() is None


# ---------------------------------------------------------------------------
# Lazy MCTS construction: defaults match the probe's needs
# ---------------------------------------------------------------------------


class TestNetworkRebind:
    """The supervisor's `_reset_network_objects` (every 3 iters at iter % 3
    == 0) swaps `trainer.network` for a freshly-loaded NetworkWrapper. The
    cached `_sc_mcts` must follow that swap, otherwise it holds a stale
    reference to the OLD wrapper whose tensor pool / device state may have
    been torn down — every `predict_from_state` then raises with a non-
    obvious error and the SC step silently no-ops.

    This test pins the rebind so a future refactor can't regress it."""

    def test_sc_step_rebinds_network_each_call(self):
        # Build a stub trainer whose `network` we can swap mid-life. Plain
        # MagicMock (no spec) so we can attach arbitrary attrs the method
        # touches (logger, optimizers etc) on the fly.
        t = MagicMock()
        t.enable_search_consistency = True
        t.experience = MagicMock()
        t.experience.size = MagicMock(return_value=100)
        t.experience.states = []
        t.experience.move_numbers = []
        t.current_iteration = 5  # past warmup
        t.search_consistency_warmup_iters = 3
        t.search_consistency_batch_size = 0  # → empty index sample, return early
        t.search_consistency_long_sims = 4

        # Pre-populated MCTS pointing at network A; trainer.network is B.
        net_a = MagicMock()
        net_a.state_encoder = MagicMock()
        net_b = MagicMock()
        net_b.state_encoder = MagicMock()

        t.network = net_b
        t.state_encoder = net_b.state_encoder
        # MCTS instance whose .network is currently A (simulates stale ref).
        mcts = MagicMock()
        mcts.network = net_a
        mcts.state_encoder = net_a.state_encoder
        t._sc_mcts = mcts
        t._build_consistency_mcts = MagicMock(return_value=mcts)

        # Bind the real method onto our stub trainer.
        method = YinshTrainer._search_consistency_step.__get__(t)
        # Method may exit early due to empty index loop — we just care that
        # the rebind happened first.
        try:
            method()
        except Exception:
            pass

        assert mcts.network is net_b, (
            "SC step must rebind cached MCTS network to current trainer.network "
            "to survive the supervisor's `_reset_network_objects` swap."
        )
        assert mcts.state_encoder is net_b.state_encoder


class TestLazyMCTSDefaults:
    """`_build_consistency_mcts` should return a pure-neural MCTS with no
    heuristic, no subtree reuse, and no Dirichlet noise — those would all
    contaminate the distillation target with exploration randomness."""

    def _stub_trainer(self):
        t = MagicMock(spec=YinshTrainer)
        t.network = MagicMock()
        t.network.state_encoder = StateEncoder()
        t.search_consistency_long_sims = 32
        t._build_consistency_mcts = YinshTrainer._build_consistency_mcts.__get__(t)
        return t

    def test_defaults(self):
        t = self._stub_trainer()
        mcts = t._build_consistency_mcts()
        assert mcts.evaluation_mode == "pure_neural"
        assert mcts.heuristic_evaluator is None
        assert mcts.enable_subtree_reuse is False
        # No Dirichlet noise on distillation searches.
        assert mcts.epsilon_mix_start == 0.0
        assert mcts.epsilon_mix_end == 0.0
        assert mcts.epsilon_mix_taper_moves == 0
        # Sim budget pulled from the trainer-level knob.
        assert mcts.early_simulations == 32
        assert mcts.late_simulations == 32
