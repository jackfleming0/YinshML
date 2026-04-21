"""Tests for the EMA shadow used by the tournament eval gate.

Covers:
  * Decay math: shadow tracks a known-good analytical average.
  * Buffer handling: BN running stats (float) get EMA'd; integer buffers
    (num_batches_tracked) are copied through.
  * swap_into / restore round-trip: weights leave the module unchanged.
  * state_dict round-trip: save + reload recovers the shadow.
  * Tournament wiring: `_load_model` prefers `_ema.pt` sibling when present
    and `use_ema_for_eval=True`.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from yinsh_ml.training.ema import EMAShadow


class _Tiny(nn.Module):
    """Minimal module with a float parameter + a BN layer, so we exercise
    both parameter and buffer paths (including the int `num_batches_tracked`).
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.bn = nn.BatchNorm1d(2)


class TestDecayMath:
    """shadow ← decay·shadow + (1−decay)·live, applied each update()."""

    def test_single_update_closed_form(self):
        m = _Tiny()
        # Pin live weights to a known value.
        with torch.no_grad():
            m.linear.weight.fill_(1.0)
            m.linear.bias.fill_(0.0)
        ema = EMAShadow(m, decay=0.9)
        # Mutate live, then update shadow once. Shadow should move 10% of
        # the way toward the new live values.
        with torch.no_grad():
            m.linear.weight.fill_(2.0)
        ema.update()
        assert torch.allclose(
            ema.shadow['linear.weight'],
            torch.full_like(ema.shadow['linear.weight'], 0.9 * 1.0 + 0.1 * 2.0),
        )

    def test_many_updates_converge(self):
        m = _Tiny()
        with torch.no_grad():
            m.linear.weight.fill_(0.0)
        ema = EMAShadow(m, decay=0.9)
        with torch.no_grad():
            m.linear.weight.fill_(1.0)
        for _ in range(100):
            ema.update()
        # After many steps at a constant target the shadow converges to it.
        assert torch.allclose(
            ema.shadow['linear.weight'],
            torch.ones_like(ema.shadow['linear.weight']),
            atol=1e-4,
        )

    def test_shadow_does_not_alias_live(self):
        m = _Tiny()
        ema = EMAShadow(m, decay=0.9)
        with torch.no_grad():
            m.linear.weight.fill_(99.0)
        # Shadow was cloned at construction and should be unaffected.
        assert not torch.allclose(ema.shadow['linear.weight'],
                                   m.linear.weight.detach())


class TestBufferHandling:
    """BatchNorm's float buffers get EMA'd; int buffers are copied."""

    def test_int_buffer_is_copied_not_averaged(self):
        m = _Tiny()
        ema = EMAShadow(m, decay=0.9)
        # nn.BatchNorm1d tracks batches as an int; shouldn't be averaged.
        with torch.no_grad():
            m.bn.num_batches_tracked.fill_(7)
        ema.update()
        assert ema.shadow['bn.num_batches_tracked'].item() == 7

    def test_running_mean_is_averaged(self):
        m = _Tiny()
        # Force a known running_mean on the live module.
        with torch.no_grad():
            m.bn.running_mean.fill_(0.0)
        ema = EMAShadow(m, decay=0.5)
        with torch.no_grad():
            m.bn.running_mean.fill_(1.0)
        ema.update()
        assert torch.allclose(
            ema.shadow['bn.running_mean'],
            torch.full_like(ema.shadow['bn.running_mean'], 0.5),
        )


class TestSwapRestore:
    """swap_into then restore should leave the module byte-identical."""

    def test_roundtrip_preserves_live_weights(self):
        m = _Tiny()
        torch.manual_seed(0)
        for _ in range(3):  # scramble params a bit
            for p in m.parameters():
                with torch.no_grad():
                    p.copy_(torch.randn_like(p))
        # Freeze a snapshot of the live weights.
        live_snapshot = {k: v.detach().clone() for k, v in m.state_dict().items()}
        ema = EMAShadow(m, decay=0.9)
        # Mutate shadow so it differs from live.
        with torch.no_grad():
            for k in ema.shadow:
                if ema.shadow[k].is_floating_point():
                    ema.shadow[k].add_(1.0)
        ema.swap_into(m)
        # Live now matches shadow, not the snapshot.
        for k in live_snapshot:
            if live_snapshot[k].is_floating_point():
                assert not torch.allclose(m.state_dict()[k], live_snapshot[k])
        ema.restore(m)
        # After restore, every tensor matches the snapshot again.
        for k, v in live_snapshot.items():
            assert torch.allclose(m.state_dict()[k], v)

    def test_double_swap_raises(self):
        m = _Tiny()
        ema = EMAShadow(m, decay=0.9)
        ema.swap_into(m)
        with pytest.raises(RuntimeError, match="swap_into called twice"):
            ema.swap_into(m)
        # Cleanup so other tests don't leak a pending backup.
        ema.restore(m)

    def test_restore_without_swap_is_noop(self):
        m = _Tiny()
        ema = EMAShadow(m, decay=0.9)
        # Should not raise, even though nothing was swapped in.
        ema.restore(m)


class TestToleratesMissingBuffers:
    """Regression: `supervisor.py::train_iteration` nulls out BN buffers
    in the post-iteration cleanup, so by the time iter N+1 saves the
    EMA checkpoint, `module.state_dict()` has fewer keys than the shadow
    (which was built when buffers existed). swap_into must not crash."""

    def test_swap_with_module_missing_keys(self):
        m = _Tiny()
        ema = EMAShadow(m, decay=0.9)
        # Simulate the supervisor's cleanup: None the BN running stats.
        m.bn._buffers['running_mean'] = None
        m.bn._buffers['running_var'] = None
        m.bn._buffers['num_batches_tracked'] = None
        # Shadow still has those keys but module no longer exposes them.
        live_keys = set(m.state_dict().keys())
        shadow_keys = set(ema.shadow.keys())
        assert shadow_keys - live_keys, \
            "test setup wrong — shadow should have extra keys"
        # Should not raise.
        ema.swap_into(m)
        ema.restore(m)


class TestStateDictRoundtrip:
    def test_save_and_reload(self):
        m = _Tiny()
        ema = EMAShadow(m, decay=0.7)
        with torch.no_grad():
            for k in ema.shadow:
                if ema.shadow[k].is_floating_point():
                    ema.shadow[k].fill_(0.42)
        saved = ema.state_dict()
        # Fresh EMA on a fresh module, then load the saved state.
        m2 = _Tiny()
        ema2 = EMAShadow(m2, decay=0.9)
        ema2.load_state_dict(saved)
        assert ema2.decay == pytest.approx(0.7)
        for k in ema.shadow:
            if ema.shadow[k].is_floating_point():
                assert torch.allclose(ema2.shadow[k], ema.shadow[k])


class TestRejectsBadDecay:
    @pytest.mark.parametrize("bad", [-0.1, 0.0, 1.0, 1.5])
    def test_decay_out_of_range(self, bad):
        m = _Tiny()
        with pytest.raises(ValueError, match="decay"):
            EMAShadow(m, decay=bad)


class TestTournamentPrefersEma:
    """Integration-light: the tournament's _load_model picks the EMA
    sibling when present and the flag is on. Uses a mock NetworkWrapper
    so we don't need a real checkpoint on disk."""

    def test_picks_ema_when_flag_on_and_sibling_exists(self, tmp_path):
        from yinsh_ml.utils.tournament import ModelTournament

        regular = tmp_path / "checkpoint_iteration_3.pt"
        regular.write_text("")  # placeholder
        ema = tmp_path / "checkpoint_iteration_3_ema.pt"
        ema.write_text("")

        tm = ModelTournament(
            training_dir=tmp_path, device='cpu',
            use_ema_for_eval=True,
        )
        with patch('yinsh_ml.utils.tournament.NetworkWrapper') as WrapperCls:
            instance = MagicMock()
            WrapperCls.return_value = instance
            tm._load_model(regular)
            instance.load_model.assert_called_once_with(str(ema))

    def test_picks_regular_when_flag_off(self, tmp_path):
        from yinsh_ml.utils.tournament import ModelTournament

        regular = tmp_path / "checkpoint_iteration_3.pt"
        regular.write_text("")
        ema = tmp_path / "checkpoint_iteration_3_ema.pt"
        ema.write_text("")

        tm = ModelTournament(
            training_dir=tmp_path, device='cpu',
            use_ema_for_eval=False,
        )
        with patch('yinsh_ml.utils.tournament.NetworkWrapper') as WrapperCls:
            instance = MagicMock()
            WrapperCls.return_value = instance
            tm._load_model(regular)
            instance.load_model.assert_called_once_with(str(regular))

    def test_falls_back_to_regular_when_ema_missing(self, tmp_path):
        from yinsh_ml.utils.tournament import ModelTournament

        regular = tmp_path / "checkpoint_iteration_3.pt"
        regular.write_text("")

        tm = ModelTournament(
            training_dir=tmp_path, device='cpu',
            use_ema_for_eval=True,
        )
        with patch('yinsh_ml.utils.tournament.NetworkWrapper') as WrapperCls:
            instance = MagicMock()
            WrapperCls.return_value = instance
            tm._load_model(regular)
            instance.load_model.assert_called_once_with(str(regular))
