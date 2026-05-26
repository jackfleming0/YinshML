"""Regression tests for cross-encoder phase decoding.

Background — 2026-05-26 B1+B2+B3 post-mortem: the trainer's
`decode_phase` helper read `state[5]` unconditionally. That's correct
for the 6-channel basic encoder (CH_GAME_PHASE=5) but wrong for the
15-channel enhanced encoder (CH_GAME_PHASE=12; channel 5 is a row-threat
channel). Because row-threats are sparse on most boards, the avg-abs
classifier always landed below the 0.2 threshold and labelled every
sample RING_PLACEMENT.

Blast radius: the silently-mislabelled `phases` field is consumed by
ReplayBuffer.sample_batch's phase-aware weighting (trainer.py:446-447).
The configured `phase_weights: {MAIN_GAME: 2.0}` boost was silently
disabled for every 15-channel run (D.2 and B1+B2+B3), so MAIN_GAME
positions were under-sampled by 2× throughout.

The fix routes both encodings through `decode_phase_from_state`, which
uses `phase_channel_index(num_channels)` as the single source of truth
for the phase channel location. These tests pin that contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from yinsh_ml.utils.encoding import (
    StateEncoder,
    phase_channel_index,
    decode_phase_from_state,
)
from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder


PHASE_VALUES = {
    "RING_PLACEMENT": 0.0,
    "MAIN_GAME": 0.5,
    "RING_REMOVAL": 1.0,
}


def _make_state(num_channels: int, phase_value: float) -> np.ndarray:
    """Build a minimal state tensor with the phase channel set to a uniform
    `phase_value` broadcast (matching how the encoders write the channel).
    Other channels are left at zero — they don't influence the decoder.
    """
    s = np.zeros((num_channels, 11, 11), dtype=np.float32)
    ch = phase_channel_index(num_channels)
    s[ch] = phase_value
    return s


@pytest.mark.parametrize("phase_label, phase_value", PHASE_VALUES.items())
def test_decode_phase_basic_encoder(phase_label: str, phase_value: float):
    """6-channel encoder: phase channel is CH_GAME_PHASE = 5."""
    state = _make_state(StateEncoder.NUM_CHANNELS, phase_value)
    assert decode_phase_from_state(state) == phase_label


@pytest.mark.parametrize("phase_label, phase_value", PHASE_VALUES.items())
def test_decode_phase_enhanced_encoder(phase_label: str, phase_value: float):
    """15-channel encoder: phase channel is CH_GAME_PHASE = 12. This is
    the case that previously failed silently — `decode_phase` read
    state[5] which is a row-threat channel.
    """
    state = _make_state(EnhancedStateEncoder.NUM_CHANNELS, phase_value)
    assert decode_phase_from_state(state) == phase_label


def test_phase_channel_index_named_constants_align():
    """The named CH_GAME_PHASE constants on each encoder MUST match what
    `phase_channel_index` returns. If you renumber channels in an encoder,
    update the constant; if you add an encoder, extend the helper. This
    test fails loudly if either side drifts.
    """
    assert phase_channel_index(StateEncoder.NUM_CHANNELS) == StateEncoder.CH_GAME_PHASE
    assert phase_channel_index(EnhancedStateEncoder.NUM_CHANNELS) == EnhancedStateEncoder.CH_GAME_PHASE


def test_phase_channel_index_unknown_encoder_raises():
    """A new encoder (different channel count) must announce itself —
    silently returning '5' the way the old magic-index code did is the
    failure mode we're trying to prevent forever.
    """
    with pytest.raises(ValueError, match=r"unknown encoder"):
        phase_channel_index(7)
    with pytest.raises(ValueError, match=r"unknown encoder"):
        phase_channel_index(20)


def test_15ch_channel5_no_longer_classifies_phase():
    """The specific failure mode: a 15-channel state with arbitrary
    row-threat data in channel 5 must NOT influence phase classification.
    Phase comes from channel 12 only.
    """
    state = _make_state(EnhancedStateEncoder.NUM_CHANNELS, PHASE_VALUES["MAIN_GAME"])
    # Set the OLD (wrong) phase-read channel to a high value — under the
    # old logic this would have flipped the classification.
    state[5] = 1.0  # row-threat channel, max value everywhere
    # Phase channel still says MAIN_GAME, and that's what wins.
    assert decode_phase_from_state(state) == "MAIN_GAME"
