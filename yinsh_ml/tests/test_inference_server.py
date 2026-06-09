"""Tests for the process-based shared inference server (E20 throughput).

Two layers:

1. `test_predict_batch_encoded_matches_predict_batch` — the encode/forward
   split must be numerically identical to `predict_batch`. This is the
   correctness contract the whole design rests on: workers encode, the server
   runs `predict_batch_encoded`, and the result must equal what a single
   process would have computed with `predict_batch`. No subprocesses — fast.

2. `test_inference_server_roundtrip` (slow) — spin up a real server process and
   drive it through `ProcessEvaluatorClient` from the parent (acting as a
   worker). Validates the IPC transport, coalescing, and result routing
   end-to-end against a direct `predict_batch` reference.
"""

import copy
import os
import random
import tempfile

import numpy as np
import pytest
import torch

from yinsh_ml.network.wrapper import NetworkWrapper
from yinsh_ml.game.game_state import GameState


def _make_states(n, seed=0):
    """A few distinct, legal positions via a seeded random playout."""
    rng = random.Random(seed)
    gs = GameState()
    states = [copy.deepcopy(gs)]
    guard = 0
    while len(states) < n and guard < 2000:
        guard += 1
        if gs.is_terminal():
            gs = GameState()
            continue
        moves = gs.get_valid_moves()
        if not moves:
            gs = GameState()
            continue
        gs.make_move(rng.choice(moves))
        states.append(copy.deepcopy(gs))
    assert len(states) == n, f"only generated {len(states)} states"
    return states


def test_predict_batch_encoded_matches_predict_batch():
    wrapper = NetworkWrapper(device="cpu")
    wrapper.network.eval()
    states = _make_states(5, seed=1)

    logits_a, values_a = wrapper.predict_batch(states)

    encoded = np.stack(
        [wrapper.state_encoder.encode_state(s).astype(np.float32) for s in states]
    )
    logits_b, values_b = wrapper.predict_batch_encoded(encoded)

    assert logits_a.shape == logits_b.shape
    assert values_a.shape == values_b.shape
    assert torch.allclose(logits_a, logits_b, atol=1e-5), "encoded forward diverged from predict_batch (logits)"
    assert torch.allclose(values_a, values_b, atol=1e-5), "encoded forward diverged from predict_batch (values)"


def test_predict_batch_encoded_rejects_wrong_channel_count():
    wrapper = NetworkWrapper(device="cpu")  # basic = 6 channels
    bad = np.zeros((2, 15, 11, 11), dtype=np.float32)
    with pytest.raises(ValueError):
        wrapper.predict_batch_encoded(bad)


@pytest.mark.slow
def test_inference_server_roundtrip():
    import multiprocessing as mp

    from yinsh_ml.network.inference_server import (
        run_inference_server,
        ProcessEvaluatorClient,
    )

    ctx = mp.get_context("spawn")

    wrapper = NetworkWrapper(device="cpu")
    wrapper.network.eval()
    states = _make_states(6, seed=2)
    ref_logits, ref_values = wrapper.predict_batch(states)

    # Persist weights the same way SelfPlay.generate_games does (bare
    # state_dict), so the server's load_model path is exercised identically.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        torch.save(wrapper.network.state_dict(), tmp.name)
        model_path = tmp.name

    request_queue = ctx.Queue()
    response_queues = [ctx.Queue()]
    stop_event = ctx.Event()
    ready_event = ctx.Event()
    server_cfg = {
        "use_enhanced_encoding": False,
        "value_head_type": None,
        "max_batch": 64,
        "max_wait_ms": 5.0,
    }

    server = ctx.Process(
        target=run_inference_server,
        args=(model_path, server_cfg, request_queue, response_queues, stop_event, ready_event),
        daemon=True,
    )
    server.start()
    try:
        assert ready_event.wait(timeout=180), "server never became ready"
        client = ProcessEvaluatorClient(0, request_queue, response_queues[0], wrapper.state_encoder)

        logits, values = client.evaluate_batch(states)
        # The reference runs on CPU; the server may run on CUDA (this test is
        # also exercised on GPU boxes). fp32 matmul accumulation differs across
        # devices, so raw logits diverge by ~1e-2 even though the transport is
        # bit-exact — what must hold is the *decision* (argmax) and value
        # agreement within device precision, not bit-equality of CPU vs GPU.
        assert torch.equal(ref_logits.argmax(dim=1), logits.argmax(dim=1)), \
            "server changed the argmax move vs reference"
        assert torch.allclose(ref_logits, logits, atol=1e-2, rtol=1e-3), "server logits diverged beyond device precision"
        assert torch.allclose(ref_values, values, atol=1e-3, rtol=1e-3), "server values diverged beyond device precision"

        # A second call confirms the per-worker seq accounting holds across
        # multiple in-flight cycles.
        logits2, values2 = client.evaluate_batch(states[:2])
        assert logits2.shape[0] == 2
    finally:
        stop_event.set()
        server.join(timeout=20)
        if server.is_alive():
            server.terminate()
        try:
            os.unlink(model_path)
        except OSError:
            pass
