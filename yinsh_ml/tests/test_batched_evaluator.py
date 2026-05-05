"""Unit tests for BatchedEvaluator (Phases 0-1 of IMPLEMENTATION_PLAN.md).

Phase 0 (this module's primary scope) — three correctness pieces:

  T1. *Coalescence works.* N concurrent `evaluate(...)` calls collapse
      into a small number of `network.predict_batch` calls — not N
      separate ones. This is the whole reason the evaluator exists.

  T2. *Hard cap on batch size.* `max_batch` is respected even when the
      queue has more requests waiting. A regression here would silently
      blow up GPU memory usage at high concurrency.

  T3. *Routing correctness.* Each future receives the result for *its*
      input, not a sibling's. This is the determinism story; if it
      fails, MCTS will silently train on cross-talked values.

Phase 1 — integration into self_play.MCTS:

  T4. *MCTS-with-evaluator matches direct path.* Routing through the
      evaluator must not change search outcomes. Given the same seed,
      `search_batch` should produce identical visit counts and move
      probabilities whether `evaluator=None` or
      `evaluator=BatchedEvaluator(network)`.

These tests do not need a GPU — they use recording fake networks. The
real-network smoke is in Phase 3 (Mac + cloud sweep).
"""

from __future__ import annotations

import random
import threading
from typing import List

import numpy as np
import pytest
import torch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.network.batched_evaluator import BatchedEvaluator
from yinsh_ml.utils.encoding import StateEncoder


TOTAL_MOVES = StateEncoder().total_moves


# --------------------------------------------------------------------------- #
# Fakes                                                                       #
# --------------------------------------------------------------------------- #


class CountingFakeNetwork:
    """Records every `predict_batch` call's batch size; returns zeros.

    Mirrors `RecordingNetwork` from test_gpu_scaling_investigation.py but
    without the single-state `predict` paths the evaluator doesn't use.
    """

    def __init__(self):
        self.state_encoder = StateEncoder()
        self.predict_batch_calls: List[int] = []
        self._lock = threading.Lock()

    def predict_batch(self, states):
        n = len(states)
        with self._lock:
            self.predict_batch_calls.append(n)
        return torch.zeros(n, TOTAL_MOVES), torch.zeros(n, 1)


class GlobalCounterFakeNetwork:
    """Assigns each input state a globally-unique integer value.

    Lets routing tests verify that no two futures get the same value,
    which would only happen via cross-talk between concurrent requests.
    Counter is global (not per-call) so unique-across-batches holds.
    """

    def __init__(self):
        self.state_encoder = StateEncoder()
        self._counter = 0
        self._lock = threading.Lock()

    def predict_batch(self, states):
        n = len(states)
        with self._lock:
            base = self._counter
            self._counter += n
        values = torch.tensor(
            [[float(base + i)] for i in range(n)],
            dtype=torch.float32,
        )
        return torch.zeros(n, TOTAL_MOVES), values


# --------------------------------------------------------------------------- #
# T1 — coalescence                                                            #
# --------------------------------------------------------------------------- #


def test_concurrent_requests_collapse_into_few_predict_batch_calls():
    """T1: 100 concurrent requests should produce ≤ a handful of calls.

    With max_wait_ms=20 the drain thread waits long enough for the
    queue to fill before flushing — exactly the regime self-play hits
    when N parallel MCTS workers all flush their per-game batches near
    the same wall-clock instant.
    """
    fake = CountingFakeNetwork()
    n_requests = 100

    with BatchedEvaluator(fake, max_batch=128, max_wait_ms=20.0) as ev:
        results: List = []
        results_lock = threading.Lock()

        def submit():
            out = ev.evaluate(GameState())
            with results_lock:
                results.append(out)

        threads = [threading.Thread(target=submit) for _ in range(n_requests)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(results) == n_requests
    assert sum(fake.predict_batch_calls) == n_requests, (
        f"total positions evaluated ({sum(fake.predict_batch_calls)}) != "
        f"requests ({n_requests}); requests are being lost or duplicated"
    )
    # The coalescence assertion. In practice we typically see 1-3 calls
    # under this regime; allow up to 8 to absorb scheduler jitter without
    # making the test flaky. If it's >> 8 the drain loop is flushing
    # too eagerly and the evaluator's purpose is defeated.
    assert len(fake.predict_batch_calls) <= 8, (
        f"100 requests fanned out to {len(fake.predict_batch_calls)} "
        f"predict_batch calls (sizes: {fake.predict_batch_calls}). "
        f"The drain loop is not coalescing — investigate _collect_batch."
    )


# --------------------------------------------------------------------------- #
# T2 — max_batch cap                                                          #
# --------------------------------------------------------------------------- #


def test_max_batch_caps_individual_call_size():
    """T2: no single predict_batch call exceeds max_batch.

    Even with 500 concurrent requests and a generous wait, the cap is
    a hard limit. A regression here pushes batches into VRAM trouble
    on the GPU.
    """
    fake = CountingFakeNetwork()
    n_requests = 500
    max_batch = 64

    with BatchedEvaluator(fake, max_batch=max_batch, max_wait_ms=50.0) as ev:
        threads = [
            threading.Thread(target=lambda: ev.evaluate(GameState()))
            for _ in range(n_requests)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert sum(fake.predict_batch_calls) == n_requests
    over = [n for n in fake.predict_batch_calls if n > max_batch]
    assert not over, (
        f"max_batch={max_batch} violated; oversized calls: {over}. "
        f"All sizes: {fake.predict_batch_calls}"
    )


# --------------------------------------------------------------------------- #
# T3 — routing correctness                                                    #
# --------------------------------------------------------------------------- #


def test_each_request_receives_a_unique_value():
    """T3: routing back to per-request futures is correct.

    The fake assigns a globally-unique integer to every state it sees.
    If routing is correct, the N futures resolve to N distinct values.
    A regression where one future gets another's value (cross-talk in
    the drain loop's index→future mapping) shows up here as a
    duplicate.

    A regression here is silent: training would proceed without
    errors, but with the wrong value targets per position. This is the
    test that catches "off-by-one in the drain loop" or "wrong
    state-to-future mapping under concurrency".
    """
    fake = GlobalCounterFakeNetwork()
    n_requests = 50
    received: List[int] = []
    received_lock = threading.Lock()

    with BatchedEvaluator(fake, max_batch=32, max_wait_ms=20.0) as ev:
        def submit():
            _, value = ev.evaluate(GameState())
            with received_lock:
                received.append(int(value.item()))

        threads = [threading.Thread(target=submit) for _ in range(n_requests)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(received) == n_requests
    duplicates = len(received) - len(set(received))
    assert duplicates == 0, (
        f"{duplicates} duplicate value(s) found across {n_requests} "
        f"futures: {received}. The drain loop is misrouting results "
        f"between futures — multiple futures got the same value."
    )


# --------------------------------------------------------------------------- #
# Smaller assertions — shapes, shutdown                                       #
# --------------------------------------------------------------------------- #


def test_evaluate_batch_returns_predict_batch_shapes():
    """`evaluate_batch` must return tensors of the same shape as
    `predict_batch` so the call-site swap in `_evaluate_and_backup_batch`
    is one line.
    """
    fake = CountingFakeNetwork()
    states = [GameState() for _ in range(8)]
    with BatchedEvaluator(fake, max_batch=16, max_wait_ms=10.0) as ev:
        logits, values = ev.evaluate_batch(states)
    assert logits.shape == (8, TOTAL_MOVES)
    assert values.shape == (8, 1)


def test_shutdown_is_idempotent():
    """`shutdown()` may be called more than once (context-manager exit
    plus an explicit call, for example). Second call must be a no-op.
    """
    fake = CountingFakeNetwork()
    ev = BatchedEvaluator(fake)
    ev.shutdown()
    ev.shutdown()  # must not raise


# --------------------------------------------------------------------------- #
# T4 — MCTS routed through evaluator matches direct-network path              #
# --------------------------------------------------------------------------- #


def _build_mcts(network, evaluator):
    """Construct a self-play MCTS with deterministic-friendly settings.

    Disables Dirichlet noise + epsilon-mix (no randomness from those
    paths). The remaining randomness — the UCB tiebreak ε-noise —
    consumes np.random the same number of times in both runs, so a
    single seed before each run is enough to match.
    """
    from yinsh_ml.training.self_play import MCTS as SelfPlayMCTS

    return SelfPlayMCTS(
        network=network,
        evaluation_mode="pure_neural",
        num_simulations=32,
        late_simulations=32,
        simulation_switch_ply=20,
        c_puct=1.0,
        dirichlet_alpha=0.0,        # no Dirichlet noise
        value_weight=1.0,
        max_depth=200,
        initial_temp=1.0,
        final_temp=1.0,
        annealing_steps=1,
        enable_subtree_reuse=True,  # keep _cached_root for inspection
        epsilon_mix_start=0.0,      # no root-noise mixing
        epsilon_mix_end=0.0,
        epsilon_mix_taper_moves=0,
        evaluator=evaluator,
    )


def test_mcts_with_evaluator_matches_direct_path():
    """T4: integration determinism guard.

    Run `search_batch` twice — once with `evaluator=None` (direct
    path), once with `evaluator=BatchedEvaluator(network)` — using the
    same fake network and the same seed. Visit counts on the root's
    children must match exactly. If they don't, the evaluator is
    introducing nondeterminism (e.g. dropping/duplicating requests, or
    routing them in the wrong order under coalescence).

    The fake network returns zeros — uniform priors after softmax — so
    any divergence is necessarily structural, not noise from value
    deltas.
    """
    def run(use_evaluator: bool):
        np.random.seed(42)
        random.seed(42)
        network = CountingFakeNetwork()
        if use_evaluator:
            evaluator = BatchedEvaluator(network, max_wait_ms=20.0)
        else:
            evaluator = None
        try:
            mcts = _build_mcts(network, evaluator)
            move_probs = mcts.search_batch(GameState(), move_number=0,
                                           batch_size=16)
            root = mcts._cached_root
            assert root is not None and root.children, (
                "search produced no children — fixture is too small to "
                "exercise the integration"
            )
            visits = {action: root.children[action].visit_count
                      for action in sorted(root.children.keys(), key=str)}
        finally:
            if evaluator is not None:
                evaluator.shutdown()
        return visits, move_probs

    visits_direct, probs_direct = run(use_evaluator=False)
    visits_eval, probs_eval = run(use_evaluator=True)

    assert visits_direct == visits_eval, (
        "Visit counts diverge between direct and evaluator paths. "
        "First few mismatches: "
        + str([(k, visits_direct[k], visits_eval.get(k))
               for k in visits_direct
               if visits_direct[k] != visits_eval.get(k)][:5])
    )
    np.testing.assert_array_almost_equal(probs_direct, probs_eval, decimal=6)


# Note on orphan-failure behavior:
# `shutdown()` is supposed to fail any requests still in the queue with
# RuntimeError("...shut down..."), so worker threads can't deadlock
# waiting on a never-arriving result. We don't unit-test that path here
# because it requires stalling `predict_batch` to keep the drain thread
# from picking the request up — too much machinery for a unit test. It
# gets exercised naturally by Phase 2 integration (cancelling a
# threaded SelfPlay run mid-iteration).
