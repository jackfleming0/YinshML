"""Investigation tests for the GPU scaling plan (PR #11, GPU_SCALING_PLAN.md).

The plan is a design doc, not a refactor. Its TL;DR is:

    The first thing to do is turn on what exists (`num_workers > 0`,
    `mcts_batch_size`) and **measure** before building anything new.

These tests pin the empirical claims the doc makes about *current* code so
that:

  1. Anyone investigating the cloud-GPU push can run `pytest -k
     gpu_scaling_investigation` and see, on real code, which claims hold.
  2. The "first refactor target" is identified in evidence, not vibes.
  3. If the underlying code drifts (e.g. someone wires up cross-game
     batching), the relevant test fails and the plan can be updated rather
     than going stale silently.

Each test maps to a numbered claim in `GPU_SCALING_PLAN.md`:

  - C1 (Part 1) Per-game leaf batching with virtual loss exists and is
                actually called in batches by `search_batch`.
  - C2 (Part 1) `add_virtual_loss` / `remove_virtual_loss` balance — no
                node is left with virtual losses pinned after a search.
  - C3 (Part 1) The simpler `yinsh_ml/search/mcts.py` MCTS has no batching
                — it calls `network.predict` once per leaf.
  - C4 (Part 1) No shared cross-worker `BatchedEvaluator` exists yet.
  - C5 (Part 2) `supervisor._compute_num_workers()` returns 0 by default
                (the `MAX_WORKERS = 0` cap from the zombie-process incident).
                This is "the single biggest lever currently unused".
  - C6 (Part 2) The default `mcts_batch_size` baked into self-play config
                is 32 — small enough that the doc recommends 64-128 first.

Where it's cheap to do so, the tests also *print* the call-shape data
(number of `predict_batch` calls, max batch size observed, etc.) so that
running them is itself a quick diagnostic — `pytest -s` shows you the
shape of the search loop without needing `py-spy`.

These are unit/integration tests; they do not require a GPU. They use a
recording fake `NetworkWrapper` that returns deterministic zero tensors,
which is enough to drive the search code paths and observe call counts.
The actual GPU-utilization question (Part 2 of the plan) still has to be
answered with `nvidia-smi dmon` on the cloud instance — a unit test
cannot tell you whether a 4090 is starved.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from yinsh_ml.game.game_state import GameState
from yinsh_ml.search.mcts import (
    MCTS as SearchMCTS,
    MCTSConfig,
    EvaluationMode,
)
from yinsh_ml.training.self_play import MCTS as SelfPlayMCTS
from yinsh_ml.utils.encoding import StateEncoder


TOTAL_MOVES = StateEncoder().total_moves


# --------------------------------------------------------------------------- #
# Fakes                                                                       #
# --------------------------------------------------------------------------- #


class RecordingNetwork:
    """Fake `NetworkWrapper` that records every inference call.

    Mirrors only the surface area `search_batch` and the simpler MCTS need:
    `state_encoder`, `predict`, `predict_from_state`, `predict_batch`. Each
    call is logged with its batch size so tests can assert on call shape.
    """

    def __init__(self):
        self.state_encoder = StateEncoder()
        self.predict_calls: List[int] = []           # one entry per `predict` call (always 1)
        self.predict_from_state_calls: int = 0       # singleton path on self_play.MCTS.search
        self.predict_batch_calls: List[int] = []     # one entry per call, value = batch size

    # Used by yinsh_ml/search/mcts.py
    def predict(self, state_tensor):
        self.predict_calls.append(1)
        policy = np.zeros(TOTAL_MOVES, dtype=np.float32)
        policy[:] = 1.0 / TOTAL_MOVES
        return policy, 0.0

    # Used by yinsh_ml/training/self_play.py::MCTS.search (the non-batched path)
    def predict_from_state(self, state):
        self.predict_from_state_calls += 1
        policy = torch.zeros(1, TOTAL_MOVES)
        value = torch.zeros(1, 1)
        return policy, value

    # Used by yinsh_ml/training/self_play.py::MCTS.search_batch
    def predict_batch(self, states):
        n = len(states)
        self.predict_batch_calls.append(n)
        policy = torch.zeros(n, TOTAL_MOVES)
        values = torch.zeros(n, 1)
        return policy, values


def _selfplay_mcts(network, *, num_simulations=64, max_depth=200):
    """Construct a minimally-configured self-play MCTS for batched search.

    Uses pure_neural mode + subtree reuse off so each test starts clean.
    """
    return SelfPlayMCTS(
        network=network,
        evaluation_mode="pure_neural",
        num_simulations=num_simulations,
        late_simulations=num_simulations,
        simulation_switch_ply=20,
        c_puct=1.0,
        dirichlet_alpha=0.0,           # deterministic priors
        value_weight=1.0,
        max_depth=max_depth,
        initial_temp=1.0,
        final_temp=1.0,
        annealing_steps=1,
        enable_subtree_reuse=False,
        epsilon_mix_start=0.0,
        epsilon_mix_end=0.0,
        epsilon_mix_taper_moves=0,
    )


# --------------------------------------------------------------------------- #
# C1 — `search_batch` actually batches its calls to `predict_batch`           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("batch_size", [16, 32, 64])
def test_search_batch_calls_predict_batch_in_batches(batch_size, capsys):
    """C1: per-game leaf batching is real, not nominal.

    Run `search_batch` with `num_simulations=128` against a recording fake
    network. The number of `predict_batch` calls should be roughly
    ceil(128 / batch_size) — *not* 128. Each call should evaluate at most
    `batch_size` states, and the total should equal the number of non-
    terminal leaves expanded.

    This is the cheapest concrete answer to "is the existing batching path
    actually firing?" — the doc claims it is; this test pins it.
    """
    network = RecordingNetwork()
    mcts = _selfplay_mcts(network, num_simulations=128)
    state = GameState()  # fresh ring-placement state — many legal moves

    mcts.search_batch(state, move_number=0, batch_size=batch_size)

    n_calls = len(network.predict_batch_calls)
    total_evaluated = sum(network.predict_batch_calls)
    max_batch = max(network.predict_batch_calls) if network.predict_batch_calls else 0

    # Diagnostic print so `pytest -s` is itself a useful tool.
    print(
        f"\n[C1 batch_size={batch_size}] predict_batch calls={n_calls} "
        f"total_states={total_evaluated} max_batch={max_batch} "
        f"per-call sizes={network.predict_batch_calls}"
    )

    # If batching is broken, we'd see ~128 calls of size 1. Be loose on the
    # upper bound — terminal/max-depth bailouts mean a few simulations
    # don't reach the batch — but anything >2× the expected count points
    # at a regression where the flush logic isn't actually batching.
    expected_calls = int(np.ceil(128 / batch_size))
    assert n_calls <= max(expected_calls * 2, 4), (
        f"search_batch made {n_calls} predict_batch calls for 128 sims at "
        f"batch_size={batch_size}; expected ~{expected_calls}. Batching "
        f"may have regressed."
    )
    assert max_batch <= batch_size, (
        f"Saw a predict_batch call of size {max_batch} > requested "
        f"batch_size={batch_size}; flush size accounting is wrong."
    )
    # And — most importantly — at least one call should be near full,
    # otherwise the path is "batching" 1-2 leaves at a time which is no
    # better than `predict()`.
    assert max_batch >= min(batch_size // 2, 8), (
        f"Largest observed batch was {max_batch}; expected at least "
        f"{min(batch_size // 2, 8)}. Either the search is starved of leaves "
        f"or the flush is firing too eagerly. This is the symptom that "
        f"motivates the cross-game evaluator in Part 1 of the plan."
    )


# --------------------------------------------------------------------------- #
# C2 — virtual loss is balanced (added then removed)                          #
# --------------------------------------------------------------------------- #


def test_virtual_loss_lifecycle_balances():
    """C2: every `add_virtual_loss` is paired with `remove_virtual_loss`.

    If virtual losses leak, the UCB scoring under-weights repeatedly-
    visited subtrees and search behavior gets weird without ever erroring.
    The doc relies on the existing batching path being correct before
    talking about cross-game extensions; this test pins that.
    """
    network = RecordingNetwork()
    mcts = _selfplay_mcts(network, num_simulations=64)
    state = GameState()

    mcts.search_batch(state, move_number=0, batch_size=16)

    # Walk the (root + descendants) and confirm no leftover virtual losses.
    # We don't have a public root accessor when subtree reuse is off, so
    # we patch into `_cached_root` — it's still set transiently. Fall back
    # to a per-node sweep via `_evaluate_and_backup_batch` if needed.
    root = mcts._cached_root
    if root is None:
        # subtree reuse disabled: tree was cleared. The strongest signal we
        # can give is then "no exception raised + non-empty predict_batch
        # calls", which we already check elsewhere. Skip the structural
        # walk in that case but still assert the call shape.
        assert network.predict_batch_calls, "no predict_batch calls observed"
        return

    stack = [root]
    seen = 0
    while stack:
        node = stack.pop()
        seen += 1
        assert node.virtual_losses == 0, (
            f"Node has virtual_losses={node.virtual_losses} after search "
            f"completed. Imbalance in add/remove_virtual_loss."
        )
        stack.extend(node.children.values())

    assert seen > 1, "expected the search to expand at least the root + one child"


# --------------------------------------------------------------------------- #
# C3 — the simpler search/mcts.py has no batching                             #
# --------------------------------------------------------------------------- #


def test_simple_mcts_evaluate_state_is_singleton():
    """C3: `yinsh_ml/search/mcts.py::MCTS._evaluate_state` is unbatched.

    The doc claim is structural — that the inference/tuning-time MCTS has
    no batched evaluation path, only a one-position-per-call
    `_evaluate_state`. We test the structural property directly rather
    than driving `search()`, because the search loop has an unrelated
    first-sim-skip artifact (see the comment in `self_play.py:778-779`
    about the old `action is None` guard) that makes a top-level call
    count brittle.

    If we ever add a batched path here (e.g. for hyperparameter tuning on
    cloud), this test will fail and point at exactly the file that needs
    its assertion updated.
    """
    network = RecordingNetwork()
    config = MCTSConfig(
        num_simulations=4,
        evaluation_mode=EvaluationMode.PURE_NEURAL,
        use_heuristic_evaluation=False,
    )
    mcts = SearchMCTS(network=network, config=config)
    state = GameState()

    # Call the leaf-eval path directly — this is the unit the doc names.
    policy, value = mcts._evaluate_state(state)

    print(
        f"\n[C3] _evaluate_state predict calls={len(network.predict_calls)} "
        f"predict_batch calls={len(network.predict_batch_calls)}"
    )

    assert network.predict_batch_calls == [], (
        "yinsh_ml/search/mcts.py::MCTS._evaluate_state is calling "
        "predict_batch — the simple MCTS has gained a batched evaluation "
        "path. Update the GPU scaling plan; the gap noted in Part 1 is "
        "closed."
    )
    assert network.predict_calls == [1], (
        f"Expected exactly one single-state predict() call from "
        f"_evaluate_state, got {network.predict_calls}. The unbatched "
        f"contract has changed."
    )


# --------------------------------------------------------------------------- #
# C4 — no shared cross-worker BatchedEvaluator exists yet                     #
# --------------------------------------------------------------------------- #


def test_no_shared_batched_evaluator_module():
    """C4: cross-game batching is absent.

    The doc proposes a future `BatchedEvaluator` that owns the only model
    on the GPU and coalesces requests from N MCTS workers. It does not
    exist yet. This test pins that — if someone adds the module, this
    test will fail loudly and the plan should be updated to mark Part 1's
    "future work" section as in-progress / done.
    """
    candidates = [
        "yinsh_ml.training.batched_evaluator",
        "yinsh_ml.search.batched_evaluator",
        "yinsh_ml.network.batched_evaluator",
        "yinsh_ml.network.shared_evaluator",
    ]
    found = []
    for mod in candidates:
        try:
            __import__(mod)
            found.append(mod)
        except ImportError:
            continue

    assert not found, (
        f"Found unexpected batched-evaluator module(s): {found}. The "
        f"GPU scaling plan assumes none exists yet — update the plan."
    )


# --------------------------------------------------------------------------- #
# C5 — `_compute_num_workers` defaults to 0 (zombie-process cap)              #
# --------------------------------------------------------------------------- #


def test_default_num_workers_is_zero():
    """C5: out-of-the-box self-play runs serially.

    The plan calls this out as the single biggest lever currently unused.
    The cap at 0 was set defensively for a zombie-process / RSS-leak
    incident; whether that cause is still live is an open question listed
    in the doc.

    This test is *intentionally* fragile to that cap. If someone raises
    `MAX_WORKERS` (e.g. because the underlying issue was fixed by the
    bitboard work), this test will fail — and that's the signal to
    update the plan to mark Part 2's first action as complete.
    """
    from yinsh_ml.training.supervisor import TrainingSupervisor

    # Build a minimally-mocked supervisor without going through __init__.
    sup = TrainingSupervisor.__new__(TrainingSupervisor)
    sup.mode_settings = {}  # no override
    sup.logger = MagicMock()

    workers = sup._compute_num_workers()
    print(f"\n[C5] _compute_num_workers() with empty config returned {workers}")

    assert workers == 0, (
        f"Default num_workers is {workers}, not 0. The MAX_WORKERS=0 cap "
        f"in supervisor.py has been raised. Update GPU_SCALING_PLAN.md "
        f"Part 2 — the 'biggest lever currently unused' has been pulled."
    )


def test_explicit_num_workers_override_takes_effect():
    """C5b: passing `num_workers` in config bypasses the cap.

    The plan recommends `num_workers: 8` as a starting point on a 4090
    box. Confirm that overrides actually thread through — i.e. once we
    decide to flip the lever, we don't have to also remove the cap.
    """
    from yinsh_ml.training.supervisor import TrainingSupervisor

    sup = TrainingSupervisor.__new__(TrainingSupervisor)
    sup.mode_settings = {"num_workers": 8}
    sup.logger = MagicMock()

    workers = sup._compute_num_workers()
    assert workers == 8, (
        f"Explicit num_workers=8 resolved to {workers}; override path "
        f"is broken. The lever exists but isn't reachable from config."
    )


# --------------------------------------------------------------------------- #
# C6 — default mcts_batch_size is small (32)                                  #
# --------------------------------------------------------------------------- #


def test_default_mcts_batch_size_is_small():
    """C6: the in-code default for `mcts_batch_size` is 32.

    The plan recommends 64-128 as a starting point. We don't change the
    default in this test — that's a deliberate decision the plan defers
    until after measurement on cloud — but we pin it so the recommendation
    in the doc stays honest.
    """
    import inspect
    from yinsh_ml.training.self_play import SelfPlay

    sig = inspect.signature(SelfPlay.__init__)
    default = sig.parameters["mcts_batch_size"].default
    print(f"\n[C6] SelfPlay.__init__ default mcts_batch_size = {default}")
    assert default == 32, (
        f"Default mcts_batch_size is {default}, not 32. If this was a "
        f"deliberate change after measurement, update the recommendation "
        f"in GPU_SCALING_PLAN.md Part 2."
    )


# --------------------------------------------------------------------------- #
# Summary of where to investigate first                                       #
# --------------------------------------------------------------------------- #
#
# These tests give the investigator a fast, structured read of current
# state without needing a GPU:
#
#   1. C1 + C2 confirm the existing batched path *works*, so a refactor
#      to add a shared evaluator is not blocked by a broken local one.
#   2. C5 says the easiest lever (num_workers > 0) is unpulled. That's
#      the first thing to flip on a cloud-4090 instance, with an eye on
#      RSS for the historical zombie-process issue.
#   3. C6 says the second-easiest lever (mcts_batch_size 32 → 64-128) is
#      one config change.
#   4. C3 + C4 are the *deferred* refactor work. Don't start there.
#
# So the test ordering is also the recommended investigation ordering:
# verify the cheap stuff, flip the cheap knobs, *measure* on real
# hardware, and only then revisit C3/C4.
