"""T4.10: GameStatePool worker safety + worker_crash_count.

Two surfaces under test:

1. ``_run_game_loop`` must release the GameState back to its pool on
   every exit path — including when the inner game loop raises. Before
   the T4.10 fix, the pool return only ran on the happy path; a worker
   crash leaked one GameState per run. Today the cleanest place to
   notice a leak is by checking ``pool.size()`` before/after.

2. ``SelfPlay.worker_crash_count`` must increment when a worker /
   thread / sequential run raises. This is the parent-side counter ops
   dashboards consume. We exercise the sequential path because it
   doesn't require a real ProcessPoolExecutor and can monkey-patch the
   inner worker function deterministically.

Process-pool leak verdict: a *non-issue in practice* because each
worker process creates its OWN GameStatePool (line ~1851 in
``self_play.py``), and when the worker process dies the whole pool
dies with it — there's nothing in the parent for the pool to leak
into. The try/finally guard still matters for the thread-pool path
(shared parent process) and for any future caller that reuses a pool
across games. See the docstring on ``_run_game_loop``.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from yinsh_ml.memory import GameStatePool, GameStatePoolConfig
from yinsh_ml.game import GameState
from yinsh_ml.training.self_play import _run_game_loop


# --------------------------------------------------------------------- #
# T4.10 — Surface 1: _run_game_loop releases the GameState on exception #
# --------------------------------------------------------------------- #


def _make_pool(initial_size: int = 5) -> GameStatePool:
    cfg = GameStatePoolConfig(
        initial_size=initial_size,
        factory_func=GameState,
        enable_statistics=True,
    )
    return GameStatePool(cfg)


def test_run_game_loop_returns_state_on_inner_crash():
    """Game loop crash mid-game → GameState still returned to pool.

    We force `mcts.search_batch` to raise on the very first call. If the
    try/finally guard is missing, the GameState we acquired never goes
    back to the pool and pool.size() will be one short of baseline.
    """
    pool = _make_pool(initial_size=5)
    baseline_size = pool.size()

    mcts = MagicMock()
    mcts.search_batch.side_effect = RuntimeError("synthetic worker crash")
    mcts.search.side_effect = RuntimeError("synthetic worker crash")
    # heuristic_evaluator inspection at function tail must not run, but
    # be defensive in case the loop exit path changes.
    mcts.heuristic_evaluator = None

    state_encoder = MagicMock()
    state_encoder.total_moves = 7433  # canonical value
    worker_logger = logging.getLogger("test._run_game_loop")

    with pytest.raises(RuntimeError, match="synthetic worker crash"):
        _run_game_loop(
            mcts=mcts,
            state_encoder=state_encoder,
            use_cpp_engine=False,
            local_game_state_pool=pool,
            game_id=0,
            worker_logger=worker_logger,
            use_batched_mcts=True,
            mcts_batch_size=1,
        )

    # Critical assertion: pool size returned to baseline. Pre-T4.11 fix
    # this would be baseline_size - 1 (the leaked GameState).
    assert pool.size() == baseline_size, (
        f"GameState leaked on crash: pool.size()={pool.size()}, "
        f"expected {baseline_size}. The try/finally guard in "
        f"_run_game_loop is missing or broken."
    )


def test_run_game_loop_no_pool_when_cpp_engine():
    """CppGameState path doesn't touch the pool — no return needed."""
    mcts = MagicMock()
    mcts.search_batch.side_effect = RuntimeError("cpp path crash")
    mcts.heuristic_evaluator = None
    state_encoder = MagicMock()
    state_encoder.total_moves = 7433

    # use_cpp_engine=True will try `from ..game_cpp import CppGameState`.
    # Skip cleanly if that build artifact isn't present in this checkout.
    try:
        from yinsh_ml.game_cpp import CppGameState  # noqa: F401
    except Exception:
        pytest.skip("game_cpp module not available in this build")

    worker_logger = logging.getLogger("test._run_game_loop_cpp")
    with pytest.raises(RuntimeError, match="cpp path crash"):
        _run_game_loop(
            mcts=mcts,
            state_encoder=state_encoder,
            use_cpp_engine=True,
            local_game_state_pool=None,
            game_id=0,
            worker_logger=worker_logger,
            use_batched_mcts=True,
            mcts_batch_size=1,
        )
    # No pool to inspect; the assertion is that we got the exception
    # without an additional cleanup-side error.


# --------------------------------------------------------------------- #
# T4.10 — Surface 2: worker_crash_count increments in the parent       #
# --------------------------------------------------------------------- #


def _build_minimal_self_play():
    """Build a SelfPlay instance with the absolute minimum machinery.

    We can't easily construct a real NetworkWrapper without dragging in
    PyTorch model weights, so we stand in a MagicMock. Only the bits
    `generate_games` actually touches in the sequential path are needed.
    """
    from yinsh_ml.training.self_play import SelfPlay

    network = MagicMock()
    # SelfPlay reads network.state_encoder, network.network.cpu(),
    # network.network.state_dict(), network.network.to(...), and
    # network.use_enhanced_encoding.
    network.state_encoder = MagicMock()
    network.state_encoder.total_moves = 7433
    network.network = MagicMock()
    network.network.cpu = MagicMock()
    network.network.state_dict = MagicMock(return_value={})
    network.network.to = MagicMock()
    network.use_enhanced_encoding = False
    network.device = "cpu"

    # num_workers=0 → sequential path → no ProcessPoolExecutor needed.
    sp = SelfPlay(
        network=network,
        num_workers=0,
        num_simulations=1,
        late_simulations=1,
    )
    return sp


def test_worker_crash_count_increments_sequential():
    """Sequential generate_games() bumps worker_crash_count on each crash."""
    sp = _build_minimal_self_play()
    assert sp.worker_crash_count == 0, "fresh SelfPlay must start at 0"

    # Patch the worker function so every call raises. The parent's
    # try/except in the sequential branch is what we're exercising.
    with patch(
        "yinsh_ml.training.self_play.play_game_worker",
        side_effect=RuntimeError("boom"),
    ):
        result = sp.generate_games(num_games=3)

    assert result == [], "all games failed → empty result list"
    assert sp.worker_crash_count == 3, (
        f"expected 3 crashes counted, got {sp.worker_crash_count}"
    )


def test_worker_crash_count_does_not_double_count_success():
    """Successful games don't bump the counter."""
    sp = _build_minimal_self_play()

    # Worker returns a minimal valid 5-tuple.
    fake_result = ([], [], [], {}, [])
    with patch(
        "yinsh_ml.training.self_play.play_game_worker",
        return_value=fake_result,
    ):
        sp.generate_games(num_games=2)

    assert sp.worker_crash_count == 0, (
        f"successful games should not bump crash counter, got "
        f"{sp.worker_crash_count}"
    )
