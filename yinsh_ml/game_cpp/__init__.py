"""C++ bitboard engine for YINSH (in development).

This package will eventually expose a drop-in replacement for
``yinsh_ml.game.GameState`` backed by a pybind11 bitboard engine. While
the port is in progress, the only public surface is ``_engine`` — the
compiled extension module — and a small set of probes used by the
build/perf harness.

Status: minimal slice — validates the pybind11 build chain and the
121-cell bitboard layout. Move generation and full state are not yet
implemented.
"""
from __future__ import annotations

try:
    from . import _engine  # noqa: F401  (compiled extension)
except ImportError as exc:  # pragma: no cover — surfaces build problems early
    raise ImportError(
        "yinsh_ml.game_cpp._engine is not built. Run `pip install -e .` "
        "from the repo root to compile the C++ extension."
    ) from exc

from .game_state import CppBoard, CppGameState  # noqa: E402

__all__ = ["_engine", "CppGameState", "CppBoard"]
