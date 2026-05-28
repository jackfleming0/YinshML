"""Python wrapper around the yngine_driver subprocess.

Typical use::

    from yinsh_ml.yngine import Yngine
    from yinsh_ml.game.game_state import GameState

    state = GameState()
    with Yngine.start() as eng:
        # Feed our model's moves to yngine so its MCTS tree can be reused.
        eng.apply(our_move)
        # Ask yngine for its move.
        yngine_mv = eng.get_move(sims=1000, player=state.current_player)
        state.make_move(yngine_mv)

The driver process maintains its own copy of the game state and an MCTS
tree. `apply` and `get_move` keep them in sync with our `GameState`.

Build the driver first::

    bash third_party/yngine_driver/build.sh

Then this module finds the binary at
``third_party/yngine_driver/build-release/yngine_driver`` automatically.
Override via the ``YNGINE_DRIVER`` environment variable or by passing
``binary_path=`` to :py:meth:`Yngine.start`.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import threading
from pathlib import Path
from typing import Optional

from yinsh_ml.game.constants import Player
from yinsh_ml.game.types import Move

from .move_codec import move_to_wire, wire_to_move

logger = logging.getLogger("yinsh_ml.yngine")


class YngineError(RuntimeError):
    """Raised when the driver returns ``err …`` or exits unexpectedly."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_binary_path() -> Path:
    """Resolve the driver binary. ``YNGINE_DRIVER`` env var overrides."""
    env = os.environ.get("YNGINE_DRIVER")
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root() / "third_party" / "yngine_driver" / "build-release" / "yngine_driver"


class Yngine:
    """Subprocess wrapper around the yngine_driver protocol."""

    def __init__(self, binary_path: Path, threads: int = 1):
        self._binary = Path(binary_path)
        self._threads = max(1, int(threads))
        self._proc: Optional[subprocess.Popen[str]] = None
        # Drain stderr in a background thread so the pipe never blocks the
        # driver if it logs something unexpected.
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_buf: list[str] = []

    # ----- lifecycle -----

    @classmethod
    def start(
        cls,
        binary_path: Optional[Path] = None,
        threads: int = 1,
    ) -> "Yngine":
        """Spawn the driver and wait for its ``ready`` handshake."""
        path = Path(binary_path) if binary_path else default_binary_path()
        if not path.exists():
            raise YngineError(
                f"yngine_driver binary not found at {path}. Build with:\n"
                "  bash third_party/yngine_driver/build.sh\n"
                "Or set YNGINE_DRIVER=/abs/path/to/yngine_driver."
            )
        eng = cls(path, threads=threads)
        eng._spawn()
        return eng

    def _spawn(self) -> None:
        # Force unbuffered stdin/stdout. The C++ driver line-buffers its
        # output via setvbuf; Python defaults to line-buffered for text-mode
        # universal newline pipes which is what we want.
        self._proc = subprocess.Popen(
            [str(self._binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()
        handshake = self._readline()
        if handshake != "ready":
            raise YngineError(
                f"yngine_driver handshake failed: expected 'ready', got {handshake!r}"
            )

    def _drain_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        for line in self._proc.stderr:
            self._stderr_buf.append(line.rstrip("\n"))
            # Keep memory bounded — the driver itself doesn't log to stderr,
            # but yngine's underlying ArenaAllocator/abort path can.
            if len(self._stderr_buf) > 256:
                del self._stderr_buf[:64]

    def stop(self) -> int:
        """Send ``quit`` and wait for the driver to exit. Returns its exit code."""
        if self._proc is None:
            return 0
        try:
            self._write("quit")
            self._readline()  # "bye"
        except (BrokenPipeError, YngineError):
            pass
        try:
            rc = self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            rc = self._proc.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
        self._proc = None
        return rc

    def __enter__(self) -> "Yngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ----- protocol primitives -----

    def _write(self, line: str) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise YngineError("yngine_driver subprocess is not running")
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            raise YngineError(
                f"yngine_driver pipe broken — stderr: {' | '.join(self._stderr_buf[-5:])}"
            ) from exc

    def _readline(self) -> str:
        if self._proc is None or self._proc.stdout is None:
            raise YngineError("yngine_driver subprocess is not running")
        line = self._proc.stdout.readline()
        if not line:
            tail = " | ".join(self._stderr_buf[-5:]) if self._stderr_buf else ""
            raise YngineError(
                f"yngine_driver closed stdout unexpectedly (stderr: {tail})"
            )
        return line.rstrip("\n")

    def _request(self, line: str) -> str:
        self._write(line)
        reply = self._readline()
        if reply.startswith("err "):
            raise YngineError(
                f"yngine_driver rejected {line!r}: {reply}"
            )
        return reply

    # ----- protocol commands -----

    def new_game(self) -> None:
        """Reset the driver's board and MCTS tree."""
        reply = self._request("new")
        if reply != "ok":
            raise YngineError(f"unexpected reply to 'new': {reply!r}")

    def apply(self, move: Move) -> None:
        """Apply one of our moves to the driver's board (advances MCTS too)."""
        wire = move_to_wire(move)
        reply = self._request(f"apply {wire}")
        if reply != "ok":
            raise YngineError(f"unexpected reply to 'apply': {reply!r}")

    def apply_wire(self, wire: str) -> None:
        """Apply a move already in wire format (used to echo yngine's pick)."""
        reply = self._request(f"apply {wire}")
        if reply != "ok":
            raise YngineError(f"unexpected reply to 'apply': {reply!r}")

    def get_move(
        self,
        player: Player,
        sims: Optional[int] = None,
        seconds: Optional[float] = None,
    ) -> tuple[Move, str]:
        """Ask yngine for its chosen move at the given budget.

        Exactly one of `sims` (iteration count) or `seconds` (wall-clock
        budget) must be set. Returns `(our_Move, wire_string)`; pass
        `wire_string` to `apply_wire()` after also applying the move to your
        own GameState to keep yngine's tree in sync.
        """
        if (sims is None) == (seconds is None):
            raise ValueError("specify exactly one of sims= or seconds=")
        if sims is not None:
            cmd = f"go sims {int(sims)} threads {self._threads}"
        else:
            cmd = f"go time {float(seconds)} threads {self._threads}"
        reply = self._request(cmd)
        if not reply.startswith("move "):
            raise YngineError(f"expected 'move …', got {reply!r}")
        wire = reply[len("move "):]
        return wire_to_move(wire, player), wire

    def state(self) -> dict:
        """Query the driver's current state for debugging."""
        reply = self._request("state")
        # Format: "state next=<a> turn=<W|B> result=<...>"
        parts = reply.split()
        out: dict[str, str] = {}
        for kv in parts[1:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                out[k] = v
        return out

    # ----- diagnostics -----

    @property
    def binary_path(self) -> Path:
        return self._binary

    @property
    def stderr_tail(self) -> list[str]:
        return list(self._stderr_buf[-20:])
