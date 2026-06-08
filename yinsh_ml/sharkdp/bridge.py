"""Python wrapper around the ``yinsh-driver`` subprocess (sharkdp/yinsh).

``Sharkdp`` subclasses :class:`yinsh_ml.yngine.bridge.Yngine` because the two
drivers speak an identical line protocol. The only differences:

* the binary lives in ``third_party/sharkdp_yinsh/target/release/yinsh-driver``
  (override with ``SHARKDP_DRIVER``);
* sharkdp searches to a fixed negamax *depth* rather than an MCTS sim budget,
  so :meth:`get_move` sends ``go depth N`` (or a bare ``go`` to use the
  driver's default / startup depth) instead of ``go sims N``.

Everything else — the spawn/handshake, ``new``/``apply``/``apply_wire``,
stderr draining, and the wire <-> ``Move`` codec — is inherited unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from yinsh_ml.game.constants import Player
from yinsh_ml.game.types import Move

from yinsh_ml.yngine.bridge import Yngine, YngineError, _repo_root
from yinsh_ml.yngine.move_codec import wire_to_move


class SharkdpError(YngineError):
    """Raised when the sharkdp driver errors or exits unexpectedly."""


def default_sharkdp_binary_path() -> Path:
    """Resolve the driver binary. ``SHARKDP_DRIVER`` env var overrides."""
    env = os.environ.get("SHARKDP_DRIVER")
    if env:
        return Path(env).expanduser().resolve()
    return (
        _repo_root()
        / "third_party"
        / "sharkdp_yinsh"
        / "target"
        / "release"
        / "yinsh-driver"
    )


class Sharkdp(Yngine):
    """Subprocess wrapper around the sharkdp ``yinsh-driver`` protocol."""

    def __init__(self, binary_path: Path, depth: int = 6):
        # threads is irrelevant for sharkdp's single-threaded negamax, but the
        # base class wires it into a couple of messages; keep it at 1.
        super().__init__(binary_path, threads=1)
        self._depth = int(depth)

    @classmethod
    def start(  # type: ignore[override]
        cls,
        binary_path: Optional[Path] = None,
        depth: int = 6,
        threads: int = 1,
    ) -> "Sharkdp":
        """Spawn the driver and wait for its ``ready`` handshake."""
        path = Path(binary_path) if binary_path else default_sharkdp_binary_path()
        if not path.exists():
            raise SharkdpError(
                f"yinsh-driver binary not found at {path}. Build with:\n"
                "  cd third_party/sharkdp_yinsh && "
                "cargo build --release -p yinsh_driver\n"
                "Or set SHARKDP_DRIVER=/abs/path/to/yinsh-driver."
            )
        eng = cls(path, depth=depth)
        eng._spawn()
        # Make the engine's persistent default depth match what we asked for,
        # so a bare `go` (and any internal default) uses it too.
        eng._set_depth(eng._depth)
        return eng

    def _set_depth(self, depth: int) -> None:
        reply = self._request(f"depth {int(depth)}")
        if reply != "ok":
            raise SharkdpError(f"unexpected reply to 'depth': {reply!r}")
        self._depth = int(depth)

    def get_move(  # type: ignore[override]
        self,
        player: Player,
        depth: Optional[int] = None,
        # accepted for signature-compatibility with Yngine.get_move; ignored.
        sims: Optional[int] = None,
        seconds: Optional[float] = None,
    ) -> tuple[Move, str]:
        """Ask sharkdp for its chosen move at the given (or default) depth.

        Returns ``(our_Move, wire_string)``. As with the yngine bridge, apply
        the move to your own ``GameState`` *and* call ``apply_wire(wire)`` to
        keep the driver's board in sync.
        """
        d = self._depth if depth is None else int(depth)
        reply = self._request(f"go depth {d}")
        if not reply.startswith("move "):
            raise SharkdpError(f"expected 'move …', got {reply!r}")
        wire = reply[len("move "):]
        if wire.strip() == "S":
            raise SharkdpError("sharkdp returned pass (S) — degenerate position")
        return wire_to_move(wire, player), wire
