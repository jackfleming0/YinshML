"""Basic type definitions for YINSH game."""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .constants import Player, Position

class MoveType(Enum):
    PLACE_RING = "PLACE_RING"
    MOVE_RING = "MOVE_RING"
    REMOVE_MARKERS = "REMOVE_MARKERS"
    REMOVE_RING = "REMOVE_RING"

class GamePhase(Enum):
    RING_PLACEMENT = 0
    MAIN_GAME = 1
    ROW_COMPLETION = 2
    RING_REMOVAL = 3
    GAME_OVER = 4

@dataclass(frozen=True)
class Move:
    """Represents a move in YINSH."""
    type: MoveType
    player: Player
    source: Optional[Position] = None
    destination: Optional[Position] = None
    markers: Optional[Tuple[Position, ...]] = None  # Changed from List to Optional[Tuple]

    def __post_init__(self):
        """Convert markers list to tuple if needed."""
        if self.markers is not None and isinstance(self.markers, list):
            object.__setattr__(self, 'markers', tuple(self.markers))

    def __str__(self) -> str:
        """String representation of the move."""
        if self.type == MoveType.PLACE_RING:
            return f"{self.player.name} places ring at {self.source}"
        elif self.type == MoveType.MOVE_RING:
            return f"{self.player.name} moves ring {self.source}->{self.destination}"
        elif self.type == MoveType.REMOVE_MARKERS:
            markers_str = ",".join(str(pos) for pos in (self.markers or []))
            return f"{self.player.name} removes markers at {markers_str}"
        else:  # REMOVE_RING
            return f"{self.player.name} removes ring at {self.source}"

    def __hash__(self):
        """Make Move hashable.

        Hash result is memoized in ``self.__dict__['_hash']`` after the
        first call. Profiling on cloud 4090 (BITBOARD_FOLLOWUP_PLAN.md
        Candidate A') showed Move.__hash__ at ~30s cum/game from 12M
        recursive hash chains across MCTS dict ops. Caching collapses
        each repeat call to a single dict lookup.

        ``object.__setattr__`` is required because frozen dataclasses
        block direct attribute assignment; ``__post_init__`` already
        uses the same escape hatch for the markers tuple normalization,
        so the precedent is established. ``__getstate__`` strips the
        cache before pickling — string hashes depend on PYTHONHASHSEED,
        which is per-process randomized, so a cached hash from a
        self-play worker is wrong in the parent process.
        """
        cached = self.__dict__.get('_hash')
        if cached is not None:
            return cached
        markers_tuple = None
        if self.markers is not None:
            # Sort markers for consistent hashing
            markers_tuple = tuple(sorted(self.markers, key=lambda p: (p.column, p.row)))
        h = hash((
            self.type,
            self.player,
            self.source,
            self.destination,
            markers_tuple
        ))
        object.__setattr__(self, '_hash', h)
        return h

    def __getstate__(self):
        """Pickle hook: drop the per-process hash cache before serialization.

        Without this, a Move hashed in a worker process (where strings
        hash with one PYTHONHASHSEED) would carry the worker's hash into
        the parent process (different seed) and silently break dict
        invariants — equal Moves landing in different buckets.
        """
        state = self.__dict__.copy()
        state.pop('_hash', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)