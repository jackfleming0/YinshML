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
        """Make Move hashable."""
        markers_tuple = None
        if self.markers is not None:
            # Sort markers for consistent hashing
            markers_tuple = tuple(sorted(self.markers, key=lambda p: (p.column, p.row)))
        return hash((
            self.type,
            self.player,
            self.source,
            self.destination,
            markers_tuple
        ))