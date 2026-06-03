"""BoardGameArena YINSH game, table 862307561 (Black wins 3-2).

A strong human game, transcribed from the BGA move-history table. Black opens
with a board-wide ring spread (edges + corners) against White's dense B-file
cluster, plays a long middlegame for tempo/optionality rather than immediate
scoring, and converts three rows in a tight endgame window to win 3-2.

This game is used as:

* an **engine regression fixture** — every move (including all marker-row and
  ring removals) is legal in our ``GameState`` engine and the game ends 3-2
  Black. See ``yinsh_ml/tests/test_human_game_replay.py``.
* **analysis material** — ``scripts/review_human_game.py`` replays it and scores
  every position with the heuristic feature set. That review surfaced two dead
  heuristic features and motivated the experimental feature palette in
  ``yinsh_ml/heuristics/experimental_features.py``. See
  ``docs/game_reviews/bga_862307561_review.md``.

The transcription is intentionally close to the raw BGA log so it is easy to
audit against the source table.
"""

from typing import Iterator, List, Tuple

from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import Move, MoveType, GamePhase
from yinsh_ml.game.constants import Player, Position


# --- Source identity -------------------------------------------------------

BGA_TABLE_ID = "862307561"
RESULT = "Black wins 3-2"


# --- Ring placement (10 placements, alternating White/Black) ---------------

PLACEMENTS: List[Tuple[str, str]] = [
    ("W", "F6"), ("B", "A5"), ("W", "B6"), ("B", "C6"), ("W", "B4"),
    ("B", "B1"), ("W", "B7"), ("B", "K7"), ("W", "B5"), ("B", "B3"),
]


# --- Main game -------------------------------------------------------------
# Each entry: (player, "<src>-<dst>" ring move, removal).
# removal is None for a normal move, otherwise
#   (marker_run_start, marker_run_end, ring_removed)
# describing the row the move completed (the 5 markers removed, plus the ring).

MAIN: List[Tuple[str, str, object]] = [
    ("W", "B6-C7", None),
    ("B", "B1-K10", None),
    ("W", "B7-C8", None),
    ("B", "K10-E10", None),
    ("W", "C8-D9", None),
    ("B", "E10-E1", None),
    ("W", "D9-E9", None),
    ("B", "C6-D7", None),
    ("W", "C7-D8", None),
    ("B", "E1-E8", None),
    ("W", "E9-F10", None),
    ("B", "B3-J11", None),
    ("W", "F10-F9", None),
    ("B", "J11-G11", None),
    ("W", "F9-F8", None),
    ("B", "K7-F7", None),
    ("W", "B4-C5", None),
    ("B", "G11-G10", None),
    ("W", "B5-B2", None),
    ("B", "A5-A2", None),
    ("W", "C5-A3", ("B3", "B7", "B2")),    # White scores -> 1:0
    ("B", "E8-E7", None),
    ("W", "F8-G9", None),
    ("B", "F7-G8", None),
    ("W", "F6-E6", None),
    ("B", "E7-D6", None),
    ("W", "D8-B6", None),
    ("B", "D6-B4", None),
    ("W", "E6-E5", None),
    ("B", "G10-J10", None),
    ("W", "G9-H9", None),
    ("B", "D7-H11", None),
    ("W", "B6-B7", None),
    ("B", "J10-G7", None),
    ("W", "H9-I10", None),
    ("B", "H11-H10", None),
    ("W", "A3-A4", None),
    ("B", "H10-H8", None),
    ("W", "A4-B5", None),
    ("B", "A2-D5", None),
    ("W", "I10-I9", None),
    ("B", "H8-H7", None),
    ("W", "I9-I6", None),
    ("B", "G7-G6", ("C7", "G7", "D5")),    # Black scores -> 1:1
    ("W", "B7-C7", None),
    ("B", "H7-I8", ("H7", "H11", "B4")),   # Black scores -> 2:1
    ("W", "C7-E7", ("B6", "F10", "I6")),   # White scores -> 2:2
    ("B", "G8-D8", None),
    ("W", "B5-D7", None),
    ("B", "G6-B6", None),
    ("W", "E7-F7", None),
    ("B", "B6-B4", None),
    ("W", "E5-G7", None),
    ("B", "B4-H10", None),
    ("W", "D7-H11", None),
    ("B", "I8-I11", None),
    ("W", "F7-H9", None),
    ("B", "D8-H8", None),
    ("W", "H9-D5", None),
    ("B", "H10-F10", ("D6", "H10", "F10")),  # Black scores -> 3:2 WIN
]


# --- Helpers ---------------------------------------------------------------

def _pos(s: str) -> Position:
    return Position(s[0], int(s[1:]))


def _player(tag: str) -> Player:
    return Player.WHITE if tag == "W" else Player.BLACK


def expand_line(start: str, end: str) -> Tuple[Position, ...]:
    """Inclusive list of positions from ``start`` to ``end`` along a hex line.

    Used to expand a "B3-B7" style row endpoint pair into the 5 marker
    positions that get removed.
    """
    s, e = _pos(start), _pos(end)
    dc = ord(e.column) - ord(s.column)
    dr = e.row - s.row
    steps = max(abs(dc), abs(dr))
    sc = (dc // steps) if dc else 0
    sr = (dr // steps) if dr else 0
    return tuple(
        Position(chr(ord(s.column) + sc * i), s.row + sr * i)
        for i in range(steps + 1)
    )


class IllegalReplayMove(RuntimeError):
    """Raised when a transcribed move is rejected by the engine."""


def _apply(state: GameState, move: Move, label: str) -> None:
    if not state.make_move(move):
        raise IllegalReplayMove(
            f"engine rejected {label}: {move} "
            f"(phase={state.phase}, current={state.current_player.name}, "
            f"score W{state.white_score}-B{state.black_score})"
        )


def play_placements(state: GameState) -> None:
    """Apply the 10 ring placements to ``state`` (which must be fresh)."""
    for tag, pos in PLACEMENTS:
        _apply(state, Move(type=MoveType.PLACE_RING, player=_player(tag),
                           source=_pos(pos)), f"place {tag} {pos}")


def replay() -> GameState:
    """Replay the full game and return the terminal ``GameState``.

    Raises ``IllegalReplayMove`` if any move is rejected — i.e. this doubles as
    an engine correctness assertion.
    """
    state = GameState()
    play_placements(state)
    for tag, ringmv, removal in MAIN:
        _apply_turn(state, tag, ringmv, removal)
    return state


def _apply_turn(state: GameState, tag: str, ringmv: str, removal) -> None:
    pl = _player(tag)
    src, dst = ringmv.split("-")
    _apply(state, Move(type=MoveType.MOVE_RING, player=pl,
                       source=_pos(src), destination=_pos(dst)),
           f"{tag} {ringmv}")
    if removal is not None:
        mstart, mend, ring = removal
        _apply(state, Move(type=MoveType.REMOVE_MARKERS, player=pl,
                           markers=expand_line(mstart, mend)),
               f"{tag} remove markers {mstart}-{mend}")
        _apply(state, Move(type=MoveType.REMOVE_RING, player=pl,
                           source=_pos(ring)),
               f"{tag} remove ring {ring}")


def iter_states() -> Iterator[Tuple[int, str, GameState]]:
    """Yield ``(turn_number, mover_tag, state_copy)`` after each main-game turn.

    ``turn_number`` follows BGA numbering (main game starts at turn 11). The
    yielded state is a copy taken after the full turn (ring move plus any
    row/ring removals), safe for the caller to mutate.
    """
    state = GameState()
    play_placements(state)
    turn_no = len(PLACEMENTS)
    for tag, ringmv, removal in MAIN:
        turn_no += 1
        _apply_turn(state, tag, ringmv, removal)
        yield turn_no, tag, state.copy()
