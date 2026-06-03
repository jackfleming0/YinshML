"""Experimental heuristic feature palette ("let a thousand flowers bloom").

This module is a *staging ground* for candidate strategic signals surfaced by
reviewing strong human games (see ``docs/game_reviews/bga_862307561_review.md``).
The review showed our production 7-feature set is essentially attack-oriented,
all-differential, and — crucially — that two of its top-weighted features are
inert during real play. The features here capture strategic ideas the current
set structurally cannot express: ring optionality, immediate scoring pressure,
defense/disruption, board tempo, and ring degradation.

**These features are deliberately NOT wired into the production evaluator's
default weights.** They are an opt-in palette: compute them, log them, feed
them to weight-learning or ablation experiments, and let the data decide which
ones matter. Nothing here pre-commits to a "best" strategy or a fixed weight.

All functions follow the house convention: return a float **differential**
(player value minus opponent value) where *higher is better for ``player``*,
and take ``(game_state, player)``. They are pure (no mutation of state).
"""

from typing import Callable, Dict, List

from ..game.game_state import GameState
from ..game.constants import Player, PieceType, Position, is_valid_position
from .features import _maximal_marker_runs


# --- small geometry/board helpers -----------------------------------------

def _ring_positions(board, player: Player) -> List[Position]:
    ring = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING
    return board.get_pieces_positions(ring)


def _marker_type(player: Player) -> PieceType:
    return PieceType.WHITE_MARKER if player == Player.WHITE else PieceType.BLACK_MARKER


def _path_between(src: Position, dst: Position) -> List[Position]:
    """Cells strictly between ``src`` and ``dst`` along a hex line (exclusive).

    These are exactly the cells a ring jumps over — any opponent markers among
    them get flipped when the ring moves.
    """
    dc = ord(dst.column) - ord(src.column)
    dr = dst.row - src.row
    steps = max(abs(dc), abs(dr))
    if steps == 0:
        return []
    sc = dc // steps if dc else 0
    sr = dr // steps if dr else 0
    return [
        Position(chr(ord(src.column) + sc * i), src.row + sr * i)
        for i in range(1, steps)
    ]


def _run_extension_cells(run) -> List[Position]:
    """The two cells continuing a maximal run's line beyond each end.

    ``run`` is a sorted tuple of (column, row) cells (as produced by
    ``_maximal_marker_runs``). Returns the on-board extension cells.
    """
    cells = [Position(c[0], c[1]) for c in sorted(run)]
    p0, p1, plast = cells[0], cells[1], cells[-1]
    sc = (ord(p1.column) - ord(p0.column))
    sr = (p1.row - p0.row)
    # normalize (already unit by construction, but be safe)
    step = max(abs(sc), abs(sr)) or 1
    sc //= step
    sr //= step
    low = Position(chr(ord(p0.column) - sc), p0.row - sr)
    high = Position(chr(ord(plast.column) + sc), plast.row + sr)
    return [c for c in (low, high) if is_valid_position(c)]


# --- the palette -----------------------------------------------------------

def ring_mobility_differential(game_state: GameState, player: Player) -> float:
    """Total legal ring-move destinations available, mine minus opponent's.

    Strategic idea: **optionality / mobility preservation.** YINSH rings
    degrade as markers fill the board; the player who keeps more rings free to
    threaten multiple future rows tends to win the conversion race. The human
    game was won largely on this axis, which the geometric ``ring_positioning``
    / ``ring_spread`` features (position, not freedom) do not capture.
    """
    b = game_state.board
    mine = sum(len(b.valid_move_positions(p)) for p in _ring_positions(b, player))
    opp = sum(len(b.valid_move_positions(p)) for p in _ring_positions(b, player.opponent))
    return float(mine - opp)


def ring_confinement_pressure(game_state: GameState, player: Player) -> float:
    """Boxed-in rings: (opponent rings with <=1 move) - (mine with <=1 move).

    Strategic idea: **ring degradation / death.** A ring with no moves is dead
    weight (and, with no legal move anywhere, a loss). Positive = the opponent
    is more cramped than I am. Complements mobility with a tail-risk view.
    """
    b = game_state.board
    mine = sum(1 for p in _ring_positions(b, player) if len(b.valid_move_positions(p)) <= 1)
    opp = sum(1 for p in _ring_positions(b, player.opponent) if len(b.valid_move_positions(p)) <= 1)
    return float(opp - mine)


def near_completion_threats(game_state: GameState, player: Player) -> float:
    """Immediately-completable rows, mine minus opponent's.

    A maximal 4-run is an *immediate* scoring threat if one of its extension
    cells holds a same-color **ring that has at least one legal move**: moving
    that ring leaves a marker on the extension cell, completing a 5-row.

    Strategic idea: a live replacement for ``completed_runs_differential``,
    which reads ~0 in real play because completed rows are removed within the
    same turn. This measures scoring pressure *before* the cash-in.
    """
    return float(_completion_threats(game_state, player)
                 - _completion_threats(game_state, player.opponent))


def _completion_threats(game_state: GameState, player: Player) -> int:
    b = game_state.board
    ring = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING
    count = 0
    for run in _maximal_marker_runs(b, _marker_type(player)):
        if len(run) != 4:
            continue
        for ext in _run_extension_cells(run):
            if b.get_piece(ext) == ring and len(b.valid_move_positions(ext)) >= 1:
                count += 1
                break  # count the run once
    return count


def defensive_disruption(game_state: GameState, player: Player) -> float:
    """Disruption initiative: (opponent near-rows I can break) minus (my
    near-rows the opponent can break).

    A near-row is a maximal run of length >= 3. It is "breakable" by a side if
    one of its markers lies on the path of a legal ring move of that side (a
    passed-over marker flips color, severing the run).

    Strategic idea: **the missing defensive term.** Every production feature is
    additive self-progress; none rewards *denying* the opponent. Positive =
    I can spoil more of the opponent's development than they can of mine — which
    should make shallow-depth play far less suicidal.
    """
    return float(_breakable_runs(game_state, player.opponent, by=player)
                 - _breakable_runs(game_state, player, by=player.opponent))


def _breakable_runs(game_state: GameState, owner: Player, by: Player) -> int:
    """Count ``owner``'s maximal runs (length >= 3) that ``by`` can sever by
    flipping one of their markers via a legal ring move."""
    b = game_state.board
    owner_marker = _marker_type(owner)
    flippable = set()
    for p in _ring_positions(b, by):
        for dst in b.valid_move_positions(p):
            for cell in _path_between(p, dst):
                if b.get_piece(cell) == owner_marker:
                    flippable.add((cell.column, cell.row))
    if not flippable:
        return 0
    return sum(
        1 for run in _maximal_marker_runs(b, owner_marker)
        if len(run) >= 3 and any(c in flippable for c in run)
    )


def marker_tempo_differential(game_state: GameState, player: Player) -> float:
    """Markers on board, mine minus opponent's.

    Strategic idea: **board presence / tempo.** A coarse proxy for who has been
    dictating play. Cheap, and a useful normalizer alongside the run features.
    """
    b = game_state.board
    mine = len(b.get_pieces_positions(_marker_type(player)))
    opp = len(b.get_pieces_positions(_marker_type(player.opponent)))
    return float(mine - opp)


# --- registry + aggregator -------------------------------------------------

EXPERIMENTAL_FEATURE_FNS: Dict[str, Callable[[GameState, Player], float]] = {
    "ring_mobility_differential": ring_mobility_differential,
    "ring_confinement_pressure": ring_confinement_pressure,
    "near_completion_threats": near_completion_threats,
    "defensive_disruption": defensive_disruption,
    "marker_tempo_differential": marker_tempo_differential,
}


def extract_experimental_features(game_state: GameState, player: Player) -> Dict[str, float]:
    """Compute the full experimental palette for ``(game_state, player)``.

    Returns a name -> float dict. Intended for logging, ablation, and
    weight-learning experiments — NOT consumed by the production evaluator's
    default weights.
    """
    return {name: fn(game_state, player) for name, fn in EXPERIMENTAL_FEATURE_FNS.items()}
