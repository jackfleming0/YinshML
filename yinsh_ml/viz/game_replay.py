"""Replay recorded YINSH games turn-by-turn.

Reads the parquet schema written by ``yinsh_ml.self_play.data_storage``
(one row per turn) and reconstructs per-turn ``Board`` snapshots by
replaying serialized moves through ``GameState``. Used by the live
viewer to step through games and surface per-turn metrics (the 7
differential heuristic features, search timing, etc.).

The schema stores moves but not full board states — replaying is cheap
enough to do once per game on load. Heuristic feature columns are
preserved verbatim from the parquet for downstream display.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

from ..game.board import Board
from ..game.constants import Player, Position
from ..game.game_state import GameState
from ..game.types import Move, MoveType

logger = logging.getLogger(__name__)


# Columns we treat as game-level metadata vs. turn-level data when
# building a per-game summary. Anything not in this set is assumed to be
# a turn-level feature column.
_GAME_LEVEL_COLS = {
    "game_id", "start_time", "end_time", "duration",
    "total_turns", "winner", "white_score", "black_score",
    "final_phase", "total_moves", "feature_count",
}
_TURN_LEVEL_COLS = {
    "turn_number", "current_player",
    "move_type", "move_source", "move_destination", "move_markers",
    "turn_timestamp",
}


def _parse_position(s: object) -> Optional[Position]:
    if s is None or (isinstance(s, float) and pd.isna(s)) or s == "":
        return None
    return Position.from_string(str(s))


def _deserialize_move(row: pd.Series) -> Move:
    """Reconstruct a ``Move`` from a single parquet row.

    Mirrors ``GameRecorder._serialize_move`` in
    ``yinsh_ml/self_play/game_recorder.py`` — keep the two in sync if
    that serializer changes.
    """
    move_type = MoveType(row["move_type"])
    player = Player(row["current_player"])
    source = _parse_position(row.get("move_source"))
    destination = _parse_position(row.get("move_destination"))

    markers: Optional[Tuple[Position, ...]] = None
    raw_markers = row.get("move_markers")
    if raw_markers is not None and not (isinstance(raw_markers, float) and pd.isna(raw_markers)):
        try:
            marker_list = json.loads(raw_markers) if isinstance(raw_markers, str) else list(raw_markers)
            if marker_list:
                markers = tuple(Position.from_string(s) for s in marker_list)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Could not parse move_markers=%r: %s", raw_markers, e)

    return Move(
        type=move_type, player=player,
        source=source, destination=destination, markers=markers,
    )


@dataclass
class GameReplay:
    """A single replayable game.

    Attributes
    ----------
    game_id:
        Identifier from the parquet record.
    moves:
        Per-turn ``Move`` objects in play order.
    states:
        Board snapshots, length ``len(moves) + 1``. ``states[0]`` is the
        empty starting board; ``states[i+1]`` is the board after
        ``moves[i]`` is applied.
    features:
        Per-turn dict of feature values inlined into the source parquet
        (whatever was beyond the standard game/turn columns). Length
        matches ``moves``. Empty when constructed from a move list with
        no source parquet.
    annotations:
        Per-turn dict of values added by annotators after load — e.g.
        on-the-fly heuristic features, network value predictions, MCTS
        visit distributions. Open-ended schema; consumers look up keys
        they care about. Length matches ``moves`` after any annotator
        has run; otherwise empty. See ``yinsh_ml/viz/annotators.py``.
    winner, white_score, black_score, final_phase, total_turns:
        End-of-game metadata.
    """

    game_id: str
    moves: List[Move] = field(default_factory=list)
    states: List[Board] = field(default_factory=list)
    features: List[Dict[str, float]] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    winner: Optional[str] = None
    white_score: int = 0
    black_score: int = 0
    final_phase: Optional[str] = None
    total_turns: int = 0
    replay_truncated_at: Optional[int] = None
    """Turn index where replay aborted (e.g. an illegal recorded move), or None."""

    def __len__(self) -> int:
        return len(self.moves)

    def board_after(self, turn_index: int) -> Board:
        """Board immediately after the move at ``turn_index`` (0-indexed)."""
        return self.states[turn_index + 1]

    def board_before(self, turn_index: int) -> Board:
        return self.states[turn_index]

    def iter_turns(self) -> Iterator[Tuple[int, Move, Board, Dict[str, float]]]:
        """Yield ``(turn_index, move, board_after, features)`` per turn."""
        for i, move in enumerate(self.moves):
            yield i, move, self.states[i + 1], self.features[i]

    def iter_states(self) -> Iterator[Tuple[int, GameState]]:
        """Yield ``(turn_index, game_state_after_move)`` for every turn.

        Single forward pass through the move list, so this is O(N) — use
        in preference to repeatedly rebuilding state from scratch when
        you need full ``GameState`` (not just ``Board``) at every turn,
        e.g. for computing phase- or score-dependent heuristic features.
        """
        state = GameState()
        for i, move in enumerate(self.moves):
            try:
                state.make_move(move)
            except Exception as e:
                logger.warning(
                    "Game %s: iter_states aborting at turn %d (%s)",
                    self.game_id, i, e,
                )
                return
            yield i, state


def replay_from_moves(
    moves: Sequence[Move],
    *,
    game_id: str = "ad_hoc",
    features: Optional[Sequence[Dict[str, float]]] = None,
    winner: Optional[str] = None,
    white_score: int = 0,
    black_score: int = 0,
) -> GameReplay:
    """Build a ``GameReplay`` from a list of ``Move`` objects.

    Source-agnostic loader — useful for BGA scraper output, in-memory
    move lists from a running self-play worker, synthetic moves in
    tests, or any other producer. Replays the moves through a fresh
    ``GameState`` to materialise per-turn ``Board`` snapshots; aborts
    gracefully and sets ``replay_truncated_at`` on illegal moves.

    ``features`` is an optional per-turn dict (e.g. from a parquet that
    inlined extra columns). Length must match ``moves`` if provided.
    """
    if features is not None and len(features) != len(moves):
        raise ValueError(
            f"features length ({len(features)}) must match moves "
            f"length ({len(moves)})"
        )

    state = GameState()
    applied_moves: List[Move] = []
    states: List[Board] = [copy.deepcopy(state.board)]
    applied_features: List[Dict[str, float]] = []

    for i, move in enumerate(moves):
        try:
            state.make_move(move)
        except Exception as e:
            logger.warning(
                "Game %s: aborting replay at turn %d (%s)", game_id, i + 1, e
            )
            return GameReplay(
                game_id=game_id,
                moves=applied_moves,
                states=states,
                features=applied_features,
                winner=winner,
                white_score=white_score,
                black_score=black_score,
                total_turns=len(applied_moves),
                replay_truncated_at=i,
            )
        applied_moves.append(move)
        states.append(copy.deepcopy(state.board))
        if features is not None:
            applied_features.append(features[i])

    return GameReplay(
        game_id=game_id,
        moves=applied_moves,
        states=states,
        features=applied_features,
        winner=winner,
        white_score=white_score,
        black_score=black_score,
        total_turns=len(applied_moves),
    )


def _build_replay(game_id: str, rows: pd.DataFrame, feature_cols: Sequence[str]) -> GameReplay:
    """Replay one parquet game by deserialising its rows then calling
    ``replay_from_moves``.

    Thin adapter that handles the parquet-specific concerns
    (deserialisation, endgame metadata extraction) and delegates the
    actual replay loop to the source-agnostic primitive.
    """
    rows = rows.sort_values("turn_number")

    moves: List[Move] = []
    features: List[Dict[str, float]] = []
    for _, row in rows.iterrows():
        try:
            moves.append(_deserialize_move(row))
        except Exception as e:
            logger.warning(
                "Game %s: move deserialisation failed at turn %s (%s)",
                game_id, row.get("turn_number"), e,
            )
            break
        features.append({c: row[c] for c in feature_cols if c in row.index})

    replay = replay_from_moves(
        moves,
        game_id=game_id,
        features=features if len(features) == len(moves) else None,
    )
    _fill_endgame_metadata(replay, rows)
    return replay


def _fill_endgame_metadata(replay: GameReplay, rows: pd.DataFrame) -> None:
    first = rows.iloc[0]
    replay.winner = _maybe(first.get("winner"))
    replay.white_score = int(first.get("white_score", 0) or 0)
    replay.black_score = int(first.get("black_score", 0) or 0)
    replay.final_phase = _maybe(first.get("final_phase"))
    replay.total_turns = int(first.get("total_turns", len(replay.moves)) or len(replay.moves))


def _maybe(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return v


def list_games(parquet_dir: Path) -> pd.DataFrame:
    """Summarize available games in a parquet directory.

    Returns one row per game with ``game_id, winner, total_turns,
    white_score, black_score, duration, final_phase, source_file``.
    Cheap — reads only game-level columns from each parquet file.
    """
    parquet_dir = Path(parquet_dir)
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=[
            "game_id", "winner", "total_turns",
            "white_score", "black_score", "duration",
            "final_phase", "source_file",
        ])

    keep_cols = [
        "game_id", "winner", "total_turns",
        "white_score", "black_score", "duration", "final_phase",
    ]
    dfs = []
    for fp in files:
        try:
            df = pd.read_parquet(fp, columns=keep_cols)
        except Exception as e:
            logger.warning("Could not read %s: %s", fp, e)
            continue
        df = df.drop_duplicates(subset="game_id")
        df["source_file"] = fp.name
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_game(parquet_dir: Path, game_id: str) -> GameReplay:
    """Load and replay a single game by id.

    Scans parquet files until the game is found. For interactive use
    over a small set of games this is fine; for high-throughput access
    consider caching the index of which file each game lives in.
    """
    parquet_dir = Path(parquet_dir)
    for fp in sorted(parquet_dir.glob("*.parquet")):
        df = pd.read_parquet(fp)
        if "game_id" not in df.columns:
            continue
        game_rows = df[df["game_id"] == game_id]
        if game_rows.empty:
            continue
        feature_cols = [
            c for c in df.columns
            if c not in _GAME_LEVEL_COLS and c not in _TURN_LEVEL_COLS
        ]
        return _build_replay(game_id, game_rows, feature_cols)
    raise KeyError(f"Game {game_id!r} not found under {parquet_dir}")


def replay_from_dataframe(df: pd.DataFrame, game_id: Optional[str] = None) -> GameReplay:
    """Build a ``GameReplay`` directly from an in-memory DataFrame.

    Useful for unit tests and for replaying a single game that's already
    in memory (e.g. just produced by a self-play harness). If
    ``game_id`` is None, the value is taken from the first row.
    """
    if game_id is None:
        game_id = str(df["game_id"].iloc[0])
    rows = df[df["game_id"] == game_id]
    if rows.empty:
        raise KeyError(f"No rows for game_id={game_id!r}")
    feature_cols = [
        c for c in df.columns
        if c not in _GAME_LEVEL_COLS and c not in _TURN_LEVEL_COLS
    ]
    return _build_replay(game_id, rows, feature_cols)


__all__ = [
    "GameReplay",
    "list_games",
    "load_game",
    "replay_from_dataframe",
    "replay_from_moves",
]
