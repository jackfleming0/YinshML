"""Convert yngine volume corpus (text format) → .npz training data.

Reads `shard_*.txt` files emitted by `yngine_volume`, replays each game
through our `GameState` to validate + encode states, writes the resulting
(state, one-hot policy, value) tuples in the same .npz schema that
`scripts/run_supervised_pretraining.py` consumes.

yngine ↔ our coordinate mapping (derived in /tmp/find_mapping.py):
  yngine (x, y) → our (col=chr('A'+x), row=11-y)
All 6 yngine hex directions land on our 3 hex axes.

Input line formats:
  G <gid>                              -- game start
  P <W|B> <x> <y>                       -- place ring
  M <W|B> <fx> <fy> <tx> <ty> <dir>     -- ring move
  R <W|B> <fx> <fy> <dir>               -- remove row (5 markers along dir)
  X <W|B> <x> <y>                       -- remove ring
  S                                     -- pass
  E <W|B|D> <total_moves>               -- game end

Usage:
  python scripts/yngine_corpus_to_npz.py \\
      --corpus-dir /tmp/yngine_corpus \\
      --output expert_games/yngine_corpus.npz
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yinsh_ml.game.constants import Player, Position, PieceType  # noqa: E402
from yinsh_ml.game.game_state import GameState  # noqa: E402
from yinsh_ml.game.types import Move, MoveType  # noqa: E402
from yinsh_ml.utils.encoding import StateEncoder  # noqa: E402
from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder  # noqa: E402

logger = logging.getLogger("yngine_corpus")

# yngine direction → (Δx, Δy), matches common.hpp::direction_to_vec2.
YNGINE_DIR_VEC = {
    0: (+1,  0),  # SE
    1: ( 0, +1),  # NE
    2: (-1, +1),  # N
    3: (-1,  0),  # NW
    4: ( 0, -1),  # SW
    5: (+1, -1),  # S
}

# yngine signs convention: white plays first; W = WHITE, B = BLACK.
PLAYER_MAP = {'W': Player.WHITE, 'B': Player.BLACK}


def yngine_xy_to_pos(x: int, y: int) -> Position:
    """yngine (x, y) → our Position (col=chr('A'+x), row=11-y)."""
    if not (0 <= x <= 10 and 0 <= y <= 10):
        raise ValueError(f"yngine (x, y) out of range: ({x}, {y})")
    col = chr(ord('A') + x)
    row = 11 - y
    return Position(col, row)


def yngine_row_positions(x: int, y: int, direction: int) -> List[Position]:
    """A length-5 row of markers starting at yngine (x, y) in given direction."""
    dx, dy = YNGINE_DIR_VEC[direction]
    return [yngine_xy_to_pos(x + i * dx, y + i * dy) for i in range(5)]


class CorpusGame:
    """One parsed yngine game."""
    __slots__ = ("gid", "moves", "outcome", "n_moves")

    def __init__(self, gid: int):
        self.gid = gid
        self.moves: List[Tuple[str, Player, dict]] = []  # (kind, player, fields)
        self.outcome: str = ""
        self.n_moves: int = 0


def parse_shard(path: Path) -> Iterator[CorpusGame]:
    """Stream games from a shard file."""
    current: CorpusGame | None = None
    for raw in path.open("r"):
        line = raw.strip()
        if not line:
            continue
        kind = line[0]
        if kind == 'G':
            if current is not None:
                logger.warning(f"{path}: game {current.gid} had no E line, skipping")
            current = CorpusGame(int(line.split()[1]))
        elif kind == 'E':
            if current is None:
                continue
            parts = line.split()
            current.outcome = parts[1]
            current.n_moves = int(parts[2])
            yield current
            current = None
        elif kind in ('P', 'M', 'R', 'X', 'S'):
            if current is None:
                continue
            parts = line.split()
            if kind == 'P':
                _, p, x, y = parts
                current.moves.append(('P', PLAYER_MAP[p], {'x': int(x), 'y': int(y)}))
            elif kind == 'M':
                _, p, fx, fy, tx, ty, d = parts
                current.moves.append(('M', PLAYER_MAP[p], {
                    'fx': int(fx), 'fy': int(fy),
                    'tx': int(tx), 'ty': int(ty), 'dir': int(d),
                }))
            elif kind == 'R':
                _, p, fx, fy, d = parts
                current.moves.append(('R', PLAYER_MAP[p], {
                    'fx': int(fx), 'fy': int(fy), 'dir': int(d),
                }))
            elif kind == 'X':
                _, p, x, y = parts
                current.moves.append(('X', PLAYER_MAP[p], {'x': int(x), 'y': int(y)}))
            elif kind == 'S':
                # PassMove has no player in yngine output; treat as current side's pass.
                current.moves.append(('S', None, {}))


def replay_game(game: CorpusGame, encoder: StateEncoder
                ) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """Replay a yngine game through our GameState. Returns (states, move_idxs, players).

    - states: encoded state tensors at each turn (before move applied)
    - move_idxs: the played move's slot in the policy vector
    - players: who played at that turn (for POV-correct value labeling)

    Raises on rule-violation mismatch — that's the validation.
    """
    gs = GameState()
    states: List[np.ndarray] = []
    move_idxs: List[int] = []
    players: List[int] = []  # 0 = WHITE, 1 = BLACK; for value POV

    for kind, player, f in game.moves:
        # Snapshot state BEFORE applying move
        state_tensor = encoder.encode_state(gs).astype(np.float32)

        # Translate yngine move → our Move
        if kind == 'P':
            mv = Move(type=MoveType.PLACE_RING, player=player,
                      source=yngine_xy_to_pos(f['x'], f['y']))
        elif kind == 'M':
            mv = Move(type=MoveType.MOVE_RING, player=player,
                      source=yngine_xy_to_pos(f['fx'], f['fy']),
                      destination=yngine_xy_to_pos(f['tx'], f['ty']))
        elif kind == 'R':
            mv = Move(type=MoveType.REMOVE_MARKERS, player=player,
                      markers=tuple(yngine_row_positions(f['fx'], f['fy'], f['dir'])))
        elif kind == 'X':
            mv = Move(type=MoveType.REMOVE_RING, player=player,
                      source=yngine_xy_to_pos(f['x'], f['y']))
        elif kind == 'S':
            # PassMove — our codebase may not have this; treat as no-op and skip recording.
            continue
        else:
            raise ValueError(f"unknown move kind: {kind}")

        mv_idx = encoder.move_to_index(mv)

        # Apply through GameState
        ok = gs.make_move(mv)
        if not ok:
            raise RuntimeError(f"game {game.gid}: move {mv} rejected by GameState at turn {len(states)}")

        states.append(state_tensor)
        move_idxs.append(mv_idx)
        players.append(0 if player == Player.WHITE else 1)

    return states, move_idxs, players


def outcome_value(outcome: str, player_idx: int) -> float:
    """Game terminal value from `player_idx`'s POV. outcome ∈ {W, B, D}."""
    if outcome == 'D':
        return 0.0
    winner_is_white = (outcome == 'W')
    player_is_white = (player_idx == 0)
    return +1.0 if winner_is_white == player_is_white else -1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-games", type=int, default=None,
                        help="cap on games to convert (debug)")
    parser.add_argument("--shard-glob", type=str, default="shard_*.txt")
    parser.add_argument("--use-enhanced-encoding", action="store_true",
                        help="Encode states with EnhancedStateEncoder (15 channels) "
                             "instead of StateEncoder (6 channels). Required for "
                             "Branch D.2 corpus generation. Move-encoding API is "
                             "shared so policy indices stay identical between the two.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.use_enhanced_encoding:
        encoder = EnhancedStateEncoder()
        logger.info("Using EnhancedStateEncoder (15 channels)")
    else:
        encoder = StateEncoder()
        logger.info("Using StateEncoder (6 channels)")
    total_moves_slots = encoder.total_moves

    all_states: List[np.ndarray] = []
    all_policy_idx: List[int] = []     # int32; expand to one-hot at load time
    all_values: List[float] = []

    shards = sorted(args.corpus_dir.glob(args.shard_glob))
    logger.info(f"Reading {len(shards)} shard(s) from {args.corpus_dir}")

    games_ok = 0
    games_failed = 0
    positions = 0
    for shard in shards:
        for game in parse_shard(shard):
            if args.max_games and games_ok >= args.max_games:
                break
            try:
                states, mv_idxs, players = replay_game(game, encoder)
            except Exception as e:
                games_failed += 1
                logger.debug(f"game {game.gid} replay failed: {e}")
                continue
            for state_tensor, mv_idx, pidx in zip(states, mv_idxs, players):
                value = outcome_value(game.outcome, pidx)
                all_states.append(state_tensor)
                all_policy_idx.append(int(mv_idx))
                all_values.append(value)
                positions += 1
            games_ok += 1
            if games_ok % 1000 == 0:
                logger.info(f"  processed {games_ok} games, {positions} positions")
        if args.max_games and games_ok >= args.max_games:
            break

    logger.info(f"Done: {games_ok} games OK, {games_failed} failed, {positions} positions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Note: schema differs from the supervised-pretrain expected (`policies`
    # as full (N, total_moves) one-hot). For 13.6M positions × 7433 slots
    # one-hot is ~400GB; we instead save the argmax index per position
    # (4 bytes/position) plus the total_moves size as metadata so the
    # downstream dataloader can materialize on-demand.
    np.savez_compressed(
        args.output,
        states=np.asarray(all_states, dtype=np.float32),
        policy_indices=np.asarray(all_policy_idx, dtype=np.int32),
        values=np.asarray(all_values, dtype=np.float32),
        total_moves=np.int32(total_moves_slots),
    )
    logger.info(f"Wrote {args.output} ({len(all_states)} positions, "
                f"policy stored as int32 indices to save space — see comment)")


if __name__ == "__main__":
    main()
