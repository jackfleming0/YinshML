#!/usr/bin/env python3
"""Replay parsed BGA games through the engine to find rule divergences.

Loads every JSON in expert_games/bga/parsed/ and feeds the move stream
into a fresh GameState via make_move. Logs each failure with enough
context to diagnose: move index, move, phase, current player, and
a short reason. Also compares the final result (winner) to BGA's
recorded result.

Usage:
    python scripts/audit_bga_replays.py [--limit N] [--verbose]
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yinsh_ml.game.constants import Player, Position
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import Move, MoveType, GamePhase


# Silence engine's own debug logs (we want a clean audit report).
logging.getLogger('yinsh_ml.game.game_state').setLevel(logging.CRITICAL)
logging.getLogger('yinsh_ml.game.board').setLevel(logging.CRITICAL)


PLAYER = {'white': Player.WHITE, 'black': Player.BLACK}


def _pos(s: str) -> Position:
    return Position.from_string(s)


def _build_move(m: dict, phase: GamePhase, current_player: Player) -> Move:
    """Convert parsed dict into Move, letting phase disambiguate where needed.

    The parsed schema has MOVE_RING for main-game ring motion. It does NOT
    know about the engine's phase transitions (ring-placement uses PLACE_RING,
    ring-removal uses REMOVE_RING).
    """
    mt = m['move_type']
    player = PLAYER[m['player']]
    if mt == 'PLACE_RING':
        return Move(type=MoveType.PLACE_RING, player=player,
                    source=_pos(m['position']))
    if mt == 'MOVE_RING':
        return Move(type=MoveType.MOVE_RING, player=player,
                    source=_pos(m['source']),
                    destination=_pos(m['destination']))
    if mt == 'REMOVE_MARKERS':
        return Move(type=MoveType.REMOVE_MARKERS, player=player,
                    markers=tuple(_pos(p) for p in m['markers']))
    if mt == 'REMOVE_RING':
        return Move(type=MoveType.REMOVE_RING, player=player,
                    source=_pos(m['position']))
    raise ValueError(f"Unknown move_type: {mt}")


def replay_game(game: dict) -> dict:
    """Replay a single parsed game. Returns a diagnostic dict."""
    gid = game['game_id']
    state = GameState()
    moves = game['moves']
    failure = None
    replayed = 0

    for i, m in enumerate(moves):
        try:
            mv = _build_move(m, state.phase, state.current_player)
        except Exception as e:
            failure = {
                'kind': 'build_error',
                'move_index': i,
                'raw_move': m,
                'error': str(e),
                'phase': state.phase.name,
                'current_player': state.current_player.name,
            }
            break

        ok = state.make_move(mv)
        if not ok:
            # Reconstruct the reason — replay validation with more context.
            reason = diagnose_rejection(state, mv)
            failure = {
                'kind': 'rejected',
                'move_index': i,
                'raw_move': m,
                'mv_str': str(mv),
                'phase': state.phase.name,
                'current_player': state.current_player.name,
                'white_rings_placed': state.rings_placed[Player.WHITE],
                'black_rings_placed': state.rings_placed[Player.BLACK],
                'white_score': state.white_score,
                'black_score': state.black_score,
                'reason': reason,
            }
            break
        replayed += 1

    # Final state check.
    final_winner = None
    if state.white_score >= 3:
        final_winner = 'white'
    elif state.black_score >= 3:
        final_winner = 'black'

    parsed_result = game.get('result', 'unknown')

    winner_match = True
    if parsed_result in ('white', 'black'):
        winner_match = (final_winner == parsed_result)

    return {
        'game_id': gid,
        'total_moves': len(moves),
        'replayed': replayed,
        'completed': failure is None,
        'failure': failure,
        'parsed_result': parsed_result,
        'final_winner': final_winner,
        'final_phase': state.phase.name,
        'white_score': state.white_score,
        'black_score': state.black_score,
        'winner_match': winner_match,
    }


def diagnose_rejection(state: GameState, mv: Move) -> str:
    """Give a best-guess reason a move was rejected. Read-only on state."""
    if mv.player != state.current_player:
        return f'player_mismatch(move={mv.player.name}, cur={state.current_player.name})'

    if mv.type == MoveType.PLACE_RING:
        if state.phase != GamePhase.RING_PLACEMENT:
            return f'wrong_phase(expected=RING_PLACEMENT, got={state.phase.name})'
        piece = state.board.get_piece(mv.source)
        if piece is not None:
            return f'square_occupied_at_{mv.source}_by_{piece}'
        return 'unknown_place_ring_reject'

    if mv.type == MoveType.MOVE_RING:
        if state.phase != GamePhase.MAIN_GAME:
            return f'wrong_phase(expected=MAIN_GAME, got={state.phase.name})'
        src_piece = state.board.get_piece(mv.source)
        if src_piece is None:
            return f'source_empty_at_{mv.source}'
        if not src_piece.is_ring():
            return f'source_not_ring(piece={src_piece})'
        if src_piece.get_player() != mv.player:
            return f'source_wrong_color(piece={src_piece}, mover={mv.player.name})'
        dst_piece = state.board.get_piece(mv.destination)
        if dst_piece is not None:
            return f'dest_occupied_at_{mv.destination}_by_{dst_piece}'
        valid = state.board.valid_move_positions(mv.source)
        if mv.destination not in valid:
            return f'dest_not_reachable({mv.source}->{mv.destination})'
        return 'unknown_move_ring_reject'

    if mv.type == MoveType.REMOVE_MARKERS:
        if state.phase != GamePhase.ROW_COMPLETION:
            return f'wrong_phase(expected=ROW_COMPLETION, got={state.phase.name})'
        if not mv.markers or len(mv.markers) != 5:
            return f'bad_marker_count({len(mv.markers) if mv.markers else 0})'
        # Deep check: per-square inspection
        from yinsh_ml.game.constants import PieceType
        expected = (PieceType.WHITE_MARKER if mv.player == Player.WHITE
                    else PieceType.BLACK_MARKER)
        bad = []
        for p in mv.markers:
            piece = state.board.get_piece(p)
            if piece != expected:
                bad.append(f'{p}={piece}')
        if bad:
            return f'marker_squares_not_own_color[{",".join(bad[:3])}]'
        return 'sequence_not_accepted_by_is_valid_marker_sequence'

    if mv.type == MoveType.REMOVE_RING:
        if state.phase != GamePhase.RING_REMOVAL:
            return f'wrong_phase(expected=RING_REMOVAL, got={state.phase.name})'
        from yinsh_ml.game.constants import PieceType
        expected = (PieceType.WHITE_RING if mv.player == Player.WHITE
                    else PieceType.BLACK_RING)
        piece = state.board.get_piece(mv.source)
        if piece != expected:
            return f'wrong_piece_at_{mv.source}(got={piece}, expected={expected})'
        return 'unknown_remove_ring_reject'

    return 'unrecognized'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parsed-dir', default='expert_games/bga/parsed')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    parsed_dir = Path(args.parsed_dir)
    files = sorted(parsed_dir.glob('*.json'))
    if args.limit:
        files = files[:args.limit]

    print(f"Auditing {len(files)} parsed BGA games from {parsed_dir}")
    print('=' * 72)

    results = []
    for f in files:
        with open(f) as fh:
            game = json.load(fh)
        res = replay_game(game)
        results.append(res)

    # Summary
    succ = [r for r in results if r['completed']]
    fail = [r for r in results if not r['completed']]
    print(f"\nREPLAY: succeeded={len(succ)}  failed={len(fail)}  total={len(results)}")

    # Failure categorization
    cat = Counter()
    examples = {}
    for r in fail:
        f = r['failure']
        key = f.get('reason', f.get('kind', 'unknown'))
        # Collapse position-specific keys to families
        # e.g., square_occupied_at_F6 -> square_occupied_at_*
        fam = _family(key)
        cat[fam] += 1
        examples.setdefault(fam, r)

    print("\nFAILURE CATEGORIES:")
    for fam, n in cat.most_common():
        print(f"  [{n:3d}] {fam}")
        r = examples[fam]
        f = r['failure']
        print(f"        e.g. {r['game_id']} move#{f['move_index']} "
              f"phase={f.get('phase')} cur={f.get('current_player')}")
        print(f"        raw: {f.get('raw_move')}")
        print(f"        reason: {f.get('reason')}")

    # Final-state comparison
    print("\nFINAL-STATE (winner) AGREEMENT:")
    wm = Counter()
    for r in results:
        wm[r['winner_match']] += 1
    print(f"  match={wm[True]}  mismatch={wm[False]}")

    mismatches = [r for r in results if not r['winner_match']]
    for r in mismatches[:8]:
        print(f"  - {r['game_id']}: parsed={r['parsed_result']} "
              f"engine_winner={r['final_winner']} "
              f"(W={r['white_score']} B={r['black_score']} "
              f"phase={r['final_phase']} "
              f"replayed={r['replayed']}/{r['total_moves']})")

    # Phase-at-failure
    phase_at_fail = Counter(
        r['failure'].get('phase', '?') for r in fail
    )
    print("\nPHASE WHEN REJECTION HAPPENED:")
    for k, n in phase_at_fail.most_common():
        print(f"  {k}: {n}")

    # Move-type-at-failure
    mt_at_fail = Counter(
        r['failure'].get('raw_move', {}).get('move_type', '?') for r in fail
    )
    print("\nMOVE TYPE WHEN REJECTION HAPPENED:")
    for k, n in mt_at_fail.most_common():
        print(f"  {k}: {n}")

    # Median progress before failure (helps estimate where in a game bugs trigger)
    if fail:
        progresses = sorted(
            (r['replayed'] / max(r['total_moves'], 1), r)
            for r in fail
        )
        med = progresses[len(progresses) // 2]
        print(f"\nMedian failure progress: {med[0]:.0%} "
              f"(e.g. {med[1]['game_id']} "
              f"{med[1]['replayed']}/{med[1]['total_moves']})")


def _family(key: str) -> str:
    """Collapse instance-specific keys (positions, pieces) into families."""
    import re
    k = re.sub(r'_at_[A-K]\d+', '_at_*', key)
    k = re.sub(r'\([^)]*\)', '(...)', k)
    k = re.sub(r'\[[^\]]*\]', '[...]', k)
    return k


if __name__ == '__main__':
    main()
