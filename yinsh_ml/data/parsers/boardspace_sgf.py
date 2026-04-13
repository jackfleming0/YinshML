"""Parser for Boardspace.net YINSH SGF files.

Boardspace uses a proprietary SGF variant where moves are encoded as
text commands inside SGF property values:

  Header:
    GM[24] SU[Yinsh] P0[id "white_player"] P1[id "black_player"]
    RE[Game won by PlayerName] or RE[The game is a draw]

  Move nodes (semicolon-delimited):
    ; P0[1 place wr  G 11]TM[3261]     — place white ring
    ; P0[21 place w  K 8]TM[19220]     — place marker (step 1 of move)
    ; P0[22 drop board  K 9]TM[20057]  — drop ring (step 2 of move)
    ; P1[24 move G 9 F 9]TM[8343]      — combined ring move
    ; P1[91 remove b C 2 C 6]TM[76592] — remove markers (endpoints)
    ; P1[93 remove br I 6]TM[77760]    — remove ring
    ; P1[23 Resign ]TM[5000]           — resignation

Coordinates are space-separated: "G 11" → Position('G', 11).
P0 = White (first player), P1 = Black.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

from ...game.constants import Position, is_valid_position
from .utils import positions_on_line

logger = logging.getLogger(__name__)

# Regex patterns for SGF parsing
_PROP_RE = re.compile(r'(\w+)\[([^\]]*)\]')
_NODE_RE = re.compile(r';\s*(P[01])\[([^\]]*)\]')
_PLAYER_ID_RE = re.compile(r'id\s+"([^"]*)"')


def parse_boardspace_sgf(sgf_text: str) -> Optional[Dict]:
    """Parse a Boardspace SGF file into a standardized game dict.

    Args:
        sgf_text: Full SGF file contents.

    Returns:
        Standardized game dict, or None if parsing fails.
    """
    try:
        header = _parse_header(sgf_text)
        if header is None:
            return None

        moves = _parse_moves(sgf_text, header)

        return {
            'source': 'boardspace',
            'game_id': header.get('game_name', 'unknown'),
            'players': {
                'white': {
                    'name': header.get('white_name', 'unknown'),
                    'rating': 0,
                },
                'black': {
                    'name': header.get('black_name', 'unknown'),
                    'rating': 0,
                },
            },
            'result': header.get('result', 'unknown'),
            'moves': moves,
            'quality_tier': 'human',
        }
    except Exception as e:
        logger.error(f"Failed to parse Boardspace SGF: {e}")
        return None


def _parse_header(sgf_text: str) -> Optional[Dict]:
    """Extract header properties from SGF text."""
    props = {}
    for match in _PROP_RE.finditer(sgf_text):
        key, value = match.group(1), match.group(2).strip()
        if key not in props:
            props[key] = value

    # Validate game type
    if props.get('GM') != '24' and props.get('SU', '').lower() not in ('yinsh', 'yinsh-blitz'):
        logger.warning(f"Not a YINSH game: GM={props.get('GM')}, SU={props.get('SU')}")
        return None

    # Extract player names
    white_name = 'unknown'
    black_name = 'unknown'
    p0_val = props.get('P0', '')
    p1_val = props.get('P1', '')

    m0 = _PLAYER_ID_RE.search(p0_val)
    m1 = _PLAYER_ID_RE.search(p1_val)
    if m0:
        white_name = m0.group(1)
    if m1:
        black_name = m1.group(1)

    # Extract result
    result = _parse_result(props.get('RE', ''), white_name, black_name)

    return {
        'white_name': white_name,
        'black_name': black_name,
        'result': result,
        'game_name': props.get('GN', 'unknown'),
        'game_type': props.get('SU', 'Yinsh'),
    }


def _parse_result(re_value: str, white_name: str, black_name: str) -> str:
    """Parse the RE property to determine game result.

    Handles multiple locales — Boardspace uses the client's language for
    result strings. Known formats:
      English: "Game won by PlayerName"
      Russian: "Игру выиграл(а) PlayerName."
      General: any string containing a player's name indicates they won.
    """
    re_lower = re_value.lower()

    if 'draw' in re_lower:
        return 'draw'

    # Try to find a player name anywhere in the result string.
    # This handles all locales without needing to know each language's
    # "won by" phrasing.
    re_clean = re_value.rstrip('.')  # strip trailing period
    w_lower = white_name.lower()
    b_lower = black_name.lower()

    if w_lower in re_clean.lower():
        return 'white'
    if b_lower in re_clean.lower():
        return 'black'

    return 'unknown'


def _parse_moves(sgf_text: str, header: Dict) -> List[Dict]:
    """Parse all move nodes from SGF text."""
    moves = []
    # State machine for the place+drop two-step move
    pending_place_pos = None  # Position from "place w/b" waiting for "drop board"
    pending_place_player = None

    # Find all move nodes
    for match in _NODE_RE.finditer(sgf_text):
        player_tag = match.group(1)  # P0 or P1
        command_str = match.group(2).strip()

        # P0 = white, P1 = black
        player = 'white' if player_tag == 'P0' else 'black'

        # Strip sequence number from front if present: "21 place w  K 8" → "place w  K 8"
        # Old-format SGFs sometimes omit the sequence number (e.g. "; P0[move B 5 H 11]").
        parts = command_str.split(None, 1)
        if len(parts) == 0:
            continue
        if len(parts) == 2 and parts[0].isdigit():
            cmd = parts[1].strip()
        else:
            cmd = command_str.strip()

        # Skip pure metadata commands like "time 0:04:03"
        if cmd.startswith('time '):
            continue

        parsed = _parse_command(cmd, player)
        if parsed is None:
            # Skip Done, Start, pick, Resign (handled below), etc.
            if cmd.startswith('Resign'):
                # Mark the opponent as winner
                opponent = 'black' if player == 'white' else 'white'
                header['result'] = opponent
            continue

        cmd_type = parsed.get('_type')

        if cmd_type == 'place_marker':
            # First step of a two-step move: "place w K 8"
            # Buffer the position, wait for "drop board"
            pending_place_pos = parsed['position']
            pending_place_player = player
            continue

        if cmd_type == 'drop_board':
            # Second step: "drop board K 9"
            if pending_place_pos is not None:
                moves.append({
                    'move_type': 'MOVE_RING',
                    'player': pending_place_player or player,
                    'source': pending_place_pos,
                    'destination': parsed['position'],
                })
                pending_place_pos = None
                pending_place_player = None
            else:
                logger.warning(f"drop board without preceding place: {cmd}")
            continue

        # If we get here with a pending place, something went wrong
        if pending_place_pos is not None:
            logger.warning(f"Orphaned place w/b at {pending_place_pos}, "
                          f"next command was: {cmd}")
            pending_place_pos = None
            pending_place_player = None

        # Normal commands
        if cmd_type == 'place_ring':
            moves.append({
                'move_type': 'PLACE_RING',
                'player': player,
                'position': parsed['position'],
            })
        elif cmd_type == 'move_ring':
            moves.append({
                'move_type': 'MOVE_RING',
                'player': player,
                'source': parsed['source'],
                'destination': parsed['destination'],
            })
        elif cmd_type == 'remove_markers':
            moves.append({
                'move_type': 'REMOVE_MARKERS',
                'player': player,
                'markers': parsed['markers'],
            })
        elif cmd_type == 'remove_ring':
            moves.append({
                'move_type': 'REMOVE_RING',
                'player': player,
                'position': parsed['position'],
            })

    return moves


def _parse_command(cmd: str, player: str) -> Optional[Dict]:
    """Parse a single Boardspace command string.

    Returns a dict with a '_type' key indicating the command type,
    or None for commands that should be skipped.
    """
    # Normalize whitespace
    cmd = ' '.join(cmd.split())

    # Skip non-move commands
    if cmd.startswith('Done') or cmd.startswith('Start'):
        return None
    if cmd.startswith('pick '):
        return None
    if cmd.startswith('Resign'):
        return None

    # place wr/br {col} {row} — ring placement
    m = re.match(r'place\s+(wr|br)\s+([A-K])\s+(\d+)', cmd)
    if m:
        pos = _make_position(m.group(2), m.group(3))
        if pos:
            return {'_type': 'place_ring', 'position': pos}
        return None

    # place w/b {col} {row} — marker placement (first step of move)
    m = re.match(r'place\s+(w|b)\s+([A-K])\s+(\d+)', cmd)
    if m:
        pos = _make_position(m.group(2), m.group(3))
        if pos:
            return {'_type': 'place_marker', 'position': pos}
        return None

    # drop board {col} {row} — ring drop (second step of move)
    m = re.match(r'drop\s+board\s+([A-K])\s+(\d+)', cmd)
    if m:
        pos = _make_position(m.group(1), m.group(2))
        if pos:
            return {'_type': 'drop_board', 'position': pos}
        return None

    # move {col1} {row1} {col2} {row2} — combined ring move
    m = re.match(r'move\s+([A-K])\s+(\d+)\s+([A-K])\s+(\d+)', cmd)
    if m:
        src = _make_position(m.group(1), m.group(2))
        dst = _make_position(m.group(3), m.group(4))
        if src and dst:
            return {'_type': 'move_ring', 'source': src, 'destination': dst}
        return None

    # remove w/b {col1} {row1} {col2} {row2} — marker removal (endpoints)
    m = re.match(r'remove\s+(w|b)\s+([A-K])\s+(\d+)\s+([A-K])\s+(\d+)', cmd)
    if m:
        start = Position(m.group(2), int(m.group(3)))
        end = Position(m.group(4), int(m.group(5)))
        markers = positions_on_line(start, end)
        if markers:
            return {
                '_type': 'remove_markers',
                'markers': [str(p) for p in markers],
            }
        return None

    # remove wr/br {col} {row} — ring removal
    m = re.match(r'remove\s+(wr|br)\s+([A-K])\s+(\d+)', cmd)
    if m:
        pos = _make_position(m.group(2), m.group(3))
        if pos:
            return {'_type': 'remove_ring', 'position': pos}
        return None

    # Unknown command — log and skip
    logger.debug(f"Skipping unknown Boardspace command: {cmd!r}")
    return None


def _make_position(col: str, row_str: str) -> Optional[str]:
    """Create a position string from column letter and row number.

    Returns the position as a string (e.g., "G11") or None if invalid.
    """
    try:
        row = int(row_str)
        pos = Position(col, row)
        if is_valid_position(pos):
            return str(pos)
        logger.warning(f"Invalid Boardspace position: {col} {row_str}")
        return None
    except (ValueError, TypeError):
        return None
