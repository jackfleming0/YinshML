"""Tests for the Boardspace SGF parser."""

import pytest
from pathlib import Path

from yinsh_ml.game.constants import Position, is_valid_position
from yinsh_ml.data.parsers.boardspace_sgf import (
    parse_boardspace_sgf, _parse_command, _parse_header, _parse_result,
)
from yinsh_ml.data.validator import GameValidator


# ============================================================
# Minimal SGF templates for testing
# ============================================================

HEADER_TEMPLATE = """\
(;GM[24]VV[2]SU[Yinsh]
P0[id "{white}"]P1[id "{black}"]
RE[{result}]GN[test-game]
"""

def _make_sgf(white="alice", black="bob", result="Game won by alice", moves=""):
    return HEADER_TEMPLATE.format(white=white, black=black, result=result) + moves + ")"


# ============================================================
# Command Parsing Tests
# ============================================================

class TestCommandParsing:
    """Test individual Boardspace command parsing."""

    def test_place_white_ring(self):
        result = _parse_command("place wr G 11", "white")
        assert result is not None
        assert result['_type'] == 'place_ring'
        assert result['position'] == 'G11'

    def test_place_black_ring(self):
        result = _parse_command("place br E 5", "black")
        assert result is not None
        assert result['_type'] == 'place_ring'
        assert result['position'] == 'E5'

    def test_place_marker(self):
        result = _parse_command("place w K 8", "white")
        assert result is not None
        assert result['_type'] == 'place_marker'
        assert result['position'] == 'K8'

    def test_drop_board(self):
        result = _parse_command("drop board K 9", "white")
        assert result is not None
        assert result['_type'] == 'drop_board'
        assert result['position'] == 'K9'

    def test_combined_move(self):
        result = _parse_command("move G 9 F 9", "black")
        assert result is not None
        assert result['_type'] == 'move_ring'
        assert result['source'] == 'G9'
        assert result['destination'] == 'F9'

    def test_remove_markers_vertical(self):
        result = _parse_command("remove b C 2 C 6", "black")
        assert result is not None
        assert result['_type'] == 'remove_markers'
        assert len(result['markers']) == 5
        assert result['markers'][0] == 'C2'
        assert result['markers'][-1] == 'C6'

    def test_remove_markers_horizontal(self):
        result = _parse_command("remove w D 5 H 5", "white")
        assert result is not None
        assert result['_type'] == 'remove_markers'
        assert len(result['markers']) == 5

    def test_remove_markers_diagonal(self):
        result = _parse_command("remove b D 3 H 7", "black")
        assert result is not None
        assert result['_type'] == 'remove_markers'
        assert len(result['markers']) == 5
        assert result['markers'][0] == 'D3'
        assert result['markers'][-1] == 'H7'

    def test_remove_white_ring(self):
        result = _parse_command("remove wr F 8", "white")
        assert result is not None
        assert result['_type'] == 'remove_ring'
        assert result['position'] == 'F8'

    def test_remove_black_ring(self):
        result = _parse_command("remove br H 11", "black")
        assert result is not None
        assert result['_type'] == 'remove_ring'
        assert result['position'] == 'H11'

    def test_skip_done(self):
        assert _parse_command("Done ", "white") is None

    def test_skip_start(self):
        assert _parse_command("Start White", "white") is None

    def test_skip_pick(self):
        assert _parse_command("pick wr", "white") is None

    def test_skip_resign(self):
        assert _parse_command("Resign ", "black") is None

    def test_extra_whitespace(self):
        """Handle Boardspace's double-space formatting."""
        result = _parse_command("place wr  G 11", "white")
        assert result is not None
        assert result['position'] == 'G11'


# ============================================================
# Header Parsing Tests
# ============================================================

class TestHeaderParsing:
    def test_parse_result_white_wins(self):
        assert _parse_result("Game won by alice", "alice", "bob") == 'white'

    def test_parse_result_black_wins(self):
        assert _parse_result("Game won by bob", "alice", "bob") == 'black'

    def test_parse_result_draw(self):
        assert _parse_result("The game is a draw", "alice", "bob") == 'draw'

    def test_parse_result_unknown(self):
        assert _parse_result("", "alice", "bob") == 'unknown'

    def test_parse_header_yinsh(self):
        sgf = _make_sgf()
        header = _parse_header(sgf)
        assert header is not None
        assert header['white_name'] == 'alice'
        assert header['black_name'] == 'bob'
        assert header['result'] == 'white'


# ============================================================
# Full SGF Parsing Tests
# ============================================================

class TestFullSGFParsing:

    def test_ring_placement_only(self):
        """Parse a game with only ring placements."""
        moves_sgf = """
; P0[0 Start White]TM[100]
; P0[1 place wr  E 5]TM[200]
; P0[2 Done ]TM[300]
; P1[3 place br  G 7]TM[400]
; P1[4 Done ]TM[500]
"""
        game = parse_boardspace_sgf(_make_sgf(moves=moves_sgf))
        assert game is not None
        assert len(game['moves']) == 2
        assert game['moves'][0]['move_type'] == 'PLACE_RING'
        assert game['moves'][0]['player'] == 'white'
        assert game['moves'][0]['position'] == 'E5'
        assert game['moves'][1]['move_type'] == 'PLACE_RING'
        assert game['moves'][1]['player'] == 'black'
        assert game['moves'][1]['position'] == 'G7'

    def test_place_drop_move(self):
        """Parse the two-step place+drop ring move format."""
        moves_sgf = """
; P0[21 place w  K 8]TM[19220]
; P0[22 drop board  K 9]TM[20057]
; P0[23 Done ]TM[21139]
"""
        game = parse_boardspace_sgf(_make_sgf(moves=moves_sgf))
        assert game is not None
        ring_moves = [m for m in game['moves'] if m['move_type'] == 'MOVE_RING']
        assert len(ring_moves) == 1
        assert ring_moves[0]['source'] == 'K8'
        assert ring_moves[0]['destination'] == 'K9'

    def test_combined_move_format(self):
        """Parse the single-line combined move format."""
        moves_sgf = """
; P1[24 move G 9 F 9]TM[8343]
; P1[25 Done ]TM[10625]
"""
        game = parse_boardspace_sgf(_make_sgf(moves=moves_sgf))
        ring_moves = [m for m in game['moves'] if m['move_type'] == 'MOVE_RING']
        assert len(ring_moves) == 1
        assert ring_moves[0]['source'] == 'G9'
        assert ring_moves[0]['destination'] == 'F9'
        assert ring_moves[0]['player'] == 'black'

    def test_row_capture_sequence(self):
        """Parse marker removal + ring removal."""
        moves_sgf = """
; P1[91 remove b C 2 C 6]TM[76592]
; P1[92 Done ]TM[77063]
; P1[93 remove br I 6]TM[77760]
; P1[94 Done ]TM[79769]
"""
        game = parse_boardspace_sgf(_make_sgf(moves=moves_sgf))
        assert len(game['moves']) == 2
        assert game['moves'][0]['move_type'] == 'REMOVE_MARKERS'
        assert game['moves'][0]['player'] == 'black'
        assert len(game['moves'][0]['markers']) == 5
        assert game['moves'][1]['move_type'] == 'REMOVE_RING'
        assert game['moves'][1]['position'] == 'I6'

    def test_resign(self):
        """Resignation sets correct winner."""
        moves_sgf = """
; P0[1 place wr  E 5]TM[200]
; P0[2 Done ]TM[300]
; P1[3 Resign ]TM[5000]
; P1[4 Done ]TM[5100]
"""
        game = parse_boardspace_sgf(_make_sgf(
            result="Game won by alice", moves=moves_sgf
        ))
        assert game['result'] == 'white'

    def test_draw_result(self):
        game = parse_boardspace_sgf(_make_sgf(result="The game is a draw"))
        assert game['result'] == 'draw'

    def test_full_ring_placement_validates(self):
        """A complete ring placement phase should pass validation."""
        # Place 5 rings for white (P0) and 5 for black (P1), alternating
        positions_white = ['E5', 'G7', 'D4', 'H8', 'F3']
        positions_black = ['G4', 'C5', 'I6', 'B3', 'E8']

        moves_sgf = "; P0[0 Start White]TM[100]\n"
        seq = 1
        for w_pos, b_pos in zip(positions_white, positions_black):
            w_col, w_row = w_pos[0], w_pos[1:]
            b_col, b_row = b_pos[0], b_pos[1:]
            moves_sgf += f"; P0[{seq} place wr  {w_col} {w_row}]TM[{seq*100}]\n"
            seq += 1
            moves_sgf += f"; P0[{seq} Done ]TM[{seq*100}]\n"
            seq += 1
            moves_sgf += f"; P1[{seq} place br  {b_col} {b_row}]TM[{seq*100}]\n"
            seq += 1
            moves_sgf += f"; P1[{seq} Done ]TM[{seq*100}]\n"
            seq += 1

        game = parse_boardspace_sgf(_make_sgf(moves=moves_sgf))
        assert game is not None
        assert len(game['moves']) == 10

        # Validate through GameState
        validator = GameValidator(strict=False)
        result = validator.validate_game(game)
        assert result.valid, f"Validation failed: {result.error_message}"

    def test_non_yinsh_game_returns_none(self):
        """SGF for a different game type should return None."""
        sgf = "(;GM[1]SU[Chess]P0[id \"alice\"]P1[id \"bob\"]RE[draw])"
        assert parse_boardspace_sgf(sgf) is None


class TestRealSGFFiles:
    """Test against real SGF files downloaded from Boardspace."""

    FIXTURES_DIR = Path(__file__).parent / "fixtures" / "boardspace"

    @pytest.fixture
    def sgf_files(self):
        path = Path(self.FIXTURES_DIR)
        if not path.exists():
            pytest.skip("Boardspace fixtures not available")
        files = sorted(path.glob("*.sgf"))
        if not files:
            pytest.skip("No SGF files in fixtures")
        return files

    def test_all_real_sgfs_parse(self, sgf_files):
        """All real SGF files should parse without errors."""
        for sgf_path in sgf_files:
            with open(sgf_path) as f:
                sgf_text = f.read()
            game = parse_boardspace_sgf(sgf_text)
            assert game is not None, f"Failed to parse {sgf_path.name}"
            assert len(game['moves']) > 0, f"No moves in {sgf_path.name}"
            assert game['result'] in ('white', 'black', 'draw'), \
                f"Bad result '{game['result']}' in {sgf_path.name}"

    def test_all_real_sgfs_validate(self, sgf_files):
        """All real SGF files should pass game validation."""
        validator = GameValidator(strict=False)
        for sgf_path in sgf_files:
            with open(sgf_path) as f:
                sgf_text = f.read()
            game = parse_boardspace_sgf(sgf_text)
            if game is None:
                continue
            result = validator.validate_game(game)
            assert result.valid, \
                f"{sgf_path.name}: validation failed at move {result.error_move}: {result.error_message}"
