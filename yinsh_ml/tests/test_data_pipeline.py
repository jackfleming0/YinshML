"""Tests for the expert game data pipeline.

Covers:
- LG notation parsing (coordinate translation)
- CG notation parsing (coordinate translation + ring mapping)
- Game validation
- Converter integration
"""

import pytest
import json
import tempfile
from pathlib import Path

from yinsh_ml.game.constants import Position, is_valid_position, VALID_POSITIONS
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import Move, MoveType, GamePhase
from yinsh_ml.data.parsers.lg_notation import (
    parse_position, parse_game_record, _positions_between,
)
from yinsh_ml.data.parsers.cg_notation import (
    cg_to_position, position_to_cg, parse_cg_moves,
    get_mapping_stats, _RING_TO_POS,
)
from yinsh_ml.data.validator import GameValidator
from yinsh_ml.data.converter import GameConverter


# ============================================================
# LG Notation Tests
# ============================================================

class TestLGNotationParsing:
    """Test official GIPF notation parsing."""

    def test_parse_position_lowercase(self):
        pos = parse_position("e5")
        assert pos == Position("E", 5)

    def test_parse_position_uppercase(self):
        pos = parse_position("E5")
        assert pos == Position("E", 5)

    def test_parse_position_two_digit_row(self):
        pos = parse_position("g11")
        assert pos == Position("G", 11)

    def test_parse_position_corners(self):
        """Test parsing of corner positions."""
        assert parse_position("a2") == Position("A", 2)
        assert parse_position("a5") == Position("A", 5)
        assert parse_position("k7") == Position("K", 7)
        assert parse_position("k10") == Position("K", 10)

    def test_parse_position_invalid(self):
        """Invalid positions should raise ValueError."""
        with pytest.raises(ValueError):
            parse_position("a1")  # A1 is not valid
        with pytest.raises(ValueError):
            parse_position("k6")  # K6 is not valid
        with pytest.raises(ValueError):
            parse_position("z5")  # Z is not a column

    def test_parse_ring_placement(self):
        """Ring placement: just a position."""
        moves = parse_game_record("1.e5")
        assert len(moves) == 1
        assert moves[0]['move_type'] == 'PLACE_RING'
        assert moves[0]['player'] == 'white'
        assert moves[0]['position'] == 'E5'

    def test_parse_ring_movement(self):
        """Ring movement: source-destination."""
        moves = parse_game_record("24.a2-h9")
        assert len(moves) == 1
        assert moves[0]['move_type'] == 'MOVE_RING'
        assert moves[0]['source'] == 'A2'
        assert moves[0]['destination'] == 'H9'

    def test_parse_ring_movement_with_removal(self):
        """Movement with row removal and ring removal."""
        moves = parse_game_record("24.e3-e8;xe5")
        assert len(moves) == 2
        assert moves[0]['move_type'] == 'MOVE_RING'
        assert moves[1]['move_type'] == 'REMOVE_RING'

    def test_parse_multiple_moves(self):
        """Multiple numbered moves."""
        moves = parse_game_record("1.e5 2.g7 3.d4")
        assert len(moves) == 3
        assert moves[0]['player'] == 'white'
        assert moves[1]['player'] == 'black'
        assert moves[2]['player'] == 'white'

    def test_positions_between_vertical(self):
        """Positions on a vertical line."""
        positions = _positions_between(
            Position("E", 3), Position("E", 7)
        )
        assert len(positions) == 5
        assert positions[0] == Position("E", 3)
        assert positions[-1] == Position("E", 7)

    def test_positions_between_horizontal(self):
        """Positions on a horizontal line."""
        positions = _positions_between(
            Position("C", 5), Position("G", 5)
        )
        assert len(positions) == 5
        assert positions[0] == Position("C", 5)
        assert positions[-1] == Position("G", 5)

    def test_positions_between_diagonal(self):
        """Positions on a diagonal line."""
        positions = _positions_between(
            Position("C", 3), Position("F", 6)
        )
        assert len(positions) == 4
        for p in positions:
            assert is_valid_position(p)


# ============================================================
# CG Notation Tests
# ============================================================

class TestCGCoordinateMapping:
    """Test CodinGame concentric ring coordinate mapping."""

    def test_mapping_total_positions(self):
        """Must map exactly 85 positions."""
        stats = get_mapping_stats()
        assert stats['total_positions'] == 85
        assert stats['valid'] is True

    def test_ring_counts(self):
        """Ring sizes: 1, 6, 12, 18, 24, 24."""
        stats = get_mapping_stats()
        expected = {0: 1, 1: 6, 2: 12, 3: 18, 4: 24, 5: 24}
        assert stats['positions_per_ring'] == expected

    def test_center_is_f6(self):
        """Ring 0, position 0 should map to F6 (board center)."""
        pos = cg_to_position(0, 0)
        assert pos == Position("F", 6)

    def test_all_mapped_positions_are_valid(self):
        """Every mapped position must be a valid YINSH board position."""
        for (ring, idx), pos in _RING_TO_POS.items():
            assert is_valid_position(pos), \
                f"({ring}, {idx}) mapped to invalid position {pos}"

    def test_all_valid_positions_are_mapped(self):
        """Every valid board position must appear in the mapping."""
        mapped_positions = {str(pos) for pos in _RING_TO_POS.values()}
        for col, rows in VALID_POSITIONS.items():
            for row in rows:
                pos_str = f"{col}{row}"
                assert pos_str in mapped_positions, \
                    f"Position {pos_str} not in CG mapping"

    def test_roundtrip(self):
        """Position → CG → Position roundtrip."""
        for (ring, idx), pos in _RING_TO_POS.items():
            r, i = position_to_cg(pos)
            assert (r, i) == (ring, idx), \
                f"Roundtrip failed for ({ring}, {idx}) → {pos}"

    def test_ring1_neighbors_of_center(self):
        """Ring 1 positions should all be adjacent to F6."""
        center_col = ord('F') - ord('A')
        center_row = 6
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)]

        ring1_positions = {str(pos) for (r, _), pos in _RING_TO_POS.items()
                          if r == 1}

        for dc, dr in dirs:
            col = chr(ord('A') + center_col + dc)
            row = center_row + dr
            pos_str = f"{col}{row}"
            assert pos_str in ring1_positions, \
                f"Expected {pos_str} in ring 1"

    def test_invalid_cg_coordinates_raise(self):
        """Invalid CG coordinates should raise ValueError."""
        with pytest.raises(ValueError):
            cg_to_position(0, 1)  # Ring 0 only has position 0
        with pytest.raises(ValueError):
            cg_to_position(6, 0)  # Ring 6 doesn't exist


class TestCGMoveParsing:
    """Test CodinGame move string parsing."""

    def test_parse_ring_placement(self):
        moves = parse_cg_moves("P 0 0", "white")
        assert len(moves) == 1
        assert moves[0]['move_type'] == 'PLACE_RING'
        assert moves[0]['position'] == 'F6'

    def test_parse_ring_movement(self):
        moves = parse_cg_moves("S 0 0 M 1 0", "white")
        assert len(moves) == 1
        assert moves[0]['move_type'] == 'MOVE_RING'
        assert moves[0]['source'] == 'F6'

    def test_parse_move_with_removal(self):
        """S M RS RE X sequence."""
        moves = parse_cg_moves("S 1 0 M 2 0 RS 1 0 RE 2 0 X 3 0", "black")
        assert len(moves) == 3
        assert moves[0]['move_type'] == 'MOVE_RING'
        assert moves[1]['move_type'] == 'REMOVE_MARKERS'
        assert moves[2]['move_type'] == 'REMOVE_RING'

    def test_parse_ring_placement_ring1(self):
        """Ring 1 placements should map to F6's neighbors."""
        moves = parse_cg_moves("P 1 0", "white")
        pos = Position.from_string(moves[0]['position'])
        assert is_valid_position(pos)


# ============================================================
# Validator Tests
# ============================================================

class TestGameValidator:
    """Test game validation by replaying through GameState."""

    def _make_simple_game(self):
        """Create a valid game with just ring placements."""
        # Both players place 5 rings each (10 moves)
        gs = GameState()
        moves = []
        valid = gs.get_valid_moves()

        for i in range(10):
            move = valid[i % len(valid)]
            player = 'white' if gs.current_player.name == 'WHITE' else 'black'

            moves.append({
                'move_type': 'PLACE_RING',
                'player': player,
                'position': str(move.source),
            })

            gs.make_move(move)
            valid = gs.get_valid_moves()

        return {
            'source': 'test',
            'game_id': 'test_001',
            'players': {
                'white': {'name': 'test_w', 'rating': 1500},
                'black': {'name': 'test_b', 'rating': 1500},
            },
            'result': 'draw',  # Won't match terminal state, but that's OK for non-strict
            'moves': moves,
        }

    def test_validate_ring_placement_phase(self):
        """Validate a game through ring placement phase."""
        validator = GameValidator(strict=False)
        game = self._make_simple_game()
        result = validator.validate_game(game)
        assert result.valid, f"Validation failed: {result.error_message}"
        assert result.total_moves == 10

    def test_validate_invalid_move(self):
        """A game with an illegal move should fail validation."""
        validator = GameValidator()
        game = {
            'source': 'test',
            'game_id': 'test_bad',
            'result': 'white',
            'moves': [
                # Place ring at valid position
                {'move_type': 'PLACE_RING', 'player': 'white', 'position': 'E5'},
                # Try to move a ring during placement phase (illegal)
                {'move_type': 'MOVE_RING', 'player': 'black',
                 'source': 'E5', 'destination': 'E8'},
            ],
        }
        result = validator.validate_game(game)
        assert not result.valid
        assert result.error_move == 1

    def test_validate_no_moves(self):
        validator = GameValidator()
        result = validator.validate_game({
            'game_id': 'empty', 'result': 'white', 'moves': []
        })
        assert not result.valid

    def test_validate_bad_result(self):
        validator = GameValidator()
        result = validator.validate_game({
            'game_id': 'bad', 'result': 'invalid',
            'moves': [{'move_type': 'PLACE_RING', 'player': 'white',
                       'position': 'E5'}],
        })
        assert not result.valid


# ============================================================
# Converter Tests
# ============================================================

class TestGameConverter:
    """Test conversion of games to training pairs."""

    def test_convert_ring_placement(self):
        """Converting ring placements should produce training pairs."""
        converter = GameConverter()
        game = {
            'game_id': 'test',
            'result': 'white',
            'moves': [
                {'move_type': 'PLACE_RING', 'player': 'white', 'position': 'E5'},
                {'move_type': 'PLACE_RING', 'player': 'black', 'position': 'G7'},
            ],
        }
        pairs = converter.convert_game(game)
        assert len(pairs) == 2

        # First pair should have state shape (6, 11, 11)
        assert pairs[0]['state'].shape == (6, 11, 11)
        # Policy should be one-hot
        assert pairs[0]['policy'].sum() == pytest.approx(1.0, abs=0.01)
        # Value should be +1 for white (winner) playing first
        assert pairs[0]['value'] == 1.0
        # Value should be -1 for black (loser)
        assert pairs[1]['value'] == -1.0

    def test_save_and_load_roundtrip(self):
        """Save and load training data."""
        converter = GameConverter()
        game = {
            'game_id': 'roundtrip_test',
            'result': 'black',
            'moves': [
                {'move_type': 'PLACE_RING', 'player': 'white', 'position': 'F6'},
                {'move_type': 'PLACE_RING', 'player': 'black', 'position': 'E5'},
            ],
        }
        pairs = converter.convert_game(game)
        assert len(pairs) > 0

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            converter.save_training_data(pairs, f.name)
            states, policies, values = GameConverter.load_training_data(f.name)

        assert states.shape[0] == len(pairs)
        assert states.shape[1:] == (6, 11, 11)


# ============================================================
# Integration Tests
# ============================================================

class TestPipelineIntegration:
    """End-to-end pipeline tests."""

    def test_lg_notation_to_training_pairs(self):
        """Parse LG notation → standardized dict → training pairs."""
        move_text = "1.e5 2.g7 3.d4 4.h8 5.f3 6.c5 7.b3 8.i6 9.g4 10.e8"
        moves = parse_game_record(move_text)
        assert len(moves) == 10

        game = {
            'source': 'gipf_notation',
            'game_id': 'integration_test',
            'players': {
                'white': {'name': 'player1', 'rating': 1800},
                'black': {'name': 'player2', 'rating': 1700},
            },
            'result': 'white',
            'moves': moves,
        }

        # Validate
        validator = GameValidator(strict=False)
        result = validator.validate_game(game)
        assert result.valid, f"Validation failed: {result.error_message}"

        # Convert
        converter = GameConverter()
        pairs = converter.convert_game(game)
        assert len(pairs) == 10

    def test_cg_notation_to_training_pairs(self):
        """Parse CG notation → standardized dict → training pairs."""
        # Simulate a CG game: 10 ring placements
        cg_moves_white = [
            "P 0 0",  # F6
            "P 2 0",  # ring 2, pos 0
            "P 2 6",
            "P 3 0",
            "P 3 6",
        ]
        cg_moves_black = [
            "P 1 0",  # ring 1, pos 0
            "P 2 3",
            "P 2 9",
            "P 3 3",
            "P 3 9",
        ]

        all_moves = []
        for w, b in zip(cg_moves_white, cg_moves_black):
            all_moves.extend(parse_cg_moves(w, 'white'))
            all_moves.extend(parse_cg_moves(b, 'black'))

        game = {
            'source': 'codingame',
            'game_id': 'cg_integration',
            'players': {
                'white': {'name': 'bot_a', 'rating': 30},
                'black': {'name': 'bot_b', 'rating': 25},
            },
            'result': 'white',
            'moves': all_moves,
        }

        # Validate
        validator = GameValidator(strict=False)
        result = validator.validate_game(game)
        assert result.valid, f"Validation failed at move {result.error_move}: {result.error_message}"
