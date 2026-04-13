"""Unit tests for EnhancedStateEncoder.

Tests verify:
1. Correct output shape (15, 11, 11)
2. Channel contents for each of the 15 channels
3. Side-to-move normalization
4. Consistency with basic encoder for shared channels
5. Edge cases and error handling
"""

import pytest
import numpy as np
from yinsh_ml.utils.enhanced_encoding import EnhancedStateEncoder, compare_encodings
from yinsh_ml.utils.encoding import StateEncoder
from yinsh_ml.game.game_state import GameState, GamePhase
from yinsh_ml.game.constants import Player, PieceType, Position


class TestEnhancedStateEncoderBasics:
    """Basic functionality tests."""

    @pytest.fixture
    def encoder(self):
        return EnhancedStateEncoder(enable_stats=True)

    @pytest.fixture
    def basic_encoder(self):
        return StateEncoder()

    @pytest.fixture
    def initial_state(self):
        """Create a fresh game state."""
        return GameState()

    @pytest.fixture
    def midgame_state(self):
        """Create a mid-game state with pieces on the board."""
        state = GameState()
        # Place some rings and markers for a realistic state
        # Place 5 rings for each player
        white_ring_positions = ['E5', 'D4', 'F6', 'C3', 'G7']
        black_ring_positions = ['E7', 'D6', 'F4', 'C5', 'G5']

        for pos_str in white_ring_positions:
            pos = Position.from_string(pos_str)
            state.board.place_piece(pos, PieceType.WHITE_RING)

        for pos_str in black_ring_positions:
            pos = Position.from_string(pos_str)
            state.board.place_piece(pos, PieceType.BLACK_RING)

        # Add some markers
        state.board.place_piece(Position('E', 6), PieceType.WHITE_MARKER)
        state.board.place_piece(Position('D', 5), PieceType.BLACK_MARKER)

        state.phase = GamePhase.MAIN_GAME
        state.current_player = Player.WHITE
        return state

    def test_output_shape(self, encoder, initial_state):
        """Test that encoder produces correct output shape."""
        encoded = encoder.encode_state(initial_state)
        assert encoded.shape == (15, 11, 11), f"Expected (15, 11, 11), got {encoded.shape}"

    def test_output_dtype(self, encoder, initial_state):
        """Test that encoder produces float32 output."""
        encoded = encoder.encode_state(initial_state)
        assert encoded.dtype == np.float32, f"Expected float32, got {encoded.dtype}"

    def test_channel_count_constant(self, encoder):
        """Test that NUM_CHANNELS constant is correct."""
        assert EnhancedStateEncoder.NUM_CHANNELS == 15

    def test_initial_state_encoding(self, encoder, initial_state):
        """Test encoding of initial (empty) game state."""
        encoded = encoder.encode_state(initial_state)

        # No pieces on board initially
        assert np.sum(encoded[0:4]) == 0, "Initial state should have no pieces"

        # Center distance should be non-zero (precomputed)
        assert np.sum(encoded[9]) > 0, "Center distance channel should be populated"

    def test_midgame_encoding(self, encoder, midgame_state):
        """Test encoding of midgame state with pieces."""
        encoded = encoder.encode_state(midgame_state)

        # Should have pieces in ring channels
        assert np.sum(encoded[0]) > 0, "Current rings channel should have pieces"
        assert np.sum(encoded[1]) > 0, "Opponent rings channel should have pieces"

        # Should have pieces in marker channels
        assert np.sum(encoded[2]) > 0, "Current markers channel should have pieces"
        assert np.sum(encoded[3]) > 0, "Opponent markers channel should have pieces"


class TestSideToMoveNormalization:
    """Test side-to-move normalization (channels flip based on current player)."""

    @pytest.fixture
    def encoder(self):
        return EnhancedStateEncoder()

    def test_white_perspective(self, encoder):
        """Test encoding from white's perspective."""
        state = GameState()
        state.current_player = Player.WHITE

        # Place white ring at E5
        state.board.place_piece(Position('E', 5), PieceType.WHITE_RING)
        # Place black ring at F6
        state.board.place_piece(Position('F', 6), PieceType.BLACK_RING)

        encoded = encoder.encode_state(state)

        # White ring should be in "current" channel (0)
        assert encoded[0, 4, 4] == 1.0, "White ring should be in channel 0 (current)"
        # Black ring should be in "opponent" channel (1)
        assert encoded[1, 5, 5] == 1.0, "Black ring should be in channel 1 (opponent)"

    def test_black_perspective(self, encoder):
        """Test encoding from black's perspective."""
        state = GameState()
        state.current_player = Player.BLACK

        # Place white ring at E5
        state.board.place_piece(Position('E', 5), PieceType.WHITE_RING)
        # Place black ring at F6
        state.board.place_piece(Position('F', 6), PieceType.BLACK_RING)

        encoded = encoder.encode_state(state)

        # Black ring should be in "current" channel (0)
        assert encoded[0, 5, 5] == 1.0, "Black ring should be in channel 0 (current)"
        # White ring should be in "opponent" channel (1)
        assert encoded[1, 4, 4] == 1.0, "White ring should be in channel 1 (opponent)"

    def test_perspective_consistency(self, encoder):
        """Test that encoding is consistent with perspective changes."""
        state = GameState()

        # Place symmetric pieces
        state.board.place_piece(Position('E', 5), PieceType.WHITE_RING)
        state.board.place_piece(Position('F', 6), PieceType.BLACK_RING)

        # Encode from white's perspective
        state.current_player = Player.WHITE
        white_encoded = encoder.encode_state(state)

        # Encode from black's perspective
        state.current_player = Player.BLACK
        black_encoded = encoder.encode_state(state)

        # Current/opponent channels should swap
        assert np.array_equal(white_encoded[0], black_encoded[1]), \
            "White's current should equal Black's opponent"
        assert np.array_equal(white_encoded[1], black_encoded[0]), \
            "White's opponent should equal Black's current"


class TestRowThreatsChannel:
    """Test row threat detection (channels 4-5)."""

    @pytest.fixture
    def encoder(self):
        return EnhancedStateEncoder(enable_stats=True)

    def test_four_in_row_creates_threat(self, encoder):
        """Test that 4 markers in a row create threat positions."""
        state = GameState()
        state.current_player = Player.WHITE
        state.phase = GamePhase.MAIN_GAME

        # Place 4 white markers in a row (vertical)
        for row in range(3, 7):  # E3, E4, E5, E6
            state.board.place_piece(Position('E', row), PieceType.WHITE_MARKER)

        encoded = encoder.encode_state(state)
        stats = encoder.get_last_stats()

        # Should have row threats (E2 and E7 would complete the row)
        assert stats.row_threats_found > 0, "Should find row threats for 4-in-a-row"

        # Check specific threat positions
        threat_channel = encoded[4]  # Current player threats
        # E2 is at row_idx=1, col_idx=4
        # E7 is at row_idx=6, col_idx=4
        has_threat = threat_channel[1, 4] == 1.0 or threat_channel[6, 4] == 1.0
        assert has_threat, "Threat should be at position that completes the row"

    def test_no_threats_with_few_markers(self, encoder):
        """Test that fewer than 4 markers don't create threats."""
        state = GameState()
        state.current_player = Player.WHITE
        state.phase = GamePhase.MAIN_GAME

        # Place only 2 markers
        state.board.place_piece(Position('E', 5), PieceType.WHITE_MARKER)
        state.board.place_piece(Position('E', 6), PieceType.WHITE_MARKER)

        encoded = encoder.encode_state(state)
        stats = encoder.get_last_stats()

        assert stats.row_threats_found == 0, "Should not find threats with only 2 markers"


class TestPartialRowsChannel:
    """Test partial row detection (channels 6-7)."""

    @pytest.fixture
    def encoder(self):
        return EnhancedStateEncoder(enable_stats=True)

    def test_three_in_row_detected(self, encoder):
        """Test that 3 markers in a row are detected as partial row."""
        state = GameState()
        state.current_player = Player.WHITE
        state.phase = GamePhase.MAIN_GAME

        # Place 3 white markers in a row
        for row in range(4, 7):  # E4, E5, E6
            state.board.place_piece(Position('E', row), PieceType.WHITE_MARKER)

        encoded = encoder.encode_state(state)
        stats = encoder.get_last_stats()

        assert stats.partial_rows_found > 0, "Should find partial rows for 3-in-a-row"

        # Check that the markers are marked in partial row channel
        partial_channel = encoded[6]  # Current player partial rows
        assert partial_channel[3, 4] == 1.0, "E4 should be in partial row"
        assert partial_channel[4, 4] == 1.0, "E5 should be in partial row"
        assert partial_channel[5, 4] == 1.0, "E6 should be in partial row"


class TestRingMobilityChannel:
    """Test ring mobility encoding (channel 8)."""

    @pytest.fixture
    def encoder(self):
        return EnhancedStateEncoder(enable_stats=True)

    def test_ring_mobility_nonzero(self, encoder):
        """Test that rings have mobility values."""
        state = GameState()
        state.phase = GamePhase.MAIN_GAME

        # Place a ring in a central position
        state.board.place_piece(Position('E', 5), PieceType.WHITE_RING)

        encoded = encoder.encode_state(state)
        stats = encoder.get_last_stats()

        # Ring should have mobility
        assert stats.ring_mobility_sum > 0, "Ring should have some mobility"
        assert encoded[8, 4, 4] > 0, "Mobility channel should be non-zero at ring position"

    def test_ring_mobility_blocked(self, encoder):
        """Test that blocked rings have lower mobility."""
        state = GameState()
        state.phase = GamePhase.MAIN_GAME

        # Place a ring
        state.board.place_piece(Position('E', 5), PieceType.WHITE_RING)

        # Encode without blockers
        encoded_free = encoder.encode_state(state)
        mobility_free = encoded_free[8, 4, 4]

        # Add blocking rings around it
        state.board.place_piece(Position('D', 5), PieceType.BLACK_RING)
        state.board.place_piece(Position('F', 5), PieceType.BLACK_RING)
        state.board.place_piece(Position('E', 4), PieceType.BLACK_RING)
        state.board.place_piece(Position('E', 6), PieceType.BLACK_RING)

        # Encode with blockers
        encoded_blocked = encoder.encode_state(state)
        mobility_blocked = encoded_blocked[8, 4, 4]

        # Mobility should be lower when blocked
        assert mobility_blocked <= mobility_free, \
            "Blocked ring should have equal or lower mobility"


class TestCenterDistanceChannel:
    """Test center distance heatmap (channel 9)."""

    @pytest.fixture
    def encoder(self):
        return EnhancedStateEncoder()

    def test_center_has_highest_value(self, encoder):
        """Test that board center has highest distance value."""
        state = GameState()
        encoded = encoder.encode_state(state)

        center_channel = encoded[9]

        # Center of YINSH board (approximately F6)
        center_value = center_channel[5, 5]  # F6

        # Should be close to 1.0 at center
        assert center_value > 0.8, f"Center value should be high, got {center_value}"

    def test_edges_have_lower_values(self, encoder):
        """Test that edges have lower distance values than center."""
        state = GameState()
        encoded = encoder.encode_state(state)

        center_channel = encoded[9]
        center_value = center_channel[5, 5]

        # Check some edge positions
        # B2 is at row_idx=1, col_idx=1
        if center_channel[1, 1] > 0:  # If valid position
            edge_value = center_channel[1, 1]
            assert edge_value < center_value, "Edge should have lower value than center"

    def test_center_distance_is_static(self, encoder):
        """Test that center distance doesn't change with game state."""
        state1 = GameState()
        state2 = GameState()

        # Add pieces to state2
        state2.board.place_piece(Position('E', 5), PieceType.WHITE_RING)

        encoded1 = encoder.encode_state(state1)
        encoded2 = encoder.encode_state(state2)

        assert np.array_equal(encoded1[9], encoded2[9]), \
            "Center distance channel should be identical regardless of pieces"


class TestRingInfluenceChannel:
    """Test ring influence encoding (channel 10)."""

    @pytest.fixture
    def encoder(self):
        return EnhancedStateEncoder(enable_stats=True)

    def test_ring_influence_coverage(self, encoder):
        """Test that ring influence marks reachable cells."""
        state = GameState()
        state.current_player = Player.WHITE
        state.phase = GamePhase.MAIN_GAME

        # Place a ring in a central position
        state.board.place_piece(Position('E', 5), PieceType.WHITE_RING)

        encoded = encoder.encode_state(state)
        stats = encoder.get_last_stats()

        # Should have some influence coverage
        assert stats.ring_influence_coverage > 0, "Ring should have influence"

        # Influence channel should have non-zero cells
        influence_channel = encoded[10]
        assert np.sum(influence_channel) > 0, "Influence channel should have marked cells"


class TestGamePhaseChannel:
    """Test game phase encoding (channel 12)."""

    @pytest.fixture
    def encoder(self):
        return EnhancedStateEncoder()

    def test_ring_placement_phase(self, encoder):
        """Test encoding during ring placement phase."""
        state = GameState()
        state.phase = GamePhase.RING_PLACEMENT

        encoded = encoder.encode_state(state)
        phase_value = encoded[12, 0, 0]

        # RING_PLACEMENT = 0, GamePhase has 5 values (0-4), normalized = 0/4 = 0
        assert phase_value == 0.0, f"Ring placement phase should be 0, got {phase_value}"

    def test_main_game_phase(self, encoder):
        """Test encoding during main game phase."""
        state = GameState()
        state.phase = GamePhase.MAIN_GAME

        encoded = encoder.encode_state(state)
        phase_value = encoded[12, 0, 0]

        # MAIN_GAME = 1, GamePhase has 5 values (0-4), normalized = 1/4 = 0.25
        assert phase_value == 0.25, f"Main game phase should be 0.25, got {phase_value}"


class TestTurnNumberChannel:
    """Test turn number encoding (channel 13)."""

    @pytest.fixture
    def encoder(self):
        return EnhancedStateEncoder()

    def test_turn_zero(self, encoder):
        """Test encoding at turn 0."""
        state = GameState()
        state.move_count = 0

        encoded = encoder.encode_state(state)
        turn_value = encoded[13, 0, 0]

        assert turn_value == 0.0, f"Turn 0 should encode to 0, got {turn_value}"

    def test_turn_fifty(self, encoder):
        """Test encoding at turn 50."""
        state = GameState()
        state.move_count = 50

        encoded = encoder.encode_state(state)
        turn_value = encoded[13, 0, 0]

        assert turn_value == 0.5, f"Turn 50 should encode to 0.5, got {turn_value}"

    def test_turn_capped_at_hundred(self, encoder):
        """Test that turn number is capped at 100."""
        state = GameState()
        state.move_count = 150

        encoded = encoder.encode_state(state)
        turn_value = encoded[13, 0, 0]

        assert turn_value == 1.0, f"Turn 150 should cap at 1.0, got {turn_value}"


class TestScoreDifferentialChannel:
    """Test score differential encoding (channel 14)."""

    @pytest.fixture
    def encoder(self):
        return EnhancedStateEncoder()

    def test_even_score(self, encoder):
        """Test encoding with even score."""
        state = GameState()
        state.white_score = 1
        state.black_score = 1
        state.current_player = Player.WHITE

        encoded = encoder.encode_state(state)
        score_diff = encoded[14, 0, 0]

        assert score_diff == 0.0, f"Even score should encode to 0, got {score_diff}"

    def test_winning_score(self, encoder):
        """Test encoding when current player is winning."""
        state = GameState()
        state.white_score = 2
        state.black_score = 0
        state.current_player = Player.WHITE

        encoded = encoder.encode_state(state)
        score_diff = encoded[14, 0, 0]

        # 2-0 = 2, normalized = 2/3 = 0.667
        assert abs(score_diff - 2/3) < 0.01, f"Winning by 2 should encode to ~0.667, got {score_diff}"

    def test_losing_score(self, encoder):
        """Test encoding when current player is losing."""
        state = GameState()
        state.white_score = 0
        state.black_score = 2
        state.current_player = Player.WHITE

        encoded = encoder.encode_state(state)
        score_diff = encoded[14, 0, 0]

        # 0-2 = -2, normalized = -2/3 = -0.667
        assert abs(score_diff + 2/3) < 0.01, f"Losing by 2 should encode to ~-0.667, got {score_diff}"


class TestBasicEncoderCompatibility:
    """Test compatibility with basic StateEncoder."""

    @pytest.fixture
    def encoder(self):
        return EnhancedStateEncoder()

    @pytest.fixture
    def basic_encoder(self):
        return StateEncoder()

    def test_first_four_channels_similar(self, encoder, basic_encoder):
        """Test that piece channels match basic encoder (conceptually)."""
        state = GameState()
        state.current_player = Player.WHITE
        state.phase = GamePhase.MAIN_GAME

        # Place some pieces
        state.board.place_piece(Position('E', 5), PieceType.WHITE_RING)
        state.board.place_piece(Position('F', 6), PieceType.BLACK_RING)
        state.board.place_piece(Position('D', 4), PieceType.WHITE_MARKER)
        state.board.place_piece(Position('G', 7), PieceType.BLACK_MARKER)

        basic_encoded = basic_encoder.encode_state(state)
        enhanced_encoded = encoder.encode_state(state)

        # First 4 channels should be identical
        assert np.allclose(basic_encoded[:4], enhanced_encoded[:4]), \
            "First 4 channels (pieces) should match basic encoder"


class TestCompareEncodingsUtility:
    """Test the compare_encodings utility function."""

    def test_compare_function_runs(self):
        """Test that compare_encodings function works."""
        state = GameState()
        state.phase = GamePhase.MAIN_GAME

        result = compare_encodings(state)

        assert 'basic_shape' in result
        assert 'enhanced_shape' in result
        assert result['basic_shape'] == (6, 11, 11)
        assert result['enhanced_shape'] == (15, 11, 11)


class TestEncodingStatsCollection:
    """Test statistics collection functionality."""

    def test_stats_disabled_by_default(self):
        """Test that stats are disabled by default."""
        encoder = EnhancedStateEncoder()  # No enable_stats
        state = GameState()

        encoder.encode_state(state)
        stats = encoder.get_last_stats()

        assert stats is None, "Stats should be None when disabled"

    def test_stats_enabled(self):
        """Test that stats are collected when enabled."""
        encoder = EnhancedStateEncoder(enable_stats=True)
        state = GameState()

        encoder.encode_state(state)
        stats = encoder.get_last_stats()

        assert stats is not None, "Stats should be collected when enabled"
        assert hasattr(stats, 'row_threats_found')
        assert hasattr(stats, 'partial_rows_found')
        assert hasattr(stats, 'ring_mobility_sum')
        assert hasattr(stats, 'ring_influence_coverage')


class TestDescribeChannels:
    """Test channel description functionality."""

    def test_describe_returns_all_channels(self):
        """Test that describe_channels returns all 15 channels."""
        encoder = EnhancedStateEncoder()
        descriptions = encoder.describe_channels()

        assert len(descriptions) == 15, f"Expected 15 descriptions, got {len(descriptions)}"
        for i in range(15):
            assert i in descriptions, f"Channel {i} not in descriptions"
            assert isinstance(descriptions[i], str), f"Channel {i} description not a string"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
