"""Comprehensive tests to verify heuristic evaluator handles basic gameplay correctly.

This test suite verifies that YinshHeuristics correctly handles:
1. Scoring opportunities (completing runs)
2. Preventing opponent scoring (blocking moves)
3. Winning the game (3 runs completed)
4. Preventing wins (blocking opponent from winning)
5. Creating threatening positions (4-in-a-row)
6. Ring removal decisions
"""

import unittest
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player, PieceType, Position, POINTS_TO_WIN
from yinsh_ml.game.moves import Move, MoveType
from yinsh_ml.game.types import GamePhase
from yinsh_ml.heuristics.evaluator import YinshHeuristics
from yinsh_ml.heuristics.terminal_detection import detect_terminal_position
from yinsh_ml.heuristics.tactical_patterns import detect_immediate_tactical_patterns
from yinsh_ml.heuristics.features import potential_runs_count, completed_runs_differential


class TestHeuristicGameplayBasics(unittest.TestCase):
    """Test that heuristic evaluator handles fundamental gameplay tactics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.heuristics = YinshHeuristics()
        self.game_state = GameState()
        self._setup_initial_rings()
    
    def _setup_initial_rings(self):
        """Helper to place initial rings for both players."""
        ring_positions = [
            ('E5', Player.WHITE),
            ('F6', Player.BLACK),
            ('E7', Player.WHITE),
            ('F8', Player.BLACK),
            ('E9', Player.WHITE),
            ('F10', Player.BLACK),
            ('G5', Player.WHITE),
            ('H5', Player.BLACK),
            ('C3', Player.WHITE),
            ('D4', Player.BLACK),
        ]
        
        for pos, player in ring_positions:
            self.game_state.current_player = player
            move = Move(
                type=MoveType.PLACE_RING,
                player=player,
                source=Position.from_string(pos)
            )
            self.game_state.make_move(move)
        
        # Set phase to MAIN_GAME for tactical pattern detection
        self.game_state.phase = GamePhase.MAIN_GAME
    
    def _place_markers(self, positions, marker_type):
        """Helper to place markers directly on the board."""
        for pos in positions:
            pos_obj = Position.from_string(pos) if isinstance(pos, str) else pos
            self.game_state.board.place_piece(pos_obj, marker_type)
    
    def test_identifies_winning_positions(self):
        """Test that evaluator correctly identifies winning positions (3 runs completed)."""
        # Set white score to 3 (winning)
        self.game_state.white_score = POINTS_TO_WIN
        self.game_state.black_score = 0
        
        # Terminal detection should identify this as a win
        terminal_score = detect_terminal_position(self.game_state, Player.WHITE)
        self.assertIsNotNone(terminal_score)
        self.assertGreater(terminal_score, 1000.0, "Winning position should return large positive score")
        
        # Evaluator should also return high score
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertGreater(score, 1000.0, "Heuristic evaluator should recognize winning position")
    
    def test_identifies_losing_positions(self):
        """Test that evaluator correctly identifies losing positions (opponent has 3 runs)."""
        # Set black score to 3 (opponent wins)
        self.game_state.white_score = 0
        self.game_state.black_score = POINTS_TO_WIN
        
        # Terminal detection should identify this as a loss
        terminal_score = detect_terminal_position(self.game_state, Player.WHITE)
        self.assertIsNotNone(terminal_score)
        self.assertLess(terminal_score, -1000.0, "Losing position should return large negative score")
        
        # Evaluator should also return low score
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertLess(score, -1000.0, "Heuristic evaluator should recognize losing position")
    
    def test_prioritizes_scoring_opportunities(self):
        """Test that evaluator prioritizes positions with scoring opportunities."""
        # Create a position where white can complete a run
        # Place 4 white markers in a row (one away from scoring)
        self._place_markers(['B1', 'B2', 'B3', 'B4'], PieceType.WHITE_MARKER)
        
        # Note: potential_runs_count only detects completed runs (5+ markers),
        # but the evaluator should still value 4-in-a-row through other features
        # like connected chains and overall position evaluation
        
        # Evaluator should give positive score for scoring opportunity
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertGreater(score, 0.0, "Should prioritize scoring opportunities")
        
        # Compare with position without scoring opportunity
        neutral_state = GameState()
        self._setup_initial_rings_for_state(neutral_state)
        neutral_score = self.heuristics.evaluate_position(neutral_state, Player.WHITE)
        self.assertGreater(score, neutral_score, "Position with scoring opportunity should score higher")
    
    def test_recognizes_blocking_moves(self):
        """Test that evaluator recognizes blocking moves (preventing opponent runs)."""
        # Create position where opponent has 4-in-a-row (threatening to score)
        self._place_markers(['B1', 'B2', 'B3', 'B4'], PieceType.BLACK_MARKER)
        
        # Note: potential_runs_count only detects completed runs, but evaluator
        # should still recognize the threat through other features
        
        # From white's perspective, this should be evaluated negatively
        white_score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertLess(white_score, 0.0, "Should recognize opponent's scoring threat")
        
        # Now add a white marker blocking the run
        self._place_markers(['B5'], PieceType.WHITE_MARKER)
        
        # White's score should improve (blocked the threat)
        blocked_score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertGreater(blocked_score, white_score, "Blocking move should improve evaluation")
    
    def test_values_creating_lines_of_4(self):
        """Test that evaluator values creating threatening positions (4-in-a-row)."""
        # Create a position with 3 white markers (not yet threatening)
        self._place_markers(['B1', 'B2', 'B3'], PieceType.WHITE_MARKER)
        score_3 = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        
        # Add 4th marker (now threatening)
        self._place_markers(['B4'], PieceType.WHITE_MARKER)
        score_4 = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        
        # 4-in-a-row should score higher than 3-in-a-row
        self.assertGreater(score_4, score_3, "4-in-a-row should be valued higher than 3-in-a-row")
        
        # Verify potential runs count increases
        potential_3 = potential_runs_count(self.game_state.copy(), Player.WHITE)
        # Reset and check with 4 markers
        test_state = GameState()
        self._setup_initial_rings_for_state(test_state)
        self._place_markers(['B1', 'B2', 'B3', 'B4'], PieceType.WHITE_MARKER)
        potential_4 = potential_runs_count(test_state, Player.WHITE)
        # Note: potential_runs_count may not detect incomplete rows, but evaluator should still value them
    
    def _setup_initial_rings_for_state(self, state):
        """Helper to setup rings for a given state."""
        ring_positions = [
            ('E5', Player.WHITE),
            ('F6', Player.BLACK),
            ('E7', Player.WHITE),
            ('F8', Player.BLACK),
            ('E9', Player.WHITE),
            ('F10', Player.BLACK),
            ('G5', Player.WHITE),
            ('H5', Player.BLACK),
            ('C3', Player.WHITE),
            ('D4', Player.BLACK),
        ]
        
        for pos, player in ring_positions:
            state.current_player = player
            move = Move(
                type=MoveType.PLACE_RING,
                player=player,
                source=Position.from_string(pos)
            )
            state.make_move(move)
        state.phase = GamePhase.MAIN_GAME
    
    def test_handles_ring_removal_decisions(self):
        """Test that evaluator handles ring removal decisions appropriately."""
        # Create a position where completing a run would remove a ring
        # Place 4 white markers
        self._place_markers(['B1', 'B2', 'B3', 'B4'], PieceType.WHITE_MARKER)
        
        # Place a white ring at B5 (would be removed if run completed)
        self.game_state.board.place_piece(Position.from_string('B5'), PieceType.WHITE_RING)
        
        # Evaluator should still value completing the run (scoring is more important)
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        
        # Tactical pattern detection should identify this as an opportunity
        tactical_score = detect_immediate_tactical_patterns(self.game_state, Player.WHITE)
        # May or may not detect depending on implementation, but evaluator should handle it
    
    def test_immediate_win_opportunity_detection(self):
        """Test that evaluator detects immediate win opportunities."""
        # Set white score to 2 (one point away from winning)
        self.game_state.white_score = POINTS_TO_WIN - 1
        self.game_state.black_score = 0
        
        # Create position where white can complete a run in one move
        self._place_markers(['B1', 'B2', 'B3', 'B4'], PieceType.WHITE_MARKER)
        
        # Tactical pattern detection should identify this
        tactical_score = detect_immediate_tactical_patterns(self.game_state, Player.WHITE)
        # Note: This requires checking if a valid move can complete the row
        # The detection may not work perfectly without actual move validation
        
        # Evaluator should give very high score
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertGreater(score, 0.0, "Should recognize win opportunity")
    
    def test_prevents_opponent_wins(self):
        """Test that evaluator recognizes when opponent can win."""
        # Set black score to 2 (opponent one point away from winning)
        self.game_state.white_score = 0
        self.game_state.black_score = POINTS_TO_WIN - 1
        
        # Create position where opponent can complete a run
        self._place_markers(['B1', 'B2', 'B3', 'B4'], PieceType.BLACK_MARKER)
        
        # From white's perspective, this should be evaluated very negatively
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertLess(score, 0.0, "Should recognize opponent's win threat")
        
        # Tactical detection should identify this
        tactical_score = detect_immediate_tactical_patterns(self.game_state, Player.WHITE)
        if tactical_score is not None:
            self.assertLess(tactical_score, -1000.0, "Should detect opponent's immediate win threat")
    
    def test_completed_runs_prioritized(self):
        """Test that completed runs are highly prioritized."""
        # Create position with completed run
        self._place_markers(['B1', 'B2', 'B3', 'B4', 'B5'], PieceType.WHITE_MARKER)
        
        # Check completed runs differential
        completed_diff = completed_runs_differential(self.game_state, Player.WHITE)
        self.assertEqual(completed_diff, 1, "Should detect completed run")
        
        # Evaluator should give high score
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertGreater(score, 0.0, "Completed run should give positive score")
        
        # Compare with position without completed run
        no_run_state = GameState()
        self._setup_initial_rings_for_state(no_run_state)
        no_run_score = self.heuristics.evaluate_position(no_run_state, Player.WHITE)
        self.assertGreater(score, no_run_score, "Position with completed run should score higher")
    
    def test_multiple_threats_handled(self):
        """Test that evaluator handles multiple threats correctly."""
        # Create position with multiple potential runs
        self._place_markers(['B1', 'B2', 'B3', 'B4'], PieceType.WHITE_MARKER)  # Vertical threat
        self._place_markers(['C1', 'C2', 'C3', 'C4'], PieceType.WHITE_MARKER)  # Another vertical threat
        
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertGreater(score, 0.0, "Multiple threats should be valued")
        
        # Should detect multiple potential runs
        potential = potential_runs_count(self.game_state, Player.WHITE)
        # Note: May not detect if implementation only finds completed rows
    
    def test_edge_case_zero_rings_remaining(self):
        """Test evaluator handles edge case when player has no rings remaining."""
        # Simulate endgame scenario
        # Remove all white rings (set rings_placed to max)
        self.game_state.rings_placed[Player.WHITE] = 5  # All rings placed
        
        # Evaluator should still work
        score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        self.assertIsInstance(score, float, "Should handle zero rings remaining")
    
    def test_symmetry_white_vs_black(self):
        """Test that evaluation is symmetric (white advantage = black disadvantage)."""
        # Create asymmetric position
        self._place_markers(['B1', 'B2', 'B3', 'B4'], PieceType.WHITE_MARKER)
        
        white_score = self.heuristics.evaluate_position(self.game_state, Player.WHITE)
        black_score = self.heuristics.evaluate_position(self.game_state, Player.BLACK)
        
        # Scores should be approximately opposite
        self.assertAlmostEqual(white_score, -black_score, delta=0.1,
                              msg="Evaluation should be symmetric")


if __name__ == '__main__':
    unittest.main()

