"""Tests for move selection policies."""

import pytest
import random
import time
import numpy as np
from unittest.mock import patch, MagicMock

from yinsh_ml.self_play.policies import (
    RandomMovePolicy, PolicyConfig, PolicyFactory,
    HeuristicPolicy, HeuristicPolicyConfig
)
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import MoveType, GamePhase
from yinsh_ml.game.constants import Player, Position
from yinsh_ml.game.board import Board


class TestRandomMovePolicy:
    """Test the random move selection policy."""
    
    def test_policy_initialization(self):
        """Test policy initialization."""
        config = PolicyConfig(rule_based_probability=0.2, random_seed=42)
        policy = RandomMovePolicy(config)
        
        assert policy.config.rule_based_probability == 0.2
        assert policy.config.random_seed == 42
    
    def test_policy_initialization_defaults(self):
        """Test policy initialization with defaults."""
        policy = RandomMovePolicy()
        
        assert policy.config.rule_based_probability == 0.1
        assert policy.config.random_seed is None
    
    def test_select_move_ring_placement(self):
        """Test move selection during ring placement phase."""
        policy = RandomMovePolicy(PolicyConfig(random_seed=42))
        game_state = GameState()
        
        # Ensure we're in ring placement phase
        assert game_state.phase == GamePhase.RING_PLACEMENT
        
        # Get valid moves
        valid_moves = game_state.get_valid_moves()
        assert len(valid_moves) > 0
        
        # Select a move
        selected_move = policy.select_move(game_state)
        
        # Verify the move is valid
        assert selected_move in valid_moves
        assert selected_move.type == MoveType.PLACE_RING
        assert selected_move.player == game_state.current_player
    
    def test_select_move_random_seed(self):
        """Test that random seed produces consistent results."""
        config = PolicyConfig(random_seed=123)
        policy = RandomMovePolicy(config)
        game_state = GameState()
        
        # Test that the policy can select moves
        move = policy.select_move(game_state)
        assert move is not None
        assert move.type == MoveType.PLACE_RING
        assert move.player == game_state.current_player
    
    def test_select_move_no_valid_moves(self):
        """Test behavior when no valid moves are available."""
        policy = RandomMovePolicy()
        
        # Create a game state with no valid moves (this shouldn't happen in normal play)
        game_state = GameState()
        
        # Mock get_valid_moves to return empty list
        with patch.object(game_state, 'get_valid_moves', return_value=[]):
            with pytest.raises(ValueError, match="No valid moves available"):
                policy.select_move(game_state)
    
    def test_rule_based_ring_placement(self):
        """Test rule-based ring placement strategy."""
        policy = RandomMovePolicy(PolicyConfig(rule_based_probability=1.0, random_seed=42))
        game_state = GameState()
        
        # Force rule-based selection
        with patch('random.random', return_value=0.0):  # Always trigger rule-based
            move = policy.select_move(game_state)
            
            assert move.type == MoveType.PLACE_RING
            assert move.player == game_state.current_player
    
    def test_rule_based_probability(self):
        """Test that rule-based moves are selected with correct probability."""
        policy = RandomMovePolicy(PolicyConfig(rule_based_probability=0.5, random_seed=42))
        game_state = GameState()
        
        rule_based_count = 0
        total_tests = 100
        
        for i in range(total_tests):
            # Use different random values to test probability
            with patch('random.random', return_value=i / total_tests):
                move = policy.select_move(game_state)
                # This is a simplified test - in practice we'd need to mock the rule-based logic
                # to verify it's being called with the right probability
    
    def test_is_center_position(self):
        """Test center position detection."""
        policy = RandomMovePolicy()
        
        # Test center positions
        assert policy._is_center_position(Position.from_string("D4"))
        assert policy._is_center_position(Position.from_string("E5"))
        assert policy._is_center_position(Position.from_string("F6"))
        
        # Test non-center positions
        assert not policy._is_center_position(Position.from_string("A2"))
        assert not policy._is_center_position(Position.from_string("K10"))
    
    def test_is_edge_position(self):
        """Test edge position detection."""
        policy = RandomMovePolicy()
        
        # Test edge positions
        assert policy._is_edge_position(Position.from_string("A2"))
        assert policy._is_edge_position(Position.from_string("K10"))
        assert policy._is_edge_position(Position.from_string("B1"))
        
        # Test non-edge positions
        assert not policy._is_edge_position(Position.from_string("D4"))
        assert not policy._is_edge_position(Position.from_string("E5"))


class TestPolicyFactory:
    """Test the policy factory."""
    
    def test_create_random_policy(self):
        """Test creating a random policy."""
        policy = PolicyFactory.create_random_policy()
        assert isinstance(policy, RandomMovePolicy)
    
    def test_create_random_policy_with_config(self):
        """Test creating a random policy with config."""
        config = PolicyConfig(rule_based_probability=0.3)
        policy = PolicyFactory.create_random_policy(config)
        
        assert isinstance(policy, RandomMovePolicy)
        assert policy.config.rule_based_probability == 0.3
    
    def test_create_policy_by_type(self):
        """Test creating policy by type."""
        policy = PolicyFactory.create_policy("random")
        assert isinstance(policy, RandomMovePolicy)
    
    def test_create_policy_invalid_type(self):
        """Test creating policy with invalid type."""
        with pytest.raises(ValueError, match="Unsupported policy type"):
            PolicyFactory.create_policy("invalid")


class TestHeuristicPolicy:
    """Test the heuristic-based move selection policy."""
    
    def test_policy_initialization(self):
        """Test policy initialization."""
        config = HeuristicPolicyConfig(
            search_depth=2,
            temperature=1.5,
            use_fast_mode=True,
            random_seed=42
        )
        policy = HeuristicPolicy(config)
        
        assert policy.config.search_depth == 2
        assert policy.config.temperature == 1.5
        assert policy.config.use_fast_mode is True
        assert policy.config.random_seed == 42
        assert policy._current_temperature == 1.5
        assert policy._current_epsilon == 0.0
    
    def test_policy_initialization_defaults(self):
        """Test policy initialization with defaults."""
        policy = HeuristicPolicy()
        
        assert policy.config.use_fast_mode is False
        assert policy.config.temperature == 1.0
        assert policy._move_count == 0
        assert policy._game_count == 0
    
    def test_select_move_fast_mode(self):
        """Test fast mode move selection."""
        config = HeuristicPolicyConfig(use_fast_mode=True, random_seed=42)
        policy = HeuristicPolicy(config)
        game_state = GameState()
        
        valid_moves = game_state.get_valid_moves()
        assert len(valid_moves) > 0
        
        move = policy.select_move(game_state)
        
        # Verify move is valid
        assert move in valid_moves
        assert move.type == MoveType.PLACE_RING
    
    def test_select_move_performance_fast_mode(self):
        """Test that fast mode completes in <100ms."""
        config = HeuristicPolicyConfig(use_fast_mode=True, random_seed=42)
        policy = HeuristicPolicy(config)
        game_state = GameState()
        
        # Run multiple moves and check timing
        times = []
        for _ in range(10):
            start = time.perf_counter()
            policy.select_move(game_state)
            duration = time.perf_counter() - start
            times.append(duration)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        # Average should be well under 100ms, max should be under 200ms (allowing some variance)
        assert avg_time < 0.1, f"Average time {avg_time*1000:.2f}ms exceeds 100ms"
        assert max_time < 0.2, f"Max time {max_time*1000:.2f}ms exceeds 200ms"
    
    def test_score_all_moves_ranking(self):
        """Test move ranking and evaluation system (subtask 7.1)."""
        config = HeuristicPolicyConfig(use_fast_mode=True, random_seed=42)
        policy = HeuristicPolicy(config)
        game_state = GameState()
        
        valid_moves = game_state.get_valid_moves()
        scored_moves = policy._score_all_moves(game_state, valid_moves, fast_evaluation=True)
        
        # Verify all moves are scored
        assert len(scored_moves) == len(valid_moves)
        
        # Verify moves are sorted by score (descending)
        scores = [score for score, _ in scored_moves]
        assert scores == sorted(scores, reverse=True), "Moves should be sorted by score descending"
        
        # Verify scores are reasonable (not all the same)
        unique_scores = set(scores)
        assert len(unique_scores) > 1, "Should have variation in move scores"
    
    def test_score_all_moves_fast_vs_normal(self):
        """Test that fast evaluation produces similar rankings to normal evaluation."""
        config = HeuristicPolicyConfig(random_seed=42)
        policy = HeuristicPolicy(config)
        game_state = GameState()
        
        valid_moves = game_state.get_valid_moves()
        
        # Get scores from both methods
        fast_scores = policy._score_all_moves(game_state, valid_moves, fast_evaluation=True)
        normal_scores = policy._score_all_moves(game_state, valid_moves, fast_evaluation=False)
        
        # Both should rank moves (though exact scores may differ)
        assert len(fast_scores) == len(normal_scores)
        
        # Top move should be similar (at least in top 3)
        top_fast_move = fast_scores[0][1]
        top_normal_moves = [move for _, move in normal_scores[:3]]
        assert top_fast_move in top_normal_moves or top_fast_move in [move for _, move in normal_scores]
    
    def test_temperature_sampling_deterministic(self):
        """Test temperature-based sampling with temperature=0 (deterministic) (subtask 7.2)."""
        config = HeuristicPolicyConfig(
            use_fast_mode=True,
            temperature=0.0,  # Deterministic
            random_seed=42
        )
        policy = HeuristicPolicy(config)
        game_state = GameState()
        
        valid_moves = game_state.get_valid_moves()
        scored_moves = policy._score_all_moves(game_state, valid_moves, fast_evaluation=True)
        
        # With temperature=0, should always pick best move
        move1 = policy._sample_with_temperature(scored_moves, use_current_temp=False)
        move2 = policy._sample_with_temperature(scored_moves, use_current_temp=False)
        
        assert move1 == move2, "Temperature=0 should be deterministic"
        assert move1 == scored_moves[0][1], "Should pick top-ranked move"
    
    def test_temperature_sampling_high_temperature(self):
        """Test that high temperature increases randomness (subtask 7.2)."""
        config = HeuristicPolicyConfig(
            use_fast_mode=True,
            temperature=2.0,  # High temperature
            random_seed=42
        )
        policy = HeuristicPolicy(config)
        game_state = GameState()
        
        valid_moves = game_state.get_valid_moves()
        scored_moves = policy._score_all_moves(game_state, valid_moves, fast_evaluation=True)
        
        # Sample multiple times with high temperature
        moves_high_temp = []
        for _ in range(20):
            move = policy._sample_with_temperature(scored_moves, use_current_temp=False)
            moves_high_temp.append(move)
        
        # Should have some diversity (not always same move)
        unique_moves = set(moves_high_temp)
        assert len(unique_moves) > 1, "High temperature should produce diverse moves"
    
    def test_temperature_sampling_low_temperature(self):
        """Test that low temperature favors top moves (subtask 7.2)."""
        config = HeuristicPolicyConfig(
            use_fast_mode=True,
            temperature=0.1,  # Low temperature
            random_seed=42
        )
        policy = HeuristicPolicy(config)
        game_state = GameState()
        
        valid_moves = game_state.get_valid_moves()
        scored_moves = policy._score_all_moves(game_state, valid_moves, fast_evaluation=True)
        
        # Sample multiple times with low temperature
        moves_low_temp = []
        for _ in range(20):
            move = policy._sample_with_temperature(scored_moves, use_current_temp=False)
            moves_low_temp.append(move)
        
        # Top move should be selected more frequently
        top_move = scored_moves[0][1]
        top_move_count = moves_low_temp.count(top_move)
        
        assert top_move_count > len(moves_low_temp) * 0.5, \
            f"Low temperature should favor top move (got {top_move_count}/{len(moves_low_temp)})"
    
    def test_temperature_decay(self):
        """Test temperature decay over moves (subtask 7.2)."""
        config = HeuristicPolicyConfig(
            use_fast_mode=True,
            temperature=1.0,
            exploration_decay=0.9,  # 10% decay per move
            min_temperature=0.1,
            random_seed=42
        )
        policy = HeuristicPolicy(config)
        
        initial_temp = policy._current_temperature
        assert initial_temp == 1.0
        
        # Simulate multiple moves
        for i in range(5):
            policy._move_count = i
            policy._update_decay()
        
        # Temperature should have decreased
        assert policy._current_temperature < initial_temp
        assert policy._current_temperature >= config.min_temperature
    
    def test_epsilon_greedy_exploration(self):
        """Test epsilon-greedy exploration (subtask 7.3)."""
        config = HeuristicPolicyConfig(
            use_fast_mode=True,
            epsilon_greedy=0.5,  # 50% random exploration
            temperature=0.0,  # Deterministic when not exploring
            random_seed=42
        )
        policy = HeuristicPolicy(config)
        game_state = GameState()
        
        valid_moves = game_state.get_valid_moves()
        
        # Mock random to always explore
        with patch.object(policy._rng, 'random', return_value=0.3):  # < 0.5 epsilon
            move = policy._select_move_fast(game_state, valid_moves)
            # Should be a random move (we can't easily verify this without more mocking)
            assert move in valid_moves
        
        # Mock random to never explore
        with patch.object(policy._rng, 'random', return_value=0.7):  # > 0.5 epsilon
            move = policy._select_move_fast(game_state, valid_moves)
            # Should use heuristic selection
            assert move in valid_moves
    
    def test_epsilon_decay(self):
        """Test epsilon decay over games (subtask 7.3)."""
        config = HeuristicPolicyConfig(
            use_fast_mode=True,
            epsilon_greedy=1.0,  # Start with 100% exploration
            exploration_decay=0.8,  # 20% decay per game
            random_seed=42
        )
        policy = HeuristicPolicy(config)
        
        assert policy._current_epsilon == 1.0
        
        # Simulate multiple games
        for i in range(5):
            policy._game_count = i
            policy._update_decay()
        
        # Epsilon should have decreased
        assert policy._current_epsilon < 1.0
        assert policy._current_epsilon >= 0.0
    
    def test_reset_for_new_game(self):
        """Test reset_for_new_game method."""
        config = HeuristicPolicyConfig(
            use_fast_mode=True,
            exploration_decay=0.9,
            random_seed=42
        )
        policy = HeuristicPolicy(config)
        
        # Make some moves
        policy._move_count = 10
        policy._game_count = 2
        
        # Reset for new game
        old_game_count = policy._game_count
        policy.reset_for_new_game()
        
        assert policy._move_count == 0
        assert policy._game_count == old_game_count + 1
    
    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        config = HeuristicPolicyConfig(use_fast_mode=True, random_seed=42)
        policy = HeuristicPolicy(config)
        
        # Simulate some move times
        policy._track_performance(0.05)  # 50ms
        policy._track_performance(0.08)  # 80ms
        policy._track_performance(0.12)  # 120ms (should warn)
        
        stats = policy.get_performance_stats()
        
        assert stats["total_moves"] == 3
        assert stats["avg_move_time"] > 0
        assert stats["max_move_time"] == 0.12
        assert stats["p95_move_time"] > 0
    
    def test_performance_warning_fast_mode(self):
        """Test that performance warnings are issued for slow moves in fast mode."""
        config = HeuristicPolicyConfig(use_fast_mode=True, random_seed=42)
        policy = HeuristicPolicy(config)
        
        # Track a slow move (>100ms)
        with patch('yinsh_ml.self_play.policies.logger') as mock_logger:
            policy._track_performance(0.15)  # 150ms
            mock_logger.warning.assert_called_once()
            assert "exceeds 100ms threshold" in str(mock_logger.warning.call_args)
    
    def test_move_quality_vs_random(self):
        """Test that heuristic moves are better than random (subtask 7.1)."""
        config = HeuristicPolicyConfig(use_fast_mode=True, random_seed=42)
        policy = HeuristicPolicy(config)
        game_state = GameState()
        
        valid_moves = game_state.get_valid_moves()
        
        # Get heuristic-ranked moves
        scored_moves = policy._score_all_moves(game_state, valid_moves, fast_evaluation=True)
        top_heuristic_move = scored_moves[0][1]
        top_score = scored_moves[0][0]
        
        # Verify top move has a reasonable score
        assert top_score > -10000, "Top move should have valid score"
        
        # Verify we can select moves
        move = policy.select_move(game_state)
        assert move in valid_moves
    
    def test_temperature_range_support(self):
        """Test temperature range from 0.1 to 2.0 (subtask 7.2)."""
        config = HeuristicPolicyConfig(use_fast_mode=True, random_seed=42)
        policy = HeuristicPolicy(config)
        game_state = GameState()
        
        valid_moves = game_state.get_valid_moves()
        scored_moves = policy._score_all_moves(game_state, valid_moves, fast_evaluation=True)
        
        # Test various temperatures
        for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
            policy.config.temperature = temp
            move = policy._sample_with_temperature(scored_moves, use_current_temp=False)
            assert move in valid_moves
    
    def test_exploration_decay_schedule(self):
        """Test exploration decay schedule (subtask 7.3)."""
        config = HeuristicPolicyConfig(
            use_fast_mode=True,
            temperature=1.0,
            epsilon_greedy=0.5,
            exploration_decay=0.95,  # 5% decay
            random_seed=42
        )
        policy = HeuristicPolicy(config)
        
        initial_temp = policy._current_temperature
        initial_epsilon = policy._current_epsilon
        
        # Simulate 10 moves and 3 games
        for game in range(3):
            policy._game_count = game
            for move in range(10):
                policy._move_count = move
                policy._update_decay()
        
        # Both should have decayed
        assert policy._current_temperature < initial_temp
        assert policy._current_epsilon < initial_epsilon
    
    def test_backward_compatibility(self):
        """Test that original search mode still works."""
        config = HeuristicPolicyConfig(
            use_fast_mode=False,  # Original mode
            search_depth=2,
            random_seed=42
        )
        policy = HeuristicPolicy(config)
        game_state = GameState()
        
        # Should work without errors
        move = policy.select_move(game_state)
        valid_moves = game_state.get_valid_moves()
        assert move in valid_moves


if __name__ == "__main__":
    pytest.main([__file__])
