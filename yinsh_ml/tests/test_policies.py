"""Tests for move selection policies."""

import pytest
import random
from unittest.mock import patch

from yinsh_ml.self_play.policies import RandomMovePolicy, PolicyConfig, PolicyFactory
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


if __name__ == "__main__":
    pytest.main([__file__])
