#!/usr/bin/env python3
"""Demo script for the random move policy."""

import logging
from yinsh_ml.self_play.policies import RandomMovePolicy, PolicyConfig
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.types import GamePhase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_random_policy():
    """Demonstrate the random move policy."""
    print("=== Random Move Policy Demo ===\n")
    
    # Create policy with default settings
    policy = RandomMovePolicy()
    print(f"Created policy with rule-based probability: {policy.config.rule_based_probability}")
    
    # Create a new game state
    game_state = GameState()
    print(f"Initial game phase: {game_state.phase}")
    print(f"Current player: {game_state.current_player}")
    
    # Get initial valid moves
    valid_moves = game_state.get_valid_moves()
    print(f"Number of valid moves: {len(valid_moves)}")
    
    # Select moves for several turns
    print("\n=== Move Selection Demo ===")
    for turn in range(5):
        if not game_state.get_valid_moves():
            print("No more valid moves available")
            break
        
        # Select a move
        selected_move = policy.select_move(game_state)
        print(f"Turn {turn + 1}: Selected {selected_move}")
        
        # Apply the move (simplified - just update the game state)
        try:
            game_state.apply_move(selected_move)
            print(f"  Game phase: {game_state.phase}")
            print(f"  Current player: {game_state.current_player}")
        except Exception as e:
            print(f"  Error applying move: {e}")
            break
    
    print("\n=== Policy Configuration Demo ===")
    
    # Test different configurations
    configs = [
        PolicyConfig(rule_based_probability=0.0, random_seed=42),  # Pure random
        PolicyConfig(rule_based_probability=0.5, random_seed=42),  # 50% rule-based
        PolicyConfig(rule_based_probability=1.0, random_seed=42), # Pure rule-based
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i + 1}:")
        print(f"  Rule-based probability: {config.rule_based_probability}")
        print(f"  Random seed: {config.random_seed}")
        
        policy = RandomMovePolicy(config)
        game_state = GameState()
        
        # Select a few moves
        for j in range(3):
            if not game_state.get_valid_moves():
                break
            move = policy.select_move(game_state)
            print(f"    Move {j + 1}: {move}")
            try:
                game_state.apply_move(move)
            except Exception as e:
                print(f"    Error: {e}")
                break


if __name__ == "__main__":
    demo_random_policy()
