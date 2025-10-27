"""Enhanced self-play system using analysis-integrated MCTS.

This module provides a drop-in replacement for the standard self-play system
that incorporates analysis findings for improved performance.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from ..game.game_state import GameState
from ..game.constants import Player
from ..utils.encoding import StateEncoder
from ..utils.mcts_metrics import MCTSMetrics
from ..memory.game_state_pool import GameStatePool
from ..network.wrapper import NetworkWrapper
from .enhanced_mcts import EnhancedMCTS, EnhancedMCTSConfig
from .self_play import SelfPlay


class EnhancedSelfPlay(SelfPlay):
    """Enhanced self-play using analysis-integrated MCTS."""
    
    def __init__(self,
                 network: NetworkWrapper,
                 num_workers: int,
                 # Enhanced MCTS parameters
                 enhanced_config: EnhancedMCTSConfig,
                 # Standard parameters (for compatibility)
                 num_simulations: int = 100,
                 late_simulations: Optional[int] = None,
                 simulation_switch_ply: int = 20,
                 c_puct: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 value_weight: float = 1.0,
                 max_depth: int = 50,
                 # Temperature parameters
                 initial_temp: float = 1.0,
                 final_temp: float = 0.1,
                 annealing_steps: int = 30,
                 temp_clamp_fraction: float = 0.8,
                 # Optional components
                 metrics_logger: Optional[Any] = None,
                 mcts_metrics: Optional[MCTSMetrics] = None,
                 game_state_pool: Optional[GameStatePool] = None):
        """
        Initialize enhanced self-play system.
        
        Args:
            network: Neural network wrapper
            num_workers: Number of worker processes
            enhanced_config: Enhanced MCTS configuration
            Other parameters: Standard self-play parameters for compatibility
        """
        self.logger = logging.getLogger("EnhancedSelfPlay")
        
        # Store enhanced configuration
        self.enhanced_config = enhanced_config
        
        # Initialize parent class with standard parameters
        super().__init__(
            network=network,
            num_workers=num_workers,
            num_simulations=num_simulations,
            late_simulations=late_simulations,
            simulation_switch_ply=simulation_switch_ply,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            value_weight=value_weight,
            max_depth=max_depth,
            initial_temp=initial_temp,
            final_temp=final_temp,
            annealing_steps=annealing_steps,
            temp_clamp_fraction=temp_clamp_fraction,
            metrics_logger=metrics_logger,
            mcts_metrics=mcts_metrics,
            game_state_pool=game_state_pool
        )
        
        # Override MCTS with enhanced version
        self.mcts = EnhancedMCTS(
            network=self.network,
            config=self.enhanced_config,
            mcts_metrics=self.mcts_metrics,
            game_state_pool=self.game_state_pool
        )
        
        self.logger.info("Enhanced Self-Play Initialized:")
        self.logger.info(f"  Using Enhanced MCTS with analysis integration")
        self.logger.info(f"  Heuristic Evaluation: {enhanced_config.use_heuristic_evaluation}")
        self.logger.info(f"  Phase-aware Budget: {enhanced_config.use_phase_aware_budget}")
        self.logger.info(f"  Enhanced UCB: {enhanced_config.use_enhanced_ucb}")
    
    def _create_worker_mcts(self, worker_id: int) -> EnhancedMCTS:
        """Create enhanced MCTS instance for worker process."""
        return EnhancedMCTS(
            network=self.network,
            config=self.enhanced_config,
            mcts_metrics=self.mcts_metrics,
            game_state_pool=self.game_state_pool
        )
    
    def play_game(self, game_id: int = 0) -> Tuple[List[np.ndarray], List[np.ndarray], float, List[Dict]]:
        """
        Play a single game using enhanced MCTS.
        
        Returns:
            Tuple of (training_states, training_policies, game_outcome, game_metadata)
        """
        self.logger.debug(f"Starting enhanced self-play game {game_id}")
        
        # Initialize game
        state = GameState()
        training_states = []
        training_policies = []
        move_number = 0
        
        # Game metadata for analysis
        game_metadata = {
            'game_id': game_id,
            'total_moves': 0,
            'phase_distribution': {'early': 0, 'mid': 0, 'late': 0},
            'enhanced_features_used': {
                'heuristic_evaluation': self.enhanced_config.use_heuristic_evaluation,
                'phase_aware_budget': self.enhanced_config.use_phase_aware_budget,
                'enhanced_ucb': self.enhanced_config.use_enhanced_ucb
            }
        }
        
        while not state.is_terminal():
            # Get enhanced MCTS policy
            policy = self.mcts.search(state, move_number)
            
            # Select move based on policy
            move = self._select_move_from_policy(policy, state, move_number)
            
            # Store training data
            training_states.append(self.state_encoder.encode_state(state))
            training_policies.append(policy)
            
            # Make move
            state.make_move(move)
            move_number += 1
            
            # Track phase distribution
            from ..analysis.phase_analyzer import PhaseConfig
            phase_config = PhaseConfig()
            current_phase = phase_config.get_phase(move_number)
            game_metadata['phase_distribution'][current_phase.value] += 1
        
        # Determine game outcome
        winner = state.get_winner()
        if winner == Player.WHITE:
            outcome = 1.0
        elif winner == Player.BLACK:
            outcome = -1.0
        else:
            outcome = 0.0
        
        game_metadata['total_moves'] = move_number
        game_metadata['outcome'] = outcome
        game_metadata['winner'] = winner.value if winner else 'draw'
        
        self.logger.debug(f"Game {game_id} completed: {move_number} moves, outcome: {outcome}")
        
        return training_states, training_policies, outcome, [game_metadata]
    
    def _select_move_from_policy(self, policy: np.ndarray, state: GameState, move_number: int) -> int:
        """Select move from policy vector."""
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Get valid move indices
        valid_indices = []
        valid_probs = []
        for move in valid_moves:
            move_idx = self.state_encoder.move_to_index(move)
            if 0 <= move_idx < len(policy):
                valid_indices.append(move_idx)
                valid_probs.append(policy[move_idx])
        
        if not valid_indices:
            # Fallback to random selection
            return valid_moves[0]
        
        # Normalize probabilities
        valid_probs = np.array(valid_probs)
        if valid_probs.sum() > 0:
            valid_probs /= valid_probs.sum()
        else:
            valid_probs = np.ones_like(valid_probs) / len(valid_probs)
        
        # Sample move
        selected_idx = np.random.choice(len(valid_indices), p=valid_probs)
        return valid_moves[selected_idx]
    
    def generate_games(self, num_games: int) -> List[Tuple[List[np.ndarray], List[np.ndarray], float, List[Dict]]]:
        """
        Generate multiple games using enhanced self-play.
        
        Args:
            num_games: Number of games to generate
            
        Returns:
            List of game results
        """
        self.logger.info(f"Generating {num_games} games with enhanced self-play")
        
        games = []
        for game_id in range(num_games):
            try:
                game_result = self.play_game(game_id)
                games.append(game_result)
                
                if (game_id + 1) % 10 == 0:
                    self.logger.info(f"Generated {game_id + 1}/{num_games} games")
                    
            except Exception as e:
                self.logger.error(f"Error generating game {game_id}: {e}")
                continue
        
        self.logger.info(f"Successfully generated {len(games)} games")
        return games


def create_enhanced_self_play_from_config(network: NetworkWrapper, 
                                        config_dict: Dict[str, Any]) -> EnhancedSelfPlay:
    """
    Create enhanced self-play instance from configuration dictionary.
    
    Args:
        network: Neural network wrapper
        config_dict: Configuration dictionary
        
    Returns:
        Enhanced self-play instance
    """
    # Extract enhanced MCTS configuration
    enhanced_config = EnhancedMCTSConfig(
        num_simulations=config_dict.get('num_simulations', 100),
        late_simulations=config_dict.get('late_simulations'),
        simulation_switch_ply=config_dict.get('simulation_switch_ply', 20),
        c_puct=config_dict.get('c_puct', 1.0),
        dirichlet_alpha=config_dict.get('dirichlet_alpha', 0.3),
        value_weight=config_dict.get('value_weight', 1.0),
        max_depth=config_dict.get('max_depth', 50),
        initial_temp=config_dict.get('initial_temp', 1.0),
        final_temp=config_dict.get('final_temp', 0.1),
        annealing_steps=config_dict.get('annealing_steps', 30),
        temp_clamp_fraction=config_dict.get('temp_clamp_fraction', 0.8),
        use_heuristic_evaluation=config_dict.get('use_heuristic_evaluation', True),
        use_phase_aware_budget=config_dict.get('use_phase_aware_budget', True),
        use_enhanced_ucb=config_dict.get('use_enhanced_ucb', True),
        heuristic_weight=config_dict.get('heuristic_weight', 0.3)
    )
    
    return EnhancedSelfPlay(
        network=network,
        num_workers=config_dict.get('num_workers', 1),
        enhanced_config=enhanced_config,
        num_simulations=config_dict.get('num_simulations', 100),
        late_simulations=config_dict.get('late_simulations'),
        simulation_switch_ply=config_dict.get('simulation_switch_ply', 20),
        c_puct=config_dict.get('c_puct', 1.0),
        dirichlet_alpha=config_dict.get('dirichlet_alpha', 0.3),
        value_weight=config_dict.get('value_weight', 1.0),
        max_depth=config_dict.get('max_depth', 50),
        initial_temp=config_dict.get('initial_temp', 1.0),
        final_temp=config_dict.get('final_temp', 0.1),
        annealing_steps=config_dict.get('annealing_steps', 30),
        temp_clamp_fraction=config_dict.get('temp_clamp_fraction', 0.8)
    )
