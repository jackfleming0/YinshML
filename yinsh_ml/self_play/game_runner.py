"""Self-play game runner for Yinsh."""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path

from ..game import GameState, Player, GamePhase
from ..game.moves import MoveGenerator
from .random_policy import RandomMovePolicy, PolicyConfig
from .policies import PolicyFactory, HeuristicPolicyConfig, MCTSPolicyConfig, AdaptivePolicyConfig
from .game_recorder import GameRecorder, GameRecord
from .data_storage import SelfPlayDataManager, StorageConfig
from .quality_metrics import QualityAnalyzer, GameQualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for the self-play game runner."""
    target_games: int = 100
    max_games_per_batch: int = 10
    save_interval: int = 10  # Save every N games
    progress_interval: int = 100  # Log progress every N games
    rule_based_probability: float = 0.1
    random_seed: Optional[int] = None
    max_turns_per_game: int = 200  # Safety limit
    output_dir: str = "self_play_data"
    use_parquet_storage: bool = True
    parquet_batch_size: int = 100
    validation_enabled: bool = True
    # Policy configuration
    policy_type: str = "random"  # random | heuristic | mcts | adaptive
    policy_config: Optional[Dict[str, Any]] = None  # Config dict for selected policy
    network: Any = None  # Optional neural network wrapper
    training_progress_callback: Optional[Callable[[int], float]] = None  # For adaptive policy
    # Quality metrics
    compute_quality_metrics: bool = False  # Enable quality analysis
    quality_baseline_path: Optional[str] = None  # Path to baseline for comparison


@dataclass
class RunnerStats:
    """Statistics for the self-play runner."""
    games_completed: int = 0
    games_failed: int = 0
    total_turns: int = 0
    total_duration: float = 0.0
    average_game_length: float = 0.0
    average_duration: float = 0.0
    games_per_hour: float = 0.0
    start_time: float = 0.0
    last_update_time: float = 0.0
    # Enhanced performance tracking
    data_storage_size_mb: float = 0.0
    storage_efficiency_mb_per_game: float = 0.0
    validation_time_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    compression_ratio: float = 0.0
    # Quality metrics
    average_move_diversity: float = 0.0
    average_strategic_coherence: float = 0.0
    average_tactical_patterns: float = 0.0
    quality_metrics: List[GameQualityMetrics] = None


class SelfPlayRunner:
    """Runs self-play games using random move selection."""
    
    def __init__(self, config: RunnerConfig = None):
        """Initialize the self-play runner.
        
        Args:
            config: Runner configuration. If None, uses default config.
        """
        self.config = config or RunnerConfig()
        self.stats = RunnerStats()
        self.stats.quality_metrics = []
        self.feature_history: List[Dict[str, Any]] = []
        
        # Initialize policy using factory
        self.policy = self._create_policy()
        
        self.recorder = GameRecorder(self.config.output_dir, save_json=not self.config.use_parquet_storage)
        
        # Initialize quality analyzer if enabled
        self.quality_analyzer = None
        if self.config.compute_quality_metrics:
            self.quality_analyzer = QualityAnalyzer()
            logger.info("Quality metrics analysis enabled")
        
        # Initialize parquet storage if enabled
        if self.config.use_parquet_storage:
            storage_config = StorageConfig(
                output_dir=self.config.output_dir,
                batch_size=self.config.parquet_batch_size,
                validation_enabled=self.config.validation_enabled
            )
            self.data_manager = SelfPlayDataManager(storage_config)
        else:
            self.data_manager = None
        
        # Progress tracking
        self.progress_callbacks: List[Callable[[RunnerStats], None]] = []
        
        logger.info(f"Initialized SelfPlayRunner with target: {self.config.target_games} games, "
                   f"policy: {self.config.policy_type}")
    
    def _create_policy(self):
        """Create policy instance based on configuration.
        
        Returns:
            Policy instance
        """
        policy_config = self.config.policy_config or {}
        
        if self.config.policy_type == "random":
            from .random_policy import PolicyConfig
            config = PolicyConfig(
                rule_based_probability=self.config.rule_based_probability,
                random_seed=self.config.random_seed
            )
            return PolicyFactory.create_random_policy(config)
        
        elif self.config.policy_type == "heuristic":
            config = HeuristicPolicyConfig(
                search_depth=policy_config.get("search_depth", 3),
                randomness=policy_config.get("randomness", 0.1),
                time_limit=policy_config.get("time_limit", 1.0),
                temperature=policy_config.get("temperature", 1.0),
                random_seed=self.config.random_seed
            )
            return PolicyFactory.create_heuristic_policy(config)
        
        elif self.config.policy_type == "mcts":
            config = MCTSPolicyConfig(
                num_simulations=policy_config.get("num_simulations", 100),
                evaluation_mode=policy_config.get("evaluation_mode", "hybrid"),
                heuristic_weight=policy_config.get("heuristic_weight", 0.5),
                use_dirichlet=policy_config.get("use_dirichlet", True),
                random_seed=self.config.random_seed
            )
            return PolicyFactory.create_mcts_policy(config, self.config.network)
        
        elif self.config.policy_type == "adaptive":
            heuristic_config = None
            if "heuristic" in policy_config:
                heuristic_config = HeuristicPolicyConfig(**policy_config["heuristic"])
            
            mcts_config = None
            if "mcts" in policy_config:
                mcts_config = MCTSPolicyConfig(**policy_config["mcts"])
            
            config = AdaptivePolicyConfig(
                initial_policy=policy_config.get("initial_policy", "heuristic"),
                target_policy=policy_config.get("target_policy", "neural"),
                transition_schedule=policy_config.get("transition_schedule", "linear"),
                transition_steps=policy_config.get("transition_steps", 10000),
                checkpoint_interval=policy_config.get("checkpoint_interval", 1000),
                heuristic_config=heuristic_config,
                mcts_config=mcts_config,
                min_neural_quality=policy_config.get("min_neural_quality", 0.5),
                random_seed=self.config.random_seed
            )
            return PolicyFactory.create_adaptive_policy(
                config, 
                self.config.network,
                self.config.training_progress_callback
            )
        
        else:
            raise ValueError(f"Unsupported policy type: {self.config.policy_type}")
    
    def add_progress_callback(self, callback: Callable[[RunnerStats], None]) -> None:
        """Add a progress callback function.
        
        Args:
            callback: Function to call with stats updates
        """
        self.progress_callbacks.append(callback)
    
    def run_games(self) -> List[GameRecord]:
        """Run the configured number of self-play games.
        
        Returns:
            List of completed game records
        """
        logger.info(f"Starting self-play with {self.config.target_games} target games")
        
        self.stats.start_time = time.time()
        self.stats.last_update_time = self.stats.start_time
        self.feature_history.clear()
        
        completed_games = []
        
        try:
            while self.stats.games_completed < self.config.target_games:
                # Run a batch of games
                batch_games = self._run_batch()
                completed_games.extend(batch_games)
                
                # Update statistics
                self._update_stats()
                
                # Save progress periodically
                if self.stats.games_completed % self.config.save_interval == 0:
                    self._save_progress(completed_games)
                
                # Log progress
                if self.stats.games_completed % self.config.progress_interval == 0:
                    self._log_progress()
                
                # Call progress callbacks
                self._notify_progress()
                
        except KeyboardInterrupt:
            logger.info("Self-play interrupted by user")
        except Exception as e:
            logger.error(f"Self-play failed: {e}")
            raise
        
        # Flush any remaining data
        if self.data_manager:
            self.data_manager.flush_storage()
        
        # Final statistics
        self._log_final_stats()
        
        return completed_games
    
    def _run_batch(self) -> List[GameRecord]:
        """Run a batch of games.
        
        Returns:
            List of completed game records from this batch
        """
        batch_size = min(
            self.config.max_games_per_batch,
            self.config.target_games - self.stats.games_completed
        )
        
        batch_games = []
        
        for i in range(batch_size):
            try:
                game_record = self._run_single_game()
                if game_record:
                    batch_games.append(game_record)
                    self.stats.games_completed += 1
                else:
                    self.stats.games_failed += 1
                    
            except Exception as e:
                logger.error(f"Game {self.stats.games_completed + 1} failed: {e}")
                self.stats.games_failed += 1
        
        return batch_games
    
    def _run_single_game(self) -> Optional[GameRecord]:
        """Run a single self-play game.
        
        Returns:
            Game record if successful, None otherwise
        """
        # Start game recording
        game_id = self.recorder.start_game()
        
        # Initialize game state
        game_state = GameState()
        
        turn_count = 0
        start_time = time.time()
        
        try:
            # Play the game
            while game_state.phase.value != GamePhase.GAME_OVER.value:
                # Safety check
                if turn_count >= self.config.max_turns_per_game:
                    logger.warning(f"Game {game_id} exceeded max turns ({self.config.max_turns_per_game})")
                    break
                
                # Get valid moves
                valid_moves = MoveGenerator.get_valid_moves(game_state.board, game_state)
                if not valid_moves:
                    logger.warning(f"Game {game_id} has no valid moves at turn {turn_count}")
                    break
                
                # Select move
                move = self.policy.select_move(game_state)
                if not move:
                    logger.warning(f"Game {game_id} failed to select move at turn {turn_count}")
                    break
                # Apply move
                player_before_move = game_state.current_player
                success = game_state.make_move(move)
                if not success:
                    logger.warning(f"Game {game_id} failed to apply move: {move}")
                    break
                
                turn_count += 1
                
                # Update turn count in stats
                self.stats.total_turns += 1

                turn = self.recorder.record_turn(game_state, move, player=player_before_move)
                if turn:
                    self.feature_history.append({
                        "game_id": game_id,
                        "turn_number": turn.turn_number,
                        "player": turn.current_player,
                        "features": turn.features,
                        "timestamp": turn.timestamp
                    })
            
            # End game recording
            winner = game_state.get_winner()
            logger.debug(f"Game {game_id} winner: {winner.value if winner is not None else None}")
            game_record = self.recorder.end_game(game_state, winner)
            logger.debug(f"Game {game_id} recording completed")
            
            if game_record:
                # Compute quality metrics if enabled
                if self.quality_analyzer:
                    quality_metrics = self.quality_analyzer.analyze_game(game_record)
                    self.stats.quality_metrics.append(quality_metrics)
                    logger.debug(f"Game {game_id} quality: diversity={quality_metrics.move_diversity:.3f}, "
                               f"coherence={quality_metrics.strategic_coherence:.3f}, "
                               f"tactical={quality_metrics.tactical_patterns}")
                
                # Store in parquet if enabled
                if self.data_manager:
                    logger.info(f"Game {game_id} storing in parquet")
                    validation_result = self.data_manager.store_game(game_record)
                    if not validation_result.valid:
                        logger.warning(f"Game {game_id} validation failed: {validation_result.errors}")
                    else:
                        logger.info(f"Game {game_id} stored successfully")
                
                # Update game-specific stats
                game_duration = time.time() - start_time
                self.stats.total_duration += game_duration
                
                # Increment game count for adaptive policy
                if hasattr(self.policy, 'increment_game_count'):
                    self.policy.increment_game_count()
                
                logger.debug(f"Completed game {game_id}: {turn_count} turns, "
                           f"{game_duration:.2f}s, winner: {winner.value if winner is not None else None}")
            
            return game_record
            
        except Exception as e:
            import traceback
            logger.error(f"Game {game_id} failed with exception: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Try to end the game recording even if it failed
            try:
                self.recorder.end_game(game_state, None)
            except:
                pass
            return None
    
    def _update_stats(self) -> None:
        """Update runner statistics."""
        current_time = time.time()
        
        if self.stats.games_completed > 0:
            self.stats.average_game_length = self.stats.total_turns / self.stats.games_completed
            self.stats.average_duration = self.stats.total_duration / self.stats.games_completed
        
        # Calculate games per hour
        elapsed_time = current_time - self.stats.start_time
        if elapsed_time > 0:
            self.stats.games_per_hour = (self.stats.games_completed * 3600) / elapsed_time
        
        # Update quality metrics averages
        if self.quality_analyzer and self.stats.quality_metrics:
            self.stats.average_move_diversity = (
                sum(m.move_diversity for m in self.stats.quality_metrics) / 
                len(self.stats.quality_metrics)
            )
            self.stats.average_strategic_coherence = (
                sum(m.strategic_coherence for m in self.stats.quality_metrics) / 
                len(self.stats.quality_metrics)
            )
            self.stats.average_tactical_patterns = (
                sum(m.tactical_patterns for m in self.stats.quality_metrics) / 
                len(self.stats.quality_metrics)
            )
        
        # Update storage and performance metrics
        self._update_storage_metrics()
        self._update_memory_metrics()
        
        self.stats.last_update_time = current_time
    
    def _update_storage_metrics(self) -> None:
        """Update storage-related performance metrics."""
        if self.data_manager and self.stats.games_completed > 0:
            try:
                storage_info = self.data_manager.get_storage_info()
                self.stats.data_storage_size_mb = storage_info.get('total_size_mb', 0.0)
                self.stats.storage_efficiency_mb_per_game = (
                    self.stats.data_storage_size_mb / self.stats.games_completed
                    if self.stats.games_completed > 0 else 0.0
                )
                
                # Calculate compression ratio if available
                if 'compression' in storage_info and storage_info['compression'] != 'none':
                    # Estimate uncompressed size (rough approximation)
                    estimated_uncompressed = self.stats.storage_efficiency_mb_per_game * 2.0
                    if estimated_uncompressed > 0:
                        self.stats.compression_ratio = (
                            self.stats.storage_efficiency_mb_per_game / estimated_uncompressed
                        )
            except Exception as e:
                logger.debug(f"Failed to update storage metrics: {e}")
    
    def _update_memory_metrics(self) -> None:
        """Update memory usage metrics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory_mb = memory_info.rss / (1024 * 1024)
            
            self.stats.memory_usage_mb = current_memory_mb
            if current_memory_mb > self.stats.peak_memory_usage_mb:
                self.stats.peak_memory_usage_mb = current_memory_mb
                
        except ImportError:
            logger.debug("psutil not available for memory tracking")
        except Exception as e:
            logger.debug(f"Failed to update memory metrics: {e}")
    
    def _log_progress(self) -> None:
        """Log current progress."""
        progress_pct = (self.stats.games_completed / self.config.target_games) * 100
        elapsed_time = self.stats.last_update_time - self.stats.start_time
        
        logger.info(f"Progress: {self.stats.games_completed}/{self.config.target_games} "
                   f"({progress_pct:.1f}%) - "
                   f"Games/hour: {self.stats.games_per_hour:.1f} - "
                   f"Avg duration: {self.stats.average_duration:.2f}s - "
                   f"Storage: {self.stats.data_storage_size_mb:.1f}MB - "
                   f"Memory: {self.stats.memory_usage_mb:.1f}MB")
    
    def _log_final_stats(self) -> None:
        """Log final statistics."""
        total_time = self.stats.last_update_time - self.stats.start_time
        
        logger.info("=== Self-Play Complete ===")
        logger.info(f"Games completed: {self.stats.games_completed}")
        logger.info(f"Games failed: {self.stats.games_failed}")
        logger.info(f"Total turns: {self.stats.total_turns}")
        logger.info(f"Total duration: {total_time:.2f}s")
        logger.info(f"Average game length: {self.stats.average_game_length:.1f} turns")
        logger.info(f"Average game duration: {self.stats.average_duration:.2f}s")
        logger.info(f"Games per hour: {self.stats.games_per_hour:.1f}")
        logger.info(f"Success rate: {(self.stats.games_completed / (self.stats.games_completed + self.stats.games_failed)) * 100:.1f}%")
        logger.info("=== Performance Metrics ===")
        logger.info(f"Data storage size: {self.stats.data_storage_size_mb:.1f}MB")
        logger.info(f"Storage efficiency: {self.stats.storage_efficiency_mb_per_game:.3f}MB/game")
        logger.info(f"Peak memory usage: {self.stats.peak_memory_usage_mb:.1f}MB")
        logger.info(f"Compression ratio: {self.stats.compression_ratio:.2f}")
        logger.info(f"Validation time: {self.stats.validation_time_seconds:.2f}s")
        
        # Quality metrics
        if self.quality_analyzer and self.stats.quality_metrics:
            logger.info("=== Quality Metrics ===")
            logger.info(f"Average move diversity: {self.stats.average_move_diversity:.3f}")
            logger.info(f"Average strategic coherence: {self.stats.average_strategic_coherence:.3f}")
            logger.info(f"Average tactical patterns: {self.stats.average_tactical_patterns:.1f}")
            
            # Comparison with baseline if available
            if self.config.quality_baseline_path:
                self._log_quality_comparison()
    
    def _save_progress(self, completed_games: List[GameRecord]) -> None:
        """Save progress to disk.
        
        Args:
            completed_games: List of completed game records
        """
        try:
            # Save runner state
            state_file = Path(self.config.output_dir) / "runner_state.json"
            state_data = {
                "config": self.config.__dict__,
                "stats": self.stats.__dict__,
                "timestamp": time.time()
            }
            
            import json
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.debug(f"Saved progress to {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
    
    def _notify_progress(self) -> None:
        """Notify progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(self.stats)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")

    def get_feature_history(self) -> List[Dict[str, Any]]:
        """Return the runner-level feature history collected so far."""
        return list(self.feature_history)
    
    def get_current_stats(self) -> RunnerStats:
        """Get current runner statistics.
        
        Returns:
            Current statistics
        """
        self._update_stats()
        return self.stats
    
    def load_progress(self) -> bool:
        """Load progress from disk.
        
        Returns:
            True if progress was loaded, False otherwise
        """
        try:
            state_file = Path(self.config.output_dir) / "runner_state.json"
            if not state_file.exists():
                return False
            
            import json
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore statistics
            self.stats = RunnerStats(**state_data["stats"])
            
            logger.info(f"Loaded progress: {self.stats.games_completed} games completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            return False
    
    def export_results(self, output_file: str) -> None:
        """Export all game results to CSV.
        
        Args:
            output_file: Path to output CSV file
        """
        self.recorder.export_to_csv(output_file)
        logger.info(f"Exported results to {output_file}")
    
    def get_game_statistics(self) -> Dict[str, Any]:
        """Get statistics about recorded games.
        
        Returns:
            Dictionary with game statistics
        """
        stats = self.recorder.get_statistics()
        
        # Add parquet storage stats if available
        if self.data_manager:
            storage_stats = self.data_manager.get_storage_info()
            stats['parquet_storage'] = storage_stats
        
        return stats
    
    def _log_quality_comparison(self) -> None:
        """Log quality comparison with baseline dataset."""
        try:
            from pathlib import Path
            baseline_path = Path(self.config.quality_baseline_path)
            
            # Load baseline games (simplified - assumes JSON files)
            baseline_games = []
            if baseline_path.is_dir():
                for json_file in baseline_path.glob("*.json"):
                    game_record = self.recorder.load_game_record(json_file.stem)
                    if game_record:
                        baseline_games.append(game_record)
            
            if baseline_games:
                # Get current games
                current_games = []
                for game_id in self.recorder.list_game_records():
                    game_record = self.recorder.load_game_record(game_id)
                    if game_record:
                        current_games.append(game_record)
                
                if current_games:
                    comparison = self.quality_analyzer.compare_datasets(baseline_games, current_games)
                    logger.info("=== Quality Comparison vs Baseline ===")
                    logger.info(f"Length improvement: {comparison.length_improvement:+.1f}%")
                    logger.info(f"Diversity improvement: {comparison.diversity_improvement:+.1f}%")
                    logger.info(f"Coherence improvement: {comparison.coherence_improvement:+.1f}%")
                    logger.info(f"Tactical patterns improvement: {comparison.tactical_improvement:+.1f}%")
                    logger.info(f"Overall quality score: {comparison.overall_quality_score:.3f}")
        except Exception as e:
            logger.warning(f"Failed to generate quality comparison: {e}")
    
    def validate_stored_data(self) -> Dict[str, Any]:
        """Validate all stored data.
        
        Returns:
            Dictionary with validation results
        """
        if not self.data_manager:
            return {'error': 'Parquet storage not enabled'}
        
        validation_result = self.data_manager.validate_stored_data()
        return {
            'valid': validation_result.valid,
            'errors': validation_result.errors,
            'warnings': validation_result.warnings,
            'stats': validation_result.stats
        }
