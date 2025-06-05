# training/supervisor.py

import logging
from pathlib import Path
import time
from typing import Optional, List, Tuple, Dict
import numpy as np
import json
import psutil, platform
from collections import defaultdict
import math
import torch

# --- YINSH ML Imports ---
from ..network.wrapper import NetworkWrapper
from .self_play import SelfPlay
from .trainer import YinshTrainer
# from ..utils.visualization import TrainingVisualizer # Keep if used, remove otherwise
from ..utils.encoding import StateEncoder
from ..game.constants import Player, PieceType
from ..game.game_state import GameState
from ..utils.metrics_manager import TrainingMetrics # Keep if used for internal tracking
from ..utils.tournament import ModelTournament, _canon # Import _canon if used directly
# --- Memory Pool Imports ---
from ..memory import (
    GameStatePool, TensorPool, 
    GameStatePoolConfig, TensorPoolConfig
)

# --- Logging Setup ---
# Configure logger for this module
logger = logging.getLogger("TrainingSupervisor")
# Note: The level is often set by the root logger configuration in the entry point (runner/test_experiments)

# Make sure root logger is configured if running supervisor standalone
if not logging.getLogger().hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TrainingSupervisor:
    def __init__(self,
                 network: NetworkWrapper,
                 save_dir: str,
                 device: str = 'cpu',
                 tournament_games: int = 20, # Games per match-up in tournament
                 # --- Core MCTS Setting ---
                 # This defines the *early* simulation budget.
                 mcts_simulations: int = 100,
                 # --- All other config settings arrive here ---
                 mode_settings: Optional[Dict] = None
                ):
        """
        Supervises the full training loop, configured via direct args and mode_settings.

        Args:
            network: The NetworkWrapper instance to manage.
            save_dir: Base directory for saving checkpoints, logs for this run.
            device: Computation device ('cuda', 'mps', 'cpu').
            tournament_games: Number of games per head-to-head match in tournaments.
            mcts_simulations: Base number of MCTS simulations (used for early game).
            mode_settings: Dictionary containing all other configuration parameters
                           from the CombinedConfig (e.g., learning rates, other MCTS params,
                           temperature settings, loss weights, etc.).
        """
        self.network = network
        self.save_dir = Path(save_dir)
        self.device = device
        self.tournament_games = tournament_games
        self.logger = logging.getLogger("TrainingSupervisor") # Use module logger

        if mode_settings is None:
            mode_settings = {}
        self.mode_settings = mode_settings  # <<< STORE mode_settings AS INSTANCE VARIABLE

        self.logger.info(f"Initializing TrainingSupervisor in '{self.save_dir}'")
        self.logger.info(f"Base MCTS Simulations (early game): {mcts_simulations}")
        # Log received mode_settings for debugging
        # Be careful logging sensitive info if any exists in config
        # self.logger.debug(f"Received mode_settings: {mode_settings}")

        # ==================================================================
        # 1. Initialize Memory Management Pools
        # ==================================================================
        self._initialize_memory_pools(mcts_simulations)

        # ==================================================================
        # 2. Extract Parameters from mode_settings with Defaults
        # ==================================================================
        # --- SelfPlay / MCTS specific ---
        late_simulations = self.mode_settings.get('late_simulations', None)
        if late_simulations is None:
            late_simulations = mcts_simulations
            self.logger.info(f"Late MCTS simulations not specified, using early game value: {late_simulations}")
        else:
             self.logger.info(f"Late MCTS simulations (after ply {mode_settings.get('simulation_switch_ply', 20)}): {late_simulations}")

        simulation_switch_ply = self.mode_settings.get('simulation_switch_ply', 20)
        c_puct = self.mode_settings.get('c_puct', 1.0)
        dirichlet_alpha = self.mode_settings.get('dirichlet_alpha', 0.3)
        # *** Naming Consistency: Use 'value_weight' from config for MCTS ***
        # MCTS's internal parameter should probably be just 'value_weight' too.
        # For now, we pass the config value to the MCTS arg named 'initial_value_weight'.
        # We will rename the MCTS arg later.
        mcts_value_weight = self.mode_settings.get('value_weight', 1.0)
        max_depth = self.mode_settings.get('max_depth', 500)

        # --- Temperature specific ---
        initial_temp = self.mode_settings.get('initial_temp', 1.0)
        final_temp = self.mode_settings.get('final_temp', 0.1)
        annealing_steps = self.mode_settings.get('annealing_steps', 30)
        temp_schedule = self.mode_settings.get('temp_schedule', 'linear')
        temp_clamp_fraction = self.mode_settings.get('temp_clamp_fraction', 0.6)

        # --- Trainer specific ---
        # Learning Rate: Base LR is handled by runner creating optimizer maybe? Or should trainer handle it?
        # Assuming trainer handles base LR setup, we only need factors/weights here.
        # Let's fetch all relevant trainer params from mode_settings.
        trainer_batch_size = self.mode_settings.get('batch_size', 256)
        trainer_l2_reg = self.mode_settings.get('l2_reg', 0.0)
        value_head_lr_factor = self.mode_settings.get('value_head_lr_factor', 5.0)
        value_loss_weights = self.mode_settings.get('value_loss_weights', (0.5, 0.5))
        base_lr = self.mode_settings.get('lr', 0.001)
        lr_schedule = self.mode_settings.get('lr_schedule', 'constant')
        warmup_steps = self.mode_settings.get('warmup_steps', 0)

        # ==================================================================
        # 2. Instantiate Components with Extracted Parameters
        # ==================================================================

        _mcts_config_for_init = { # Use temporary name to avoid confusion
            'num_simulations': mcts_simulations, # Early sims from direct arg
            'late_simulations': late_simulations,
            'simulation_switch_ply': simulation_switch_ply,
            'c_puct': c_puct,
            'dirichlet_alpha': dirichlet_alpha,
            # *** THIS IS THE PROBLEM LINE ***
            # It should pass 'value_weight', not 'initial_value_weight'
            'value_weight': mcts_value_weight, # <<< CHANGE THIS KEY
            'max_depth': max_depth,
            'initial_temp': initial_temp,
            'final_temp': final_temp,
            'annealing_steps': annealing_steps,
            'temp_clamp_fraction': temp_clamp_fraction,
        }

        self.self_play = SelfPlay(
            network=self.network,
            num_workers=self._compute_num_workers(),
            game_state_pool=self.game_state_pool,  # Pass memory pool
            **_mcts_config_for_init # Pass the explicitly constructed dict
            # metrics_logger=... , mcts_metrics=... # Add if needed
        )

        self.logger.info(f"SelfPlay Initialized: Early Sims={mcts_simulations}, Late Sims={late_simulations}, Switch Ply={simulation_switch_ply}, cPUCT={c_puct}, Alpha={dirichlet_alpha}, ValueWeight={mcts_value_weight}")
        self.logger.info(f"Temperature Initialized: Initial={initial_temp}, Final={final_temp}, Anneal Steps={annealing_steps}, Clamp Frac={temp_clamp_fraction}")


        # Ensure the replay buffer path is within the run's save directory
        replay_buffer_file = self.save_dir / "replay_buffer.pkl"

        self.trainer = YinshTrainer(
            network=self.network, # Pass the managed network
            device=self.device,
            batch_size=trainer_batch_size, # Pass batch size from config
            l2_reg=trainer_l2_reg, # Pass L2 reg from config
            value_head_lr_factor=value_head_lr_factor, # Pass factor from config
            value_loss_weights=value_loss_weights, # Pass weights from config
            replay_buffer_path=str(replay_buffer_file), # Pass path for persistence
            # --- Pass LR info if Trainer sets up optimizers/schedulers ---
            # base_lr = base_lr,
            # lr_schedule = lr_schedule,
            # warmup_steps = warmup_steps
            # metrics_logger=self.metrics_logger, # If using a shared logger instance
        )
        # --- Configure Trainer Optimizers/Schedulers based on mode_settings ---
        # This assumes YinshTrainer exposes methods or attributes to configure these
        # If Trainer creates optimizers in __init__, we need to pass parameters there.
        # Let's assume Trainer's __init__ handles this based on passed params (like base_lr, value_head_lr_factor, etc.)
        # We already passed necessary factors/weights above. If base_lr needs explicit setting:
        if hasattr(self.trainer, 'policy_optimizer') and hasattr(self.trainer, 'value_optimizer'):
            self.trainer.policy_optimizer.param_groups[0]['lr'] = base_lr
            self.trainer.value_optimizer.param_groups[0]['lr'] = base_lr * value_head_lr_factor
            # Re-initialize schedulers if LR changes significantly or schedule type changes
            # self.trainer.reinitialize_schedulers(lr_schedule, warmup_steps, ...) # Hypothetical method
            self.logger.info(f"Trainer optimizers configured: Base LR={base_lr}, Value Factor={value_head_lr_factor}")
        else:
             self.logger.warning("Could not find optimizers on Trainer to configure LR directly. Assuming Trainer handles it internally.")


        # self.visualizer = TrainingVisualizer() # Keep if used
        self.state_encoder = StateEncoder() # Useful for decoding states if needed
        self.metrics = TrainingMetrics() # Keep for internal aggregation if used

        self.tournament_manager = ModelTournament(
            training_dir=self.save_dir, # Tournaments evaluate models within this run's directory
            device=self.device,
            games_per_match=self.tournament_games # Use the value passed to supervisor
        )
        self.logger.info(f"Tournament Manager Initialized: Games/Match={self.tournament_games}")


        # --- Best Model Tracking ---
        self.best_model_elo: float = -float('inf')
        self.best_model_iteration: int = -1
        self.best_model_path: Optional[Path] = None
        # Save best model directly in the run directory for simplicity
        self.best_model_save_path: Path = self.save_dir / "best_model.pt"
        self._iteration_counter: int = 0 # Initialize iteration counter

        self._load_best_model_state() # Load previous state if exists for this run directory
        self.logger.info("=== Training Supervisor Initialized ===")
        if self.best_model_path:
            self.logger.info(f"Loaded best model state: Iteration {self.best_model_iteration}, ELO {self.best_model_elo:.1f}")
            # Load best model weights into the network managed by the supervisor
            if self.best_model_path.exists():
                try:
                    self.logger.info(f"Loading weights from best model path: {self.best_model_path}")
                    self.network.load_model(str(self.best_model_path))
                    self.logger.info("Successfully loaded best model weights.")
                except Exception as e:
                    self.logger.error(f"Failed to load best model weights from {self.best_model_path}: {e}. Using initial network weights.", exc_info=True)
            else:
                self.logger.warning(f"Best model path {self.best_model_path} not found. Using initial network weights.")
                self._reset_best_model_state() # Reset tracking if path is invalid

    def _initialize_memory_pools(self, mcts_simulations: int):
        """Initialize memory pools for efficient resource management."""
        self.logger.info("Initializing memory management pools...")
        
        # Calculate pool sizes based on MCTS simulations and worker count
        num_workers = self._compute_num_workers()
        
        # GameStatePool: Size based on MCTS simulations per worker
        # Each worker needs states for simulations + some buffer
        game_state_pool_size = max(100, mcts_simulations * 2)  # At least 100, or 2x simulations
        game_state_config = GameStatePoolConfig(
            initial_size=game_state_pool_size,
            max_capacity=game_state_pool_size * 3,  # Allow 3x growth
            mcts_batch_size=max(10, mcts_simulations // 10),  # Reasonable batch size
            enable_statistics=True,  # Enable stats for monitoring
            training_mode=True  # Enable training optimizations
        )
        self.game_state_pool = GameStatePool(game_state_config)
        
        # TensorPool: Size based on network operations
        # Need tensors for input states, policy outputs, value outputs
        tensor_pool_size = max(50, num_workers * 10)  # At least 50, or 10 per worker
        tensor_config = TensorPoolConfig(
            initial_size=tensor_pool_size,
            max_capacity=tensor_pool_size * 2,  # Allow 2x growth
            enable_statistics=True,  # Enable stats for monitoring
            auto_device_selection=True,  # Enable device-specific pooling
            enable_adaptive_sizing=True  # Enable adaptive pool sizing
        )
        self.tensor_pool = TensorPool(tensor_config)
        
        # Update network to use the tensor pool if it doesn't have one
        if not hasattr(self.network, 'tensor_pool') or self.network.tensor_pool is None:
            self.network.tensor_pool = self.tensor_pool
            self.network._pool_enabled = True
        
        self.logger.info(f"Memory pools initialized:")
        self.logger.info(f"  GameStatePool: {game_state_pool_size} initial, {game_state_pool_size * 3} max")
        self.logger.info(f"  TensorPool: {tensor_pool_size} initial, {tensor_pool_size * 2} max")
        self.logger.info(f"  Workers: {num_workers}")

    def cleanup_memory_pools(self):
        """Clean up memory pools and release all resources."""
        self.logger.info("Cleaning up memory pools...")
        
        if hasattr(self, 'game_state_pool') and self.game_state_pool is not None:
            # Log statistics before cleanup
            if hasattr(self.game_state_pool, 'get_statistics'):
                stats = self.game_state_pool.get_statistics()
                self.logger.info(f"GameStatePool final stats: {stats}")
            self.game_state_pool = None
            
        if hasattr(self, 'tensor_pool') and self.tensor_pool is not None:
            # Log statistics before cleanup
            if hasattr(self.tensor_pool, 'get_statistics'):
                stats = self.tensor_pool.get_statistics()
                self.logger.info(f"TensorPool final stats: {stats}")
            self.tensor_pool = None
            
        self.logger.info("Memory pools cleaned up")

    def train_iteration(self, num_games: int, epochs: int):
        """
        One full self-play -> training -> evaluation -> model-selection cycle.
        Uses configuration parameters stored in self.self_play and self.trainer.
        Memory-optimized version to reduce RAM usage.

        Args:
            num_games: Number of self-play games to generate for this iteration.
            epochs: Number of training epochs for this iteration.
        """
        # Initialize memory monitoring
        try:
            import psutil
            process = psutil.Process()
            initial_memory_mb = process.memory_info().rss / (1024 * 1024)
            self.logger.info(f"Initial memory usage: {initial_memory_mb:.1f} MB")
        except:
            initial_memory_mb = 0

        # Memory pools handle resource management now - no need for manual GC
        current_iteration = self._iteration_counter
        self.logger.info(f"\n{'=' * 15} Starting Iteration {current_iteration} {'=' * 15}")

        # Iteration-specific directory within the main save_dir
        iteration_dir = self.save_dir / f"iteration_{current_iteration}"
        iteration_dir.mkdir(exist_ok=True, parents=True)

        # ------------------------------------------------------------------ #
        # 1. SELF-PLAY (MEMORY-OPTIMIZED)
        # ------------------------------------------------------------------ #
        # Ensure self_play uses the *current* network weights (potentially reverted from previous iter)
        self.self_play.network = self.network
        # Pass the number of games for this iteration
        self.self_play.current_iteration = current_iteration

        self.logger.info(f"Generating {num_games} self-play games...")
        # Log MCTS parameters being used by SelfPlay for this iteration for verification
        self.logger.info(f"[SelfPlay Config] Early Sims: {self.self_play.mcts.early_simulations}, "
                         f"Late Sims: {self.self_play.mcts.late_simulations}, "
                         f"Switch Ply: {self.self_play.mcts.switch_ply}, "
                         f"cPUCT: {self.self_play.mcts.c_puct}, "
                         f"Alpha: {self.self_play.mcts.dirichlet_alpha}, "
                         f"Value Weight: {self.self_play.mcts.value_weight}")

        t0 = time.time()

        # MEMORY-OPTIMIZATION: Process games in batches to reduce peak memory usage
        batch_size = min(25, num_games)  # Generate in smaller batches
        games = []

        for batch_start in range(0, num_games, batch_size):
            batch_end = min(batch_start + batch_size, num_games)
            batch_size_actual = batch_end - batch_start

            self.logger.info(f"Generating game batch {batch_start + 1}-{batch_end} of {num_games}")

            # Generate batch of games
            batch_games = self.self_play.generate_games(num_games=batch_size_actual)
            games.extend(batch_games)

            # Log progress and memory usage
            try:
                if initial_memory_mb > 0:
                    current_memory_mb = process.memory_info().rss / (1024 * 1024)
                    self.logger.info(
                        f"Memory after batch: {current_memory_mb:.1f} MB (Change: {current_memory_mb - initial_memory_mb:+.1f} MB)")
            except:
                pass

            self.logger.info(f"Completed {len(games)}/{num_games} games")

        game_time = time.time() - t0

        if not games:
            # Handle case where no games are generated (e.g., worker errors)
            self.logger.error("Self-play generated zero games! Skipping training and evaluation for this iteration.")
            # Increment counter and return an empty summary or raise error
            self._iteration_counter += 1
            return {'error': 'No games generated'}

        self.logger.info(f"Generated {len(games)} games in {game_time:.1f}s")

        # --- Basic game stats ---
        game_lengths = [len(g[0]) for g in games if g[0]]
        avg_game_len = np.mean(game_lengths) if game_lengths else 0

        # MEMORY-OPTIMIZATION: Process last states more efficiently
        last_states_decoded = []
        for game_idx, game_data in enumerate(games):
            try:
                if len(game_data) >= 4 and isinstance(game_data[3], list) and game_data[3]:
                    if 'state' in game_data[3][-1] and isinstance(game_data[3][-1]['state'], GameState):
                        last_states_decoded.append(game_data[3][-1]['state'])
                    elif game_data[0]:
                        last_states_decoded.append(self.state_encoder.decode_state(game_data[0][-1]))
                elif game_data[0]:
                    last_states_decoded.append(self.state_encoder.decode_state(game_data[0][-1]))

                # MEMORY-OPTIMIZATION: Process every 10 games to avoid memory buildup
                if game_idx % 10 == 9:
                    pass  # Memory pools handle cleanup automatically
            except Exception as e:
                self.logger.warning(f"Error processing game {game_idx} for stats: {e}")

        ring_mobility_list = [self._calculate_ring_mobility_from_state(gs) for gs in last_states_decoded]
        avg_ring_mobility = np.mean(ring_mobility_list) if ring_mobility_list else 0.0

        outcomes = [g[2] for g in games]
        pseudo_wins_white = sum(1 for o in outcomes if o > 0.1)
        pseudo_wins_black = sum(1 for o in outcomes if o < -0.1)
        pseudo_draws = len(outcomes) - pseudo_wins_white - pseudo_wins_black
        pseudo_win_rate = (pseudo_wins_white + pseudo_wins_black) / len(outcomes) if outcomes else 0

        self.logger.info(
            f"Self-Play Stats: avg_len={avg_game_len:.1f} | W/B/D (estimated) = {pseudo_wins_white}/{pseudo_wins_black}/{pseudo_draws}")

        # ------------------------------------------------------------------ #
        # 2. ADD EXPERIENCE & TRAIN (MEMORY-OPTIMIZED)
        # ------------------------------------------------------------------ #
        # MEMORY-OPTIMIZATION: Clear some memory before adding experience
        last_states_decoded = None
        # Memory pools handle cleanup automatically

        # Ensure trainer uses the current network
        self.trainer.network = self.network

        # MEMORY-OPTIMIZATION: Process games in smaller chunks to limit peak memory
        chunk_size = 10  # Process 10 games at a time
        total_games = len(games)

        self.logger.info(f"Adding game experience in {(total_games + chunk_size - 1) // chunk_size} chunks...")

        for i in range(0, total_games, chunk_size):
            chunk = games[i:i + chunk_size]
            chunk_end = min(i + chunk_size, total_games)
            self.logger.info(f"Processing game chunk {i + 1}-{chunk_end} of {total_games}")

            # Process each game in the chunk
            for game_data in chunk:
                # Unpack game data (assuming format: states, policies, outcome, history)
                states, policies, outcome, game_history = game_data
                if not states or not policies:
                    self.logger.warning("Skipping empty game data.")
                    continue

                try:
                    # Decode last state to get scores for add_game_experience
                    if game_history and 'state' in game_history[-1]:
                        final_state = game_history[-1]['state']
                        if isinstance(final_state, GameState):
                            final_scores = (final_state.white_score, final_state.black_score)
                        else:
                            decoded_state = self.state_encoder.decode_state(final_state)
                            final_scores = (decoded_state.white_score, decoded_state.black_score)
                    else:
                        decoded_state = self.state_encoder.decode_state(states[-1])
                        final_scores = (decoded_state.white_score, decoded_state.black_score)

                    self.trainer.add_game_experience(states, policies, final_scores)
                except Exception as e:
                    self.logger.error(
                        f"Error adding game experience: {e}. State/Policy lengths: {len(states)}/{len(policies)}",
                        exc_info=True)

            # MEMORY-OPTIMIZATION: Force garbage collection after each chunk
            # Also free the chunk data itself
            chunk = None
            # Memory pools handle cleanup automatically

            # Log memory usage after each chunk
            try:
                if initial_memory_mb > 0:
                    current_memory_mb = process.memory_info().rss / (1024 * 1024)
                    self.logger.info(
                        f"Memory after chunk: {current_memory_mb:.1f} MB (Change: {current_memory_mb - initial_memory_mb:+.1f} MB)")
            except:
                pass

        # MEMORY-OPTIMIZATION: Free up games data now that it's been processed
        games = None
        # Memory pools handle cleanup automatically

        # Train the network using data in the replay buffer
        self.logger.info(f"Training network for {epochs} epochs...")
        t1 = time.time()
        self.trainer.policy_losses.clear()
        self.trainer.value_losses.clear()
        avg_epoch_stats = defaultdict(list)

        # MEMORY-OPTIMIZATION: Calculate optimal batch count
        buffer_size = self.trainer.experience.size()

        for epoch in range(epochs):
            self.logger.info(f"Starting Epoch {epoch + 1}/{epochs}")

            if buffer_size < self.trainer.batch_size:
                self.logger.warning(
                    f"Replay buffer size ({buffer_size}) is smaller than batch size ({self.trainer.batch_size}). Skipping training epoch.")
                continue

            # MEMORY-OPTIMIZATION: Use optimal batch count based on buffer size
            optimal_batches = max(10, (buffer_size // self.trainer.batch_size) // 2)
            # Use config if provided, otherwise use calculated optimal batches
            batches_per_epoch = self.mode_settings.get('batches_per_epoch', optimal_batches)
            actual_batches = min(batches_per_epoch, optimal_batches)

            self.logger.info(
                f"Using {actual_batches} batches (buffer size: {buffer_size}, batch size: {self.trainer.batch_size})")

            epoch_stats = self.trainer.train_epoch(
                batch_size=self.trainer.batch_size,
                batches_per_epoch=actual_batches
            )

            # Aggregate stats
            for key, value in epoch_stats.items():
                if isinstance(value, dict):
                    if not isinstance(avg_epoch_stats[key], list):
                        avg_epoch_stats[key] = [defaultdict(list) for _ in range(len(value))]
                    # Skip complex aggregation
                    pass
                elif isinstance(value, (int, float)):
                    avg_epoch_stats[key].append(value)

            # MEMORY-OPTIMIZATION: Force garbage collection between epochs
            if epoch < epochs - 1:  # Skip on last epoch
                pass  # Memory pools handle cleanup automatically

            # Log memory usage every other epoch
            if epoch % 2 == 0:
                try:
                    if initial_memory_mb > 0:
                        current_memory_mb = process.memory_info().rss / (1024 * 1024)
                        self.logger.info(
                            f"Memory after epoch {epoch + 1}: {current_memory_mb:.1f} MB (Change: {current_memory_mb - initial_memory_mb:+.1f} MB)")
                except:
                    pass

        train_time = time.time() - t1

        # Calculate average losses over the epochs trained
        final_pol_loss = np.mean(self.trainer.policy_losses) if self.trainer.policy_losses else 0
        final_val_loss = np.mean(self.trainer.value_losses) if self.trainer.value_losses else 0

        self.logger.info(
            f"Training done in {train_time:.1f}s (Avg Policy Loss={final_pol_loss:.4f}, Avg Value Loss={final_val_loss:.4f})")

        # ------------------------------------------------------------------ #
        # 3. SAVE CANDIDATE CHECKPOINT (MEMORY-OPTIMIZED)
        # ------------------------------------------------------------------ #
        # MEMORY-OPTIMIZATION: Force garbage collection before saving
        pass  # Memory pools handle cleanup automatically

        checkpoint_path = iteration_dir / f"checkpoint_iteration_{current_iteration}.pt"
        self.network.save_model(str(checkpoint_path))
        self.logger.info(f"Candidate checkpoint saved to {checkpoint_path}")

        # MEMORY-OPTIMIZATION: Check checkpoint file size
        try:
            import os
            checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            self.logger.info(f"Checkpoint file size: {checkpoint_size_mb:.1f} MB")
        except:
            pass

        # ------------------------------------------------------------------ #
        # 4. TOURNAMENT EVALUATION
        # ------------------------------------------------------------------ #
        # The rest of the code remains mostly unchanged as tournament evaluation
        # typically doesn't have significant memory issues

        self.logger.info("Running tournament evaluation...")

        try:
            self.tournament_manager.run_full_round_robin_tournament(current_iteration)
            model_id_canon = _canon(str(checkpoint_path))
            self.logger.info(f"Fetching tournament performance for canonical ID: {model_id_canon}")
            tournament_stats = self.tournament_manager.get_model_performance(model_id_canon) or {}
            current_elo = tournament_stats.get('current_rating', self.tournament_manager.glicko_tracker.initial_rating)
            tourn_win_rate = tournament_stats.get('win_rate', 0.0)
            self.logger.info(
                f"Tournament Results for {model_id_canon}: Elo={current_elo:.1f}, Win Rate={tourn_win_rate:.1%}")

        except FileNotFoundError as e:
            self.logger.error(f"Tournament failed: Could not find a model checkpoint. Check paths. Error: {e}",
                              exc_info=True)
            current_elo = self.best_model_elo
            tourn_win_rate = 0.0
            tournament_stats = {}
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during tournament: {e}", exc_info=True)
            current_elo = self.best_model_elo
            tourn_win_rate = 0.0
            tournament_stats = {}

        # ------------------------------------------------------------------ #
        # 5. PROMOTION / REVERSION (using Wilson Gate)
        # ------------------------------------------------------------------ #
        candidate_iteration = current_iteration
        candidate_id_canon = _canon(str(checkpoint_path))

        tournament_stats = self.tournament_manager.get_model_performance(candidate_id_canon) or {}
        candidate_elo = tournament_stats.get('current_rating', self.tournament_manager.glicko_tracker.initial_rating)

        wins, total = 0, 0
        perform_wilson_check = False

        # --- Check if comparison is needed and possible ---
        if self.best_model_iteration >= 0 and self.best_model_iteration != candidate_iteration and self.best_model_path:
            best_id_canon = _canon(str(self.best_model_path))
            self.logger.info(f"Comparing Candidate ({candidate_id_canon}) vs Best ({best_id_canon}) for Wilson gate.")
            try:
                wins, total = self.tournament_manager.get_head_to_head(candidate_id_canon, best_id_canon)
                if total > 0:
                    perform_wilson_check = True
                    self.logger.info(f"Head-to-head results found: Candidate Wins={wins}, Total Games={total}")
                else:
                    self.logger.warning(
                        f"No head-to-head games recorded between {candidate_id_canon} and {best_id_canon}. Cannot run Wilson gate.")
            except Exception as e:
                self.logger.error(f"Error getting head-to-head results: {e}", exc_info=True)

        elif self.best_model_iteration < 0:
            self.logger.info("No previous best model recorded. Wilson gate skipped.")
        else:
            self.logger.info(
                f"Candidate ({candidate_id_canon}) is the current best model ({_canon(str(self.best_model_path))}). Wilson gate skipped.")

        # --- Run Wilson Gate if applicable ---
        promote_by_wilson = False
        if perform_wilson_check:
            wilson_threshold = self.mode_settings.get('promotion_threshold', 0.55)
            promote_by_wilson = self._should_promote(wins, total, threshold=wilson_threshold)
            self.logger.info(
                f"Wilson Gate Check: Wins={wins}, Total={total}, LB={self._wilson_lower_bound(wins, total):.3f}, Threshold={wilson_threshold} -> {'PROMOTE' if promote_by_wilson else 'REJECT'}")

        # --- Final Promotion Decision Logic ---
        promote = False
        reverted = False
        kept_current_best = (self.best_model_iteration == candidate_iteration)

        if promote_by_wilson:
            promote = True
            self.logger.info(f"✅ PROMOTED: Iter {candidate_iteration} ({candidate_id_canon}) passed Wilson gate.")
        elif self.best_model_iteration < 0:
            promote = True
            self.logger.info(f"✅ PROMOTED: Iter {candidate_iteration} ({candidate_id_canon}) is the first model.")
        elif candidate_elo > self.best_model_elo:
            promote = True
            reason = f"Elo improved ({candidate_elo:.1f} > {self.best_model_elo:.1f})"
            if perform_wilson_check and not promote_by_wilson:
                reason = f"Wilson failed ({wins}/{total}) but Elo improved"
            self.logger.info(f"✅ PROMOTED: Iter {candidate_iteration} ({candidate_id_canon}) - {reason}.")

        # --- Apply Decision ---
        if promote:
            self.best_model_elo = candidate_elo
            self.best_model_iteration = candidate_iteration
            self.best_model_path = checkpoint_path
            if self.best_model_path.exists():
                self.network.save_model(str(self.best_model_save_path))
                self.logger.info(f"Copied {self.best_model_path.name} to {self.best_model_save_path.name}")
            else:
                self.logger.error(f"Cannot copy best model: Source checkpoint {self.best_model_path} does not exist!")
            self._save_best_model_state()

        elif kept_current_best:
            self.logger.info(f" Kandidat ({candidate_id_canon}) is already the best model. Keeping current state.")
            self._save_best_model_state()

        else:
            self.logger.info(f"🚫 REJECTED: Candidate Iter {candidate_iteration} ({candidate_id_canon}) not promoted.")
            if self.best_model_path and self.best_model_path.exists():
                self.logger.info(
                    f"Reverting network weights to previous best: {self.best_model_path.name} (Iter {self.best_model_iteration})")
                try:
                    self.network.load_model(str(self.best_model_path))
                    reverted = True
                except Exception as e:
                    self.logger.error(
                        f"Failed to revert weights from {self.best_model_path}: {e}. Network weights remain as rejected candidate.",
                        exc_info=True)
            else:
                self.logger.warning(
                    f"Cannot revert: Previous best model path invalid or not found ({self.best_model_path}). Keeping rejected candidate weights.")
            self._save_best_model_state()

        # Determine active network weights for clarity
        active_iter = self.best_model_iteration if (promote or reverted or kept_current_best) else candidate_iteration
        self.logger.info(f"Active network weights in memory correspond to iteration {active_iter}")

        # ------------------------------------------------------------------ #
        # 6. METRICS, VISUALS, HOUSE-KEEPING (MEMORY-OPTIMIZED)
        # ------------------------------------------------------------------ #

        if hasattr(self.tournament_manager, 'glicko_tracker'):
            # Force garbage collection
            pass  # Memory pools handle cleanup automatically

        # MEMORY-OPTIMIZATION: Final garbage collection
        pass  # Memory pools handle cleanup automatically

        # Log final memory usage
        try:
            if initial_memory_mb > 0:
                final_memory_mb = process.memory_info().rss / (1024 * 1024)
                self.logger.info(
                    f"Final memory usage: {final_memory_mb:.1f} MB (Change: {final_memory_mb - initial_memory_mb:+.1f} MB)")
        except:
            pass

        # Aggregate metrics for this iteration
        self.metrics.add_iteration_metrics(
            avg_game_length=avg_game_len,
            avg_ring_mobility=avg_ring_mobility,
            win_rate=pseudo_win_rate,
            draw_rate=pseudo_draws / len(outcomes) if outcomes else 0,
            policy_loss=final_pol_loss,
            value_loss=final_val_loss,
            tournament_rating=current_elo,
            tournament_win_rate=tourn_win_rate
        )
        self._save_metrics(iteration_dir)

        # --- Advance counter for the next iteration ---
        self._iteration_counter += 1

        self.prune_old_experiences()
        self.clear_pytorch_memory()

        # Every 2-3 iterations, reset network objects completely
        if self._iteration_counter % 3 == 0:
            self._reset_network_objects()

        # MEMORY OPTIMIZATION: Prune old experiences to prevent memory growth
        self.prune_old_experiences()  # Add this line here

        # ------------------------------------------------------------------ #
        # 7. RETURN SUMMARY
        # ------------------------------------------------------------------ #
        return {
            'iteration': current_iteration,
            'training_games': {
                'pseudo_wins_white': pseudo_wins_white,
                'pseudo_wins_black': pseudo_wins_black,
                'pseudo_draws': pseudo_draws,
                'win_rate': pseudo_win_rate,
                'avg_game_length': avg_game_len
            },
            'training': {
                'game_time': game_time,
                'training_time': train_time,
                'policy_loss': final_pol_loss,
                'value_loss': final_val_loss
            },
            'evaluation': {
                'tournament': {
                    'rating': current_elo,
                    'win_rate': tourn_win_rate,
                    'raw_stats': tournament_stats
                }
            },
            'model_selection': {
                'best_iteration': self.best_model_iteration,
                'best_elo': self.best_model_elo,
                'reverted_to_best': reverted,
                'active_network_iteration': active_iter
            }
        }

    # --- Helper Methods ---
    # Keep _compute_num_workers, _load/save_best_model_state, _save_metrics,
    # _calculate_ring_mobility_from_state (new helper), _wilson_lower_bound, _should_promote

    def clear_pytorch_memory(self):
        """Clear PyTorch memory caches and force garbage collection."""
        self.logger.info("Clearing PyTorch memory...")

        # Clear PyTorch's CUDA cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Move model to CPU temporarily to clear GPU memory
        if hasattr(self.network, 'network') and hasattr(self.network.network, 'to'):
            original_device = next(self.network.network.parameters()).device
            if original_device.type != 'cpu':
                self.network.network.to(torch.device('cpu'))
                # Memory pools handle cleanup automatically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.network.network.to(original_device)

        # Try advanced memory release techniques
        try:
            import ctypes
            # Try calling malloc_trim to release memory back to the OS
            try:
                ctypes.CDLL('libc.so.6').malloc_trim(0)
            except:
                pass  # May not work on macOS
        except Exception as e:
            self.logger.warning(f"Error attempting advanced memory cleanup: {e}")

        # Memory pools handle cleanup automatically

    def _reset_network_objects(self):
        """Recreate network objects to eliminate memory leaks."""
        import tempfile, torch, os

        self.logger.info("Recreating network objects to eliminate memory leaks")

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                temp_path = tmp.name
                # Save current network state
                self.network.save_model(temp_path)

                # Get device information
                device = getattr(self.network, 'device', torch.device('cpu'))

                # Recreate network with tensor pool
                from yinsh_ml.network.wrapper import NetworkWrapper
                new_network = NetworkWrapper(device=device, tensor_pool=self.tensor_pool)
                new_network.load_model(temp_path)

                # Replace old network
                self.network = new_network
                self.self_play.network = self.network
                self.trainer.network = self.network

                os.unlink(temp_path)
        except Exception as e:
            self.logger.error(f"Error recreating network objects: {e}")

    def prune_old_experiences(self):
        """
        Trim experience buffer to maintain memory efficiency between iterations.
        Add this method to your TrainingSupervisor class.
        """
        if hasattr(self.trainer, 'experience'):
            buffer_size = self.trainer.experience.size()
            keep_size = 15000  # Target size to maintain

            if buffer_size > 20000:  # Only prune if significantly over target
                # We need to add a prune method to the GameExperience class first
                if not hasattr(self.trainer.experience, 'prune_oldest'):
                    # Add the prune method dynamically if it doesn't exist
                    def prune_oldest(exp_obj, keep_newest):
                        """Keep only the most recent experiences."""
                        if len(exp_obj.states) <= keep_newest:
                            return  # Nothing to prune

                        # Calculate how many to remove
                        to_remove = len(exp_obj.states) - keep_newest

                        # Remove oldest entries (those at the start of the deque)
                        for _ in range(to_remove):
                            exp_obj.states.popleft()
                            exp_obj.move_probs.popleft()
                            exp_obj.values.popleft()
                            exp_obj.phases.popleft()

                        self.logger.info(f"Pruned {to_remove} experiences, keeping {keep_newest}")

                    # Attach the method to the experience object
                    import types
                    self.trainer.experience.prune_oldest = types.MethodType(
                        prune_oldest, self.trainer.experience)

                # Now we can call it
                self.logger.info(f"Pruning experience buffer from {buffer_size} to {keep_size} states")
                self.trainer.experience.prune_oldest(keep_size)

                # Memory pools handle cleanup automatically

                # Log memory usage after pruning
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    self.logger.info(f"Memory after pruning: {memory_mb:.1f} MB")
                except:
                    pass

    def _compute_num_workers(self) -> int:
        """Calculate optimal number of workers based on CPU cores."""
        # Use platform detection for Apple Silicon
        if platform.system() == "Darwin" and platform.processor() == "arm":
            # Suggest slightly fewer than physical cores for M-series to leave room for system/GPU
            physical_cores = psutil.cpu_count(logical=False)
            # M1/M2 often have 8 or 10 cores (e.g., 4+4 or 8+2 performance/efficiency)
            # Let's be conservative, maybe leave 2-3 cores free.
            optimal_workers = max(1, physical_cores - 3) # Ensure at least 1 worker
            self.logger.info(f"Apple Silicon detected. Using {optimal_workers} workers (Physical cores: {physical_cores}).")
            return optimal_workers

        # Original logic for other platforms (Linux/Windows, Intel/AMD)
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)

        if logical_cores >= 32:
            optimal_workers = min(24, logical_cores - 4)
        elif logical_cores >= 16:
            optimal_workers = min(12, logical_cores - 2)
        else:
            optimal_workers = max(4, physical_cores - 1 if physical_cores else logical_cores -1) # Use physical if available

        self.logger.info(f"CPU Info: Logical={logical_cores}, Physical={physical_cores}. Using {optimal_workers} workers.")
        return optimal_workers

    def _load_best_model_state(self):
        """Loads the state of the best model tracking if file exists."""
        state_path = self.save_dir / "best_model_state.json"
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                self.best_model_elo = state.get('best_model_elo', -float('inf'))
                self.best_model_iteration = state.get('best_model_iteration', -1)
                best_path_str = state.get('best_model_path')
                self._iteration_counter = state.get('_iteration_counter', 0) # Load counter

                if best_path_str:
                    # Ensure path is relative to the current save_dir for portability
                    potential_path = self.save_dir / Path(best_path_str).name # Reconstruct path relative to save_dir
                    if potential_path.exists():
                         self.best_model_path = potential_path
                         self.logger.info(f"Loaded best model state from {state_path}")
                    else:
                         self.logger.warning(f"Best model path '{potential_path}' from state file not found. Resetting tracking.")
                         self._reset_best_model_state() # Reset completely if path invalid
                else:
                     self.best_model_path = None # No path stored
                     self.logger.info(f"Loaded best model state from {state_path} (no best model path recorded yet).")


            except (json.JSONDecodeError, TypeError, KeyError) as e:
                self.logger.error(f"Failed to load or parse best model state from {state_path}: {e}. Starting fresh.", exc_info=True)
                self._reset_best_model_state()
        else:
            self.logger.info(f"No previous best_model_state.json found in {self.save_dir}. Starting fresh.")
            self._reset_best_model_state()

    def _save_best_model_state(self):
        """Saves the state of the best model tracking."""
        # Store the best model path relative to the save_dir if it exists
        relative_best_path = self.best_model_path.relative_to(self.save_dir).as_posix() if self.best_model_path and self.best_model_path.is_relative_to(self.save_dir) else None
        # Fallback: store just the filename if not relative (should ideally not happen with current logic)
        if self.best_model_path and relative_best_path is None:
             relative_best_path = self.best_model_path.name
             self.logger.warning(f"Best model path {self.best_model_path} was not relative to save dir {self.save_dir}. Saving only filename.")


        state = {
            'best_model_elo': self.best_model_elo,
            'best_model_iteration': self.best_model_iteration,
            'best_model_path': relative_best_path, # Store relative path or filename
            '_iteration_counter': self._iteration_counter # Save counter
        }
        state_path = self.save_dir / "best_model_state.json"
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=4)
            self.logger.debug(f"Saved best model state to {state_path}") # Use debug level
        except Exception as e:
            self.logger.error(f"Failed to save best model state to {state_path}: {e}", exc_info=True)

    def _reset_best_model_state(self):
        """Resets the best model tracking state."""
        self.best_model_elo = -float('inf')
        self.best_model_iteration = -1
        self.best_model_path = None
        self._iteration_counter = 0 # Reset counter too
        self.logger.info("Reset best model tracking state.")

    def _save_metrics(self, iteration_dir: Path) -> None:
        """Save training metrics for the completed iteration."""
        metrics_data = self.metrics.get_latest_metrics() # Assuming method exists
        metrics_data['timestamp'] = time.time()
        # Convert numpy types for JSON serialization
        serializable_data = {}
        for k, v in metrics_data.items():
             if isinstance(v, (np.number, np.bool_)): # Broader numpy type check
                 serializable_data[k] = v.item() # Use .item() to convert to Python native type
             elif isinstance(v, np.ndarray):
                  serializable_data[k] = v.tolist() # Convert arrays if needed
             elif isinstance(v, (int, float, bool, str, list, dict)) or v is None:
                  serializable_data[k] = v
             else:
                  try:
                      json.dumps(v) # Check if serializable directly
                      serializable_data[k] = v
                  except TypeError:
                      serializable_data[k] = str(v) # Fallback for other types


        metrics_path = iteration_dir / "metrics.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(serializable_data, f, indent=4)
            self.logger.info(f"Saved iteration metrics to {metrics_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics to {metrics_path}: {e}", exc_info=True)

    # Renamed helper to avoid conflict with method using encoded state tensor
    def _calculate_ring_mobility_from_state(self, game_state: GameState) -> float:
        """Calculate average number of valid moves available per ring from GameState object."""
        if not isinstance(game_state, GameState):
             self.logger.error("Invalid input: _calculate_ring_mobility_from_state requires a GameState object.")
             return 0.0

        total_moves = 0
        num_rings = 0
        try:
            for player in [Player.WHITE, Player.BLACK]:
                ring_type = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING
                ring_positions = game_state.board.get_pieces_positions(ring_type)
                num_rings += len(ring_positions)
                for pos in ring_positions:
                    # This assumes game_state.board.valid_move_positions exists and works correctly
                    valid_moves = game_state.board.valid_move_positions(pos)
                    total_moves += len(valid_moves)

            return total_moves / num_rings if num_rings > 0 else 0.0
        except AttributeError as e:
             self.logger.error(f"Error accessing board or methods during mobility calculation: {e}", exc_info=True)
             return 0.0
        except Exception as e:
            self.logger.error(f"Unexpected error calculating ring mobility: {e}", exc_info=True)
            return 0.0


    @staticmethod
    def _wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
        """Wilson score confidence interval lower bound."""
        if total == 0: return 0.0
        p_hat = wins / total
        try:
            term_inside_sqrt = (p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2))
            # Handle potential negative value due to floating point errors near 0 or 1
            if term_inside_sqrt < 0:
                 term_inside_sqrt = 0
            lower_bound = (p_hat + z**2 / (2 * total) - z * math.sqrt(term_inside_sqrt)) / (1 + z**2 / total)
        except ValueError: # math domain error if term_inside_sqrt is negative
             lower_bound = 0.0 # Should be handled by the check above, but belt and suspenders
        return max(0.0, lower_bound) # Ensure non-negative


    def _should_promote(self, wins: int, total: int,
                        threshold: float = 0.55, conf: float = 0.95) -> bool:
        """Determine if candidate should be promoted based on Wilson score."""
        if total == 0:
            self.logger.warning("Promotion gate check skipped: No head-to-head games played (total=0).")
            return False # Cannot promote without data

        z = 1.96 if conf == 0.95 else (2.576 if conf == 0.99 else 1.645) # Common z-scores
        lb = self._wilson_lower_bound(wins, total, z)

        self.logger.debug(f"Promotion Gate Check: Wins={wins}, Total={total}, Confidence={conf*100:.0f}%, Wilson LB={lb:.4f}, Threshold={threshold}")
        return lb > threshold