"""Training supervisor for YINSH ML model."""

import logging
from pathlib import Path
import time
from typing import Optional, List, Tuple, Dict
import numpy as np
import os
import json
import psutil # Make sure psutil is imported
import sys # Import sys for stdout checking

from ..network.wrapper import NetworkWrapper
from .self_play import SelfPlay
from .trainer import YinshTrainer
from ..utils.visualization import TrainingVisualizer
from ..utils.encoding import StateEncoder
from ..game.constants import Player, PieceType
from ..game.game_state import GameState
from ..utils.metrics_manager import TrainingMetrics
from ..utils.tournament import ModelTournament

class TrainingSupervisor:
    def __init__(self,
                 network: NetworkWrapper,
                 save_dir: str,
                 mcts_simulations: int = 100,
                 mode: str = 'dev',
                 device='cpu',
                 tournament_games: int = 10,
                 **mode_settings
        ):
        """
        Initialize the training supervisor.
        (Docstring remains the same)
        """
        self.network = network
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.mode = mode
        self.device = device
        self.tournament_games = tournament_games

        # --- Setup Logging ---
        self.logger = logging.getLogger("TrainingSupervisor")
        # Set the desired level for this specific logger
        self.logger.setLevel(logging.INFO)

        # Add a handler specifically for this logger if none exists at the root or for this logger
        # This helps ensure output even if basicConfig wasn't called, but basicConfig is preferred
        if not logging.root.hasHandlers() and not self.logger.hasHandlers():
             print("WARNING: No logging handlers found. Adding default StreamHandler for TrainingSupervisor.", file=sys.stderr)
             handler = logging.StreamHandler(sys.stdout) # Explicitly use stdout
             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
             handler.setFormatter(formatter)
             self.logger.addHandler(handler)
             # Propagate messages to root logger if it might get configured later
             self.logger.propagate = True
        elif not self.logger.hasHandlers() and self.logger.propagate:
             # If root has handlers, we likely don't need a specific one here
             pass # Messages will go to root's handlers
        elif not self.logger.propagate:
             print(f"WARNING: Logger {self.logger.name} has propagate=False and no handlers. It may not output messages.", file=sys.stderr)


        # Calculate optimal workers
        try:
            cpu_count_logical = psutil.cpu_count(logical=True)
            cpu_count_physical = psutil.cpu_count(logical=False)
            if cpu_count_logical is None: cpu_count_logical = 4 # Default
            if cpu_count_physical is None: cpu_count_physical = cpu_count_logical // 2 if cpu_count_logical > 1 else 1

            if cpu_count_logical >= 32:
                num_workers = min(24, cpu_count_logical - 4)
            elif cpu_count_logical >= 16:
                num_workers = min(12, cpu_count_logical - 2)
            else:
                # Prefer physical cores if significantly fewer, otherwise use logical - 1
                num_workers = max(2, cpu_count_physical -1) if cpu_count_physical < cpu_count_logical / 1.5 else max(2, cpu_count_logical - 1)
                num_workers = min(num_workers, 8) # Cap lower end too

        except Exception as e:
             self.logger.warning(f"Could not detect CPU counts ({e}), defaulting to 4 workers.")
             num_workers = 4

        self.num_workers = num_workers
        self.logger.info(f"Using {self.num_workers} workers for self-play.")

        # --- Initialize Components ---
        self.self_play = SelfPlay(
            network=self.network, # Pass the managed network
            num_simulations=mcts_simulations,
            num_workers=self.num_workers, # Use calculated workers
            initial_temp=mode_settings.get('initial_temp', 1.0),
            final_temp=mode_settings.get('final_temp', 0.2),
            annealing_steps=mode_settings.get('annealing_steps', 30),
            c_puct=mode_settings.get('c_puct', 1.0),
            max_depth=mode_settings.get('max_depth', 20)
        )

        self.trainer = YinshTrainer(
            self.network, # Pass the managed network
            device=device,
            l2_reg=mode_settings.get('l2_reg', 0.0),
            # Pass other relevant trainer settings from mode_settings if needed
            value_head_lr_factor=mode_settings.get('value_head_lr_factor', 1.0),
            value_loss_weights=mode_settings.get('value_loss_weights', (0.5, 0.5))
        )
        self.visualizer = TrainingVisualizer()
        self.state_encoder = StateEncoder()
        self.metrics = TrainingMetrics()

        self.tournament_manager = ModelTournament(
            training_dir=self.save_dir,
            device=device,
            games_per_match=tournament_games
        )

        # --- Explicit Initialization for Best Model Tracking ---
        self.best_model_elo: float = -float('inf') # Initialize ELO to negative infinity
        self.best_model_iteration: int = -1
        self.best_model_path: Optional[Path] = None
        self.best_model_save_path: Path = self.save_dir / "best_model.pt"
        self._iteration_counter: int = 0 # Initialize iteration counter

        # Try to load previous best model state if restarting
        self._load_best_model_state() # Sets internal vars and _iteration_counter
        self.logger.info("=== Training Supervisor Initialized ===")
        if self.best_model_path:
            self.logger.info(f"Loaded previous best model state: Iteration {self.best_model_iteration}, ELO {self.best_model_elo:.1f}")
            # Optionally load the best model weights into self.network here if desired on startup
            if self.best_model_path.exists():
                try:
                    self.logger.info(f"Attempting to load weights from {self.best_model_path} on startup...")
                    self.network.load_model(str(self.best_model_path))
                    self.logger.info("Successfully loaded best model weights on startup.")
                except Exception as e:
                    self.logger.error(f"Failed to load best model weights on startup from {self.best_model_path}: {e}. Current network weights unchanged.")
                    # Keep the loaded state info but log that the network wasn't updated yet
            else:
                 self.logger.warning(f"Best model path {self.best_model_path} from state file does not exist. Resetting tracking.")
                 self._reset_best_model_state()

    def _reset_best_model_state(self):
        """Resets the best model tracking state."""
        self.best_model_elo = -float('inf')
        self.best_model_iteration = -1
        self.best_model_path = None
        self._iteration_counter = 0 # Reset counter too

    def _save_best_model_state(self):
        """Saves the state of the best model tracking."""
        state = {
            'best_model_elo': self.best_model_elo,
            'best_model_iteration': self.best_model_iteration,
            'best_model_path': str(self.best_model_path) if self.best_model_path else None,
            '_iteration_counter': self._iteration_counter # Save counter
        }
        state_path = self.save_dir / "best_model_state.json"
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=4)
            self.logger.debug(f"Saved best model state to {state_path}") # Use debug level
        except Exception as e:
            self.logger.error(f"Failed to save best model state: {e}")

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
                    potential_path = Path(best_path_str)
                    if potential_path.exists():
                         self.best_model_path = potential_path
                    else:
                         self.logger.warning(f"Best model path '{best_path_str}' from state file not found. Resetting tracking.")
                         self._reset_best_model_state() # Reset completely if path invalid
                else:
                     self.best_model_path = None # No path stored

                self.logger.info(f"Loaded best model state from {state_path}")

            except Exception as e:
                self.logger.error(f"Failed to load or parse best model state from {state_path}: {e}. Starting fresh.")
                self._reset_best_model_state()
        else:
            self.logger.info("No previous best_model_state.json found. Starting fresh.")
            self._reset_best_model_state() # Ensure state is fresh


    def train_iteration(self, num_games: int = 100, epochs: int = 10):
        """Perform one training iteration."""
        current_iteration = self._iteration_counter # Use the internal counter
        self.logger.info(f"\n{'='*15} Starting Iteration {current_iteration} {'='*15}")

        iteration_dir = self.save_dir / f"iteration_{current_iteration}" # Use counter
        iteration_dir.mkdir(exist_ok=True)

        # --- Self-Play ---
        self.logger.info(f"Generating {num_games} self-play games using model from iteration {self.best_model_iteration if self.best_model_path else 'initial'}...")
        start_time = time.time()
        # Ensure SelfPlay uses the potentially reverted network
        self.self_play.network = self.network
        games = self.self_play.generate_games(num_games=num_games)
        game_time = time.time() - start_time
        if not games:
            self.logger.error("CRITICAL: No games generated in self-play. Stopping training.")
            raise RuntimeError("Self-play failed to generate games.")
        self.logger.info(f"Generated {len(games)} games in {game_time:.1f}s")

        # --- Game Metrics ---
        game_lengths = [len(game[0]) for game in games]
        avg_game_length = np.mean(game_lengths) if game_lengths else 0
        # Calculate ring mobility
        ring_mobilities = [self._calculate_ring_mobility(game[0][-1]) for game in games]
        avg_ring_mobility = np.mean(ring_mobilities) if ring_mobilities else 0
        # Calculate win/draw rates
        outcomes = [game[2] for game in games]
        white_wins = sum(1 for o in outcomes if o == 1)
        black_wins = sum(1 for o in outcomes if o == -1)
        draws = len(outcomes) - white_wins - black_wins
        win_rate = (white_wins + black_wins) / len(outcomes) if outcomes else 0
        draw_rate = draws / len(outcomes) if outcomes else 0

        self.logger.info("Self-Play Game Stats:")
        self.logger.info(f"  Avg Length: {avg_game_length:.1f}, Avg Mobility: {avg_ring_mobility:.1f}")
        self.logger.info(f"  Outcomes (W/B/D): {white_wins}/{black_wins}/{draws}, Win Rate: {win_rate:.2f}, Draw Rate: {draw_rate:.2f}")

        # --- Experience & Training ---
        games_path = iteration_dir / "games.npy"
        self.self_play.export_games(games, str(games_path))

        # Ensure Trainer uses the potentially reverted network
        self.trainer.network = self.network
        for states, policies, outcome in games:
            self.trainer.add_game_experience(states, policies, outcome)

        self.logger.info("Training network...")
        training_start = time.time()
        self.trainer.policy_losses.clear()
        self.trainer.value_losses.clear()
        for epoch in range(epochs):
            self.trainer.train_epoch(batch_size=self.trainer.batch_size, batches_per_epoch=100) # Use config batch size? or hardcode? Let's assume trainer has it
        training_time = time.time() - training_start
        self.logger.info(f"Training completed in {training_time:.1f}s")

        policy_loss = np.mean(self.trainer.policy_losses) if self.trainer.policy_losses else float('nan')
        value_loss = np.mean(self.trainer.value_losses) if self.trainer.value_losses else float('nan')
        self.logger.info(f"Training Losses - Policy: {policy_loss:.4f}, Value: {value_loss:.4f}")

        # --- Save Checkpoint ---
        checkpoint_path = iteration_dir / f"checkpoint_iteration_{current_iteration}.pt"
        self.network.save_model(str(checkpoint_path))
        self.logger.info(f"Saved checkpoint for iteration {current_iteration} to {checkpoint_path}")

        # --- Tournament Evaluation (with robust error handling) ---
        self.logger.info(f"\nRunning tournament evaluation for iteration {current_iteration}...")
        current_elo = -float('inf') # Default ELO if tournament fails
        tournament_win_rate = 0.0
        tournament_stats = {} # Initialize empty dict

        try:
            # Ensure tournament manager discovers the new model checkpoint
            self.tournament_manager.discover_models()

            # Run tournament including the newly trained model
            # Pass current_iteration if your manager needs it to identify the new model
            self.tournament_manager.run_full_round_robin_tournament(current_iteration)

            # Get ELO specifically for the model just trained
            model_id = f"iteration_{current_iteration}" # Adjust if your naming is different
            tournament_stats = self.tournament_manager.get_model_performance(model_id)

            if not tournament_stats:
                 self.logger.warning(f"No tournament stats returned for {model_id}. Cannot perform model selection based on ELO.")
                 # Keep current_elo as -inf
            else:
                 self.logger.info(f"Tournament Results for {model_id}:")
                 # Log the raw dict for debugging:
                 self.logger.debug(f"  Raw stats dict: {tournament_stats}")
                 # Safely get the rating, provide a default if key is missing
                 current_elo = tournament_stats.get('current_rating', -float('inf'))
                 if current_elo == -float('inf'):
                     self.logger.warning(f"  'current_rating' key missing or invalid in tournament_stats for {model_id}.")
                 else:
                      self.logger.info(f"  Current Rating (ELO): {current_elo:.1f}")
                 # Safely get win rate
                 tournament_win_rate = tournament_stats.get('win_rate', 0.0)
                 self.logger.info(f"  Win Rate vs Previous: {tournament_win_rate:.2%}")

        except Exception as e:
             self.logger.error(f"CRITICAL ERROR during tournament evaluation or stats retrieval for iteration {current_iteration}: {e}", exc_info=True)
             self.logger.error("Skipping model selection for this iteration due to tournament error.")
             # current_elo remains -inf, tournament_win_rate remains 0.0

        # --- Explicit Debug Logs Before Model Selection ---
        self.logger.info(f"DEBUG: Reached point just before model selection. Determined ELO for Iter {current_iteration}: {current_elo}")
        print(f"DEBUG PRINT: Reached point just before model selection. Determined ELO for Iter {current_iteration}: {current_elo}")

        # --- Model Selection Logic ---
        self.logger.info(f"\n===== MODEL SELECTION (Iteration {current_iteration}) =====")
        self.logger.info(f"Current Best: Iteration {self.best_model_iteration}, ELO {self.best_model_elo:.1f}, Path: {self.best_model_path}")
        self.logger.info(f"Candidate Model: Iteration {current_iteration}, ELO {current_elo:.1f}, Path: {checkpoint_path}")

        reverted = False
        # Ensure comparison happens only if current_elo is valid (not -inf)
        if current_elo > -float('inf') and current_elo > self.best_model_elo:
            self.logger.info(f"SUCCESS: New best model found! Iteration {current_iteration} (ELO {current_elo:.1f}) > Iteration {self.best_model_iteration} (ELO {self.best_model_elo:.1f})")
            self.best_model_elo = current_elo
            self.best_model_iteration = current_iteration
            self.best_model_path = checkpoint_path # Store path to the *.pt file

            # Save a separate copy to the dedicated best model path
            try:
                self.network.save_model(str(self.best_model_save_path))
                self.logger.info(f"Saved new best model weights to {self.best_model_save_path}")
                # Persist this new best state immediately (including iteration counter)
                self._save_best_model_state()
            except Exception as e:
                 self.logger.error(f"CRITICAL: Failed to save new best model to {self.best_model_save_path}: {e}")

        elif self.best_model_path is not None: # Only revert if there IS a previous best model
            # Log decision regardless of whether current_elo was valid, as long as a best exists
            if current_elo <= self.best_model_elo:
                 reason = f"Current ELO {current_elo:.1f} <= Best ELO {self.best_model_elo:.1f}"
            else: # This case means current_elo was -inf (tournament failed)
                 reason = "Tournament failed or returned invalid ELO"
            self.logger.warning(f"WEAKER/INVALID: Candidate model (Iter {current_iteration}) not better than best ({reason}).")

            # Attempt to revert only if the best model path exists
            if self.best_model_path.exists():
                self.logger.info(f"Attempting to REVERT to best model weights from: {self.best_model_path}")
                try:
                    # Load the weights from the previously saved best checkpoint
                    self.network.load_model(str(self.best_model_path))
                    self.logger.info(f"Successfully REVERTED network weights to best model (Iteration {self.best_model_iteration}).")
                    reverted = True
                    # No need to save state here, it hasn't changed (best model remains the same)
                except Exception as e:
                    self.logger.error(f"CRITICAL ERROR: Failed to load best model from {self.best_model_path}: {e}", exc_info=True)
                    self.logger.error("CONTINUING WITH THE CURRENT (POST-TRAINING) MODEL. Training might degrade.")
                    # Decide how to handle this: Stop? Alert?
                    # raise RuntimeError(f"Failed to revert to best model: {e}") # Option to stop
            else:
                 self.logger.error(f"CRITICAL ERROR: Best model path {self.best_model_path} not found. Cannot revert. Continuing with current model.")

        else:
            # This case handles the very first iteration where no 'best' exists yet, OR if tournament failed on first iter
            if current_elo > -float('inf'):
                self.logger.info(f"INITIAL BEST: Setting Iteration {current_iteration} (ELO {current_elo:.1f}) as the initial best model.")
                self.best_model_elo = current_elo
                self.best_model_iteration = current_iteration
                self.best_model_path = checkpoint_path
                try:
                    self.network.save_model(str(self.best_model_save_path))
                    self.logger.info(f"Saved initial best model to {self.best_model_save_path}")
                    self._save_best_model_state() # Save the initial state
                except Exception as e:
                     self.logger.error(f"CRITICAL: Failed to save initial best model to {self.best_model_save_path}: {e}")
            else:
                 self.logger.warning(f"Tournament failed on first iteration. No initial best model set.")


        # Log final state for the iteration
        if reverted:
            active_iter = self.best_model_iteration
        elif self.best_model_iteration == current_iteration: # New best was found this iter
            active_iter = current_iteration
        elif self.best_model_path is None: # Still no best model (e.g., first iter failed tournament)
             active_iter = "initial/untrained" # Or maybe current_iteration if revert failed? Needs careful thought. Let's say current for now.
             active_iter = current_iteration # Let's be explicit it's the one just trained, even if bad
        else: # Kept current model because revert failed or wasn't needed but wasn't new best
             active_iter = current_iteration


        self.logger.info(f"Model Selection Complete. Network in memory holds weights from iteration: {active_iter}")
        self.logger.info(f"===== END MODEL SELECTION =====")


        # --- Update & Save Metrics ---
        # Ensure metrics are added even if tournament failed
        self.metrics.add_iteration_metrics(
            avg_game_length=avg_game_length,
            avg_ring_mobility=avg_ring_mobility,
            win_rate=win_rate, # Self-play win rate
            draw_rate=draw_rate,# Self-play draw rate
            policy_loss=policy_loss,
            value_loss=value_loss,
            # eval_win_rate=eval_win_rate, # If you run separate eval
            # eval_draw_rate=eval_draw_rate,
            # Use the potentially -inf ELO and 0.0 win rate if tournament failed
            tournament_rating=current_elo,
            tournament_win_rate=tournament_win_rate
        )
        self._save_metrics(iteration_dir) # Save metrics for this specific iteration

        # --- Stability Checks & Visualization ---
        try: # Wrap stability/plotting in case of errors
            stability_checks = self.metrics.assess_stability()
            self.logger.info("\nStability Check Results:")
            for check, result in stability_checks.items():
                self.logger.info(f"  {check}: {'PASS' if result else 'FAIL'}")

            # Generate visualizations
            plot_path = iteration_dir / "training_history.png"
            self.visualizer.plot_training_history(
                self.metrics.get_plotting_data(), # Assume metrics manager provides data
                save_path=str(plot_path)
            )
            self.logger.info(f"Saved training history plot to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error during stability check or plotting: {e}", exc_info=True)

        # --- Increment Iteration Counter ---
        self._iteration_counter += 1
        # Save state again AFTER incrementing counter if you want the *next* iteration number saved
        # Or save before incrementing if you want the completed iteration number saved. Let's save before:
        # self._save_best_model_state() # Moved saving to happen only when best model changes or initially

        # Return metrics
        return {
            'iteration': current_iteration,
            'training_games': {
                'white_wins': white_wins, 'black_wins': black_wins, 'draws': draws,
                'win_rate': win_rate, 'draw_rate': draw_rate,
                'avg_game_length': avg_game_length, 'avg_ring_mobility': avg_ring_mobility
            },
            'training': {
                'game_time': game_time, 'training_time': training_time,
                'policy_loss': policy_loss, 'value_loss': value_loss
            },
            'evaluation': {
                # 'self_play': {'win_rate': eval_win_rate, 'draw_rate': eval_draw_rate}, # If separate eval run
                'tournament': {
                    'rating': current_elo,
                    'win_rate': tournament_win_rate,
                    'raw_stats': tournament_stats # Include raw stats for inspection
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

    def _save_metrics(self, iteration_dir: Path) -> None:
        """Save training metrics for the completed iteration."""
        metrics_data = self.metrics.get_latest_metrics() # Assuming method exists
        # Add timestamp and ensure numpy types are converted
        metrics_data['timestamp'] = time.time()
        serializable_data = {}
        for k, v in metrics_data.items():
             if isinstance(v, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                 serializable_data[k] = int(v)
             elif isinstance(v, (np.floating, np.float_, np.float16, np.float32, np.float64)):
                 serializable_data[k] = float(v) if not np.isnan(v) else None # Handle NaN
             elif isinstance(v, np.ndarray):
                  serializable_data[k] = v.tolist() # Convert arrays if needed
             elif isinstance(v, (int, float, bool, str, list, dict)) or v is None:
                  serializable_data[k] = v
             else:
                  serializable_data[k] = str(v) # Fallback for other types


        metrics_path = iteration_dir / "metrics.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(serializable_data, f, indent=4, allow_nan=True) # Allow NaN -> null
            self.logger.info(f"Saved iteration metrics to {metrics_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics to {metrics_path}: {e}")

    def _calculate_ring_mobility(self, state_tensor: np.ndarray) -> float:
        """Calculate average number of valid moves available per ring."""
        # Ensure state_encoder is available
        if not hasattr(self, 'state_encoder'):
            self.logger.error("StateEncoder not initialized in TrainingSupervisor.")
            return 0.0

        try:
            game_state = self.state_encoder.decode_state(state_tensor)
            total_moves = 0
            num_rings = 0

            for player in [Player.WHITE, Player.BLACK]:
                ring_type = PieceType.WHITE_RING if player == Player.WHITE else PieceType.BLACK_RING
                ring_positions = game_state.board.get_pieces_positions(ring_type)

                for pos in ring_positions:
                    # Ensure board object and method exist
                    if hasattr(game_state, 'board') and hasattr(game_state.board, 'valid_move_positions'):
                        valid_moves = game_state.board.valid_move_positions(pos)
                        total_moves += len(valid_moves)
                    else:
                        self.logger.warning("game_state.board or valid_move_positions not found during mobility calc.")
                num_rings += len(ring_positions)

            return total_moves / num_rings if num_rings > 0 else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating ring mobility: {e}", exc_info=True)
            return 0.0


    # Remove duplicate/unused methods like evaluate_model, _evaluate_model, _handle_model_selection
    # if the logic is fully integrated into train_iteration now.
    # Keep evaluate_self_play if you intend to call it separately.
    def evaluate_self_play(self, num_games: int = 25) -> Tuple[float, float]:
        # Placeholder for self-play evaluation if needed separately
        self.logger.info(f"Running self-play evaluation ({num_games} games)...")
        # Use a temporary SelfPlay instance or ensure the main one uses low temp?
        eval_self_play = SelfPlay(
            network=self.network, # Use current network state
            num_simulations=self.self_play.num_simulations, # Use same sims?
            num_workers=self.num_workers,
            initial_temp=0.1, # Low temp for evaluation
            final_temp=0.1,
            annealing_steps=1,
            c_puct=self.self_play.c_puct,
            max_depth=self.self_play.max_depth
        )
        games = eval_self_play.generate_games(num_games=num_games)
        if not games: return 0.0, 0.0
        outcomes = [game[2] for game in games]
        wins = sum(1 for o in outcomes if o != 0)
        draws = len(outcomes) - wins
        win_rate = wins / len(outcomes) if outcomes else 0.0
        draw_rate = draws / len(outcomes) if outcomes else 0.0
        self.logger.info(f"Self-play Eval Results - Win Rate: {win_rate:.2f}, Draw Rate: {draw_rate:.2f}")
        return win_rate, draw_rate