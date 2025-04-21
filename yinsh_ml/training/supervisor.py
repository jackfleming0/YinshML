"""Training supervisor for YINSH ML model."""

import logging
from pathlib import Path
import time
from typing import Optional, List, Tuple, Dict
import numpy as np
import json
import psutil, platform
import math

from ..network.wrapper import NetworkWrapper
from .self_play import SelfPlay
from .trainer import YinshTrainer
from ..utils.visualization import TrainingVisualizer
from ..utils.encoding import StateEncoder
from ..game.constants import Player, PieceType
from ..game.game_state import GameState
from ..utils.metrics_manager import TrainingMetrics
from ..utils.tournament import ModelTournament

# -----------------------------------------------------------------------------
# ONE‑TIME log‑file setup (main.py or experiment entry point)
# -----------------------------------------------------------------------------
import logging, sys, os, pathlib

root = logging.getLogger()
if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
    fh = logging.FileHandler("run.log", mode="w")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root.addHandler(fh)           # ← everything (incl. TrainingSupervisor) now ends up in run.log
    root.setLevel(logging.INFO)   # INFO is enough for the gate
print(f"[LOG] ‑‑ writing full log to {pathlib.Path('run.log').resolve()}", file=sys.stderr)

class TrainingSupervisor:
    def __init__(self,
                 network: NetworkWrapper,
                 save_dir: str,
                 # You can still override the early‑rollout budget here
                 mcts_simulations: int = 100,
                 mode: str = 'dev',
                 device: str = 'cpu',
                 tournament_games: int = 10, #number of games per match for tourney
                 **mode_settings     # ←‑‑ all config fields arrive here
    ):
        """
        Supervises the full training loop.

        New MCTS knobs expected in **mode_settings**:
        ------------------------------------------------
        late_simulations        (int)   – rollout budget *after* switch‑ply
        simulation_switch_ply   (int)   – ply at which we drop the budget
        temp_clamp_fraction     (float) – fraction of anneal steps after
                                          which temperature is clamped
        """
        # ─────────────────────────────────────────────────────────────────
        # 0.  House‑keeping & logging (unchanged, omitted here for brevity)
        # ─────────────────────────────────────────────────────────────────
        self.network  = network
        self.save_dir = Path(save_dir); self.save_dir.mkdir(exist_ok=True)
        self.mode     = mode
        self.device   = device
        self.tournament_games = tournament_games
        self.logger = logging.getLogger("TrainingSupervisor")   # ← add
        self.logger.setLevel(logging.INFO)

        # -----------------------------------------------------------------
        # 1.  Pull new hyper‑parameters out of **mode_settings**
        # -----------------------------------------------------------------
        early_simulations  = mcts_simulations                               # keep arg name
        late_simulations   = mode_settings.get('late_simulations',
                                               early_simulations)           # default = same
        switch_ply         = mode_settings.get('simulation_switch_ply', 20)
        temp_clamp_frac    = mode_settings.get('temp_clamp_fraction', 0.60)

        initial_temp       = mode_settings.get('initial_temp', 1.0)
        final_temp         = mode_settings.get('final_temp',   0.2)
        anneal_steps       = mode_settings.get('annealing_steps', 30)
        c_puct             = mode_settings.get('c_puct', 1.0)
        max_depth          = mode_settings.get('max_depth', 20)
        dirichlet_alpha = mode_settings.get('dirichlet_alpha', 0.3)
        value_weight = mode_settings.get('value_weight', 0.5)  # optional

        # -----------------------------------------------------------------
        # 2.  Instantiate SelfPlay with the full schedule
        # -----------------------------------------------------------------
        self.self_play = SelfPlay(
            network           = self.network,
            num_simulations   = early_simulations,      # early budget
            late_simulations  = late_simulations,       # NEW
            simulation_switch_ply = switch_ply,         # NEW
            temp_clamp_fraction   = temp_clamp_frac,    # NEW
            num_workers       = self._compute_num_workers(),  # helper as before
            initial_temp      = initial_temp,
            final_temp        = final_temp,
            annealing_steps   = anneal_steps,
            c_puct            = c_puct,
            max_depth         = max_depth,
            dirichlet_alpha   = dirichlet_alpha,   # ← NEW
            value_weight      = value_weight       # ← NEW (forwarded to MCTS via **extras)
        )

        # -----------------------------------------------------------------
        # 3.  Trainers, tournaments, etc.  (unchanged)
        # -----------------------------------------------------------------

        self.trainer = YinshTrainer(
            self.network, # Pass the managed network
            device=device,
            batch_size = mode_settings.get('batch_size', 256),   # ← NEW
            l2_reg=mode_settings.get('l2_reg', 0.0),
            # Pass other relevant trainer settings from mode_settings if needed
            value_head_lr_factor=mode_settings.get('value_head_lr_factor', 1.0),
            value_loss_weights=mode_settings.get('value_loss_weights', (0.5, 0.5))
        )
        #self.trainer.batch_size = mode_settings.get('batch_size', 256)

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

    def _compute_num_workers(self) -> int:
        """
        Heuristic identical to SelfPlay._get_optimal_workers().
        • On Apple Silicon (arm) → 6 workers (leaves 2 cores free on an M‑series 8‑core).
        • Otherwise scale with CPU count and leave a couple of cores for the OS.
        """
        if platform.processor() == "arm":
            return 6                                            # M‑series default
        logical  = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False) or logical   # fallback

        if logical >= 32:   # big server / A10G
            return min(24, logical - 4)
        elif logical >= 16: # mid‑tier (T4‑16CPU, etc.)
            return min(12, logical - 2)
        else:               # laptop / small VM
            return max(4, physical - 1)

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
        """
        One full self‑play → training → evaluation → model‑selection cycle.

        Key additions
        -------------
        • Promotion now uses a Wilson‑score lower‑bound gate
          (`_should_promote`) before looking at Elo.

        • The rest of the function is identical to your previous
          implementation (comments trimmed for readability).
        """
        # ------------------------------------------------------------------ #
        # 0.  Book‑keeping / directory setup
        # ------------------------------------------------------------------ #
        current_iteration = self._iteration_counter
        self.logger.info(f"\n{'='*15}  Starting Iteration {current_iteration}  {'='*15}")

        iteration_dir = self.save_dir / f"iteration_{current_iteration}"
        iteration_dir.mkdir(exist_ok=True, parents=True)

        # ------------------------------------------------------------------ #
        # 1.  SELF‑PLAY
        # ------------------------------------------------------------------ #
        self.logger.info(
            f"Generating {num_games} self‑play games using model "
            f"{self.best_model_iteration if self.best_model_path else 'initial'}…"
        )
        t0 = time.time()
        self.self_play.network = self.network          # ensure sync
        games = self.self_play.generate_games(num_games=num_games)
        game_time = time.time() - t0
        if not games:
            raise RuntimeError("Self‑play produced zero games!")
        self.logger.info(f"Generated {len(games)} games in {game_time:.1f}s")

        # -------- basic stats ------------------------------------------------
        game_lengths   = [len(g[0]) for g in games]
        avg_game_len   = np.mean(game_lengths)
        ring_mobility  = np.mean([self._calculate_ring_mobility(g[0][-1]) for g in games])
        outcomes       = [g[2] for g in games]
        w_wins         = outcomes.count(1)
        b_wins         = outcomes.count(-1)
        draws          = outcomes.count(0)
        win_rate       = (w_wins + b_wins) / len(outcomes)

        self.logger.info("Self‑Play Stats : "
                         f"avg_len={avg_game_len:.1f} | W/B/D = {w_wins}/{b_wins}/{draws}")

        # ------------------------------------------------------------------ #
        # 2.  ADD EXPERIENCE & TRAIN
        # ------------------------------------------------------------------ #
        games_path = iteration_dir / "games.npy"
        self.self_play.export_games(games, str(games_path))

        self.trainer.network = self.network            # assure same weights
        for states, policies, _numeric_outcome, *_ in games:
            # decode last state to get the final scores
            terminal_state = self.state_encoder.decode_state(states[-1])
            final_scores = (terminal_state.white_score,
                            terminal_state.black_score)  # e.g. (3,1)

            self.trainer.add_game_experience(states, policies, final_scores)
        self.logger.info("Training network …")
        t1 = time.time()
        self.trainer.policy_losses.clear(); self.trainer.value_losses.clear()
        for _ in range(epochs):
            self.trainer.train_epoch(
                batch_size=self.trainer.batch_size,
                batches_per_epoch=2,
            )
        train_time = time.time() - t1
        pol_loss = np.mean(self.trainer.policy_losses)
        val_loss = np.mean(self.trainer.value_losses)
        self.logger.info(f"Training done in {train_time:.1f}s "
                         f"(policy={pol_loss:.4f}, value={val_loss:.4f})")

        # ------------------------------------------------------------------ #
        # 3.  SAVE CANDIDATE CHECKPOINT
        # ------------------------------------------------------------------ #
        checkpoint_path = iteration_dir / f"checkpoint_iteration_{current_iteration}.pt"
        self.network.save_model(str(checkpoint_path))

        # ──────────────────────────────────────────────────────────────────────────
        # 4.  TOURNAMENT EVALUATION
        # ──────────────────────────────────────────────────────────────────────────
        self.logger.info("Running tournament evaluation …")

        self.tournament_manager.discover_models()
        self.tournament_manager.run_full_round_robin_tournament(current_iteration)

        model_id = f"iteration_{current_iteration}"
        tournament_stats = self.tournament_manager.get_model_performance(model_id) or {}
        current_elo = tournament_stats.get('current_rating', -float('inf'))
        tourn_win_rate = tournament_stats.get('win_rate', 0.0)

        # ──────────────────────────────────────────────────────────────────────────
        # 5.  PROMOTION / REVERSION
        # ──────────────────────────────────────────────────────────────────────────
        candidate_iteration = current_iteration  # The iteration just trained/evaluated
        candidate_id = f"iteration_{candidate_iteration}"

        # Get performance info for the candidate from the latest tournament
        tournament_stats = self.tournament_manager.get_model_performance(candidate_id) or {}
        candidate_elo = tournament_stats.get('current_rating', -float('inf'))  # Use a distinct name

        wins, total = 0, 0  # Default values for head-to-head
        perform_wilson_check = False  # Flag to indicate if H2H comparison is meaningful

        # Check if there *is* a previous best model recorded AND
        # if the candidate is different from the previous best.
        if self.best_model_iteration >= 0 and self.best_model_iteration != candidate_iteration:
            best_id = f"iteration_{self.best_model_iteration}"
            # Get head-to-head results between the *previous best* and the *new candidate*
            wins, total = self.tournament_manager.get_head_to_head(best_id, candidate_id)
            perform_wilson_check = True  # It makes sense to run the check

            # Log the comparison being made
            self.logger.info(
                f"Comparing Candidate (Iter {candidate_iteration}) vs Best (Iter {self.best_model_iteration}) for Wilson gate."
            )
        elif self.best_model_iteration < 0:
            self.logger.info("No previous best model to compare against for Wilson gate (first iteration).")
        else:  # best_model_iteration == candidate_iteration
            self.logger.info(
                f"Candidate (Iter {candidate_iteration}) is the same as the current best. Skipping Wilson gate check.")

        promote_by_wilson = False
        if perform_wilson_check:
            promote_by_wilson = self._should_promote(wins, total)
            # Show the gate’s raw numbers at INFO level so it’s always visible
            self.logger.info(
                f"Wilson Gate Check — Wins (candidate):{wins}, Total Games:{total}, "
                f"Wilson Lower Bound:{self._wilson_lower_bound(wins, total):.3f}, "
                f"Result: {'PROMOTE' if promote_by_wilson else 'REJECT'}"
            )
        # else: If not performing check, promote_by_wilson remains False

        # --- Final Promotion Decision ---
        # Promote if Wilson gate passed OR if it's the first model OR if Elo improved significantly.

        promote = False
        kept_current_best = False # Flag for the specific case candidate == best

        if promote_by_wilson:
            promote = True
            self.logger.info("Promoting based on successful Wilson gate check.")
        elif self.best_model_iteration < 0:
             promote = True
             self.logger.info("Promoting the first model automatically.")
        # Check Elo improvement as a fallback or primary reason
        elif candidate_elo > self.best_model_elo:
            promote = True
            reason = "Elo improved"
            if perform_wilson_check and not promote_by_wilson: # Wilson failed but Elo improved
                reason = f"Wilson gate failed ({wins}/{total}) but Elo improved"
            # This case should no longer happen if Wilson wasn't run because candidate == best
            # elif not perform_wilson_check:
            #     reason = f"Wilson gate skipped (candidate==best) but Elo improved"

            self.logger.info(f"{reason} ({candidate_elo:.1f} > {self.best_model_elo:.1f}) – promoting.")
        # --- ADD THIS BLOCK ---
        elif not perform_wilson_check and self.best_model_iteration == candidate_iteration:
             # Wilson check was skipped specifically because the candidate *is* the best.
             # No need to promote over itself, just keep it. Elo check already failed (candidate_elo <= self.best_model_elo).
             self.logger.info(f"Candidate (Iter {candidate_iteration}) is already the best model. Retaining current best state.")
             kept_current_best = True
             # promote remains False, but we won't reject/revert either.
        # --- END ADD ---


        # --- Apply Decision ---
        reverted = False
        if promote:
            # ---------- accept new model ------------------------------------
            self.logger.info(f"✅ PROMOTED: Iter {candidate_iteration} becomes new best model.")
            self.best_model_elo = candidate_elo
            self.best_model_iteration = candidate_iteration
            # The checkpoint path was saved earlier in the iteration
            self.best_model_path = iteration_dir / f"checkpoint_iteration_{candidate_iteration}.pt"
            # Save the *promoted* model's weights to the canonical "best_model.pt"
            # Ensure the path exists before saving
            if not self.best_model_path.exists():
                 self.logger.error(f"Cannot save best model: Checkpoint path {self.best_model_path} does not exist!")
            else:
                 self.network.save_model(str(self.best_model_save_path))
            self._save_best_model_state() # Save the updated best state info

        # Only reject/revert if NOT promoting AND NOT explicitly keeping the current best
        elif not kept_current_best:
            # ---------- keep / revert to previous best ---------------------
            self.logger.info(f"🚫 REJECTED: Candidate Iter {candidate_iteration} not promoted (Wilson Fail or Elo insufficient).")

            if self.best_model_iteration >= 0 and self.best_model_path and self.best_model_path.exists():
                self.logger.info(f"Reverting network weights to previous best model (Iter {self.best_model_iteration}) from {self.best_model_path}.")
                try:
                    self.network.load_model(str(self.best_model_path))
                    reverted = True
                except Exception as e:
                     self.logger.error(f"Failed to revert to best model weights from {self.best_model_path}: {e}. Network weights remain as the rejected candidate's weights.", exc_info=True)
                     # Keep reverted = False as the revert failed
            else:
                # This case should ideally not happen after the first iteration if saving works
                 log_msg = "No valid previous best model path found to revert to."
                 if self.best_model_iteration >= 0:
                      log_msg += f" Expected path: {self.best_model_path}"
                 self.logger.warning(log_msg + " Keeping candidate weights in memory, but not promoting.")
        # If kept_current_best is True, we do nothing here - the candidate weights are already active.

        # Determine which iteration's weights are active in the network object *now*
        if promote:
            active_iter = candidate_iteration
        elif reverted:
            active_iter = self.best_model_iteration
        elif kept_current_best:
            active_iter = self.best_model_iteration # Which is same as candidate_iteration
        else: # Rejected but failed to revert (or no prior best)
             active_iter = candidate_iteration # The rejected candidate's weights remain

        self.logger.info(f"Active network weights in memory correspond to iteration {active_iter}")


        # ------------------------------------------------------------------ #
        # 6.  METRICS, VISUALS, HOUSE‑KEEPING
        # ------------------------------------------------------------------ #
        self.metrics.add_iteration_metrics(
            avg_game_length = avg_game_len,
            avg_ring_mobility = ring_mobility,
            win_rate = win_rate,
            draw_rate = draws / len(outcomes),
            policy_loss = pol_loss,
            value_loss = val_loss,
            tournament_rating = current_elo,
            tournament_win_rate = tourn_win_rate
        )
        self._save_metrics(iteration_dir)

        # (stability checks / plots unchanged – omitted for brevity)

        # advance counter *after* everything else
        self._iteration_counter += 1

        # ------------------------------------------------------------------ #
        # 7.  RETURN SUMMARY
        # ------------------------------------------------------------------ #
        return {
            'iteration': current_iteration,
            'training_games': {
                'white_wins': w_wins, 'black_wins': b_wins, 'draws': draws,
                'win_rate': win_rate, 'avg_game_length': avg_game_len
            },
            'training': {
                'game_time': game_time, 'training_time': train_time,
                'policy_loss': pol_loss, 'value_loss': val_loss
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

    # ──────────────────────────────────────────────────────────────────────────
    # PROMOTION GATE helpers
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
        """
        Wilson score 95 % lower bound on win‑rate.
        Returns 0 if `total` is 0.
        """
        if total == 0:
            return 0.0
        p_hat = wins / total
        denom = 1 + z**2 / total
        centre = p_hat + z**2 / (2 * total)
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)
        return (centre - margin) / denom

    def _should_promote(self, wins: int, total: int,
                        threshold: float = 0.55, conf: float = 0.95) -> bool:
        if total == 0:  # ← ➊ short‑circuit
            self.logger.warning("Promotion gate skipped: tournament returned 0 games")
            return False  # never promote on no data

        z = 1.96 if conf == 0.95 else 2.58
        lb = self._wilson_lower_bound(wins, total, z)
        win_rate = wins / total  # ← safe: total > 0
        self.logger.debug(
            f"Promotion gate: win_rate={win_rate:.3f}, "
            f"Wilson‑LB={lb:.3f} (need >{threshold})")
        return lb > threshold