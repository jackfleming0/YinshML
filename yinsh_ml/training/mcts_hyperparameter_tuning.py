"""MCTS hyperparameter tuning and optimization framework.

This module implements automated hyperparameter tuning for the enhanced MCTS
using grid search and Bayesian optimization methods.
"""

import copy
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from enum import Enum

from ..game.game_state import GameState
from ..game.constants import Player
from ..training.enhanced_mcts import EnhancedMCTS, EnhancedMCTSConfig
from ..network.wrapper import NetworkWrapper
from ..analysis.phase_analyzer import GamePhase


def _to_json_safe(obj):
    """Recursively convert objects to JSON-serializable forms.

    - Enum keys/values -> their value or string
    - numpy scalars -> Python scalars
    - dict keys ensured to be strings
    """
    # Enum values
    if isinstance(obj, Enum):
        # Prefer the enum value if present, fallback to name
        return getattr(obj, "value", str(obj))

    # numpy scalar types
    if isinstance(obj, np.generic):
        return obj.item()

    # dicts: ensure keys are strings and values are converted
    if isinstance(obj, dict):
        converted = {}
        for k, v in obj.items():
            if isinstance(k, Enum):
                key_str = str(getattr(k, "value", str(k)))
            else:
                key_str = str(k)
            converted[key_str] = _to_json_safe(v)
        return converted

    # sequences
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(x) for x in obj]

    # dataclass dicts or plain python values pass through
    return obj

@dataclass
class HyperparameterSpace:
    """Define hyperparameter search space."""
    
    # Exploration constant C (UCB1)
    c_puct_min: float = 0.5
    c_puct_max: float = 2.0
    c_puct_step: float = 0.25
    
    # Heuristic weight alpha
    heuristic_alpha_min: float = 0.1
    heuristic_alpha_max: float = 0.7
    heuristic_alpha_step: float = 0.1
    
    # Epsilon for greedy rollouts
    epsilon_greedy_min: float = 0.2
    epsilon_greedy_max: float = 0.6
    epsilon_greedy_step: float = 0.1
    
    # Simulation budget
    num_simulations_min: int = 1000
    num_simulations_max: int = 5000
    num_simulations_step: int = 1000
    
    # Simulation depth limits
    max_depth_min: int = 30
    max_depth_max: int = 80
    max_depth_step: int = 10
    
    def get_grid_search_space(self) -> List[Dict[str, Any]]:
        """Generate grid search parameter combinations."""
        c_puct_values = np.arange(self.c_puct_min, self.c_puct_max + self.c_puct_step, self.c_puct_step)
        heuristic_alpha_values = np.arange(self.heuristic_alpha_min, self.heuristic_alpha_max + self.heuristic_alpha_step, self.heuristic_alpha_step)
        epsilon_greedy_values = np.arange(self.epsilon_greedy_min, self.epsilon_greedy_max + self.epsilon_greedy_step, self.epsilon_greedy_step)
        num_simulations_values = range(self.num_simulations_min, self.num_simulations_max + self.num_simulations_step, self.num_simulations_step)
        max_depth_values = range(self.max_depth_min, self.max_depth_max + self.max_depth_step, self.max_depth_step)
        
        combinations = []
        for combo in itertools.product(c_puct_values, heuristic_alpha_values, epsilon_greedy_values, 
                                     num_simulations_values, max_depth_values):
            combinations.append({
                'c_puct': combo[0],
                'heuristic_alpha': combo[1],
                'epsilon_greedy': combo[2],
                'num_simulations': combo[3],
                'max_depth': combo[4]
            })
        
        return combinations


@dataclass
class TuningResult:
    """Container for hyperparameter tuning results."""
    
    parameters: Dict[str, Any]
    win_rate: float
    move_time_avg: float
    move_time_std: float
    games_played: int
    evaluation_time: float
    timestamp: str
    # Optional diagnostics (color split and termination stats)
    tuned_white_win_rate: Optional[float] = None
    tuned_black_win_rate: Optional[float] = None
    balanced_win_rate: Optional[float] = None
    terminal_rate: Optional[float] = None
    nonterminal_tiebreak_rate: Optional[float] = None
    move_cap_rate: Optional[float] = None
    timeout_rate: Optional[float] = None
    avg_moves: Optional[float] = None
    avg_white_score: Optional[float] = None
    avg_black_score: Optional[float] = None
    avg_score_diff: Optional[float] = None
    no_valid_moves_rate: Optional[float] = None
    invalid_move_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    
    # Evaluation parameters
    evaluation_games: int = 20  # Number of games per parameter set
    max_move_time: float = 3.0  # Maximum allowed move time in seconds
    min_win_rate_improvement: float = 0.05  # Minimum win rate improvement for significance
    enforce_move_time_cap: bool = True  # If False, do not cap per-move time
    max_moves_per_game: int = 300  # Hard cap on moves to prevent infinite games
    
    # Optimization parameters
    use_bayesian_optimization: bool = False  # Use Bayesian optimization instead of grid search
    bayesian_iterations: int = 50  # Number of Bayesian optimization iterations
    bayesian_acquisition: str = 'EI'  # Acquisition function: 'EI', 'PI', 'UCB'
    
    # Statistical testing
    confidence_level: float = 0.95  # Confidence level for statistical tests
    min_games_for_significance: int = 10  # Minimum games for statistical significance
    
    # Parallel execution
    max_workers: int = 4  # Maximum number of parallel workers
    timeout_per_game: int = 300  # Timeout per game in seconds
    enforce_game_timeout: bool = False  # If True, enforce per-game timeout; default off

    # Head-to-head evaluation settings (vs baseline pool)
    use_head_to_head: bool = True  # Evaluate tuned params vs baselines instead of self-play
    baselines: Optional[List[Dict[str, Any]]] = None  # List of baseline param dicts
    paired_colors: bool = True  # Play both colors for each seed when head-to-head

    # Successive Halving (SH) settings
    sh_eta: int = 3  # Reduction factor per round
    sh_initial_games: Optional[int] = None  # If None, uses evaluation_games
    sh_max_rounds: Optional[int] = None  # If None, derived from number of configs

    # Debug / diagnostics
    record_game_logs: bool = False  # If True, write per-game logs for head-to-head series
    debug_log_dir: Optional[str] = None  # Optional directory for debug logs
    debug_log_prefix: str = "h2h_debug"  # File prefix for debug logs
    # Fairness options
    randomize_starting_player: bool = True


class MCTSHyperparameterTuner:
    """Main class for MCTS hyperparameter tuning."""
    
    def __init__(self, 
                 network: NetworkWrapper,
                 baseline_config: EnhancedMCTSConfig,
                 tuning_config: TuningConfig = None):
        """
        Initialize hyperparameter tuner.
        
        Args:
            network: Neural network wrapper
            baseline_config: Baseline MCTS configuration
            tuning_config: Tuning configuration
        """
        self.network = network
        self.baseline_config = baseline_config
        self.tuning_config = tuning_config or TuningConfig()
        self.logger = logging.getLogger("MCTSHyperparameterTuner")
        
        # Results storage
        self.results: List[TuningResult] = []
        self.best_result: Optional[TuningResult] = None
        
        # Hyperparameter space
        self.hyperparameter_space = HyperparameterSpace()
        
        self.logger.info("MCTS Hyperparameter Tuner Initialized")
        self.logger.info(f"Evaluation games per parameter set: {self.tuning_config.evaluation_games}")
        self.logger.info(f"Maximum move time: {self.tuning_config.max_move_time}s")
        self.logger.info(f"Max workers: {self.tuning_config.max_workers}")
    
    def _create_mcts_with_params(self, params: Dict[str, Any]) -> EnhancedMCTS:
        """Create MCTS instance with given parameters."""
        config = copy.deepcopy(self.baseline_config)
        
        # Update configuration with tuned parameters
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return EnhancedMCTS(self.network, config)
    
    def _play_h2h_series(self,
                         tuned_params: Dict[str, Any],
                         baseline_params: Dict[str, Any],
                         seeds: int) -> Tuple[int, int, int, List[float], Dict[str, Any]]:
        """Play a head-to-head mini-series between tuned params and a baseline.

        Returns (tuned_wins, draws, total_games, tuned_move_times)
        """
        # Build MCTS instances once per series
        tuned_mcts_white = self._create_mcts_with_params(tuned_params)
        tuned_mcts_black = self._create_mcts_with_params(tuned_params)
        base_mcts_white = self._create_mcts_with_params(baseline_params)
        base_mcts_black = self._create_mcts_with_params(baseline_params)

        tuned_wins = 0
        draws = 0
        total_games = 0
        tuned_move_times: List[float] = []
        # Color-split and termination diagnostics
        tuned_white_wins = 0
        tuned_black_wins = 0
        tuned_white_games = 0
        tuned_black_games = 0
        white_wins = 0
        black_wins = 0
        term_counts = {
            'terminal_state': 0,
            'non_terminal_tiebreak': 0,
            'move_cap': 0,
            'timeout': 0,
            'no_valid_moves': 0,
            'invalid_move': 0
        }
        moves_total = 0
        games_count = 0
        series_logs: List[Dict[str, Any]] = []
        sum_white_score = 0
        sum_black_score = 0

        def play_game(tuned_is_white: bool, seed_index: int) -> Tuple[float, List[float], Dict[str, Any]]:
            nonlocal tuned_white_wins, tuned_black_wins, tuned_white_games, tuned_black_games
            nonlocal white_wins, black_wins, moves_total, games_count
            nonlocal sum_white_score, sum_black_score
            state = GameState()
            # Randomize starting player deterministically by seed to reduce first-move bias
            state.current_player = Player.WHITE if (seed_index % 2 == 0) else Player.BLACK
            move_count = 0
            tuned_times: List[float] = []
            break_reason: Optional[str] = None
            game_log: Dict[str, Any] = {
                'seed_index': seed_index,
                'tuned_is_white': tuned_is_white,
                'start_player': 'WHITE' if (seed_index % 2 == 0) else 'BLACK',
                'moves_played': 0,
                'termination_reason': '',
                'cap_hit': False,
                'timeout_triggered': False,
                'state_terminal': False,
                'winner': None,
                'tuned_score': None,
                'white_score': None,
                'black_score': None
            }

            cap = self.tuning_config.max_moves_per_game
            while not state.is_terminal():
                move_start = time.time()

                # Select current MCTS by player to move
                if state.current_player == Player.WHITE:
                    current = tuned_mcts_white if tuned_is_white else base_mcts_white
                    tuned_to_move = tuned_is_white
                else:
                    current = tuned_mcts_black if not tuned_is_white else base_mcts_black
                    tuned_to_move = (not tuned_is_white)

                policy = current.search(state, move_count + 1)

                valid_moves = state.get_valid_moves()
                if not valid_moves:
                    break_reason = 'no_valid_moves'
                    break

                move_probs = []
                for move in valid_moves:
                    move_idx = current.state_encoder.move_to_index(move)
                    if 0 <= move_idx < len(policy):
                        move_probs.append(policy[move_idx])
                    else:
                        move_probs.append(0.0)

                if move_probs and max(move_probs) > 0:
                    selected_idx = int(np.argmax(move_probs))
                else:
                    selected_idx = 0

                success = state.make_move(valid_moves[selected_idx])
                if not success:
                    break_reason = 'invalid_move'
                    break
                move_count += 1

                move_time = time.time() - move_start
                if tuned_to_move:
                    tuned_times.append(move_time)

                if self.tuning_config.enforce_move_time_cap and move_time > self.tuning_config.max_move_time:
                    # Respect per-move cap when enabled
                    if not game_log['termination_reason']:
                        game_log['termination_reason'] = 'move_time_cap'
                        game_log['timeout_triggered'] = True
                    break

                if cap and cap > 0 and move_count >= cap:
                    # Respect max moves only if positive
                    if not game_log['termination_reason']:
                        game_log['termination_reason'] = 'move_cap'
                        game_log['cap_hit'] = True
                    break

            # Score outcome from tuned perspective (use heuristic tiebreak on non-terminal)
            if state.is_terminal():
                winner = state.get_winner()
                if winner == Player.WHITE:
                    tuned_score = 1.0 if tuned_is_white else 0.0
                elif winner == Player.BLACK:
                    tuned_score = 1.0 if not tuned_is_white else 0.0
                else:
                    tuned_score = 0.5
            else:
                # Tiebreak at cap: try heuristic first, then fall back to score differential
                tuned_score = 0.5
                evaluator = tuned_mcts_white.heuristic_evaluator or base_mcts_white.heuristic_evaluator
                if evaluator is not None:
                    try:
                        white_eval = evaluator.evaluate_position_fast(state, Player.WHITE)
                        black_eval = evaluator.evaluate_position_fast(state, Player.BLACK)
                        delta = white_eval - black_eval
                        if delta > 1e-6:
                            tuned_score = 1.0 if tuned_is_white else 0.0
                        elif delta < -1e-6:
                            tuned_score = 1.0 if not tuned_is_white else 0.0
                    except Exception:
                        tuned_score = 0.5
                if tuned_score == 0.5:
                    try:
                        # Use built-in score differential if available on state
                        white_score = getattr(state, 'white_score', None)
                        black_score = getattr(state, 'black_score', None)
                        if white_score is not None and black_score is not None:
                            if white_score > black_score:
                                tuned_score = 1.0 if tuned_is_white else 0.0
                            elif black_score > white_score:
                                tuned_score = 1.0 if not tuned_is_white else 0.0
                    except Exception:
                        pass

            # Populate log metadata
            game_log['moves_played'] = move_count
            game_log['state_terminal'] = state.is_terminal()
            if not game_log['termination_reason']:
                if state.is_terminal():
                    game_log['termination_reason'] = 'terminal_state'
                else:
                    game_log['termination_reason'] = break_reason or 'non_terminal_tiebreak'

            winner = state.get_winner() if state.is_terminal() else None
            if winner == Player.WHITE:
                game_log['winner'] = 'WHITE'
            elif winner == Player.BLACK:
                game_log['winner'] = 'BLACK'
            elif winner is None and state.is_terminal():
                game_log['winner'] = 'DRAW'
            else:
                game_log['winner'] = None

            game_log['tuned_score'] = float(tuned_score)
            game_log['white_score'] = getattr(state, 'white_score', None)
            game_log['black_score'] = getattr(state, 'black_score', None)

            # Aggregate diagnostics
            if game_log['winner'] == 'WHITE':
                white_wins += 1
            elif game_log['winner'] == 'BLACK':
                black_wins += 1

            # Count tuned results by color
            if tuned_is_white:
                tuned_white_games += 1
                if tuned_score == 1.0:
                    tuned_white_wins += 1
            else:
                tuned_black_games += 1
                if tuned_score == 1.0:
                    tuned_black_wins += 1

            term = game_log['termination_reason']
            if term in term_counts:
                term_counts[term] += 1
            moves_total += move_count
            games_count += 1
            # Score sums
            if isinstance(game_log['white_score'], (int, float)):
                sum_white_score += int(game_log['white_score'])
            if isinstance(game_log['black_score'], (int, float)):
                sum_black_score += int(game_log['black_score'])

            return tuned_score, tuned_times, game_log

        for s in range(seeds):
            # Randomize game color order per seed to reduce sequence effects
            if random.random() < 0.5:
                order = (True, False)
            else:
                order = (False, True)
            score_a, times_a, log_a = play_game(tuned_is_white=order[0], seed_index=s)
            score_b, times_b, log_b = play_game(tuned_is_white=order[1], seed_index=s)

            tuned_move_times.extend(times_a)
            tuned_move_times.extend(times_b)

            if self.tuning_config.record_game_logs:
                series_logs.append(log_a)
                series_logs.append(log_b)

            # Convert scores to counts
            for score in (score_a, score_b):
                if score == 1.0:
                    tuned_wins += 1
                elif score == 0.5:
                    draws += 1
                # losses contribute 0
                total_games += 1

        if self.tuning_config.record_game_logs and series_logs:
            try:
                log_dir = Path(self.tuning_config.debug_log_dir or "logs/mcts_h2h_debug")
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / f"{self.tuning_config.debug_log_prefix}_{int(time.time())}_{uuid.uuid4().hex}.json"
                log_payload = {
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'tuned_params': _to_json_safe(tuned_params),
                    'baseline_params': _to_json_safe(baseline_params),
                    'seeds': seeds,
                    'summary': {
                        'tuned_wins': tuned_wins,
                        'draws': draws,
                        'total_games': total_games,
                        'win_rate': ((tuned_wins) + 0.5 * draws) / total_games if total_games > 0 else None,
                        'tuned_white_win_rate': (tuned_white_wins / tuned_white_games) if tuned_white_games > 0 else None,
                        'tuned_black_win_rate': (tuned_black_wins / tuned_black_games) if tuned_black_games > 0 else None,
                        'terminal_rate': (term_counts['terminal_state'] / games_count) if games_count > 0 else None,
                        'nonterminal_tiebreak_rate': (term_counts['non_terminal_tiebreak'] / games_count) if games_count > 0 else None,
                        'move_cap_rate': (term_counts['move_cap'] / games_count) if games_count > 0 else None,
                        'timeout_rate': (term_counts['timeout'] / games_count) if games_count > 0 else None,
                        'avg_moves': (moves_total / games_count) if games_count > 0 else None
                    },
                    'games': _to_json_safe(series_logs)
                }
                with open(log_file, 'w') as f:
                    json.dump(log_payload, f, indent=2)
                self.logger.info(f"Recorded head-to-head debug log to {log_file}")
            except Exception as exc:
                self.logger.warning(f"Failed to record head-to-head debug log: {exc}")

        series_stats = {
            'tuned_white_wins': tuned_white_wins,
            'tuned_black_wins': tuned_black_wins,
            'tuned_white_games': tuned_white_games,
            'tuned_black_games': tuned_black_games,
            'white_wins': white_wins,
            'black_wins': black_wins,
            'draws': draws,
            'terminal_state': term_counts['terminal_state'],
            'non_terminal_tiebreak': term_counts['non_terminal_tiebreak'],
            'move_cap': term_counts['move_cap'],
            'timeout': term_counts['timeout'],
            'no_valid_moves': term_counts['no_valid_moves'],
            'invalid_move': term_counts['invalid_move'],
            'moves_total': moves_total,
            'games_count': games_count,
            'sum_white_score': sum_white_score,
            'sum_black_score': sum_black_score
        }

        return tuned_wins, draws, total_games, tuned_move_times, series_stats

    def _evaluate_with_budget(self, params: Dict[str, Any], games_budget: int) -> TuningResult:
        """Evaluate params using a specific games budget (head-to-head preferred)."""
        start_time = time.time()

        use_h2h = self.tuning_config.use_head_to_head
        baselines = self.tuning_config.baselines or []

        if use_h2h and baselines:
            per_baseline_games_budget = max(1, games_budget // max(1, len(baselines)))
            games_per_seed = 2 if self.tuning_config.paired_colors else 1
            seeds = max(1, per_baseline_games_budget // games_per_seed)

            total_wins = 0
            total_draws = 0
            total_games = 0
            tuned_move_times_all: List[float] = []
            # Diagnostics accumulators
            sum_tuned_white_wins = 0
            sum_tuned_black_wins = 0
            sum_tuned_white_games = 0
            sum_tuned_black_games = 0
            sum_terminal = 0
            sum_nonterminal = 0
            sum_move_cap = 0
            sum_timeout = 0
            sum_moves_total = 0
            sum_games_count = 0
            sum_sum_white_score = 0
            sum_sum_black_score = 0
            sum_no_valid_moves = 0
            sum_invalid_move = 0

            for base in baselines:
                wins_b, draws_b, games_b, tuned_times_b, stats_b = self._play_h2h_series(
                    tuned_params=params,
                    baseline_params=base,
                    seeds=seeds
                )
                total_wins += wins_b
                total_draws += draws_b
                total_games += games_b
                tuned_move_times_all.extend(tuned_times_b)
                # Aggregate diagnostics
                sum_tuned_white_wins += stats_b['tuned_white_wins']
                sum_tuned_black_wins += stats_b['tuned_black_wins']
                sum_tuned_white_games += stats_b['tuned_white_games']
                sum_tuned_black_games += stats_b['tuned_black_games']
                sum_terminal += stats_b['terminal_state']
                sum_nonterminal += stats_b['non_terminal_tiebreak']
                sum_move_cap += stats_b['move_cap']
                sum_timeout += stats_b['timeout']
                sum_moves_total += stats_b['moves_total']
                sum_games_count += stats_b['games_count']
                sum_sum_white_score += stats_b['sum_white_score']
                sum_sum_black_score += stats_b['sum_black_score']
                sum_no_valid_moves += stats_b['no_valid_moves']
                sum_invalid_move += stats_b['invalid_move']

            evaluation_time = time.time() - start_time
            win_rate = ((total_wins) + 0.5 * total_draws) / total_games if total_games > 0 else 0.0
            move_time_avg = np.mean(tuned_move_times_all) if tuned_move_times_all else float('inf')
            move_time_std = np.std(tuned_move_times_all) if tuned_move_times_all else 0.0
            games_played = total_games
            # Compute diagnostics
            tuned_white_win_rate = (sum_tuned_white_wins / sum_tuned_white_games) if sum_tuned_white_games > 0 else None
            tuned_black_win_rate = (sum_tuned_black_wins / sum_tuned_black_games) if sum_tuned_black_games > 0 else None
            balanced_win_rate = None
            if tuned_white_win_rate is not None and tuned_black_win_rate is not None:
                balanced_win_rate = 0.5 * (tuned_white_win_rate + tuned_black_win_rate)
            terminal_rate = (sum_terminal / sum_games_count) if sum_games_count > 0 else None
            nonterminal_tiebreak_rate = (sum_nonterminal / sum_games_count) if sum_games_count > 0 else None
            move_cap_rate = (sum_move_cap / sum_games_count) if sum_games_count > 0 else None
            timeout_rate = (sum_timeout / sum_games_count) if sum_games_count > 0 else None
            avg_moves = (sum_moves_total / sum_games_count) if sum_games_count > 0 else None
            avg_white_score = (sum_sum_white_score / sum_games_count) if sum_games_count > 0 else None
            avg_black_score = (sum_sum_black_score / sum_games_count) if sum_games_count > 0 else None
            avg_score_diff = (avg_white_score - avg_black_score) if (avg_white_score is not None and avg_black_score is not None) else None
            no_valid_moves_rate = (sum_no_valid_moves / sum_games_count) if sum_games_count > 0 else None
            invalid_move_rate = (sum_invalid_move / sum_games_count) if sum_games_count > 0 else None
        else:
            # Fallback to legacy self-play
            mcts = self._create_mcts_with_params(params)
            move_times: List[float] = []
            wins = 0
            draws = 0
            games_played = 0
            for game_id in range(games_budget):
                try:
                    game_start = time.time()
                    game_result = self._play_evaluation_game(mcts, game_id)
                    _ = time.time() - game_start
                    move_times.extend(game_result['move_times'])
                    if game_result['outcome'] == 1.0:
                        wins += 1
                    elif game_result['outcome'] == 0.5:
                        draws += 1
                    games_played += 1
                    if self.tuning_config.enforce_game_timeout and _ > self.tuning_config.timeout_per_game:
                        self.logger.warning(f"Game {game_id} timed out")
                        break
                except Exception as e:
                    self.logger.error(f"Error in game {game_id}: {e}")
                    continue
            evaluation_time = time.time() - start_time
            win_rate = ((wins) + 0.5 * draws) / games_played if games_played > 0 else 0.0
            move_time_avg = np.mean(move_times) if move_times else float('inf')
            move_time_std = np.std(move_times) if move_times else 0.0
            tuned_white_win_rate = None
            tuned_black_win_rate = None
            balanced_win_rate = None
            terminal_rate = None
            nonterminal_tiebreak_rate = None
            move_cap_rate = None
            timeout_rate = None
            avg_moves = None
            avg_white_score = None
            avg_black_score = None
            avg_score_diff = None
            no_valid_moves_rate = None
            invalid_move_rate = None

        result = TuningResult(
            parameters=params,
            win_rate=win_rate,
            move_time_avg=move_time_avg,
            move_time_std=move_time_std,
            games_played=games_played,
            evaluation_time=evaluation_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            tuned_white_win_rate=tuned_white_win_rate,
            tuned_black_win_rate=tuned_black_win_rate,
            balanced_win_rate=balanced_win_rate,
            terminal_rate=terminal_rate,
            nonterminal_tiebreak_rate=nonterminal_tiebreak_rate,
            move_cap_rate=move_cap_rate,
            timeout_rate=timeout_rate,
            avg_moves=avg_moves,
            avg_white_score=avg_white_score,
            avg_black_score=avg_black_score,
            avg_score_diff=avg_score_diff,
            no_valid_moves_rate=no_valid_moves_rate,
            invalid_move_rate=invalid_move_rate
        )
        return result

    def _evaluate_parameters(self, params: Dict[str, Any]) -> TuningResult:
        """Evaluate a single parameter set."""
        self.logger.info(f"Evaluating parameters: {params}")
        
        start_time = time.time()
        
        use_h2h = self.tuning_config.use_head_to_head
        baselines = self.tuning_config.baselines or []

        if use_h2h and baselines:
            # Distribute the evaluation budget approximately across baselines and pairing
            per_baseline_games_budget = max(1, self.tuning_config.evaluation_games // max(1, len(baselines)))
            games_per_seed = 2 if self.tuning_config.paired_colors else 1
            seeds = max(1, per_baseline_games_budget // games_per_seed)

            total_wins = 0
            total_draws = 0
            total_games = 0
            tuned_move_times_all: List[float] = []
            # Diagnostics accumulators
            sum_tuned_white_wins = 0
            sum_tuned_black_wins = 0
            sum_tuned_white_games = 0
            sum_tuned_black_games = 0
            sum_terminal = 0
            sum_nonterminal = 0
            sum_move_cap = 0
            sum_timeout = 0
            sum_moves_total = 0
            sum_games_count = 0

            for base in baselines:
                wins_b, draws_b, games_b, tuned_times_b, stats_b = self._play_h2h_series(
                    tuned_params=params,
                    baseline_params=base,
                    seeds=seeds
                )
                total_wins += wins_b
                total_draws += draws_b
                total_games += games_b
                tuned_move_times_all.extend(tuned_times_b)
                # Aggregate diagnostics
                sum_tuned_white_wins += stats_b['tuned_white_wins']
                sum_tuned_black_wins += stats_b['tuned_black_wins']
                sum_tuned_white_games += stats_b['tuned_white_games']
                sum_tuned_black_games += stats_b['tuned_black_games']
                sum_terminal += stats_b['terminal_state']
                sum_nonterminal += stats_b['non_terminal_tiebreak']
                sum_move_cap += stats_b['move_cap']
                sum_timeout += stats_b['timeout']
                sum_moves_total += stats_b['moves_total']
                sum_games_count += stats_b['games_count']

            evaluation_time = time.time() - start_time

            # Win rate with draws as 0.5
            win_rate = ((total_wins) + 0.5 * total_draws) / total_games if total_games > 0 else 0.0
            move_time_avg = np.mean(tuned_move_times_all) if tuned_move_times_all else float('inf')
            move_time_std = np.std(tuned_move_times_all) if tuned_move_times_all else 0.0
            games_played = total_games
            # Compute diagnostics
            tuned_white_win_rate = (sum_tuned_white_wins / sum_tuned_white_games) if sum_tuned_white_games > 0 else None
            tuned_black_win_rate = (sum_tuned_black_wins / sum_tuned_black_games) if sum_tuned_black_games > 0 else None
            # Derive balanced metric here for logging and result
            balanced_win_rate = None
            if tuned_white_win_rate is not None and tuned_black_win_rate is not None:
                balanced_win_rate = 0.5 * (tuned_white_win_rate + tuned_black_win_rate)
            terminal_rate = (sum_terminal / sum_games_count) if sum_games_count > 0 else None
            nonterminal_tiebreak_rate = (sum_nonterminal / sum_games_count) if sum_games_count > 0 else None
            move_cap_rate = (sum_move_cap / sum_games_count) if sum_games_count > 0 else None
            timeout_rate = (sum_timeout / sum_games_count) if sum_games_count > 0 else None
            avg_moves = (sum_moves_total / sum_games_count) if sum_games_count > 0 else None
            # Scores not tracked in this code path; leave as None
            avg_white_score = None
            avg_black_score = None
            avg_score_diff = None
        else:
            # Fallback: self-play with single MCTS (legacy behavior)
            mcts = self._create_mcts_with_params(params)

            move_times = []
            wins = 0
            draws = 0
            games_played = 0

            for game_id in range(self.tuning_config.evaluation_games):
                try:
                    game_start = time.time()
                    game_result = self._play_evaluation_game(mcts, game_id)
                    _ = time.time() - game_start
                    move_times.extend(game_result['move_times'])

                    if game_result['outcome'] == 1.0:
                        wins += 1
                    elif game_result['outcome'] == 0.5:
                        draws += 1
                    games_played += 1

                    if self.tuning_config.enforce_game_timeout and _ > self.tuning_config.timeout_per_game:
                        self.logger.warning(f"Game {game_id} timed out")
                        break
                except Exception as e:
                    self.logger.error(f"Error in game {game_id}: {e}")
                    continue

            evaluation_time = time.time() - start_time
            win_rate = ((wins) + 0.5 * draws) / games_played if games_played > 0 else 0.0
            move_time_avg = np.mean(move_times) if move_times else float('inf')
            move_time_std = np.std(move_times) if move_times else 0.0
            tuned_white_win_rate = None
            tuned_black_win_rate = None
            terminal_rate = None
            nonterminal_tiebreak_rate = None
            move_cap_rate = None
            timeout_rate = None
            avg_moves = None
        
        result = TuningResult(
            parameters=params,
            win_rate=win_rate,
            move_time_avg=move_time_avg,
            move_time_std=move_time_std,
            games_played=games_played,
            evaluation_time=evaluation_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            tuned_white_win_rate=tuned_white_win_rate,
            tuned_black_win_rate=tuned_black_win_rate,
            balanced_win_rate=balanced_win_rate,
            terminal_rate=terminal_rate,
            nonterminal_tiebreak_rate=nonterminal_tiebreak_rate,
            move_cap_rate=move_cap_rate,
            timeout_rate=timeout_rate,
            avg_moves=avg_moves
        )
        
        # Compact diagnostic summary
        if tuned_white_win_rate is not None and tuned_black_win_rate is not None and terminal_rate is not None:
            self.logger.info(
                f"Results: Win rate={win_rate:.3f}, Balanced={(0.5*(tuned_white_win_rate+tuned_black_win_rate)):.3f}, W/B={tuned_white_win_rate:.3f}/{tuned_black_win_rate:.3f}, "
                f"Terminal={terminal_rate:.2f}, NTiebreak={nonterminal_tiebreak_rate:.2f}, "
                f"Cap={move_cap_rate:.2f}, TO={timeout_rate:.2f}, AvgMoves={avg_moves:.1f}, "
                f"Avg move time={move_time_avg:.3f}s"
            )
        else:
            self.logger.info(f"Results: Win rate={win_rate:.3f}, Avg move time={move_time_avg:.3f}s")
        
        return result
    
    def _play_evaluation_game(self, mcts: EnhancedMCTS, game_id: int) -> Dict[str, Any]:
        """Play a single evaluation game."""
        state = GameState()
        move_times = []
        move_count = 0
        
        cap = self.tuning_config.max_moves_per_game
        while not state.is_terminal():
            move_start = time.time()
            
            # Get move from MCTS
            policy = mcts.search(state, move_count + 1)
            
            # Select move (greedy selection for evaluation)
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                break
            
            # Select move with highest probability
            move_probs = []
            for move in valid_moves:
                move_idx = mcts.state_encoder.move_to_index(move)
                if 0 <= move_idx < len(policy):
                    move_probs.append(policy[move_idx])
                else:
                    move_probs.append(0.0)
            
            if move_probs and max(move_probs) > 0:
                selected_idx = np.argmax(move_probs)
                selected_move = valid_moves[selected_idx]
            else:
                selected_move = valid_moves[0]  # Fallback
            
            # Make move
            state.make_move(selected_move)
            move_count += 1
            
            move_time = time.time() - move_start
            move_times.append(move_time)
            
            # Check move time limit
            if self.tuning_config.enforce_move_time_cap and move_time > self.tuning_config.max_move_time:
                self.logger.warning(f"Move {move_count} exceeded time limit: {move_time:.3f}s")
                break

            if cap and cap > 0 and move_count >= cap:
                # Respect max moves only if positive
                break
        
        # Determine outcome (use heuristic tiebreak on non-terminal)
        if state.is_terminal():
            winner = state.get_winner()
            if winner == Player.WHITE:
                outcome = 1.0
            elif winner == Player.BLACK:
                outcome = -1.0
            else:
                outcome = 0.5
        else:
            # Non-terminal tiebreak: heuristic if available, else score differential
            outcome = 0.5
            evaluator = mcts.heuristic_evaluator
            if evaluator is not None:
                try:
                    white_eval = evaluator.evaluate_position_fast(state, Player.WHITE)
                    black_eval = evaluator.evaluate_position_fast(state, Player.BLACK)
                    delta = white_eval - black_eval
                    if delta > 1e-6:
                        outcome = 1.0
                    elif delta < -1e-6:
                        outcome = -1.0
                except Exception:
                    outcome = 0.5
            if outcome == 0.5:
                try:
                    white_score = getattr(state, 'white_score', None)
                    black_score = getattr(state, 'black_score', None)
                    if white_score is not None and black_score is not None:
                        if white_score > black_score:
                            outcome = 1.0
                        elif black_score > white_score:
                            outcome = -1.0
                except Exception:
                    pass
        
        return {
            'outcome': outcome,
            'move_times': move_times,
            'total_moves': move_count
        }
    
    def grid_search(self) -> List[TuningResult]:
        """Perform grid search over hyperparameter space."""
        self.logger.info("Starting grid search hyperparameter tuning")
        
        # Get parameter combinations
        param_combinations = self.hyperparameter_space.get_grid_search_space()
        total_combinations = len(param_combinations)
        
        self.logger.info(f"Testing {total_combinations} parameter combinations")
        
        # Evaluate parameters in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.tuning_config.max_workers) as executor:
            # Submit all evaluation tasks
            future_to_params = {
                executor.submit(self._evaluate_parameters, params): params 
                for params in param_combinations
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_params):
                try:
                    result = future.result()
                    results.append(result)
                    self.results.append(result)
                    
                    self.logger.info(f"Completed {len(results)}/{total_combinations} evaluations")
                    # Periodic autosave to avoid losing long-run progress
                    if len(self.results) % 25 == 0:
                        try:
                            self.save_results('mcts_tuning_autosave.json')
                        except Exception as e:
                            self.logger.warning(f"Autosave failed: {e}")
                    
                except Exception as e:
                    params = future_to_params[future]
                    self.logger.error(f"Error evaluating parameters {params}: {e}")
        
        # Sort results by color-balanced win rate when available
        def _sel(r):
            if r.tuned_white_win_rate is not None and r.tuned_black_win_rate is not None:
                return 0.5 * (r.tuned_white_win_rate + r.tuned_black_win_rate)
            return r.win_rate
        results.sort(key=_sel, reverse=True)
        
        # Find best result
        if results:
            self.best_result = results[0]
            self.logger.info(f"Best parameters: {self.best_result.parameters}")
            self.logger.info(f"Best win rate: {self.best_result.win_rate:.3f}")
            self.logger.info(f"Best move time: {self.best_result.move_time_avg:.3f}s")
        
        return results

    def successive_halving(self) -> List[TuningResult]:
        """Perform Successive Halving over the grid using head-to-head evaluation.

        Uses games budget per config that multiplies by eta each round and
        prunes the bottom (1 - 1/eta) fraction.
        """
        self.logger.info("Starting Successive Halving hyperparameter tuning")

        params_all = self.hyperparameter_space.get_grid_search_space()
        n0 = len(params_all)
        if n0 == 0:
            return []

        eta = max(2, int(self.tuning_config.sh_eta))
        initial_games = int(self.tuning_config.sh_initial_games or self.tuning_config.evaluation_games)

        # Derive number of rounds if not provided
        if self.tuning_config.sh_max_rounds is not None:
            rounds = int(self.tuning_config.sh_max_rounds)
        else:
            # s_max = floor(log_eta(n0))
            s_max = 0
            temp = n0
            while temp >= eta:
                temp //= eta
                s_max += 1
            rounds = max(1, s_max + 1)

        current_params = params_all
        all_results: List[TuningResult] = []

        for r in range(rounds):
            budget = max(1, initial_games * (eta ** r))
            self.logger.info(f"SH Round {r+1}/{rounds}: candidates={len(current_params)}, games_per_config={budget}")

            # Evaluate all current params in parallel with this round's budget
            round_results: List[TuningResult] = []
            with ThreadPoolExecutor(max_workers=self.tuning_config.max_workers) as executor:
                future_to_params = {
                    executor.submit(self._evaluate_with_budget, p, budget): p for p in current_params
                }
                for future in as_completed(future_to_params):
                    try:
                        res = future.result()
                        round_results.append(res)
                        self.results.append(res)
                        if len(self.results) % 25 == 0:
                            try:
                                self.save_results('mcts_tuning_autosave.json')
                            except Exception as e:
                                self.logger.warning(f"Autosave failed: {e}")
                    except Exception as e:
                        p = future_to_params[future]
                        self.logger.error(f"Error evaluating parameters {p}: {e}")

            # Rank and select top 1/eta using color-balanced metric
            def _sel(r):
                if r.tuned_white_win_rate is not None and r.tuned_black_win_rate is not None:
                    return 0.5 * (r.tuned_white_win_rate + r.tuned_black_win_rate)
                return r.win_rate
            round_results.sort(key=_sel, reverse=True)
            keep = max(1, len(round_results) // eta)
            current_params = [res.parameters for res in round_results[:keep]]
            all_results.extend(round_results)

            self.logger.info(f"SH Round {r+1}: kept {keep} configs")

            if keep <= 1:
                break

        # Final selection
        if all_results:
            def _sel(r):
                if r.tuned_white_win_rate is not None and r.tuned_black_win_rate is not None:
                    return 0.5 * (r.tuned_white_win_rate + r.tuned_black_win_rate)
                return r.win_rate
            all_results.sort(key=_sel, reverse=True)
            self.best_result = all_results[0]
            self.logger.info(f"Best parameters (SH): {self.best_result.parameters}")
            self.logger.info(f"Best win rate (SH): {self.best_result.win_rate:.3f}")

        return all_results
    
    def bayesian_optimization(self) -> List[TuningResult]:
        """Perform Bayesian optimization over hyperparameter space."""
        self.logger.info("Starting Bayesian optimization hyperparameter tuning")
        
        # This is a simplified implementation
        # In practice, you'd use a library like scikit-optimize or optuna
        
        results = []
        
        # Initialize with random samples
        n_init = 10
        for i in range(n_init):
            params = self._sample_random_parameters()
            result = self._evaluate_parameters(params)
            results.append(result)
        
        # Bayesian optimization loop
        for iteration in range(self.tuning_config.bayesian_iterations):
            # Fit Gaussian Process
            X = np.array([[r.parameters['c_puct'], r.parameters['heuristic_alpha'], 
                          r.parameters['epsilon_greedy']] for r in results])
            y = np.array([r.win_rate for r in results])
            
            # Simple RBF kernel
            kernel = C(1.0) * RBF(1.0)
            gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
            gp.fit(X, y)
            
            # Acquisition function (simplified)
            # In practice, use Expected Improvement or other acquisition functions
            next_params = self._sample_random_parameters()
            
            # Evaluate next parameters
            result = self._evaluate_parameters(next_params)
            results.append(result)
            
            self.logger.info(f"Bayesian iteration {iteration + 1}/{self.tuning_config.bayesian_iterations}")
        
        # Sort results by color-balanced win rate when available
        def _sel(r):
            if r.tuned_white_win_rate is not None and r.tuned_black_win_rate is not None:
                return 0.5 * (r.tuned_white_win_rate + r.tuned_black_win_rate)
            return r.win_rate
        results.sort(key=_sel, reverse=True)
        
        # Find best result
        if results:
            self.best_result = results[0]
            self.logger.info(f"Best parameters: {self.best_result.parameters}")
            self.logger.info(f"Best win rate: {self.best_result.win_rate:.3f}")
        
        return results
    
    def _sample_random_parameters(self) -> Dict[str, Any]:
        """Sample random parameters from hyperparameter space."""
        return {
            'c_puct': np.random.uniform(self.hyperparameter_space.c_puct_min, 
                                       self.hyperparameter_space.c_puct_max),
            'heuristic_alpha': np.random.uniform(self.hyperparameter_space.heuristic_alpha_min,
                                               self.hyperparameter_space.heuristic_alpha_max),
            'epsilon_greedy': np.random.uniform(self.hyperparameter_space.epsilon_greedy_min,
                                              self.hyperparameter_space.epsilon_greedy_max),
            'num_simulations': np.random.randint(self.hyperparameter_space.num_simulations_min,
                                               self.hyperparameter_space.num_simulations_max + 1),
            'max_depth': np.random.randint(self.hyperparameter_space.max_depth_min,
                                         self.hyperparameter_space.max_depth_max + 1)
        }
    
    def statistical_significance_test(self, result1: TuningResult, result2: TuningResult) -> Dict[str, Any]:
        """Perform statistical significance test between two results."""
        # This is a simplified implementation
        # In practice, you'd use proper statistical tests
        
        win_rate_diff = result1.win_rate - result2.win_rate
        
        # Simple significance test based on sample size
        n1, n2 = result1.games_played, result2.games_played
        
        if n1 < self.tuning_config.min_games_for_significance or n2 < self.tuning_config.min_games_for_significance:
            return {
                'significant': False,
                'reason': 'Insufficient sample size',
                'win_rate_diff': win_rate_diff,
                'n1': n1,
                'n2': n2
            }
        
        # Simplified significance test
        # In practice, use proper statistical tests like chi-square or t-test
        significant = abs(win_rate_diff) > self.tuning_config.min_win_rate_improvement
        
        return {
            'significant': significant,
            'win_rate_diff': win_rate_diff,
            'n1': n1,
            'n2': n2,
            'confidence_level': self.tuning_config.confidence_level
        }
    
    def save_results(self, filepath: str):
        """Save tuning results to file."""
        results_data = {
            'tuning_config': _to_json_safe(asdict(self.tuning_config)),
            'hyperparameter_space': _to_json_safe(asdict(self.hyperparameter_space)),
            # baseline_config may contain Enum keys (e.g., GamePhase) -> make JSON-safe
            'baseline_config': _to_json_safe(asdict(self.baseline_config)),
            'results': [_to_json_safe(result.to_dict()) for result in self.results],
            'best_result': _to_json_safe(self.best_result.to_dict()) if self.best_result else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        self.logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load tuning results from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.results = [TuningResult(**result) for result in data['results']]
        if data['best_result']:
            self.best_result = TuningResult(**data['best_result'])
        
        self.logger.info(f"Results loaded from {filepath}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive tuning report."""
        if not self.results:
            return "No results available for report generation."
        
        report = []
        report.append("MCTS Hyperparameter Tuning Report")
        report.append("=" * 50)
        report.append(f"Total parameter combinations tested: {len(self.results)}")
        report.append(f"Evaluation games per combination: {self.tuning_config.evaluation_games}")
        report.append(f"Maximum move time limit: {self.tuning_config.max_move_time}s")
        report.append("")
        
        # Best result
        if self.best_result:
            report.append("Best Parameters:")
            report.append("-" * 20)
            for param, value in self.best_result.parameters.items():
                report.append(f"  {param}: {value}")
            report.append(f"  Win Rate: {self.best_result.win_rate:.3f}")
            report.append(f"  Avg Move Time: {self.best_result.move_time_avg:.3f}s")
            report.append(f"  Games Played: {self.best_result.games_played}")
            report.append("")
        
        # Top 5 results
        report.append("Top 5 Parameter Combinations:")
        report.append("-" * 30)
        for i, result in enumerate(self.results[:5]):
            report.append(f"{i+1}. Win Rate: {result.win_rate:.3f}, Move Time: {result.move_time_avg:.3f}s")
            report.append(f"   Parameters: {result.parameters}")
            report.append("")
        
        # Statistical summary
        win_rates = [r.win_rate for r in self.results]
        move_times = [r.move_time_avg for r in self.results]
        
        report.append("Statistical Summary:")
        report.append("-" * 20)
        report.append(f"Win Rate - Mean: {np.mean(win_rates):.3f}, Std: {np.std(win_rates):.3f}")
        report.append(f"Move Time - Mean: {np.mean(move_times):.3f}s, Std: {np.std(move_times):.3f}s")
        report.append(f"Best Win Rate: {np.max(win_rates):.3f}")
        report.append(f"Best Move Time: {np.min(move_times):.3f}s")
        
        return "\n".join(report)
