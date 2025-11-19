"""Tournament evaluation utilities for heuristic agents."""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import os
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, TYPE_CHECKING, Any, Dict, Callable

from .heuristic_agent import HeuristicAgent, HeuristicAgentConfig

if TYPE_CHECKING:  # pragma: no cover
    from ..game.game_state import GameState
    from ..game.constants import Player
    from ..self_play.random_policy import RandomMovePolicy
else:
    GameState = Any
    Player = Any
    RandomMovePolicy = Any

logger = logging.getLogger(__name__)


def _play_game_worker(
    game_num: int,
    starting_player: int,
    opponent_seed: Optional[int],
    agent_config_dict: Dict[str, Any],
    agent_factory_name: str,
    opponent_factory_name: Optional[str],
) -> Dict[str, Any]:
    """Module-level worker function for parallel game execution.
    
    Args:
        game_num: Game number (for logging)
        starting_player: Starting player (1=WHITE, -1=BLACK)
        opponent_seed: Random seed for opponent
        agent_config_dict: Serialized agent configuration
        agent_factory_name: Name of agent factory function/class
        opponent_factory_name: Name of opponent factory function/class
        
    Returns:
        Dictionary with game results
    """
    # Import here to avoid pickling issues
    from ..game.game_state import GameState
    from ..game.constants import Player
    from ..game.moves import MoveGenerator
    
    # Reconstruct agent from config
    agent_config = HeuristicAgentConfig(**agent_config_dict)
    agent = HeuristicAgent(config=agent_config)
    
    # Reconstruct opponent
    if opponent_factory_name:
        # For custom opponent factories, we'd need to handle this differently
        # For now, use default random policy
        from ..self_play.random_policy import RandomMovePolicy
        from ..self_play.policies import PolicyConfig
        opponent = RandomMovePolicy(PolicyConfig(random_seed=opponent_seed))
    else:
        from ..self_play.random_policy import RandomMovePolicy
        from ..self_play.policies import PolicyConfig
        opponent = RandomMovePolicy(PolicyConfig(random_seed=opponent_seed))
    
    state = GameState()
    state.current_player = Player.WHITE if starting_player == 1 else Player.BLACK
    turn_count = 0
    move_times: List[float] = []
    max_turns = 200
    
    while turn_count < max_turns and not state.is_terminal():
        valid_moves = MoveGenerator.get_valid_moves(state.board, state)
        if not valid_moves:
            break
        
        if state.current_player == Player.WHITE:
            start = time.perf_counter()
            move = agent.select_move(state)
            move_times.append(time.perf_counter() - start)
        else:
            move = opponent.select_move(state)
        
        if not state.make_move(move):
            break
        
        turn_count += 1
    
    winner = state.get_winner()
    wins = 1 if winner == Player.WHITE else 0
    losses = 1 if winner == Player.BLACK else 0
    draws = 1 if winner is None else 0
    duration = sum(move_times)
    max_move_time = max(move_times) if move_times else 0.0
    nodes = agent.last_search_stats.get("nodes_evaluated", 0)
    num_moves = len(move_times)  # Number of moves made by heuristic agent
    
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "duration": duration,
        "max_move_time": max_move_time,
        "nodes": nodes,
        "turns": turn_count,
        "num_moves": num_moves,  # Track number of heuristic agent moves
    }


@dataclass
class TournamentConfig:
    """Configuration for large-scale tournament execution."""
    num_games: int = 1000
    """Total number of games to play."""
    
    concurrent_workers: int = 4
    """Number of parallel workers for concurrent game execution."""
    
    save_interval: int = 100
    """Save intermediate results every N games."""
    
    output_path: Optional[str] = None
    """Path to save tournament results (JSON format)."""
    
    format: str = "round-robin"
    """Tournament format: 'round-robin' or 'elimination'."""
    
    max_turns_per_game: int = 200
    """Maximum turns per game before timeout."""
    
    resume: bool = False
    """If True, resume from existing results file."""


@dataclass
class TournamentMetrics:
    total_games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    average_game_length: float
    std_game_length: float
    average_move_time: float
    max_move_time: float
    nodes_per_second: float

    @property
    def loss_rate(self) -> float:
        return self.losses / self.total_games if self.total_games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.total_games if self.total_games else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TournamentMetrics:
        """Create metrics from dictionary."""
        return cls(**data)


class TournamentEvaluator:
    """Utility for running head-to-head tournaments."""

    def __init__(
        self,
        heuristic_agent_factory=HeuristicAgent,
        heuristic_config: Optional[HeuristicAgentConfig] = None,
        opponent_factory: Optional[callable] = None,
    ) -> None:
        self._agent_factory = heuristic_agent_factory
        self._agent_config = heuristic_config or HeuristicAgentConfig()
        self._opponent_factory = opponent_factory
        self._results_cache: Optional[Dict[str, Any]] = None

    def play_game(
        self,
        heuristic_agent: HeuristicAgent,
        opponent_policy: RandomMovePolicy,
        starting_player: Player,
        max_turns: int = 200,
    ) -> tuple[int, float, float]:
        from ..game.game_state import GameState
        from ..game.constants import Player
        from ..game.moves import MoveGenerator

        state = GameState()
        state.current_player = starting_player
        turn_count = 0
        move_times: List[float] = []

        while turn_count < max_turns and not state.is_terminal():
            valid_moves = MoveGenerator.get_valid_moves(state.board, state)
            if not valid_moves:
                break

            if state.current_player == Player.WHITE:
                start = time.perf_counter()
                move = heuristic_agent.select_move(state)
                move_times.append(time.perf_counter() - start)
            else:
                move = opponent_policy.select_move(state)

            if not state.make_move(move):
                break

            turn_count += 1

        winner = state.get_winner()
        wins = 1 if winner == Player.WHITE else 0
        losses = 1 if winner == Player.BLACK else 0
        draws = 1 if winner is None else 0
        duration = sum(move_times)
        max_move_time = max(move_times) if move_times else 0.0
        nodes = heuristic_agent.last_search_stats.get("nodes_evaluated", 0)
        return (
            (wins, losses, draws),
            duration,
            max_move_time,
            nodes,
            turn_count,
        )

    def run_tournament(
        self,
        games: int = 50,
        opponent_seed: Optional[int] = 1234,
    ) -> TournamentMetrics:
        if self._opponent_factory is not None:
            opponent_policy = self._opponent_factory()
        else:
            from ..self_play.random_policy import RandomMovePolicy
            from ..self_play.policies import PolicyConfig

            opponent_policy = RandomMovePolicy(PolicyConfig(random_seed=opponent_seed))
        heuristic_agent = self._agent_factory(self._agent_config)
        wins = losses = draws = 0
        total_duration = 0.0
        max_move_time = 0.0
        total_nodes = 0
        lengths: List[int] = []

        # Use stub player values for alternation (actual Player imported in play_game)
        PLAYER_WHITE = 1
        PLAYER_BLACK = -1

        for idx in range(games):
            starting_player = PLAYER_WHITE if idx % 2 == 0 else PLAYER_BLACK
            (wld, duration, peak, nodes, turns) = self.play_game(
                heuristic_agent,
                opponent_policy,
                starting_player=starting_player,
            )
            win, loss, draw = wld
            wins += win
            losses += loss
            draws += draw
            total_duration += duration
            max_move_time = max(max_move_time, peak)
            total_nodes += nodes
            lengths.append(turns)

        win_rate = wins / games if games else 0.0
        avg_length = statistics.mean(lengths) if lengths else 0.0
        std_length = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
        avg_move_time = total_duration / (wins + losses + draws) if games else 0.0
        nodes_per_second = total_nodes / total_duration if total_duration else 0.0

        return TournamentMetrics(
            total_games=games,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=win_rate,
            average_game_length=avg_length,
            std_game_length=std_length,
            average_move_time=avg_move_time,
            max_move_time=max_move_time,
            nodes_per_second=nodes_per_second,
        )
    
    def run_large_scale_tournament(
        self,
        config: TournamentConfig,
        opponent_seed: Optional[int] = 1234,
    ) -> TournamentMetrics:
        """Run a large-scale tournament with concurrent execution and persistence.
        
        Args:
            config: Tournament configuration
            opponent_seed: Random seed for opponent policy
            
        Returns:
            Aggregated tournament metrics
        """
        # Load existing results if resuming
        start_game = 0
        accumulated_metrics = None
        if config.resume and config.output_path and os.path.exists(config.output_path):
            accumulated_metrics = self._load_results(config.output_path)
            if accumulated_metrics:
                start_game = accumulated_metrics.total_games
                logger.info(f"Resuming tournament from game {start_game}")
        
        # Initialize accumulated metrics if not resuming
        if accumulated_metrics is None:
            accumulated_metrics = TournamentMetrics(
                total_games=0,
                wins=0,
                losses=0,
                draws=0,
                win_rate=0.0,
                average_game_length=0.0,
                std_game_length=0.0,
                average_move_time=0.0,
                max_move_time=0.0,
                nodes_per_second=0.0,
            )
        
        remaining_games = config.num_games - start_game
        if remaining_games <= 0:
            logger.info("Tournament already complete")
            return accumulated_metrics
        
        # Prepare game arguments for parallel execution
        agent_config_dict = asdict(self._agent_config)
        opponent_factory_name = None
        if self._opponent_factory is not None:
            opponent_factory_name = getattr(self._opponent_factory, "__name__", None)
        
        game_args = []
        for idx in range(remaining_games):
            game_num = start_game + idx
            starting_player = 1 if game_num % 2 == 0 else -1  # WHITE/BLACK alternation
            game_args.append((
                game_num,
                starting_player,
                opponent_seed,
                agent_config_dict,
                "HeuristicAgent",  # agent_factory_name
                opponent_factory_name,
            ))
        
        # Run games concurrently
        logger.info(f"Starting {remaining_games} games with {config.concurrent_workers} workers")
        start_time = time.perf_counter()
        
        with multiprocessing.Pool(processes=config.concurrent_workers) as pool:
            results = []
            for i, result in enumerate(pool.starmap(
                _play_game_worker,
                game_args
            ), start=start_game):
                results.append(result)
                
                # Save intermediate results periodically
                if (i + 1) % config.save_interval == 0:
                    partial_metrics = self._aggregate_results(results, start_game)
                    self._save_results(config.output_path, partial_metrics)
                    elapsed = time.perf_counter() - start_time
                    rate = (i + 1 - start_game) / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {i + 1}/{config.num_games} games "
                        f"({rate:.1f} games/sec)"
                    )
        
        # Aggregate final results
        final_metrics = self._aggregate_results(results, start_game, accumulated_metrics)
        
        # Save final results
        if config.output_path:
            self._save_results(config.output_path, final_metrics)
        
        elapsed = time.perf_counter() - start_time
        logger.info(
            f"Tournament complete: {final_metrics.total_games} games in {elapsed:.1f}s "
            f"({final_metrics.total_games / elapsed:.1f} games/sec)"
        )
        
        return final_metrics
    
    def _aggregate_results(
        self,
        results: List[Dict[str, Any]],
        start_game: int = 0,
        existing_metrics: Optional[TournamentMetrics] = None,
    ) -> TournamentMetrics:
        """Aggregate game results into tournament metrics.
        
        Args:
            results: List of game result dictionaries
            start_game: Starting game number (for total count)
            existing_metrics: Existing metrics to merge with
            
        Returns:
            Aggregated tournament metrics
        """
        if existing_metrics:
            wins = existing_metrics.wins
            losses = existing_metrics.losses
            draws = existing_metrics.draws
            # Reconstruct total_duration from average_move_time and estimated total moves
            # Estimate total moves from average game length (heuristic agent plays ~half)
            estimated_total_moves = (existing_metrics.average_game_length * (wins + losses + draws)) // 2
            total_duration = existing_metrics.average_move_time * estimated_total_moves if estimated_total_moves > 0 else 0.0
            max_move_time = existing_metrics.max_move_time
            total_nodes = existing_metrics.nodes_per_second * total_duration if total_duration > 0 else 0
            total_moves = estimated_total_moves
            lengths = []
        else:
            wins = losses = draws = 0
            total_duration = 0.0
            max_move_time = 0.0
            total_nodes = 0
            total_moves = 0
            lengths = []
        
        for result in results:
            wins += result["wins"]
            losses += result["losses"]
            draws += result["draws"]
            total_duration += result["duration"]
            max_move_time = max(max_move_time, result["max_move_time"])
            total_nodes += result["nodes"]
            lengths.append(result["turns"])
            # Use actual move count if available, otherwise estimate
            num_moves = result.get("num_moves", result["turns"] // 2)
            total_moves += num_moves
        
        total_games = wins + losses + draws
        win_rate = wins / total_games if total_games > 0 else 0.0
        avg_length = statistics.mean(lengths) if lengths else 0.0
        std_length = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
        # Calculate average move time: total duration divided by total moves made by heuristic agent
        avg_move_time = total_duration / total_moves if total_moves > 0 else 0.0
        nodes_per_second = total_nodes / total_duration if total_duration > 0 else 0.0
        
        return TournamentMetrics(
            total_games=total_games,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=win_rate,
            average_game_length=avg_length,
            std_game_length=std_length,
            average_move_time=avg_move_time,
            max_move_time=max_move_time,
            nodes_per_second=nodes_per_second,
        )
    
    def _save_results(self, output_path: Optional[str], metrics: TournamentMetrics) -> None:
        """Save tournament results to file.
        
        Args:
            output_path: Path to save results (JSON format)
            metrics: Tournament metrics to save
        """
        if not output_path:
            return
        
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "metrics": metrics.to_dict(),
                "timestamp": time.time(),
            }
            
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved tournament results to {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save tournament results: {e}")
    
    def _load_results(self, output_path: str) -> Optional[TournamentMetrics]:
        """Load tournament results from file.
        
        Args:
            output_path: Path to results file
            
        Returns:
            Tournament metrics if file exists and is valid, None otherwise
        """
        try:
            with open(output_path, "r") as f:
                data = json.load(f)
            
            if "metrics" in data:
                return TournamentMetrics.from_dict(data["metrics"])
            return None
        except Exception as e:
            logger.warning(f"Failed to load tournament results: {e}")
            return None

