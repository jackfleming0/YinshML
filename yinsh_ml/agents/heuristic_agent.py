"""Standalone heuristic-driven agent for Yinsh.

This module provides a lightweight search agent that relies exclusively on the
existing `YinshHeuristics` evaluator.  It is intended as a strong baseline for
tests, benchmarking, and heuristic-guided self-play without invoking the neural
network or MCTS pipeline.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List, TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..game.game_state import GameState
    from ..game.types import Move
    from ..game.constants import Player
    from ..heuristics import YinshHeuristics
else:
    GameState = Any
    Move = Any
    Player = Any
    YinshHeuristics = Any


logger = logging.getLogger(__name__)


@dataclass
class HeuristicAgentConfig:
    """Configuration options for :class:`HeuristicAgent`."""

    min_depth: int = 1
    """Minimum search depth (plies) that must be fully evaluated before timing out."""

    max_depth: int = 3
    """Maximum search depth (plies) for negamax search."""

    time_limit_seconds: float = 1.0
    """Soft wall-clock budget per move. Set <=0 for no limit."""

    time_buffer_seconds: float = 0.01
    """Grace period retained from the time budget to allow orderly shutdown."""

    max_branching_factor: Optional[int] = 24
    """Optional cap on candidate moves searched after initial ordering."""

    use_iterative_deepening: bool = True
    """Enable iterative-deepening search up to ``max_depth``."""

    score_cap: float = 10_000.0
    """Clamp heuristic scores to avoid runaway values."""

    random_tiebreak: bool = True
    """Perturb equal scores slightly to reduce deterministic oscillation."""

    debug: bool = False
    """Emit detailed timing information via the agent's `last_search_stats`."""

    slow_warning_threshold: float = 0.75
    """Emit debug warnings when a single depth takes longer than this many seconds."""

    random_seed: Optional[int] = None
    """Optional seed for deterministic fallback behaviour."""


class HeuristicAgent:
    """Baseline agent that selects moves using heuristic-guided search."""

    def __init__(
        self,
        config: Optional[HeuristicAgentConfig] = None,
        evaluator: Optional["YinshHeuristics"] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.config = config or HeuristicAgentConfig()
        self._validate_config()

        self._rng = rng or random.Random(self.config.random_seed)
        if evaluator is not None:
            self._evaluator = evaluator
        else:
            self._evaluator = self._create_default_evaluator()

        self.last_search_stats: dict = {}
        self._nodes_searched: int = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def select_move(self, game_state: Any) -> Move:
        """Return the best move found for the provided ``game_state``."""
        if not hasattr(game_state, "get_valid_moves"):
            raise TypeError("game_state must provide get_valid_moves()")
        if not hasattr(game_state, "copy") or not callable(game_state.copy):
            raise TypeError("game_state must provide copy()")
        if not hasattr(game_state, "make_move"):
            raise TypeError("game_state must provide make_move()")
        if not hasattr(game_state, "is_terminal"):
            raise TypeError("game_state must provide is_terminal()")
        if not hasattr(game_state, "current_player"):
            raise TypeError("game_state must expose current_player")

        valid_moves = list(game_state.get_valid_moves())
        if not valid_moves:
            raise ValueError("No legal moves available for the current state.")

        start = time.perf_counter()
        player = game_state.current_player
        self._nodes_searched = 0

        ordered_moves = self._order_moves(game_state, valid_moves, player, start)
        if not ordered_moves:
            ordered_moves = list(valid_moves)
        if self.config.max_branching_factor is not None:
            ordered_moves = ordered_moves[: self.config.max_branching_factor]

        (
            best_move,
            best_score,
            depth_reached,
            timed_out,
            depth_metrics,
        ) = self._iterative_deepening_search(
            game_state,
            ordered_moves,
            player,
            start,
        )

        # Fallback if the search failed to identify a move within constraints.
        if best_move is None:
            best_move = self._fallback_move(game_state, valid_moves, player, start)
            best_score = float("nan")
            timed_out = True

        duration = time.perf_counter() - start
        self._record_stats(
            best_move=best_move,
            score=best_score,
            depth_reached=depth_reached,
            move_count=len(ordered_moves),
            timed_out=timed_out,
            duration=duration,
            depth_metrics=depth_metrics,
        )
        return best_move

    def get_move(self, game_state: Any) -> Move:
        """Alias for :meth:`select_move` (keeps interface parity with policies)."""
        return self.select_move(game_state)

    # ------------------------------------------------------------------ #
    # Search orchestration
    # ------------------------------------------------------------------ #
    def _iterative_deepening_search(
        self,
        root_state: Any,
        moves: Sequence[Move],
        perspective: Player,
        start_time: float,
    ) -> Tuple[Optional[Move], float, int, bool, List[dict]]:
        """Run iterative deepening (when enabled) and return best move info."""
        best_move: Optional[Move] = None
        best_score: float = -math.inf
        last_completed_depth = 0
        timed_out = False
        depth_metrics: List[dict] = []

        if self.config.use_iterative_deepening:
            depth_range = range(1, self.config.max_depth + 1)
        else:
            depth_range = range(self.config.max_depth, self.config.max_depth + 1)

        for depth in depth_range:
            enforce_completion = depth <= self.config.min_depth
            if self._is_time_exceeded(start_time, allow_overrun=enforce_completion):
                timed_out = True
                break

            depth_timer_start = time.perf_counter()
            nodes_before_depth = self._nodes_searched
            depth_best: Optional[Move] = None
            depth_score = -math.inf
            completed_depth = True

            for move in moves:
                if self._is_time_exceeded(start_time, allow_overrun=enforce_completion):
                    timed_out = True
                    completed_depth = False
                    break

                score, finished = self._evaluate_move(
                    root_state,
                    move,
                    depth,
                    perspective,
                    start_time,
                    enforce_completion,
                )
                if not finished:
                    completed_depth = False
                    timed_out = True

                if score > depth_score or (
                    self.config.random_tiebreak
                    and math.isclose(score, depth_score)
                    and self._rng.random() < 0.5
                ):
                    depth_score = score
                    depth_best = move

            depth_duration = time.perf_counter() - depth_timer_start
            nodes_this_depth = self._nodes_searched - nodes_before_depth
            depth_completed = completed_depth and depth_best is not None

            depth_metrics.append(
                {
                    "depth": depth,
                    "completed": depth_completed,
                    "duration_seconds": depth_duration,
                    "nodes": nodes_this_depth,
                    "best_score": depth_score if depth_best is not None else None,
                }
            )

            if self.config.debug:
                logger.debug(
                    "HeuristicAgent depth %d completed=%s nodes=%d duration=%.4fs best=%s",
                    depth,
                    depth_completed,
                    nodes_this_depth,
                    depth_duration,
                    f"{depth_score:.3f}" if depth_best is not None else "N/A",
                )
                if depth_duration >= self.config.slow_warning_threshold:
                    logger.warning(
                        "HeuristicAgent depth %d exceeded slow threshold (%.4fs >= %.4fs)",
                        depth,
                        depth_duration,
                        self.config.slow_warning_threshold,
                    )

            if depth_completed:
                best_move = depth_best
                best_score = depth_score
                last_completed_depth = depth
            elif timed_out and self.config.debug:
                logger.debug(
                    "HeuristicAgent depth %d aborted due to time limit (elapsed %.4fs of %.4fs).",
                    depth,
                    time.perf_counter() - start_time,
                    self.config.time_limit_seconds,
                )

            if not completed_depth:
                break

        return best_move, best_score, last_completed_depth, timed_out, depth_metrics

    def _evaluate_move(
        self,
        game_state: Any,
        move: Move,
        depth: int,
        perspective: Player,
        start_time: float,
        enforce_completion: bool,
    ) -> Tuple[float, bool]:
        """Evaluate a candidate move using negamax search."""
        state_copy = game_state.copy()
        if not state_copy.make_move(move):
            return -math.inf, True

        search_depth = max(depth - 1, 0)
        score, complete = self._negamax(
            state_copy,
            search_depth,
            -self.config.score_cap,
            self.config.score_cap,
            perspective,
            start_time,
            enforce_completion,
        )
        return score, complete

    def _negamax(
        self,
        state: Any,
        depth: int,
        alpha: float,
        beta: float,
        perspective: Player,
        start_time: float,
        allow_overrun: bool,
    ) -> Tuple[float, bool]:
        """Negamax search with alpha-beta pruning and time checks."""
        if self._is_time_exceeded(start_time, allow_overrun=allow_overrun):
            return 0.0, False

        self._nodes_searched += 1

        if depth == 0 or state.is_terminal():
            return self._evaluate_position(state, perspective), True

        moves = state.get_valid_moves()
        if not moves:
            return self._evaluate_position(state, perspective), True

        best_value = -math.inf
        all_complete = True

        for move in moves:
            if self._is_time_exceeded(start_time, allow_overrun=allow_overrun):
                return (best_value if best_value > -math.inf else 0.0), False

            child = state.copy()
            if not child.make_move(move):
                continue

            value, child_complete = self._negamax(
                child,
                depth - 1,
                -beta,
                -alpha,
                perspective,
                start_time,
                allow_overrun,
            )
            if not child_complete:
                all_complete = False

            value = -value
            best_value = max(best_value, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        if best_value == -math.inf:
            best_value = self._evaluate_position(state, perspective)

        return best_value, all_complete

    # ------------------------------------------------------------------ #
    # Move ordering and fallbacks
    # ------------------------------------------------------------------ #
    def _order_moves(
        self,
        game_state: Any,
        moves: Sequence[Move],
        perspective: Player,
        start_time: float,
    ) -> List[Move]:
        """Rank moves with a shallow heuristic rollout for move ordering."""
        scored: List[Tuple[float, Move]] = []

        for move in moves:
            if self._is_time_exceeded(start_time):
                break

            state_copy = game_state.copy()
            if not state_copy.make_move(move):
                continue

            score = self._evaluate_position(state_copy, perspective)
            scored.append((score, move))

        if not scored:
            return list(moves)

        scored.sort(key=lambda item: item[0], reverse=True)
        return [move for _, move in scored]

    def _fallback_move(
        self,
        game_state: Any,
        moves: Sequence[Move],
        perspective: Player,
        start_time: float,
    ) -> Move:
        """Choose a move when the primary search fails to finish."""
        ordered = self._order_moves(game_state, moves, perspective, start_time)
        if ordered:
            return ordered[0]
        return moves[0]

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #
    def _evaluate_position(self, state: Any, perspective: Player) -> float:
        """Evaluate a position, clamped to the configured score window."""
        score = self._evaluator.evaluate_position(state, perspective)
        return max(-self.config.score_cap, min(self.config.score_cap, score))

    def _is_time_exceeded(self, start_time: float, allow_overrun: bool = False) -> bool:
        """Return True if the configured time budget has been exhausted."""
        if self.config.time_limit_seconds is None or self.config.time_limit_seconds <= 0:
            return False
        elapsed = time.perf_counter() - start_time
        if allow_overrun:
            return elapsed >= self.config.time_limit_seconds
        threshold = max(0.0, self.config.time_limit_seconds - self.config.time_buffer_seconds)
        return elapsed >= threshold

    def _record_stats(
        self,
        best_move: Move,
        score: float,
        depth_reached: int,
        move_count: int,
        timed_out: bool,
        duration: float,
        depth_metrics: List[dict],
    ) -> None:
        """Store debugging information for consumers that want diagnostics."""
        time_budget = self.config.time_limit_seconds if self.config.time_limit_seconds > 0 else None
        time_remaining = None
        if time_budget is not None:
            time_remaining = max(0.0, time_budget - duration)
        nodes_per_second = self._nodes_searched / duration if duration > 0 else None

        self.last_search_stats = {
            "best_move": best_move,
            "best_move_str": str(best_move),
            "score": score,
            "depth_reached": depth_reached,
            "nodes_evaluated": self._nodes_searched,
            "candidate_moves": move_count,
            "timed_out": timed_out,
            "duration_seconds": duration,
            "depth_metrics": depth_metrics,
            "last_completed_depth": depth_reached,
            "time_budget_seconds": time_budget,
            "time_remaining_seconds": time_remaining,
            "nodes_per_second": nodes_per_second,
        }
        if not self.config.debug:
            return
        logger.debug(
            "HeuristicAgent summary: depth=%d nodes=%d duration=%.4fs timed_out=%s",
            depth_reached,
            self._nodes_searched,
            duration,
            timed_out,
        )
        if nodes_per_second:
            logger.debug("HeuristicAgent throughput: %.1f nodes/sec", nodes_per_second)

    def _validate_config(self) -> None:
        """Ensure configuration values are within expected bounds."""
        if self.config.min_depth <= 0:
            raise ValueError("min_depth must be positive.")
        if self.config.max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        if self.config.min_depth > self.config.max_depth:
            raise ValueError("min_depth cannot exceed max_depth.")
        if self.config.score_cap <= 0:
            raise ValueError("score_cap must be positive.")
        if self.config.max_branching_factor is not None and self.config.max_branching_factor <= 0:
            raise ValueError("max_branching_factor must be positive when provided.")
        if self.config.time_buffer_seconds < 0:
            raise ValueError("time_buffer_seconds cannot be negative.")
        if self.config.slow_warning_threshold < 0:
            raise ValueError("slow_warning_threshold cannot be negative.")

    def _create_default_evaluator(self) -> "YinshHeuristics":
        """Instantiate the default heuristic evaluator lazily."""
        from ..heuristics import YinshHeuristics  # Local import to avoid heavy dependency at import time

        return YinshHeuristics()

