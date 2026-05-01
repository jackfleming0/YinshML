# training/self_play.py

import numpy as np
import torch
import time
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import logging
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor

# Force 'spawn' start method for worker processes. On Linux (where cloud
# GPU boxes run) the default is 'fork', which inherits the parent's CUDA
# context — and CUDA can't be re-initialized in a forked child, so the
# first worker to touch torch.cuda.* crashes. 'spawn' gives each worker a
# clean Python process that initializes CUDA from scratch. On macOS
# 'spawn' is already the default, so this is a no-op there.
_SELF_PLAY_MP_CONTEXT = _mp.get_context('spawn')
import time
import tempfile
import os
from pathlib import Path
import concurrent.futures
import psutil
import platform
import random
import copy

# --- YINSH ML Imports ---
# Assuming these paths are correct relative to this file
from ..utils.mcts_metrics import MCTSMetrics
from ..utils.TemperatureMetrics import TemperatureMetrics # Keep if used
from ..utils.metrics_logger import MetricsLogger, GameMetrics # Keep if used
from ..utils.encoding import StateEncoder
from ..game.game_state import GameState, GamePhase # Added GamePhase
from ..game.constants import Player, PieceType # Added PieceType
from ..network.wrapper import NetworkWrapper
from ..game.moves import Move
# --- Memory Pool Imports ---
from ..memory import GameStatePool, TensorPool, GameStatePoolConfig, TensorPoolConfig
# --- Heuristic Evaluation Import ---
from ..heuristics.evaluator import YinshHeuristics

# Configure the logger at the module level
# Use getLogger to ensure consistency if already configured elsewhere
logger = logging.getLogger("SelfPlay")
# Set default level if not configured by runner/supervisor
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
     logger.addHandler(logging.StreamHandler()) # Add console handler if none exist
     logger.setLevel(logging.INFO)


class Node:
    """Monte Carlo Tree Search node with virtual loss support for batched evaluation."""
    def __init__(self, state: GameState, parent=None, prior_prob=0.0, c_puct=1.0, state_pool=None):
        self.state = state
        self.parent = parent
        self.children = {}  # Dictionary of move -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.is_expanded = False
        self.c_puct = c_puct # Store c_puct as instance variable
        self._state_pool = state_pool  # Reference to state pool for cleanup
        self._owns_state = False  # Track if this node owns the state for cleanup

        # Virtual loss mechanism for batched MCTS
        self.virtual_losses = 0  # Track in-flight evaluations

    def __del__(self):
        """Clean up GameState when node is destroyed."""
        if self._owns_state and self._state_pool is not None and self.state is not None:
            try:
                self._state_pool.return_game_state(self.state)
            except Exception as e:
                # Use basic logging since logger might not be available
                import logging
                logging.getLogger("Node").warning(f"Failed to release state in Node destructor: {e}")

    def value(self) -> float:
        """Get mean value of node, accounting for virtual losses."""
        adjusted_visits = self.visit_count + self.virtual_losses
        if adjusted_visits == 0:
            return 0.0
        return self.value_sum / adjusted_visits

    def clear_tree(self) -> None:
        """Recursively break circular references to allow garbage collection.

        MCTS trees have parent↔child circular references that can prevent
        Python's garbage collector from freeing memory promptly. Call this
        on the root node after search completes.
        """
        # First, recursively clear all children
        for child in self.children.values():
            child.clear_tree()

        # Return state to pool if this node owns it
        if self._owns_state and self._state_pool is not None and self.state is not None:
            try:
                self._state_pool.return_game_state(self.state)
            except:
                pass
            self._owns_state = False  # Prevent double-return in __del__

        # Break circular references
        self.children.clear()
        self.parent = None
        self.state = None
        self._state_pool = None  # Clear pool reference too

    def add_virtual_loss(self) -> None:
        """Mark this node as being evaluated (used in batched MCTS)."""
        self.virtual_losses += 1

    def remove_virtual_loss(self) -> None:
        """Remove virtual loss after evaluation completes."""
        self.virtual_losses -= 1
        if self.virtual_losses < 0:
            self.virtual_losses = 0  # Safety check

    def get_ucb_score(self, parent_visit_count: int) -> float:
        """Calculate Upper Confidence Bound score, accounting for virtual losses."""
        q_value = self.value()
        # Add small epsilon to prevent division by zero, account for virtual losses
        adjusted_visits = self.visit_count + self.virtual_losses
        u_value = (self.c_puct * self.prior_prob *
                  np.sqrt(parent_visit_count) / (1 + adjusted_visits))
        return q_value + u_value

class MCTS:
    """Monte-Carlo Tree-Search with temperature annealing and ply-adaptive rollouts."""

    def __init__(self,
                 network: NetworkWrapper,
                 # --- Evaluation Mode ---
                 evaluation_mode: str = "pure_neural",  # Options: "pure_neural", "pure_heuristic", "hybrid"
                 heuristic_evaluator: Optional[YinshHeuristics] = None,
                 heuristic_weight: float = 0.5,  # Weight for heuristic in hybrid mode (0.0 = pure neural, 1.0 = pure heuristic)
                 # --- MCTS Search Params ---
                 num_simulations: int = 100, # Base simulations (early game)
                 c_puct: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 # *** Rename 'initial_value_weight' to 'value_weight' for clarity ***
                 value_weight: float = 1.0, # Weighting for value in UCB selection
                 max_depth: int = 500,
                 # --- Rollout Scheduling Params ---
                 late_simulations: Optional[int] = None, # Use Optional[int] for type hint clarity
                 simulation_switch_ply: int = 20,
                 # --- Probabilistic Fast/Slow Sim Split (RiverNewbury alphazero-general style) ---
                 # When fast_sim_prob > 0 and fast_simulations > 0, with that probability
                 # per move the per-move budget is replaced with `fast_simulations`. Default
                 # off (0.0) preserves current behavior.
                 fast_simulations: int = 0,
                 fast_sim_prob: float = 0.0,
                 # --- Temperature Params (passed down) ---
                 initial_temp: float = 1.0,
                 final_temp: float = 0.1,
                 annealing_steps: int = 30,
                 temp_clamp_fraction: float = 0.6,
                 # --- Optional Metrics ---
                 mcts_metrics: Optional[MCTSMetrics] = None,
                 # --- Memory Pools ---
                 game_state_pool: Optional[GameStatePool] = None,
                 # --- Subtree reuse ---
                 enable_subtree_reuse: bool = True,
                 # --- First-Play Urgency (PUCT) ---
                 fpu_reduction: float = 0.25,
                 # --- Root Dirichlet noise mixing ---
                 epsilon_mix_start: float = 0.25,
                 epsilon_mix_end: float = 0.0,
                 epsilon_mix_taper_moves: int = 20,
                 # --- Root policy temperature (alphazero-general style) ---
                 # T > 1 flattens root prior (more exploration); T < 1 sharpens.
                 # Applied to root child priors BEFORE Dirichlet noise mixing.
                 # 1.0 is a no-op; default off so existing configs are unchanged.
                 root_policy_temp: float = 1.0,
                ):
        """
        Initialize MCTS.

        Args:
            network: The neural network wrapper.
            evaluation_mode: Evaluation mode - "pure_neural", "pure_heuristic", or "hybrid".
            heuristic_evaluator: YinshHeuristics instance for heuristic evaluation (required for pure_heuristic/hybrid modes).
            heuristic_weight: Weight for heuristic in hybrid mode (0.0-1.0, where 0.0=pure neural, 1.0=pure heuristic).
            num_simulations: Base simulation budget (used before switch_ply).
            c_puct: Exploration constant for UCB.
            dirichlet_alpha: Alpha parameter for Dirichlet noise at the root.
            value_weight: Factor to scale the node's value in the UCB calculation.
            max_depth: Maximum depth for the search path during simulation.
            late_simulations: Simulation budget *after* simulation_switch_ply. If None, defaults to num_simulations.
            simulation_switch_ply: Ply number to switch simulation budget.
            initial_temp: Starting temperature for action selection sampling.
            final_temp: Final temperature value.
            annealing_steps: Number of moves over which temperature anneals.
            temp_clamp_fraction: Fraction of annealing_steps before clamping temperature.
            mcts_metrics: Optional MCTSMetrics collector instance.
            game_state_pool: Optional GameStatePool for memory management.
            enable_subtree_reuse: If True, the search tree built at move N is
                carried across to move N+1 via `advance_root(played_move)`, so
                visits accumulated under the played subtree are kept instead of
                discarded. Doubles effective sims-per-move with no extra NN
                compute. Disable to restore the old "fresh root every move"
                behavior for A/B testing.
            fpu_reduction: First-Play Urgency reduction coefficient. Unvisited
                children score their Q as ``q_parent - fpu_reduction *
                sqrt(Σ π(c) for visited c)`` instead of 0, so an unexplored
                low-prior move doesn't get to coast past a visited sibling with
                modestly-worse actual Q just because the prior is nonzero.
                KataGo's default is 0.25. Set to 0 to restore the old
                prior-only scoring for unvisited children.
            epsilon_mix_start: Dirichlet-noise mixing fraction at move 0.
                Classic AlphaZero value is 0.25 (new_prior = 0.75·prior +
                0.25·noise). Higher values inject more randomness at the root,
                encouraging self-play diversity.
            epsilon_mix_end: Dirichlet-noise mixing fraction after the taper
                completes. 0 turns root noise off entirely in late-game /
                tactical positions, where randomness most often hurts. Set
                equal to `epsilon_mix_start` to disable the taper.
            epsilon_mix_taper_moves: Number of moves over which
                `epsilon_mix` linearly interpolates from start → end.
                At move_number ≥ this value, `epsilon_mix_end` is used.
                0 disables the taper (start value used at every move).
        """
        self.network = network
        # Use network's encoder to ensure consistent encoding (basic or enhanced)
        self.state_encoder = network.state_encoder
        self.logger = logging.getLogger("MCTS") # Use module-level logger

        # Evaluation Mode Configuration
        self.evaluation_mode = evaluation_mode.lower()
        self.heuristic_evaluator = heuristic_evaluator
        self.heuristic_weight = np.clip(heuristic_weight, 0.0, 1.0)  # Ensure weight is in [0, 1]

        # Validate evaluation mode configuration
        if self.evaluation_mode not in ["pure_neural", "pure_heuristic", "hybrid"]:
            raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}. Must be 'pure_neural', 'pure_heuristic', or 'hybrid'.")
        if self.evaluation_mode in ["pure_heuristic", "hybrid"] and heuristic_evaluator is None:
            raise ValueError(f"heuristic_evaluator required for evaluation_mode='{evaluation_mode}'")

        # Memory Pool Management
        self.game_state_pool = game_state_pool
        self._pool_enabled = game_state_pool is not None

        # MCTS Core Parameters
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        # *** Store using the new name 'value_weight' ***
        self.value_weight = value_weight
        self.max_depth = max_depth # Store max_depth

        # Rollout Budget Scheduling
        self.early_simulations = num_simulations
        # Default late_simulations to early_simulations if None is passed
        self.late_simulations = late_simulations if late_simulations is not None else num_simulations
        self.switch_ply = simulation_switch_ply

        # Probabilistic fast-sim split — independent of early/late switch.
        # On each move, with `fast_sim_prob` probability the budget is replaced
        # with `fast_simulations` (0 ⇒ disabled). Mirrors alphazero-general's
        # `numFastSims=20, probFastSim=0.75` default.
        self.fast_simulations = int(fast_simulations)
        self.fast_sim_prob = float(fast_sim_prob)

        # Temperature Schedule Parameters (stored for get_temperature)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.annealing_steps = annealing_steps
        self.temp_clamp_frac = temp_clamp_fraction

        # Metrics Tracking
        self.metrics = mcts_metrics if mcts_metrics is not None else MCTSMetrics()
        self.current_iteration = 0 # Can be updated externally if needed

        # Subtree reuse state — carries the search tree across moves within a game.
        # Ownership contract: the caller drives the tree lifecycle by calling
        # `advance_root(played_move)` after every outer-game move and (optionally)
        # `reset_tree()` at game start / end. `search()` only reads this state.
        self.enable_subtree_reuse = enable_subtree_reuse
        self._cached_root: Optional[Node] = None

        # First-Play Urgency coefficient. 0 disables the mechanism (unvisited
        # children fall back to the old q=0 default).
        self.fpu_reduction = float(fpu_reduction)

        # Root Dirichlet noise mixing schedule. Linear taper from start → end
        # over taper_moves; constant `start` if taper_moves == 0.
        self.epsilon_mix_start = float(epsilon_mix_start)
        self.epsilon_mix_end = float(epsilon_mix_end)
        self.epsilon_mix_taper_moves = int(epsilon_mix_taper_moves)

        # Root policy temperature: re-shape priors at root before noise mixing.
        # T > 1 flattens (more exploration); T < 1 sharpens; 1.0 is no-op.
        self.root_policy_temp = float(root_policy_temp)

        self.logger.info(f"MCTS Initialized:")
        self.logger.info(f"  Evaluation Mode: {self.evaluation_mode} (heuristic_weight={self.heuristic_weight:.3f})")
        self.logger.info(f"  Memory: Pool enabled={self._pool_enabled}")
        self.logger.info(f"  Subtree Reuse: {'ENABLED' if self.enable_subtree_reuse else 'DISABLED'}")
        self.logger.info(f"  FPU Reduction: {self.fpu_reduction:.3f}")
        self.logger.info(
            f"  Epsilon Mix: {self.epsilon_mix_start:.3f} → {self.epsilon_mix_end:.3f} "
            f"over {self.epsilon_mix_taper_moves} moves"
        )
        self.logger.info(f"  Sims: Early={self.early_simulations}, Late={self.late_simulations} (Switch Ply {self.switch_ply})")
        if self.fast_sim_prob > 0.0 and self.fast_simulations > 0:
            self.logger.info(
                f"  Fast-sim split: fast={self.fast_simulations} sims with p={self.fast_sim_prob:.2f}"
            )
        self.logger.info(f"  Search: cPUCT={self.c_puct:.2f}, Alpha={self.dirichlet_alpha:.2f}, Value Weight={self.value_weight:.2f}, Max Depth={self.max_depth}")
        self.logger.info(f"  Temperature: Initial={self.initial_temp:.2f}, Final={self.final_temp:.2f}, Steps={self.annealing_steps}, Clamp Frac={self.temp_clamp_frac:.2f}")

        self.debug_config()

    def debug_config(self):
        """Log all configuration parameters to verify they're being set correctly."""
        self.logger.info("====== MCTS Configuration Debug ======")
        self.logger.info(f"Early Simulations: {self.early_simulations}")
        self.logger.info(f"Late Simulations: {self.late_simulations}")
        self.logger.info(f"Switch Ply: {self.switch_ply}")
        self.logger.info(f"c_puct: {self.c_puct}")
        self.logger.info(f"Dirichlet Alpha: {self.dirichlet_alpha}")
        self.logger.info(f"Value Weight: {self.value_weight}")
        self.logger.info(f"Max Depth: {self.max_depth}")
        self.logger.info("======================================")

    def _acquire_state_copy(self, original_state: GameState) -> GameState:
        """Get a GameState copy from pool or create new one."""
        if self._pool_enabled:
            try:
                pooled_state = self.game_state_pool.get()
                # Copy state data from original to pooled state
                # This is more efficient than creating a brand new state
                pooled_state.copy_from(original_state)
                return pooled_state
            except Exception as e:
                self.logger.warning(f"Failed to get state from pool: {e}, falling back to copy()")
                return copy.deepcopy(original_state)
        else:
            return copy.deepcopy(original_state)

    def _release_state(self, state: GameState) -> None:
        """Release a GameState back to the pool."""
        if self._pool_enabled and state is not None:
            try:
                self.game_state_pool.return_game_state(state)
            except Exception as e:
                self.logger.warning(f"Failed to return state to pool: {e}")

    def _create_child_node(self, state: GameState, parent, prior_prob: float) -> 'Node':
        """Create a child node WITHOUT copying the game state.

        Child node states were never read during search (the traversal uses a
        running current_state copy advanced by make_move), so the ~85 deep-copies
        per expansion were pure waste — the single biggest contributor to
        per-game wall-clock.
        """
        return Node(None, parent=parent, prior_prob=prior_prob, c_puct=self.c_puct)

    def reset_tree(self) -> None:
        """Drop any cached search tree and free its pooled game states."""
        if self._cached_root is not None:
            self._cached_root.clear_tree()
            self._cached_root = None

    def advance_root(self, played_move) -> None:
        """Reseat the cached root on the subtree for `played_move`. Siblings are
        freed; the chosen subtree's visits/priors/child structure are kept so the
        next `search()` can build on top of them.

        Contract: call this once per outer-game move, with the move just played.
        If subtree reuse is disabled or the cached root has no child for
        `played_move` (unknown-move fallback), the cache is cleared and the next
        `search()` rebuilds from scratch.
        """
        if not self.enable_subtree_reuse:
            # Even with reuse disabled, callers may invoke this; keep it a no-op
            # rather than paying for a tree-clear (search() already clears per call).
            return
        if self._cached_root is None:
            return
        kept = self._cached_root.children.pop(played_move, None)
        if kept is None:
            # Unknown move — can happen if the caller drove search with a
            # different state than the cache represents, or if an untracked move
            # (not enumerated as a valid child at search time) was played. Drop
            # the tree entirely so the next search builds from a fresh root.
            self._cached_root.clear_tree()
            self._cached_root = None
            return
        # Free the old root + siblings (kept was popped out already, so
        # recursive clear_tree on the old root won't touch kept).
        self._cached_root.clear_tree()
        kept.parent = None
        # `dirichlet_applied` tracks "was root-noise applied during expansion of
        # this node as root." The kept subtree was never a root before, so we
        # clear the flag just in case and let search() apply fresh noise via
        # `_apply_root_dirichlet_noise`.
        if hasattr(kept, "dirichlet_applied"):
            delattr(kept, "dirichlet_applied")
        # Same idea for `policy_temp_applied`: this node's children had their
        # priors set as a non-root expansion, so root-policy temperature has
        # not yet been applied. Clear the flag so the next search() call
        # reshapes the (now-root) child priors before the next noise mix.
        if hasattr(kept, "policy_temp_applied"):
            delattr(kept, "policy_temp_applied")
        self._cached_root = kept

    def _compute_epsilon_mix(self, move_number: int) -> float:
        """Linearly interpolate `epsilon_mix` from start → end over
        `epsilon_mix_taper_moves`. Returns `epsilon_mix_start` if the taper
        is disabled (taper_moves ≤ 0) so callers never have to special-case
        the no-taper path."""
        if self.epsilon_mix_taper_moves <= 0:
            return self.epsilon_mix_start
        alpha = min(max(move_number, 0) / self.epsilon_mix_taper_moves, 1.0)
        return self.epsilon_mix_start + alpha * (self.epsilon_mix_end - self.epsilon_mix_start)

    def _apply_root_policy_temperature(self, root: 'Node') -> None:
        """Re-shape root child priors with temperature T = ``root_policy_temp``.

        new_p_i ∝ p_i^(1/T). T > 1 flattens, T < 1 sharpens, 1.0 is a no-op.
        Applied at the root only — interior nodes keep their NN priors verbatim.
        Run BEFORE Dirichlet noise mixing so noise sees the reshaped distribution.

        Idempotent: gates on `policy_temp_applied` so reuse of a cached root
        (which already had temperature applied during its first expansion as
        root) doesn't compound. Cleared by `advance_root` on subtree-reuse.
        """
        if self.root_policy_temp == 1.0 or not root.children:
            return
        if hasattr(root, "policy_temp_applied"):
            return
        inv_t = 1.0 / self.root_policy_temp
        priors = np.array(
            [child.prior_prob for child in root.children.values()],
            dtype=np.float64,
        )
        # Guard against negative or NaN priors before exponentiation.
        priors = np.clip(priors, 0.0, None)
        reshaped = np.power(priors, inv_t)
        total = reshaped.sum()
        if total <= 1e-12:
            return  # Degenerate distribution — leave priors untouched.
        reshaped /= total
        for i, child in enumerate(root.children.values()):
            child.prior_prob = float(reshaped[i])
        root.policy_temp_applied = True

    def _apply_root_dirichlet_noise(self, root: 'Node', move_number: int) -> None:
        """Mix fresh Dirichlet noise into an already-expanded root's child priors.

        Fresh roots get noise via the normal expansion path (inside `search`),
        but a reused root is already expanded — selection skips the expansion
        branch for the root entirely, so we apply noise here to preserve
        AlphaZero's "fresh exploration noise at every move" guarantee.

        The mixing fraction tapers with `move_number` via
        `_compute_epsilon_mix`, so late-game tactical positions don't keep
        getting flooded with root randomness.
        """
        if not root.is_expanded or not root.children:
            return
        # Apply root policy temperature once per root identity (gated inside
        # the helper). Must run BEFORE noise mixing so noise sees reshaped prior.
        self._apply_root_policy_temperature(root)
        if self.dirichlet_alpha <= 0:
            return
        epsilon_mix = self._compute_epsilon_mix(move_number)
        if epsilon_mix <= 0:
            return  # Taper has fully shut noise off.
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        for i, move in enumerate(moves):
            child = root.children[move]
            child.prior_prob = (
                (1 - epsilon_mix) * child.prior_prob + epsilon_mix * noise[i]
            )

    def get_temperature(self, move_number: int) -> float:
        """
        Calculate temperature based on linear annealing schedule with clamping.
        """
        # Ensure annealing_steps is positive to avoid division by zero
        if self.annealing_steps <= 0:
            return self.final_temp

        clamp_moves = int(self.annealing_steps * self.temp_clamp_frac)

        # Ensure clamp_moves is at least 1 if annealing_steps > 0
        clamp_moves = max(1, clamp_moves)

        if move_number >= clamp_moves:
            temperature = self.final_temp
        else:
            progress = move_number / clamp_moves # Progress from 0 to 1
            temperature = self.initial_temp - (self.initial_temp - self.final_temp) * progress

        # Add debug logging for temperature calculation if needed
        # self.logger.debug(f"Move: {move_number}, Anneal Steps: {self.annealing_steps}, Clamp Frac: {self.temp_clamp_frac}, Clamp Moves: {clamp_moves}, Temp: {temperature:.3f}")
        return temperature

    def _get_budget(self, move_number: int) -> int:
        """Per-move simulation budget.

        Two-stage:
        1. Pick base budget by move_number (early < switch_ply, else late).
        2. With probability `fast_sim_prob`, replace base with `fast_simulations`.

        The fast/slow split is independent of the early/late switch and exists
        to amortize sim cost across moves (alphazero-general's `probFastSim`
        idea — 75% of moves at 20 sims, 25% at 100).
        """
        base = self.early_simulations if move_number < self.switch_ply else self.late_simulations
        if self.fast_sim_prob > 0.0 and self.fast_simulations > 0:
            if random.random() < self.fast_sim_prob:
                return self.fast_simulations
        return base

    def search(self, state: GameState, move_number: int) -> np.ndarray:
        """
        Run MCTS simulations for the given state and move number.
        """
        # Choose rollout budget based on the current move number (+ optional fast/slow split)
        budget = self._get_budget(move_number)
        # self.logger.debug(f"Move {move_number}, Using budget: {budget} (Early: {self.early_simulations}, Late: {self.late_simulations}, Switch: {self.switch_ply})")

        if self.enable_subtree_reuse and self._cached_root is not None:
            root = self._cached_root
            # Reused root is already expanded — the normal expansion path won't
            # apply root-level Dirichlet noise, so do it here instead.
            self._apply_root_dirichlet_noise(root, move_number=move_number)
        else:
            root = Node(state, c_puct=self.c_puct) # Pass c_puct to root node
            if self.enable_subtree_reuse:
                self._cached_root = root
        # Initialize policy vector
        move_probs = np.zeros(self.state_encoder.total_moves, dtype=np.float32)

        # --- Run Simulations ---
        simulation_states = []  # Track states acquired from pool for cleanup
        
        for sim in range(budget):
            node = root
            search_path = [node]
            current_state = self._acquire_state_copy(state)
            simulation_states.append(current_state)  # Track for cleanup
            # Track player-to-move at each path node so backprop can flip the
            # running value only across real player transitions (YINSH capture
            # sequences keep the same player for 3+ plies).
            players_path = [current_state.current_player]
            depth = 0
            action = None

            # 1. Selection ...
            while node.is_expanded and node.children:
                action = self._select_action(node)
                if action not in node.children:
                     self.logger.error(f"MCTS Error: Selected action {action} not in node children {list(node.children.keys())} during Selection phase. State: {current_state}")
                     break
                current_state.make_move(action)
                node = node.children[action]
                search_path.append(node)
                players_path.append(current_state.current_player)
                depth += 1
                if depth >= self.max_depth:
                     break

            # Skip only when selection ran and bailed out mid-loop because the
            # chosen action wasn't in the parent's children (error-logged above).
            # The extra `node.is_expanded and node.children` guards against the
            # normal "descended to an unexpanded leaf" case (where `action` is
            # set at the parent level but the leaf's own children dict is empty).
            # The old check also fired on `action is None`, which meant "while
            # loop didn't run because root was unexpanded" — that's the normal
            # first-sim path and should fall through to expansion, not skip.
            # Silently skipping made MCTS uniform-fallback on every first sim.
            if action is not None and action not in node.children and node.is_expanded and node.children:
                 self.logger.debug(f"Skipping simulation {sim+1} due to selection error.")
                 continue # Skip to next simulation

            # 2. Expansion & Evaluation: If a leaf node is reached
            value = self._get_value(current_state) # Check if terminal first

            if value is None and depth < self.max_depth:
                policy, net_value = self._evaluate_state(current_state) # Get NN prediction
                value = net_value # Use network's value if not terminal

                # *** Add check: Ensure state didn't become terminal between _get_value and _evaluate_state ***
                if current_state.is_terminal():
                     self.logger.debug(f"State became terminal unexpectedly before expansion at depth {depth}. Using terminal value.")
                     value = self._get_value(current_state) # Re-get terminal value
                else:
                    # Proceed with expansion only if truly non-terminal
                    valid_moves = current_state.get_valid_moves()
                    if valid_moves:
                        node.is_expanded = True
                        policy = self._mask_invalid_moves(policy, valid_moves)
                        for move in valid_moves:
                            node.children[move] = self._create_child_node(
                                current_state,
                                parent=node,
                                prior_prob=self._get_move_prob(policy, move)
                            )

                    # Apply root_policy_temperature once per root, then mix
                    # Dirichlet noise. Both gated independently — temperature
                    # via `policy_temp_applied`, noise via `dirichlet_applied`.
                    if node is root:
                        self._apply_root_policy_temperature(root)
                    if node is root and not hasattr(root, "dirichlet_applied"):
                        epsilon_mix = self._compute_epsilon_mix(move_number)
                        if self.dirichlet_alpha > 0 and len(valid_moves) > 0 and epsilon_mix > 0:
                            noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))
                            for i, move in enumerate(valid_moves):
                                child_node = root.children[move]
                                child_node.prior_prob = (1 - epsilon_mix) * child_node.prior_prob + epsilon_mix * noise[i]
                            root.dirichlet_applied = True

                # else: # No valid moves means state is terminal (should have been caught by _get_value)
                #     self.logger.warning(f"MCTS Warning: Reached non-terminal state with no valid moves during expansion? State:\n{current_state}")
                #     value = self._get_value(current_state) or 0.0 # Assign terminal value if possible

            # elif value is None and depth >= self.max_depth:
                 # Reached max depth, use network evaluation for value, but don't expand
                 # _, value = self._evaluate_state(current_state)
                 # self.logger.debug(f"MCTS Search hit max depth ({self.max_depth}) during expansion. Using value {value:.3f}.")


            # 3. Backpropagation: Update values and visit counts along the path
            if value is None: # Should not happen if logic above is correct, but safeguard
                 self.logger.error(f"MCTS Error: Value is None before backpropagation. State terminal: {current_state.is_terminal()}")
                 value = 0.0 # Default value if something went wrong
            self._backpropagate(search_path, players_path, value)

        # Clean up simulation states
        for state in simulation_states:
            self._release_state(state)

        # --- After all simulations ---
        # Calculate final move probabilities based on visit counts
        temp = self.get_temperature(move_number)
        # self.logger.debug(f"Move {move_number}: Calculating policy with Temp={temp:.3f}")

        # Ensure root has children before proceeding
        if not root.children:
            self.logger.debug(f"MCTS Warning: Root node has no children after {budget} simulations. Returning uniform policy over valid moves (if any). State:\n{state}")
            valid_moves = state.get_valid_moves()
            if valid_moves:
                prob = 1.0 / len(valid_moves)
                for move in valid_moves:
                    move_idx = self.state_encoder.move_to_index(move)
                    if 0 <= move_idx < len(move_probs):
                         move_probs[move_idx] = prob
                    else:
                         self.logger.error(f"Invalid move index {move_idx} for move {move} generated by state.get_valid_moves(). Max index: {len(move_probs)-1}")
            return move_probs # Return uniform or zero vector


        valid_moves_at_root = list(root.children.keys())
        visit_counts = np.array([
            root.children[move].visit_count for move in valid_moves_at_root
        ], dtype=np.float32)

        if visit_counts.sum() == 0:
             self.logger.warning(f"MCTS Warning: Root node children have zero total visits after {budget} simulations. Returning uniform policy.")
             prob = 1.0 / len(valid_moves_at_root) if valid_moves_at_root else 0.0
             visit_probs = np.full(len(valid_moves_at_root), prob, dtype=np.float32)
        else:
            # Apply temperature
            if temp == 0:
                # Greedy selection: highest visit count gets probability 1
                visit_probs = np.zeros_like(visit_counts)
                visit_probs[np.argmax(visit_counts)] = 1.0
            else:
                # Apply temperature scaling
                visit_counts_temp = np.power(visit_counts, 1.0 / temp)
                # Normalize to get probabilities
                total_visits_temp = visit_counts_temp.sum()
                if total_visits_temp > 1e-6 : # Avoid division by zero
                     visit_probs = visit_counts_temp / total_visits_temp
                else: # Handle case where all temp-scaled visits are near zero
                     self.logger.warning("MCTS Warning: Sum of temperature-scaled visit counts is near zero. Falling back to uniform.")
                     prob = 1.0 / len(valid_moves_at_root) if valid_moves_at_root else 0.0
                     visit_probs = np.full(len(valid_moves_at_root), prob, dtype=np.float32)


        # Store probabilities in the full policy vector
        for move, prob in zip(valid_moves_at_root, visit_probs):
            move_idx = self.state_encoder.move_to_index(move)
            if 0 <= move_idx < len(move_probs):
                 move_probs[move_idx] = prob
            else:
                 self.logger.error(f"Invalid move index {move_idx} for move {move} from root children. Max index: {len(move_probs)-1}")

        # Store root value for use as training target (mirrors search_batch).
        # Search-consistency probe (and any future caller) needs both `search()`
        # and `search_batch()` to expose the root value the same way.
        self.last_root_value = root.value() if root.visit_count > 0 else 0.0

        # With subtree reuse enabled, the tree is preserved for the next search
        # call — cleanup happens in `advance_root`/`reset_tree`. Without reuse,
        # break circular references immediately so Python GC can free nodes.
        if not self.enable_subtree_reuse:
            root.clear_tree()

        return move_probs

    def search_batch(self, state: GameState, move_number: int, batch_size: int = 32) -> np.ndarray:
        """
        Run MCTS simulations with batched leaf node evaluation for improved throughput.

        Instead of evaluating each leaf node individually, this method collects multiple
        leaf nodes and evaluates them all at once using the network's predict_batch method.
        This provides 10-20x speedup on M2 by utilizing CPU/Neural Engine more efficiently.

        Args:
            state: The current game state
            move_number: Current move number (for temperature scheduling)
            batch_size: Number of leaf nodes to collect before batching evaluation (default: 32)

        Returns:
            Policy vector of move probabilities
        """
        # Choose rollout budget based on the current move number (+ optional fast/slow split)
        budget = self._get_budget(move_number)

        if self.enable_subtree_reuse and self._cached_root is not None:
            root = self._cached_root
            # Reused root is already expanded — apply Dirichlet noise here since
            # the expansion path (which normally handles root noise) won't run
            # for an already-expanded root.
            self._apply_root_dirichlet_noise(root, move_number=move_number)
        else:
            root = Node(state, c_puct=self.c_puct)
            if self.enable_subtree_reuse:
                self._cached_root = root
        move_probs = np.zeros(self.state_encoder.total_moves, dtype=np.float32)

        # Track states acquired from pool for cleanup
        simulation_states = []

        # Collect leaves for batched evaluation
        batch_leaves = []  # List of (node, search_path, current_state, depth)

        for sim in range(budget):
            node = root
            search_path = [node]
            current_state = self._acquire_state_copy(state)
            simulation_states.append(current_state)
            # Parallel to search_path; passed through to _backpropagate so it
            # can do conditional sign-flip across real player transitions.
            players_path = [current_state.current_player]
            depth = 0
            action = None

            # 1. Selection phase - traverse tree until we reach a leaf
            while node.is_expanded and node.children:
                action = self._select_action(node)
                if action not in node.children:
                    self.logger.error(f"MCTS Error: Selected action {action} not in node children during batched selection.")
                    break
                current_state.make_move(action)
                node = node.children[action]
                search_path.append(node)
                players_path.append(current_state.current_player)
                depth += 1
                if depth >= self.max_depth:
                    break

            # Skip only when selection ran and bailed out mid-loop because the
            # chosen action wasn't in the parent's children. The extra
            # `node.is_expanded and node.children` guards against the normal
            # "descended to an unexpanded leaf" case (action is set at parent
            # level but the leaf's own children dict is empty). See matching
            # comment in `search()` — the old `action is None` guard was a bug
            # that silently skipped first-sim expansion.
            if action is not None and action not in node.children and node.is_expanded and node.children:
                continue

            # 2. Check if this is a terminal state
            terminal_value = self._get_value(current_state)
            if terminal_value is not None:
                # Terminal state - backpropagate immediately
                self._backpropagate(search_path, players_path, terminal_value)
                continue

            # 3. Check if we hit max depth
            if depth >= self.max_depth:
                # At max depth - need to evaluate but don't expand
                # Add to batch for evaluation
                node.add_virtual_loss()  # Mark as in-flight
                batch_leaves.append((node, search_path, players_path, current_state, depth, False))
            else:
                # Normal leaf node - add to batch for expansion and evaluation
                node.add_virtual_loss()  # Mark as in-flight
                batch_leaves.append((node, search_path, players_path, current_state, depth, True))

            # 4. Process batch when it's full or we're at the end
            if len(batch_leaves) >= batch_size or sim == budget - 1:
                if batch_leaves:
                    self._evaluate_and_backup_batch(batch_leaves, root, move_number=move_number)
                    batch_leaves = []

        # Process any remaining leaves
        if batch_leaves:
            self._evaluate_and_backup_batch(batch_leaves, root, move_number=move_number)

        # Clean up simulation states
        for s in simulation_states:
            self._release_state(s)

        # Calculate final move probabilities (same as serial version)
        temp = self.get_temperature(move_number)

        if not root.children:
            self.logger.debug(f"MCTS Warning: Root node has no children after {budget} simulations.")
            valid_moves = state.get_valid_moves()
            if valid_moves:
                prob = 1.0 / len(valid_moves)
                for move in valid_moves:
                    move_idx = self.state_encoder.move_to_index(move)
                    if 0 <= move_idx < len(move_probs):
                        move_probs[move_idx] = prob
            return move_probs

        valid_moves_at_root = list(root.children.keys())
        visit_counts = np.array([
            root.children[move].visit_count for move in valid_moves_at_root
        ], dtype=np.float32)

        if visit_counts.sum() == 0:
            prob = 1.0 / len(valid_moves_at_root) if valid_moves_at_root else 0.0
            visit_probs = np.full(len(valid_moves_at_root), prob, dtype=np.float32)
        else:
            if temp == 0:
                visit_probs = np.zeros_like(visit_counts)
                visit_probs[np.argmax(visit_counts)] = 1.0
            else:
                visit_counts_temp = np.power(visit_counts, 1.0 / temp)
                total_visits_temp = visit_counts_temp.sum()
                if total_visits_temp > 1e-6:
                    visit_probs = visit_counts_temp / total_visits_temp
                else:
                    prob = 1.0 / len(valid_moves_at_root) if valid_moves_at_root else 0.0
                    visit_probs = np.full(len(valid_moves_at_root), prob, dtype=np.float32)

        for move, prob in zip(valid_moves_at_root, visit_probs):
            move_idx = self.state_encoder.move_to_index(move)
            if 0 <= move_idx < len(move_probs):
                move_probs[move_idx] = prob

        # Store root value for use as training target (Fix #1)
        self.last_root_value = root.value() if root.visit_count > 0 else 0.0

        # With subtree reuse enabled, the tree is preserved for the next search
        # call — cleanup happens in `advance_root`/`reset_tree`.
        if not self.enable_subtree_reuse:
            root.clear_tree()

        return move_probs

    def _evaluate_and_backup_batch(self, batch_leaves: List[Tuple], root: Node, move_number: int = 0):
        """
        Evaluate a batch of leaf nodes and back up the results.

        Args:
            batch_leaves: List of tuples
                (node, search_path, players_path, current_state, depth, can_expand).
                ``players_path`` is parallel to ``search_path`` and lets
                ``_backpropagate`` flip the running value only across real
                player transitions (YINSH same-player capture sequences).
            root: Root node (for Dirichlet noise application)
            move_number: Outer-game move number, used to taper root Dirichlet
                noise mixing via `_compute_epsilon_mix`. 0 defaults keep early-
                game mixing strong; callers in `search_batch` pass it through.
        """
        if not batch_leaves:
            return

        # Separate states for batch evaluation
        states_to_evaluate = [item[3] for item in batch_leaves]

        # Batch evaluate all states
        if self.evaluation_mode in ["pure_neural", "hybrid"]:
            # Use neural network for batch evaluation
            policy_logits_batch, values_batch = self.network.predict_batch(states_to_evaluate)

            # Convert to numpy
            policy_logits_batch = policy_logits_batch.cpu().numpy()
            values_batch = values_batch.cpu().numpy().flatten()
        else:
            # Pure heuristic mode - evaluate individually (heuristics don't support batching)
            policy_logits_batch = None
            values_batch = np.array([
                self.heuristic_evaluator.evaluate_position(s, s.current_player)
                for s in states_to_evaluate
            ])

        # Process each leaf in the batch
        for i, (node, search_path, players_path, current_state, depth, can_expand) in enumerate(batch_leaves):
            # Get policy and value for this state
            if self.evaluation_mode in ["pure_neural", "hybrid"]:
                policy_logits = policy_logits_batch[i]
                policy_nn = torch.softmax(torch.from_numpy(policy_logits), dim=0).numpy()
                value_nn = values_batch[i]
            else:
                policy_nn = None
                value_nn = 0.0

            # Combine with heuristics if in hybrid mode
            if self.evaluation_mode == "hybrid":
                value_heuristic = self.heuristic_evaluator.evaluate_position(
                    current_state, current_state.current_player
                )
                valid_moves = current_state.get_valid_moves()
                policy_heuristic = np.zeros_like(policy_nn)
                if len(valid_moves) > 0:
                    for move in valid_moves:
                        idx = self.state_encoder.move_to_index(move)
                        if 0 <= idx < len(policy_heuristic):
                            policy_heuristic[idx] = 1.0 / len(valid_moves)

                w_h = self.heuristic_weight
                w_n = 1.0 - w_h
                policy = w_n * policy_nn + w_h * policy_heuristic
                value = w_n * value_nn + w_h * value_heuristic
            elif self.evaluation_mode == "pure_heuristic":
                valid_moves = current_state.get_valid_moves()
                policy = np.zeros(self.state_encoder.total_moves)
                if len(valid_moves) > 0:
                    for move in valid_moves:
                        idx = self.state_encoder.move_to_index(move)
                        if 0 <= idx < len(policy):
                            policy[idx] = 1.0 / len(valid_moves)
                value = values_batch[i]
            else:  # pure_neural
                policy = policy_nn
                value = value_nn

            # Expand node if allowed
            if can_expand and not current_state.is_terminal():
                valid_moves = current_state.get_valid_moves()
                if valid_moves:
                    node.is_expanded = True
                    policy = self._mask_invalid_moves(policy, valid_moves)
                    for move in valid_moves:
                        node.children[move] = self._create_child_node(
                            current_state,
                            parent=node,
                            prior_prob=self._get_move_prob(policy, move)
                        )

                    # Apply root_policy_temperature once per root, then mix
                    # Dirichlet noise. Both gated independently — temperature
                    # via `policy_temp_applied`, noise via `dirichlet_applied`.
                    if node is root:
                        self._apply_root_policy_temperature(root)
                    if node is root and not hasattr(root, "dirichlet_applied"):
                        epsilon_mix = self._compute_epsilon_mix(move_number)
                        if self.dirichlet_alpha > 0 and len(valid_moves) > 0 and epsilon_mix > 0:
                            noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))
                            for j, move in enumerate(valid_moves):
                                child_node = root.children[move]
                                child_node.prior_prob = (
                                    (1 - epsilon_mix) * child_node.prior_prob + epsilon_mix * noise[j]
                                )
                            root.dirichlet_applied = True

            # Remove virtual loss and backpropagate
            node.remove_virtual_loss()
            self._backpropagate(search_path, players_path, value)

    def _select_action(self, node: Node) -> Move:
        """Select action using UCB formula with configured value_weight.

        Vectorized over child stats. The previous scalar version was 20%
        of self-play wall-clock at sim=400 (BITBOARD_FOLLOWUP_PLAN.md
        Candidate B'); per-iter ``np.sqrt`` / ``np.random.uniform`` C-call
        overhead dominated. Materializing children's stats into numpy
        arrays once and computing UCB / argmax with vectorized ops drops
        per-call cost from ~130µs to ~10µs.

        Preserves the original semantics:
          - Visited children: q = child.value(), u = c_puct·π·√parent /
            (1 + visits).
          - Unvisited children: q = fpu_q, u = c_puct·π·√(parent+ε)
            (NO ``/ (1 + visits)`` division — original formula
            difference, intentional).
          - ε-noise added to every score for tie-breaking.

        ``c_puct`` is identical for every child of a given MCTS instance
        (set from ``self.c_puct`` in ``_create_child_node``), so we read
        it from ``self`` rather than per-child.
        """
        valid_moves = list(node.children.keys())
        if not valid_moves:
            return None
        parent_visit_count = node.visit_count

        # Handle case where node might not have been visited yet (shouldn't happen if called after expansion)
        if parent_visit_count == 0:
            self.logger.warning("MCTS _select_action called on node with zero visits. Selecting randomly.")
            return random.choice(valid_moves)

        epsilon = 1e-8
        n = len(valid_moves)

        # Materialize child stats. One Python pass over the dict; everything
        # else is vectorized. Note: child.value() = value_sum / (visit_count
        # + virtual_losses), so we pull both components and divide in numpy.
        visits = np.empty(n, dtype=np.float64)
        value_sums = np.empty(n, dtype=np.float64)
        virt_losses = np.empty(n, dtype=np.float64)
        priors = np.empty(n, dtype=np.float64)
        for i, m in enumerate(valid_moves):
            ch = node.children[m]
            visits[i] = ch.visit_count
            value_sums[i] = ch.value_sum
            virt_losses[i] = ch.virtual_losses
            priors[i] = ch.prior_prob

        visited_mask = visits > 0

        # First-Play Urgency baseline for unvisited children. KataGo-style:
        # q_fpu = q_parent − fpu_reduction · sqrt(Σ π(c) for visited c).
        # ``-node.value()`` flips grandparent-POV (how this codebase
        # stores it) into node's-own-POV. fpu_reduction=0 reduces this to
        # the old prior-only scoring for unvisited children.
        if self.fpu_reduction > 0:
            visited_policy_sum = float(priors[visited_mask].sum())
            q_parent_pov = -node.value()
            fpu_q = q_parent_pov - self.fpu_reduction * np.sqrt(visited_policy_sum)
        else:
            fpu_q = 0.0

        # Q-vector. value() denominator (visits + virtual_losses) is safe
        # here because we only consult q_visited where visited_mask=True
        # (visit_count > 0 ⇒ adjusted_visits > 0).
        adjusted_visits = visits + virt_losses
        # Avoid /0 in the unvisited slots; np.where below picks fpu_q anyway.
        safe_denom = np.where(adjusted_visits > 0, adjusted_visits, 1.0)
        q_visited = value_sums / safe_denom
        q_values = np.where(visited_mask, q_visited, fpu_q)
        scaled_q = self.value_weight * q_values

        # U-vector. Original code uses two different formulas:
        #   visited:   c_puct · π · √parent / (1 + visits)
        #   unvisited: c_puct · π · √(parent + ε)    [no division]
        # parent_visit_count >= 1 here (we short-circuited 0 above), so
        # √(parent + ε) ≈ √parent — but we keep both terms separately to
        # avoid drift if anyone later changes the early-return condition.
        c_puct = self.c_puct  # constant across children of this MCTS
        sqrt_parent = np.sqrt(parent_visit_count)
        sqrt_parent_eps = np.sqrt(parent_visit_count + epsilon)
        u_visited = c_puct * priors * sqrt_parent / (1.0 + visits)
        u_unvisited = c_puct * priors * sqrt_parent_eps
        u_values = np.where(visited_mask, u_visited, u_unvisited)

        # ε-noise for tiebreaking. One vector draw replaces N scalar draws.
        noise = np.random.uniform(0.0, epsilon, size=n)
        scores = scaled_q + u_values + noise

        return valid_moves[int(np.argmax(scores))]

    def _evaluate_state(self, state: GameState) -> Tuple[np.ndarray, float]:
        """Get policy and value from neural network and/or heuristic evaluator.

        Supports three evaluation modes:
        - pure_neural: Only use neural network
        - pure_heuristic: Only use heuristic evaluator
        - hybrid: Weighted combination of neural network and heuristic
        """
        # Get neural network evaluation
        if self.evaluation_mode in ["pure_neural", "hybrid"]:
            with torch.no_grad():
                policy_tensor, value_tensor = self.network.predict_from_state(state)
                # Convert to numpy and extract values
                policy_nn = torch.softmax(policy_tensor, dim=1).squeeze(0).cpu().numpy()
                value_nn = value_tensor.item()  # Get scalar value
        else:
            policy_nn = None
            value_nn = 0.0

        # Get heuristic evaluation
        if self.evaluation_mode in ["pure_heuristic", "hybrid"]:
            # Heuristic evaluator returns differential score for current player
            current_player = state.current_player
            value_heuristic = self.heuristic_evaluator.evaluate_position(state, current_player)

            # For policy, use uniform distribution over legal moves (heuristics don't provide policy guidance)
            valid_moves = state.get_valid_moves()
            policy_heuristic = np.zeros_like(policy_nn) if policy_nn is not None else np.zeros(self.state_encoder.total_moves)
            if len(valid_moves) > 0:
                for move in valid_moves:
                    idx = self.state_encoder.move_to_index(move)
                    if 0 <= idx < len(policy_heuristic):
                        policy_heuristic[idx] = 1.0 / len(valid_moves)
        else:
            policy_heuristic = None
            value_heuristic = 0.0

        # Combine based on evaluation mode
        if self.evaluation_mode == "pure_neural":
            return policy_nn, value_nn
        elif self.evaluation_mode == "pure_heuristic":
            return policy_heuristic, value_heuristic
        else:  # hybrid mode
            # Weighted combination
            w_h = self.heuristic_weight
            w_n = 1.0 - w_h

            # Combine policy (weighted sum)
            policy_combined = w_n * policy_nn + w_h * policy_heuristic

            # Combine value (weighted sum)
            value_combined = w_n * value_nn + w_h * value_heuristic

            return policy_combined, value_combined

    def _get_value(self, state: GameState) -> Optional[float]:
        """Terminal value from the leaf player's POV, or None if non-terminal.

        Returns the score margin clipped to [-1, +1] in `state.current_player`'s
        POV. Network and heuristic evaluators already produce leaf-player-POV
        values (the encoder is side-normalized), so callers can treat all leaf
        values as same-convention and pass them straight to `_backpropagate`,
        which expects leaf-player POV.
        """
        if not state.is_terminal():
            return None

        score_diff = state.white_score - state.black_score
        normalized_margin = score_diff / 3.0  # Max score difference is 3 (3-0 or 0-3)
        white_pov_value = float(np.clip(normalized_margin, -1.0, 1.0))

        # Flip to leaf-player POV. White-POV is positive when white scored more;
        # if Black is the player to move at the terminal state, that's a loss
        # for Black and the value should be negated.
        if state.current_player == Player.WHITE:
            return white_pov_value
        return -white_pov_value


    def _mask_invalid_moves(self, policy: np.ndarray, valid_moves: List[Move]) -> np.ndarray:
        """Mask out invalid moves from the policy vector and re-normalize."""
        mask = np.zeros_like(policy, dtype=bool)
        valid_indices = []
        for move in valid_moves:
             idx = self.state_encoder.move_to_index(move)
             if 0 <= idx < len(policy):
                  mask[idx] = True
                  valid_indices.append(idx)
             else:
                  self.logger.error(f"Invalid move index {idx} encountered in _mask_invalid_moves for move {move}. Max index: {len(policy)-1}")

        # Apply mask
        masked_policy = np.where(mask, policy, 0.0)

        # Re-normalize
        policy_sum = masked_policy.sum()
        if policy_sum > 1e-6: # Use a small epsilon for floating point comparison
            normalized_policy = masked_policy / policy_sum
        else:
            # If the sum is zero (e.g., network assigned zero probability to all valid moves),
            # assign uniform probability to valid moves.
            self.logger.warning("MCTS: All valid moves have near-zero probability in policy. Using uniform distribution.")
            num_valid = len(valid_indices)
            if num_valid > 0:
                uniform_prob = 1.0 / num_valid
                normalized_policy = np.zeros_like(policy)
                for idx in valid_indices:
                     normalized_policy[idx] = uniform_prob
            else: # Should not happen if valid_moves is not empty
                 normalized_policy = masked_policy # Return the zero vector

        return normalized_policy


    def _backpropagate(self, path: List[Node], players: List[Player], value: float):
        """Backpropagate the leaf evaluation up the search path.

        Storage convention (preserved from prior code): each node stores
        ``value_sum`` from its parent's player's POV; root, having no real
        parent, stores from the opposite-of-root POV. PUCT (`_select_action`)
        consumes ``child.value()`` directly as Q from the parent's POV.

        YINSH-aware: ``GameState._switch_player`` does not fire on every move
        (capture sequences MOVE_RING→ROW_COMPLETION→REMOVE_MARKERS→REMOVE_RING
        keep the same player to move across 3+ plies). Walking up the path,
        the running value's POV only flips at edges where the player actually
        changes, not unconditionally per ply.

        Args:
            path: nodes from root to leaf, len(path) >= 1.
            players: parallel list, ``players[i]`` is the player to move at
                ``path[i]``. ``len(players) == len(path)``.
            value: leaf evaluation in ``players[-1]``'s POV.
        """
        running = value
        n = len(path)
        for i in range(n - 1, -1, -1):
            node = path[i]
            node.visit_count += 1
            # Storage POV at this node:
            #   non-root with same-player parent -> parent-POV == running's POV: store +running
            #   non-root with different-player parent -> parent-POV is flipped: store -running
            #   root (no parent) -> legacy convention, store -running
            same_as_parent = (i > 0 and players[i] == players[i - 1])
            if same_as_parent:
                node.value_sum += running
            else:
                node.value_sum += -running
            # Update running for the parent's iteration (running should be in
            # players[i-1]'s POV at the next step).
            if i > 0 and players[i] != players[i - 1]:
                running = -running


    def _get_move_prob(self, policy: np.ndarray, move: Move) -> float:
        """Get move probability from the (normalized) policy array."""
        move_idx = self.state_encoder.move_to_index(move)
        if 0 <= move_idx < len(policy):
            return policy[move_idx]
        else:
            self.logger.error(f"Invalid move index {move_idx} requested in _get_move_prob for move {move}. Max index: {len(policy)-1}")
            return 0.0


class SelfPlay:
    """Handles self-play game generation using configured MCTS and temperature."""

    def __init__(self,
                 network: NetworkWrapper,
                 num_workers: int,
                 # --- Evaluation Mode ---
                 evaluation_mode: str = "pure_neural",
                 heuristic_weight: float = 0.5,
                 # --- MCTS Params (to be passed to MCTS instance) ---
                 num_simulations: int = 100, # Early game sims
                 late_simulations: Optional[int] = None,
                 simulation_switch_ply: int = 20,
                 # Probabilistic fast/slow sim split (default off)
                 fast_simulations: int = 0,
                 fast_sim_prob: float = 0.0,
                 c_puct: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 # *** Use consistent 'value_weight' naming ***
                 value_weight: float = 1.0,
                 max_depth: int = 500,
                 # --- Batched MCTS Params ---
                 use_batched_mcts: bool = True,  # Enable batched MCTS for 10-20x speedup
                 mcts_batch_size: int = 32,  # Number of leaves to collect before batching
                 # --- Temp Params (to be passed to MCTS instance) ---
                 initial_temp: float = 1.0,
                 final_temp: float = 0.1,
                 annealing_steps: int = 30,
                 temp_clamp_fraction: float = 0.6,
                 # --- Optional Metrics ---
                 metrics_logger: Optional[MetricsLogger] = None,
                 mcts_metrics: Optional[MCTSMetrics] = None,
                 # --- Memory Pools ---
                 game_state_pool: Optional[GameStatePool] = None,
                 # --- Subtree reuse ---
                 enable_subtree_reuse: bool = True,
                 # --- First-Play Urgency ---
                 fpu_reduction: float = 0.25,
                 # --- Root Dirichlet noise mixing ---
                 epsilon_mix_start: float = 0.25,
                 epsilon_mix_end: float = 0.0,
                 epsilon_mix_taper_moves: int = 20,
                 root_policy_temp: float = 1.0,
                 # --- C++ bitboard engine ---
                 use_cpp_engine: bool = False,
                ):
        """
        Initialize SelfPlay with explicit MCTS and temperature parameters.
        """
        self.network = network
        self.num_workers = num_workers # Use the calculated number passed in
        self.metrics_logger = metrics_logger
        # Use network's encoder to ensure consistent encoding (basic or enhanced)
        self.state_encoder = network.state_encoder
        self.logger = logging.getLogger("SelfPlay") # Use module logger
        self.current_iteration = 0 # Track current training iteration if needed

        # Store memory pools
        self.game_state_pool = game_state_pool

        # Initialize heuristic evaluator
        self.evaluation_mode = evaluation_mode
        self.heuristic_weight = heuristic_weight
        if evaluation_mode in ["pure_heuristic", "hybrid"]:
            # See yinsh_ml/search/mcts.py for the rationale: MCTS already
            # does multi-ply lookahead via simulation; the heuristic's own
            # forced-sequence detector duplicates that and dominates wall-
            # clock. Disable it for MCTS-side use.
            self.heuristic_evaluator = YinshHeuristics(
                enable_forced_sequence_detection=False,
            )
            self.logger.info(
                f"Initialized YinshHeuristics evaluator for {evaluation_mode} mode "
                f"(forced-sequence detection disabled)"
            )
        else:
            self.heuristic_evaluator = None

        # Store batched MCTS configuration
        self.use_batched_mcts = use_batched_mcts
        self.mcts_batch_size = mcts_batch_size

        # Store all parameters needed by MCTS and workers explicitly
        self.mcts_config = {
            'evaluation_mode': evaluation_mode,
            'heuristic_evaluator': self.heuristic_evaluator,
            'heuristic_weight': heuristic_weight,
            'num_simulations': num_simulations,
            'late_simulations': late_simulations,
            'simulation_switch_ply': simulation_switch_ply,
            'fast_simulations': fast_simulations,
            'fast_sim_prob': fast_sim_prob,
            'c_puct': c_puct,
            'dirichlet_alpha': dirichlet_alpha,
            'value_weight': value_weight, # Store with consistent name
            'max_depth': max_depth,
            'initial_temp': initial_temp,
            'final_temp': final_temp,
            'annealing_steps': annealing_steps,
            'temp_clamp_fraction': temp_clamp_fraction,
            'use_batched_mcts': use_batched_mcts,  # Add batched MCTS flag
            'mcts_batch_size': mcts_batch_size,  # Add batch size
            'use_enhanced_encoding': getattr(self.network, 'use_enhanced_encoding', False),  # Enhanced encoding flag
            'enable_subtree_reuse': enable_subtree_reuse,  # Carry MCTS tree across moves within a game
            'fpu_reduction': fpu_reduction,  # First-Play Urgency coefficient (KataGo-style PUCT)
            'epsilon_mix_start': epsilon_mix_start,  # Root Dirichlet mixing fraction at move 0
            'epsilon_mix_end': epsilon_mix_end,  # Root mixing fraction after taper (0 = no late-game noise)
            'epsilon_mix_taper_moves': epsilon_mix_taper_moves,  # Linear taper horizon
            'root_policy_temp': root_policy_temp,  # Reshape root prior; >1 flattens, <1 sharpens
            'use_cpp_engine': use_cpp_engine,  # Opt-in to game_cpp engine in workers
        }
        # Store optional metrics instances
        self.mcts_metrics = mcts_metrics # Store the instance

        # Instantiate the main MCTS object for the SelfPlay process itself (if needed, e.g., single-threaded play)
        # This MCTS instance is primarily used if play_game is called directly,
        # worker processes will create their own MCTS instances.

        # Filter out batched MCTS params, encoding flag, and engine flag
        # that don't belong in MCTS.__init__()
        mcts_init_config = {k: v for k, v in self.mcts_config.items()
                           if k not in ['use_batched_mcts', 'mcts_batch_size',
                                        'use_enhanced_encoding', 'use_cpp_engine']}

        self.mcts = MCTS(
            network=self.network,
            mcts_metrics=self.mcts_metrics, # Pass metrics instance
            game_state_pool=self.game_state_pool, # Pass memory pool
            **mcts_init_config # Unpack the filtered config dict
        )

        self.logger.info("SelfPlay Initialized:")
        self.logger.info(f"  Workers: {self.num_workers}")
        self.logger.info(f"  Evaluation Mode: {self.evaluation_mode} (heuristic_weight={self.heuristic_weight:.3f})")
        self.logger.info(f"  Batched MCTS: {'ENABLED' if self.use_batched_mcts else 'DISABLED'} (batch_size={self.mcts_batch_size})")
        self.logger.info(f"  Enhanced Encoding: {'ENABLED (15 channels)' if self.mcts_config.get('use_enhanced_encoding', False) else 'DISABLED (6 channels)'}")
        self.logger.info(f"  MCTS Config: {self.mcts_config}")


    def generate_games(self, num_games: int) -> List[Tuple[List[np.ndarray], List[np.ndarray], List[float], List[Dict]]]:
        """Generate self-play games in parallel using multiple workers.

        Fix #1: Now returns MCTS root values (List[float]) instead of single game outcome (float)
        """
        self.logger.info(f"Starting generation of {num_games} games using {self.num_workers} workers...")
        # Fix #1: Changed from outcome (float) to values (List[float]) - MCTS root values per position
        games_data: List[Tuple[List[np.ndarray], List[np.ndarray], List[float], List[Dict]]] = []

        # Ensure the main network is on the CPU before saving state dict for workers
        self.network.network.cpu()
        network_params = self.network.network.state_dict()
        # Move network back to original device if needed
        self.network.network.to(self.network.device)


        # Create a temporary file to share the model state_dict
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
                torch.save(network_params, tmp.name)
                model_path = tmp.name
                self.logger.debug(f"Saved temporary model state_dict for workers to: {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to create temporary model file: {e}", exc_info=True)
            return [] # Return empty list if model cannot be shared


        start_time = time.time()
        games_completed = 0
        futures = []

        self.logger.info(f"MCTS config being used: sims={self.mcts.early_simulations}, "
                         f"late_sims={self.mcts.late_simulations}, "
                         f"switch_ply={self.mcts.switch_ply}, c_puct={self.mcts.c_puct}")

        # Handle sequential generation when workers=0 (memory-optimized mode)
        if self.num_workers <= 0:
            self.logger.info("Running sequential game generation (workers=0, memory-optimized mode)")
            for game_id in range(num_games):
                try:
                    result = play_game_worker(
                        model_path=model_path,
                        game_id=game_id,
                        mcts_config=self.mcts_config,
                    )
                    if result is not None:
                        states, policies, values, temp_data, game_history = result
                        games_data.append((states, policies, values, game_history))
                        games_completed += 1

                        # Log progress
                        if games_completed % max(1, num_games // 10) == 0 or games_completed == num_games:
                            elapsed = time.time() - start_time
                            rate = games_completed / elapsed if elapsed > 0 else 0
                            self.logger.info(f"Games Generated: {games_completed}/{num_games} ({rate:.2f} games/s)")
                    else:
                        self.logger.warning(f"Game {game_id} returned None result.")
                except Exception as e:
                    self.logger.error(f"Error generating game {game_id}: {e}", exc_info=True)

            # Cleanup temp model file
            if 'model_path' in locals() and os.path.exists(model_path):
                try:
                    os.unlink(model_path)
                except OSError:
                    pass

            total_time = time.time() - start_time
            final_rate = games_completed / total_time if total_time > 0 else 0
            self.logger.info(f"\nGame generation complete:")
            self.logger.info(f"- Games generated: {games_completed}/{num_games}")
            self.logger.info(f"- Total time: {total_time:.1f} seconds")
            self.logger.info(f"- Final rate: {final_rate:.2f} games/second")
            return games_data

        # Parallel generation with workers > 0
        try:
            with ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=_SELF_PLAY_MP_CONTEXT,
            ) as executor:
                for game_id in range(num_games):
                    # Submit worker jobs, passing the *explicit* MCTS configuration
                    futures.append(
                        executor.submit(
                            play_game_worker,
                            model_path=model_path,
                            game_id=game_id,
                            # --- Pass all necessary config parameters ---
                            mcts_config=self.mcts_config, # Pass the whole config dict
                            # mcts_metrics=self.mcts_metrics # Pass metrics collector if needed by worker
                            # --- Pass other necessary args ---
                        )
                    )

                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        # Check if worker returned valid data (not None or error)
                        if result is not None:
                             # Fix #1: Unpack values (MCTS root values) instead of outcome
                             states, policies, values, temp_data, game_history = result
                             games_data.append((states, policies, values, game_history))
                             games_completed += 1

                             # Log progress periodically
                             if games_completed % max(1, num_games // 10) == 0 or games_completed == num_games:
                                 elapsed = time.time() - start_time
                                 rate = games_completed / elapsed if elapsed > 0 else 0
                                 self.logger.info(f"Games Generated: {games_completed}/{num_games} ({rate:.2f} games/s)")
                                 # Log CPU usage (optional)
                                 # cpu_percent = psutil.cpu_percent(percpu=True)
                                 # avg_cpu = sum(cpu_percent) / len(cpu_percent) if cpu_percent else 0
                                 # self.logger.debug(f"CPU Usage: {avg_cpu:.1f}%")

                             # --- Optional: Log detailed game metrics ---
                             # if self.metrics_logger:
                             #     game_metrics = GameMetrics(...) # Populate with data from result
                             #     self.metrics_logger.log_game(game_metrics)

                        else:
                             self.logger.warning(f"Worker returned None result for a game.")

                    except Exception as e:
                        # Log errors from worker processes
                        self.logger.error(f"Error processing game result from worker: {e}", exc_info=True)

        except Exception as e:
             self.logger.error(f"Error during parallel game generation: {e}", exc_info=True)
        finally:
            # --- Cleanup ---
            # Ensure the temporary model file is deleted
            if 'model_path' in locals() and os.path.exists(model_path):
                try:
                    os.unlink(model_path)
                    self.logger.debug(f"Deleted temporary model file: {model_path}")
                except OSError as e:
                    self.logger.error(f"Error deleting temporary model file {model_path}: {e}")

        total_time = time.time() - start_time
        final_rate = games_completed / total_time if total_time > 0 else 0
        self.logger.info(f"\nGame generation complete:")
        self.logger.info(f"- Games generated: {games_completed}/{num_games}")
        self.logger.info(f"- Total time: {total_time:.1f} seconds")
        if games_completed > 0:
            self.logger.info(f"- Average time per game: {total_time / games_completed:.2f} seconds")
        self.logger.info(f"- Final rate: {final_rate:.2f} games/second")

        # Return collected game data (states, policies, outcome, history)
        return games_data


    # Removed play_game method as it's superseded by play_game_worker
    # Removed export_games method (can be handled by supervisor or runner if needed)
    # Removed _collect_phase_values (can be done by supervisor/runner from game history)
    # Removed _get_optimal_workers (now done by supervisor)


def play_game_worker(
        model_path: str,
        game_id: int,
        mcts_config: Dict,
) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], float, Dict, List[Dict]]]:
    """
    Memory-optimized worker function for self-play game generation.
    """
    # Configure minimal logging to reduce memory overhead
    worker_logger = logging.getLogger(f"Worker-{game_id}")
    if not worker_logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        worker_logger.addHandler(handler)
        worker_logger.propagate = False
    worker_logger.setLevel(logging.INFO)

    worker_logger.info(f"Starting game {game_id}")

    try:
        # Worker device: prefer CUDA when available (cloud GPU boxes),
        # otherwise CPU. Historically this was hardcoded to CPU because
        # Mac+fork+MPS was unstable — with the spawn start method that's
        # no longer a concern. Each worker gets its own CUDA context
        # (spawn, not fork), so multiple workers share the GPU safely.
        # Can be overridden via mcts_config['worker_device'] for debugging.
        _worker_device_override = mcts_config.get('worker_device')
        if _worker_device_override:
            device = torch.device(_worker_device_override)
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        # Create local tensor pool for network operations
        from ..memory import TensorPool, TensorPoolConfig
        tensor_pool_config = TensorPoolConfig(
            initial_size=30,   # Enough for several concurrent predictions
            enable_statistics=False,  # Keep overhead low in workers
            enable_adaptive_sizing=True,  # Enable adaptive sizing
            enable_tensor_reshaping=True,  # Enable tensor reshaping
            auto_device_selection=True    # Enable device-specific pooling
        )
        local_tensor_pool = TensorPool(tensor_pool_config)

        # Create network with tensor pool (use enhanced encoding if specified)
        use_enhanced_encoding = mcts_config.get('use_enhanced_encoding', False)
        network = NetworkWrapper(device=device, tensor_pool=local_tensor_pool,
                                 use_enhanced_encoding=use_enhanced_encoding)
        network.load_model(model_path)
        network.network.eval()

        # Engine selection. The C++ bitboard engine has its own ~zero-cost
        # clone() (struct memcpy) so the GameStatePool's whole purpose —
        # amortizing GameState alloc + dict-resize cost — is moot. When
        # enabled, skip pool wiring entirely and instantiate a
        # CppGameState directly. Both paths land at MCTS via the same
        # duck-typed surface.
        use_cpp_engine = bool(mcts_config.get('use_cpp_engine', False))
        if use_cpp_engine:
            from ..game_cpp import CppGameState
            local_game_state_pool = None
            worker_logger.info(f"Game {game_id}: using C++ bitboard engine")
        else:
            from ..memory import GameStatePool, GameStatePoolConfig
            from ..game import GameState
            pool_config = GameStatePoolConfig(
                initial_size=50,  # Start with moderate pool size for worker
                enable_statistics=False,  # Disable stats in workers for performance
                factory_func=GameState  # Use GameState constructor as factory
            )
            local_game_state_pool = GameStatePool(pool_config)

        # Use network's encoder to ensure consistent encoding (basic or enhanced)
        state_encoder = network.state_encoder

        # Extract batched MCTS settings from config (with defaults)
        use_batched_mcts = mcts_config.get('use_batched_mcts', True)
        mcts_batch_size = mcts_config.get('mcts_batch_size', 32)

        # Remove these from mcts_config before passing to MCTS __init__
        mcts_init_config = {k: v for k, v in mcts_config.items()
                           if k not in ['use_batched_mcts', 'mcts_batch_size',
                                        'use_enhanced_encoding', 'use_cpp_engine']}

        mcts = MCTS(network=network, game_state_pool=local_game_state_pool, **mcts_init_config)

        # --- Initial state: pool-allocated GameState or fresh CppGameState ---
        if use_cpp_engine:
            state = CppGameState()
        else:
            state = local_game_state_pool.get()  # Get initial state from pool
            from ..memory import reset_game_state
            reset_game_state(state)  # Ensure it's properly initialized

        # Use lists instead of deques (less overhead)
        states = []
        policies = []
        players = []  # Track which player is to move at each position (for outcome perspective)

        # Keep minimal game history - just essentials for debugging
        game_history = []

        # Use a lightweight dict for temperature data
        temp_data = {'temperatures': [], 'entropies': [], 'search_times': []}

        move_count = 0
        max_game_moves = 300  # Safety limit

        # --- Main Game Loop ---
        while not state.is_terminal() and move_count < max_game_moves:
            # MCTS search with timing (use batched or serial based on configuration)
            search_start = time.time()
            if use_batched_mcts:
                move_probs = mcts.search_batch(state, move_count, batch_size=mcts_batch_size)
            else:
                move_probs = mcts.search(state, move_count)
            search_time = time.time() - search_start

            # Record search time
            temp_data['search_times'].append(search_time)

            # --- Action Selection ---
            temp = mcts.get_temperature(move_count)
            valid_moves = state.get_valid_moves()

            if not valid_moves:
                worker_logger.warning(f"No valid moves at move {move_count}. Ending game.")
                break

            # Get valid move probabilities
            valid_indices = [state_encoder.move_to_index(move) for move in valid_moves]
            valid_move_probs = move_probs[valid_indices]

            # Normalize probabilities
            prob_sum = valid_move_probs.sum()
            if prob_sum < 1e-6:
                valid_move_probs = np.ones(len(valid_moves), dtype=np.float32) / len(valid_moves)
            else:
                valid_move_probs /= prob_sum

            # Sample action based on temperature
            if temp == 0 or temp < 0.01:  # Greedy selection for very low temp
                selected_idx_in_valid = np.argmax(valid_move_probs)
            else:
                # Sample move based on probabilities
                try:
                    selected_idx_in_valid = np.random.choice(len(valid_moves), p=valid_move_probs)
                except ValueError as e:
                    worker_logger.error(f"Sampling error: {e}")
                    selected_idx_in_valid = np.argmax(valid_move_probs)

            selected_move = valid_moves[selected_idx_in_valid]

            # --- Record State and Policy ---
            # Store encoded state - using np.float32 instead of float64 reduces memory by half
            encoded_state = state_encoder.encode_state(state).astype(np.float32)
            states.append(encoded_state)
            policies.append(move_probs)
            players.append(state.current_player)  # Track player for outcome perspective

            # Calculate entropy for temperature adaptation
            entropy = -np.sum(valid_move_probs * np.log(valid_move_probs + 1e-9))

            # Record temperature data - just essentials
            temp_data['temperatures'].append(temp)
            temp_data['entropies'].append(entropy)

            # Only store minimal state info in history to save memory
            if move_count % 5 == 0:  # Only record every 5th move to history to save memory
                game_history.append({
                    'move_number': move_count,
                    'move': str(selected_move),
                    'phase': str(state.game_phase) if hasattr(state, 'game_phase') else "UNKNOWN",
                })

            # Make move
            state.make_move(selected_move)
            # Carry the MCTS tree across to next move: reseat the root on the
            # subtree for the move just played, freeing siblings. No-op if
            # subtree reuse is disabled.
            mcts.advance_root(selected_move)
            move_count += 1

            # Perform occasional garbage collection during very long games
            if move_count > 100 and move_count % 50 == 0:
                pass  # Memory pools handle cleanup automatically

            # Check for terminal state
            if state.is_terminal():
                worker_logger.debug(f"Terminal state reached after {move_count} moves")
                break

        # --- After Game Loop Ends ---
        # Record final state
        final_encoded_state = state_encoder.encode_state(state).astype(np.float32)
        states.append(final_encoded_state)

        # Add dummy policy for final state
        dummy_policy = np.zeros_like(policies[0]) if policies else np.zeros(state_encoder.total_moves, dtype=np.float32)
        policies.append(dummy_policy)
        players.append(state.current_player)  # Track player for final state too

        # Add final game info
        game_history.append({
            'move_number': move_count,
            'final_state': True,
            'white_score': state.white_score,
            'black_score': state.black_score
        })

        # Calculate outcome from White's perspective (+1 = White wins, -1 = Black wins)
        score_diff = state.white_score - state.black_score
        outcome_white = float(np.clip(score_diff / 3.0, -1.0, 1.0))

        # Backfill values for all positions based on game outcome
        # Each position's value = outcome from that player's perspective
        from ..game.types import Player
        values = []
        for player in players:
            if player == Player.WHITE:
                values.append(outcome_white)
            else:
                values.append(-outcome_white)  # Flip for Black's perspective

        outcome = outcome_white  # For logging

        # Log final summary - minimal logging to reduce overhead
        worker_logger.info(
            f"Game {game_id} finished in {move_count} moves. "
            f"Score: W={state.white_score}, B={state.black_score}. Outcome={outcome:.3f}"
        )

        # Heuristic eval-cache hit-rate. Surfacing this alongside the game
        # summary makes it easy to spot when the cache is misconfigured
        # (e.g. evaluator getting reconstructed per-call) without re-running
        # the cProfile script. See BITBOARD_FOLLOWUP_PLAN.md Candidate A.
        if (mcts.heuristic_evaluator is not None
                and hasattr(mcts.heuristic_evaluator, "cache_stats")):
            cs = mcts.heuristic_evaluator.cache_stats()
            if cs["hits"] + cs["misses"] > 0:
                worker_logger.info(
                    f"Game {game_id} eval-cache: hit_rate={cs['hit_rate']:.1%} "
                    f"hits={cs['hits']} misses={cs['misses']} size={cs['size']}/{cs['capacity']}"
                )

        # Clean up memory before returning
        # Release state if possible (CppGameState path runs without a pool)
        if ('local_game_state_pool' in locals()
                and local_game_state_pool is not None
                and 'state' in locals() and state is not None):
            local_game_state_pool.return_game_state(state)

        # Free any retained MCTS search tree (subtree reuse holds it between moves).
        mcts.reset_tree()
        del network
        del mcts.network  # Explicitly clear network reference
        del mcts

        # Memory pools handle cleanup automatically

        # Return outcome-based values (standard AlphaZero approach)
        # Each position's value = game outcome from that player's perspective
        return states, policies, values, temp_data, game_history

    except Exception as e:
        worker_logger.error(f"Error in game {game_id}: {e}", exc_info=True)
        # Ensure cleanup even on error
        try:
            # Release state if possible
            if ('local_game_state_pool' in locals()
                    and local_game_state_pool is not None
                    and 'state' in locals() and state is not None):
                local_game_state_pool.return_game_state(state)
            if 'mcts' in locals():
                mcts.reset_tree()
            del network
            del mcts
            # Memory pools handle cleanup automatically
        except:
            pass
        return None