"""Tournament system for evaluating YINSH models within a training run."""

import hashlib
import logging
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import torch
from datetime import datetime
import json

import numpy as np

from ..network.wrapper import NetworkWrapper
from ..game.game_state import GameState
from ..game.constants import Player
from .elo_manager import EloTracker, MatchResult

if TYPE_CHECKING:  # type-only import to avoid circular import at runtime
    from .metrics_logger import MetricsLogger

import math


def win_rate_to_elo_delta(win_rate: float) -> float:
    """Convert a candidate win rate against a fixed opponent into an Elo
    delta. Used to render anchor eval results as Elo numbers so dashboards
    can plot both raw-policy and MCTS strength on the same axis.

    Edge cases: 0.0 → -inf, 1.0 → +inf. Clamp to ±999 (~99.9% / 0.1%) so we
    return a finite, plottable number. The anchor is fixed across iterations,
    so this is a real Elo delta on the anchor's scale (anchor Elo unchanged).
    """
    eps = 1e-3
    p = max(min(float(win_rate), 1.0 - eps), eps)
    return 400.0 * math.log10(p / (1.0 - p))


def derive_match_seed(base_seed: int, white_id: str, black_id: str, game_num: int) -> int:
    """Stable per-game seed. Different orientations / game indices get different
    seeds; same (base, white, black, game_num) always reproduces. Uses blake2b
    rather than Python's built-in hash so it survives `PYTHONHASHSEED` randomization."""
    h = hashlib.blake2b(digest_size=8)
    h.update(str(base_seed).encode())
    h.update(b"|")
    h.update(white_id.encode())
    h.update(b"|")
    h.update(black_id.encode())
    h.update(b"|")
    h.update(str(game_num).encode())
    return int.from_bytes(h.digest(), 'big') & 0x7FFFFFFF

q = math.log(10) / 400  # constant used in Glicko formulas

def _canon(model_id: str) -> str:
    """
    Canonicalise a model identifier so everybody uses the same key:

    • strips directory components
    • strips the `.pt` extension
    • keeps the "iteration_X" stem
    """
    return Path(model_id).stem            # "…/checkpoint_iteration_3.pt" → "checkpoint_iteration_3"

class GlickoPlayer:
    def __init__(self, rating=1500.0, rd=350.0):
        self.rating = rating
        self.rd = rd  # Rating Deviation

class GlickoTracker:
    """
    Robust Glicko-1 tracker.

    Records match results during a rating period and then updates each player's
    rating and rating deviation (RD) using the full set of matches.
    """
    def __init__(self, training_dir, initial_rating=1500.0, initial_rd=350.0, K_factor=0.00001):
        self.training_dir = training_dir
        self.initial_rating = initial_rating
        self.initial_rd = initial_rd
        self.K_factor = K_factor  # New dampening factor
        self.players = {}  # model_id -> GlickoPlayer
        self.match_history = []  # list of match records for the rating period

    def add_model(self, model_id):
        """Ensure a model is in the tracker."""
        if model_id not in self.players:
            self.players[model_id] = GlickoPlayer(self.initial_rating, self.initial_rd)

    def g(self, rd):
        """Scaling function of opponent's RD."""
        return 1 / math.sqrt(1 + 3 * (q ** 2) * (rd ** 2) / (math.pi ** 2))

    def E(self, rating, opp_rating, opp_rd):
        """Expected score for a player against an opponent."""
        return 1 / (1 + 10 ** (-self.g(opp_rd) * (rating - opp_rating) / 400.0))

    def record_match(self, white_model, black_model, white_wins, black_wins, draws):
        """
        Record a match result.

        For an aggregated match (e.g. 50 games), total_games = white_wins+black_wins+draws.
        For white, S = (white_wins + 0.5 * draws) / total_games; similarly for black.
        """
        total_games = white_wins + black_wins + draws
        if total_games == 0:
            return
        white_score = (white_wins + 0.5 * draws) / total_games
        black_score = (black_wins + 0.5 * draws) / total_games
        self.match_history.append({
            'white_model': white_model,
            'black_model': black_model,
            'white_score': white_score,
            'black_score': black_score
        })
        # Ensure both players are in the tracker.
        self.add_model(white_model)
        self.add_model(black_model)

    def update_ratings(self):
        """
        Update all players' ratings and RDs using Glicko-1 formulas over the entire rating period.

        For each player, gather all matches played in this period and then:
          1. Compute the variance v.
          2. Compute the rating change delta.
          3. Update the rating and RD.
        The final rating update is scaled by self.K_factor to prevent over-large changes.
        """
        # Organize results per player:
        results = {}  # player_id -> list of (opp_rating, opp_rd, score)
        for match in self.match_history:
            white = match['white_model']
            black = match['black_model']
            white_score = match['white_score']
            black_score = match['black_score']
            if white not in results:
                results[white] = []
            if black not in results:
                results[black] = []
            # For white: opponent is black.
            opp_black = self.players[black]
            results[white].append((opp_black.rating, opp_black.rd, white_score))
            # For black: opponent is white.
            opp_white = self.players[white]
            results[black].append((opp_white.rating, opp_white.rd, black_score))

        # Update each player's rating and RD based on all matches:
        for player_id, matches in results.items():
            player = self.players[player_id]
            if not matches:
                continue

            # Compute variance v:
            sum_term = 0.0
            for opp_rating, opp_rd, score in matches:
                E_val = self.E(player.rating, opp_rating, opp_rd)
                sum_term += (self.g(opp_rd) ** 2) * E_val * (1 - E_val)
            v = 1 / (q ** 2 * sum_term) if sum_term != 0 else float('inf')

            # Compute delta:
            delta = 0.0
            for opp_rating, opp_rd, score in matches:
                E_val = self.E(player.rating, opp_rating, opp_rd)
                delta += self.g(opp_rd) * (score - E_val)
            delta *= v * q

            # Compute the update denominator:
            rating_inv = 1 / (player.rd ** 2) + 1 / v

            # Apply dampening via K_factor:
            new_rating = player.rating + self.K_factor * (delta / rating_inv)
            new_rd = math.sqrt(1 / rating_inv)

            player.rating = new_rating
            player.rd = new_rd

        # Clear match history after processing the rating period.
        self.match_history = []

    def clear_model_cache(self):
        """
        Clear cached models to free memory.
        Add this method to your GlickoTracker class.
        """
        # If your GlickoTracker stores models in a dictionary or other structure:
        if hasattr(self, 'players'):
            # We only want to clear cached data, not ratings
            # So we'll keep the ratings but clear any large objects
            current_ratings = {}
            for model_id, player in self.players.items():
                # Store just the essential rating data
                current_ratings[model_id] = {
                    'rating': player.rating,
                    'rd': player.rd
                }

            # Clear the full players dictionary
            self.players.clear()

            # Restore just the ratings data
            for model_id, rating_data in current_ratings.items():
                if model_id not in self.players:
                    from yinsh_ml.utils.tournament import GlickoPlayer
                    self.players[model_id] = GlickoPlayer(
                        rating=rating_data['rating'],
                        rd=rating_data['rd']
                    )

            # Force garbage collection
            # Memory pools handle cleanup automatically
            pass

            print(f"[Tournament] Cleared model cache, keeping {len(self.players)} player ratings")

    def get_rating(self, model_id):
        """Return the current rating of the specified model."""
        if model_id in self.players:
            return self.players[model_id].rating
        return self.initial_rating

    def get_rd(self, model_id):
        """Return the current RD of the specified model."""
        if model_id in self.players:
            return self.players[model_id].rd
        return self.initial_rd

    def get_match_history(self, model_id):
        """Return all recorded match results for a given model (as stored during the rating period)."""
        # This returns the matches recorded before update_ratings() is called.
        return [m for m in self.match_history if m['white_model'] == model_id or m['black_model'] == model_id]

class ModelTournament:
    """Manages tournaments between YINSH models from the current training run."""

    def __init__(self,
                 training_dir: Path,
                 device: str = 'cpu',
                 games_per_match: int = 50,
                 temperature: float = 0.1,
                 sliding_window_size: int = 5,
                 use_ema_for_eval: bool = True,
                 eval_seed: Optional[int] = None,
                 use_enhanced_encoding: bool = False,
                 value_head_type: Optional[str] = None):
        """
        Initialize tournament manager.

        Args:
            training_dir: Directory containing the current training run
            device: Device to run models on
            games_per_match: Number of games to play per match
            temperature: Temperature for move selection
            sliding_window_size: Max number of recent models to include in tournament.
                                 Set to 0 for full round-robin (not recommended for long runs).
                                 Default 5 = constant O(1) complexity: 10 pairs × 200 games = 4000 games.
            use_ema_for_eval: If True and a sibling `<ckpt>_ema.pt` exists, load
                the EMA-smoothed weights for match play. The model still keys
                off the non-EMA path (so promotion-gate lookups and tournament
                history remain aligned with the iteration checkpoint), but the
                weights used in games are the smoothed ones — reducing the
                single-iteration-noise bleed into gate decisions.
            eval_seed: If not None, enables deterministic match play. Each game
                seeds torch / numpy / random from a stable hash of
                ``(eval_seed, white_id, black_id, game_num)`` so reruns with the
                same models reproduce exactly, different (white, black)
                orientations still diverge, and games within a match aren't
                clones of each other. RNG state is saved and restored around
                the seeded region so nothing leaks into callers.
        """
        self.training_dir = Path(training_dir)
        self.device = device
        self.games_per_match = games_per_match
        self.sliding_window_size = sliding_window_size
        self.temperature = temperature
        self.use_ema_for_eval = use_ema_for_eval
        self.eval_seed = eval_seed
        # Encoder + head config used when constructing NetworkWrapper instances
        # for tournament / anchor evaluation. Must match the training network's
        # architecture or load_model hard-fails on channel / head mismatch.
        self.use_enhanced_encoding = use_enhanced_encoding
        self.value_head_type = value_head_type
        self.latest_summary_stats: Dict[str, Dict] = {}   # NEW
        self._pair_results: Dict[Tuple[str, str], Dict[str, int]] = {}


        # Initialize ELO tracker
        #self.elo_tracker = EloTracker(self.training_dir)

        #initialize glicko tracker
        self.glicko_tracker = GlickoTracker(self.training_dir)

        # Setup logging
        self.logger = logging.getLogger("ModelTournament")

        # Tournament tracking
        self.current_tournament_id = None
        self.tournament_history_file = self.training_dir / "tournament_history.json"

        # Load tournament history if exists
        self.tournament_history = self._load_tournament_history()

    def _load_tournament_history(self) -> Dict:
        """Load tournament history from file."""
        if self.tournament_history_file.exists():
            try:
                with open(self.tournament_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading tournament history: {e}")
        return {}

    def _save_tournament_history(self):
        """Save tournament history to file."""
        try:
            with open(self.tournament_history_file, 'w') as f:
                json.dump(self.tournament_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving tournament history: {e}")

    def _load_model(self, checkpoint_path: Path) -> NetworkWrapper:
        """Load a model from checkpoint.

        When `use_ema_for_eval` is set and a `<stem>_ema.pt` sibling exists,
        that EMA-smoothed weight file is used for play instead of the raw
        iteration checkpoint. The model is still keyed by the original path
        upstream so Glicko/promotion tracking stays consistent.
        """
        actual_path = checkpoint_path
        if self.use_ema_for_eval:
            ema_path = checkpoint_path.with_name(checkpoint_path.stem + '_ema.pt')
            if ema_path.exists():
                actual_path = ema_path
                self.logger.debug(f"Loading EMA weights for {checkpoint_path.name}: {ema_path}")
        wrapper_kwargs = {'device': self.device,
                          'use_enhanced_encoding': self.use_enhanced_encoding}
        if self.value_head_type is not None:
            wrapper_kwargs['value_head_type'] = self.value_head_type
        model = NetworkWrapper(**wrapper_kwargs)
        model.load_model(str(actual_path))
        return model

    def _play_match(self,
                    white_model: NetworkWrapper,
                    black_model: NetworkWrapper,
                    white_id: str,
                    black_id: str) -> MatchResult:
        """Play a match (several games) between two models.

        MEMORY OPTIMIZATION: Uses tensor pools to avoid creating new MPS tensors
        for each move. This prevents MPS driver memory from growing unbounded.
        """
        import gc

        white_wins = 0
        black_wins = 0
        draws = 0
        total_moves = 0

        # Pre-allocate reusable tensors for this match to minimize MPS allocations
        # These stay allocated for the entire match, avoiding repeated alloc/free cycles
        white_input_tensor = white_model._acquire_input_tensor(batch_size=1)
        black_input_tensor = black_model._acquire_input_tensor(batch_size=1)

        for game_num in range(self.games_per_match):
            self.logger.debug(f"Playing game {game_num + 1}/{self.games_per_match}: "
                              f"{white_id} (White) vs {black_id} (Black)")

            # Snapshot + seed RNGs per game when deterministic eval is enabled.
            # State is restored in the finally below so tournament play doesn't
            # leak into subsequent training / self-play RNG consumers.
            rng_snapshot = None
            if self.eval_seed is not None:
                rng_snapshot = (
                    torch.get_rng_state(),
                    np.random.get_state(),
                    random.getstate(),
                )
                game_seed = derive_match_seed(
                    self.eval_seed, white_id, black_id, game_num
                )
                torch.manual_seed(game_seed)
                np.random.seed(game_seed)
                random.seed(game_seed)

            game_state = GameState()
            move_count = 0

            while not game_state.is_terminal() and move_count < 500:
                current_model = white_model if game_state.current_player == Player.WHITE else black_model
                input_tensor = white_input_tensor if game_state.current_player == Player.WHITE else black_input_tensor

                # Get valid moves
                valid_moves = game_state.get_valid_moves()
                if not valid_moves:
                    break

                # MEMORY OPTIMIZATION: Encode state into pre-allocated tensor
                # This avoids creating a new MPS tensor for every move
                state_array = current_model.state_encoder.encode_state(game_state)
                input_tensor.copy_(torch.from_numpy(np.array(state_array)).unsqueeze(0))

                move_probs, _ = current_model.predict(input_tensor)

                # Select move
                selected_move = current_model.select_move(
                    move_probs, valid_moves, self.temperature
                )

                # Only delete move_probs (input_tensor is reused)
                del move_probs

                # Make move
                success = game_state.make_move(selected_move)
                if not success:
                    self.logger.error(
                        f"Invalid move by {game_state.current_player.name}"
                    )
                    break

                move_count += 1

            # Record game result
            winner = game_state.get_winner()
            if winner == Player.WHITE:
                white_wins += 1
            elif winner == Player.BLACK:
                black_wins += 1
            else:
                draws += 1

            total_moves += move_count

            # MEMORY: Clear game state after each game
            del game_state

            # Restore RNG state snapshotted above, so tournament seeding doesn't
            # bleed into training / self-play consumers between matches.
            if rng_snapshot is not None:
                torch_state, np_state, py_state = rng_snapshot
                torch.set_rng_state(torch_state)
                np.random.set_state(np_state)
                random.setstate(py_state)

            # MEMORY: Periodic cleanup every 20 games to prevent MPS memory accumulation
            if (game_num + 1) % 20 == 0:
                gc.collect()
                if torch.backends.mps.is_available():
                    try:
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                    except Exception:
                        pass

        # Release pre-allocated tensors back to pools
        white_model._release_tensor(white_input_tensor)
        black_model._release_tensor(black_input_tensor)

        match_result = MatchResult(
            white_model=white_id,
            black_model=black_id,
            white_wins=white_wins,
            black_wins=black_wins,
            draws=draws,
            avg_game_length=total_moves / self.games_per_match if self.games_per_match > 0 else 0
        )

        pair = tuple(sorted((white_id, black_id)))
        if pair not in self._pair_results:
            # wins, draws are always *from the perspective of the first element in `pair`*
            self._pair_results[pair] = {'wins_a': 0, 'wins_b': 0, 'draws': 0}

        if pair[0] == white_id:
            self._pair_results[pair]['wins_a'] += match_result.white_wins
            self._pair_results[pair]['wins_b'] += match_result.black_wins
        else:
            self._pair_results[pair]['wins_a'] += match_result.black_wins
            self._pair_results[pair]['wins_b'] += match_result.white_wins

        self._pair_results[pair]['draws'] += match_result.draws

        self.logger.debug(f"Match complete: White: {white_wins}, Black: {black_wins}, Draws: {draws}")
        return match_result

    def get_head_to_head(self, model_a: str, model_b: str) -> Tuple[int, int]:
        """
        Return (wins_by_A, total_games_between_A_and_B).
        model_a and model_b are expected in the format "iteration_X".
        The internal storage uses keys like "checkpoint_iteration_X".
        If the pair never met, return (0, 0) so the supervisor can decide what to do.
        """
        # Construct the keys used internally for storage
        # Example: model_a="iteration_1" -> key_a="checkpoint_iteration_1"
        try:
            # Extract the iteration number from the input IDs
            iter_a = model_a.split('_')[-1]
            iter_b = model_b.split('_')[-1]
            # Construct the keys as they are stored in _pair_results
            key_a = f"checkpoint_iteration_{iter_a}"
            key_b = f"checkpoint_iteration_{iter_b}"
        except IndexError:
            # Handle cases where the input ID format is unexpected
            self.logger.error(f"Could not parse iteration numbers from model IDs: {model_a}, {model_b}")
            return 0, 0  # Cannot proceed if IDs are malformed

        # Use the constructed internal keys for lookup
        pair = tuple(sorted((key_a, key_b)))

        rec = self._pair_results.get(pair)
        if rec is None:
            # Log which pair was missed for debugging
            self.logger.warning(
                f"Head-to-head data not found for lookup pair: {pair}. Available stored pairs: {list(self._pair_results.keys())}")
            return 0, 0

        # Determine wins based on which key is first in the sorted tuple 'pair'
        # Remember: rec['wins_a'] are wins for pair[0], rec['wins_b'] are wins for pair[1]
        if pair[0] == key_a:
            # model_a corresponds to the first element in the pair tuple
            wins_a = rec['wins_a']
        else:  # pair[0] must be key_b, so model_a corresponds to the second element
            wins_a = rec['wins_b']

        total = rec['wins_a'] + rec['wins_b'] + rec['draws']

        # Optional: Add a debug log to confirm successful retrieval
        self.logger.debug(
            f"Head-to-head lookup for ({model_a}, {model_b}): Found pair {pair}, wins_a={wins_a}, total={total}")

        return wins_a, total

    def _clear_model_memory(self):
        """Aggressively clear GPU/MPS memory after unloading models.

        MPS (Metal) memory management is lazy and requires explicit synchronization
        and multiple GC passes to properly release memory.
        """
        import gc
        import time

        # First GC pass - collect Python objects
        gc.collect()

        if torch.backends.mps.is_available():
            try:
                # CRITICAL: Synchronize MPS to ensure all GPU operations complete
                # Without this, empty_cache() may not release memory that's still "in flight"
                torch.mps.synchronize()

                # Clear the MPS cache
                torch.mps.empty_cache()

                # Second GC pass after MPS sync - catches objects that were waiting on GPU
                gc.collect()

                # Small delay to let MPS memory manager catch up
                time.sleep(0.1)

                # Final cache clear
                torch.mps.synchronize()
                torch.mps.empty_cache()

            except Exception as e:
                self.logger.debug(f"MPS cleanup warning: {e}")

        elif torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        # Final GC pass
        gc.collect()

    # ------------------------------------------------------------------ #
    # Absolute evaluation anchor (CLOUD_TRAINING_PLAN §1.3)
    # ------------------------------------------------------------------ #
    # The anchor eval plays a candidate NetworkWrapper against a fixed
    # HeuristicAgent opponent (same depth, same weights, same seed every
    # iteration). This gives an ABSOLUTE measure of playing strength
    # independent of the relative Elo from the round-robin, which is the
    # only way to interpret long training trajectories — relative ratings
    # drift with the pool, but anchor_win_rate is anchored to a fixed
    # baseline for the life of the run.
    #
    # Kept as a separate call (not folded into the round-robin) so it
    # survives any future changes to the Elo / promotion-gate logic.
    # TODO(W1-NEW B2 / W1e): Wire B2 value-outcome correlation here.
    #
    # The safeguard wants, per evaluation pass, a Pearson r between:
    #   (a) MCTS root value at the start of each evaluated position, in the
    #       POV of the player to move there
    #   (b) terminal outcome of that game in the same POV (+1 candidate win,
    #       -1 candidate loss, 0 draw, with sign-flip for white/black-as-
    #       candidate symmetry)
    # logged via ``metrics_logger.compute_and_log_value_outcome_correlation()``
    # at the end of the eval pass.
    #
    # Integration points when W1e lands:
    #   1. Plumb a `MetricsLogger` into `ModelTournament` (constructor arg),
    #      or set it on the instance from the supervisor right before the
    #      anchor eval call.
    #   2. In `run_anchor_eval` (use_mcts=True branch only — raw-policy mode
    #      has no MCTS root value), after each game:
    #        rv = game_mcts.last_root_value  # root value at game start
    #        outcome_in_pov = +1 if winner == candidate_color else -1 if winner else 0
    #        metrics_logger.log_eval_value_pair(rv, outcome_in_pov)
    #   3. After the game loop, call
    #        metrics_logger.compute_and_log_value_outcome_correlation(step=current_iteration)
    #
    # NOT done here yet: the W1e workstream is restructuring exactly the
    # game-loop block where (2) lands, and doing it twice is worse than
    # waiting. The MetricsLogger API for B2 IS in place (see
    # ``log_eval_value_pair`` / ``compute_and_log_value_outcome_correlation``)
    # and is unit-tested, so wiring is a 5-line change once W1e is stable.
    def run_anchor_eval(
        self,
        candidate_network: NetworkWrapper,
        candidate_label: str,
        num_games: int = 40,
        depth: int = 3,
        seed: int = 1337,
        max_moves_per_game: int = 200,
        use_mcts: bool = False,
        mcts_simulations: int = 64,
        heuristic_time_limit_seconds: float = 0.0,
        # BREAKING (T4.9): default changed 0.0 -> 0.5 so the eval doesn't
        # silently fall into argmax determinism mirages. Set explicitly to
        # 0.0 for legacy argmax behavior (and read the deterministic_sides
        # warning carefully if you do).
        candidate_temperature: float = 0.5,
        metrics_logger: Optional["MetricsLogger"] = None,
        iteration: Optional[int] = None,
    ) -> Dict:
        """Play the candidate network against a fixed HeuristicAgent baseline.

        Games are split half white / half black for the candidate, using a
        deterministic per-game seed derived from ``seed``. The baseline
        HeuristicAgent is constructed once per call with depth ``depth``,
        default phase-aware weights, and a fixed ``random_seed`` so its
        tie-breaking is reproducible across iterations.

        Args:
            candidate_network: Loaded NetworkWrapper to evaluate.
            candidate_label: Identifier for logs (e.g. ``checkpoint_iteration_7``).
            num_games: Total games (split 50/50 between colors).
            depth: HeuristicAgent max search depth.
            seed: Base RNG seed used to derive per-game seeds. KEEP STABLE
                across iterations — the whole point of the anchor is that
                the baseline doesn't drift.
            use_mcts: If True, the candidate plays via pure-neural MCTS
                instead of raw policy-head argmax. Tests how the model
                actually plays in deployment; raw-policy mode is kept as a
                diagnostic for the policy head in isolation.
            mcts_simulations: Per-move sim budget when ``use_mcts=True``.
                Subtree reuse is ON within a game; root Dirichlet noise is
                disabled so the eval is deterministic across iterations.
            heuristic_time_limit_seconds: Per-move wall-clock cap on the
                HeuristicAgent's alpha-beta search. 0.0 (default) is "no
                limit" — preserves existing training-loop determinism.
                Set >0 (e.g. 30.0) for offline eval at depth=3, where the
                pathological alpha-beta blow-up on certain network-produced
                positions (see WARMSTART_PHASE_LOG.md §4b/§5b) would
                otherwise hang the eval indefinitely. With iterative
                deepening on, the agent gracefully reports the deepest
                COMPLETED depth's best move when the budget is hit, so this
                is a liveness fix that costs only the depth on positions
                where depth=3 wasn't reachable anyway.
            candidate_temperature: Move-selection temperature for the
                candidate (T4.9). Default 0.5 — argmax (0.0) silently
                produces the deterministic-side artifact where every game
                on a side replays the same line, so the win rate measures
                side-coverage rather than skill. 0.5 is enough variance to
                surface real strength while still being heavily skewed
                toward the network's preferred move. Pass 0.0 only when
                you specifically want to diagnose deterministic behavior.
            metrics_logger: Optional ``MetricsLogger`` (T5.4). When
                provided, deterministic-collapse alerts at the end of the
                eval are routed through ``log_event`` and ``log_scalar`` so
                they surface in the per-iteration metrics JSON and the
                experiment tracker — not just the python logger.
            iteration: Training iteration this eval belongs to. Used to
                annotate metrics_logger events. Defaults to ``None``
                (events still emit, just without iteration context).

        Returns:
            Dict with keys ``games_played``, ``candidate_wins``,
            ``anchor_wins``, ``draws``, ``win_rate`` (candidate wins /
            games_played; draws count as 0 in numerator), and ``mode``
            (``'raw_policy'`` or ``'mcts'``).

            On failure (e.g. HeuristicAgent construction blows up), returns
            a dict with ``games_played=0`` and a ``skipped_reason`` key —
            callers should treat that as "skip, don't fail the iteration."
        """
        # Import here so a broken HeuristicAgent / heuristics subtree can't
        # crash the tournament module at import time.
        try:
            from ..agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig
            from ..heuristics import YinshHeuristics
        except Exception as e:  # pragma: no cover - defensive
            self.logger.warning(f"Anchor eval skipped — could not import HeuristicAgent: {e}")
            return {
                'games_played': 0, 'candidate_wins': 0, 'anchor_wins': 0,
                'draws': 0, 'win_rate': 0.0, 'skipped_reason': f'import: {e}',
            }

        try:
            # Anchor opponent uses the FAST heuristic (no forced-sequence
            # detection). The anchor agent does its own alpha-beta search on
            # top of the heuristic, so the anchor's per-move strength comes
            # from search depth, not from the heuristic's forced-sequence
            # mini-search. With detection ON, each anchor move was ~9-10s
            # (60 child positions × ~159ms eval). Disabling brings each anchor
            # move to ~30-40ms and each game from ~7 min to a few seconds.
            # The anchor stays the same across iterations and configs, so
            # the relative-comparison signal of the sweep is preserved.
            fast_evaluator = YinshHeuristics(
                enable_forced_sequence_detection=False,
            )
            anchor_agent = HeuristicAgent(
                config=HeuristicAgentConfig(
                    max_depth=depth,
                    min_depth=min(depth, 1),
                    use_iterative_deepening=True,
                    # Default 0.0 = no wall-clock limit (determinism wins for
                    # training-loop anchor eval). Offline depth=3 eval should
                    # pass a positive value to prevent alpha-beta hangs on
                    # pathological positions; iterative deepening guarantees
                    # the deepest-completed-depth result is returned.
                    time_limit_seconds=float(heuristic_time_limit_seconds),
                    random_tiebreak=False,   # deterministic across iterations
                    random_seed=seed,
                    use_transposition_table=True,
                    zobrist_seed=f"anchor-seed-{seed}",
                ),
                evaluator=fast_evaluator,
            )
        except Exception as e:
            self.logger.warning(f"Anchor eval skipped — HeuristicAgent construction failed: {e}")
            return {
                'games_played': 0, 'candidate_wins': 0, 'anchor_wins': 0,
                'draws': 0, 'win_rate': 0.0, 'skipped_reason': f'construct: {e}',
            }

        # Build a per-game MCTS factory when use_mcts is True. We rebuild
        # per game (not per move) so subtree reuse benefits within a game
        # without leaking visits across games. Pure-neural mode + zero
        # Dirichlet noise + deterministic temp ensures the eval is
        # reproducible game-over-game.
        mcts_factory = None
        if use_mcts:
            try:
                from ..training.self_play import MCTS as _AnchorMCTS

                def _build_anchor_mcts():
                    return _AnchorMCTS(
                        network=candidate_network,
                        evaluation_mode="pure_neural",
                        heuristic_evaluator=None,
                        num_simulations=mcts_simulations,
                        late_simulations=mcts_simulations,
                        simulation_switch_ply=10_000,
                        enable_subtree_reuse=True,
                        epsilon_mix_start=0.0,
                        epsilon_mix_end=0.0,
                        epsilon_mix_taper_moves=0,
                        initial_temp=1.0,
                        final_temp=1.0,
                        annealing_steps=1,
                    )
                mcts_factory = _build_anchor_mcts
            except Exception as e:
                self.logger.warning(
                    f"Anchor eval (MCTS): failed to build MCTS, falling back to raw policy: {e}"
                )
                use_mcts = False

        mode_label = 'mcts' if use_mcts else 'raw_policy'
        anchor_label = f"anchor_heuristic_d{depth}"
        half = num_games // 2
        # Fixed pairing sequence: first `half` games candidate plays White,
        # remaining games candidate plays Black. Same order every iteration.
        color_order = ['white'] * half + ['black'] * (num_games - half)

        candidate_wins = 0
        anchor_wins = 0
        draws = 0
        games_played = 0
        total_moves = 0
        # Per-side bookkeeping so we can detect the deterministic-side artifact:
        # when both candidate and heuristic play argmax, every game on a given
        # side replays the same line, so the eval splits {0/half, half/half}
        # by side-coverage rather than by skill. We surface that as a warning
        # so callers don't read a 50% win rate as "tied" when it's really
        # "wins all white, loses all black, deterministically."
        per_side_stats = {
            'white': {'cand_wins': 0, 'games': 0, 'move_counts': []},
            'black': {'cand_wins': 0, 'games': 0, 'move_counts': []},
        }

        # Pre-allocate a reusable input tensor for the candidate network.
        cand_input_tensor = candidate_network._acquire_input_tensor(batch_size=1)

        try:
            for game_num, cand_color in enumerate(color_order):
                # Per-game progress log — important for multi-hour evals
                # where the user otherwise has no idea what's happening.
                game_t0 = time.time()
                self.logger.info(
                    f"[anchor {mode_label}] game {game_num + 1}/{num_games} "
                    f"start (cand={cand_color} vs {anchor_label})"
                )
                # Deterministic per-game seed, stable across iterations, so
                # the SAME pairing sequence is replayed every time.
                rng_snapshot = (
                    torch.get_rng_state(),
                    np.random.get_state(),
                    random.getstate(),
                )
                game_seed = derive_match_seed(
                    seed, candidate_label, anchor_label, game_num
                )
                torch.manual_seed(game_seed)
                np.random.seed(game_seed)
                random.seed(game_seed)
                # Reset anchor's internal RNG each game so identical
                # positions produce identical moves across iterations.
                anchor_agent._rng = random.Random(seed)
                anchor_agent.clear_transposition_table()

                # Fresh MCTS per game so subtree reuse stays in-game only.
                game_mcts = mcts_factory() if mcts_factory is not None else None

                game_state = GameState()
                move_count = 0

                # Map color → side
                candidate_is_white = (cand_color == 'white')

                # B2 telemetry buffer: per-move (root_value, terminal_outcome)
                # pairs collected during the MCTS path; resolved to terminal
                # outcome after the game ends and pushed to metrics_logger.
                game_root_values: List[float] = []

                try:
                    while not game_state.is_terminal() and move_count < max_moves_per_game:
                        valid_moves = game_state.get_valid_moves()
                        if not valid_moves:
                            break

                        cand_to_move = (
                            (game_state.current_player == Player.WHITE and candidate_is_white)
                            or (game_state.current_player == Player.BLACK and not candidate_is_white)
                        )

                        if cand_to_move:
                            if game_mcts is not None:
                                # MCTS path: visit-distribution → greedy argmax
                                # over valid moves. select_move with temp=0
                                # picks the highest-prob valid move. Use the
                                # batched search so each move evaluates leaves
                                # in NN batches — single-leaf search at 64+
                                # sims is ~10× slower on MPS / CUDA.
                                visit_probs = game_mcts.search_batch(
                                    game_state, move_count, batch_size=32
                                )
                                # B2 (W1b): capture root value at this position
                                # in side-to-move POV. The MCTS post-Wave-2
                                # negates `root.value()` at assignment so
                                # `last_root_value` agrees with the
                                # AlphaZero "side-to-move POV" contract
                                # (positive = current player winning). Since
                                # `cand_to_move` is True here, that's the
                                # candidate's POV — directly comparable to
                                # `outcome_in_pov` below without a sign flip.
                                rv = getattr(game_mcts, 'last_root_value', None)
                                if rv is not None and np.isfinite(rv):
                                    game_root_values.append(float(rv))
                                visit_probs_t = torch.from_numpy(np.asarray(visit_probs)).to(
                                    candidate_network.device
                                )
                                selected_move = candidate_network.select_move(
                                    visit_probs_t, valid_moves, temperature=candidate_temperature
                                )
                                del visit_probs_t
                            else:
                                state_array = candidate_network.state_encoder.encode_state(game_state)
                                cand_input_tensor.copy_(
                                    torch.from_numpy(np.array(state_array)).unsqueeze(0)
                                )
                                move_probs, _ = candidate_network.predict(cand_input_tensor)
                                # candidate_temperature=0 (default) → argmax (deterministic given seeds).
                                # >0 → sample from softmax(logits / temperature) over valid moves.
                                selected_move = candidate_network.select_move(
                                    move_probs, valid_moves, temperature=candidate_temperature
                                )
                                del move_probs
                        else:
                            selected_move = anchor_agent.select_move(game_state)

                        if selected_move is None:
                            break
                        success = game_state.make_move(selected_move)
                        if not success:
                            self.logger.warning(
                                f"Anchor eval: invalid move at game {game_num} "
                                f"move {move_count} — aborting game."
                            )
                            break
                        # Keep MCTS's cached root in sync with the game tree
                        # so subtree-reuse benefits accrue across moves.
                        if game_mcts is not None:
                            game_mcts.advance_root(selected_move)
                        move_count += 1
                except Exception as e:
                    self.logger.warning(
                        f"Anchor eval: game {game_num} crashed ({e}); counting as draw."
                    )
                    winner = None
                else:
                    winner = game_state.get_winner()

                if winner == Player.WHITE:
                    if candidate_is_white:
                        candidate_wins += 1
                    else:
                        anchor_wins += 1
                elif winner == Player.BLACK:
                    if candidate_is_white:
                        anchor_wins += 1
                    else:
                        candidate_wins += 1
                else:
                    draws += 1

                games_played += 1
                total_moves += move_count

                # Per-side bookkeeping
                side_key = 'white' if candidate_is_white else 'black'
                per_side_stats[side_key]['games'] += 1
                per_side_stats[side_key]['move_counts'].append(move_count)
                cand_won = (
                    (winner == Player.WHITE and candidate_is_white)
                    or (winner == Player.BLACK and not candidate_is_white)
                )
                if cand_won:
                    per_side_stats[side_key]['cand_wins'] += 1

                # B2 (W1b): resolve per-position root values to terminal
                # outcome in candidate POV and push to metrics_logger. Skip
                # when no metrics_logger or no MCTS root values were captured
                # (raw-policy mode, or candidate never moved). Outcome is
                # +1 candidate-win, -1 candidate-loss, 0 draw — all root
                # values were captured in candidate POV (see the cand_to_move
                # branch above), so no per-color sign flip is needed here.
                if metrics_logger is not None and game_root_values:
                    if cand_won:
                        outcome_in_pov = 1.0
                    elif winner is None:
                        outcome_in_pov = 0.0
                    else:
                        outcome_in_pov = -1.0
                    for rv in game_root_values:
                        metrics_logger.log_eval_value_pair(rv, outcome_in_pov)

                # Per-game completion log — running totals so the user can
                # estimate ETA and track win rate as the eval progresses.
                game_dt = time.time() - game_t0
                if winner == Player.WHITE:
                    outcome = 'cand_W' if candidate_is_white else 'anchor_W'
                elif winner == Player.BLACK:
                    outcome = 'anchor_W' if candidate_is_white else 'cand_W'
                else:
                    outcome = 'draw'
                self.logger.info(
                    f"[anchor {mode_label}] game {game_num + 1}/{num_games} "
                    f"done: {outcome} in {move_count} moves ({game_dt/60:.1f} min). "
                    f"running W/L/D = {candidate_wins}/{anchor_wins}/{draws}"
                )

                del game_state

                # Restore outer RNG state so anchor seeding doesn't leak.
                torch_state, np_state, py_state = rng_snapshot
                torch.set_rng_state(torch_state)
                np.random.set_state(np_state)
                random.setstate(py_state)
        finally:
            candidate_network._release_tensor(cand_input_tensor)

        win_rate = (candidate_wins / games_played) if games_played > 0 else 0.0

        # Per-side aggregation + deterministic-collapse detection.
        side_summary = {}
        deterministic_sides = []
        for side, s in per_side_stats.items():
            if s['games'] == 0:
                continue
            mc = s['move_counts']
            length_min = min(mc)
            length_max = max(mc)
            length_range = length_max - length_min
            side_summary[side] = {
                'games': s['games'],
                'cand_wins': s['cand_wins'],
                'cand_win_rate': s['cand_wins'] / s['games'],
                'avg_game_length': sum(mc) / len(mc),
                'game_length_min': length_min,
                'game_length_max': length_max,
                'game_length_range': length_range,
            }
            # If 2+ games on a side and ALL games have identical length, the
            # candidate's argmax + anchor's argmax replayed the same line every
            # time on that side. That's the deterministic-collapse fingerprint
            # — the win rate doesn't measure skill, it measures side-coverage.
            if s['games'] >= 2 and length_range == 0:
                deterministic_sides.append(side)

        result = {
            'games_played': games_played,
            'candidate_wins': candidate_wins,
            'anchor_wins': anchor_wins,
            'draws': draws,
            'win_rate': win_rate,
            'depth': depth,
            'seed': seed,
            'avg_game_length': (total_moves / games_played) if games_played > 0 else 0.0,
            'mode': mode_label,
            'mcts_simulations': mcts_simulations if use_mcts else 0,
            'per_side': side_summary,
            'deterministic_sides': deterministic_sides,
        }
        self.logger.info(
            f"Anchor eval [{mode_label}]: {candidate_label} vs {anchor_label} → "
            f"W/L/D = {candidate_wins}/{anchor_wins}/{draws} "
            f"({win_rate:.1%} win rate over {games_played} games)"
        )
        # Per-side breakdown — helps catch the deterministic-collapse case
        # where overall win rate hides a 100% / 0% side split.
        for side, ss in side_summary.items():
            self.logger.info(
                f"  side={side}: cand_wins={ss['cand_wins']}/{ss['games']} "
                f"({ss['cand_win_rate']:.1%})  "
                f"game_length min/avg/max = "
                f"{ss['game_length_min']}/{ss['avg_game_length']:.1f}/{ss['game_length_max']}"
            )
        if deterministic_sides:
            self.logger.warning(
                f"⚠️  Deterministic-collapse detected on side(s): "
                f"{', '.join(deterministic_sides)}. Every game on these sides "
                f"had identical move counts — candidate's argmax + anchor's "
                f"argmax replayed the same line. The win rate measures "
                f"side-coverage, not skill. Re-run with candidate_temperature "
                f">= 0.5 or use_mcts=True to break the determinism."
            )
            # T5.4: route the alert into the metrics system so dashboards
            # (and post-run analysis) see the collapse, not just the run log.
            # We emit BOTH a counter scalar (count of collapsing sides) and
            # a structured event with the side list + mode for context.
            if metrics_logger is not None:
                try:
                    metrics_logger.log_event(
                        'deterministic_collapse_alert',
                        severity='warning',
                        iteration=iteration,
                        details={
                            'sides': list(deterministic_sides),
                            'mode': mode_label,
                            'candidate_label': candidate_label,
                            'candidate_temperature': float(candidate_temperature),
                            'use_mcts': bool(use_mcts),
                        },
                    )
                    metrics_logger.log_scalar(
                        'eval/deterministic_collapse_count',
                        float(len(deterministic_sides)),
                        iteration=iteration,
                    )
                except Exception as e:  # never let metrics routing kill eval
                    self.logger.warning(
                        f"Failed to route deterministic_collapse_alert: {e}"
                    )

        # B2 (W1b): aggregate buffered (root_value, terminal_outcome) pairs
        # into eval/value_outcome_correlation for this iteration. Only
        # meaningful in MCTS mode (raw-policy mode never appends pairs).
        # The helper handles too-few-pairs / zero-variance internally; we
        # just call it unconditionally and let it no-op when appropriate.
        if metrics_logger is not None and use_mcts:
            try:
                # W2 B3: name the series after the candidate so each checkpoint
                # in the sliding-window eval lands on its own series. The
                # canonical ``eval/value_outcome_correlation`` is still
                # populated as a mirror for backwards-compat.
                metrics_logger.compute_and_log_value_outcome_correlation(
                    step=iteration,
                    metric_name=f'eval/value_outcome_correlation/{candidate_label}',
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to log value_outcome_correlation: {e}"
                )
        return result

    def run_anchor_eval_batch(
        self,
        models: List[Tuple[str, Path]],
        num_games: int = 40,
        depth: int = 3,
        seed: int = 1337,
        max_moves_per_game: int = 200,
        use_mcts: bool = False,
        mcts_simulations: int = 64,
        candidate_temperature: float = 0.5,
        metrics_logger: Optional["MetricsLogger"] = None,
        iteration: Optional[int] = None,
    ) -> Dict[str, Dict]:
        """Run anchor eval for a list of (label, checkpoint_path) entries.

        Uses lazy model loading so memory stays bounded — only one candidate
        is resident at a time. Missing checkpoints are skipped gracefully.
        Set ``use_mcts=True`` to evaluate with pure-neural MCTS instead of
        raw policy argmax.

        ``candidate_temperature`` defaults to 0.5 (T4.9) — see
        ``run_anchor_eval`` for the rationale. Pass ``metrics_logger`` and
        ``iteration`` to surface deterministic-collapse alerts in dashboards
        (T5.4).
        """
        results: Dict[str, Dict] = {}
        for label, ckpt in models:
            if ckpt is None:
                continue
            ckpt = Path(ckpt)
            if not ckpt.exists():
                self.logger.info(f"Anchor eval: skipping {label} — checkpoint not found at {ckpt}")
                continue
            try:
                net = self._load_model(ckpt)
            except Exception as e:
                self.logger.warning(f"Anchor eval: failed to load {label} from {ckpt}: {e}")
                continue
            try:
                results[label] = self.run_anchor_eval(
                    candidate_network=net,
                    candidate_label=label,
                    num_games=num_games,
                    depth=depth,
                    seed=seed,
                    max_moves_per_game=max_moves_per_game,
                    use_mcts=use_mcts,
                    mcts_simulations=mcts_simulations,
                    candidate_temperature=candidate_temperature,
                    metrics_logger=metrics_logger,
                    iteration=iteration,
                )
            finally:
                if hasattr(net, 'cleanup'):
                    try:
                        net.cleanup()
                    except Exception:
                        pass
                del net
                self._clear_model_memory()
        return results

    def run_dual_anchor_eval(
        self,
        candidate_network: NetworkWrapper,
        candidate_label: str,
        num_games: int = 40,
        depth: int = 3,
        seed: int = 1337,
        max_moves_per_game: int = 200,
        mcts_simulations: int = 64,
        candidate_temperature: float = 0.5,
        run_raw: bool = True,
        run_mcts: bool = True,
        metrics_logger: Optional["MetricsLogger"] = None,
        iteration: Optional[int] = None,
    ) -> Dict:
        """Run anchor eval TWICE — raw-policy AND MCTS — against the same
        anchor opponent (T4.9). The deployed player uses MCTS, but the raw
        policy is the diagnostic we trained the head against, so we want
        both numbers side-by-side every iteration.

        Cost note: doubles the anchor eval wall-clock. The supervisor wires
        this on for canonical recipes (anchor_dual_eval_enabled=true) because
        the cost is bounded (anchor eval is ~minutes, not hours) and the
        observability win is large. For one-off probes you can disable
        either side via ``run_raw`` / ``run_mcts``.

        Returns:
            Dict with keys:
              - ``raw``: full result dict from raw-policy eval (or ``None`` if skipped)
              - ``mcts``: full result dict from MCTS eval (or ``None`` if skipped)
              - ``raw_elo``: Elo delta vs the anchor for raw-policy mode
                (or ``None`` if skipped). Anchor = 0.
              - ``mcts_elo``: Elo delta vs the anchor for MCTS mode
                (or ``None`` if skipped).
              - ``raw_collapse``: list of sides that collapsed in raw-policy
                mode (empty list = none). Always present if raw ran.
              - ``mcts_collapse``: same, for MCTS mode.
        """
        raw_result: Optional[Dict] = None
        mcts_result: Optional[Dict] = None

        if run_raw:
            raw_result = self.run_anchor_eval(
                candidate_network=candidate_network,
                candidate_label=candidate_label,
                num_games=num_games,
                depth=depth,
                seed=seed,
                max_moves_per_game=max_moves_per_game,
                use_mcts=False,
                mcts_simulations=mcts_simulations,
                candidate_temperature=candidate_temperature,
                metrics_logger=metrics_logger,
                iteration=iteration,
            )

        if run_mcts:
            mcts_result = self.run_anchor_eval(
                candidate_network=candidate_network,
                candidate_label=candidate_label,
                num_games=num_games,
                depth=depth,
                seed=seed,
                max_moves_per_game=max_moves_per_game,
                use_mcts=True,
                mcts_simulations=mcts_simulations,
                candidate_temperature=candidate_temperature,
                metrics_logger=metrics_logger,
                iteration=iteration,
            )

        def _elo(res: Optional[Dict]) -> Optional[float]:
            if not res or res.get('games_played', 0) <= 0:
                return None
            return win_rate_to_elo_delta(float(res.get('win_rate', 0.0)))

        def _collapse(res: Optional[Dict]) -> Optional[List[str]]:
            if not res:
                return None
            return list(res.get('deterministic_sides') or [])

        return {
            'raw': raw_result,
            'mcts': mcts_result,
            'raw_elo': _elo(raw_result),
            'mcts_elo': _elo(mcts_result),
            'raw_collapse': _collapse(raw_result),
            'mcts_collapse': _collapse(mcts_result),
        }

    def run_full_round_robin_tournament(self, current_iteration: int):
        """
        Run a round-robin tournament among recent models (sliding window).

        SLIDING WINDOW: Only includes the most recent N models (controlled by
        sliding_window_size). This keeps tournament time constant O(N²) instead
        of growing O(n²) with total iterations.

        With sliding_window_size=5: 10 pairs × 2 directions × 200 games = 4,000 games
        (constant regardless of how many iterations have run)

        Uses LAZY MODEL LOADING to prevent OOM:
          - Models are loaded on-demand for each match pair
          - Only 2 models are in memory at a time
          - Memory is cleared between match pairs

        Uses a robust Glicko-1 rating system:
          - It resets all model ratings to an initial value (1500) and RD (350)
          - It records every match outcome using record_match
          - After all matches, it updates the ratings in a batch via update_ratings
        """
        # Create a unique tournament ID
        tournament_id = f"full_round_robin_{current_iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_tournament_id = tournament_id

        # Gather all model checkpoints up to current_iteration
        all_model_paths = {}  # model_id -> path (dict for lazy loading)
        for i in range(current_iteration + 1):
            # NEW PATH: Looks inside the iteration subdirectory
            ckpt = self.training_dir / f"iteration_{i}" / f"checkpoint_iteration_{i}.pt"

            if ckpt.exists():
                model_id = _canon(ckpt)
                all_model_paths[model_id] = ckpt
            else:
                self.logger.debug(f"Checkpoint not found at expected path: {ckpt}")

        if len(all_model_paths) < 2:
            self.logger.warning(f"Skipping tournament - found only {len(all_model_paths)} models (need >= 2). Checked paths like: {self.training_dir / 'iteration_X' / 'checkpoint_iteration_X.pt'}")
            if all_model_paths:
                 self.logger.warning(f"Found model paths: {list(all_model_paths.values())}")
            return

        # SLIDING WINDOW: Only keep the most recent N models
        all_model_ids = sorted(all_model_paths.keys(), key=lambda x: int(x.split('_')[-1]))

        if self.sliding_window_size > 0 and len(all_model_ids) > self.sliding_window_size:
            # Keep only the most recent models
            selected_ids = all_model_ids[-self.sliding_window_size:]
            model_paths = {mid: all_model_paths[mid] for mid in selected_ids}
            self.logger.info(f"Sliding window: selected {len(model_paths)} most recent models "
                           f"(window={self.sliding_window_size}, total available={len(all_model_ids)})")
        else:
            model_paths = all_model_paths
            if self.sliding_window_size > 0:
                self.logger.info(f"Using all {len(model_paths)} models (below sliding window size of {self.sliding_window_size})")
            else:
                self.logger.info(f"Sliding window disabled, using all {len(model_paths)} models")

        # Reset Glicko ratings for all models by initializing a new tracker
        self.logger.info("Resetting Glicko ratings for tournament models...")
        self.glicko_tracker = GlickoTracker(self.training_dir, initial_rating=1500.0, initial_rd=350.0)
        for model_id in model_paths.keys():
            self.glicko_tracker.add_model(model_id)

        # Keep track of individual match results
        round_robin_results = []

        # Round-robin among selected models - LAZY LOADING
        model_ids = sorted(model_paths.keys(), key=lambda x: int(x.split('_')[-1]))
        num_pairs = len(model_ids) * (len(model_ids) - 1) // 2
        total_games = num_pairs * 2 * self.games_per_match
        self.logger.info(f"Starting round-robin among {len(model_ids)} models: "
                        f"{num_pairs} pairs × 2 directions × {self.games_per_match} games = {total_games} games")

        for i in range(len(model_ids)):
            for j in range(i + 1, len(model_ids)):
                id_i = model_ids[i]
                id_j = model_ids[j]

                # LAZY LOAD: Load only the two models needed for this match pair
                self.logger.debug(f"Loading models for match: {id_i} vs {id_j}")
                model_i = self._load_model(model_paths[id_i])
                model_j = self._load_model(model_paths[id_j])

                self.logger.info(f"Match: {id_i} vs {id_j} (i as White, j as Black)")
                result_white = self._play_match(
                    white_model=model_i,
                    black_model=model_j,
                    white_id=id_i,
                    black_id=id_j
                )
                # Record the match result for the white side
                self.glicko_tracker.record_match(id_i, id_j,
                                                 white_wins=result_white.white_wins,
                                                 black_wins=result_white.black_wins,
                                                 draws=result_white.draws)
                round_robin_results.append(result_white)

                self.logger.info(f"Reverse Match: {id_j} vs {id_i} (j as White, i as Black)")
                result_black = self._play_match(
                    white_model=model_j,
                    black_model=model_i,
                    white_id=id_j,
                    black_id=id_i
                )
                # Record the match result for the reverse match
                self.glicko_tracker.record_match(id_j, id_i,
                                                 white_wins=result_black.white_wins,
                                                 black_wins=result_black.black_wins,
                                                 draws=result_black.draws)
                round_robin_results.append(result_black)

                # MEMORY CLEANUP: Explicitly cleanup models before deletion to prevent OOM
                # The cleanup() method releases tensor pool memory immediately
                if hasattr(model_i, 'cleanup'):
                    model_i.cleanup()
                if hasattr(model_j, 'cleanup'):
                    model_j.cleanup()
                del model_i
                del model_j
                self._clear_model_memory()
                self.logger.debug(f"Cleared models from memory after {id_i} vs {id_j}")

        # After recording all matches, update the Glicko ratings in a batch
        self.glicko_tracker.update_ratings()

        # Build summary stats for each model and include Glicko ratings and RD
        summary_stats = self._aggregate_round_robin_stats(round_robin_results)
        for model_id in summary_stats.keys():
            summary_stats[model_id]['glicko_rating'] = self.glicko_tracker.get_rating(model_id)
            summary_stats[model_id]['rd'] = self.glicko_tracker.get_rd(model_id)
        self.latest_summary_stats = summary_stats.copy()

        # Per-pair Wilson 95% CIs — surfaces statistical noise that the gate alone hides.
        pair_cis = self._compute_pair_cis()

        # Save tournament results
        self.tournament_history[tournament_id] = {
            'iteration': current_iteration,
            'timestamp': datetime.now().isoformat(),
            'round_robin_results': [vars(r) for r in round_robin_results],
            'stats': summary_stats,
            'pair_cis': pair_cis,
        }
        self._save_tournament_history()

        # Log final summary
        self.logger.info(f"\n{'=' * 20} Round-Robin Summary {'=' * 20}")
        for model_id in sorted(summary_stats.keys(), key=lambda m: int(m.split('_')[-1])):
            st = summary_stats[model_id]
            self.logger.info(
                f"{model_id} | Glicko Rating: {st['glicko_rating']:.1f} (RD: {st['rd']:.1f}) | "
                f"W-L-D: {st['wins']}-{st['losses']}-{st['draws']} "
                f"({st['win_rate'] * 100:.1f}% win) | "
                f"WhiteWinRate: {st['white_win_rate'] * 100:.1f}% | "
                f"BlackWinRate: {st['black_win_rate'] * 100:.1f}%"
            )
        if pair_cis:
            self.logger.info(f"\n{'-' * 20} Per-Pair 95% CI {'-' * 20}")
            for entry in pair_cis:
                self.logger.info(
                    f"{entry['model_a']} vs {entry['model_b']}: "
                    f"{entry['wins_a']}/{entry['total']} "
                    f"(p={entry['win_rate_a']:.3f}, SE={entry['se']:.3f}, "
                    f"CI95=[{entry['ci_lower']:.3f}, {entry['ci_upper']:.3f}])"
                )
        self.logger.info("=" * 60)

        # Final memory cleanup
        self._clear_model_memory()
        self.logger.info("Tournament complete (lazy loading kept memory usage low)")

    def _compute_pair_cis(self) -> List[Dict]:
        """Per-pair Wilson 95% CI on model_a's win rate (decisives only — draws excluded
        from numerator and denominator so the proportion is well-defined). Returned as
        a list ordered by iteration ascending. Empty list if no pairs were played."""
        from .stats import wilson_bounds, standard_error

        out = []
        for (model_a, model_b), rec in self._pair_results.items():
            wins_a = rec.get('wins_a', 0)
            wins_b = rec.get('wins_b', 0)
            draws = rec.get('draws', 0)
            decisive = wins_a + wins_b
            total = decisive + draws
            if total == 0:
                continue
            lower, upper = wilson_bounds(wins_a, decisive)
            se = standard_error(wins_a, decisive)
            out.append({
                'model_a': model_a,
                'model_b': model_b,
                'wins_a': wins_a,
                'wins_b': wins_b,
                'draws': draws,
                'total': total,
                'win_rate_a': (wins_a / decisive) if decisive > 0 else 0.0,
                'ci_lower': lower,
                'ci_upper': upper,
                'se': se,
            })

        def _iter_key(model_id: str) -> int:
            try:
                return int(model_id.split('_')[-1])
            except (ValueError, IndexError):
                return -1

        out.sort(key=lambda e: (_iter_key(e['model_a']), _iter_key(e['model_b'])))
        return out

    def _aggregate_round_robin_stats(self, match_results: List[MatchResult]) -> Dict[str, Dict]:
        """
        Aggregate overall and color-specific stats for each model:
          - total wins, draws, losses
          - color-based stats
          - final Glicko rating (retrieved via get_rating)
          - overall and color-specific win rates
        """
        stats = {}
        for mr in match_results:
            if mr.white_model not in stats:
                stats[mr.white_model] = {
                    'wins': 0, 'losses': 0, 'draws': 0,
                    'white_wins': 0, 'white_losses': 0, 'white_draws': 0,
                    'black_wins': 0, 'black_losses': 0, 'black_draws': 0,
                    'games': 0, 'white_games': 0, 'black_games': 0
                }
            if mr.black_model not in stats:
                stats[mr.black_model] = {
                    'wins': 0, 'losses': 0, 'draws': 0,
                    'white_wins': 0, 'white_losses': 0, 'white_draws': 0,
                    'black_wins': 0, 'black_losses': 0, 'black_draws': 0,
                    'games': 0, 'white_games': 0, 'black_games': 0
                }

            total_g = mr.total_games()
            # White side
            w = mr.white_wins
            d = mr.draws
            l = mr.black_wins  # from White's perspective
            stats[mr.white_model]['wins'] += w
            stats[mr.white_model]['draws'] += d
            stats[mr.white_model]['losses'] += l
            stats[mr.white_model]['white_wins'] += w
            stats[mr.white_model]['white_draws'] += d
            stats[mr.white_model]['white_losses'] += l
            stats[mr.white_model]['games'] += total_g
            stats[mr.white_model]['white_games'] += total_g

            # Black side
            w_b = mr.black_wins
            d_b = mr.draws
            l_b = mr.white_wins  # from Black's perspective
            stats[mr.black_model]['wins'] += w_b
            stats[mr.black_model]['draws'] += d_b
            stats[mr.black_model]['losses'] += l_b
            stats[mr.black_model]['black_wins'] += w_b
            stats[mr.black_model]['black_draws'] += d_b
            stats[mr.black_model]['black_losses'] += l_b
            stats[mr.black_model]['games'] += total_g
            stats[mr.black_model]['black_games'] += total_g

        # Calculate win rates and attach final Glicko ratings
        for model_id, data in stats.items():
            g = data['games']
            w = data['wins']
            data['win_rate'] = w / g if g > 0 else 0.0

            wg = data['white_games']
            data['white_win_rate'] = data['white_wins'] / wg if wg > 0 else 0.0

            bg = data['black_games']
            data['black_win_rate'] = data['black_wins'] / bg if bg > 0 else 0.0

            # Use the get_rating method instead of accessing a non-existent 'ratings' attribute
            data['elo'] = self.glicko_tracker.get_rating(model_id)

        return stats

    def get_latest_tournament_summary(self) -> Dict:
        """Get summary of the most recent tournament."""
        if not self.current_tournament_id:
            return None
        return self.tournament_history.get(self.current_tournament_id)

    # def run_tournament(self, current_iteration: int):
    #     """Run tournament for current iteration against previous models."""
    #     tournament_id = f"tournament_{current_iteration}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    #     self.current_tournament_id = tournament_id
    #
    #     # Silently get model paths
    #     model_paths = [
    #         (i, self.training_dir / f"checkpoint_iteration_{i}.pt")
    #         for i in range(current_iteration + 1)
    #         if (self.training_dir / f"checkpoint_iteration_{i}.pt").exists()
    #     ]
    #
    #     if len(model_paths) < 2:
    #         self.logger.info("Skipping tournament - need at least 2 models")
    #         return
    #
    #     # Single log at start
    #     self.logger.info(f"\nStarting tournament for iteration {current_iteration}")
    #     self.logger.info(f"Playing against {len(model_paths) - 1} previous models")
    #
    #     tournament_results = []
    #     current_model = self._load_model(model_paths[-1][1])
    #     current_id = f"iteration_{current_iteration}"
    #
    #     for prev_iter, prev_path in model_paths[:-1]:
    #         prev_id = f"iteration_{prev_iter}"
    #         prev_model = self._load_model(prev_path)
    #
    #         # Log start of match
    #         self.logger.info(f"\nMatching iteration {current_iteration} vs {prev_iter}")
    #
    #         # Play both colors
    #         white_result = self._play_match(current_model, prev_model, current_id, prev_id)
    #         black_result = self._play_match(prev_model, current_model, prev_id, current_id)
    #
    #         tournament_results.extend([white_result, black_result])
    #         self.elo_tracker.update_ratings(white_result)
    #         self.elo_tracker.update_ratings(black_result)
    #
    #         # Log match result
    #         total_wins = white_result.white_wins + black_result.black_wins
    #         total_games = white_result.total_games() + black_result.total_games()
    #         self.logger.info(f"Win rate vs iter {prev_iter}: {total_wins / total_games:.1%}")
    #
    #     # Save results silently
    #     self.tournament_history[tournament_id] = {
    #         'iteration': current_iteration,
    #         'timestamp': datetime.now().isoformat(),
    #         'results': [vars(result) for result in tournament_results]
    #     }
    #     self._save_tournament_history()
    #
    #     # Single summary at end
    #     self.logger.info(f"\n{'=' * 20} Tournament Summary {'=' * 20}")
    #     self.logger.info(f"Current model: iteration {current_iteration}")
    #     self.logger.info("\nELO Ratings:")
    #     for model_id, rating in sorted(self.elo_tracker.ratings.items()):
    #         prefix = "→" if model_id == current_id else " "
    #         self.logger.info(f"{prefix} {model_id}: {rating:.1f}")
    #     self.logger.info("=" * 50)

    def get_model_performance(self, model_id: str) -> Dict:
        """
        Return wins / draws / losses and current rating for `model_id`
        using the summary produced in the most recent tournament.
        Fallback to the old (match‑history) method if we don't have stats yet.
        """
        if model_id in self.latest_summary_stats:
            st = self.latest_summary_stats[model_id]
            total_games = st['games']
            wins        = st['wins']
            draws       = st['draws']
            losses      = st['losses']
            rating      = self.glicko_tracker.get_rating(model_id)
            return {
                'total_games': total_games,
                'wins':        wins,
                'draws':       draws,
                'losses':      losses,
                'win_rate':    wins / total_games if total_games else 0.0,
                'current_rating': rating
            }

        # ---------- fallback (old behaviour) ---------------------------------
        matches = self.glicko_tracker.get_match_history(model_id)
        total_games = sum(m['total_games'] for m in matches)
        wins  = sum(m['white_wins'] if m['white_model']==model_id else m['black_wins']
                    for m in matches)
        draws = sum(m['draws'] for m in matches)
        losses = total_games - wins - draws
        return {
            'total_games': total_games,
            'wins':        wins,
            'draws':       draws,
            'losses':      losses,
            'win_rate':    wins / total_games if total_games else 0.0,
            'current_rating': self.glicko_tracker.get_rating(model_id)
        }

    def discover_models(self) -> None:
        """
        Populate an internal list of (iteration, checkpoint_path).
        Called by TrainingSupervisor before each tournament.
        """
        self._model_paths = []          # clear cache
        for ckpt in sorted(self.training_dir.glob("checkpoint_iteration_*.pt")):
            # Extract iteration number
            try:
                iter_num = int(ckpt.stem.split("_")[-1])
                self._model_paths.append((iter_num, ckpt))
            except ValueError:
                continue    # skip odd files