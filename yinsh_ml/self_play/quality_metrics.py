"""Game quality metrics for evaluating self-play training data."""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from collections import Counter, defaultdict

from ..game.game_state import GameState
from ..game.types import Move
from .game_recorder import GameRecord, GameTurn

logger = logging.getLogger(__name__)


@dataclass
class GameQualityMetrics:
    """Metrics for evaluating game quality."""
    game_length: int  # Number of turns
    move_diversity: float  # Unique moves / total moves (0-1)
    positional_entropy: float  # Board position variety measure
    score_differential: float  # Final score difference
    strategic_coherence: float  # Move consistency measure (0-1)
    tactical_patterns: int  # Number of detected tactical sequences
    # Additional metrics
    average_move_time: float = 0.0  # Average time per move
    position_repetition_rate: float = 0.0  # Rate of repeated positions
    move_type_distribution: Dict[str, int] = field(default_factory=dict)  # Move type counts
    phase_distribution: Dict[str, int] = field(default_factory=dict)  # Phase duration


@dataclass
class ComparisonReport:
    """Report comparing two datasets."""
    baseline_avg_length: float
    seeded_avg_length: float
    length_improvement: float  # Percentage improvement
    
    baseline_avg_diversity: float
    seeded_avg_diversity: float
    diversity_improvement: float
    
    baseline_avg_coherence: float
    seeded_avg_coherence: float
    coherence_improvement: float
    
    baseline_avg_score_diff: float
    seeded_avg_score_diff: float
    
    baseline_tactical_patterns: float
    seeded_tactical_patterns: float
    tactical_improvement: float
    
    overall_quality_score: float  # Composite quality score (0-1)


class QualityAnalyzer:
    """Analyzer for computing game quality metrics."""
    
    def __init__(self):
        """Initialize the quality analyzer."""
        self.logger = logging.getLogger(__name__)
    
    def analyze_game(self, game_record: GameRecord) -> GameQualityMetrics:
        """Compute quality metrics for a single game.
        
        Args:
            game_record: Game record to analyze
            
        Returns:
            Game quality metrics
        """
        if not game_record.turns:
            return GameQualityMetrics(
                game_length=0,
                move_diversity=0.0,
                positional_entropy=0.0,
                score_differential=0.0,
                strategic_coherence=0.0,
                tactical_patterns=0
            )
        
        # Basic metrics
        game_length = game_record.total_turns
        move_diversity = self._compute_move_diversity(game_record)
        positional_entropy = self._compute_positional_entropy(game_record)
        score_differential = abs(game_record.final_score.get("white", 0) - 
                                game_record.final_score.get("black", 0))
        strategic_coherence = self._compute_strategic_coherence(game_record)
        tactical_patterns = self._detect_tactical_patterns(game_record)
        
        # Additional metrics
        average_move_time = game_record.duration / game_length if game_length > 0 else 0.0
        position_repetition_rate = self._compute_position_repetition_rate(game_record)
        move_type_distribution = self._compute_move_type_distribution(game_record)
        phase_distribution = self._compute_phase_distribution(game_record)
        
        return GameQualityMetrics(
            game_length=game_length,
            move_diversity=move_diversity,
            positional_entropy=positional_entropy,
            score_differential=score_differential,
            strategic_coherence=strategic_coherence,
            tactical_patterns=tactical_patterns,
            average_move_time=average_move_time,
            position_repetition_rate=position_repetition_rate,
            move_type_distribution=move_type_distribution,
            phase_distribution=phase_distribution
        )
    
    def _compute_move_diversity(self, game_record: GameRecord) -> float:
        """Calculate move diversity (unique moves / total moves).
        
        Args:
            game_record: Game record
            
        Returns:
            Diversity score (0-1)
        """
        if not game_record.turns:
            return 0.0
        
        # Serialize moves for comparison
        move_strings = []
        for turn in game_record.turns:
            move_str = self._serialize_move_for_comparison(turn.move)
            move_strings.append(move_str)
        
        unique_moves = len(set(move_strings))
        total_moves = len(move_strings)
        
        return unique_moves / total_moves if total_moves > 0 else 0.0
    
    def _serialize_move_for_comparison(self, move_dict: Dict[str, Any]) -> str:
        """Serialize a move dictionary to a comparable string.
        
        Args:
            move_dict: Move dictionary
            
        Returns:
            Serialized move string
        """
        parts = [move_dict.get("type", "unknown")]
        if "source" in move_dict:
            parts.append(str(move_dict["source"]))
        if "destination" in move_dict:
            parts.append(str(move_dict["destination"]))
        if "markers" in move_dict:
            markers = sorted(move_dict["markers"]) if isinstance(move_dict["markers"], list) else []
            parts.append(",".join(str(m) for m in markers))
        return "|".join(parts)
    
    def _compute_positional_entropy(self, game_record: GameRecord) -> float:
        """Calculate positional entropy (variety of board positions).
        
        Args:
            game_record: Game record
            
        Returns:
            Entropy score (higher = more variety)
        """
        if not game_record.turns:
            return 0.0
        
        # Extract position features from each turn
        positions = []
        for turn in game_record.turns:
            # Use move source/destination as position indicators
            move = turn.move
            if "source" in move:
                positions.append(move["source"])
            if "destination" in move:
                positions.append(move["destination"])
        
        if not positions:
            return 0.0
        
        # Calculate Shannon entropy
        position_counts = Counter(positions)
        total = len(positions)
        
        entropy = 0.0
        for count in position_counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _compute_strategic_coherence(self, game_record: GameRecord) -> float:
        """Calculate strategic coherence (consistency of move patterns).
        
        Args:
            game_record: Game record
            
        Returns:
            Coherence score (0-1)
        """
        if len(game_record.turns) < 2:
            return 1.0
        
        # Analyze move sequences for patterns
        move_types = [turn.move.get("type", "unknown") for turn in game_record.turns]
        
        # Count transitions between move types
        transitions = defaultdict(int)
        for i in range(len(move_types) - 1):
            transition = (move_types[i], move_types[i + 1])
            transitions[transition] += 1
        
        # Calculate coherence as consistency of transitions
        total_transitions = len(move_types) - 1
        if total_transitions == 0:
            return 1.0
        
        # Measure how concentrated transitions are (higher = more coherent)
        max_transition_count = max(transitions.values()) if transitions else 0
        coherence = max_transition_count / total_transitions if total_transitions > 0 else 0.0
        
        return coherence
    
    def _detect_tactical_patterns(self, game_record: GameRecord) -> int:
        """Detect tactical patterns in the game.
        
        Args:
            game_record: Game record
            
        Returns:
            Number of detected tactical sequences
        """
        if len(game_record.turns) < 3:
            return 0
        
        patterns = 0
        
        # Pattern 1: Consecutive marker removals (tactical sequence)
        consecutive_removals = 0
        for turn in game_record.turns:
            if turn.move.get("type") == "REMOVE_MARKERS":
                consecutive_removals += 1
                if consecutive_removals >= 2:
                    patterns += 1
            else:
                consecutive_removals = 0
        
        # Pattern 2: Ring placement followed by immediate movement (setup)
        for i in range(len(game_record.turns) - 1):
            if (game_record.turns[i].move.get("type") == "PLACE_RING" and
                game_record.turns[i + 1].move.get("type") == "MOVE_RING"):
                patterns += 1
        
        # Pattern 3: Multiple markers removed in single turn (tactical)
        for turn in game_record.turns:
            if turn.move.get("type") == "REMOVE_MARKERS":
                markers = turn.move.get("markers", [])
                if len(markers) >= 3:  # Removing 3+ markers is tactical
                    patterns += 1
        
        return patterns
    
    def _compute_position_repetition_rate(self, game_record: GameRecord) -> float:
        """Calculate rate of repeated board positions.
        
        Args:
            game_record: Game record
            
        Returns:
            Repetition rate (0-1)
        """
        if not game_record.turns:
            return 0.0
        
        # Track positions visited
        positions_visited = []
        for turn in game_record.turns:
            move = turn.move
            if "source" in move:
                positions_visited.append(move["source"])
            if "destination" in move:
                positions_visited.append(move["destination"])
        
        if len(positions_visited) < 2:
            return 0.0
        
        # Count repeated positions
        position_counts = Counter(positions_visited)
        repeated = sum(1 for count in position_counts.values() if count > 1)
        total_unique = len(position_counts)
        
        return repeated / total_unique if total_unique > 0 else 0.0
    
    def _compute_move_type_distribution(self, game_record: GameRecord) -> Dict[str, int]:
        """Compute distribution of move types.
        
        Args:
            game_record: Game record
            
        Returns:
            Dictionary mapping move types to counts
        """
        move_types = [turn.move.get("type", "unknown") for turn in game_record.turns]
        return dict(Counter(move_types))
    
    def _compute_phase_distribution(self, game_record: GameRecord) -> Dict[str, int]:
        """Compute distribution of game phases.
        
        Args:
            game_record: Game record
            
        Returns:
            Dictionary mapping phases to turn counts
        """
        # Note: Phase information may not be directly available in GameRecord
        # This is a placeholder that could be enhanced with actual phase tracking
        return {"main_game": game_record.total_turns}
    
    def compare_datasets(self, baseline_games: List[GameRecord], 
                        seeded_games: List[GameRecord]) -> ComparisonReport:
        """Compare two datasets and generate a comparison report.
        
        Args:
            baseline_games: List of baseline (e.g., random) game records
            seeded_games: List of heuristic-seeded game records
            
        Returns:
            Comparison report
        """
        # Analyze baseline games
        baseline_metrics = [self.analyze_game(game) for game in baseline_games]
        seeded_metrics = [self.analyze_game(game) for game in seeded_games]
        
        # Compute averages
        baseline_avg_length = sum(m.game_length for m in baseline_metrics) / len(baseline_metrics) if baseline_metrics else 0.0
        seeded_avg_length = sum(m.game_length for m in seeded_metrics) / len(seeded_metrics) if seeded_metrics else 0.0
        
        baseline_avg_diversity = sum(m.move_diversity for m in baseline_metrics) / len(baseline_metrics) if baseline_metrics else 0.0
        seeded_avg_diversity = sum(m.move_diversity for m in seeded_metrics) / len(seeded_metrics) if seeded_metrics else 0.0
        
        baseline_avg_coherence = sum(m.strategic_coherence for m in baseline_metrics) / len(baseline_metrics) if baseline_metrics else 0.0
        seeded_avg_coherence = sum(m.strategic_coherence for m in seeded_metrics) / len(seeded_metrics) if seeded_metrics else 0.0
        
        baseline_avg_score_diff = sum(m.score_differential for m in baseline_metrics) / len(baseline_metrics) if baseline_metrics else 0.0
        seeded_avg_score_diff = sum(m.score_differential for m in seeded_metrics) / len(seeded_metrics) if seeded_metrics else 0.0
        
        baseline_tactical_patterns = sum(m.tactical_patterns for m in baseline_metrics) / len(baseline_metrics) if baseline_metrics else 0.0
        seeded_tactical_patterns = sum(m.tactical_patterns for m in seeded_metrics) / len(seeded_metrics) if seeded_metrics else 0.0
        
        # Calculate improvements (percentage)
        length_improvement = ((seeded_avg_length - baseline_avg_length) / baseline_avg_length * 100) if baseline_avg_length > 0 else 0.0
        diversity_improvement = ((seeded_avg_diversity - baseline_avg_diversity) / baseline_avg_diversity * 100) if baseline_avg_diversity > 0 else 0.0
        coherence_improvement = ((seeded_avg_coherence - baseline_avg_coherence) / baseline_avg_coherence * 100) if baseline_avg_coherence > 0 else 0.0
        tactical_improvement = ((seeded_tactical_patterns - baseline_tactical_patterns) / baseline_tactical_patterns * 100) if baseline_tactical_patterns > 0 else 0.0
        
        # Compute overall quality score (normalized composite)
        # Higher is better for: length (up to reasonable max), diversity, coherence, tactical patterns
        # Normalize each metric to [0, 1] and average
        normalized_length = min(1.0, seeded_avg_length / 100.0)  # Assume 100 turns is good
        normalized_diversity = seeded_avg_diversity
        normalized_coherence = seeded_avg_coherence
        normalized_tactical = min(1.0, seeded_tactical_patterns / 10.0)  # Assume 10 patterns is good
        
        overall_quality_score = (
            normalized_length * 0.2 +
            normalized_diversity * 0.3 +
            normalized_coherence * 0.3 +
            normalized_tactical * 0.2
        )
        
        return ComparisonReport(
            baseline_avg_length=baseline_avg_length,
            seeded_avg_length=seeded_avg_length,
            length_improvement=length_improvement,
            baseline_avg_diversity=baseline_avg_diversity,
            seeded_avg_diversity=seeded_avg_diversity,
            diversity_improvement=diversity_improvement,
            baseline_avg_coherence=baseline_avg_coherence,
            seeded_avg_coherence=seeded_avg_coherence,
            coherence_improvement=coherence_improvement,
            baseline_avg_score_diff=baseline_avg_score_diff,
            seeded_avg_score_diff=seeded_avg_score_diff,
            baseline_tactical_patterns=baseline_tactical_patterns,
            seeded_tactical_patterns=seeded_tactical_patterns,
            tactical_improvement=tactical_improvement,
            overall_quality_score=overall_quality_score
        )


