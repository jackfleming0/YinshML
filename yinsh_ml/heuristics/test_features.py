"""Feature extraction testing and validation framework.

This module provides comprehensive testing utilities for validating
heuristic feature extraction methods. Includes performance benchmarks,
correlation validation, and edge case testing.
"""

import time
from typing import List, Dict, Any, Tuple
from ..game.game_state import GameState
from ..game.constants import Player
from .features import (
    completed_runs_differential,
    potential_runs_count,
    connected_marker_chains,
    ring_positioning,
    ring_spread,
    board_control,
    extract_all_features,
)


class FeatureValidator:
    """Validates feature extraction methods with comprehensive testing."""
    
    # Performance requirements
    MAX_EXTRACTION_TIME_MS = 1.0  # Maximum time per feature extraction in milliseconds
    
    def __init__(self):
        """Initialize the feature validator."""
        self.feature_functions = {
            'completed_runs_differential': completed_runs_differential,
            'potential_runs_count': potential_runs_count,
            'connected_marker_chains': connected_marker_chains,
            'ring_positioning': ring_positioning,
            'ring_spread': ring_spread,
            'board_control': board_control,
        }
    
    def validate_all_features(self, game_state: GameState, player: Player) -> Dict[str, Any]:
        """Validate all feature extraction functions for a game state.
        
        Args:
            game_state: The game state to test
            player: The player to extract features for
            
        Returns:
            Dictionary with validation results including:
            - feature_values: Extracted feature values
            - performance_times: Time taken for each feature (ms)
            - performance_passed: Whether performance requirements met
            - edge_cases_passed: Whether edge cases handled correctly
        """
        results = {
            'feature_values': {},
            'performance_times': {},
            'performance_passed': True,
            'edge_cases_passed': True,
            'errors': []
        }
        
        # Extract all features and measure performance
        for feature_name, feature_func in self.feature_functions.items():
            try:
                start_time = time.perf_counter()
                value = feature_func(game_state, player)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                
                results['feature_values'][feature_name] = value
                results['performance_times'][feature_name] = elapsed_ms
                
                if elapsed_ms > self.MAX_EXTRACTION_TIME_MS:
                    results['performance_passed'] = False
                    results['errors'].append(
                        f"{feature_name} exceeded time limit: {elapsed_ms:.3f}ms > {self.MAX_EXTRACTION_TIME_MS}ms"
                    )
            except Exception as e:
                results['edge_cases_passed'] = False
                results['errors'].append(f"{feature_name} raised exception: {str(e)}")
        
        return results
    
    def benchmark_features(
        self,
        game_states: List[GameState],
        player: Player,
        iterations: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Benchmark feature extraction performance across multiple positions.
        
        Args:
            game_states: List of game states to test
            player: The player to extract features for
            iterations: Number of iterations per feature (default: 100)
            
        Returns:
            Dictionary with benchmark statistics for each feature:
            - mean_time_ms: Average extraction time
            - min_time_ms: Minimum extraction time
            - max_time_ms: Maximum extraction time
            - std_time_ms: Standard deviation of extraction times
        """
        benchmark_results = {}
        
        for feature_name, feature_func in self.feature_functions.items():
            times = []
            
            # Run benchmark for this feature
            for _ in range(iterations):
                for game_state in game_states:
                    start_time = time.perf_counter()
                    try:
                        feature_func(game_state, player)
                    except Exception:
                        pass  # Skip errors during benchmarking
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    times.append(elapsed_ms)
            
            if times:
                benchmark_results[feature_name] = {
                    'mean_time_ms': sum(times) / len(times),
                    'min_time_ms': min(times),
                    'max_time_ms': max(times),
                    'std_time_ms': sum((t - sum(times) / len(times))**2 for t in times) / len(times)**0.5 if len(times) > 1 else 0.0,
                    'samples': len(times),
                }
        
        return benchmark_results
    
    def test_edge_cases(self) -> Dict[str, bool]:
        """Test feature extraction with edge case scenarios.
        
        Tests include:
        - Empty board
        - Full board (if possible)
        - Single marker/ring scenarios
        - Invalid inputs
        
        Returns:
            Dictionary mapping test case names to pass/fail status
        """
        test_results = {}
        
        # Test 1: Empty board
        try:
            empty_state = GameState()
            features = extract_all_features(empty_state, Player.WHITE)
            test_results['empty_board'] = all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in features.values()
            )
        except Exception as e:
            test_results['empty_board'] = False
            test_results['empty_board_error'] = str(e)
        
        # Test 2: Single ring placement
        try:
            single_ring_state = GameState()
            from ..game.types import Move, MoveType
            from ..game.constants import Position
            
            # Place one ring
            move = Move(
                type=MoveType.PLACE_RING,
                player=Player.WHITE,
                source=Position('E5')
            )
            single_ring_state.make_move(move)
            features = extract_all_features(single_ring_state, Player.WHITE)
            test_results['single_ring'] = all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in features.values()
            )
        except Exception as e:
            test_results['single_ring'] = False
            test_results['single_ring_error'] = str(e)
        
        # Test 3: Invalid inputs
        try:
            # Test with None as game_state (should raise TypeError)
            try:
                extract_all_features(None, Player.WHITE)  # type: ignore
                test_results['invalid_game_state'] = False
            except TypeError:
                test_results['invalid_game_state'] = True
            except Exception:
                test_results['invalid_game_state'] = False
        except Exception as e:
            test_results['invalid_game_state'] = False
            test_results['invalid_game_state_error'] = str(e)
        
        return test_results
    
    def validate_differential_calculations(
        self,
        game_state: GameState
    ) -> Dict[str, bool]:
        """Validate that all features return differential values.
        
        Checks that features return my_value - opponent_value correctly
        by verifying that swapping players reverses the sign.
        
        Args:
            game_state: The game state to test
            
        Returns:
            Dictionary mapping feature names to validation status
        """
        validation_results = {}
        
        white_features = extract_all_features(game_state, Player.WHITE)
        black_features = extract_all_features(game_state, Player.BLACK)
        
        for feature_name in white_features.keys():
            white_value = white_features[feature_name]
            black_value = black_features[feature_name]
            
            # For differential features, white_value should be approximately -black_value
            # (allowing for floating point precision)
            if isinstance(white_value, (int, float)) and isinstance(black_value, (int, float)):
                expected_diff = abs(white_value + black_value)
                validation_results[feature_name] = expected_diff < 1e-6  # Floating point tolerance
            else:
                validation_results[feature_name] = False
        
        return validation_results
    
    def generate_test_report(
        self,
        game_states: List[GameState],
        player: Player = Player.WHITE
    ) -> str:
        """Generate a comprehensive test report.
        
        Args:
            game_states: List of game states to test
            player: The player to test features for
            
        Returns:
            Formatted test report string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("HEURISTIC FEATURES TEST REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Edge case tests
        report_lines.append("EDGE CASE TESTS:")
        report_lines.append("-" * 60)
        edge_results = self.test_edge_cases()
        for test_name, passed in edge_results.items():
            if 'error' not in test_name:
                status = "PASS" if passed else "FAIL"
                report_lines.append(f"  {test_name:30s} {status}")
        report_lines.append("")
        
        # Performance benchmarks
        report_lines.append("PERFORMANCE BENCHMARKS:")
        report_lines.append("-" * 60)
        benchmarks = self.benchmark_features(game_states, player, iterations=10)
        for feature_name, stats in benchmarks.items():
            report_lines.append(f"  {feature_name}:")
            report_lines.append(f"    Mean: {stats['mean_time_ms']:.3f}ms")
            report_lines.append(f"    Min:  {stats['min_time_ms']:.3f}ms")
            report_lines.append(f"    Max:  {stats['max_time_ms']:.3f}ms")
            report_lines.append(f"    Samples: {stats['samples']}")
            if stats['mean_time_ms'] > self.MAX_EXTRACTION_TIME_MS:
                report_lines.append(f"    ⚠️  EXCEEDS LIMIT ({self.MAX_EXTRACTION_TIME_MS}ms)")
        report_lines.append("")
        
        # Differential validation
        if game_states:
            report_lines.append("DIFFERENTIAL VALIDATION:")
            report_lines.append("-" * 60)
            diff_results = self.validate_differential_calculations(game_states[0])
            for feature_name, passed in diff_results.items():
                status = "PASS" if passed else "FAIL"
                report_lines.append(f"  {feature_name:30s} {status}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)

