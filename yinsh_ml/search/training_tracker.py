"""Training progress tracker for adaptive heuristic weight reduction.

This module tracks neural network training progress and calculates improvement
metrics to enable gradual reduction of heuristic influence as the network improves.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np


@dataclass
class PerformanceMetrics:
    """Container for performance metrics at a training iteration."""
    iteration: int
    win_rate: float
    evaluation_accuracy: float
    loss: Optional[float] = None
    timestamp: Optional[float] = None


class TrainingTracker:
    """Tracks training progress and calculates improvement metrics."""
    
    def __init__(self, window_size: int = 10):
        """
        Initialize training tracker.
        
        Args:
            window_size: Number of recent iterations to consider for improvement calculation
        """
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)
        self.logger = logging.getLogger("TrainingTracker")
        
        # Baseline metrics (initial performance)
        self.baseline_win_rate: Optional[float] = None
        self.baseline_eval_accuracy: Optional[float] = None
    
    def record_iteration(self,
                        iteration: int,
                        win_rate: float,
                        evaluation_accuracy: float,
                        loss: Optional[float] = None):
        """
        Record performance metrics for a training iteration.
        
        Args:
            iteration: Training iteration number
            win_rate: Win rate against baseline (0.0 to 1.0)
            evaluation_accuracy: Evaluation accuracy metric (0.0 to 1.0)
            loss: Optional training loss value
        """
        metrics = PerformanceMetrics(
            iteration=iteration,
            win_rate=win_rate,
            evaluation_accuracy=evaluation_accuracy,
            loss=loss
        )
        
        self.metrics_history.append(metrics)
        
        # Set baseline on first record
        if self.baseline_win_rate is None:
            self.baseline_win_rate = win_rate
            self.baseline_eval_accuracy = evaluation_accuracy
            self.logger.info(f"Baseline metrics set: win_rate={win_rate:.3f}, "
                           f"eval_accuracy={evaluation_accuracy:.3f}")
    
    def get_improvement_factor(self) -> float:
        """
        Calculate improvement factor based on recent performance.
        
        Returns:
            Improvement factor between 0.0 (no improvement) and 1.0 (maximum improvement)
        """
        if len(self.metrics_history) < 2:
            return 0.0
        
        if self.baseline_win_rate is None:
            return 0.0
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)
        
        # Calculate average recent performance
        recent_win_rate = np.mean([m.win_rate for m in recent_metrics])
        recent_eval_accuracy = np.mean([m.evaluation_accuracy for m in recent_metrics])
        
        # Calculate improvement from baseline
        win_rate_improvement = max(0.0, recent_win_rate - self.baseline_win_rate)
        eval_accuracy_improvement = max(0.0, recent_eval_accuracy - self.baseline_eval_accuracy)
        
        # Normalize improvements (assuming max possible improvement is 1.0 - baseline)
        max_win_rate_improvement = 1.0 - self.baseline_win_rate
        max_eval_accuracy_improvement = 1.0 - self.baseline_eval_accuracy
        
        normalized_win_rate = (win_rate_improvement / max_win_rate_improvement 
                              if max_win_rate_improvement > 0 else 0.0)
        normalized_eval_accuracy = (eval_accuracy_improvement / max_eval_accuracy_improvement
                                   if max_eval_accuracy_improvement > 0 else 0.0)
        
        # Combined improvement factor (weighted average)
        improvement_factor = 0.6 * normalized_win_rate + 0.4 * normalized_eval_accuracy
        
        # Clamp to [0, 1]
        improvement_factor = max(0.0, min(1.0, improvement_factor))
        
        return improvement_factor
    
    def get_recent_performance(self) -> Dict[str, float]:
        """
        Get recent performance metrics.
        
        Returns:
            Dictionary with recent performance statistics
        """
        if len(self.metrics_history) == 0:
            return {
                'win_rate': 0.0,
                'evaluation_accuracy': 0.0,
                'iterations_tracked': 0
            }
        
        recent_metrics = list(self.metrics_history)
        return {
            'win_rate': np.mean([m.win_rate for m in recent_metrics]),
            'evaluation_accuracy': np.mean([m.evaluation_accuracy for m in recent_metrics]),
            'iterations_tracked': len(recent_metrics),
            'latest_iteration': recent_metrics[-1].iteration
        }
    
    def reset_baseline(self):
        """Reset baseline metrics to current performance."""
        if len(self.metrics_history) > 0:
            recent_metrics = list(self.metrics_history)
            self.baseline_win_rate = np.mean([m.win_rate for m in recent_metrics])
            self.baseline_eval_accuracy = np.mean([m.evaluation_accuracy for m in recent_metrics])
            self.logger.info(f"Baseline reset: win_rate={self.baseline_win_rate:.3f}, "
                           f"eval_accuracy={self.baseline_eval_accuracy:.3f}")
    
    def clear_history(self):
        """Clear all recorded metrics."""
        self.metrics_history.clear()
        self.baseline_win_rate = None
        self.baseline_eval_accuracy = None
        self.logger.info("Training history cleared")

