"""Utility functions for feature extraction.

This module provides common utility functions used across feature
extraction methods for consistency and code reuse.
"""

from typing import Union


def calculate_differential(
    my_value: Union[int, float],
    opponent_value: Union[int, float]
) -> Union[int, float]:
    """Calculate differential score between two values.
    
    Utility function for consistent differential calculations across
    all features. Ensures all features follow the pattern:
    feature = my_value - opponent_value
    
    Args:
        my_value: Value for the current player
        opponent_value: Value for the opponent
        
    Returns:
        Differential value (same type as inputs): my_value - opponent_value
        
    Example:
        >>> calculate_differential(5, 3)
        2
        >>> calculate_differential(3.5, 2.1)
        1.4
    """
    return my_value - opponent_value

