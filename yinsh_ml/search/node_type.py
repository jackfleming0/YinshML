"""Node type classification for transposition table entries."""

from enum import Enum


class NodeType(Enum):
    """Node type classification for transposition table entries.
    
    EXACT: Node value is exact (alpha <= value <= beta)
    LOWER_BOUND: Node value is a lower bound (value >= beta, beta cutoff)
    UPPER_BOUND: Node value is an upper bound (value <= alpha, no improvement)
    """
    EXACT = "exact"
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"

