"""Transposition table for caching search results in game tree search algorithms.

This module provides a high-performance transposition table implementation that uses
Zobrist hashing to cache evaluated positions, improving search performance by avoiding
redundant position evaluations.

The transposition table stores search results (evaluated scores, best moves, search depth)
using Zobrist hash values as keys, enabling O(1) lookup and storage operations.

Key Features:
    - Configurable table size (default: 2^20 entries)
    - Depth-preferred replacement policy
    - Comprehensive metrics tracking
    - Thread-safe concurrent access
    - Optimized memory layout for cache efficiency

Example Usage:
    >>> from yinsh_ml.search.transposition_table import TranspositionTable
    >>> from yinsh_ml.game.zobrist import ZobristHasher
    >>> from yinsh_ml.game.game_state import GameState
    >>>
    >>> # Create transposition table
    >>> tt = TranspositionTable(size_power=20)
    >>> hasher = ZobristHasher(seed="test")
    >>>
    >>> # Store a position evaluation
    >>> state = GameState()
    >>> hash_key = hasher.hash_state(state)
    >>> tt.store(hash_key, depth=3, value=0.5, best_move=None, node_type=NodeType.EXACT)
    >>>
    >>> # Lookup stored position
    >>> entry = tt.lookup(hash_key)
    >>> if entry and entry.depth >= 3:
    ...     print(f"Found cached value: {entry.value}")
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

from yinsh_ml.game.types import Move

# Import NodeType using relative import to avoid triggering __init__.py
from .node_type import NodeType


@dataclass
class TranspositionTableEntry:
    """Entry structure for transposition table.
    
    Attributes:
        hash_key: 64-bit Zobrist hash key
        depth: Search depth at which this entry was created
        value: Evaluated score for this position
        best_move: Best move found at this position (optional)
        node_type: Type of node (EXACT, LOWER_BOUND, UPPER_BOUND)
        age: Entry age for replacement policy (optional)
    """
    hash_key: int
    depth: int
    value: float
    best_move: Optional[Move]
    node_type: NodeType
    age: int = 0


class TranspositionTable:
    """High-performance transposition table for caching search results.
    
    Uses Zobrist hashes as keys to store evaluated positions, enabling
    efficient position caching in game tree search algorithms.
    
    The table uses open addressing with linear probing for collision handling
    and implements a depth-preferred replacement policy.
    
    Args:
        size_power: Table size as power of 2 (default: 20, i.e., 2^20 = 1,048,576 entries)
        enable_metrics: Enable metrics collection (default: True)
    
    Attributes:
        size: Actual table size (2^size_power)
        metrics: Dictionary containing performance metrics
    
    Example:
        >>> tt = TranspositionTable(size_power=20)
        >>> hash_key = 1234567890
        >>> tt.store(hash_key, depth=3, value=0.5, best_move=None, node_type=NodeType.EXACT)
        >>> entry = tt.lookup(hash_key)
        >>> print(entry.value if entry else "Not found")
        0.5
    """
    
    def __init__(self, size_power: int = 20, enable_metrics: bool = True):
        """Initialize transposition table.
        
        Args:
            size_power: Table size as power of 2 (must be between 4 and 30)
            enable_metrics: Enable metrics collection
        
        Raises:
            ValueError: If size_power is out of valid range
        """
        if not (4 <= size_power <= 30):
            raise ValueError(f"size_power must be between 4 and 30, got {size_power}")
        
        self.size_power = size_power
        self.size = 1 << size_power  # 2^size_power
        self.mask = self.size - 1  # Bit mask for fast modulo
        
        # Initialize table as list of None (empty slots)
        self._table: list[Optional[TranspositionTableEntry]] = [None] * self.size
        
        # Metrics tracking
        self._enable_metrics = enable_metrics
        self._metrics_lock = threading.Lock() if enable_metrics else None
        self._hits = 0
        self._misses = 0
        self._collisions = 0
        self._replacements = 0
        self._stores = 0
        
        # Thread safety (will be enhanced in subtask 2.5)
        self._lock = threading.RLock()
    
    def _get_index(self, hash_key: int) -> int:
        """Get table index from hash key.
        
        Args:
            hash_key: Zobrist hash key
        
        Returns:
            Table index (0 to size-1)
        """
        return hash_key & self.mask
    
    def _should_replace(
        self,
        existing: TranspositionTableEntry,
        new_entry: TranspositionTableEntry,
    ) -> bool:
        """Determine if new entry should replace existing entry (depth-preferred policy).
        
        Replacement policy priorities:
        1. Higher depth entries are preferred
        2. If depths are equal, prefer EXACT over bounds
        3. If depths and node types are equal, prefer newer entry (higher age)
        
        Args:
            existing: Existing entry in table
            new_entry: New entry to potentially store
        
        Returns:
            True if new_entry should replace existing, False otherwise
        """
        # Priority 1: Higher depth is always preferred
        if new_entry.depth > existing.depth:
            return True
        if new_entry.depth < existing.depth:
            return False
        
        # Priority 2: Same depth - prefer EXACT over bounds
        # EXACT > LOWER_BOUND > UPPER_BOUND
        node_type_priority = {
            NodeType.EXACT: 3,
            NodeType.LOWER_BOUND: 2,
            NodeType.UPPER_BOUND: 1,
        }
        
        existing_priority = node_type_priority.get(existing.node_type, 0)
        new_priority = node_type_priority.get(new_entry.node_type, 0)
        
        if new_priority > existing_priority:
            return True
        if new_priority < existing_priority:
            return False
        
        # Priority 3: Same depth and node type - prefer newer entry (higher age)
        # Note: Age typically increments, so higher age = newer
        return new_entry.age >= existing.age
    
    def store(
        self,
        hash_key: int,
        depth: int,
        value: float,
        best_move: Optional[Move] = None,
        node_type: NodeType = NodeType.EXACT,
        age: int = 0,
    ) -> None:
        """Store an entry in the transposition table.
        
        Uses linear probing for collision handling. If a collision occurs,
        the replacement policy determines whether to replace the existing entry.
        
        Args:
            hash_key: Zobrist hash key for the position
            depth: Search depth at which this entry was created
            value: Evaluated score
            best_move: Best move found (optional)
            node_type: Node type classification
            age: Entry age for replacement policy (optional)
        """
        with self._lock:
            index = self._get_index(hash_key)
            entry = TranspositionTableEntry(
                hash_key=hash_key,
                depth=depth,
                value=value,
                best_move=best_move,
                node_type=node_type,
                age=age,
            )
            
            # Linear probing to find slot
            original_index = index
            collision_detected = False
            first_collision_index = None
            
            while True:
                existing = self._table[index]
                
                if existing is None:
                    # Empty slot found
                    self._table[index] = entry
                    if self._enable_metrics:
                        with self._metrics_lock:
                            self._stores += 1
                            if collision_detected:
                                self._collisions += 1
                    return
                
                # Check if this is the same position (same hash key)
                if existing.hash_key == hash_key:
                    # Same position - use replacement policy to decide
                    if self._should_replace(existing, entry):
                        self._table[index] = entry
                        if self._enable_metrics:
                            with self._metrics_lock:
                                self._stores += 1
                                if existing.depth != entry.depth or existing.node_type != entry.node_type:
                                    self._replacements += 1
                    else:
                        # Don't replace - keep existing entry
                        if self._enable_metrics:
                            with self._metrics_lock:
                                # Still count as a store attempt
                                pass
                    return
                
                # Collision detected - different hash key at this slot
                if not collision_detected:
                    collision_detected = True
                    first_collision_index = index
                    if self._enable_metrics:
                        with self._metrics_lock:
                            self._collisions += 1
                
                # Move to next slot (linear probing)
                index = (index + 1) & self.mask
                
                # Prevent infinite loop if table is full
                if index == original_index:
                    # Table is full - find best replacement candidate using depth-preferred policy
                    # Check all collision positions and find one to replace
                    if first_collision_index is not None:
                        # Check if we should replace at first collision
                        candidate = self._table[first_collision_index]
                        if candidate is None or self._should_replace(candidate, entry):
                            self._table[first_collision_index] = entry
                            if self._enable_metrics:
                                with self._metrics_lock:
                                    if candidate is not None:
                                        self._replacements += 1
                                    self._stores += 1
                            return
                    
                    # Fallback: replace at original index
                    candidate = self._table[original_index]
                    if candidate is None or self._should_replace(candidate, entry):
                        self._table[original_index] = entry
                        if self._enable_metrics:
                            with self._metrics_lock:
                                if candidate is not None:
                                    self._replacements += 1
                                self._stores += 1
                    return
    
    def lookup(self, hash_key: int) -> Optional[TranspositionTableEntry]:
        """Lookup an entry in the transposition table.
        
        Uses linear probing to handle collisions.
        
        Args:
            hash_key: Zobrist hash key for the position
        
        Returns:
            TranspositionTableEntry if found, None otherwise
        """
        with self._lock:
            index = self._get_index(hash_key)
            original_index = index
            
            while True:
                entry = self._table[index]
                
                if entry is None:
                    # Empty slot - not found
                    if self._enable_metrics:
                        with self._metrics_lock:
                            self._misses += 1
                    return None
                
                if entry.hash_key == hash_key:
                    # Found matching entry
                    if self._enable_metrics:
                        with self._metrics_lock:
                            self._hits += 1
                    return entry
                
                # Continue probing
                index = (index + 1) & self.mask
                
                # Prevent infinite loop
                if index == original_index:
                    # Not found after full table scan
                    if self._enable_metrics:
                        with self._metrics_lock:
                            self._misses += 1
                    return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Dictionary containing metrics:
            - hits: Number of successful lookups
            - misses: Number of failed lookups
            - collisions: Number of hash collisions encountered
            - replacements: Number of entries replaced
            - stores: Total number of store operations
            - utilization: Percentage of table slots used
            - hit_rate: Hit rate percentage (hits / (hits + misses))
        """
        if not self._enable_metrics:
            return {}
        
        with self._metrics_lock:
            total_lookups = self._hits + self._misses
            hit_rate = (self._hits / total_lookups * 100.0) if total_lookups > 0 else 0.0
            
            # Count non-empty slots for utilization
            non_empty = sum(1 for entry in self._table if entry is not None)
            utilization = (non_empty / self.size * 100.0) if self.size > 0 else 0.0
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "collisions": self._collisions,
                "replacements": self._replacements,
                "stores": self._stores,
                "utilization": utilization,
                "hit_rate": hit_rate,
            }
    
    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        if not self._enable_metrics:
            return
        
        with self._metrics_lock:
            self._hits = 0
            self._misses = 0
            self._collisions = 0
            self._replacements = 0
            self._stores = 0
    
    def get_hit_rate(self) -> float:
        """Calculate hit rate percentage.
        
        Returns:
            Hit rate as percentage (0.0 to 100.0)
        """
        if not self._enable_metrics:
            return 0.0
        
        with self._metrics_lock:
            total = self._hits + self._misses
            return (self._hits / total * 100.0) if total > 0 else 0.0
    
    def get_utilization_rate(self) -> float:
        """Calculate table utilization percentage.
        
        Returns:
            Utilization as percentage (0.0 to 100.0)
        """
        with self._lock:
            non_empty = sum(1 for entry in self._table if entry is not None)
            return (non_empty / self.size * 100.0) if self.size > 0 else 0.0
    
    def clear(self) -> None:
        """Clear all entries from the table."""
        with self._lock:
            self._table = [None] * self.size
            if self._enable_metrics:
                self.reset_metrics()

