# Search System Documentation

Complete documentation for the Yinsh search system components, including Zobrist hashing and Transposition Table implementations.

## Overview

The search system provides high-performance components for game tree search algorithms:

1. **Zobrist Hashing** (Task 1) - Fast, deterministic position hashing
2. **Transposition Table** (Task 2) - Efficient position caching for search results

These components work together to dramatically improve search performance by avoiding redundant position evaluations.

## Architecture

```
┌─────────────────┐
│  Search Algorithm │
│  (negamax/MCTS)  │
└────────┬──────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌──────────────────┐
│ Zobrist Hasher  │  │ Transposition    │
│                 │  │ Table            │
│ - Hash positions│  │ - Cache results  │
│ - Incremental   │  │ - Depth-preferred│
│   updates       │  │   replacement    │
└─────────────────┘  └──────────────────┘
         │                 │
         └────────┬─────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  GameState/Board │
         └─────────────────┘
```

## Quick Start

### Basic Usage

```python
from yinsh_ml.search.transposition_table import TranspositionTable, NodeType
from yinsh_ml.game.zobrist import ZobristHasher
from yinsh_ml.game.game_state import GameState

# Create components
hasher = ZobristHasher(seed="my-game")
tt = TranspositionTable(size_power=20)  # 1M entries

# Hash a game state
state = GameState()
hash_key = hasher.hash_state(state)

# Store evaluation result
tt.store(
    hash_key=hash_key,
    depth=3,
    value=0.5,
    best_move=None,
    node_type=NodeType.EXACT,
)

# Lookup cached result
entry = tt.lookup(hash_key)
if entry and entry.depth >= 3:
    print(f"Cached value: {entry.value}")
```

### Integration with Search Algorithm

The `HeuristicAgent` now includes built-in transposition table support. The integration is enabled by default and can be configured via `HeuristicAgentConfig`:

```python
from yinsh_ml.agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig
from yinsh_ml.game.game_state import GameState

# Create agent with transposition table enabled (default)
config = HeuristicAgentConfig(
    max_depth=3,
    use_transposition_table=True,  # Enabled by default
    transposition_table_size_power=20,  # 1M entries
    zobrist_seed="deterministic-seed",  # Optional: for reproducibility
)
agent = HeuristicAgent(config=config)

# Use the agent - transposition table is used automatically
game_state = GameState()
best_move = agent.select_move(game_state)

# Access transposition table metrics
stats = agent.last_search_stats
if stats.get('transposition_table_metrics'):
    print(f"Hit rate: {stats['transposition_table_metrics']['hit_rate']:.2f}%")
```

**How It Works:**

The `HeuristicAgent._negamax` method automatically:
1. **Hashes the position** using Zobrist hashing
2. **Checks the transposition table** before searching
3. **Uses cached results** when available and sufficient depth
4. **Adjusts alpha-beta window** based on node type (EXACT, LOWER_BOUND, UPPER_BOUND)
5. **Stores search results** after evaluation
6. **Uses best moves** from the table for move ordering

**Disabling Transposition Table:**

```python
# Disable transposition table for testing or comparison
config_no_tt = HeuristicAgentConfig(use_transposition_table=False)
agent_no_tt = HeuristicAgent(config=config_no_tt)
```

**Clearing the Table:**

```python
# Clear transposition table between games
agent.clear_transposition_table()
```

## Component Details

### Zobrist Hashing

Zobrist hashing provides fast, deterministic hash values for game positions. See the [Zobrist Hashing Guide](../docs/zobrist_hashing_guide.md) for complete documentation.

**Key Features:**
- O(1) incremental updates
- Deterministic (same position = same hash)
- Low collision probability
- Thread-safe for read operations

**Basic Usage:**
```python
from yinsh_ml.game.zobrist import ZobristHasher
from yinsh_ml.game.game_state import GameState

hasher = ZobristHasher(seed="deterministic")
state = GameState()

# Full board hashing
hash_value = hasher.hash_state(state)

# Incremental update
new_hash = hasher.place_marker(
    Position.from_string("F6"),
    Player.WHITE,
    hash_value
)
```

### Transposition Table

The transposition table caches search results to avoid redundant evaluations.

**Key Features:**
- Configurable size (default: 2^20 = 1M entries)
- Depth-preferred replacement policy
- Comprehensive metrics tracking
- Thread-safe concurrent access
- Optimized memory layout

**Configuration:**
```python
# Small table for testing
tt = TranspositionTable(size_power=10)  # 1K entries

# Default production size
tt = TranspositionTable(size_power=20)  # 1M entries

# Large table for deep searches
tt = TranspositionTable(size_power=24)  # 16M entries

# Disable metrics for maximum performance
tt = TranspositionTable(size_power=20, enable_metrics=False)
```

**Replacement Policy:**
The table uses a depth-preferred replacement policy:
1. **Higher depth entries** are always preserved over lower depth
2. **EXACT nodes** are preferred over bounds when depths are equal
3. **Newer entries** (higher age) are preferred when depth and type match

**Node Types:**
- `EXACT`: Exact value (alpha <= value <= beta)
- `LOWER_BOUND`: Lower bound (value >= beta, beta cutoff)
- `UPPER_BOUND`: Upper bound (value <= alpha, no improvement)

## Performance Characteristics

### Benchmarks

Based on testing with 1M entry table:

- **Lookup**: < 1μs per operation
- **Store**: < 2μs per operation
- **Memory**: ~40-50 bytes per entry
- **Hit Rate**: 60-80% typical for game tree search

### Metrics

The transposition table tracks comprehensive metrics:

```python
metrics = tt.get_metrics()
print(f"Hit rate: {metrics['hit_rate']:.2f}%")
print(f"Utilization: {metrics['utilization']:.2f}%")
print(f"Collisions: {metrics['collisions']}")
print(f"Replacements: {metrics['replacements']}")
```

**Typical Metrics:**
- Hit rate: 60-80% (higher is better)
- Utilization: 30-70% (depends on table size)
- Collisions: Varies with hash distribution
- Replacements: Indicates table pressure

## Integration Guide

### With HeuristicAgent

The transposition table is **fully integrated** into `HeuristicAgent` and enabled by default. The integration provides:

1. **Automatic position hashing** using Zobrist hashing
2. **Transparent caching** of search results
3. **Automatic move ordering** using best moves from the table
4. **Alpha-beta window adjustment** based on cached node types
5. **Performance metrics** included in search statistics

**Configuration Options:**

```python
from yinsh_ml.agents.heuristic_agent import HeuristicAgentConfig

config = HeuristicAgentConfig(
    use_transposition_table=True,  # Enable/disable TT (default: True)
    transposition_table_size_power=20,  # Table size: 2^20 = 1M entries
    zobrist_seed="my-seed",  # Deterministic hashing (optional)
)
```

**Accessing Metrics:**

```python
agent = HeuristicAgent(config=config)
move = agent.select_move(game_state)

# Metrics are automatically included in search stats
stats = agent.last_search_stats
tt_metrics = stats.get('transposition_table_metrics', {})
print(f"Hit rate: {tt_metrics.get('hit_rate', 0):.2f}%")
print(f"Utilization: {tt_metrics.get('utilization', 0):.2f}%")
```

**Best Practices:**

- Use deterministic `zobrist_seed` for reproducible results
- Clear table between games: `agent.clear_transposition_table()`
- Monitor hit rate to optimize table size
- Larger tables (2^22-2^24) for deeper searches

### With MCTS

For MCTS integration:

```python
from yinsh_ml.search.mcts import MCTS
from yinsh_ml.search.transposition_table import TranspositionTable

class MCTSWithTT(MCTS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tt = TranspositionTable(size_power=20)
        self.hasher = ZobristHasher()
    
    def _evaluate_state(self, state):
        # Check transposition table
        hash_key = self.hasher.hash_state(state)
        entry = self.tt.lookup(hash_key)
        
        if entry:
            return entry.value
        
        # ... evaluate state ...
        
        # Store result
        self.tt.store(hash_key, depth=0, value=result, ...)
        return result
```

## API Reference

### TranspositionTable

#### `__init__(size_power=20, enable_metrics=True)`

Initialize transposition table.

**Parameters:**
- `size_power`: Table size as power of 2 (10-30, default: 20)
- `enable_metrics`: Enable metrics collection (default: True)

#### `store(hash_key, depth, value, best_move=None, node_type=NodeType.EXACT, age=0)`

Store an entry in the transposition table.

**Parameters:**
- `hash_key`: Zobrist hash key (64-bit integer)
- `depth`: Search depth
- `value`: Evaluated score
- `best_move`: Best move found (optional)
- `node_type`: Node type classification
- `age`: Entry age for replacement policy

#### `lookup(hash_key) -> Optional[TranspositionTableEntry]`

Lookup an entry by hash key.

**Returns:** `TranspositionTableEntry` if found, `None` otherwise

#### `get_metrics() -> Dict[str, Any]`

Get current performance metrics.

**Returns:** Dictionary with hits, misses, collisions, utilization, hit_rate

#### `reset_metrics()`

Reset all metrics counters.

#### `get_hit_rate() -> float`

Calculate hit rate percentage.

#### `get_utilization_rate() -> float`

Calculate table utilization percentage.

#### `clear()`

Clear all entries from the table.

### TranspositionTableEntry

Entry structure containing:
- `hash_key`: 64-bit Zobrist hash
- `depth`: Search depth
- `value`: Evaluated score
- `best_move`: Best move (optional)
- `node_type`: Node type (EXACT, LOWER_BOUND, UPPER_BOUND)
- `age`: Entry age

### NodeType

Enumeration of node types:
- `EXACT`: Exact value
- `LOWER_BOUND`: Lower bound
- `UPPER_BOUND`: Upper bound

## Performance Tuning

### Table Size

Choose table size based on available memory and search depth:

- **Small (2^10 - 2^14)**: Testing, shallow searches
- **Medium (2^16 - 2^20)**: Typical game play, moderate depth
- **Large (2^22 - 2^24)**: Deep searches, tournament play

### Memory Considerations

Each entry uses ~40-50 bytes:
- 1M entries ≈ 40-50 MB
- 16M entries ≈ 640-800 MB

### Hit Rate Optimization

To improve hit rate:
1. Increase table size
2. Use depth-preferred replacement (already implemented)
3. Clear table between games
4. Use appropriate hash key generation

## Troubleshooting

### Low Hit Rate

**Symptoms:** Hit rate < 50%

**Solutions:**
- Increase table size
- Verify hash key generation is deterministic
- Check that positions are being stored correctly
- Consider clearing table between unrelated searches

### High Collision Rate

**Symptoms:** Many collisions reported in metrics

**Solutions:**
- Increase table size
- Verify Zobrist hash distribution is uniform
- Check for hash key collisions (different positions, same hash)

### Performance Issues

**Symptoms:** Slow lookup/store operations

**Solutions:**
- Disable metrics (`enable_metrics=False`)
- Reduce table size if memory constrained
- Check for lock contention in multi-threaded scenarios
- Profile to identify bottlenecks

### Memory Usage

**Symptoms:** High memory consumption

**Solutions:**
- Reduce table size
- Clear table periodically
- Use smaller entry structures (already optimized)

## Related Documentation

- [Zobrist Hashing Guide](../docs/zobrist_hashing_guide.md) - Complete Zobrist hashing documentation
- [Heuristic Agent API](../docs/heuristic_agent_api.md) - Agent integration details
- [Game State Documentation](../game/game_state.py) - Game state representation

## Examples

### Complete Search Integration

The `HeuristicAgent` in `yinsh_ml/agents/heuristic_agent.py` demonstrates a complete, production-ready integration:

- **Automatic initialization** of transposition table and Zobrist hasher
- **Seamless integration** with existing negamax search
- **Proper node type handling** (EXACT, LOWER_BOUND, UPPER_BOUND)
- **Best move storage and retrieval** for improved move ordering
- **Comprehensive metrics** tracking and reporting
- **Thread-safe operations** for concurrent access

The implementation follows best practices for transposition table integration in game tree search algorithms.

### Metrics Analysis

```python
# Run search with metrics
tt = TranspositionTable(size_power=20)
# ... perform searches ...

# Analyze performance
metrics = tt.get_metrics()
print(f"Hit rate: {metrics['hit_rate']:.2f}%")
print(f"Table utilization: {metrics['utilization']:.2f}%")

if metrics['hit_rate'] < 50:
    print("Warning: Low hit rate, consider increasing table size")
```

### Batch Evaluation

```python
# Evaluate multiple positions
hasher = ZobristHasher()
tt = TranspositionTable()

positions = [GameState() for _ in range(1000)]

for state in positions:
    hash_key = hasher.hash_state(state)
    entry = tt.lookup(hash_key)
    
    if not entry:
        value = evaluate_position(state)
        tt.store(hash_key, depth=0, value=value, node_type=NodeType.EXACT)
```

## Best Practices

1. **Use deterministic seeds** for Zobrist hasher to ensure reproducibility
2. **Clear table between games** to avoid stale entries
3. **Monitor metrics** to optimize table size and replacement policy
4. **Use appropriate node types** for alpha-beta window adjustments
5. **Store best moves** for improved move ordering
6. **Thread safety**: Table is thread-safe, but consider per-thread tables for maximum performance

## Future Enhancements

Potential improvements:
- Per-bucket locking for better concurrency
- Lock-free reads using atomic operations
- Entry compression for memory efficiency
- Persistent storage for table contents
- Adaptive replacement policies

---

**Last Updated**: See git history for latest changes.

**Related Tasks**: 
- Task 1: Zobrist Hashing Implementation ✅
- Task 2: Transposition Table Implementation ✅
- Task 3: Negamax Integration ✅

