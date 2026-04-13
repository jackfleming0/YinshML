# Heuristic Agent API Documentation

Complete API reference for the heuristic agent system, including the `HeuristicAgent` class, configuration options, and the `YinshHeuristics` evaluator.

## Table of Contents

- [HeuristicAgent](#heuristicagent)
- [HeuristicAgentConfig](#heuristicagentconfig)
- [YinshHeuristics](#yinshheuristics)
- [Usage Examples](#usage-examples)
- [Configuration and Tuning](#configuration-and-tuning)

---

## HeuristicAgent

The `HeuristicAgent` class provides a standalone search agent that uses heuristic-guided search to select moves. It is designed as a strong baseline for testing, benchmarking, and heuristic-guided self-play without requiring neural networks or MCTS.

### Class Definition

```python
from yinsh_ml.agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player

agent = HeuristicAgent(
    config: Optional[HeuristicAgentConfig] = None,
    evaluator: Optional[YinshHeuristics] = None,
    rng: Optional[random.Random] = None,
)
```

### Constructor Parameters

- **config** (`HeuristicAgentConfig`, optional): Configuration options for the agent. Defaults to `HeuristicAgentConfig()` with default values.
- **evaluator** (`YinshHeuristics`, optional): Heuristic evaluator instance. If not provided, a default evaluator is created.
- **rng** (`random.Random`, optional): Random number generator for tiebreaking. If not provided, a new generator is created.

### Methods

#### `select_move(game_state: GameState) -> Move`

Selects the best move for the given game state using heuristic-guided search.

**Parameters:**
- `game_state` (`GameState`): The current game state. Must provide:
  - `get_valid_moves()` method
  - `copy()` method
  - `make_move(move)` method
  - `is_terminal()` method
  - `current_player` attribute

**Returns:**
- `Move`: The selected move

**Raises:**
- `TypeError`: If `game_state` doesn't provide required methods
- `ValueError`: If no legal moves are available

**Example:**
```python
from yinsh_ml.game.game_state import GameState
from yinsh_ml.agents.heuristic_agent import HeuristicAgent

agent = HeuristicAgent()
game_state = GameState()
move = agent.select_move(game_state)
```

### Attributes

#### `last_search_stats: dict`

Dictionary containing statistics from the last move selection:
- `nodes_evaluated`: Number of positions evaluated
- `depth_reached`: Maximum search depth reached
- `move_count`: Number of candidate moves considered
- `timed_out`: Whether search timed out
- `duration`: Total search time in seconds
- `depth_metrics`: Per-depth timing information (if `debug=True`)

**Example:**
```python
move = agent.select_move(game_state)
stats = agent.last_search_stats
print(f"Evaluated {stats['nodes_evaluated']} positions")
print(f"Reached depth {stats['depth_reached']}")
print(f"Search took {stats['duration']:.3f} seconds")
```

---

## HeuristicAgentConfig

Configuration dataclass for `HeuristicAgent` behavior.

### Class Definition

```python
from yinsh_ml.agents.heuristic_agent import HeuristicAgentConfig

config = HeuristicAgentConfig(
    min_depth: int = 1,
    max_depth: int = 3,
    time_limit_seconds: float = 1.0,
    time_buffer_seconds: float = 0.01,
    max_branching_factor: Optional[int] = 24,
    use_iterative_deepening: bool = True,
    score_cap: float = 10_000.0,
    random_tiebreak: bool = True,
    debug: bool = False,
    slow_warning_threshold: float = 0.75,
    random_seed: Optional[int] = None,
)
```

### Configuration Parameters

#### Search Depth

- **min_depth** (`int`, default: `1`): Minimum search depth (plies) that must be fully evaluated before timing out.
- **max_depth** (`int`, default: `3`): Maximum search depth (plies) for negamax search.

#### Time Management

- **time_limit_seconds** (`float`, default: `1.0`): Soft wall-clock budget per move. Set `<=0` for no limit.
- **time_buffer_seconds** (`float`, default: `0.01`): Grace period retained from the time budget to allow orderly shutdown.

#### Search Control

- **max_branching_factor** (`Optional[int]`, default: `24`): Optional cap on candidate moves searched after initial ordering. Set to `None` to search all moves.
- **use_iterative_deepening** (`bool`, default: `True`): Enable iterative-deepening search up to `max_depth`.

#### Score Handling

- **score_cap** (`float`, default: `10_000.0`): Clamp heuristic scores to avoid runaway values.
- **random_tiebreak** (`bool`, default: `True`): Perturb equal scores slightly to reduce deterministic oscillation.

#### Debugging

- **debug** (`bool`, default: `False`): Emit detailed timing information via `last_search_stats`.
- **slow_warning_threshold** (`float`, default: `0.75`): Emit debug warnings when a single depth takes longer than this many seconds.

#### Randomization

- **random_seed** (`Optional[int]`, default: `None`): Optional seed for deterministic fallback behavior.

### Example Configurations

#### Fast Configuration (Quick Moves)
```python
fast_config = HeuristicAgentConfig(
    max_depth=2,
    time_limit_seconds=0.1,
    max_branching_factor=12,
)
```

#### Deep Search Configuration (Strong Play)
```python
strong_config = HeuristicAgentConfig(
    max_depth=5,
    time_limit_seconds=5.0,
    max_branching_factor=None,  # Search all moves
    use_iterative_deepening=True,
)
```

#### Deterministic Configuration (Reproducible)
```python
deterministic_config = HeuristicAgentConfig(
    random_seed=42,
    random_tiebreak=False,
)
```

---

## YinshHeuristics

The `YinshHeuristics` class evaluates Yinsh game positions using feature extraction and phase-aware weighting.

### Class Definition

```python
from yinsh_ml.heuristics import YinshHeuristics
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player

evaluator = YinshHeuristics(
    weights: Optional[Dict[str, Any]] = None,
    phase_config: Optional[PhaseConfig] = None,
    weight_config_file: Optional[str] = None,
)
```

### Constructor Parameters

- **weights** (`Optional[Dict[str, Any]]`): Dictionary containing feature weights for different game phases. If not provided, default weights are used.
- **phase_config** (`Optional[PhaseConfig]`): Configuration for game phase detection. If not provided, default phase configuration is used.
- **weight_config_file** (`Optional[str]`): Path to JSON configuration file containing weights. If provided, weights are loaded from this file.

### Methods

#### `evaluate_position(game_state: GameState, player: Player) -> float`

Evaluate a single game position and return a score.

**Parameters:**
- `game_state` (`GameState`): The current game state to evaluate
- `player` (`Player`): The player whose perspective to evaluate from (`Player.WHITE` or `Player.BLACK`)

**Returns:**
- `float`: Position evaluation score where:
  - Positive values indicate an advantage for the specified player
  - Negative values indicate a disadvantage
  - Zero indicates a roughly equal position

**Raises:**
- `TypeError`: If `game_state` is not a `GameState` instance
- `ValueError`: If `player` is invalid

**Example:**
```python
from yinsh_ml.heuristics import YinshHeuristics
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player

evaluator = YinshHeuristics()
game_state = GameState()
score = evaluator.evaluate_position(game_state, Player.WHITE)
print(f"Position evaluation: {score}")
```

#### `evaluate_batch(game_states: List[GameState], players: List[Player]) -> List[float]`

Evaluate multiple game positions in batch for improved efficiency.

**Parameters:**
- `game_states` (`List[GameState]`): List of game states to evaluate
- `players` (`List[Player]`): List of players (one per game state) whose perspective to evaluate from

**Returns:**
- `List[float]`: List of float scores, one per position, in the same order as input

**Raises:**
- `TypeError`: If inputs are not lists or contain invalid types
- `ValueError`: If lists have mismatched lengths

**Example:**
```python
game_states = [GameState() for _ in range(10)]
players = [Player.WHITE if i % 2 == 0 else Player.BLACK for i in range(10)]
scores = evaluator.evaluate_batch(game_states, players)
```

### Evaluation Features

The heuristic evaluator considers:

1. **Game Phase Detection**: Early/mid/late game phases based on move count
2. **Feature Extraction**:
   - Completed runs differential
   - Potential runs count
   - Connected marker chains
   - Ring positioning
   - Ring spread
   - Board control
3. **Phase-Specific Weighting**: Different feature weights for different game phases
4. **Terminal Position Detection**: Immediate win/loss detection
5. **Tactical Patterns**: Detection of immediate tactical opportunities
6. **Forced Sequences**: Multi-move forced outcomes

---

## Usage Examples

### Basic Usage

```python
from yinsh_ml.agents.heuristic_agent import HeuristicAgent
from yinsh_ml.game.game_state import GameState

# Create agent with default configuration
agent = HeuristicAgent()

# Create game state
game_state = GameState()

# Select move
move = agent.select_move(game_state)
print(f"Selected move: {move}")
```

### Custom Configuration

```python
from yinsh_ml.agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig

# Create custom configuration
config = HeuristicAgentConfig(
    max_depth=4,
    time_limit_seconds=2.0,
    max_branching_factor=30,
    debug=True,
)

# Create agent with custom config
agent = HeuristicAgent(config=config)

# Use agent
move = agent.select_move(game_state)

# Access debug statistics
stats = agent.last_search_stats
print(f"Search statistics: {stats}")
```

### Custom Evaluator

```python
from yinsh_ml.agents.heuristic_agent import HeuristicAgent
from yinsh_ml.heuristics import YinshHeuristics

# Create custom evaluator with loaded weights
evaluator = YinshHeuristics(weight_config_file="path/to/weights.json")

# Create agent with custom evaluator
agent = HeuristicAgent(evaluator=evaluator)

# Use agent
move = agent.select_move(game_state)
```

### Batch Evaluation

```python
from yinsh_ml.heuristics import YinshHeuristics
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player

evaluator = YinshHeuristics()

# Generate multiple positions
positions = [GameState() for _ in range(100)]
players = [Player.WHITE if i % 2 == 0 else Player.BLACK for i in range(100)]

# Evaluate in batch
scores = evaluator.evaluate_batch(positions, players)

# Process results
for i, score in enumerate(scores):
    print(f"Position {i}: {score}")
```

---

## Configuration and Tuning

### Weight Configuration

Heuristic weights can be configured via JSON files. See the configuration system documentation for details on weight file format.

### Performance Tuning

For optimal performance:

1. **Adjust search depth**: Increase `max_depth` for stronger play (at cost of time)
2. **Limit branching**: Set `max_branching_factor` to reduce search space
3. **Time limits**: Set `time_limit_seconds` based on your time constraints
4. **Iterative deepening**: Keep `use_iterative_deepening=True` for best results

### Integration with Self-Play

The heuristic agent can be used as a baseline opponent in self-play:

```python
from yinsh_ml.agents.heuristic_agent import HeuristicAgent
from yinsh_ml.self_play.policies import RandomMovePolicy

# Create heuristic agent
heuristic_agent = HeuristicAgent()

# Use as opponent in self-play
# (See integration guide for full examples)
```

---

## See Also

- [Integration Guide](integration_guide.md) - Step-by-step guide for integrating into AlphaZero pipeline
- [Interface Compatibility](heuristic_integration_compatibility.md) - Interface compatibility analysis
- [Configuration System](../heuristics/config_manager.py) - Weight configuration management

