# Integration Guide: Heuristic Agent for AlphaZero Training Pipeline

This guide provides step-by-step instructions for integrating the heuristic agent into the AlphaZero self-play training pipeline.

## Table of Contents

- [Overview](#overview)
- [Architecture Overview](#architecture-overview)
- [Integration Methods](#integration-methods)
  - [Method 1: Baseline Opponent](#method-1-baseline-opponent)
  - [Method 2: Heuristic Policy in Self-Play](#method-2-heuristic-policy-in-self-play)
  - [Method 3: Heuristic Evaluation in MCTS](#method-3-heuristic-evaluation-in-mcts)
- [Configuration Examples](#configuration-examples)
- [Performance Considerations](#performance-considerations)
- [Best Practices](#best-practices)
- [Transition to Phase 1](#transition-to-phase-1)

---

## Overview

The heuristic agent can be integrated into the AlphaZero training pipeline in several ways:

1. **As a baseline opponent** - Provides a strong baseline for comparison and evaluation
2. **As a policy in self-play** - Uses heuristic guidance during training game generation
3. **As evaluation guidance in MCTS** - Enhances MCTS search with heuristic evaluation

This guide covers all three integration methods with code examples and configuration templates.

---

## Architecture Overview

The heuristic system consists of:

- **`HeuristicAgent`**: Main search agent using heuristic-guided negamax search
- **`YinshHeuristics`**: Position evaluator using feature extraction and phase-aware weighting
- **`HeuristicPolicy`**: Policy wrapper for self-play integration (already implemented)

The system is designed to be:
- **Fast**: Sub-millisecond evaluation times
- **Compatible**: Direct integration with existing `GameState` interface
- **Configurable**: Tunable weights and search parameters

---

## Integration Methods

### Method 1: Baseline Opponent

Use the heuristic agent as a baseline opponent for evaluation and comparison during training.

#### Basic Integration

```python
from yinsh_ml.agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig
from yinsh_ml.self_play.policies import RandomMovePolicy, PolicyConfig
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player

# Create heuristic agent as baseline
baseline_config = HeuristicAgentConfig(
    max_depth=3,
    time_limit_seconds=1.0,
    max_branching_factor=24,
)
baseline_agent = HeuristicAgent(config=baseline_config)

# Use in evaluation games
def play_evaluation_game(network_policy, baseline_agent):
    """Play a game between network policy and baseline."""
    state = GameState()
    
    while not state.is_terminal():
        if state.current_player == Player.WHITE:
            # Network policy move
            move = network_policy.select_move(state)
        else:
            # Baseline agent move
            move = baseline_agent.select_move(state)
        
        state.make_move(move)
    
    return state.get_winner()
```

#### Integration with Tournament System

```python
from yinsh_ml.agents.tournament import TournamentEvaluator, TournamentConfig
from yinsh_ml.agents.heuristic_agent import HeuristicAgentConfig

# Create tournament evaluator
config = HeuristicAgentConfig()
evaluator = TournamentEvaluator(
    heuristic_agent_factory=None,  # Use default
    heuristic_config=config,
    opponent_factory=None,  # Use random as opponent
)

# Run evaluation tournament
tournament_config = TournamentConfig(
    num_games=100,
    concurrent_workers=4,
    output_path="evaluation_results.json",
)
metrics = evaluator.run_large_scale_tournament(tournament_config)

print(f"Win rate: {metrics.win_rate:.3f}")
print(f"Average move time: {metrics.average_move_time*1000:.3f} ms")
```

---

### Method 2: Heuristic Policy in Self-Play

Use `HeuristicPolicy` (already implemented) as an alternative policy during self-play game generation.

#### Using HeuristicPolicy

```python
from yinsh_ml.self_play.policies import HeuristicPolicy, HeuristicPolicyConfig
from yinsh_ml.self_play.game_runner import SelfPlayRunner, RunnerConfig

# Configure heuristic policy
heuristic_config = HeuristicPolicyConfig(
    search_depth=3,
    time_limit=1.0,
    use_fast_mode=False,  # Use full search
    randomness=0.1,  # 10% exploration
    temperature=0.5,  # Temperature for move selection
)

# Create heuristic policy
heuristic_policy = HeuristicPolicy(config=heuristic_config)

# Use in self-play runner
runner_config = RunnerConfig(
    num_games=100,
    policy=heuristic_policy,  # Use heuristic policy
)
runner = SelfPlayRunner(config=runner_config)
results = runner.run()
```

#### Custom Self-Play Integration

```python
from yinsh_ml.agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.constants import Player

class CustomSelfPlay:
    """Custom self-play with heuristic agent."""
    
    def __init__(self):
        # Create heuristic agent
        agent_config = HeuristicAgentConfig(
            max_depth=3,
            time_limit_seconds=0.5,
        )
        self.heuristic_agent = HeuristicAgent(config=agent_config)
    
    def play_game(self):
        """Play a self-play game using heuristic agent."""
        state = GameState()
        training_data = []
        
        while not state.is_terminal():
            # Use heuristic agent for move selection
            move = self.heuristic_agent.select_move(state)
            
            # Store training data (state, move, etc.)
            training_data.append({
                'state': state.copy(),
                'move': move,
            })
            
            state.make_move(move)
        
        # Determine outcome
        winner = state.get_winner()
        outcome = 1.0 if winner == Player.WHITE else -1.0 if winner == Player.BLACK else 0.0
        
        return training_data, outcome
```

---

### Method 3: Heuristic Evaluation in MCTS

Integrate heuristic evaluation into MCTS search for enhanced position evaluation.

#### MCTS Integration

The heuristic evaluator can be integrated into MCTS for leaf node evaluation:

```python
from yinsh_ml.heuristics import YinshHeuristics
from yinsh_ml.search.mcts import MCTS, MCTSConfig

# Create heuristic evaluator
heuristic_evaluator = YinshHeuristics()

# Configure MCTS with heuristic evaluation
mcts_config = MCTSConfig(
    num_simulations=100,
    evaluation_mode='hybrid',  # Use both neural and heuristic
    heuristic_evaluator=heuristic_evaluator,
    heuristic_weight=0.3,  # 30% heuristic, 70% neural
)

# Create MCTS instance
mcts = MCTS(network=network, config=mcts_config)

# Use in search
policy = mcts.search(game_state, move_number)
```

#### Enhanced MCTS with Heuristics

```python
from yinsh_ml.heuristics import YinshHeuristics
from yinsh_ml.training.enhanced_mcts import EnhancedMCTS

# Create heuristic evaluator
heuristic_evaluator = YinshHeuristics()

# Enhanced MCTS already supports heuristic integration
enhanced_mcts = EnhancedMCTS(
    network=network,
    heuristic_evaluator=heuristic_evaluator,
    use_heuristic_evaluation=True,
    heuristic_score_scale=100.0,  # Normalization scale
)
```

---

## Configuration Examples

### Configuration Template: Baseline Opponent

```python
# config/baseline_heuristic.json
{
    "heuristic_agent": {
        "max_depth": 3,
        "time_limit_seconds": 1.0,
        "max_branching_factor": 24,
        "use_iterative_deepening": true,
        "random_seed": 42
    }
}
```

### Configuration Template: Heuristic Policy

```python
# config/heuristic_policy.json
{
    "heuristic_policy": {
        "search_depth": 3,
        "time_limit": 1.0,
        "use_fast_mode": false,
        "randomness": 0.1,
        "temperature": 0.5,
        "random_seed": null
    }
}
```

### Configuration Template: MCTS Integration

```python
# config/mcts_heuristic.json
{
    "mcts": {
        "num_simulations": 100,
        "evaluation_mode": "hybrid",
        "heuristic_weight": 0.3,
        "heuristic_score_scale": 100.0
    },
    "heuristic_evaluator": {
        "weight_config_file": "config/heuristic_weights.json"
    }
}
```

---

## Performance Considerations

### Evaluation Time

The heuristic agent is optimized for speed:
- **Average evaluation time**: < 1ms per move
- **Batch evaluation**: Supports efficient batch processing
- **Caching**: Phase detection and feature extraction are optimized

### Memory Usage

- **Minimal overhead**: Heuristic agent uses minimal memory
- **No neural network**: Reduces memory footprint compared to MCTS
- **Efficient state copying**: Uses optimized `GameState.copy()`

### Scaling Considerations

For large-scale training:

1. **Use batch evaluation** when evaluating multiple positions:
   ```python
   scores = evaluator.evaluate_batch(game_states, players)
   ```

2. **Limit search depth** for faster moves:
   ```python
   config = HeuristicAgentConfig(max_depth=2, time_limit_seconds=0.1)
   ```

3. **Use fast mode** in `HeuristicPolicy`:
   ```python
   config = HeuristicPolicyConfig(use_fast_mode=True)
   ```

---

## Best Practices

### 1. Start with Baseline Integration

Begin by integrating the heuristic agent as a baseline opponent to establish performance benchmarks:

```python
# Run baseline evaluation
baseline_metrics = run_tournament_vs_heuristic(network_policy, num_games=100)
print(f"Network vs Heuristic: {baseline_metrics.win_rate:.3f}")
```

### 2. Use Appropriate Search Depth

Balance between strength and speed:
- **Depth 2**: Fast, suitable for rapid evaluation
- **Depth 3**: Balanced (default)
- **Depth 4+**: Stronger but slower

### 3. Monitor Performance Metrics

Track key metrics during integration:
- Win rate vs heuristic baseline
- Average move time
- Game quality metrics

### 4. Gradual Integration

Integrate gradually:
1. Start with baseline evaluation
2. Add heuristic policy to self-play
3. Integrate heuristic evaluation into MCTS

### 5. Configuration Management

Use configuration files for easy tuning:
- Store weights in JSON files
- Use `ConfigManager` for weight management
- Version control configuration files

---

## Transition to Phase 1

The heuristic system is designed as Phase 0 (baseline) before transitioning to Phase 1 (neural network training).

### Phase 0 → Phase 1 Transition Plan

1. **Validate Heuristic Performance**
   - Run final validation tournament
   - Verify success criteria (60%+ win rate, <1ms eval time)
   - Document performance metrics

2. **Establish Baseline**
   - Use heuristic agent as strong baseline
   - Compare neural network performance against heuristic
   - Track improvement over training iterations

3. **Hybrid Training**
   - Start with heuristic-guided self-play
   - Gradually transition to neural-guided self-play
   - Use `AdaptivePolicy` for smooth transition

4. **Evaluation Framework**
   - Maintain heuristic baseline for evaluation
   - Compare neural network against heuristic at checkpoints
   - Track progress metrics

### Example Transition Code

```python
from yinsh_ml.self_play.policies import AdaptivePolicy, AdaptivePolicyConfig

# Create adaptive policy that transitions from heuristic to neural
adaptive_config = AdaptivePolicyConfig(
    initial_policy='heuristic',
    target_policy='neural',
    transition_games=1000,
    min_neural_quality=0.55,  # Require 55% win rate before transition
)

adaptive_policy = AdaptivePolicy(config=adaptive_config)

# Use in training
for iteration in range(num_iterations):
    # Policy automatically transitions based on quality
    games = generate_games_with_policy(adaptive_policy, num_games=100)
    train_network(games)
    
    # Evaluate against heuristic baseline
    metrics = evaluate_vs_heuristic(network_policy)
    print(f"Iteration {iteration}: Win rate vs heuristic: {metrics.win_rate:.3f}")
```

---

## Troubleshooting

### Common Issues

#### Issue: Slow Evaluation Times

**Solution**: Reduce search depth or enable fast mode:
```python
config = HeuristicAgentConfig(max_depth=2, time_limit_seconds=0.1)
# or
policy_config = HeuristicPolicyConfig(use_fast_mode=True)
```

#### Issue: Inconsistent Results

**Solution**: Set random seed for reproducibility:
```python
config = HeuristicAgentConfig(random_seed=42)
```

#### Issue: Integration Errors

**Solution**: Verify interface compatibility:
- Ensure `GameState` provides required methods
- Check that `Player` enum values are correct
- Verify move generation compatibility

### Performance Debugging

Enable debug mode to track performance:
```python
config = HeuristicAgentConfig(debug=True)
agent = HeuristicAgent(config=config)
move = agent.select_move(game_state)
stats = agent.last_search_stats
print(f"Search stats: {stats}")
```

---

## See Also

- [API Documentation](heuristic_agent_api.md) - Complete API reference
- [Interface Compatibility](heuristic_integration_compatibility.md) - Interface compatibility analysis
- [Validation Report](validation_results/final_validation_report.md) - Performance validation results

---

## Summary

The heuristic agent provides a strong baseline for AlphaZero training with:

- ✅ Fast evaluation (<1ms per move)
- ✅ Strong play (60%+ win rate vs random)
- ✅ Easy integration (compatible interfaces)
- ✅ Flexible configuration (tunable parameters)

Follow this guide to integrate the heuristic agent into your training pipeline and establish a solid foundation for neural network training.

