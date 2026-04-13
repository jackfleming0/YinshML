# Heuristic System Documentation

Complete documentation for the Yinsh heuristic evaluation and agent system.

## Quick Start

The heuristic system provides a fast, strong baseline agent for Yinsh game evaluation and self-play. Get started in minutes:

```python
from yinsh_ml.agents.heuristic_agent import HeuristicAgent
from yinsh_ml.game.game_state import GameState

# Create agent
agent = HeuristicAgent()

# Use agent
game_state = GameState()
move = agent.select_move(game_state)
```

## Documentation Overview

### Core Documentation

- **[API Documentation](heuristic_agent_api.md)** - Complete API reference for `HeuristicAgent`, `HeuristicAgentConfig`, and `YinshHeuristics`
  - Class definitions and methods
  - Configuration parameters
  - Usage examples
  - Performance tuning guide

- **[Integration Guide](integration_guide.md)** - Step-by-step guide for integrating into AlphaZero training pipeline
  - Baseline opponent integration
  - Self-play policy integration
  - MCTS evaluation integration
  - Configuration templates
  - Best practices

- **[Interface Compatibility](heuristic_integration_compatibility.md)** - Detailed analysis of interface compatibility with game engine
  - GameState interface compatibility
  - Board interface compatibility
  - Player/PieceType enum compatibility
  - Test results and verification

### Validation and Results

- **[Validation Results](validation_results/final_validation_report.md)** - Comprehensive validation report with performance metrics
  - Tournament results (heuristic vs random, vs baseline)
  - Success criteria verification (60%+ win rate, <1ms eval time)
  - Statistical analysis (confidence intervals, significance tests)
  - Performance profiling data

## Key Features

### Performance

- **Fast Evaluation**: Average move time < 1ms
- **Strong Play**: 60%+ win rate vs random opponents
- **Efficient**: Optimized batch evaluation support

### Integration

- **Easy Integration**: Direct compatibility with `GameState` interface
- **Flexible**: Multiple integration methods (baseline, policy, MCTS)
- **Configurable**: Tunable weights and search parameters

### Architecture

- **HeuristicAgent**: Search agent using heuristic-guided negamax
- **YinshHeuristics**: Position evaluator with feature extraction
- **HeuristicPolicy**: Policy wrapper for self-play integration

## Quick Reference

### Basic Usage

```python
from yinsh_ml.agents.heuristic_agent import HeuristicAgent, HeuristicAgentConfig
from yinsh_ml.game.game_state import GameState

# Default configuration
agent = HeuristicAgent()
move = agent.select_move(game_state)

# Custom configuration
config = HeuristicAgentConfig(
    max_depth=4,
    time_limit_seconds=2.0,
)
agent = HeuristicAgent(config=config)
```

### Batch Evaluation

```python
from yinsh_ml.heuristics import YinshHeuristics
from yinsh_ml.game.constants import Player

evaluator = YinshHeuristics()
scores = evaluator.evaluate_batch(game_states, players)
```

### Tournament Evaluation

```python
from yinsh_ml.agents.tournament import TournamentEvaluator, TournamentConfig

evaluator = TournamentEvaluator()
config = TournamentConfig(num_games=1000)
metrics = evaluator.run_large_scale_tournament(config)
print(f"Win rate: {metrics.win_rate:.3f}")
```

## Integration Examples

### As Baseline Opponent

```python
from yinsh_ml.agents.heuristic_agent import HeuristicAgent

baseline = HeuristicAgent()
# Use in evaluation games
```

### As Self-Play Policy

```python
from yinsh_ml.self_play.policies import HeuristicPolicy, HeuristicPolicyConfig

policy_config = HeuristicPolicyConfig(search_depth=3)
policy = HeuristicPolicy(config=policy_config)
# Use in self-play runner
```

### In MCTS Search

```python
from yinsh_ml.heuristics import YinshHeuristics
from yinsh_ml.search.mcts import MCTS

heuristic_evaluator = YinshHeuristics()
mcts = MCTS(heuristic_evaluator=heuristic_evaluator)
# Use in MCTS search
```

## Configuration

### Weight Configuration

Heuristic weights can be configured via JSON files:

```json
{
    "early": {
        "completed_runs": 10.0,
        "potential_runs": 5.0,
        ...
    },
    "mid": { ... },
    "late": { ... }
}
```

See the [API Documentation](heuristic_agent_api.md) for details.

### Search Configuration

```python
config = HeuristicAgentConfig(
    max_depth=3,              # Search depth
    time_limit_seconds=1.0,   # Time per move
    max_branching_factor=24,   # Move limit
)
```

## Performance Benchmarks

Based on validation results:

- **Win Rate**: 60%+ vs random opponents
- **Evaluation Time**: < 1ms average per move
- **Max Evaluation Time**: < 10ms (safety threshold)
- **Game Length**: ~50-60 moves average

See [Validation Report](validation_results/final_validation_report.md) for detailed metrics.

## Troubleshooting

### Common Issues

**Slow evaluation**: Reduce `max_depth` or enable fast mode
**Inconsistent results**: Set `random_seed` for reproducibility
**Integration errors**: Verify interface compatibility

See [Integration Guide](integration_guide.md) troubleshooting section for details.

## Next Steps

1. **Read API Documentation**: Understand the complete API
2. **Review Integration Guide**: Learn integration methods
3. **Run Validation**: Execute tournament validation
4. **Integrate**: Add to your training pipeline

## Related Documentation

- [Heuristic Features](../heuristics/features.py) - Feature extraction implementation
- [Weight Management](../heuristics/weight_manager.py) - Weight configuration system
- [Tournament System](../agents/tournament.py) - Tournament evaluation framework
- [Statistics](../agents/statistics.py) - Statistical analysis tools

## Support

For issues or questions:
- Review the [API Documentation](heuristic_agent_api.md)
- Check the [Integration Guide](integration_guide.md)
- Review [Interface Compatibility](heuristic_integration_compatibility.md)

---

**Last Updated**: See validation report for latest performance metrics and validation date.

