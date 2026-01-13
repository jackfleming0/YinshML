# Training System Refactor Plan - Path to Superhuman YinshML

**Date Created:** 2026-01-13
**Goal:** Achieve superhuman YINSH play through proper AlphaZero implementation
**Status:** Planning Phase
**Estimated Timeline:** 6-8 weeks full implementation

---

## Executive Summary

### Current State Assessment

The YinshML training system has fundamental architectural issues preventing successful training:

1. **Heuristics are completely disconnected** - Two separate MCTS implementations, only the pure-neural one is used
2. **Severe under-training** - Each sample trained only ~0.01 times (needs 100-1000x more)
3. **Hardcoded hyperparameters** - Config files don't control what they should
4. **No batch processing** - M2 CPU underutilized, serial evaluations during MCTS
5. **Training/inference mismatch** - Value head trained on different objective than used

**Critical Finding:** The 100K game heuristic analysis and learned feature weights are completely unused in training. The `yinsh_ml/search/mcts.py` file with heuristic integration is orphaned code - never imported or used by the training pipeline.

### Strategic Goals

**Short-term (Weeks 1-2):** Fix critical architecture issues to enable any learning at all
**Medium-term (Weeks 3-4):** Implement proper AlphaZero-style training loop
**Long-term (Weeks 5-8):** Optimize for superhuman play with curriculum learning and advanced techniques

### Success Criteria by Phase

| Phase | Criteria | Expected ELO vs Random |
|-------|----------|------------------------|
| Phase 1 | Network learns basic patterns, beats random 60%+ | +200 |
| Phase 2 | Beats pure heuristic agent 55%+ | +400 |
| Phase 3 | Beats best heuristic+MCTS agent 55%+ | +600 |
| Phase 4 | Superhuman play, consistent tactical brilliance | +800+ |

---

## Phase 1: Critical Architecture Fixes (Weeks 1-2)

**Goal:** Make the training system actually work at a basic level
**Estimated Effort:** 20-30 hours

### 1.1: Unify MCTS Implementations

**Problem:** Two incompatible MCTS implementations exist:
- `yinsh_ml/search/mcts.py` - Has heuristic integration, never used
- `yinsh_ml/training/self_play.py` - Actually runs, pure neural only

**Decision Required:** Merge or Delete?

#### Option A: Merge into Single Implementation (RECOMMENDED)
**Pros:** Cleaner codebase, single source of truth, enables heuristic integration
**Cons:** More refactoring work upfront (4-6 hours)

**Implementation Steps:**

1. **Create unified MCTS class** in `yinsh_ml/search/mcts_unified.py`:
   ```python
   from enum import Enum
   from typing import Optional
   import numpy as np
   from yinsh_ml.heuristics.evaluator import YinshHeuristics
   from yinsh_ml.network.wrapper import NetworkWrapper

   class EvaluationMode(Enum):
       PURE_NEURAL = "pure_neural"
       PURE_HEURISTIC = "pure_heuristic"
       HYBRID = "hybrid"

   class UnifiedMCTS:
       def __init__(
           self,
           network: NetworkWrapper,
           heuristic_evaluator: Optional[YinshHeuristics],
           evaluation_mode: EvaluationMode = EvaluationMode.HYBRID,
           heuristic_weight: float = 0.5,
           num_simulations: int = 100,
           c_puct: float = 1.0,
           # ... other params
       ):
           self.network = network
           self.heuristic_evaluator = heuristic_evaluator
           self.evaluation_mode = evaluation_mode
           self.heuristic_weight = heuristic_weight
           # ...

       def _evaluate_leaf(self, state: GameState) -> Tuple[np.ndarray, float]:
           """Evaluate leaf node using configured evaluation mode."""

           # Get neural network evaluation
           if self.evaluation_mode in [EvaluationMode.PURE_NEURAL, EvaluationMode.HYBRID]:
               policy_nn, value_nn = self.network.predict_from_state(state)
           else:
               policy_nn, value_nn = None, 0.0

           # Get heuristic evaluation
           if self.evaluation_mode in [EvaluationMode.PURE_HEURISTIC, EvaluationMode.HYBRID]:
               value_heuristic = self.heuristic_evaluator.evaluate(state)
               # Policy from heuristic (uniform over legal moves for now)
               legal_moves = state.get_legal_moves()
               policy_heuristic = np.zeros_like(policy_nn) if policy_nn is not None else None
               if policy_heuristic is not None:
                   for move in legal_moves:
                       idx = self.state_encoder.move_to_index(move)
                       policy_heuristic[idx] = 1.0 / len(legal_moves)
           else:
               value_heuristic = 0.0
               policy_heuristic = None

           # Combine based on mode
           if self.evaluation_mode == EvaluationMode.PURE_NEURAL:
               return policy_nn, value_nn
           elif self.evaluation_mode == EvaluationMode.PURE_HEURISTIC:
               return policy_heuristic, value_heuristic
           else:  # HYBRID
               w_h = self.heuristic_weight
               w_n = 1.0 - w_h
               policy = w_n * policy_nn + w_h * policy_heuristic
               value = w_n * value_nn + w_h * value_heuristic
               return policy, value
   ```

2. **Update SelfPlay class** in `yinsh_ml/training/self_play.py`:
   ```python
   from yinsh_ml.search.mcts_unified import UnifiedMCTS, EvaluationMode
   from yinsh_ml.heuristics.evaluator import YinshHeuristics

   class SelfPlay:
       def __init__(self, config, network, state_encoder):
           self.config = config
           self.network = network
           self.state_encoder = state_encoder

           # Initialize heuristic evaluator
           self.heuristic_evaluator = YinshHeuristics()

           # Get evaluation mode from config
           eval_mode_str = config.get('evaluation_mode', 'hybrid')
           self.evaluation_mode = EvaluationMode[eval_mode_str.upper()]

           # Get heuristic weight from config (with decay schedule)
           self.heuristic_weight = config.get('heuristic_weight', 0.5)

           # Create unified MCTS
           self.mcts = UnifiedMCTS(
               network=network,
               heuristic_evaluator=self.heuristic_evaluator,
               evaluation_mode=self.evaluation_mode,
               heuristic_weight=self.heuristic_weight,
               num_simulations=config.get('num_simulations', 100),
               c_puct=config.get('c_puct', 1.0),
               # ... other params
           )
   ```

3. **Delete old implementations**:
   - Remove MCTS class from `yinsh_ml/training/self_play.py` (lines 82-537)
   - Remove or deprecate `yinsh_ml/search/mcts.py` (move reusable code to unified)

4. **Update imports throughout codebase**:
   ```bash
   # Find all files importing old MCTS
   grep -r "from.*training.self_play.*import.*MCTS" . --include="*.py"
   grep -r "from.*search.mcts import" . --include="*.py"

   # Update to use UnifiedMCTS
   ```

5. **Add tests** in `tests/test_unified_mcts.py`:
   ```python
   def test_pure_neural_mode():
       """Test MCTS with pure neural evaluation."""
       mcts = UnifiedMCTS(
           network=mock_network,
           heuristic_evaluator=None,
           evaluation_mode=EvaluationMode.PURE_NEURAL,
       )
       # ... test logic

   def test_pure_heuristic_mode():
       """Test MCTS with pure heuristic evaluation."""
       # ...

   def test_hybrid_mode():
       """Test MCTS with hybrid evaluation."""
       # ...

   def test_heuristic_weight_scheduling():
       """Test that heuristic weight can be updated during training."""
       # ...
   ```

**Validation:**
- [ ] All tests pass
- [ ] Self-play can run with PURE_NEURAL mode (should match old behavior)
- [ ] Self-play can run with PURE_HEURISTIC mode (new capability)
- [ ] Self-play can run with HYBRID mode (new capability)
- [ ] Heuristic weight can be adjusted during training

#### Option B: Delete search/mcts.py
**Pros:** Quick (1 hour)
**Cons:** Loses heuristic integration work, harder to add back later

Not recommended for superhuman play goal.

---

### 1.2: Connect Heuristics to Training Pipeline

**Problem:** YinshHeuristics evaluator is well-implemented but never called during training.

**Implementation Steps:**

1. **Add evaluation_mode to config** in `configs/`:
   ```yaml
   # configs/default_training.yaml
   mcts:
     evaluation_mode: "hybrid"  # Options: pure_neural, pure_heuristic, hybrid
     heuristic_weight_schedule:
       initial: 0.7        # Start with 70% heuristic weight
       final: 0.1          # End with 10% heuristic weight
       decay_iterations: 100  # Decay over first 100 iterations
       decay_type: "linear"   # Options: linear, exponential, cosine
   ```

2. **Implement heuristic weight scheduling** in `yinsh_ml/training/supervisor.py`:
   ```python
   class TrainingSupervisor:
       def __init__(self, config):
           # ...
           self.heuristic_weight_schedule = config['mcts'].get('heuristic_weight_schedule', {})
           self.initial_heuristic_weight = self.heuristic_weight_schedule.get('initial', 0.7)
           self.final_heuristic_weight = self.heuristic_weight_schedule.get('final', 0.1)
           self.decay_iterations = self.heuristic_weight_schedule.get('decay_iterations', 100)
           self.decay_type = self.heuristic_weight_schedule.get('decay_type', 'linear')

       def _get_heuristic_weight(self, iteration: int) -> float:
           """Calculate heuristic weight for current iteration."""
           if iteration >= self.decay_iterations:
               return self.final_heuristic_weight

           progress = iteration / self.decay_iterations

           if self.decay_type == 'linear':
               weight = self.initial_heuristic_weight - progress * (
                   self.initial_heuristic_weight - self.final_heuristic_weight
               )
           elif self.decay_type == 'exponential':
               weight = self.final_heuristic_weight + (
                   self.initial_heuristic_weight - self.final_heuristic_weight
               ) * np.exp(-5 * progress)
           elif self.decay_type == 'cosine':
               weight = self.final_heuristic_weight + 0.5 * (
                   self.initial_heuristic_weight - self.final_heuristic_weight
               ) * (1 + np.cos(np.pi * progress))
           else:
               weight = self.initial_heuristic_weight

           return weight

       def train_iteration(self, iteration: int):
           # Update heuristic weight for this iteration
           current_heuristic_weight = self._get_heuristic_weight(iteration)
           self.self_play.mcts.heuristic_weight = current_heuristic_weight

           self.logger.info(f"Iteration {iteration}: heuristic_weight={current_heuristic_weight:.3f}")

           # ... rest of training iteration
   ```

3. **Add heuristic evaluation metrics** to logging:
   ```python
   # Track how much heuristic vs neural is being used
   metrics = {
       'iteration': iteration,
       'heuristic_weight': current_heuristic_weight,
       'evaluation_mode': self.evaluation_mode.value,
       'neural_contribution': 1.0 - current_heuristic_weight,
       # ... other metrics
   }
   ```

**Validation:**
- [ ] Heuristic weight starts at configured initial value
- [ ] Heuristic weight decays according to schedule
- [ ] Self-play games use heuristic evaluation (can verify by adding debug logging)
- [ ] Training metrics show heuristic contribution over time

---

### 1.3: Fix Severe Under-Training

**Problem:** Each sample trained only ~0.01 times (needs 100-1000x more)
- Current: 50 games → ~2,500 samples, 4 epochs × 10-20 batches = 40-80 updates
- Needed: 1,000-10,000 updates per sample

**Root Cause:** AlphaZero uses much larger replay buffers and many more training epochs per iteration.

**Implementation Steps:**

1. **Increase replay buffer size** in `yinsh_ml/training/trainer.py`:
   ```python
   class GameExperience:
       def __init__(self, max_size: int = 100000, subsample_long_games: bool = True):
           # Increase from 10,000 to 100,000
           self.states = deque(maxlen=max_size)
           self.move_probs = deque(maxlen=max_size)
           self.values = deque(maxlen=max_size)
           self.phases = deque(maxlen=max_size)
           self.max_size = max_size
   ```

2. **Increase epochs per iteration** in `configs/default_training.yaml`:
   ```yaml
   training:
     epochs_per_iteration: 40  # Increase from 4 to 40
     batch_size: 256
     batches_per_epoch: "auto"  # Will be computed based on buffer size
   ```

3. **Implement progressive buffer growth**:
   ```python
   class TrainingSupervisor:
       def __init__(self, config):
           # ...
           self.min_buffer_size = config['training'].get('min_buffer_size', 10000)
           self.target_buffer_size = config['training'].get('target_buffer_size', 100000)
           self.buffer_growth_iterations = config['training'].get('buffer_growth_iterations', 50)

       def _get_current_buffer_size(self, iteration: int) -> int:
           """Grow buffer size over first N iterations."""
           if iteration >= self.buffer_growth_iterations:
               return self.target_buffer_size

           progress = iteration / self.buffer_growth_iterations
           current_size = int(
               self.min_buffer_size + progress * (self.target_buffer_size - self.min_buffer_size)
           )
           return current_size

       def train_iteration(self, iteration: int):
           # Update buffer size
           current_buffer_size = self._get_current_buffer_size(iteration)
           self.experience_buffer.max_size = current_buffer_size

           # Only start training once we have minimum samples
           if len(self.experience_buffer) < self.min_buffer_size:
               self.logger.info(f"Buffer size {len(self.experience_buffer)}/{self.min_buffer_size}, "
                              f"skipping training this iteration")
               return

           # ... proceed with training
   ```

4. **Add sample reuse metrics**:
   ```python
   def compute_sample_reuse_rate(buffer_size: int, batch_size: int, num_epochs: int) -> float:
       """Calculate how many times each sample is trained on average."""
       batches_per_epoch = buffer_size // batch_size
       total_samples_trained = batches_per_epoch * num_epochs * batch_size
       reuse_rate = total_samples_trained / buffer_size
       return reuse_rate

   # Log this metric
   reuse_rate = compute_sample_reuse_rate(len(self.experience_buffer), batch_size, num_epochs)
   self.logger.info(f"Sample reuse rate: {reuse_rate:.2f}x (target: 100-1000x)")
   ```

**Validation:**
- [ ] Buffer grows from 10K → 100K over first 50 iterations
- [ ] Training skipped until minimum buffer size reached
- [ ] Sample reuse rate logged and increases over time
- [ ] Each sample trained 100+ times on average (after buffer full)

**Expected Impact:** This alone should enable the network to start learning basic patterns.

---

### 1.4: Remove Hardcoded Hyperparameters

**Problem:** Learning rates, phase weights, and other hyperparameters hardcoded in `trainer.py`, conflicting with config.

**Implementation Steps:**

1. **Create comprehensive config schema** in `yinsh_ml/training/config_schema.py`:
   ```python
   from dataclasses import dataclass
   from typing import Optional, Literal

   @dataclass
   class OptimizerConfig:
       policy_lr: float = 0.001
       value_lr: float = 0.0001
       value_lr_factor: float = 1.0
       policy_weight_decay: float = 1e-4
       value_weight_decay: float = 1e-3
       value_momentum: float = 0.9

   @dataclass
   class SchedulerConfig:
       type: Literal["cyclic", "cosine", "step"] = "cyclic"
       base_lr: float = 1e-5
       max_lr: float = 1e-4
       step_size: int = 500
       # ... other scheduler params

   @dataclass
   class PhaseWeightConfig:
       ring_placement: float = 0.5
       main_game: float = 2.0
       ring_removal: float = 0.5

   @dataclass
   class TrainingConfig:
       optimizer: OptimizerConfig
       scheduler: SchedulerConfig
       phase_weights: PhaseWeightConfig
       epochs_per_iteration: int = 40
       batch_size: int = 256
       # ...
   ```

2. **Refactor YinshTrainer to use config** in `yinsh_ml/training/trainer.py`:
   ```python
   class YinshTrainer:
       def __init__(self, network, config: TrainingConfig, device='cpu'):
           self.network = network
           self.config = config
           self.device = device

           # Create optimizers from config (no hardcoded values!)
           policy_params = list(network.policy_head.parameters())
           value_params = list(network.value_head.parameters())

           self.policy_optimizer = optim.Adam(
               policy_params,
               lr=config.optimizer.policy_lr,
               weight_decay=config.optimizer.policy_weight_decay
           )

           value_lr = config.optimizer.value_lr * config.optimizer.value_lr_factor
           self.value_optimizer = optim.SGD(
               value_params,
               lr=value_lr,
               momentum=config.optimizer.value_momentum,
               weight_decay=config.optimizer.value_weight_decay
           )

           # Create schedulers from config
           if config.scheduler.type == "cyclic":
               self.policy_scheduler = optim.lr_scheduler.CyclicLR(
                   self.policy_optimizer,
                   base_lr=config.scheduler.base_lr,
                   max_lr=config.scheduler.max_lr,
                   step_size_up=config.scheduler.step_size,
               )
               # ... same for value scheduler
           # ... other scheduler types

           # Phase weights from config
           self.phase_weights = {
               'RING_PLACEMENT': config.phase_weights.ring_placement,
               'MAIN_GAME': config.phase_weights.main_game,
               'RING_REMOVAL': config.phase_weights.ring_removal,
           }
   ```

3. **Remove LR override in supervisor** (currently line 181):
   ```python
   # DELETE THIS:
   # self.trainer.policy_optimizer.param_groups[0]['lr'] = base_lr

   # Config is now the single source of truth
   ```

4. **Add hyperparameter logging**:
   ```python
   def log_hyperparameters(self):
       """Log all active hyperparameters for debugging."""
       hparams = {
           'policy_lr': self.policy_optimizer.param_groups[0]['lr'],
           'value_lr': self.value_optimizer.param_groups[0]['lr'],
           'policy_weight_decay': self.config.optimizer.policy_weight_decay,
           'value_weight_decay': self.config.optimizer.value_weight_decay,
           'phase_weights': self.phase_weights,
           'batch_size': self.config.batch_size,
           'epochs': self.config.epochs_per_iteration,
       }
       self.logger.info(f"Active hyperparameters: {hparams}")
   ```

**Validation:**
- [ ] No hardcoded learning rates in trainer.py
- [ ] All hyperparameters come from config
- [ ] Config changes actually affect training behavior
- [ ] Hyperparameters logged at start of training

---

### 1.5: Fix Value Head Training/Inference Mismatch

**Problem:** Value head trained with MSE + BCE loss, but MCTS only uses scalar value output.

**Analysis:**
Current training in `trainer.py` lines 502-512:
```python
value_loss_mse = F.mse_loss(pred_values, target_values)
target_outcomes = (target_values > 0).long()
value_loss_ce = F.binary_cross_entropy(value_probs, target_outcomes.float())
value_loss = self.value_loss_weights[0] * value_loss_mse +
             self.value_loss_weights[1] * value_loss_ce
```

But in MCTS, only `pred_values` (scalar) is used, not `value_probs`.

**Decision Required:** Keep dual loss or simplify?

#### Option A: Simplify to MSE Only (RECOMMENDED for Phase 1)
**Pros:** Matches inference, simpler, faster
**Cons:** Loses some training signal from win/loss classification

```python
# In trainer.py
def compute_value_loss(self, pred_values, target_values):
    """Compute value loss (MSE only)."""
    return F.mse_loss(pred_values, target_values)
```

#### Option B: Use Both Outputs in MCTS (More Complex)
**Pros:** Uses full network capacity
**Cons:** More complex MCTS code, slower inference

```python
# In unified MCTS
def _evaluate_leaf(self, state):
    policy, value, value_probs = self.network.predict_full(state)

    # Combine scalar value and win probability
    value_scalar = value  # Regression output
    win_prob = value_probs[1]  # Probability of winning (index 1)

    # Weighted combination
    combined_value = 0.7 * value_scalar + 0.3 * (2 * win_prob - 1)
    return policy, combined_value
```

**Recommendation:** Use Option A for Phase 1 (simplify), consider Option B in Phase 4 (optimization).

**Implementation:**

1. **Simplify value loss** in `trainer.py`:
   ```python
   def compute_value_loss(self, pred_values, target_values):
       """Compute value loss using MSE only."""
       return F.mse_loss(pred_values, target_values)
   ```

2. **Remove BCE loss code** (lines 502-512):
   ```python
   # DELETE:
   # target_outcomes = (target_values > 0).long()
   # value_loss_ce = F.binary_cross_entropy(value_probs, target_outcomes.float())
   # value_loss = self.value_loss_weights[0] * value_loss_mse + ...

   # REPLACE WITH:
   value_loss = F.mse_loss(pred_values, target_values)
   ```

3. **Update network forward pass** if needed to ensure consistent output shape.

**Validation:**
- [ ] Value loss is single scalar (MSE only)
- [ ] MCTS uses same value output as training
- [ ] Training runs without errors
- [ ] Value predictions improve over training

---

## Phase 2: Proper AlphaZero Training Loop (Weeks 3-4)

**Goal:** Implement full AlphaZero methodology for stable, effective training
**Estimated Effort:** 30-40 hours

### 2.1: Implement Batched MCTS Evaluation

**Problem:** Each MCTS simulation calls network individually (250K serial calls per iteration).
**Impact:** M2 CPU severely underutilized, could be 10-20x faster with batching.

**AlphaZero Approach:** Collect multiple leaf nodes, evaluate in single batch.

**Implementation Steps:**

1. **Create virtual loss mechanism** to handle concurrent expansions:
   ```python
   class MCTSNode:
       def __init__(self):
           self.visit_count = 0
           self.value_sum = 0.0
           self.virtual_losses = 0  # NEW: track in-flight evaluations
           self.children = {}
           self.prior = 0.0

       def get_value(self):
           """Get average value, accounting for virtual losses."""
           adjusted_visits = self.visit_count + self.virtual_losses
           if adjusted_visits == 0:
               return 0.0
           return self.value_sum / adjusted_visits

       def add_virtual_loss(self):
           """Mark this node as being evaluated."""
           self.virtual_losses += 1

       def remove_virtual_loss(self):
           """Remove virtual loss after evaluation completes."""
           self.virtual_losses -= 1
   ```

2. **Implement batched tree traversal**:
   ```python
   class BatchedMCTS:
       def __init__(self, batch_size=32, **kwargs):
           self.batch_size = batch_size
           # ... other init

       def search_batch(self, states: List[GameState], num_simulations: int):
           """Perform MCTS for multiple root states simultaneously."""
           roots = [self._create_root(state) for state in states]

           for sim in range(num_simulations):
               # Traverse all trees and collect leaf nodes
               leaves = []
               paths = []

               for root, state in zip(roots, states):
                   leaf, path = self._select_leaf(root, state.copy())
                   leaf.add_virtual_loss()
                   leaves.append((leaf, state))
                   paths.append(path)

               # Batch evaluate all leaves
               if len(leaves) >= self.batch_size or sim % 10 == 0:
                   self._evaluate_and_backup_batch(leaves, paths)
                   leaves = []
                   paths = []

               # Handle remaining leaves
               if leaves:
                   self._evaluate_and_backup_batch(leaves, paths)

           # Return policies for all roots
           return [self._get_policy(root) for root in roots]

       def _evaluate_and_backup_batch(self, leaves, paths):
           """Evaluate multiple leaf nodes in single network call."""
           states = [state for _, state in leaves]

           # Single batched forward pass
           policies, values = self.network.predict_batch(states)

           # Backup results
           for (leaf, _), path, policy, value in zip(leaves, paths, policies, values):
               leaf.remove_virtual_loss()
               self._expand_node(leaf, policy)
               self._backup(path, value)
   ```

3. **Add batched inference to NetworkWrapper**:
   ```python
   class NetworkWrapper:
       def predict_batch(self, states: List[GameState]) -> Tuple[np.ndarray, np.ndarray]:
           """Evaluate multiple states in single forward pass."""
           # Encode all states
           state_tensors = [self.state_encoder.encode_state(s) for s in states]
           batch_tensor = torch.stack(state_tensors).to(self.device)

           with torch.no_grad():
               policy_logits, values = self.model(batch_tensor)
               policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
               values = values.cpu().numpy()

           return policies, values
   ```

4. **Integrate batched MCTS into self-play**:
   ```python
   class SelfPlay:
       def __init__(self, config, network, state_encoder):
           # ...
           self.mcts_batch_size = config.get('mcts_batch_size', 32)
           self.mcts = BatchedMCTS(
               batch_size=self.mcts_batch_size,
               # ... other params
           )

       def play_game_batch(self, num_games: int):
           """Play multiple games in parallel with batched MCTS."""
           # ...
   ```

**Expected Speedup:** 10-20x faster self-play generation on M2.

**Validation:**
- [ ] Batched MCTS produces same results as serial (within numerical precision)
- [ ] Self-play throughput increases 5x+ (measure games/hour)
- [ ] Memory usage remains reasonable (<8GB)
- [ ] Network forward pass count reduced 10x+

---

### 2.2: Implement Proper Tournament Evaluation

**Problem:** Current tournament evaluation is round-robin with all models, expensive and grows over time.

**AlphaZero Approach:** New model plays fixed number of games against best model only.

**Implementation Steps:**

1. **Create head-to-head evaluator**:
   ```python
   class HeadToHeadEvaluator:
       def __init__(self, num_games=100, time_limit=None):
           self.num_games = num_games
           self.time_limit = time_limit

       def evaluate(self, candidate_network, best_network, config):
           """Play candidate vs best, return win rate."""
           wins = 0
           losses = 0
           draws = 0

           for game_idx in range(self.num_games):
               # Alternate colors
               if game_idx % 2 == 0:
                   white_net = candidate_network
                   black_net = best_network
               else:
                   white_net = best_network
                   black_net = candidate_network

               # Play game
               outcome = self._play_game(white_net, black_net, config)

               # Record result from candidate's perspective
               if game_idx % 2 == 0:
                   result = outcome  # candidate was white
               else:
                   result = -outcome  # candidate was black

               if result > 0:
                   wins += 1
               elif result < 0:
                   losses += 1
               else:
                   draws += 1

           win_rate = (wins + 0.5 * draws) / self.num_games
           return {
               'wins': wins,
               'losses': losses,
               'draws': draws,
               'win_rate': win_rate,
               'total_games': self.num_games,
           }
   ```

2. **Update promotion logic in supervisor**:
   ```python
   def _evaluate_candidate(self, iteration: int):
       """Evaluate candidate model against current best."""
       self.logger.info(f"Evaluating candidate model (iteration {iteration})")

       # Save candidate
       candidate_path = self.checkpoint_dir / f"candidate_{iteration}.pt"
       self.network.save_model(str(candidate_path))

       # Load best model for comparison
       if self.best_model_path.exists():
           best_network = self._load_network_checkpoint(self.best_model_path)
       else:
           # First model, auto-promote
           self.logger.info("First model, auto-promoting")
           return True

       # Head-to-head evaluation
       evaluator = HeadToHeadEvaluator(num_games=100)
       results = evaluator.evaluate(self.network, best_network, self.config)

       self.logger.info(f"Evaluation results: {results}")

       # Promotion threshold
       win_rate_threshold = self.config.get('promotion_win_rate', 0.55)

       if results['win_rate'] >= win_rate_threshold:
           self.logger.info(f"Promoting new model (win_rate={results['win_rate']:.3f})")
           return True
       else:
           self.logger.info(f"Keeping current best (win_rate={results['win_rate']:.3f})")
           return False
   ```

3. **Remove Wilson score gate** (complex and unnecessary for head-to-head):
   ```python
   # DELETE wilson score calculation (lines 703-790 in supervisor.py)
   # REPLACE with simple win rate threshold
   ```

**Validation:**
- [ ] Candidate plays exactly 100 games vs best (50 as white, 50 as black)
- [ ] Promotion occurs when win_rate ≥ 55%
- [ ] No reversion to old models (only keep best)
- [ ] Evaluation runs in reasonable time (<10 min on M2)

---

### 2.3: Implement Curriculum Learning

**Goal:** Smooth transfer from heuristic-heavy to pure neural network play.

**Strategy:**
1. Start with 70% heuristic weight (HYBRID mode)
2. Linearly decay to 10% over 100 iterations
3. Continue with 10% heuristic as safety net

**Implementation:** Already covered in Phase 1.2, but add advanced scheduling:

```python
class CurriculumScheduler:
    def __init__(self, config):
        self.stages = config.get('curriculum_stages', [
            {'iterations': 30, 'heuristic_weight': 0.7, 'simulations': 100},
            {'iterations': 50, 'heuristic_weight': 0.4, 'simulations': 200},
            {'iterations': 100, 'heuristic_weight': 0.1, 'simulations': 400},
            {'iterations': float('inf'), 'heuristic_weight': 0.05, 'simulations': 800},
        ])
        self.current_stage = 0
        self.iteration = 0

    def get_current_config(self):
        """Get current training configuration based on curriculum stage."""
        stage = self.stages[self.current_stage]
        return {
            'heuristic_weight': stage['heuristic_weight'],
            'num_simulations': stage['simulations'],
        }

    def step(self):
        """Advance curriculum to next stage if threshold reached."""
        self.iteration += 1
        if self.iteration >= self.stages[self.current_stage]['iterations']:
            if self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.logger.info(f"Advancing to curriculum stage {self.current_stage}")
```

**Validation:**
- [ ] Training starts with high heuristic weight
- [ ] Weight decays according to schedule
- [ ] MCTS simulations increase as heuristic weight decreases
- [ ] Network learns progressively harder patterns

---

### 2.4: Add Comprehensive Logging and Visualization

**Goal:** Understand what's happening during training.

**Implementation Steps:**

1. **Add TensorBoard logging**:
   ```python
   from torch.utils.tensorboard import SummaryWriter

   class TrainingSupervisor:
       def __init__(self, config):
           # ...
           self.tb_writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')

       def train_iteration(self, iteration):
           # ... training code ...

           # Log scalars
           self.tb_writer.add_scalar('Training/PolicyLoss', policy_loss, iteration)
           self.tb_writer.add_scalar('Training/ValueLoss', value_loss, iteration)
           self.tb_writer.add_scalar('Training/TotalLoss', total_loss, iteration)
           self.tb_writer.add_scalar('Training/LearningRate', current_lr, iteration)

           # Log curriculum
           self.tb_writer.add_scalar('Curriculum/HeuristicWeight', heuristic_weight, iteration)
           self.tb_writer.add_scalar('Curriculum/NumSimulations', num_sims, iteration)

           # Log evaluation
           self.tb_writer.add_scalar('Evaluation/WinRate', win_rate, iteration)
           self.tb_writer.add_scalar('Evaluation/ELO', elo_rating, iteration)

           # Log self-play quality
           self.tb_writer.add_scalar('SelfPlay/AvgGameLength', avg_game_length, iteration)
           self.tb_writer.add_scalar('SelfPlay/AvgBranchingFactor', avg_branching, iteration)

           # Log buffer stats
           self.tb_writer.add_scalar('Buffer/Size', buffer_size, iteration)
           self.tb_writer.add_scalar('Buffer/ReuseRate', reuse_rate, iteration)
   ```

2. **Create training dashboard script**:
   ```python
   # scripts/monitor_training.py
   import subprocess
   import os

   def launch_tensorboard(log_dir='logs/tensorboard'):
       """Launch TensorBoard for training monitoring."""
       print(f"Launching TensorBoard for {log_dir}")
       subprocess.run(['tensorboard', '--logdir', log_dir, '--port', '6006'])

   if __name__ == '__main__':
       launch_tensorboard()
   ```

3. **Add game quality metrics**:
   ```python
   def compute_game_quality_metrics(game_history):
       """Compute metrics to assess game quality."""
       metrics = {}

       # Game length
       metrics['game_length'] = len(game_history)

       # Move diversity (entropy of move distribution)
       move_counts = Counter([move for move, _ in game_history])
       total_moves = sum(move_counts.values())
       move_probs = [count / total_moves for count in move_counts.values()]
       metrics['move_entropy'] = -sum(p * np.log(p + 1e-10) for p in move_probs)

       # Value variance (should be high for interesting games)
       values = [value for _, value in game_history]
       metrics['value_variance'] = np.var(values)

       # Decisive wins (value magnitude at end)
       metrics['decisiveness'] = abs(values[-1]) if values else 0.0

       return metrics
   ```

**Validation:**
- [ ] TensorBoard launches and shows training metrics
- [ ] Can visualize loss curves, learning rate, etc.
- [ ] Game quality metrics logged and tracked
- [ ] Can diagnose training issues from logs

---

## Phase 3: Optimization for M2 Performance (Weeks 5-6)

**Goal:** Maximize training throughput on Mac Mini M2
**Estimated Effort:** 20-30 hours

### 3.1: Profile and Optimize Bottlenecks

**Implementation Steps:**

1. **Add comprehensive profiling**:
   ```python
   import cProfile
   import pstats
   from torch.profiler import profile, ProfilerActivity

   class TrainingProfiler:
       def __init__(self, enabled=True):
           self.enabled = enabled
           self.profiler = None

       def __enter__(self):
           if self.enabled:
               self.profiler = profile(
                   activities=[ProfilerActivity.CPU],
                   record_shapes=True,
                   profile_memory=True,
                   with_stack=True
               )
               self.profiler.__enter__()
           return self

       def __exit__(self, *args):
           if self.enabled and self.profiler:
               self.profiler.__exit__(*args)
               print(self.profiler.key_averages().table(
                   sort_by="cpu_time_total", row_limit=20
               ))
   ```

2. **Profile key operations**:
   ```bash
   # Profile self-play
   python -m cProfile -o selfplay.prof scripts/run_training.py --iterations 1

   # Analyze profile
   python -c "import pstats; p=pstats.Stats('selfplay.prof'); p.sort_stats('cumtime'); p.print_stats(20)"
   ```

3. **Optimize based on findings** (common bottlenecks):
   - State encoding/decoding
   - Move generation
   - Board copying
   - Network inference

**Expected Optimizations:**

```python
# Cache encoded states
class StateEncoderCached:
    def __init__(self):
        self.cache = {}
        self.max_cache_size = 10000

    def encode_state(self, state):
        state_hash = hash(state)
        if state_hash in self.cache:
            return self.cache[state_hash]

        encoded = self._encode_state_impl(state)

        if len(self.cache) < self.max_cache_size:
            self.cache[state_hash] = encoded

        return encoded

# Optimize move generation with caching
class GameState:
    def __init__(self):
        self._legal_moves_cache = None
        self._cache_valid = False

    def get_legal_moves(self):
        if self._cache_valid and self._legal_moves_cache is not None:
            return self._legal_moves_cache

        self._legal_moves_cache = self._compute_legal_moves()
        self._cache_valid = True
        return self._legal_moves_cache

    def make_move(self, move):
        self._cache_valid = False  # Invalidate cache
        # ... apply move
```

**Validation:**
- [ ] Profile data collected for all major operations
- [ ] Bottlenecks identified (aim for >80% time in network forward pass)
- [ ] Optimizations implemented for top 3 bottlenecks
- [ ] Throughput increased 2x+ from baseline

---

### 3.2: Leverage M2 Neural Engine

**Goal:** Use M2's dedicated neural engine for faster inference.

**Implementation Steps:**

1. **Convert model to CoreML**:
   ```python
   # scripts/convert_to_coreml.py
   import coremltools as ct
   import torch

   def convert_to_coreml(pytorch_model_path, output_path):
       """Convert PyTorch model to CoreML for M2 acceleration."""
       # Load PyTorch model
       model = YinshNetwork()
       model.load_state_dict(torch.load(pytorch_model_path))
       model.eval()

       # Create example input
       example_input = torch.randn(1, 6, 11, 11)

       # Trace model
       traced_model = torch.jit.trace(model, example_input)

       # Convert to CoreML
       coreml_model = ct.convert(
           traced_model,
           inputs=[ct.TensorType(name="state", shape=(1, 6, 11, 11))],
           outputs=[
               ct.TensorType(name="policy"),
               ct.TensorType(name="value")
           ],
           compute_units=ct.ComputeUnit.ALL  # Use Neural Engine + GPU + CPU
       )

       # Save
       coreml_model.save(output_path)
       print(f"CoreML model saved to {output_path}")
   ```

2. **Create CoreML inference wrapper**:
   ```python
   import coremltools as ct

   class CoreMLNetworkWrapper(NetworkWrapper):
       def __init__(self, coreml_model_path):
           self.model = ct.models.MLModel(coreml_model_path)

       def predict_from_state(self, state):
           # Encode state
           state_tensor = self.state_encoder.encode_state(state)
           state_array = state_tensor.numpy()

           # Run inference
           output = self.model.predict({'state': state_array})

           policy = output['policy']
           value = output['value']

           return policy, value
   ```

3. **Benchmark CoreML vs PyTorch**:
   ```python
   # scripts/benchmark_inference.py
   def benchmark_inference(model_path, num_iterations=1000):
       # Test PyTorch
       pytorch_model = load_pytorch_model(model_path)
       pytorch_times = []
       for _ in range(num_iterations):
           start = time.time()
           pytorch_model.predict(state)
           pytorch_times.append(time.time() - start)

       # Test CoreML
       coreml_model = load_coreml_model(model_path + '.mlmodel')
       coreml_times = []
       for _ in range(num_iterations):
           start = time.time()
           coreml_model.predict(state)
           coreml_times.append(time.time() - start)

       print(f"PyTorch: {np.mean(pytorch_times)*1000:.2f}ms ± {np.std(pytorch_times)*1000:.2f}ms")
       print(f"CoreML: {np.mean(coreml_times)*1000:.2f}ms ± {np.std(coreml_times)*1000:.2f}ms")
       print(f"Speedup: {np.mean(pytorch_times)/np.mean(coreml_times):.2f}x")
   ```

**Note:** CoreML has overhead for model switching, so batch processing may still be faster. Benchmark both approaches.

**Validation:**
- [ ] Model successfully converts to CoreML
- [ ] CoreML inference produces same results as PyTorch (within epsilon)
- [ ] Benchmarks show speedup vs PyTorch CPU
- [ ] Self-play with CoreML is faster than PyTorch

---

### 3.3: Multi-Process Self-Play Optimization

**Goal:** Maximize CPU utilization with efficient parallel self-play.

**Implementation Steps:**

1. **Implement shared memory for network weights**:
   ```python
   import torch.multiprocessing as mp

   class SharedNetworkWrapper:
       def __init__(self, model_path):
           self.model = YinshNetwork()
           self.model.load_state_dict(torch.load(model_path))
           self.model.eval()
           self.model.share_memory()  # Share weights across processes

       def update_weights(self, new_weights_path):
           """Update shared weights from training."""
           new_state_dict = torch.load(new_weights_path)
           self.model.load_state_dict(new_state_dict)
   ```

2. **Optimize data transfer between processes**:
   ```python
   class SelfPlayWorker(mp.Process):
       def __init__(self, shared_network, result_queue, config):
           super().__init__()
           self.network = shared_network
           self.result_queue = result_queue
           self.config = config

       def run(self):
           """Worker process for self-play."""
           while True:
               # Play game
               game_data = self.play_game()

               # Send results (avoid large data copies)
               compressed_data = self.compress_game_data(game_data)
               self.result_queue.put(compressed_data)

       def compress_game_data(self, game_data):
           """Compress game data before sending to main process."""
           # Use numpy arrays instead of lists
           # Quantize values if possible
           # Only send essential data
           return compressed_data
   ```

3. **Balance workers and training**:
   ```python
   class AdaptiveWorkerPool:
       def __init__(self, max_workers=5):
           self.max_workers = max_workers
           self.active_workers = max_workers

       def adjust_workers(self, training_time, selfplay_time):
           """Adjust worker count based on bottleneck."""
           ratio = training_time / selfplay_time

           if ratio > 2.0:  # Training is bottleneck
               self.active_workers = max(1, self.active_workers - 1)
           elif ratio < 0.5:  # Self-play is bottleneck
               self.active_workers = min(self.max_workers, self.active_workers + 1)
   ```

**Validation:**
- [ ] All CPU cores utilized during self-play
- [ ] Memory usage reasonable (<8GB total)
- [ ] No process starvation or deadlocks
- [ ] Data transfer overhead minimal (<10% of time)

---

## Phase 4: Advanced Techniques for Superhuman Play (Weeks 7-8)

**Goal:** Implement cutting-edge techniques for maximum performance
**Estimated Effort:** 30-40 hours

### 4.1: Advanced MCTS Enhancements

**Techniques to implement:**

1. **Progressive Widening**:
   ```python
   def should_expand_child(self, parent_visits, num_children):
       """Limit children based on parent visits."""
       max_children = int(np.ceil(parent_visits ** 0.5))
       return num_children < max_children
   ```

2. **First Play Urgency (FPU)**:
   ```python
   def compute_uct_score(self, child, parent_visits, fpu_value=-1.0):
       """Compute UCT score with First Play Urgency."""
       if child.visit_count == 0:
           return fpu_value + child.prior  # Pessimistic initialization

       q_value = child.get_value()
       u_value = self.c_puct * child.prior * np.sqrt(parent_visits) / (1 + child.visit_count)
       return q_value + u_value
   ```

3. **Transposition Table in MCTS**:
   ```python
   class TranspositionMCTS:
       def __init__(self):
           self.transposition_table = {}  # hash -> node

       def _get_or_create_node(self, state):
           state_hash = self.hasher.hash_state(state)
           if state_hash in self.transposition_table:
               return self.transposition_table[state_hash]

           node = MCTSNode()
           self.transposition_table[state_hash] = node
           return node
   ```

4. **PUCT with Variance**:
   ```python
   def compute_uct_with_variance(self, child, parent_visits):
       """UCT with variance term for exploration."""
       q_mean = child.get_value()
       q_variance = child.get_variance()

       exploration = self.c_puct * child.prior * np.sqrt(parent_visits) / (1 + child.visit_count)
       variance_bonus = self.c_var * np.sqrt(q_variance) / (1 + child.visit_count)

       return q_mean + exploration + variance_bonus
   ```

**Validation:**
- [ ] Each enhancement tested independently
- [ ] Combined enhancements don't conflict
- [ ] MCTS strength improves vs baseline
- [ ] Search efficiency increases (fewer sims needed)

---

### 4.2: Advanced Training Techniques

**Techniques to implement:**

1. **Auxiliary Tasks**:
   ```python
   class YinshNetworkWithAux(YinshNetwork):
       def __init__(self):
           super().__init__()
           # Add auxiliary prediction heads
           self.phase_head = nn.Linear(256, 3)  # Predict game phase
           self.mobility_head = nn.Linear(256, 1)  # Predict mobility
           self.run_completion_head = nn.Linear(256, 1)  # Predict run completions

       def forward(self, x):
           features = self.backbone(x)

           # Main outputs
           policy = self.policy_head(features)
           value = self.value_head(features)

           # Auxiliary outputs
           phase = self.phase_head(features)
           mobility = self.mobility_head(features)
           run_completion = self.run_completion_head(features)

           return policy, value, phase, mobility, run_completion
   ```

2. **Prioritized Experience Replay**:
   ```python
   class PrioritizedReplayBuffer:
       def __init__(self, max_size, alpha=0.6):
           self.buffer = []
           self.priorities = []
           self.alpha = alpha

       def add(self, experience, td_error):
           """Add experience with priority based on TD error."""
           priority = (abs(td_error) + 1e-5) ** self.alpha
           self.buffer.append(experience)
           self.priorities.append(priority)

       def sample(self, batch_size, beta=0.4):
           """Sample batch with importance sampling weights."""
           probs = np.array(self.priorities) / sum(self.priorities)
           indices = np.random.choice(len(self.buffer), batch_size, p=probs)

           # Compute importance sampling weights
           weights = (len(self.buffer) * probs[indices]) ** (-beta)
           weights /= weights.max()

           batch = [self.buffer[i] for i in indices]
           return batch, weights, indices

       def update_priorities(self, indices, td_errors):
           """Update priorities after training."""
           for idx, td_error in zip(indices, td_errors):
               self.priorities[idx] = (abs(td_error) + 1e-5) ** self.alpha
   ```

3. **Cyclical Learning Rate Schedule**:
   ```python
   class OneCycleLR:
       def __init__(self, optimizer, max_lr, total_steps):
           self.optimizer = optimizer
           self.max_lr = max_lr
           self.total_steps = total_steps
           self.current_step = 0

       def step(self):
           """Update learning rate according to one-cycle policy."""
           if self.current_step < self.total_steps // 2:
               # Warm up
               lr = self.max_lr * self.current_step / (self.total_steps // 2)
           else:
               # Cool down
               progress = (self.current_step - self.total_steps // 2) / (self.total_steps // 2)
               lr = self.max_lr * (1 - progress)

           for param_group in self.optimizer.param_groups:
               param_group['lr'] = lr

           self.current_step += 1
   ```

4. **Gradient Clipping and Normalization**:
   ```python
   def train_step(self, batch):
       # Forward pass
       loss = self.compute_loss(batch)

       # Backward pass
       self.optimizer.zero_grad()
       loss.backward()

       # Gradient clipping
       torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

       # Gradient normalization
       total_norm = 0.0
       for p in self.model.parameters():
           if p.grad is not None:
               total_norm += p.grad.data.norm(2).item() ** 2
       total_norm = total_norm ** 0.5

       if total_norm > 10.0:  # Unusually large gradient
           self.logger.warning(f"Large gradient norm: {total_norm}")

       self.optimizer.step()
   ```

**Validation:**
- [ ] Auxiliary tasks don't hurt main task performance
- [ ] Prioritized replay improves sample efficiency
- [ ] Learning rate schedule improves convergence
- [ ] Gradient issues (exploding/vanishing) resolved

---

### 4.3: Endgame Database and Tablebase

**Goal:** Perfect play in endgame positions.

**Implementation Steps:**

1. **Retrograde analysis for simple endgames**:
   ```python
   class EndgameTablebase:
       def __init__(self):
           self.tablebase = {}  # position_hash -> (value, best_move)

       def build_tablebase(self, max_pieces=6):
           """Build perfect play database for positions with <= max_pieces."""
           # Start from terminal positions
           terminal_positions = self.generate_terminal_positions(max_pieces)

           for pos in terminal_positions:
               self.tablebase[hash(pos)] = self.get_terminal_value(pos)

           # Work backwards
           changed = True
           while changed:
               changed = False
               for pos in self.generate_positions(max_pieces):
                   if hash(pos) in self.tablebase:
                       continue

                   value, move = self.retrograde_solve(pos)
                   if value is not None:
                       self.tablebase[hash(pos)] = (value, move)
                       changed = True

       def probe(self, state):
           """Look up position in tablebase."""
           pos_hash = hash(state)
           return self.tablebase.get(pos_hash)
   ```

2. **Integrate with MCTS**:
   ```python
   def _evaluate_leaf(self, state):
       # Check tablebase first
       if self.tablebase:
           result = self.tablebase.probe(state)
           if result is not None:
               value, best_move = result
               # Return perfect policy
               policy = np.zeros(self.num_actions)
               policy[self.move_to_index(best_move)] = 1.0
               return policy, value

       # Fall back to neural network
       return self.network.predict(state)
   ```

**Note:** YINSH endgame tablebases are complex due to large state space. Consider starting with 4-ring positions only.

**Validation:**
- [ ] Tablebase correctly solves simple endgames
- [ ] Integration doesn't slow down MCTS for non-endgame positions
- [ ] Endgame play becomes perfect in tablebase positions
- [ ] Win rate improves in close games

---

### 4.4: Meta-Learning and Hyperparameter Optimization

**Goal:** Automatically tune hyperparameters for optimal performance.

**Implementation Steps:**

1. **Implement Bayesian Optimization**:
   ```python
   from sklearn.gaussian_process import GaussianProcessRegressor
   from sklearn.gaussian_process.kernels import RBF

   class BayesianHyperparameterOptimizer:
       def __init__(self, param_space):
           self.param_space = param_space
           self.observations = []
           self.gp = GaussianProcessRegressor(kernel=RBF())

       def suggest_params(self):
           """Suggest next hyperparameters to try."""
           if len(self.observations) < 5:
               # Random exploration
               return self.sample_random_params()

           # Fit GP on observations
           X = [obs['params'] for obs in self.observations]
           y = [obs['score'] for obs in self.observations]
           self.gp.fit(X, y)

           # Optimize acquisition function
           best_params = self.optimize_acquisition()
           return best_params

       def observe(self, params, score):
           """Record observation of params -> score."""
           self.observations.append({'params': params, 'score': score})
   ```

2. **Create hyperparameter tuning script**:
   ```python
   # scripts/tune_hyperparameters.py
   def tune_hyperparameters(budget=50):
       """Run Bayesian optimization for hyperparameters."""
       param_space = {
           'learning_rate': (1e-5, 1e-2),
           'c_puct': (0.5, 2.0),
           'num_simulations': (100, 1000),
           'heuristic_weight': (0.0, 0.8),
           'temperature': (0.5, 1.5),
       }

       optimizer = BayesianHyperparameterOptimizer(param_space)

       for trial in range(budget):
           # Get suggested params
           params = optimizer.suggest_params()

           # Train for 10 iterations with these params
           score = train_and_evaluate(params, iterations=10)

           # Record result
           optimizer.observe(params, score)

           print(f"Trial {trial}: score={score}, params={params}")

       # Return best params
       best = max(optimizer.observations, key=lambda x: x['score'])
       return best['params']
   ```

**Validation:**
- [ ] Bayesian optimizer finds better params than defaults
- [ ] Tuning completes in reasonable time (1-2 days on M2)
- [ ] Best params are reproducible
- [ ] Training with tuned params achieves higher ELO

---

## Phase 5: Evaluation and Benchmarking (Ongoing)

**Goal:** Measure progress towards superhuman play
**Continuous throughout all phases**

### 5.1: Comprehensive Benchmark Suite

**Create standardized benchmarks**:

1. **Skill Ladder**:
   ```python
   BENCHMARK_OPPONENTS = [
       {'name': 'Random', 'type': 'random'},
       {'name': 'Basic Heuristic', 'type': 'heuristic', 'depth': 1},
       {'name': 'Strong Heuristic', 'type': 'heuristic', 'depth': 3},
       {'name': 'MCTS-100', 'type': 'mcts', 'simulations': 100},
       {'name': 'MCTS-400', 'type': 'mcts', 'simulations': 400},
       {'name': 'MCTS-1600', 'type': 'mcts', 'simulations': 1600},
       {'name': 'Previous Best', 'type': 'checkpoint', 'path': 'best_model.pt'},
   ]
   ```

2. **Position Test Suite**:
   ```python
   # Create suite of tactical positions
   TACTICAL_POSITIONS = [
       {'fen': '...', 'best_move': '...', 'description': 'Fork opportunity'},
       {'fen': '...', 'best_move': '...', 'description': 'Run completion'},
       {'fen': '...', 'best_move': '...', 'description': 'Defensive block'},
       # ... 100+ positions
   ]

   def benchmark_tactical_suite(model):
       """Test model on tactical positions."""
       correct = 0
       for position in TACTICAL_POSITIONS:
           state = GameState.from_fen(position['fen'])
           predicted_move = model.select_move(state)
           if predicted_move == position['best_move']:
               correct += 1

       return correct / len(TACTICAL_POSITIONS)
   ```

3. **ELO Rating System**:
   ```python
   class ELORating:
       def __init__(self, k=32):
           self.k = k
           self.ratings = {}

       def expected_score(self, rating_a, rating_b):
           """Expected score for A vs B."""
           return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

       def update_rating(self, player, opponent, score):
           """Update rating after game."""
           rating = self.ratings.get(player, 1500)
           opp_rating = self.ratings.get(opponent, 1500)

           expected = self.expected_score(rating, opp_rating)
           new_rating = rating + self.k * (score - expected)

           self.ratings[player] = new_rating
           return new_rating
   ```

**Validation:**
- [ ] Benchmark suite runs automatically after each iteration
- [ ] Results logged and visualized
- [ ] ELO ratings tracked over time
- [ ] Clear progression towards superhuman play

---

### 5.2: Analysis Tools

**Implementation Steps:**

1. **Game Analysis Tool**:
   ```python
   # scripts/analyze_game.py
   def analyze_game(game_path):
       """Provide detailed analysis of a game."""
       game = load_game(game_path)

       # Analyze each move
       for move_idx, (state, move, policy, value) in enumerate(game):
           # Get best move from strongest model
           best_move, best_value = strongest_model.search(state, simulations=1600)

           # Compare
           if move != best_move:
               value_loss = best_value - value
               if value_loss > 0.1:
                   print(f"Move {move_idx}: Mistake (value loss: {value_loss:.2f})")
                   print(f"  Played: {move} (value: {value:.2f})")
                   print(f"  Best: {best_move} (value: {best_value:.2f})")
   ```

2. **Training Progress Dashboard**:
   ```python
   # scripts/create_progress_report.py
   def create_progress_report(log_dir):
       """Generate comprehensive progress report."""
       report = {
           'iterations': iteration_count,
           'training_time': total_time,
           'games_played': total_games,
           'current_elo': current_elo,
           'elo_progress': elo_over_time,
           'win_rates': win_rates_vs_benchmarks,
           'tactical_score': tactical_suite_score,
           'average_game_quality': avg_game_quality,
       }

       # Generate visualizations
       plot_elo_progress(report['elo_progress'])
       plot_win_rates(report['win_rates'])
       plot_loss_curves(log_dir)

       # Save report
       save_report(report, 'training_progress.html')
   ```

**Validation:**
- [ ] Can analyze any game and identify mistakes
- [ ] Progress report generated after each evaluation
- [ ] Clear metrics show improvement over time
- [ ] Can diagnose training issues from analysis

---

## Success Criteria and Milestones

### Phase 1 Success (Weeks 1-2)
- [ ] Network learns basic patterns (value predictions correlate with outcomes)
- [ ] Beats random player >60% of games
- [ ] Training runs stably without crashes
- [ ] Heuristics are integrated and providing guidance
- [ ] Sample reuse rate >100x

### Phase 2 Success (Weeks 3-4)
- [ ] Beats pure heuristic agent (depth 3) >55% of games
- [ ] ELO rating >400 vs random baseline
- [ ] Tournament evaluation runs efficiently (<10 min)
- [ ] Batched MCTS implemented and 5x+ faster
- [ ] Model promotion working correctly

### Phase 3 Success (Weeks 5-6)
- [ ] Training throughput >100 games/hour on M2
- [ ] Beats MCTS-100 (with heuristic) >55% of games
- [ ] ELO rating >600 vs random baseline
- [ ] CoreML acceleration working (if beneficial)
- [ ] All CPU cores utilized efficiently

### Phase 4 Success (Weeks 7-8)
- [ ] Beats MCTS-400 (with heuristic) >55% of games
- [ ] ELO rating >800 vs random baseline
- [ ] Tactical test suite score >90%
- [ ] Demonstrates superhuman tactical play in analysis
- [ ] Consistent, strategic game patterns

### Ultimate Success (Superhuman Play)
- [ ] Beats best available YINSH engines
- [ ] Tactical test suite score >95%
- [ ] ELO rating >1000 vs random baseline
- [ ] Game analysis shows consistently brilliant moves
- [ ] Can beat strong human players

---

## Risk Mitigation

### Technical Risks

1. **Risk: Training diverges or collapses**
   - Mitigation: Careful learning rate tuning, gradient clipping, regular checkpointing
   - Detection: Monitor loss curves, value prediction accuracy
   - Recovery: Revert to last stable checkpoint, reduce learning rate

2. **Risk: Heuristic integration degrades performance**
   - Mitigation: Start with high heuristic weight, decay slowly
   - Detection: Compare vs pure neural baseline
   - Recovery: Adjust decay schedule or disable heuristics

3. **Risk: M2 too slow for effective training**
   - Mitigation: Aggressive optimization (batching, caching, CoreML)
   - Detection: Measure games/hour, compare to targets
   - Recovery: Consider cloud GPU for compute-intensive phases

4. **Risk: State space too large for superhuman play**
   - Mitigation: Focus on tactical proficiency, endgame tablebases
   - Detection: Monitor strategic coherence in games
   - Recovery: Add domain-specific knowledge (opening book, endgame DB)

### Process Risks

1. **Risk: Scope creep, never-ending optimization**
   - Mitigation: Strict phase boundaries, success criteria
   - Detection: Project timeline tracking
   - Recovery: Prioritize ruthlessly, cut non-essential features

2. **Risk: Lack of validation during development**
   - Mitigation: Comprehensive test suite, continuous benchmarking
   - Detection: Unexpected behavior in training
   - Recovery: Add tests retroactively, bisect to find regression

3. **Risk: Configuration complexity becomes unmaintainable**
   - Mitigation: Well-structured config schema, documentation
   - Detection: Difficulty reproducing results
   - Recovery: Simplify config, reduce options

---

## Resource Requirements

### Compute
- **M2 Mac Mini**: Primary training hardware
  - 8 cores for parallel self-play
  - Neural Engine for inference acceleration (if CoreML beneficial)
  - 8-24GB RAM (16GB+ recommended)
- **Estimated training time**: 6-8 weeks continuous (can pause/resume)
- **Storage**: 50-100GB for checkpoints, replay buffer, logs

### Development Time
- **Phase 1**: 20-30 hours (critical fixes)
- **Phase 2**: 30-40 hours (AlphaZero implementation)
- **Phase 3**: 20-30 hours (M2 optimization)
- **Phase 4**: 30-40 hours (advanced techniques)
- **Total**: 100-140 hours development + compute time

### Dependencies
- PyTorch 2.0+
- NumPy, SciPy
- TensorBoard (logging)
- CoreMLTools (M2 acceleration)
- Pytest (testing)
- All in requirements.txt

---

## Commit Strategy

### Branch Organization
```
main (production-ready code)
├── feature/unified-mcts (Phase 1.1)
├── feature/heuristic-integration (Phase 1.2)
├── feature/training-improvements (Phase 1.3-1.5)
├── feature/batched-mcts (Phase 2.1)
├── feature/tournament-eval (Phase 2.2)
├── feature/m2-optimization (Phase 3)
└── feature/advanced-techniques (Phase 4)
```

### Commit Guidelines
- **Atomic commits**: Each commit should be a single logical change
- **Descriptive messages**: "Add batched MCTS evaluation with virtual losses"
- **Reference issues**: "Fix #123: Heuristic weight not decaying properly"
- **Test coverage**: All commits should pass tests

### Checkpoint Commits (After Each Phase)
- Commit this planning doc first: `docs: Add comprehensive training refactor plan`
- After Phase 1: `feat: Implement critical architecture fixes (Phase 1)`
- After Phase 2: `feat: Complete AlphaZero training loop (Phase 2)`
- After Phase 3: `perf: Optimize for M2 performance (Phase 3)`
- After Phase 4: `feat: Add advanced techniques for superhuman play (Phase 4)`

---

## Next Steps

1. **Immediate**: Commit this planning document
   ```bash
   git add TRAINING_REFACTOR_PLAN.md
   git commit -m "docs: Add comprehensive training refactor plan for superhuman YinshML"
   git push origin heuristic_seeding
   ```

2. **Week 1**: Begin Phase 1.1 (Unify MCTS implementations)
   - Create new branch: `git checkout -b feature/unified-mcts`
   - Implement UnifiedMCTS class
   - Add tests
   - Update SelfPlay to use unified MCTS

3. **Week 2**: Complete Phase 1 (all critical fixes)
   - Merge unified MCTS
   - Integrate heuristics
   - Increase training intensity
   - Remove hardcoded hyperparameters
   - Validate network is learning

4. **Weeks 3-4**: Phase 2 (AlphaZero implementation)

5. **Weeks 5-6**: Phase 3 (M2 optimization)

6. **Weeks 7-8**: Phase 4 (Advanced techniques)

---

## References

1. **AlphaGo Zero Paper**: Silver et al., "Mastering the game of Go without human knowledge" (2017)
2. **AlphaZero Paper**: Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play" (2018)
3. **MuZero Paper**: Schrittwieser et al., "Mastering Atari, Go, chess and shogi by planning with a learned model" (2020)
4. **EfficientZero Paper**: Ye et al., "Mastering Atari Games with Limited Data" (2021)
5. **YINSH Game Rules**: https://www.gipf.com/yinsh/rules/rules.html

---

## Appendix: Quick Reference Commands

### Start Training (After Phase 1)
```bash
# Configure
vim configs/superhuman_training.yaml

# Launch training
python scripts/run_training.py --config configs/superhuman_training.yaml

# Monitor (separate terminal)
tensorboard --logdir logs/tensorboard
```

### Evaluate Model
```bash
# Quick evaluation vs benchmarks
python scripts/evaluate_model.py --model checkpoints/iteration_100.pt

# Tournament vs all opponents
python scripts/run_tournament.py --model checkpoints/iteration_100.pt --rounds 100

# Analyze specific game
python scripts/analyze_game.py --game data/game_12345.json
```

### Debugging
```bash
# Profile self-play
python -m cProfile -o profile.stats scripts/run_training.py --iterations 1

# Memory profiling
python -m memory_profiler scripts/run_training.py --iterations 1

# Check config validity
python scripts/validate_config.py configs/superhuman_training.yaml
```

### Testing
```bash
# Run all tests
pytest

# Run specific phase tests
pytest tests/test_unified_mcts.py -v
pytest tests/test_heuristic_integration.py -v
pytest tests/test_batched_inference.py -v

# Run with coverage
pytest --cov=yinsh_ml --cov-report=html
```

---

**End of Training Refactor Plan**

*This document is a living plan and will be updated as implementation progresses and new insights emerge.*
