# Roadmap to AlphaZero-Style Self-Play for YINSH

## 🎯 Strategic Goal

Build an AlphaZero-style YINSH AI that learns from pure self-play to achieve expert-level performance (80%+ win rate vs current baseline).

---

## 📊 Current State: What We Have

### **✅ Completed (100K Game Analysis)**

| Component | Status | Quality |
|-----------|--------|---------|
| Feature Engineering | ✅ Done | 15 features validated |
| Heuristic Evaluator | ✅ Done | 52% accuracy, 284 evals/sec |
| Training Data | ✅ Done | 100K games, 9M positions |
| Feature Importance | ✅ Done | Evidence-based rankings |
| Phase Analysis | ✅ Done | Early/Mid/Late insights |
| Random Forest Model | ✅ Done | 55.1% accuracy |
| Parquet Storage | ✅ Done | 344MB efficient storage |

### **📈 Current Performance Baseline**

- **Heuristic Function:** 52% accuracy (2% above random)
- **Random Forest:** 55.1% accuracy (5% above random)
- **Feature Set:** 7 core features + 8 secondary features
- **Training Examples:** 8,968,580 board positions

---

## 🗺️ The AlphaZero Journey: 5 Phases

### **Overview**

```
Phase 1: Enhanced Heuristic AI (Weeks 1-2)
    ↓
Phase 2: Neural Network Foundation (Weeks 3-6)
    ↓
Phase 3: MCTS Integration (Weeks 7-10)
    ↓
Phase 4: Self-Play Loop (Weeks 11-16)
    ↓
Phase 5: Scaling & Refinement (Weeks 17-24)
```

**Total Timeline: ~6 months to strong AlphaZero-style AI**

---

## 📋 Phase 1: Enhanced Heuristic AI (Weeks 1-2)

**Goal:** Maximize the utility of current heuristic before moving to neural networks

### **Task 1.1: Implement Tree Search with Heuristic**

**Deliverables:**
- [ ] Minimax search (alpha-beta pruning, depth 3-4)
- [ ] Iterative deepening framework
- [ ] Move ordering using heuristic
- [ ] Transposition table for position caching

**Expected Improvement:** 52% → 60%

**Code Location:** `yinsh_ml/search/minimax.py`

**Key Implementation:**
```python
class MinimaxSearcher:
    def __init__(self, heuristic_evaluator, max_depth=4):
        self.evaluator = heuristic_evaluator
        self.max_depth = max_depth
        self.transposition_table = {}
    
    def search(self, game_state):
        # Alpha-beta with iterative deepening
        # Use heuristic for move ordering
        # Cache positions in transposition table
        pass
```

**Validation:**
- Play 1000 games: Minimax vs Random
- Target: 60%+ win rate
- Performance: <5 seconds/move average

---

### **Task 1.2: Phase-Aware Evaluation**

**Deliverables:**
- [ ] Dynamic weight adjustment based on turn number
- [ ] Early/Mid/Late game specialized evaluators
- [ ] Transition smoothing between phases

**Expected Improvement:** 60% → 62%

**Code Location:** `yinsh_ml/evaluation/phase_aware.py`

**Key Implementation:**
```python
class PhaseAwareEvaluator:
    def evaluate(self, game_state):
        turn = game_state.turn_number
        
        if turn <= 15:  # Early
            return self.early_game_eval(game_state)
        elif turn <= 35:  # Mid
            return self.mid_game_eval(game_state)
        else:  # Late
            return self.late_game_eval(game_state)
```

**Validation:**
- Compare phase-aware vs static weights
- Target: 2-3% improvement in late game positions

---

### **Task 1.3: Improved Feature Engineering**

**Deliverables:**
- [ ] Threat detection (run-in-2, run-in-3)
- [ ] Defensive features (opponent threats)
- [ ] Tempo features (initiative, forcing moves)
- [ ] Pattern features (fork positions, trapped rings)

**Expected Improvement:** 62% → 65%

**New Features to Add:**
1. `immediate_run_threats` - Runs completable in 1 move
2. `opponent_threats` - Opponent's run opportunities
3. `forcing_moves` - Moves that limit opponent options
4. `ring_coordination` - Rings working together
5. `marker_efficiency` - Markers placed optimally

**Code Location:** `yinsh_ml/game/features.py` (extend existing)

**Validation:**
- Train new Random Forest with expanded feature set
- Target: 58%+ accuracy (up from 55.1%)

---

### **Phase 1 Milestones**

- [ ] **Week 1:** Minimax + Alpha-Beta implementation
- [ ] **Week 2:** Phase-aware evaluation + enhanced features
- [ ] **Validation:** 65% win rate vs random baseline
- [ ] **Deliverable:** `MinimaxSearcher` ready for integration

**Success Criteria:**
- ✅ 65%+ win rate vs random
- ✅ <5 seconds per move
- ✅ Self-play generates higher quality games than Phase 0

---

## 📋 Phase 2: Neural Network Foundation (Weeks 3-6)

**Goal:** Replace hand-crafted heuristic with learned neural network

### **Task 2.1: Policy Network Architecture**

**Deliverables:**
- [ ] Neural network for move prediction
- [ ] Input: Board state encoding
- [ ] Output: Move probabilities (policy head)
- [ ] Training on 100K games

**Architecture:**
```python
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: Board representation (11x11x channels)
        # Channels: rings, markers, legal moves, etc.
        
        # Convolutional layers for spatial features
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # Residual blocks (like AlphaZero)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(5)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2*11*11, move_space_size)
```

**Code Location:** `yinsh_ml/neural/policy_network.py`

**Expected Performance:** 60% accuracy predicting expert moves

---

### **Task 2.2: Value Network Architecture**

**Deliverables:**
- [ ] Neural network for position evaluation
- [ ] Input: Board state encoding
- [ ] Output: Win probability (value head)
- [ ] Training on 100K game outcomes

**Architecture:**
```python
class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared convolutional layers with policy network
        # ...
        
        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(11*11, 64)
        self.value_fc2 = nn.Linear(64, 1)  # Output: win probability
```

**Code Location:** `yinsh_ml/neural/value_network.py`

**Expected Performance:** 62% accuracy predicting game outcomes

---

### **Task 2.3: Combined Policy-Value Network (AlphaZero Style)**

**Deliverables:**
- [ ] Single network with dual heads
- [ ] Shared representation layers
- [ ] Multi-task loss function
- [ ] Efficient inference

**Architecture:**
```python
class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared trunk (convolutional + residual)
        self.trunk = SharedTrunk()
        
        # Policy head
        self.policy_head = PolicyHead()
        
        # Value head  
        self.value_head = ValueHead()
    
    def forward(self, board_state):
        features = self.trunk(board_state)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value
```

**Code Location:** `yinsh_ml/neural/alphazero_network.py`

---

### **Task 2.4: Board State Encoding**

**Deliverables:**
- [ ] Efficient board → tensor conversion
- [ ] Multi-channel representation
- [ ] Symmetry augmentation
- [ ] Fast batch processing

**Encoding Scheme:**
```python
# Shape: (batch, channels, height, width) = (N, 20, 11, 11)

Channels:
 0-1:   Player 1/2 rings (binary)
 2-3:   Player 1/2 markers (binary)
 4-5:   Player 1/2 completed runs (count)
 6-7:   Legal ring placements (binary)
 8-9:   Legal marker placements (binary)
10-11:  Potential runs (count per cell)
12-13:  Ring centrality scores
14-15:  Marker density maps
16:     Turn number (normalized)
17:     Current player (binary)
18:     P1 score (normalized)
19:     P2 score (normalized)
```

**Code Location:** `yinsh_ml/neural/encoding.py`

---

### **Task 2.5: Supervised Learning Training**

**Deliverables:**
- [ ] Training pipeline using 100K games
- [ ] Data augmentation (rotations, reflections)
- [ ] Validation split (80/10/10)
- [ ] Learning rate scheduling
- [ ] Model checkpointing

**Training Configuration:**
```python
config = {
    'batch_size': 512,
    'learning_rate': 0.001,
    'epochs': 100,
    'optimizer': 'Adam',
    'loss_policy': 'cross_entropy',
    'loss_value': 'MSE',
    'loss_weight_policy': 1.0,
    'loss_weight_value': 1.0,
}
```

**Code Location:** `yinsh_ml/training/supervised_trainer.py`

**Expected Performance:**
- Policy accuracy: 60%
- Value accuracy: 62%
- Combined network: 60% game playing strength

---

### **Phase 2 Milestones**

- [ ] **Week 3:** Board encoding + data pipeline
- [ ] **Week 4:** Network architecture + initial training
- [ ] **Week 5:** Hyperparameter tuning + validation
- [ ] **Week 6:** Integration + testing
- [ ] **Validation:** 60% win rate using neural network alone
- [ ] **Deliverable:** `PolicyValueNetwork` trained on 100K games

**Success Criteria:**
- ✅ 60%+ win rate vs random (neural network only)
- ✅ <100ms inference per position
- ✅ Better than heuristic (52%) when used without search

---

## 📋 Phase 3: MCTS Integration (Weeks 7-10)

**Goal:** Combine neural network with Monte Carlo Tree Search

### **Task 3.1: Basic MCTS Implementation**

**Deliverables:**
- [ ] UCT tree search
- [ ] Neural network for tree evaluation
- [ ] Simulation rollouts
- [ ] Move selection (temperature)

**Architecture:**
```python
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0
        self.prior_prob = 0  # From policy network

class MCTS:
    def __init__(self, network, num_simulations=800):
        self.network = network
        self.num_simulations = num_simulations
    
    def search(self, root_state):
        root = MCTSNode(root_state)
        
        for _ in range(self.num_simulations):
            # Selection (UCT)
            node = self.select(root)
            
            # Expansion
            if not node.is_terminal():
                node = self.expand(node)
            
            # Evaluation (neural network)
            value = self.evaluate(node)
            
            # Backpropagation
            self.backpropagate(node, value)
        
        return self.best_action(root)
```

**Code Location:** `yinsh_ml/search/mcts.py`

---

### **Task 3.2: AlphaZero-Style MCTS**

**Deliverables:**
- [ ] Policy-guided tree search
- [ ] Virtual loss for parallelization
- [ ] Dirichlet noise for exploration
- [ ] Temperature-based move selection

**Key Enhancements:**
```python
# UCT with policy prior
def uct_score(node, child):
    q_value = child.total_value / (child.visit_count + 1e-8)
    u_value = C_PUCT * child.prior_prob * sqrt(node.visit_count) / (1 + child.visit_count)
    return q_value + u_value

# Move selection with temperature
def select_move(root, temperature=1.0):
    if temperature == 0:
        # Deterministic (argmax)
        return max(root.children.items(), key=lambda x: x[1].visit_count)
    else:
        # Stochastic sampling
        visit_counts = [child.visit_count for child in root.children.values()]
        probs = softmax([v**(1/temperature) for v in visit_counts])
        return sample(root.children.keys(), probs)
```

**Code Location:** `yinsh_ml/search/alphazero_mcts.py`

---

### **Task 3.3: MCTS Optimization**

**Deliverables:**
- [ ] Parallel tree search (8+ threads)
- [ ] GPU batch evaluation
- [ ] Tree reuse between moves
- [ ] Position caching

**Performance Targets:**
- 800 simulations in <2 seconds
- GPU utilization >80%
- Tree reuse saves 30%+ simulations

**Code Location:** `yinsh_ml/search/parallel_mcts.py`

---

### **Task 3.4: Integration & Testing**

**Deliverables:**
- [ ] Neural Network + MCTS agent
- [ ] Performance benchmarking
- [ ] Comparison vs Phase 1 (Minimax)
- [ ] Arena for agent tournaments

**Validation:**
- [ ] NN+MCTS vs Random: Target 70%+
- [ ] NN+MCTS vs Heuristic+Minimax: Target 55%+
- [ ] Self-play game quality assessment

**Code Location:** `yinsh_ml/agents/alphazero_agent.py`

---

### **Phase 3 Milestones**

- [ ] **Week 7:** Basic MCTS + neural network integration
- [ ] **Week 8:** AlphaZero-style enhancements
- [ ] **Week 9:** Parallelization + optimization
- [ ] **Week 10:** Testing + benchmarking
- [ ] **Validation:** 70% win rate vs random
- [ ] **Deliverable:** AlphaZeroAgent ready for self-play

**Success Criteria:**
- ✅ 70%+ win rate vs random
- ✅ 800 simulations in <2 seconds
- ✅ Beats Phase 1 Minimax agent (65%) by 10%+

---

## 📋 Phase 4: Self-Play Loop (Weeks 11-16)

**Goal:** Implement iterative self-play training (the core of AlphaZero)

### **Task 4.1: Self-Play Data Generation**

**Deliverables:**
- [ ] Self-play game generator
- [ ] Parallel game execution
- [ ] Training data extraction (state, policy, value)
- [ ] Efficient storage (Parquet)

**Architecture:**
```python
class SelfPlayWorker:
    def __init__(self, network, mcts_config):
        self.agent = AlphaZeroAgent(network, mcts_config)
    
    def generate_game(self):
        game = YinshGame()
        training_examples = []
        
        while not game.is_terminal():
            # MCTS search
            mcts_policy = self.agent.get_policy(game.state)
            
            # Store training example
            training_examples.append({
                'state': game.state.encode(),
                'policy': mcts_policy,  # MCTS visit counts
                'value': None  # Filled after game ends
            })
            
            # Make move
            action = sample(mcts_policy)
            game.make_move(action)
        
        # Backfill game outcome
        outcome = game.get_winner()
        for example in training_examples:
            example['value'] = outcome
        
        return training_examples
```

**Code Location:** `yinsh_ml/selfplay/generator.py`

---

### **Task 4.2: Training Pipeline**

**Deliverables:**
- [ ] Replay buffer (recent N games)
- [ ] Mini-batch sampling
- [ ] Network training on self-play data
- [ ] Checkpoint management

**Training Loop:**
```python
# AlphaZero training loop
for iteration in range(num_iterations):
    # 1. Generate self-play games
    print(f"Iteration {iteration}: Generating {games_per_iteration} games...")
    new_games = generate_self_play_games(
        network=current_network,
        num_games=games_per_iteration
    )
    
    # 2. Add to replay buffer
    replay_buffer.add(new_games)
    
    # 3. Train network on replay buffer
    print(f"Training network on {len(replay_buffer)} examples...")
    new_network = train_network(
        old_network=current_network,
        training_data=replay_buffer.sample(batch_size),
        epochs=epochs_per_iteration
    )
    
    # 4. Evaluate new network
    print(f"Evaluating new network...")
    win_rate = arena(new_network, current_network, num_games=100)
    
    # 5. Accept or reject
    if win_rate > 0.55:  # New network must win 55%+
        print(f"✅ Accepting new network (win rate: {win_rate})")
        current_network = new_network
    else:
        print(f"❌ Rejecting new network (win rate: {win_rate})")
```

**Code Location:** `yinsh_ml/training/selfplay_trainer.py`

---

### **Task 4.3: Model Evaluation & Selection**

**Deliverables:**
- [ ] Arena for pitting models against each other
- [ ] Elo rating system
- [ ] Acceptance threshold (55% win rate)
- [ ] Model versioning

**Evaluation System:**
```python
class ModelArena:
    def __init__(self):
        self.elo_ratings = {}
    
    def evaluate(self, new_model, current_best, num_games=100):
        # Play num_games between models
        results = []
        for game_num in range(num_games):
            winner = self.play_game(new_model, current_best)
            results.append(winner)
        
        win_rate = sum(r == 'new' for r in results) / num_games
        
        # Update Elo
        self.update_elo(new_model, current_best, win_rate)
        
        return win_rate
```

**Code Location:** `yinsh_ml/evaluation/arena.py`

---

### **Task 4.4: Monitoring & Visualization**

**Deliverables:**
- [ ] TensorBoard integration
- [ ] Training metrics dashboard
- [ ] Game quality metrics
- [ ] Performance tracking

**Metrics to Track:**
- Training loss (policy + value)
- Self-play win rate vs baseline
- Average game length
- MCTS evaluation correlation
- Elo rating progression
- Model size & inference speed

**Code Location:** `yinsh_ml/monitoring/dashboard.py`

---

### **Task 4.5: First Self-Play Iteration**

**Deliverables:**
- [ ] Generate 10,000 self-play games
- [ ] Train on combined dataset (100K + 10K)
- [ ] Evaluate improvement
- [ ] Iterate if successful

**Configuration:**
```python
iteration_0_config = {
    'games_per_iteration': 10000,
    'mcts_simulations': 800,
    'training_epochs': 10,
    'batch_size': 512,
    'replay_buffer_size': 50000,
    'acceptance_threshold': 0.55,
}
```

**Expected Improvement:** 70% → 75%

---

### **Phase 4 Milestones**

- [ ] **Week 11:** Self-play generator + infrastructure
- [ ] **Week 12:** Training pipeline + replay buffer
- [ ] **Week 13:** Arena + evaluation system
- [ ] **Week 14:** Monitoring + first iteration
- [ ] **Week 15-16:** Iteration 2-3 + tuning
- [ ] **Validation:** 75% win rate vs random
- [ ] **Deliverable:** Self-improving AI system

**Success Criteria:**
- ✅ Self-play loop runs end-to-end
- ✅ New iterations consistently improve (55%+ acceptance rate)
- ✅ 75%+ win rate vs random
- ✅ Network surpasses supervised learning baseline

---

## 📋 Phase 5: Scaling & Refinement (Weeks 17-24)

**Goal:** Scale to expert-level performance through extensive self-play

### **Task 5.1: Computational Scaling**

**Deliverables:**
- [ ] Multi-GPU training
- [ ] Distributed self-play (multiple machines)
- [ ] Cloud infrastructure setup
- [ ] Cost optimization

**Infrastructure:**
- **Self-play:** 8+ CPU machines (parallel game generation)
- **Training:** 1-2 GPU machines (A100 or V100)
- **Evaluation:** 1 CPU machine (arena tournaments)

**Expected Scale:**
- 50,000 games per day
- 10 iterations per week
- 500K total games by end of phase

---

### **Task 5.2: Hyperparameter Optimization**

**Deliverables:**
- [ ] MCTS simulation count tuning
- [ ] Temperature schedule optimization
- [ ] Learning rate scheduling
- [ ] Network architecture search

**Key Parameters to Tune:**
- `c_puct`: Exploration constant (default: 1.0)
- `dirichlet_alpha`: Root exploration noise (default: 0.3)
- `temperature_threshold`: Move until temp=0 (default: turn 30)
- `num_res_blocks`: Network depth (default: 5-10)

---

### **Task 5.3: Advanced Techniques**

**Deliverables:**
- [ ] Auxiliary tasks (predicting features)
- [ ] Curriculum learning (start with endgames)
- [ ] Ensemble methods
- [ ] Opening book generation

**Auxiliary Tasks:**
```python
# Additional network heads for richer learning
class EnhancedNetwork(PolicyValueNetwork):
    def __init__(self):
        super().__init__()
        
        # Additional prediction heads
        self.run_predictor = nn.Linear(64, 1)  # Predict num runs
        self.game_length_predictor = nn.Linear(64, 1)  # Predict game length
        
    def forward(self, state):
        policy, value = super().forward(state)
        features = self.trunk(state)
        
        run_pred = self.run_predictor(features)
        length_pred = self.game_length_predictor(features)
        
        return policy, value, run_pred, length_pred
```

---

### **Task 5.4: Performance Benchmarking**

**Deliverables:**
- [ ] Compare vs baseline agents (random, heuristic, minimax)
- [ ] Compare vs previous iterations
- [ ] Human player evaluation (if available)
- [ ] Identify strength/weakness patterns

**Benchmark Suite:**
- vs Random: Target 95%+
- vs Heuristic: Target 90%+
- vs Minimax (depth 4): Target 85%+
- vs Iteration 0: Target 80%+

---

### **Task 5.5: Model Compression & Deployment**

**Deliverables:**
- [ ] Model quantization (FP32 → FP16 or INT8)
- [ ] Knowledge distillation to smaller network
- [ ] ONNX export for production
- [ ] Web deployment (WASM or API)

**Deployment Targets:**
- Mobile: <50MB model, <100ms per move
- Web: JavaScript/WASM inference
- API: FastAPI server with GPU inference

---

### **Phase 5 Milestones**

- [ ] **Week 17-18:** Infrastructure scaling
- [ ] **Week 19-20:** Hyperparameter optimization
- [ ] **Week 21-22:** Advanced techniques + iteration 10+
- [ ] **Week 23:** Benchmarking + analysis
- [ ] **Week 24:** Compression + deployment
- [ ] **Validation:** 85%+ win rate vs Phase 3 baseline
- [ ] **Deliverable:** Production-ready AlphaZero YINSH AI

**Success Criteria:**
- ✅ 85%+ win rate vs initial AlphaZero agent
- ✅ 95%+ win rate vs random
- ✅ Clear superhuman patterns visible in play
- ✅ Model ready for deployment

---

## 📊 Success Metrics by Phase

| Phase | Win Rate vs Random | Key Milestone |
|-------|-------------------|---------------|
| **Current (Phase 0)** | 52% | Heuristic baseline |
| **Phase 1** | 65% | Tree search + features |
| **Phase 2** | 60% | Neural network trained |
| **Phase 3** | 70% | MCTS integration |
| **Phase 4** | 75% | Self-play iteration 3 |
| **Phase 5** | 85%+ | Scaled self-play (20+ iterations) |
| **Target** | 90%+ | Expert-level play |

---

## 🛠️ Technical Stack

### **Core Libraries**

```python
# Deep Learning
pytorch >= 2.0
tensorboard
numpy
scipy

# Game Engine
python >= 3.9
dataclasses
typing

# Data Management
pandas
pyarrow  # For parquet
h5py     # Alternative storage

# Utilities
tqdm     # Progress bars
click    # CLI
pyyaml   # Config files
joblib   # Model serialization

# Distributed Computing (Phase 5)
ray      # Distributed self-play
```

### **Hardware Requirements**

**Phase 1-2 (Development):**
- CPU: 8+ cores
- RAM: 32GB
- GPU: Optional (speeds up Phase 2)

**Phase 3-4 (Self-Play):**
- CPU: 16+ cores for parallel MCTS
- RAM: 64GB
- GPU: 1x RTX 3090 or A100 (16GB+ VRAM)

**Phase 5 (Scaling):**
- Multiple machines or cloud setup
- 2-4 GPUs for training
- 8+ CPU machines for self-play generation

---

## 📁 Project Structure

```
YinshML/
├── yinsh_ml/
│   ├── game/              # Game logic
│   │   ├── state.py
│   │   ├── moves.py
│   │   ├── features.py   # Feature extraction
│   │   └── encoding.py   # Board encoding
│   │
│   ├── search/            # Search algorithms
│   │   ├── minimax.py    # Phase 1
│   │   ├── mcts.py       # Phase 3
│   │   └── alphazero_mcts.py
│   │
│   ├── neural/            # Neural networks
│   │   ├── policy_network.py
│   │   ├── value_network.py
│   │   ├── alphazero_network.py
│   │   └── encoding.py
│   │
│   ├── training/          # Training infrastructure
│   │   ├── supervised_trainer.py    # Phase 2
│   │   ├── selfplay_trainer.py      # Phase 4
│   │   └── replay_buffer.py
│   │
│   ├── selfplay/          # Self-play generation
│   │   ├── generator.py
│   │   ├── worker.py
│   │   └── coordinator.py
│   │
│   ├── evaluation/        # Model evaluation
│   │   ├── arena.py
│   │   ├── elo.py
│   │   └── benchmarks.py
│   │
│   ├── agents/            # Agent implementations
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   │   ├── minimax_agent.py
│   │   └── alphazero_agent.py
│   │
│   └── monitoring/        # Monitoring & visualization
│       ├── dashboard.py
│       └── metrics.py
│
├── configs/               # Configuration files
│   ├── minimax.yaml
│   ├── neural_network.yaml
│   ├── mcts.yaml
│   └── selfplay.yaml
│
├── scripts/               # Utility scripts
│   ├── train_supervised.py
│   ├── run_selfplay.py
│   ├── evaluate_models.py
│   └── compress_model.py
│
├── analysis_output/       # From Phase 0 (100K games)
│   └── heuristic_evaluator_model.pkl
│
├── models/                # Saved models
│   ├── iteration_000/
│   ├── iteration_001/
│   └── best_model.pt
│
└── data/
    ├── supervised/        # 100K games (Phase 0)
    ├── selfplay/          # Self-play games (Phase 4+)
    └── evaluation/        # Benchmark results
```

---

## ⚠️ Key Risks & Mitigation

### **Risk 1: Training Instability**

**Problem:** Self-play training can diverge or collapse

**Mitigation:**
- Start with strong supervised baseline (Phase 2)
- Use acceptance threshold (55% win rate)
- Maintain replay buffer of diverse games
- Monitor loss curves carefully
- Keep checkpoints of stable iterations

### **Risk 2: Computational Cost**

**Problem:** Self-play is expensive (time + $$$)

**Mitigation:**
- Start with smaller scale (10K games/iteration)
- Use cloud spot instances for cost savings
- Optimize MCTS simulations (find minimum needed)
- Parallelize aggressively
- Consider curriculum learning (endgames first)

### **Risk 3: Overfitting to Self-Play**

**Problem:** AI might learn suboptimal strategies

**Mitigation:**
- Maintain diversity in self-play (temperature, exploration)
- Periodic evaluation vs different baselines
- Auxiliary learning tasks
- Regular benchmarking vs external agents

### **Risk 4: Feature Engineering Ceiling**

**Problem:** Current features might limit learning

**Mitigation:**
- Let neural network learn representations end-to-end
- Minimal pre-processing (raw board state)
- Trust the network to discover features
- Add auxiliary tasks to guide representation learning

---

## 🎯 Decision Points

### **After Phase 1:**

**Decision:** Is 65% win rate achieved with Minimax?

- ✅ **Yes:** Proceed to Phase 2
- ❌ **No:** 
  - Add more features
  - Increase search depth
  - Improve move ordering

### **After Phase 2:**

**Decision:** Does neural network match heuristic (52%)?

- ✅ **Yes:** Proceed to Phase 3
- ❌ **No:**
  - Increase network capacity
  - More training epochs
  - Better data augmentation
  - Consider collecting more training data

### **After Phase 3:**

**Decision:** Does MCTS+NN beat Minimax+Heuristic?

- ✅ **Yes (70%+):** Proceed to Phase 4 self-play
- ❌ **No:**
  - Increase MCTS simulations
  - Tune c_puct parameter
  - Improve network evaluation quality
  - Consider hybrid approach

### **During Phase 4:**

**Decision:** Are iterations consistently improving?

- ✅ **Yes (55%+ acceptance):** Continue iterations
- ❌ **No:**
  - Adjust acceptance threshold
  - Increase games per iteration
  - Tune training hyperparameters
  - Check for training instability

---

## 📈 Expected Timeline

### **Optimistic Scenario (6 months)**

- Phase 1: 2 weeks
- Phase 2: 4 weeks
- Phase 3: 4 weeks
- Phase 4: 6 weeks
- Phase 5: 8 weeks
- **Total: 24 weeks**

### **Realistic Scenario (9 months)**

- Phase 1: 3 weeks (debugging + tuning)
- Phase 2: 6 weeks (network architecture exploration)
- Phase 3: 5 weeks (MCTS optimization)
- Phase 4: 10 weeks (5-7 iterations + infrastructure)
- Phase 5: 12 weeks (scaling + refinement)
- **Total: 36 weeks**

### **Conservative Scenario (12 months)**

- Add 50% buffer for unforeseen issues
- More thorough experimentation at each phase
- Higher quality standards for progression
- **Total: 48-52 weeks**

---

## 🎓 Learning Resources

### **AlphaZero Papers**

1. **AlphaGo Zero:** Mastering the game of Go without human knowledge
2. **AlphaZero:** A general reinforcement learning algorithm
3. **MuZero:** Mastering without rules

### **Implementation Guides**

1. **Alpha Zero General:** GitHub implementation reference
2. **Leela Chess Zero:** Community implementation of AlphaZero for chess
3. **KataGo:** AlphaZero for Go with improvements

### **Technical Deep Dives**

1. MCTS survey papers
2. Neural network architectures for board games
3. Self-play training stability techniques

---

## ✅ Next Immediate Actions

### **This Week:**

1. **Set up Phase 1 infrastructure**
   - [ ] Create `yinsh_ml/search/minimax.py`
   - [ ] Implement basic minimax with alpha-beta
   - [ ] Load heuristic from `analysis_output/heuristic_evaluator_model.pkl`

2. **Baseline testing**
   - [ ] Test Minimax depth 1,2,3,4 vs Random
   - [ ] Measure performance (time per move)
   - [ ] Establish current win rate

3. **Plan Phase 2**
   - [ ] Design board encoding scheme
   - [ ] Sketch network architecture
   - [ ] Set up PyTorch environment

### **This Month:**

1. Complete Phase 1 (Enhanced Heuristic AI)
2. Begin Phase 2 (Neural Network Foundation)
3. Generate first supervised learning results

---

## 🎉 End Goal

**A production-ready AlphaZero-style YINSH AI that:**

✅ Achieves 85%+ win rate vs Phase 3 baseline  
✅ Demonstrates clear strategic understanding  
✅ Discovers novel winning patterns  
✅ Runs efficiently on consumer hardware  
✅ Can be deployed as web app or API  

**With full self-play training infrastructure for continued improvement!**

---

*This is a living document. Update milestones and timelines as you progress through each phase.*

*Last Updated: October 2025*
*Current Phase: 0 (100K Game Analysis Complete)*
*Next Phase: 1 (Enhanced Heuristic AI)*

