# Architectural Enhancement Roadmap

**Date:** June 30, 2025  
**Based on:** value_recovery_20250630 analysis  
**Priority Framework:** Impact × Feasibility ÷ Risk × Resources  

---

## Executive Priority Ranking

### **TIER 1: Critical Path (Immediate - Next 2 Weeks)**

#### **1. Control Model Implementation (HIGHEST PRIORITY)**
**Impact:** ★★★★★ | **Feasibility:** ★★★★★ | **Risk:** ★ | **Resources:** ★★

**Problem:** No stable baseline for Elo comparisons across experiments  
**Evidence:** "We observed inconsistent tournament baselines making cross-experiment comparison unreliable"  
**Solution:** Implement deterministic control agents with known strength levels

**Implementation:**
- Fixed-strength MCTS agent (no learning)
- Random baseline agent  
- Simple heuristic agent
- Historical "golden" model preservation

---

#### **2. Gradient Flow Analysis (CRITICAL)**
**Impact:** ★★★★★ | **Feasibility:** ★★★★ | **Risk:** ★★ | **Resources:** ★★

**Problem:** Value head learning failure mechanism unknown  
**Evidence:** "Value accuracy stuck at 49.8% ± 0.3% despite 20x LR scaling"  
**Solution:** Instrument gradient tracking to identify learning bottlenecks

**Implementation Steps:**
1. Add gradient norm tracking per layer/head
2. Track gradient direction consistency
3. Monitor weight update magnitudes
4. Analyze gradient-to-weight update ratios
5. Detect vanishing/exploding gradient patterns

---

#### **3. Value Target Quality Analysis (CRITICAL)**
**Impact:** ★★★★★ | **Feasibility:** ★★★★ | **Risk:** ★★ | **Resources:** ★★★

**Problem:** Self-play targets may be noisy/inconsistent  
**Evidence:** "Adaptive LR reduction pattern suggests poor gradient quality"  
**Solution:** Analyze value target generation and consistency

**Implementation Steps:**
1. Track value target distribution per game phase
2. Measure target variance for identical positions
3. Analyze correlation between MCTS value and final outcome
4. Compare early vs. late game target quality
5. Validate against known game positions

---

### **TIER 2: High Impact (2-4 Weeks)**

#### **4. Separated Value/Policy Training (HIGH IMPACT)**
**Impact:** ★★★★★ | **Feasibility:** ★★★ | **Risk:** ★★★ | **Resources:** ★★★★

**Problem:** Joint training may create conflicting gradients  
**Evidence:** "Policy accuracy degraded 85% while value remained static"  
**Solution:** Implement independent training pipelines

**Implementation Approaches:**
- **Option A:** Separate optimizers (conservative)
- **Option B:** Alternating training phases (moderate)  
- **Option C:** Completely separate networks (aggressive)

---

#### **5. Loss Function Engineering (HIGH IMPACT)**
**Impact:** ★★★★ | **Feasibility:** ★★★★ | **Risk:** ★★ | **Resources:** ★★★

**Problem:** Current loss doesn't provide effective learning signal  
**Evidence:** "Value loss weights (0.9, 0.1) showed no improvement over (0.65, 0.35)"  
**Solution:** Design better value learning objectives

**Candidate Approaches:**
- Ranking loss instead of MSE
- Temporal difference learning integration
- Uncertainty-weighted loss functions
- Multi-scale value targets (short/medium/long term)

---

### **TIER 3: Diagnostic & Validation (4-6 Weeks)**

#### **6. Architecture Ablation Studies (MEDIUM IMPACT)**
**Impact:** ★★★★ | **Feasibility:** ★★★★ | **Risk:** ★★ | **Resources:** ★★★★

**Problem:** Unknown which architectural components are problematic  
**Solution:** Systematic component isolation testing

**Test Matrix:**
- Shared vs. separate backbones
- Different head architectures
- Various backbone depths/widths
- Attention vs. conventional layers

---

#### **7. Toy Problem Validation (DIAGNOSTIC)**
**Impact:** ★★★ | **Feasibility:** ★★★★★ | **Risk:** ★ | **Resources:** ★★

**Problem:** Need to isolate value learning capability  
**Solution:** Test value head on simpler, known problems

**Test Problems:**
- Tic-tac-toe position evaluation
- Simple endgame scenarios
- Pre-labeled Yinsh positions
- Synthetic regression tasks

---

### **TIER 4: Research Innovations (6+ Weeks)**

#### **8. Alternative Training Algorithms (RESEARCH)**
**Impact:** ★★★★★ | **Feasibility:** ★★ | **Risk:** ★★★★ | **Resources:** ★★★★★

**Problem:** Current training methodology fundamentally flawed  
**Solution:** Explore novel training approaches

**Research Directions:**
- Evolutionary training methods
- Meta-learning approaches
- Curriculum learning strategies
- Multi-objective optimization

---

## Control Model Design Specification

### **Control Agent Architecture**

#### **1. Fixed MCTS Agent (Primary Control)**
```python
class ControlMCTSAgent:
    """Deterministic MCTS agent with fixed parameters"""
    
    def __init__(self):
        self.simulations = 400  # Fixed simulation budget
        self.c_puct = 1.0       # Fixed exploration
        self.temperature = 0.1   # Minimal randomness
        self.use_learned_policy = False  # No neural network
        self.use_learned_value = False   # Simple heuristic evaluation
        
    def evaluate_position(self, board_state):
        """Simple material + position heuristic"""
        return self.count_material(board_state) + self.position_bonus(board_state)
        
    def get_policy_priors(self, legal_moves):
        """Uniform policy over legal moves"""
        return [1.0 / len(legal_moves)] * len(legal_moves)
```

#### **2. Random Agent (Baseline Control)**
```python
class RandomAgent:
    """Pure random move selection"""
    
    def select_move(self, legal_moves):
        return random.choice(legal_moves)
```

#### **3. Heuristic Agent (Intermediate Control)**
```python
class HeuristicAgent:
    """Rule-based agent with game knowledge"""
    
    def __init__(self):
        self.rules = [
            self.prioritize_scoring_moves,
            self.avoid_giving_opponent_advantage,
            self.control_center_positions,
            self.maintain_piece_connectivity
        ]
```

#### **4. Historical Model Preservation**
```python
class HistoricalControlManager:
    """Preserve and maintain access to historical models"""
    
    def __init__(self):
        self.control_models = {
            "baseline_v1": "models/control/baseline_iteration_2.pt",
            "best_historic": "models/control/best_historical_1516_elo.pt",
            "stable_random": RandomAgent(),
            "fixed_mcts": ControlMCTSAgent()
        }
        
    def get_elo_reference_points(self):
        return {
            "random": 1000,      # By definition
            "fixed_mcts": 1200,  # Estimated
            "heuristic": 1350,   # Estimated  
            "baseline_v1": 1516, # Known from experiments
        }
```

### **Tournament Integration**

#### **Tournament Configuration Enhancement**
```python
# Add to tournament_config.py
CONTROL_AGENTS = {
    "random_control": {
        "type": "random",
        "elo_reference": 1000,
        "games_per_tournament": 100
    },
    "mcts_control": {
        "type": "fixed_mcts", 
        "elo_reference": 1200,
        "games_per_tournament": 100
    },
    "baseline_control": {
        "type": "historical_model",
        "model_path": "models/control/iteration_2_baseline.pt",
        "elo_reference": 1516,
        "games_per_tournament": 200  # More games against known strong model
    }
}

# Always include control agents in tournaments
def run_tournament_with_controls(new_model, iteration):
    opponents = [
        new_model,  # Self-play
        CONTROL_AGENTS["random_control"],
        CONTROL_AGENTS["mcts_control"], 
        CONTROL_AGENTS["baseline_control"]
    ]
    return tournament_engine.run_round_robin(opponents)
```

---

## Implementation Timeline

### **Week 1: Foundation**
- [ ] Implement Control model system
- [ ] Add gradient tracking infrastructure  
- [ ] Set up value target analysis pipeline

### **Week 2: Data Collection**
- [ ] Run diagnostic experiments with gradient tracking
- [ ] Analyze value target quality across game phases
- [ ] Establish Control model baselines

### **Week 3-4: Architectural Changes**
- [ ] Implement separated value/policy training
- [ ] Test alternative loss functions
- [ ] Run Control-based tournaments

### **Week 5-6: Validation**
- [ ] Architecture ablation studies
- [ ] Toy problem validation
- [ ] Cross-validate findings

### **Week 7+: Innovation**
- [ ] Alternative training algorithms
- [ ] Novel architectural approaches
- [ ] Research publication preparation

---

## Success Metrics

### **Immediate (Weeks 1-2)**
- [ ] Gradient analysis reveals value head learning bottleneck
- [ ] Value target quality assessment shows clear issues
- [ ] Control models provide stable Elo baselines

### **Short-term (Weeks 3-4)**
- [ ] Separated training shows >10% value accuracy improvement
- [ ] New loss functions demonstrate learning signal improvement
- [ ] Tournament results show statistical significance vs. controls

### **Medium-term (Weeks 5-6)**
- [ ] Value accuracy >60% sustained across iterations
- [ ] Policy accuracy >8% top-1 achieved
- [ ] New models consistently beat iteration 2 baseline

### **Long-term (Weeks 7+)**
- [ ] Tournament Elo >1600 achieved
- [ ] Learning patterns show progressive improvement
- [ ] Methodology generalizes to other experiments

---

## Risk Mitigation

### **High-Risk Items**
1. **Separated Training:** May break existing infrastructure
   - **Mitigation:** Implement as configuration option first
2. **Control Model Integration:** May complicate tournament system
   - **Mitigation:** Start with simple agents, expand gradually
3. **Alternative Algorithms:** High development cost
   - **Mitigation:** Focus on diagnostic findings first

### **Resource Management**
- **Computational:** Stagger experiments to avoid resource conflicts
- **Development:** Prioritize high-impact, low-risk items first
- **Validation:** Use toy problems for quick iteration

---

**Next Action:** Implement Control model system (Week 1, Priority 1)  
**Review Date:** July 7, 2025  
**Success Gate:** Control baselines established and gradient analysis operational 