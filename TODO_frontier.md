# YinshML — Frontier Research Directions

Ambitious research agenda. Items here would each be multi-week projects and, collectively, constitute a second-generation YinshML — moving beyond the AlphaZero recipe to genuinely frontier techniques.

The running `TODO.md` covers the AlphaZero playbook (EMA, subtree reuse, FPU, augmentation, cosine LR, etc.). This file is the "what would a top lab be doing" companion. Neither file is binding — both are menus.

---

## Hardware reality check

Current compute: one Mac Mini (MPS), with a GPU possibly available ~6 months out. Several items below were written as though time and money were no obstacle — here they are, honestly, against the actual hardware:

| § | Item | Mac Mini | With GPU | Reason |
|---|---|---|---|---|
| 1 | EfficientZero / MuZero | 🔴 infeasible | 🟡 feasible but heavy | Dynamics + reanalyse multiplies compute ~5-10× per sample; published results use thousands of TPU-hours |
| 2 | Transformer backbone | 🟡 will regress | 🟢 viable | MPS attention kernels unoptimized; Transformers data-hungrier at small scales |
| 3 | League training | 🔴 infeasible | 🟡 needs fleet | 30 agents × 100 games/iter compute multiplier |
| 4 | Offline RL (IQL/CQL) | 🟢 **do now** | 🟢 | Same cost envelope as existing supervised pretraining (~4h MPS) |
| 5 | Search-consistency loss | 🟢 **do now** | 🟢 | ~10% per-iter overhead if applied every K=10 steps |
| 6 | Diffusion policy head | 🔴 kills MCTS throughput | 🟡 with distillation | Multiplicative inference cost per node expansion |
| 7 | Multi-game co-training | 🟡 engineering-bound | 🟢 | Compute similar to current; payoff scales with compute |
| 8 | Interpretability / SAE | 🟢 **do now** | 🟢 | Runs on existing checkpoint; mostly forward passes + small linear |
| 9 | QAT + quantized inference | 🔴 weak MPS support | 🟢 | CUDA quantization mature; MPS is not |
| 10 | Scaling laws | 🔴 can't reach the large end | 🟡 | 20 runs × 4h = 80+ hours minimum; need the large end to fit the law |

**The good news.** The three green-light items (§4, §5, §8) are exactly the three diagnostic probes flagged in the strategic sequencing below. The hardware constraint and the correct scientific sequencing point at the same three items. Execute them first; they're the cheapest *and* the most informative about whether the red-light items are worth the GPU budget when it arrives.

**Implication for sequencing.** On Mac Mini, you're running the AlphaZero polish sprint plus the three diagnostic probes. That's it. When the GPU lands, the first question is "what did the probes tell us?" — if §5 (search-consistency) closed the discrimination plateau, most of this file is moot and you put the GPU hours into polish scale-up. If §8 (interpretability) revealed missing concepts, add auxiliary heads, no architecture change needed. Only if both probes fail to explain the plateau do you spend GPU hours on §1 (MuZero) or §2 (Transformer).

---

## Strategic sequencing

The naïve plan is: finish `TODO.md` → polished AZ baseline → try frontier items → bake-off. That's correct for the final comparison, but wastes weeks if one of the frontier diagnostics can falsify the whole approach early.

Recommended sequence:

1. **Polish sprint (1-2 weeks).** Burn down `TODO.md` Tier 1/2. Outcome: an AlphaZero baseline whose residual weaknesses are *architectural*, not configurational.
2. **Parallel diagnostic probes during week 2.** Three cheap experiments that stress-test whether MuZero/Transformer/offline-RL are actually necessary:
   - **Search-consistency loss** (§5 below) — 1-2 day trainer patch. Does it break the 0.104 discrimination plateau?
   - **Value-head SAE / probe** (§8 below) — 1-2 days on the current checkpoint. What concepts is the network failing to represent?
   - **Offline IQL pretraining** (§4 below) — 3-5 days, runs on the already-scraped 240K Boardspace positions. Does it warm-start better than supervised BC?
3. **Decision gate.** Three outcomes:
   - Polish + search-consistency breaks the plateau → AZ recipe was correct, push Tier 3-4 of `TODO.md`, skip most of this file.
   - Polish + search-consistency doesn't move the plateau, but SAE shows missing concepts → add targeted features + auxiliary heads (§5, §8, §10). Still within AZ recipe.
   - Polish + search-consistency doesn't move it, SAE shows rich internal representations → plateau is architectural. Commit to MuZero/Transformer (§1, §2).
4. **Frontier build, if warranted.** §1 and §2 take 2-3 months each. §3 (league training) is the natural follow-on because it subsumes the tournament-gate problem entirely.

The point: **don't commit to a 3-month MuZero implementation without first running the 2-day experiment that could tell you MuZero is unnecessary.** Conversely, don't ship a fully-polished AlphaZero baseline and then discover it was architecturally capped — the polish sprint and diagnostic probes are overlapped, not sequenced.

---

## Frontier items

### 1. EfficientZero / Stochastic MuZero (learned world model)

**What.** Replace the hand-written YINSH simulator in MCTS with a learned dynamics network. The network represents a position as a latent vector, learns `dynamics(latent, action) → (next_latent, reward)`, and runs MCTS in latent space. EfficientZero adds (a) a self-supervised consistency loss (predicted next-latent vs encoded-real-next-latent), (b) a value-prefix head that predicts near-term value trajectories, and (c) off-policy value correction via reanalyse.

**Why ambitious.** Breaks the "search is the bottleneck" assumption. Current MCTS spends most of its budget on simulator calls and repeated encoding; a latent world model collapses all of that into one forward pass per node. EfficientZero reached AlphaZero-level Go strength with 50× fewer environment samples. On YINSH specifically, the ring-placement phase has trivially learnable dynamics — a learned model should be near-perfect there, freeing compute for the tactical mid-game.

**Implementation sketch.**
- Representation network `h: state → latent ∈ R^256`
- Dynamics network `g: (latent, action) → (next_latent, reward)` — residual MLP or tiny Transformer
- Prediction network `f: latent → (policy, value, value_prefix)` — policy over 7395 moves, value 7-class (or scalar), value_prefix as a sum-of-bounded-rewards prediction over the next K unroll steps
- Training: unroll 5 steps, loss = policy CE + value CE + reward MSE + consistency (cosine) at each step, weighted by 1/K
- MCTS: replace `game.apply_move(state, move)` with `g(latent, move)`; expand with `f(next_latent)`
- Reanalyse: periodically re-evaluate stored trajectories with the latest network to get fresher value targets

**References.** Schrittwieser 2020 (MuZero), Ye 2021 (EfficientZero), Antonoglou 2022 (Stochastic MuZero), Danihelka 2022 (Gumbel MuZero — relevant for policy improvement at low sim counts).

**Risk / failure modes.** Dynamics learning is unstable early; without the simulator as an oracle you can train a network that plans well in a space disconnected from real YINSH. Mitigation: retain a "ground truth" evaluation mode that uses the real simulator for tournaments during initial iterations; only switch to latent-space evaluation once tournament-measured skill tracks latent-space skill.

**Effort.** 6-10 weeks including tuning. Highest single item on this list.

**Dependencies.** Depends on polish sprint (Dirichlet, subtree reuse are standard inside MuZero). Offline RL pretraining (§4) stacks cleanly on top — pretrain representation + prediction networks on Boardspace before unrolling.

**Files to touch.** New `yinsh_ml/muzero/` module; `yinsh_ml/training/supervisor.py` to add an alt training loop; `yinsh_ml/search/mcts.py` — add latent-evaluation mode.

---

### 2. Transformer backbone replacing the ResNet

**What.** Replace `YinshNetwork` (ResNet with attention in `yinsh_ml/network/model.py`) with a pure Transformer: tokenize the 11×11 hex board into 85 position tokens plus a game-phase register token and a score-differential register token, run 8-12 Transformer layers with learned positional embeddings respecting hex adjacency, decode to policy/value heads.

**Why ambitious.** YINSH is dominated by long-range interactions — a 4-marker run halfway across the board threatens completion regardless of local context; ring mobility is a global property. ResNet receptive fields grow linearly with depth, requiring 8+ layers to see the full board. Attention is O(1) in layers for global dependencies. Recent Go work (Transformer-Go, Alrdvark, LAMBO) matches or beats CNN baselines at lower depth. Secondary: Transformers are easier to scale — you get scaling laws for free.

**Implementation sketch.**
- Tokenize: 85 board positions + K register tokens (phase, turn, score margin, captured-rings counts for each player)
- Embed: learned 128-dim token embedding + learned positional embedding per hex coordinate (not sinusoidal — hex coordinates aren't a grid)
- Backbone: 8-layer Transformer, 8 heads, FFN 4×, pre-norm, no causal mask
- Heads: policy over (from_token, to_token) ∈ R^85×R^85 masked to valid moves; value over 7 classes from register token
- Total params ≈ 5-10M, comparable to current ResNet

**References.** Nakhost 2023 (Transformer-Go), Czech 2023 (CrazyAra 5), vision-Transformer literature for tokenization strategies, "Register tokens" (Darcet 2023).

**Risk / failure modes.** Transformers are data-hungrier than ResNets at small scales; may regress at current training-data budgets. Mitigation: combine with offline IQL pretraining (§4) to stuff the Transformer with expert data before self-play begins.

**Effort.** 3-5 weeks.

**Dependencies.** Should land after offline pretraining infrastructure (§4) so you can measure whether the Transformer's data hunger is problem or strength.

**Files to touch.** `yinsh_ml/network/model.py` — new `YinshTransformer`; `yinsh_ml/utils/encoding.py` — tokenization utility (can coexist with plane encoder via `use_token_encoding` flag, same pattern as current `use_enhanced_encoding`).

---

### 3. League / population-based training

**What.** Replace the linear "iter_N vs iter_{N-1}" evaluation pipeline with an AlphaStar-style league: maintain ~30-100 agents in parallel, split into "main agents" (the ones whose strength you optimize for), "main exploiters" (trained specifically to beat main agents, then discarded/recycled), and "league exploiters" (trained to beat any agent with >X% win rate against anyone). Main agents train on a PFSP (Prioritized Fictitious Self-Play) distribution over the league rather than just their most recent self.

**Why ambitious.** The 55% promotion gate in `supervisor.py` and its false-positive problems (discussed in `TODO.md` Tier 3) exist *because* the evaluation structure is a linear chain. A league provides Elo over dozens of opponents simultaneously, making promotion decisions statistically well-founded and — more importantly — discovering *exploits* in the current agent's policy that linear self-play cycles miss. YINSH has a well-known "ring-hoarding" failure mode where agents learn to avoid capturing until the board is jammed; exploiters would find and punish this automatically.

**Implementation sketch.**
- Pool manager: SQLite (reuse existing `yinsh_ml/tracking/database.py`) tracking {agent_id, checkpoint_path, role, elo, last_trained_iter}
- PFSP matchmaking: `P(opponent) ∝ (1 − win_rate_vs_opponent)^p` for p ∈ [0.5, 2.0]
- Three roles: main (long-lived, optimized), main_exploiter (short-lived, targets main), league_exploiter (targets league)
- Training: each agent is a separate `TrainingSupervisor`; they share a central replay buffer stamped with (generator_agent_id, opponent_agent_id)
- Elo: reuse `yinsh_ml/utils/elo_manager.py`, extend to population
- Compute: 30 agents × 100 games/iter/agent = 3000 games/iter; requires compute budget the current setup lacks

**References.** Vinyals 2019 (AlphaStar league), Czarnecki 2020 (real-world games are transitive + cyclic — league theory), Marris 2021 (multi-agent RL evaluation).

**Risk / failure modes.** Compute cost is linear in population size. On current hardware (single MPS box) this is painful; genuinely needs cloud or a small GPU fleet. Mitigation: start with a 5-agent mini-league as proof-of-concept before scaling.

**Effort.** 4-6 weeks for infrastructure; ongoing compute cost.

**Dependencies.** Orthogonal to §1 and §2 — works with any core agent architecture. High synergy with §4 (offline RL) and §10 (scaling laws).

**Files to touch.** New `yinsh_ml/league/` module; `yinsh_ml/tracking/database.py` — add population tables; `yinsh_ml/utils/tournament.py` — extend to PFSP matchmaking.

---

### 4. Offline RL pretraining on expert corpus (IQL / CQL)

**What.** The current supervised pretraining (`scripts/run_supervised_pretraining.py`) is behavioral cloning: train to imitate expert moves + predict game outcomes. That's demonstrably the wrong tool — it gives a prior biased toward expert-move-distribution rather than expert-*value*. Replace with Implicit Q-Learning (IQL) or Conservative Q-Learning (CQL) on the same 240K Boardspace positions: learn a Q-function by bootstrapping from trajectories, penalize out-of-distribution actions via expectile regression (IQL) or value-function regularization (CQL).

**Why ambitious.** Offline RL provides a value estimate that is Bellman-consistent with the expert trajectories, not just a BC-style scalar target. This directly addresses the warm-start regression we just debugged: BC + MCTS-with-own-value-head mismatches because BC doesn't solve the Bellman equation, whereas IQL does. The same 240K positions, re-trained with IQL, should produce a warm-start that survives iter_1 self-play — which is the open research question in `RESEARCH_LOG.md`.

**Implementation sketch.**
- IQL training loop: expectile regression for V (τ = 0.7 or 0.9), advantage-weighted regression for policy, Q learned from (s, a, r, s') tuples from game trajectories
- Reuse the 7-class discretized value head — IQL's V and Q can both be 7-class softmax
- Policy head target: advantage-weighted expert moves, `exp((Q(s,a) − V(s)) / β)` with β tuned for temperature
- Dataset: existing `expert_games/training_data.npz` already has the (state, move, outcome) tuples; extend to (state, move, reward, next_state) by walking trajectories
- Warm-start: load the IQL-pretrained checkpoint into the self-play trainer exactly as the current supervised warm-start does

**References.** Kostrikov 2021 (IQL), Kumar 2020 (CQL), Nair 2020 (AWAC — simpler cousin).

**Risk / failure modes.** Offline RL is finicky at small data scales (240K positions is small compared to D4RL standards). Mitigation: start with AWAC (simpler, more robust than IQL/CQL) as a smoke test; escalate only if AWAC shows warm-start improvement.

**Effort.** 2-3 weeks.

**Dependencies.** Runs on existing data pipeline. Zero dependency on other frontier items. This is the cheapest and potentially highest-ROI item on this list *given what we just learned debugging the warm-start*.

**Files to touch.** New `scripts/run_offline_rl_pretraining.py`; `yinsh_ml/training/offline_rl.py` for the IQL loop; `yinsh_ml/network/wrapper.py` for Q-head loading.

---

### 5. Search-consistency regularization

**What.** Add a training loss that penalizes divergence between short-search and long-search evaluations of the same position. For a mini-batch of positions, run MCTS with N_short sims and N_long sims (e.g., 50 and 800); backprop KL(policy_long || policy_short) + MSE(value_long, value_short). Target: the network should not need the tree to reach its best policy/value — the tree should serve it, and the network should learn to internalize tree improvements.

**Why ambitious.** This is the specific technique that pushed KataGo past its AlphaZero ceiling. The discrimination plateau at 0.104 (from `RESEARCH_LOG.md`) is consistent with a network that has learned to lean on MCTS for every hard decision — removing MCTS and measuring raw-network discrimination should show the same plateau or worse. Search-consistency forces the network to *close the search-vs-no-search gap*, directly attacking the plateau's root cause.

**Implementation sketch.**
- Every K training steps, sample a small batch (32-64 positions) from the replay buffer
- Run MCTS at two sim budgets — `search_short(pos, sims=N_short)`, `search_long(pos, sims=N_long)` — reusing the same NN for both
- Loss = `λ_policy · KL(π_long || π_short) + λ_value · MSE(v_long, v_short)`, added to the standard loss
- Stop-gradient on the long-search predictions (they're the target, not the learner)
- Schedule `N_long` from `2×N_short` → `20×N_short` over training

**References.** Wu 2019 (KataGo), DeepNash (Perolat 2022), earlier Rosin 2011 (consistency in rollouts).

**Risk / failure modes.** Doubles per-step cost because of extra search. Mitigation: apply once per K=10 training steps, not every step. Also: if the long-search policy is itself poorly-calibrated early in training, you'll bake bad targets in — use a warmup period before enabling this loss.

**Effort.** 1-2 weeks.

**Dependencies.** Can ship on top of the current AZ pipeline as a drop-in loss term. **This is the top diagnostic probe in the strategic sequencing above** — if it breaks 0.104 → 0.15 discrimination, the plateau was a training-signal issue and much of this document becomes moot.

**Files to touch.** `yinsh_ml/training/trainer.py` — add `_search_consistency_step()`; `yinsh_ml/training/supervisor.py` — config plumbing for `search_consistency_weight` and `search_consistency_every_k_steps`.

---

### 6. Diffusion policy head

**What.** Replace the softmax-over-7395-moves policy head with a discrete diffusion model over move tokens. Training: add Gaussian / categorical noise to MCTS visit-count distributions, train a denoiser conditioned on the board state; inference: sample by denoising from pure noise.

**Why ambitious.** Softmax heads are known to under-weight long-tail moves — precisely the "surprise tactical" moves that break human-play styles. Multi-modal target distributions (common in YINSH opening placement where many moves are approximately equal) are compressed by a softmax but preserved by a diffusion process. Recent robotics and game-AI work (Diffusion Policy, Chi 2023) shows consistent improvements on tasks with multi-modal optimal actions.

**Implementation sketch.**
- Discrete diffusion on the simplex over valid moves (not continuous diffusion — moves are discrete)
- Conditioning: board encoding → conditioning vector → diffusion model
- Training: forward process adds noise to MCTS π over T=50 steps; denoiser learns to reverse each step
- Inference: optionally single-step via consistency distillation for low-latency MCTS expansion

**References.** Chi 2023 (Diffusion Policy), Hoogeboom 2021 (Discrete diffusion), Austin 2021 (Structured denoising diffusion).

**Risk / failure modes.** Adds inference latency; incompatible with fast batched MCTS expansion unless single-step-distilled. Unknown whether the multi-modality gains outweigh the latency cost for MCTS use.

**Effort.** 4-6 weeks — this is genuinely novel for game-AI; no published AlphaZero-diffusion hybrid exists to reference.

**Dependencies.** Should come after §2 (Transformer backbone) because the conditioning vector wants rich features.

**Files to touch.** New `yinsh_ml/network/diffusion_head.py`; `yinsh_ml/network/wrapper.py` — inference path; `yinsh_ml/search/mcts.py` — adapt policy sampling.

---

### 7. Multi-game co-training (transfer learning across abstract strategy games)

**What.** Pretrain the backbone (whether ResNet or Transformer) on a mix of hex/grid abstract games — Go, Hex, Havannah, Gomoku — before specializing on YINSH. Either full co-training (all games in one batch, per-game heads) or staged (pretrain on diverse games → fine-tune on YINSH).

**Why ambitious.** Spatial reasoning, run detection, and ring-mobility have analogues in other games. A network pretrained on Hex already has "connection detection" circuits; one pretrained on Go has "group liberties" circuits; the YINSH-specific fine-tune just repurposes these. Precedent: GPT-family models, AlphaFold-2 (diverse protein co-training), Gen-Go. The compute-per-learned-concept is dramatically lower when the concept is shared across games.

**Implementation sketch.**
- Standardize board representation to a common latent shape (e.g., 11×11×C with C including game-ID plane)
- Shared backbone + per-game {policy, value} heads
- Simulators for Hex, Havannah, Gomoku are tiny (<1k LOC each, publicly available)
- Training: sample games from uniform mixture, per-game loss, gradient accumulation across games
- Evaluation: YINSH tournament strength is the only metric that matters; other games are instruments

**References.** Gen-Go (Schwarzer 2022), "Generally capable agents" (Team et al. 2021 — DeepMind XLand).

**Risk / failure modes.** Huge engineering lift for simulators. Transfer benefits unclear for a small model (5-10M params); negative transfer possible if game dynamics are too different.

**Effort.** 6-10 weeks — dominated by simulator integration.

**Dependencies.** Better after §2 (Transformer) — attention-based backbones transfer better than CNNs in the multi-task setting per published results.

**Files to touch.** Substantial — new `yinsh_ml/games/` umbrella with per-game modules; training loop restructure.

---

### 8. Mechanistic interpretability of the value head

**What.** Train a Sparse Autoencoder (SAE) on the penultimate layer of the current value head. Identify concept neurons by correlating SAE features with hand-labeled YINSH concepts ("4-in-a-row threat," "ring trapped," "opponent must respond here"). Identify *absent* concepts by sampling positions where the value head is confidently wrong in tournament play and asking what feature would have flagged them.

**Why ambitious.** Every improvement proposed so far is pushed by vibes — "the discrimination ceiling is the issue because the paper said so," "positional threats matter because humans think so." Interpretability gives evidence-based improvement. If the SAE reveals the network has learned completed-runs + centrality + ring-mobility but no threat concept, the next move is adding a threat auxiliary head and measuring whether it moves the plateau. Anthropic, OpenAI, and DeepMind have all shipped interpretability-driven improvements; game AI has not — this is a methodology gap, not a capability gap.

**Implementation sketch.**
- Capture activations at the penultimate layer for 100K random self-play positions
- Train SAE (features = 8× layer width, L1 penalty on activations per Bricken 2023 / Cunningham 2023)
- Manual labeling pass: identify a few features with human-interpretable firing patterns
- Generate "confident error" dataset: positions where |v| > 0.7 and v disagrees with the tournament outcome
- Intervention: ablate individual features, measure which cause the largest shift in confident-error rate → those are the concepts the net leans on but shouldn't

**References.** Bricken 2023 (Anthropic monosemanticity), Cunningham 2023 (Sparse autoencoders for features), Nanda 2023 (mechanistic interpretability toolkit).

**Risk / failure modes.** SAE training can yield features that don't correspond to anything human-interpretable, making the labeling step unproductive. Mitigation: budget one week for SAE + labeling; if nothing interpretable surfaces, abandon the specific SAE but still use the "confident error" dataset for targeted auxiliary heads.

**Effort.** 1-2 weeks for the initial probe; open-ended if it works.

**Dependencies.** **This is the second top diagnostic probe in the strategic sequencing above.** It runs on the existing checkpoint, zero training-infrastructure changes needed.

**Files to touch.** New `yinsh_ml/interpretability/` module; analysis notebooks in `analysis/`.

---

### 9. Quantization-aware training + 4-bit inference

**What.** Train the network with quantization-aware training (fake-quantization ops in the forward pass simulating int8/int4 arithmetic). At inference, run the quantized model for MCTS rollouts. Effective sims-per-second increases 4-8× at fixed wall-clock.

**Why ambitious.** The highest-leverage operation in MCTS is "add another sim." A 4× sims-per-second improvement compounds with every other frontier item — MuZero with 4× sims, Transformer with 4× sims, league training with 4× sims. In published Go results (Cygnus 2022 etc.), int8 quantization costs 20-50 Elo but enables 3-5× search deepening, net +100-200 Elo.

**Implementation sketch.**
- PyTorch quantization-aware training (QAT) hooks on every linear/conv layer
- Calibrate activation scales on ~10K self-play positions
- Evaluate quantized vs fp32 in a deterministic tournament (prerequisite: deterministic eval from `TODO.md` Tier 3)
- Swap quantized model into `NetworkWrapper` for self-play; retain fp32 for the final tournament gate

**References.** Jacob 2018 (QAT), Micikevicius 2022 (fp8 training), Dettmers 2022 (LLM.int8() — decoder techniques apply).

**Risk / failure modes.** MPS quantization support is weaker than CUDA; may require CPU or ONNX runtime for inference. Mitigation: start with CUDA box before bringing back to MPS.

**Effort.** 2-3 weeks.

**Dependencies.** Orthogonal to everything else. Best done *after* the architecture is settled (don't QAT a ResNet that will be replaced by a Transformer).

**Files to touch.** `yinsh_ml/network/model.py` — add QAT hooks; `yinsh_ml/network/wrapper.py` — quantized inference path.

---

### 10. Scaling-laws study

**What.** Sweep width (64 → 512 channels) × depth (4 → 24 ResNet blocks) × data (10K → 1M training positions) across ~20 training runs. Fit a power law: `loss(N, D, C) = (A/N^α + B/D^β + C/C^γ)` for params N, data D, compute C. Identify the compute-optimal frontier for YinshML.

**Why ambitious.** Every hyperparameter decision made so far has been point-estimate guessing ("StepLR is stable," "50K buffer is enough," "4 epochs is the safe zone"). A scaling-laws study turns guessing into measurement. Chinchilla, Hoffmann 2022, showed that most LLM training before 2022 was undersized on data for its parameter count; the same is almost certainly true of game-AI training. Knowing the frontier lets you make rational trade-offs ("more data vs more params"), rather than emotional ones.

**Implementation sketch.**
- Fix everything except the 3 variables (width, depth, training tokens)
- Fit both isoFLOP curves (fix compute, vary ratio) and isoPerf curves (fix loss, find min compute)
- Metric: validation value-head classification accuracy (the metric that actually predicts tournament strength per `RESEARCH_LOG.md`)
- Deliverable: a plot + fitted equation you can consult before every future capacity decision

**References.** Kaplan 2020, Hoffmann 2022 (Chinchilla), Hilton 2023 (RL scaling).

**Risk / failure modes.** Needs a lot of compute — 20 full training runs, even small ones, is 40-100 GPU-days. Time and money are hypothetically no obstacle here, but this is where that assumption is load-bearing.

**Effort.** 4-6 weeks clock-time, dominated by run duration.

**Dependencies.** Architecture should be mostly settled (don't scale-law a design you're about to replace). Best done after §2 (Transformer) or §1 (MuZero) lands.

**Files to touch.** New `experiments/scaling_laws/` with per-run configs; `scripts/fit_scaling_laws.py` for the analysis.

---

## What this document is *not*

- Not a promise to do any of these. Items here are multi-week investments; each needs a real go/no-go based on the diagnostic probes.
- Not a replacement for `TODO.md`. The polish items there are prerequisites for any fair comparison — and some of them (subtree reuse, EMA, cosine LR) are implicit inside every published MuZero/Transformer-game result anyway.
- Not a ranking by "what's coolest." §4 (offline RL), §5 (search-consistency), §8 (interpretability) are *cheaper and possibly more impactful* than the headline-grabbers (§1 MuZero, §2 Transformer). Start with cheap diagnostics; spend the big budget only when the diagnostics justify it.
