# Humanized Poker Player: Milestone Report

**Author:** Noah Tekle  
**Email:** ntekle@usc.edu  
**Date:** December 2024

## 1. Introduction

This milestone report presents preliminary results for developing a human-like poker agent that balances game-theoretic optimality with behavioral authenticity. While traditional poker AIs like Libratus achieve superhuman performance through Nash equilibrium strategies, they lack the psychological variance that characterizes human play. Our approach integrates reinforcement learning with behavioral modeling to create an agent that exhibits realistic bluffing patterns, risk tolerance variations, and emotional responses.

### 1.1 Research Questions

Based on feedback received, we address several key questions:

1. **Nash Equilibrium in Emotion-Aware Settings:** In an emotion-aware poker environment, traditional Nash equilibrium may not exist in the same form. Emotional states introduce non-stationary preferences that can shift optimal strategies. We hypothesize that the convergent solution will be a *behavioral equilibrium* where agents adapt to opponent emotional patterns rather than pure game-theoretic optimality.

2. **Emotion State Inference:** We propose a hybrid approach where emotional states are initially inferred from observed actions (using action pattern analysis) and can be dynamically updated during gameplay. This allows the agent to adapt to opponent behavior while maintaining internal emotional consistency.

3. **Humanization Layer Architecture:** The humanization layer serves as a *modulation* component that adjusts policy outputs rather than replacing them entirely. This ensures the agent maintains competitive performance while introducing human-like variance.

## 2. Methodology

### 2.1 Environment and Baselines

We utilize a custom No-Limit Texas Hold'em simulator with the following specifications:
- **State Representation:** 320-dimensional feature vector encoding hole cards, community cards, stack sizes, pot size, betting history, and position
- **Action Space:** 6 discrete actions (fold, check/call, small bet, medium bet, large bet, all-in)
- **Starting Chips:** 1,000 per player
- **Blinds:** Small blind (5), Big blind (10)

**Baselines Implemented:**

1. **MCTS-based Agent (Baseline 1):** Traditional Monte Carlo Tree Search with neural network value/policy guidance. Trained via self-play with supervised learning on MCTS-generated trajectories.

2. **PPO Agent (Baseline 2):** Proximal Policy Optimization with Generalized Advantage Estimation. Uses opponent pool and random opponent mixing (50% probability) to mitigate distribution shift.

3. **DQN Agent (Baseline 3):** Deep Q-Network with experience replay and target network. Epsilon-greedy exploration with decay from 1.0 to 0.01.

4. **Random Agent:** Uniform random action selection over valid moves (used for evaluation).

All agents use identical network architectures: 4-layer shared backbone (1024→512→256→128) with separate policy and value heads, batch normalization, and dropout (0.3).

### 2.2 Training Configuration

**MCTS Training:**
- Self-play games: 5,000
- MCTS simulations per action: 400
- Batch size: 128
- Learning rate: 0.0001 (with ReduceLROnPlateau scheduling)
- Epochs: 200

**PPO Training:**
- Rollout length: 4,096 steps
- Update frequency: 4,096 steps
- PPO epochs: 8 per update
- Batch size: 128
- Learning rate: 1e-4
- Clip epsilon: 0.2
- Entropy coefficient: 0.075
- Opponent pool size: 8
- Random opponent probability: 0.5

**DQN Training:**
- Episodes: 20,000
- Replay buffer size: 100,000
- Batch size: 64
- Learning rate: 1e-4
- Gamma: 0.99
- Epsilon decay: 0.995
- Target network update frequency: 100 steps
- Random opponent probability: 0.5

### 2.3 Evaluation Protocol

All evaluations follow a consistent protocol:
- **Games per evaluation:** 200 matches
- **Hands per game:** 20 hands (configurable)
- **Total hands:** 4,000 per evaluation
- **Chip conservation:** Verified to ensure no chips are created or destroyed
- **Metrics:** Win rate, chip statistics (net gain/loss, average per game), pot statistics (pot won, pot contributed, pot win rate)

We evaluate across multiple seeds and report variance where applicable. Each agent plays in both Player 1 and Player 2 positions to control for positional bias.

## 3. Preliminary Results

### 3.1 Baseline Performance Comparison

Table 1 summarizes head-to-head comparisons between all baseline agents over 200 games (4,000 total hands).

| Matchup | Winner | Win Rate | Net Chips/Game | Notes |
|---------|--------|----------|----------------|-------|
| DQN vs MCTS | DQN | 97.5% | +841.6 | DQN dominates MCTS |
| PPO vs MCTS | PPO | 92.0% | +463.3 | PPO significantly stronger |
| DQN vs PPO | DQN | 53.0% | +17.2 | Close match, slight DQN advantage |
| PPO vs Random | PPO | 62.5% | +310.4 | Strong performance |
| DQN vs Random | DQN | 60.5% | +204.7 | Strong performance |
| MCTS vs Random | Random | 2.5% | -545.9 | MCTS performs poorly |

**Key Findings:**

1. **MCTS Underperformance:** The MCTS-based agent performs exceptionally poorly against random opponents (2.5% win rate), suggesting it may have overfit to self-play dynamics or lacks robustness to non-optimal play.

2. **PPO vs DQN:** DQN shows a slight edge (53.0% win rate) over PPO, though both perform well against random opponents. The close margin suggests both algorithms learn effective strategies.

3. **PPO Dominance over MCTS:** PPO achieves 92.0% win rate against MCTS, indicating that direct policy optimization outperforms tree search in this setting.

4. **Chip Conservation:** All evaluations maintain perfect chip conservation (400,000 total chips across 200 games), validating our implementation.

### 3.2 Performance Against Random Opponents

Both PPO and DQN demonstrate strong performance against random opponents:

**PPO Performance:**
- Win rate: 62.5% (125/200 games)
- Net gain: +62,071 chips (+310.4 per game)
- Pot win rate: 119.6% (winning more than contributed, indicating selective play)
- Final chip percentage: 65.5% of total chips

**DQN Performance:**
- Win rate: 60.5% (121/200 games)
- Net gain: +40,940 chips (+204.7 per game)
- Pot win rate: 104.9% (also demonstrating selective play)
- Final chip percentage: 60.2% of total chips

Both agents show "selective play" characteristics (pot win rate > 100%), meaning they win more chips than they contribute to pots—a hallmark of strong poker strategy.

### 3.3 Variance Analysis

Preliminary variance analysis across evaluation runs shows:
- **PPO:** Lower variance in chip gains (consistent +300-320 chips/game range)
- **DQN:** Slightly higher variance (ranging +180-220 chips/game)
- **MCTS:** Extremely high variance (ranging from -600 to -500 chips/game, consistently negative)

The variance patterns suggest PPO learns more stable strategies, while DQN's Q-learning approach may be more sensitive to exploration-exploitation trade-offs.

## 4. Discussion and Next Steps

### 4.1 Addressing Feedback

**Nash Equilibrium in Emotion-Aware Settings:** Our current baselines operate without explicit emotional modeling, but results suggest that even "optimal" strategies (PPO, DQN) converge to different equilibria. The 53% DQN vs PPO win rate indicates multiple viable strategies exist. When emotional states are introduced, we expect convergence to *behavioral equilibria* where agents adapt to opponent emotional patterns.

**Emotion State Inference:** We plan to implement emotion inference using:
1. **Action Pattern Analysis:** Track betting frequency, aggression metrics, fold rates
2. **Temporal Modeling:** Use LSTM/GRU to model emotional state evolution
3. **Bayesian Inference:** Update emotional state beliefs based on observed actions

**Humanization Layer:** The layer will modulate policy logits before softmax, allowing controlled variance injection:
```
humanized_policy = softmax(policy_logits + emotion_bias + noise)
```
This maintains competitive performance while introducing behavioral variance.

### 4.2 Limitations and Challenges

1. **MCTS Failure:** The poor MCTS performance suggests our self-play training may have converged to a weak local optimum. This requires investigation.

2. **Evaluation Scale:** Current evaluations (200 games) may need scaling to 1,000+ games for statistical significance, especially for close matchups like DQN vs PPO.

3. **Behavioral Metrics:** We lack explicit "human-likeness" metrics (bluff frequency, reaction variance) in current evaluations. These will be critical for the humanization layer.

### 4.3 Next Steps

1. **Implement Humanization Layer:** 
   - Start with behavior cloning from human gameplay data (if available)
   - Develop LLM-based reasoning component for emotional decision-making
   - Integrate emotional state inference module

2. **Expand Baselines:**
   - Implement CFR (Counterfactual Regret Minimization) baseline
   - Add rule-based agents (tight-aggressive, loose-passive) from literature
   - Compare against published poker AI results

3. **Behavioral Evaluation:**
   - Define "human-likeness" metrics (bluff rate, aggression index, variance in bet sizing)
   - Conduct human evaluation studies
   - Measure deviation from optimal play under different personas

4. **Multi-Seed Analysis:**
   - Run evaluations across 10+ random seeds
   - Report confidence intervals and statistical significance
   - Analyze variance as a function of agent type (aggressive vs conservative)

## 5. Conclusion

Preliminary results demonstrate that PPO and DQN agents achieve strong performance (60-62% win rate) against random opponents, while MCTS-based training fails to produce a competitive agent. The close DQN vs PPO matchup (53% vs 47%) suggests multiple viable strategies exist, supporting our hypothesis that behavioral equilibria may differ from pure Nash equilibrium.

The foundation is established for integrating humanization layers. Next steps focus on implementing emotional state modeling, behavioral variance injection, and comprehensive evaluation metrics that measure both competitive performance and human-likeness.

---

## References

[To be expanded with full citations]

- Brown, N., & Sandholm, T. (2019). Superhuman AI for multiplayer poker. *Science*, 365(6456), 885-890.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

---

## Appendix

### A. Implementation Details

**Codebase Structure:**
- `poker_game.py`: Game logic and state representation
- `value_policy_network.py`: Neural network architecture
- `ppo_trainer.py`: PPO algorithm implementation
- `dqn_trainer.py`: DQN algorithm implementation
- `train_ppo.py`, `train_dqn.py`: Training scripts
- `evaluate_ppo.py`, `evaluate_dqn.py`, `evaluate.py`: Evaluation scripts
- `compare_ppo_mcts.py`, `compare_dqn_ppo.py`, `compare_dqn_mcts.py`: Comparison scripts

**Hardware:** All training and evaluation performed on CPU (no GPU acceleration used).

**Reproducibility:** All random seeds are set for reproducibility. Evaluation scripts support `--games` and `--hands-per-game` parameters for consistent evaluation protocols.

### B. Detailed Results

**Chip Statistics (200 games, 20 hands/game):**

| Agent | Final Chips | % of Total | Net Gain | Per Game |
|-------|------------|------------|----------|----------|
| PPO (vs Random) | 262,071 | 65.5% | +62,071 | +310.4 |
| DQN (vs Random) | 240,940 | 60.2% | +40,940 | +204.7 |
| PPO (vs MCTS) | 292,653 | 73.2% | +92,653 | +463.3 |
| DQN (vs MCTS) | 368,330 | 92.1% | +168,330 | +841.6 |
| MCTS (vs Random) | 90,819 | 22.7% | -109,181 | -545.9 |

**Pot Statistics:**

| Agent | Pot Won | Pot Contributed | Pot Win Rate |
|-------|---------|-----------------|--------------|
| PPO | 502,393 | 419,963 | 119.6% |
| DQN | 1,409,509 | 1,343,334 | 104.9% |

Both agents demonstrate selective play (pot win rate > 100%), indicating they win more chips than they contribute to pots.

