# Psychology Network Phase 3: Joint Finetuning Summary

## Overview

Phase 3 enables **joint finetuning** of both the DQN network and psychology network together. This allows the psychology network to adapt its embeddings based on what helps the DQN make better decisions, rather than just predicting action frequencies.

## What Changed

### Key Differences from Phase 2:

**Phase 2:**
- Psychology network is **frozen** (eval mode)
- Only DQN network learns
- Embeddings are static (from pretrained weights)

**Phase 3:**
- Psychology network is **unfrozen** (train mode)
- Both networks learn together
- Embeddings adapt to improve DQN performance
- Separate learning rates for stability

## Implementation Details

### 1. **Unfreezing Psychology Network**
- Set to `train()` mode during updates
- Gradients flow through embeddings back to psychology network

### 2. **Separate Optimizers**
- DQN optimizer: `lr=1e-4` (standard)
- Psychology optimizer: `lr=1e-5` (10x smaller for stability)

### 3. **Gradient Clipping**
- DQN: `max_norm=1.0`
- Psychology: `max_norm=0.5` (tighter to prevent drift)

### 4. **Monitoring**
- Tracks psychology network gradient norms
- Helps detect instability

## Usage

### Phase 2 (Frozen Psychology Network)
```bash
python train_dqn.py \
    --episodes 10000 \
    --use-opponent-modeling \
    --psychology-network poker-rl/output/models/psychology_network.pt \
    --behavior-embedding-dim 16
```

### Phase 3 (Joint Finetuning)
```bash
python train_dqn.py \
    --episodes 10000 \
    --use-opponent-modeling \
    --psychology-network poker-rl/output/models/psychology_network.pt \
    --behavior-embedding-dim 16 \
    --joint-finetuning \
    --psychology-lr 1e-5
```

## Training Results Analysis

Your Phase 2 training achieved:
- **Mean reward: 0.3300** ✅
- This is **positive**, meaning the agent is winning on average
- The opponent modeling is working!

## When to Use Phase 3

**Use Phase 3 when:**
- Phase 2 training has converged
- You want to further optimize embeddings for your specific task
- You have enough training data (embeddings can overfit with small datasets)

**Stick with Phase 2 if:**
- Training is still improving
- You want stable, predictable embeddings
- You're doing initial exploration

## Phase 3 Benefits

1. **Adaptive Embeddings**: Learn what opponent features matter most for winning
2. **Task-Specific**: Embeddings optimized for your poker variant
3. **End-to-End Learning**: Both networks work together optimally

## Phase 3 Risks

1. **Instability**: Psychology network can drift too much
2. **Overfitting**: Embeddings might overfit to training opponents
3. **Slower Training**: More parameters to update

## Monitoring Phase 3

Watch for:
- **Gradient norms**: Should be stable (not exploding)
- **Reward trends**: Should continue improving
- **Loss values**: Should decrease smoothly

If you see:
- Exploding gradients → Lower `--psychology-lr`
- Decreasing rewards → Disable joint finetuning
- Unstable training → Increase gradient clipping

## Recommended Workflow

1. **Start with Phase 2** (frozen psychology network)
   - Train until convergence
   - Establish baseline performance

2. **Switch to Phase 3** (joint finetuning)
   - Use pretrained Phase 2 model
   - Start with low psychology LR (`1e-5`)
   - Monitor closely

3. **Compare Results**
   - Does Phase 3 improve over Phase 2?
   - If yes, continue; if no, stick with Phase 2

## Hyperparameter Tuning

### Psychology Learning Rate
- **Too high** (`>1e-4`): Unstable, embeddings drift
- **Too low** (`<1e-6`): No adaptation, same as Phase 2
- **Sweet spot**: `1e-5` to `5e-5`

### Gradient Clipping
- Psychology network: `0.3-0.5` (tighter)
- DQN network: `1.0` (standard)

## Example Training Command

```bash
# Phase 2 first (recommended)
python train_dqn.py \
    --episodes 5000 \
    --use-opponent-modeling \
    --psychology-network poker-rl/output/models/psychology_network.pt \
    --behavior-embedding-dim 16 \
    --output poker-rl/output/models/dqn_phase2.pt

# Then Phase 3 finetuning
python train_dqn.py \
    --episodes 5000 \
    --use-opponent-modeling \
    --psychology-network poker-rl/output/models/psychology_network.pt \
    --behavior-embedding-dim 16 \
    --joint-finetuning \
    --psychology-lr 1e-5 \
    --model poker-rl/output/models/dqn_phase2.pt \
    --output poker-rl/output/models/dqn_phase3.pt
```

## Summary

✅ **Phase 3 is now implemented and ready to use!**

- Unfreezes psychology network
- Separate optimizers with different learning rates
- Gradient clipping for stability
- Monitoring and statistics

**Your Phase 2 results (0.3300 mean reward) are excellent!** You can:
1. Continue Phase 2 training for more episodes
2. Try Phase 3 to see if it improves further
3. Evaluate the trained model against different opponents

The choice is yours! 🎯

