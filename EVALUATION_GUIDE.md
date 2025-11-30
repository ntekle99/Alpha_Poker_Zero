# Evaluation Guide for Psychology Network Integration

## Quick Start

### Basic Evaluation (Without Opponent Modeling)

```bash
# Evaluate DQN model against random player
python evaluate_dqn.py \
    --model poker-rl/output/models/dqn_poker_model.pt \
    --games 100
```

### Evaluation With Opponent Modeling

```bash
# Evaluate DQN model with psychology network
python evaluate_dqn.py \
    --model poker-rl/output/models/dqn_poker_model.pt \
    --games 100 \
    --use-opponent-modeling \
    --psychology-network poker-rl/output/models/psychology_network.pt \
    --behavior-embedding-dim 16
```

### Run All Evaluations

```bash
# Run all comparisons (without opponent modeling)
python run_all_evaluations.py \
    --dqn-model poker-rl/output/models/dqn_poker_model.pt \
    --games 100

# Run all comparisons (with opponent modeling for DQN)
python run_all_evaluations.py \
    --dqn-model poker-rl/output/models/dqn_poker_model.pt \
    --games 100 \
    --use-opponent-modeling \
    --psychology-network poker-rl/output/models/psychology_network.pt \
    --behavior-embedding-dim 16
```

## How It Works

### Automatic Detection

The evaluation scripts automatically detect if your DQN model was trained with opponent modeling:

1. **Model Loading**: Checks if model has `behavior_embedding_dim > 0`
2. **Backward Compatibility**: If no embedding dim, uses zeros (works with old models)
3. **Psychology Network**: Loads and uses psychology network if `--use-opponent-modeling` is set

### What Gets Evaluated

When you run `run_all_evaluations.py`, it tests:

1. **MCTS vs Random** - Baseline comparison
2. **MCTS vs PPO** - MCTS vs PPO agent
3. **PPO vs Random** - PPO baseline
4. **DQN vs Random** - Your DQN model (with/without opponent modeling)
5. **DQN vs PPO** - DQN vs PPO agent
6. **DQN vs MCTS** - DQN vs MCTS agent

### Opponent Modeling in Evaluation

When `--use-opponent-modeling` is enabled:

- **Tracks opponent actions** during evaluation
- **Computes behavior embeddings** on-the-fly
- **Uses embeddings** to enhance DQN decisions
- **Resets history** at start of each game

## Important Notes

### Model Compatibility

- Models trained **with** opponent modeling (`behavior_embedding_dim=16`) can be evaluated **with or without** opponent modeling
- Models trained **without** opponent modeling can only be evaluated **without** opponent modeling
- The evaluation script automatically handles this

### Performance Expectations

**With Opponent Modeling:**
- Should perform better against opponents with consistent patterns
- May perform similarly against random opponents (no patterns to exploit)
- Better adaptation to opponent behavior

**Without Opponent Modeling:**
- Baseline performance
- Works with any DQN model
- Faster evaluation (no embedding computation)

## Example Workflow

### 1. Train Model (Phase 2)
```bash
python train_dqn.py \
    --episodes 10000 \
    --use-opponent-modeling \
    --psychology-network poker-rl/output/models/psychology_network.pt \
    --behavior-embedding-dim 16 \
    --output poker-rl/output/models/dqn_with_psych.pt
```

### 2. Evaluate Without Opponent Modeling
```bash
python evaluate_dqn.py \
    --model poker-rl/output/models/dqn_with_psych.pt \
    --games 100
```

### 3. Evaluate With Opponent Modeling
```bash
python evaluate_dqn.py \
    --model poker-rl/output/models/dqn_with_psych.pt \
    --games 100 \
    --use-opponent-modeling \
    --psychology-network poker-rl/output/models/psychology_network.pt \
    --behavior-embedding-dim 16
```

### 4. Compare Results
- Compare win rates between evaluations
- Opponent modeling should help against patterned opponents
- May not help much against random opponents

## Troubleshooting

### Error: "Model incompatible: State size mismatch"

**Cause**: Model was trained with different state size (e.g., with/without opponent modeling)

**Solution**: 
- If model has `behavior_embedding_dim=16`, use `--use-opponent-modeling`
- If model has `behavior_embedding_dim=0`, don't use `--use-opponent-modeling`

### Error: "Could not load psychology network"

**Cause**: Psychology network path incorrect or model incompatible

**Solution**:
- Check path to psychology network
- Ensure it's the Phase 1 pretrained model
- Evaluation will continue without opponent modeling

### Warning: "Opponent modeling disabled"

**Cause**: `psycnet` module not available

**Solution**:
- Ensure `psycnet` module is in Python path
- Evaluation continues without opponent modeling

## Best Practices

1. **Always evaluate both ways** (with and without opponent modeling) to see the benefit
2. **Use consistent game counts** for fair comparisons
3. **Test against different opponents** (random, PPO, MCTS)
4. **Compare win rates and chip statistics**
5. **Monitor embedding statistics** if available

## Summary

✅ **Everything is linked together!**

- `evaluate_dqn.py` supports opponent modeling
- `run_all_evaluations.py` passes psychology network args
- Automatic backward compatibility
- Works with existing models

Just run:
```bash
python run_all_evaluations.py \
    --dqn-model poker-rl/output/models/dqn_poker_model.pt \
    --use-opponent-modeling \
    --psychology-network poker-rl/output/models/psychology_network.pt
```

And it will work! 🎯

