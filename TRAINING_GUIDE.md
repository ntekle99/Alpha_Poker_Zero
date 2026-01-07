# Training Guide for Alpha-Poker-Zero

This guide explains how to train your poker bot effectively using the improved training pipeline.

## Overview

The training process follows an AlphaZero-style iterative loop:
1. **Self-Play**: Generate training data using MCTS
2. **Training**: Train the neural network on collected data
3. **Evaluation**: Test the trained model
4. **Iterate**: Use the trained model for next self-play iteration

## Quick Start

### Step 1: Generate Initial Training Data

```bash
# Generate 5000 games of self-play data (with random initial model)
python selfplay.py --games 5000

# This will create: poker-rl/output/pickles/poker_training_dataset.pkl
```

### Step 2: Train the Neural Network

```bash
# Train on the collected data
python train.py --dataset poker-rl/output/pickles/poker_training_dataset.pkl

# This will save:
# - poker-rl/output/models/poker_model.pt (final model)
# - poker-rl/output/models/poker_model_best.pt (best validation model)
```

### Step 3: Evaluate the Model

```bash
# Evaluate the trained model
python evaluate.py --model poker-rl/output/models/poker_model_best.pt --games 100
```

### Step 4: Iterate (Improved Training Loop)

```bash
# Use the trained model to generate better self-play data
python selfplay.py --games 5000 --model poker-rl/output/models/poker_model_best.pt

# Train on the new data (can continue from previous model)
python train.py --dataset poker-rl/output/pickles/poker_training_dataset.pkl --model poker-rl/output/models/poker_model_best.pt

# Evaluate again
python evaluate.py --model poker-rl/output/models/poker_model_best.pt --games 100
```

## Key Improvements Made

### 1. **Better Neural Network Architecture**
- **Deeper network**: 1024 → 512 → 256 → 128 (was 512 → 256 → 128)
- **Dropout regularization**: Prevents overfitting (0.3 dropout rate)
- **Deeper heads**: Policy and value heads now have 3 layers each
- **Better capacity**: ~2x more parameters for learning complex patterns

### 2. **Improved Training Configuration**
- **More MCTS simulations**: 400 (was 100) - better action quality
- **Larger dataset**: 10,000 samples (was 1,000) - more diverse training
- **More games**: 5,000 self-play games (was 1,000) - better coverage
- **Better hyperparameters**:
  - Learning rate: 0.0001 (was 0.001) - more stable training
  - Batch size: 128 (was 64) - better gradient estimates
  - Epochs: 200 (was 100) - more training time
  - Weight decay: 1e-4 - L2 regularization

### 3. **Professional Training Script**
- **Proper loss function**: Combined policy + value loss
- **Validation split**: 90% train / 10% validation
- **Learning rate scheduling**: Reduces LR when validation plateaus
- **Gradient clipping**: Prevents exploding gradients
- **Best model saving**: Automatically saves best validation model
- **Checkpointing**: Saves models periodically

### 4. **Evaluation Tools**
- **Model evaluation**: Test single model performance
- **Model comparison**: Head-to-head comparison of two models
- **Statistics**: Win rates, rewards, action counts

## Training Tips

### For Best Results:

1. **Start with Random Model**
   ```bash
   python selfplay.py --games 5000
   python train.py
   ```

2. **Iterate Multiple Times**
   - Each iteration improves the model
   - Use the best model from previous iteration
   - Generate new self-play data with improved model
   - Train on accumulated data

3. **Monitor Training**
   - Watch validation loss - should decrease
   - If validation loss increases, model is overfitting
   - Best model is saved automatically (lowest validation loss)

4. **Adjust Hyperparameters**
   - If training is unstable: lower learning rate
   - If overfitting: increase dropout or weight decay
   - If underfitting: train for more epochs or increase model size

5. **Use More MCTS Simulations for Evaluation**
   ```bash
   # Use more simulations for better evaluation
   python evaluate.py --model poker_model_best.pt --games 100 --simulations 800
   ```

## Advanced Usage

### Custom Training Parameters

```bash
python train.py \
    --dataset poker-rl/output/pickles/poker_training_dataset.pkl \
    --model poker-rl/output/models/poker_model_best.pt \
    --output poker-rl/output/models/poker_model_v2.pt \
    --epochs 300 \
    --batch-size 256 \
    --lr 0.00005 \
    --weight-decay 1e-3
```

### Generate High-Quality Self-Play Data

```bash
# Use more MCTS simulations for better quality (slower but better)
python selfplay.py --games 5000 --simulations 800 --model poker_model_best.pt
```

### Compare Models

```bash
# Compare two models head-to-head
python evaluate.py \
    --model poker-rl/output/models/poker_model_v1.pt \
    --compare poker-rl/output/models/poker_model_v2.pt \
    --games 200
```

## Expected Training Time

- **Self-play (5000 games, 400 sims)**: ~2-4 hours (CPU) or ~30-60 min (GPU)
- **Training (10k samples, 200 epochs)**: ~1-2 hours (CPU) or ~10-20 min (GPU)
- **Evaluation (100 games)**: ~5-10 minutes

## Troubleshooting

### Problem: Training loss not decreasing
- **Solution**: Lower learning rate, check data quality, ensure enough data

### Problem: Validation loss increasing (overfitting)
- **Solution**: Increase dropout, increase weight decay, reduce model size, or use more data

### Problem: Model plays poorly
- **Solution**: 
  - Generate more self-play data
  - Use more MCTS simulations
  - Train for more epochs
  - Check if model is actually being used in self-play

### Problem: Out of memory
- **Solution**: Reduce batch size, reduce dataset size, or use CPU instead of GPU

## Next Steps

1. **Implement curriculum learning**: Start with simpler scenarios
2. **Add data augmentation**: Rotate card representations
3. **Implement experience replay**: Keep older good games
4. **Add regularization**: L1, dropout scheduling
5. **Hyperparameter tuning**: Grid search or Bayesian optimization

## Monitoring Progress

Track these metrics:
- **Win rate**: Should improve over iterations
- **Validation loss**: Should decrease and stabilize
- **Reward distribution**: Should become more balanced
- **Action diversity**: Model should explore different strategies


