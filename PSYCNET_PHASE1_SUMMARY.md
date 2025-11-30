# Psychology Network Phase 1 Implementation Summary

## Overview

This document summarizes the Phase 1 implementation of the opponent modeling ("psychology network") system for Alpha-Poker-Zero. Phase 1 focuses on creating the core components and supervised pretraining infrastructure.

## Files Created

### 1. `psycnet/opponent_model.py`
**Psychology Network Architecture**

- **PsychologyNetwork** class: LSTM-based neural network that processes action sequences
  - Embedding layers for 7 categorical features
  - Bidirectional LSTM (configurable layers, hidden dim)
  - Behavior embedding head: produces `z` vector (dimension 8-32)
  - Supervised head: predicts action frequencies (fold%, call%, raise%, big_bet%)
  - Methods:
    - `forward()`: Process batch of action sequences
    - `encode_opponent()`: Encode single opponent history to embedding

**Key Features:**
- Embedding dimension: 32 (default)
- LSTM hidden dimension: 128 (default)
- Behavior embedding dimension: 16 (default)
- Dropout: 0.2 (default)
- Supports both training and inference modes

### 2. `psycnet/opponent_history.py`
**History Management and Feature Encoding**

- **ActionFeatureEncoder** class: Encodes poker actions into 7D feature vectors
  - `action_type`: 0=fold, 1=check, 2=call, 3=bet, 4=raise, 5=all-in
  - `bet_size_bucket`: 0=none, 1=small, 2=medium, 3=large, 4=overbet, 5=all-in
  - `street`: 0=preflop, 1=flop, 2=turn, 3=river
  - `position`: 0=UTG, 1=MP, 2=CO, 3=BTN, 4=SB, 5=BB
  - `num_players_bucket`: 0=HU, 1=3-way, 2=4-way, 3=5+
  - `initiative`: 0=no, 1=yes
  - `facing_action_bucket`: 0=none, 1-4=bet sizes, 5=raise, 6=3bet, 7=4bet+

- **OpponentHistoryManager** class: Maintains rolling action histories
  - Per-opponent deque-based storage
  - Configurable max history length (K = 20-40)
  - Automatic padding/truncation
  - Tracks raise counts per betting round

**Key Methods:**
- `add_action()`: Add action to opponent history
- `get_history()`: Retrieve padded history array (K, 7)
- `reset_opponent()`: Clear history for specific opponent

### 3. `psycnet/opponent_dataset.py`
**Synthetic Data Generation**

- **OpponentDatasetGenerator** class: Generates training data
  - `generate_synthetic_history()`: Create single opponent history with labels
  - `generate_dataset()`: Generate batch of histories and frequency labels
  - Supports player types: random, tight, loose, aggressive, passive
  - Computes frequency labels: [fold_pct, call_pct, raise_pct, big_bet_pct]

- **OpponentDataset** class: PyTorch-style dataset wrapper
  - Supports indexing and batching
  - Compatible with DataLoader

**Key Features:**
- Configurable history lengths (10-40 actions)
- Multiple player behavior profiles
- Automatic frequency computation
- Ready for PyTorch training pipeline

### 4. `psycnet/train_opponent_model.py`
**Phase 1 Training Script**

- Standalone training script for supervised pretraining
- Command-line interface with argparse
- Generates synthetic dataset
- Trains network to predict action frequencies
- Saves best model based on validation loss
- Includes test encoding at end of training

**Usage:**
```bash
python -m psycnet.train_opponent_model \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --train-samples 10000 \
    --val-samples 2000 \
    --save-path poker-rl/output/models/psychology_network.pt
```

### 5. `psycnet/__init__.py`
**Module initialization with exports**

### 6. `psycnet/README.md`
**Documentation for the module**

## Architecture Details

### Input Format
- Action sequences: (batch_size, K, 7) where K = max_history_length
- Each timestep is a 7D feature vector (see ActionFeatureEncoder)

### Output Format
- Behavior embedding `z`: (batch_size, behavior_embedding_dim)
- Supervised predictions: (batch_size, 4) - [fold_pct, call_pct, raise_pct, big_bet_pct]

### Network Flow
1. Embed categorical features (7 features × embedding_dim)
2. Concatenate embeddings → (batch_size, K, 7×embedding_dim)
3. LSTM processing → (batch_size, K, lstm_hidden_dim)
4. Extract final hidden state → (batch_size, lstm_hidden_dim)
5. Behavior embedding head → (batch_size, behavior_embedding_dim)
6. (Optional) Supervised head → (batch_size, 4)

## Integration Points (Phase 2)

The following files will need modification for Phase 2:

1. **dqn_network.py**
   - Modify `forward()` to accept optional `z` parameter
   - Concatenate `z` to state input
   - Maintain backward compatibility (default zeros if `z` not provided)

2. **dqn_trainer.py**
   - Initialize psychology network
   - Maintain opponent histories during training
   - Compute embeddings before DQN forward pass
   - Pass embeddings to DQN network

3. **selfplay.py** (or equivalent)
   - Track opponent actions during games
   - Update history manager
   - Compute embeddings for action selection

## Testing

Each module includes `__main__` blocks for basic testing:
- `opponent_model.py`: Test network forward pass and encoding
- `opponent_history.py`: Test feature encoding and history management
- `opponent_dataset.py`: Test dataset generation

Run tests with:
```bash
python -m psycnet.opponent_model
python -m psycnet.opponent_history
python -m psycnet.opponent_dataset
```

## Next Steps (Phase 2)

1. **Modify DQN Network**
   - Update `dqn_network.py` to accept behavior embeddings
   - Ensure backward compatibility

2. **Integrate with Training**
   - Update `dqn_trainer.py` to use psychology network
   - Maintain opponent histories during self-play
   - Pass embeddings to DQN during action selection

3. **Update Self-Play**
   - Track opponent actions in game loop
   - Update history manager after each action
   - Compute embeddings before DQN forward pass

4. **Phase 3: Joint Finetuning**
   - Enable gradient flow through psychology network
   - Use different learning rates for DQN vs psych network
   - Monitor stability and adjust hyperparameters

## Notes

- All modules use relative imports with fallback for standalone execution
- History length K is configurable (default 30)
- Behavior embedding dimension is configurable (default 16)
- Supervised head is only used during Phase 1 pretraining
- The system is designed to be backward-compatible with existing DQN code

## Dependencies

- PyTorch
- NumPy
- tqdm (for training progress)
- Existing codebase modules: `config.py`, `card.py`, etc.

