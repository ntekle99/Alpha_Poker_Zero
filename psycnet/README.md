# Psychology Network (Opponent Modeling) Module

This module implements an opponent modeling system ("psychology network") for the Alpha-Poker-Zero DQN architecture. The system learns behavior embeddings for each opponent based on their action history.

## Architecture Overview

The psychology network takes sequences of opponent actions and produces a behavior embedding vector `z` that captures the opponent's playing style. This embedding is then used as additional input to the DQN network.

### Components

1. **PsychologyNetwork** (`opponent_model.py`)
   - LSTM-based architecture that processes action sequences
   - Produces behavior embeddings (dimension 8-32)
   - Optional supervised head for pretraining (predicts action frequencies)

2. **OpponentHistoryManager** (`opponent_history.py`)
   - Maintains rolling windows of opponent actions
   - Manages action feature encoding
   - Handles history padding/truncation

3. **ActionFeatureEncoder** (`opponent_history.py`)
   - Encodes poker actions into 7-dimensional feature vectors:
     - Action type (fold, check, call, bet, raise, all-in)
     - Bet size bucket (0-5)
     - Street (preflop, flop, turn, river)
     - Position (UTG, MP, CO, BTN, SB, BB)
     - Number of players (HU, 3-way, 4-way, 5+)
     - Initiative (binary)
     - Facing action bucket (0-7)

4. **OpponentDatasetGenerator** (`opponent_dataset.py`)
   - Generates synthetic opponent histories for pretraining
   - Computes frequency labels (fold%, call%, raise%, big_bet%)
   - Supports multiple player types (tight, loose, aggressive, passive)

## Phase 1: Pretraining

Run supervised pretraining to initialize the psychology network:

```bash
python -m psycnet.train_opponent_model \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --train-samples 10000 \
    --val-samples 2000 \
    --save-path poker-rl/output/models/psychology_network.pt
```

This will:
1. Generate synthetic opponent histories
2. Compute frequency labels
3. Train the network to predict these frequencies
4. Save pretrained weights

## Phase 2: Integration (TODO)

The next phase involves:
1. Modifying `dqn_network.py` to accept behavior embeddings `z`
2. Updating `dqn_trainer.py` to include psychology network forward passes
3. Modifying self-play code to maintain opponent histories and compute embeddings

## Phase 3: Joint Finetuning (TODO)

During DQN training:
- Allow gradients to flow through `z` into the psychology network
- Use different learning rates for DQN vs psychology network
- Ensure stability (psych network shouldn't drift too much)

## Usage Example

```python
from psycnet import PsychologyNetwork, OpponentHistoryManager, ActionFeatureEncoder

# Initialize components
model = PsychologyNetwork(behavior_embedding_dim=16)
history_manager = OpponentHistoryManager(max_history_length=30)

# Add opponent actions
for action in opponent_actions:
    features = ActionFeatureEncoder.encode_action_features(...)
    history_manager.add_action(opponent_id=1, action_features=features)

# Get behavior embedding
history = history_manager.get_history(opponent_id=1)
z, _ = model.encode_opponent(history, return_supervised=False)

# Use z as additional input to DQN network
```

## File Structure

```
psycnet/
├── __init__.py                 # Module exports
├── opponent_model.py           # Psychology network architecture
├── opponent_history.py         # History management and feature encoding
├── opponent_dataset.py         # Dataset generation for pretraining
├── train_opponent_model.py     # Phase 1 training script
└── README.md                   # This file
```

## Notes

- The psychology network is designed to be backward-compatible: if no embedding is provided, the DQN network should use zeros
- History length K is typically 20-40 actions
- Behavior embedding dimension is typically 8-32
- The supervised head is only used during Phase 1 pretraining

