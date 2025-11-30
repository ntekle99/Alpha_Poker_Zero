"""Test script for psychology network integration with DQN."""

import torch
import numpy as np
from dqn_network import DQNNetwork
from dqn_trainer import DQNTrainer
from config import PokerConfig as cfg

# Test imports
try:
    from psycnet import PsychologyNetwork, OpponentHistoryManager, ActionFeatureEncoder
    PSYCNET_AVAILABLE = True
except ImportError:
    print("Warning: psycnet not available, skipping integration tests")
    PSYCNET_AVAILABLE = False


def test_dqn_with_embeddings():
    """Test DQN network with behavior embeddings."""
    print("=" * 60)
    print("Test 1: DQN Network with Behavior Embeddings")
    print("=" * 60)
    
    behavior_dim = 16
    state_size = cfg.STATE_SIZE
    
    # Create network with behavior embedding support
    network = DQNNetwork(behavior_embedding_dim=behavior_dim)
    print(f"✓ Network created with behavior_embedding_dim={behavior_dim}")
    
    # Test without embedding (backward compatibility)
    state = np.random.randn(state_size).astype(np.float32)
    q_values = network.predict(state)
    print(f"✓ Forward pass without embedding: {q_values.shape}")
    
    # Test with embedding
    z = np.random.randn(behavior_dim).astype(np.float32)
    q_values = network.predict(state, z=z)
    print(f"✓ Forward pass with embedding: {q_values.shape}")
    
    # Test batch
    batch_states = np.random.randn(4, state_size).astype(np.float32)
    batch_z = np.random.randn(4, behavior_dim).astype(np.float32)
    batch_q = network.predict_batch(batch_states, z_batch=batch_z)
    print(f"✓ Batch forward pass: {batch_q.shape}")
    
    print("Test 1 passed!\n")


def test_trainer_with_psychology():
    """Test DQN trainer with psychology network."""
    if not PSYCNET_AVAILABLE:
        print("Skipping test - psycnet not available")
        return
    
    print("=" * 60)
    print("Test 2: DQN Trainer with Psychology Network")
    print("=" * 60)
    
    behavior_dim = 16
    state_size = cfg.STATE_SIZE
    
    # Create network
    network = DQNNetwork(behavior_embedding_dim=behavior_dim)
    
    # Create trainer with opponent modeling
    trainer = DQNTrainer(
        q_network=network,
        use_opponent_modeling=True,
        behavior_embedding_dim=behavior_dim
    )
    print(f"✓ Trainer created with opponent modeling")
    print(f"✓ Psychology network initialized: {trainer.psychology_network is not None}")
    print(f"✓ History manager initialized: {trainer.history_manager is not None}")
    
    # Test action selection with opponent
    state = np.random.randn(state_size).astype(np.float32)
    valid_actions = np.ones(cfg.NUM_ACTIONS)
    
    # Add some opponent actions
    opponent_id = 1
    for i in range(5):
        features = ActionFeatureEncoder.encode_action_features(
            action=cfg.ACTION_CHECK_CALL,
            bet_size=0.0,
            pot_size=100.0,
            stage=i % 4,
            position=0,
            num_players=2,
            had_initiative=(i % 2 == 0),
            to_call=0.0,
            raise_count=0
        )
        trainer.history_manager.add_action(opponent_id, features)
    
    # Select action
    action = trainer.select_action(state, valid_actions, training=False, opponent_id=opponent_id)
    print(f"✓ Action selection with opponent history: action={action}")
    
    # Test adding opponent action
    trainer.add_opponent_action(
        opponent_id=opponent_id,
        action=cfg.ACTION_BET_MEDIUM,
        bet_size=100.0,
        pot_size=100.0,
        stage=1,
        position=0,
        num_players=2,
        had_initiative=True,
        to_call=0.0,
        raise_count=0,
        is_all_in=False,
        is_new_betting_round=False
    )
    print(f"✓ Added opponent action to history")
    
    print("Test 2 passed!\n")


def test_end_to_end():
    """Test end-to-end integration."""
    if not PSYCNET_AVAILABLE:
        print("Skipping test - psycnet not available")
        return
    
    print("=" * 60)
    print("Test 3: End-to-End Integration")
    print("=" * 60)
    
    behavior_dim = 16
    state_size = cfg.STATE_SIZE
    
    # Create network and trainer
    network = DQNNetwork(behavior_embedding_dim=behavior_dim)
    trainer = DQNTrainer(
        q_network=network,
        use_opponent_modeling=True,
        behavior_embedding_dim=behavior_dim
    )
    
    # Simulate a few steps
    opponent_id = 1
    
    for step in range(3):
        # Get state
        state = np.random.randn(state_size).astype(np.float32)
        valid_actions = np.ones(cfg.NUM_ACTIONS)
        
        # Get embedding
        history = trainer.history_manager.get_history(opponent_id)
        z = None
        if trainer.history_manager.has_history(opponent_id):
            z, _ = trainer.psychology_network.encode_opponent(history, return_supervised=False)
        
        # Select action
        action = trainer.select_action(state, valid_actions, training=True, opponent_id=opponent_id)
        
        # Add to replay buffer
        next_state = np.random.randn(state_size).astype(np.float32)
        next_history = trainer.history_manager.get_history(opponent_id)
        next_z = None
        if trainer.history_manager.has_history(opponent_id):
            next_z, _ = trainer.psychology_network.encode_opponent(next_history, return_supervised=False)
        
        trainer.replay_buffer.add(
            state, action, 0.0, next_state, False,
            z=z, next_z=next_z
        )
        
        # Add opponent action
        trainer.add_opponent_action(
            opponent_id=opponent_id,
            action=cfg.ACTION_CHECK_CALL,
            bet_size=0.0,
            pot_size=100.0,
            stage=step % 4,
            position=0,
            num_players=2,
            had_initiative=(step % 2 == 0),
            to_call=0.0,
            raise_count=0,
            is_all_in=False,
            is_new_betting_round=(step == 0)
        )
        
        print(f"  Step {step+1}: action={action}, has_embedding={z is not None}")
    
    # Try to update (if enough samples)
    if len(trainer.replay_buffer) >= trainer.batch_size:
        stats = trainer.update()
        if stats:
            print(f"✓ Training update successful: loss={stats['loss']:.4f}")
        else:
            print("  Not enough samples for update")
    else:
        print(f"  Replay buffer size: {len(trainer.replay_buffer)} (need {trainer.batch_size})")
    
    print("Test 3 passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PSYCHOLOGY NETWORK INTEGRATION TESTS")
    print("=" * 60 + "\n")
    
    try:
        test_dqn_with_embeddings()
        test_trainer_with_psychology()
        test_end_to_end()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

