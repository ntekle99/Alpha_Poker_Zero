"""Training script for DQN poker bot."""

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from config import PokerConfig as cfg
from poker_game import PokerGame
from dqn_network import DQNNetwork
from dqn_trainer import DQNTrainer
from evaluate import get_random_action


def train_dqn(
    num_episodes=10000,
    model_path=None,
    output_path=None,
    save_interval=100,
    random_opponent_prob=0.5,
    use_opponent_modeling=False,
    psychology_network_path=None,
    behavior_embedding_dim=16,
    joint_finetuning=False,
    psychology_lr=1e-5
):
    """
    Train DQN agent.
    
    Args:
        num_episodes: Number of training episodes
        model_path: Path to load existing model (optional)
        output_path: Path to save model
        save_interval: Save model every N episodes
        random_opponent_prob: Probability of playing against random opponent
        use_opponent_modeling: Whether to use psychology network
        psychology_network_path: Path to pretrained psychology network
        behavior_embedding_dim: Dimension of behavior embedding
        joint_finetuning: Enable Phase 3 joint finetuning (unfreeze psych network)
        psychology_lr: Learning rate for psychology network (Phase 3 only)
    """
    # Setup paths
    if output_path is None:
        os.makedirs(cfg.SAVE_MODEL_PATH, exist_ok=True)
        output_path = os.path.join(cfg.SAVE_MODEL_PATH, "dqn_poker_model.pt")
    
    best_model_path = output_path.replace('.pt', '_best.pt')
    
    # Initialize network
    print("Initializing DQN network...")
    behavior_dim = behavior_embedding_dim if use_opponent_modeling else 0
    q_network = DQNNetwork(behavior_embedding_dim=behavior_dim)
    device = q_network.device
    print(f"Using device: {device}")
    
    # Load existing model if provided
    if model_path and os.path.exists(model_path):
        try:
            q_network.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting with random weights")
    
    # Initialize trainer
    print("Initializing DQN trainer...")
    trainer = DQNTrainer(
        q_network=q_network,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=100,
        use_opponent_modeling=use_opponent_modeling,
        psychology_network_path=psychology_network_path,
        behavior_embedding_dim=behavior_embedding_dim,
        joint_finetuning=joint_finetuning,
        psychology_lr=psychology_lr
    )
    
    # Initialize game
    game = PokerGame()
    
    # Statistics
    all_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': [],
        'q_values': []
    }
    
    best_mean_reward = float('-inf')
    
    print("\n" + "=" * 60)
    print("STARTING DQN TRAINING")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Learning rate: 1e-4")
    print(f"Random opponent probability: {random_opponent_prob * 100:.0f}%")
    print(f"Opponent modeling: {use_opponent_modeling}")
    if use_opponent_modeling:
        print(f"Behavior embedding dim: {behavior_embedding_dim}")
        print(f"Joint finetuning (Phase 3): {joint_finetuning}")
        if joint_finetuning:
            print(f"Psychology network LR: {psychology_lr}")
    print("=" * 60 + "\n")
    
    # Opponent ID tracking (for opponent modeling)
    OPPONENT_ID = -1  # Fixed ID for opponent
    
    for episode in tqdm(range(1, num_episodes + 1), desc="Training"):
        # Initialize episode
        state = game.init_new_hand()
        episode_reward = 0
        episode_length = 0
        
        # Reset opponent history for new episode
        if trainer.use_opponent_modeling:
            trainer.history_manager.reset_opponent(OPPONENT_ID)
        
        # Track previous stage for betting round detection
        previous_stage = state.stage
        
        # Play episode
        while not game.is_terminal(state):
            current_player = state.current_player
            
            # Only train on player 1's actions
            if current_player == 1:
                state_vector = game.get_canonical_state(state, current_player)
                valid_actions = game.get_valid_moves(state, current_player)
                
                # Get behavior embedding if opponent modeling is enabled
                z = None
                if trainer.use_opponent_modeling:
                    history = trainer.history_manager.get_history(OPPONENT_ID)
                    if trainer.history_manager.has_history(OPPONENT_ID):
                        z, _ = trainer.psychology_network.encode_opponent(history, return_supervised=False)
                
                # Select action
                action = trainer.select_action(
                    state_vector,
                    valid_actions,
                    training=True,
                    opponent_id=OPPONENT_ID if trainer.use_opponent_modeling else None
                )
                
                # Apply action
                next_state = game.apply_action(state, action, current_player)
                episode_length += 1
                
                # Get next state vector and reward
                if game.is_terminal(next_state):
                    next_state_vector = state_vector  # Terminal - use same state
                    done = True
                    reward = game.get_reward(next_state, player=1)
                    episode_reward = reward
                    next_z = None
                else:
                    next_state_vector = game.get_canonical_state(next_state, 1)
                    done = False
                    reward = 0.0  # Sparse rewards - only at terminal
                    # Get next embedding
                    next_z = None
                    if trainer.use_opponent_modeling:
                        next_history = trainer.history_manager.get_history(OPPONENT_ID)
                        if trainer.history_manager.has_history(OPPONENT_ID):
                            next_z, _ = trainer.psychology_network.encode_opponent(next_history, return_supervised=False)
                
                # Store experience in replay buffer
                trainer.replay_buffer.add(
                    state_vector,
                    action,
                    reward,
                    next_state_vector,
                    done,
                    z=z,
                    next_z=next_z
                )
                
                state = next_state
            else:
                # Opponent's turn
                if np.random.random() < random_opponent_prob:
                    # Random opponent
                    action = get_random_action(state, current_player)
                else:
                    # Use target network as opponent (self-play)
                    state_vector = game.get_canonical_state(state, current_player)
                    valid_actions = game.get_valid_moves(state, current_player)
                    action = trainer.select_action(state_vector, valid_actions, training=False)
                
                # Track opponent action for modeling
                if trainer.use_opponent_modeling:
                    # Determine action type and bet size
                    my_bet = state.player1_bet if current_player == -1 else state.player2_bet
                    to_call = state.current_bet - my_bet
                    
                    # Map action to bet size
                    bet_size = 0.0
                    is_all_in = (action == cfg.ACTION_ALL_IN)
                    if action == cfg.ACTION_BET_SMALL:
                        bet_size = state.pot * 0.5
                    elif action == cfg.ACTION_BET_MEDIUM:
                        bet_size = state.pot
                    elif action == cfg.ACTION_BET_LARGE:
                        bet_size = state.pot * 2
                    elif action == cfg.ACTION_ALL_IN:
                        my_stack = state.player1_stack if current_player == -1 else state.player2_stack
                        bet_size = my_stack - to_call
                    
                    # Detect new betting round
                    is_new_betting_round = (state.stage != previous_stage)
                    
                    # Get raise count (simplified - would need more tracking in real implementation)
                    raise_count = 0  # TODO: Track raise counts per betting round
                    
                    # Add opponent action to history
                    trainer.add_opponent_action(
                        opponent_id=OPPONENT_ID,
                        action=action,
                        bet_size=bet_size,
                        pot_size=state.pot,
                        stage=state.stage,
                        position=0,  # Simplified - would need actual position tracking
                        num_players=2,  # Heads-up
                        had_initiative=(state.current_player == 1),  # Simplified
                        to_call=to_call,
                        raise_count=raise_count,
                        is_all_in=is_all_in,
                        is_new_betting_round=is_new_betting_round
                    )
                
                state = game.apply_action(state, action, current_player)
                previous_stage = state.stage
        
        # Update Q-network
        update_stats = trainer.update()
        if update_stats:
            all_stats['losses'].append(update_stats['loss'])
            all_stats['q_values'].append(update_stats['q_value'])
        
        # Decay epsilon
        trainer.update_epsilon()
        
        # Store episode statistics
        all_stats['episode_rewards'].append(episode_reward)
        all_stats['episode_lengths'].append(episode_length)
        
        # Save best model
        if episode % 10 == 0:
            mean_reward = np.mean(all_stats['episode_rewards'][-100:])
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                torch.save(q_network.state_dict(), best_model_path)
                print(f"\n  New best model! (Mean reward: {mean_reward:.4f})")
        
        # Print statistics
        if episode % 100 == 0:
            mean_reward = np.mean(all_stats['episode_rewards'][-100:])
            mean_length = np.mean(all_stats['episode_lengths'][-100:])
            mean_loss = np.mean(all_stats['losses'][-100:]) if all_stats['losses'] else 0.0
            mean_q = np.mean(all_stats['q_values'][-100:]) if all_stats['q_values'] else 0.0
            
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"  Mean Reward (last 100): {mean_reward:.4f}")
            print(f"  Mean Length: {mean_length:.1f}")
            print(f"  Mean Loss: {mean_loss:.4f}")
            print(f"  Mean Q-value: {mean_q:.4f}")
            print(f"  Epsilon: {trainer.epsilon:.4f}")
            
            # Phase 3 stats
            if joint_finetuning and update_stats and 'psych_grad_norm' in update_stats:
                print(f"  Psych Grad Norm: {update_stats['psych_grad_norm']:.4f}")
        
        # Periodic saves
        if episode % save_interval == 0:
            checkpoint_path = output_path.replace('.pt', f'_episode_{episode}.pt')
            torch.save(q_network.state_dict(), checkpoint_path)
    
    # Save final model
    torch.save(q_network.state_dict(), output_path)
    print(f"\nTraining complete!")
    print(f"Final model saved to: {output_path}")
    print(f"Best model saved to: {best_model_path} (Mean reward: {best_mean_reward:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN poker bot")
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Number of training episodes (default: 10000)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to load existing model")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save model (default: poker-rl/output/models/dqn_poker_model.pt)")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Save model every N episodes (default: 100)")
    parser.add_argument("--random-opponent-prob", type=float, default=0.5,
                        help="Probability of random opponent (default: 0.5)")
    parser.add_argument("--use-opponent-modeling", action="store_true",
                        help="Enable opponent modeling with psychology network")
    parser.add_argument("--psychology-network", type=str, default=None,
                        help="Path to pretrained psychology network")
    parser.add_argument("--behavior-embedding-dim", type=int, default=16,
                        help="Dimension of behavior embedding (default: 16)")
    parser.add_argument("--joint-finetuning", action="store_true",
                        help="Enable Phase 3 joint finetuning (unfreeze psychology network)")
    parser.add_argument("--psychology-lr", type=float, default=1e-5,
                        help="Learning rate for psychology network in Phase 3 (default: 1e-5)")
    
    args = parser.parse_args()
    
    train_dqn(
        num_episodes=args.episodes,
        model_path=args.model,
        output_path=args.output,
        save_interval=args.save_interval,
        random_opponent_prob=args.random_opponent_prob,
        use_opponent_modeling=args.use_opponent_modeling,
        psychology_network_path=args.psychology_network,
        behavior_embedding_dim=args.behavior_embedding_dim,
        joint_finetuning=args.joint_finetuning,
        psychology_lr=args.psychology_lr
    )

