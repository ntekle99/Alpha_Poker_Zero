"""PPO training script for competitive poker bot."""

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from collections import deque
from config import PokerConfig as cfg
from poker_game import PokerGame, PokerGameState
from value_policy_network import PokerValuePolicyNetwork
from ppo_trainer import PPOTrainer
from evaluate import get_random_action


class OpponentPool:
    """Pool of opponent models for diverse self-play."""
    
    def __init__(self, pool_size=4):
        self.pool_size = pool_size
        self.opponents = deque(maxlen=pool_size)
        self.opponent_networks = []
    
    def add_opponent(self, model_state_dict):
        """Add a model checkpoint to the pool."""
        self.opponents.append(model_state_dict)
        # Create network for this opponent
        network = PokerValuePolicyNetwork()
        network.load_state_dict(model_state_dict)
        network.eval()
        self.opponent_networks.append(network)
    
    def get_opponent(self):
        """Get a random opponent from the pool."""
        if len(self.opponents) == 0:
            return None
        idx = np.random.randint(len(self.opponents))
        return self.opponent_networks[idx]
    
    def clear(self):
        """Clear the pool."""
        self.opponents.clear()
        self.opponent_networks.clear()


def collect_rollout(
    game: PokerGame,
    trainer: PPOTrainer,
    opponent_network=None,
    rollout_length=2048,
    use_opponent_pool=False,
    use_random_opponent=False
):
    """
    Collect a rollout of experiences.
    
    Args:
        game: PokerGame instance
        trainer: PPOTrainer instance
        opponent_network: Opponent network (if None, uses same network)
        rollout_length: Number of steps to collect
        use_opponent_pool: Whether to use opponent pool
    
    Returns:
        Dictionary of rollout statistics
    """
    stats = {
        'episodes': 0,
        'total_reward': 0.0,
        'episode_rewards': [],
        'episode_lengths': []
    }
    
    steps_collected = 0
    episode_start_idx = 0  # Track where current episode started in buffer
    
    # Initialize hand
    state = game.init_new_hand()
    
    while steps_collected < rollout_length:
        current_player = state.current_player
        
        # Get state representation
        state_vector = game.get_canonical_state(state, current_player)
        valid_actions = game.get_valid_moves(state, current_player)
        
        # Select action
        if current_player == -1:
            # Player 2 (opponent)
            if use_random_opponent:
                # Use random opponent
                action = get_random_action(state, current_player)
                log_prob = 0.0  # Not needed for random
                value = 0.0
            elif opponent_network is not None:
                # Use opponent network for player 2
                opponent_network.eval()
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(opponent_network.device)
                    policy, _ = opponent_network(state_tensor)
                    policy = policy.cpu().numpy()[0]
                    policy = policy * valid_actions
                    if policy.sum() > 0:
                        policy = policy / policy.sum()
                    else:
                        policy = valid_actions / valid_actions.sum()
                    action = np.random.choice(len(policy), p=policy)
                    log_prob = np.log(policy[action] + 1e-8)
                    value = 0.0  # Don't need value for opponent
            else:
                # Use training network (self-play)
                action, log_prob, value = trainer.select_action(state_vector, valid_actions, deterministic=False)
        else:
            # Player 1 (training player)
            action, log_prob, value = trainer.select_action(state_vector, valid_actions, deterministic=False)
        
        # Store experience (only for training player - player 1)
        # Store for player 1 always, and for player 2 only in self-play mode
        if current_player == 1 or (opponent_network is None and not use_random_opponent):
            trainer.buffer.add(
                state=state_vector,
                action=action,
                reward=0.0,  # Will be set at end of episode
                log_prob=log_prob,
                value=value,
                done=False
            )
        
        # Apply action
        state = game.apply_action(state, action, current_player)
        steps_collected += 1
        
        # Check if episode is done
        if game.is_terminal(state):
            # Get final rewards
            reward_p1 = game.get_reward(state, player=1)
            
            # Update rewards for all steps in this episode
            # Only update steps that belong to training player
            if opponent_network is None and not use_random_opponent:
                # Both players use same network, update all steps
                episode_steps = len(trainer.buffer.rewards) - episode_start_idx
                for i in range(episode_start_idx, len(trainer.buffer.rewards)):
                    # Assign reward based on which player made the action
                    # For simplicity, assign final reward to all steps (GAE will handle discounting)
                    trainer.buffer.rewards[i] = reward_p1
                trainer.buffer.dones[-1] = True
            else:
                # Only player 1 is training, update only their steps
                # Count how many steps player 1 took in this episode
                episode_steps = len(trainer.buffer.rewards) - episode_start_idx
                for i in range(episode_start_idx, len(trainer.buffer.rewards)):
                    trainer.buffer.rewards[i] = reward_p1
                trainer.buffer.dones[-1] = True
            
            stats['episodes'] += 1
            stats['total_reward'] += reward_p1
            stats['episode_rewards'].append(reward_p1)
            stats['episode_lengths'].append(steps_collected - episode_start_idx)
            
            # Reset for new episode
            state = game.init_new_hand()
            episode_start_idx = len(trainer.buffer.rewards)
    
    return stats


def train_ppo(
    num_updates=1000,
    rollout_length=None,
    model_path=None,
    output_path=None,
    save_interval=50,
    opponent_pool_size=None,
    keep_checkpoints=5
):
    """
    Train poker bot using PPO.
    
    Args:
        num_updates: Number of PPO update iterations
        rollout_length: Steps per rollout
        model_path: Path to pre-trained model
        output_path: Path to save trained model
        save_interval: Save model every N updates
        opponent_pool_size: Size of opponent pool
    """
    if rollout_length is None:
        rollout_length = cfg.PPO_ROLLOUT_LENGTH
    if output_path is None:
        output_path = os.path.join(cfg.SAVE_MODEL_PATH, "ppo_poker_model.pt")
    if opponent_pool_size is None:
        opponent_pool_size = cfg.PPO_OPPONENT_POOL_SIZE
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Initialize network
    print("Initializing policy network...")
    policy_network = PokerValuePolicyNetwork()
    device = policy_network.device
    print(f"Using device: {device}")
    
    # Load pre-trained model if provided
    if model_path and os.path.exists(model_path):
        try:
            policy_network.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded pre-trained model from {model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting with random weights")
    
    # Initialize PPO trainer
    print("Initializing PPO trainer...")
    trainer = PPOTrainer(
        policy_network=policy_network,
        learning_rate=cfg.PPO_LEARNING_RATE,
        clip_epsilon=cfg.PPO_CLIP_EPSILON,
        value_coef=cfg.PPO_VALUE_COEF,
        entropy_coef=cfg.PPO_ENTROPY_COEF,
        max_grad_norm=cfg.PPO_MAX_GRAD_NORM,
        gamma=cfg.PPO_GAMMA,
        gae_lambda=cfg.PPO_GAE_LAMBDA,
        ppo_epochs=cfg.PPO_EPOCHS,
        batch_size=cfg.PPO_BATCH_SIZE,
        device=device
    )
    
    # Initialize opponent pool
    opponent_pool = OpponentPool(pool_size=opponent_pool_size)
    
    # Initialize game
    game = PokerGame()
    
    # Training statistics
    all_stats = {
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'total_loss': [],
        'episode_rewards': [],
        'episode_lengths': []
    }
    
    print("\n" + "=" * 60)
    print("STARTING PPO TRAINING")
    print("=" * 60)
    print(f"Updates: {num_updates}")
    print(f"Rollout length: {rollout_length}")
    print(f"PPO epochs: {cfg.PPO_EPOCHS}")
    print(f"Batch size: {cfg.PPO_BATCH_SIZE}")
    print(f"Learning rate: {cfg.PPO_LEARNING_RATE}")
    print(f"Random opponent probability: {cfg.PPO_RANDOM_OPPONENT_PROB * 100:.0f}%")
    print("=" * 60 + "\n")
    
    best_mean_reward = float('-inf')
    best_model_path = output_path.replace('.pt', '_best.pt')
    
    for update in tqdm(range(1, num_updates + 1), desc="Training"):
        # Select opponent - sometimes use random to prevent distribution shift
        use_random = np.random.random() < cfg.PPO_RANDOM_OPPONENT_PROB
        opponent_network = None
        if not use_random and len(opponent_pool.opponents) > 0 and update % cfg.PPO_OPPONENT_UPDATE_FREQ != 0:
            opponent_network = opponent_pool.get_opponent()
        
        # Collect rollout
        rollout_stats = collect_rollout(
            game=game,
            trainer=trainer,
            opponent_network=opponent_network,
            rollout_length=rollout_length,
            use_random_opponent=use_random
        )
        
        # Update policy
        update_stats = trainer.update()
        
        # Update statistics
        if update_stats:
            all_stats['policy_loss'].append(update_stats['policy_loss'])
            all_stats['value_loss'].append(update_stats['value_loss'])
            all_stats['entropy'].append(update_stats['entropy'])
            all_stats['total_loss'].append(update_stats['total_loss'])
        
        all_stats['episode_rewards'].extend(rollout_stats['episode_rewards'])
        all_stats['episode_lengths'].extend(rollout_stats['episode_lengths'])
        
        # Update opponent pool
        if update % cfg.PPO_OPPONENT_UPDATE_FREQ == 0:
            opponent_pool.add_opponent(policy_network.state_dict().copy())
        
        # Print statistics
        if update % 10 == 0:
            recent_rewards = all_stats['episode_rewards'][-100:] if len(all_stats['episode_rewards']) >= 100 else all_stats['episode_rewards']
            mean_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            
            print(f"\nUpdate {update}/{num_updates}")
            if update_stats:
                print(f"  Policy Loss: {update_stats['policy_loss']:.6f}")
                print(f"  Value Loss: {update_stats['value_loss']:.6f}")
                print(f"  Entropy: {update_stats['entropy']:.6f}")
                print(f"  Total Loss: {update_stats['total_loss']:.6f}")
            print(f"  Mean Reward (last 100): {mean_reward:.4f}")
            print(f"  Episodes: {rollout_stats['episodes']}")
            
            # Save best model
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                torch.save(policy_network.state_dict(), best_model_path)
                print(f"  New best model! (Mean reward: {mean_reward:.4f})")
        
        # Periodic saves
        if update % save_interval == 0:
            checkpoint_path = output_path.replace('.pt', f'_update_{update}.pt')
            torch.save(policy_network.state_dict(), checkpoint_path)
            
            # Clean up old checkpoints (keep only the most recent N)
            if keep_checkpoints > 0:
                import glob
                checkpoint_pattern = output_path.replace('.pt', '_update_*.pt')
                checkpoints = sorted(glob.glob(checkpoint_pattern), key=os.path.getmtime)
                if len(checkpoints) > keep_checkpoints:
                    for old_checkpoint in checkpoints[:-keep_checkpoints]:
                        try:
                            os.remove(old_checkpoint)
                        except:
                            pass
    
    # Save final model
    torch.save(policy_network.state_dict(), output_path)
    print(f"\nTraining complete!")
    print(f"Final model saved to: {output_path}")
    print(f"Best model saved to: {best_model_path} (Mean reward: {best_mean_reward:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train poker bot with PPO")
    parser.add_argument("--updates", type=int, default=1000,
                        help="Number of PPO update iterations (default: 1000)")
    parser.add_argument("--rollout-length", type=int, default=None,
                        help=f"Steps per rollout (default: {cfg.PPO_ROLLOUT_LENGTH})")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to pre-trained model (optional)")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Output path for trained model (default: {cfg.SAVE_MODEL_PATH}/ppo_poker_model.pt)")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save checkpoint every N updates (default: 50)")
    parser.add_argument("--opponent-pool-size", type=int, default=None,
                        help=f"Size of opponent pool (default: {cfg.PPO_OPPONENT_POOL_SIZE})")
    parser.add_argument("--keep-checkpoints", type=int, default=5,
                        help="Number of checkpoint files to keep (default: 5, set to 0 to keep all)")
    
    args = parser.parse_args()
    
    train_ppo(
        num_updates=args.updates,
        rollout_length=args.rollout_length,
        model_path=args.model,
        output_path=args.output,
        save_interval=args.save_interval,
        opponent_pool_size=args.opponent_pool_size,
        keep_checkpoints=args.keep_checkpoints
    )

