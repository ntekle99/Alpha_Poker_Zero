"""DQN (Deep Q-Network) trainer for poker."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
from config import PokerConfig as cfg

# Import psychology network components
try:
    from psycnet import PsychologyNetwork, OpponentHistoryManager, ActionFeatureEncoder
    PSYCNET_AVAILABLE = True
except ImportError:
    PSYCNET_AVAILABLE = False
    print("Warning: psycnet module not available. Opponent modeling disabled.")


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity=100000, store_embeddings=False):
        self.capacity = capacity
        self.store_embeddings = store_embeddings
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done, z=None, next_z=None):
        """
        Add a single experience.
        
        Args:
            state: State vector
            action: Action index
            reward: Reward
            next_state: Next state vector
            done: Terminal flag
            z: Optional behavior embedding for state
            next_z: Optional behavior embedding for next_state
        """
        if self.store_embeddings:
            self.buffer.append((state, action, reward, next_state, done, z, next_z))
        else:
            self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        if self.store_embeddings and len(batch) > 0 and len(batch[0]) > 5:
            states = np.array([e[0] for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([e[3] for e in batch])
            dones = np.array([e[4] for e in batch])
            z_batch = [e[5] for e in batch]
            next_z_batch = [e[6] if len(e) > 6 else None for e in batch]
            return states, actions, rewards, next_states, dones, z_batch, next_z_batch
        else:
            states = np.array([e[0] for e in batch])
            actions = np.array([e[1] for e in batch])
            rewards = np.array([e[2] for e in batch])
            next_states = np.array([e[3] for e in batch])
            dones = np.array([e[4] for e in batch])
            return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    """DQN trainer for poker."""
    
    def __init__(
        self,
        q_network,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=100,
        device=None,
        use_opponent_modeling=False,
        psychology_network_path=None,
        behavior_embedding_dim=16,
        joint_finetuning=False,
        psychology_lr=1e-5
    ):
        """
        Initialize DQN trainer.
        
        Args:
            q_network: Q-network (DQNNetwork)
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Starting epsilon for epsilon-greedy
            epsilon_end: Minimum epsilon
            epsilon_decay: Epsilon decay rate
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network
            device: Device to use
            use_opponent_modeling: Whether to use psychology network
            psychology_network_path: Path to pretrained psychology network
            behavior_embedding_dim: Dimension of behavior embedding
            joint_finetuning: Whether to enable Phase 3 joint finetuning (unfreeze psych network)
            psychology_lr: Learning rate for psychology network (Phase 3 only)
        """
        self.q_network = q_network
        self.device = device or q_network.device
        self.q_network.to(self.device)
        
        # Create target network (copy of main network with same parameters)
        # Need to preserve behavior_embedding_dim for proper initialization
        behavior_dim = getattr(q_network, 'behavior_embedding_dim', 0)
        self.target_network = type(q_network)(behavior_embedding_dim=behavior_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.to(self.device)
        self.target_network.eval()
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Opponent modeling setup
        self.use_opponent_modeling = use_opponent_modeling and PSYCNET_AVAILABLE
        self.behavior_embedding_dim = behavior_embedding_dim if self.use_opponent_modeling else 0
        
        if self.use_opponent_modeling:
            # Initialize psychology network
            self.psychology_network = PsychologyNetwork(
                behavior_embedding_dim=behavior_embedding_dim,
                use_supervised_head=False  # Not needed during DQN training
            )
            self.psychology_network.to(self.device)
            
            # Phase 3: Joint finetuning - unfreeze psychology network
            self.joint_finetuning = joint_finetuning
            if joint_finetuning:
                self.psychology_network.train()  # Unfreeze for joint training
                print(f"Phase 3: Joint finetuning enabled (psych LR: {psychology_lr})")
            else:
                self.psychology_network.eval()  # Freeze during Phase 2
            
            # Load pretrained weights if provided
            if psychology_network_path:
                try:
                    state_dict = torch.load(psychology_network_path, map_location=self.device)
                    # Filter out supervised head weights (not needed for Phase 2)
                    filtered_state_dict = {
                        k: v for k, v in state_dict.items()
                        if not k.startswith('supervised_')
                    }
                    self.psychology_network.load_state_dict(filtered_state_dict, strict=False)
                    print(f"Loaded psychology network from {psychology_network_path}")
                except Exception as e:
                    print(f"Warning: Could not load psychology network: {e}")
                    print("Continuing with randomly initialized weights")
            
            # Initialize history manager
            self.history_manager = OpponentHistoryManager(max_history_length=30)
            print(f"Opponent modeling enabled (embedding dim: {behavior_embedding_dim})")
        else:
            self.psychology_network = None
            self.history_manager = None
        
        # Optimizers
        # Q-network optimizer (always active)
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
            eps=1e-5
        )
        
        # Psychology network optimizer (Phase 3 only)
        if self.use_opponent_modeling and self.joint_finetuning:
            self.psychology_optimizer = optim.Adam(
                self.psychology_network.parameters(),
                lr=psychology_lr,
                eps=1e-5
            )
            print(f"Psychology network optimizer created (LR: {psychology_lr})")
        else:
            self.psychology_optimizer = None
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000, store_embeddings=self.use_opponent_modeling)
    
    def select_action(
        self,
        state: np.ndarray,
        valid_actions: np.ndarray,
        training: bool = True,
        opponent_id: Optional[int] = None
    ):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: State vector
            valid_actions: Binary mask of valid actions
            training: If True, use epsilon-greedy; else greedy
            opponent_id: Optional opponent ID for behavior embedding
        
        Returns:
            Action index
        """
        if training and np.random.random() < self.epsilon:
            # Random action (exploration)
            valid_indices = np.where(valid_actions == 1)[0]
            return np.random.choice(valid_indices)
        else:
            # Greedy action (exploitation)
            self.q_network.eval()
            with torch.no_grad():
                # Get behavior embedding if opponent modeling is enabled
                z = None
                if self.use_opponent_modeling and opponent_id is not None:
                    history = self.history_manager.get_history(opponent_id)
                    if self.history_manager.has_history(opponent_id):
                        z, _ = self.psychology_network.encode_opponent(history, return_supervised=False)
                
                state_tensor = torch.FloatTensor(state).to(self.device)
                z_tensor = torch.FloatTensor(z).to(self.device) if z is not None else None
                q_values = self.q_network(state_tensor, z_tensor)
                q_values = q_values.cpu().numpy()
                if q_values.ndim > 1:
                    q_values = q_values[0]
                
                # Mask invalid actions
                q_values = q_values * valid_actions
                q_values = np.where(valid_actions == 1, q_values, -np.inf)
                
                action = np.argmax(q_values)
            
            return action
    
    def update_epsilon(self):
        """Decay epsilon for exploration."""
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update(self):
        """
        Update the Q-network using experience replay.
        
        Returns:
            Dictionary of training statistics
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        if self.use_opponent_modeling and self.replay_buffer.store_embeddings:
            states, actions, rewards, next_states, dones, z_batch, next_z_batch = self.replay_buffer.sample(self.batch_size)
        else:
            result = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = result[:5]
            z_batch, next_z_batch = None, None
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Prepare behavior embeddings if available
        z_tensor = None
        next_z_tensor = None
        if self.use_opponent_modeling and z_batch is not None:
            # Create embeddings array (use zeros for None values)
            z_array = []
            next_z_array = []
            for i in range(len(z_batch)):
                if z_batch[i] is not None:
                    z_array.append(z_batch[i])
                else:
                    z_array.append(np.zeros(self.behavior_embedding_dim, dtype=np.float32))
                
                if next_z_batch is not None and next_z_batch[i] is not None:
                    next_z_array.append(next_z_batch[i])
                else:
                    next_z_array.append(np.zeros(self.behavior_embedding_dim, dtype=np.float32))
            
            z_tensor = torch.FloatTensor(np.stack(z_array)).to(self.device)
            next_z_tensor = torch.FloatTensor(np.stack(next_z_array)).to(self.device)
        
        # Current Q-values
        self.q_network.train()
        # Set psychology network to train mode if joint finetuning
        if self.use_opponent_modeling and self.joint_finetuning:
            self.psychology_network.train()
        
        current_q_values = self.q_network(states_tensor, z_tensor)
        current_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor, next_z_tensor)
            # Get max Q-value for next state (mask invalid actions if needed)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        if self.psychology_optimizer is not None:
            self.psychology_optimizer.zero_grad()
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        if self.psychology_optimizer is not None:
            # Use smaller gradient norm for psychology network (stability)
            torch.nn.utils.clip_grad_norm_(self.psychology_network.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        if self.psychology_optimizer is not None:
            self.psychology_optimizer.step()
        
        # Set psychology network back to eval mode if not joint finetuning
        if self.use_opponent_modeling and not self.joint_finetuning:
            self.psychology_network.eval()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        stats = {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_value': current_q_values.mean().item()
        }
        
        # Add psychology network stats if joint finetuning
        if self.psychology_optimizer is not None:
            psych_grad_norm = 0.0
            if self.psychology_network.training:
                for p in self.psychology_network.parameters():
                    if p.grad is not None:
                        psych_grad_norm += p.grad.data.norm(2).item() ** 2
                psych_grad_norm = psych_grad_norm ** 0.5
            stats['psych_grad_norm'] = psych_grad_norm
        
        return stats
    
    def add_opponent_action(
        self,
        opponent_id: int,
        action: int,
        bet_size: float,
        pot_size: float,
        stage: int,
        position: int,
        num_players: int,
        had_initiative: bool,
        to_call: float,
        raise_count: int,
        is_all_in: bool = False,
        is_new_betting_round: bool = False
    ):
        """
        Add opponent action to history for behavior modeling.
        
        Args:
            opponent_id: Unique opponent identifier
            action: Action index from config
            bet_size: Bet size in chips
            pot_size: Pot size in chips
            stage: Game stage (0-3)
            position: Player position
            num_players: Number of players
            had_initiative: Whether opponent had initiative
            to_call: Amount to call
            raise_count: Number of raises in betting round
            is_all_in: Whether action was all-in
            is_new_betting_round: Whether this starts a new betting round
        """
        if not self.use_opponent_modeling:
            return
        
        # Encode action features
        features = ActionFeatureEncoder.encode_action_features(
            action=action,
            bet_size=bet_size,
            pot_size=pot_size,
            stage=stage,
            position=position,
            num_players=num_players,
            had_initiative=had_initiative,
            to_call=to_call,
            raise_count=raise_count,
            is_all_in=is_all_in
        )
        
        # Add to history
        self.history_manager.add_action(
            opponent_id=opponent_id,
            action_features=features,
            is_new_betting_round=is_new_betting_round
        )

