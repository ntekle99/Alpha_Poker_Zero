"""DQN (Deep Q-Network) trainer for poker."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
from collections import deque
from config import PokerConfig as cfg


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add a single experience."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
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
        device=None
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
        """
        self.q_network = q_network
        self.device = device or q_network.device
        self.q_network.to(self.device)
        
        # Create target network (copy of main network)
        self.target_network = type(q_network)()
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
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
            eps=1e-5
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000)
    
    def select_action(self, state: np.ndarray, valid_actions: np.ndarray, training: bool = True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: State vector
            valid_actions: Binary mask of valid actions
            training: If True, use epsilon-greedy; else greedy
        
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
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                q_values = q_values.cpu().numpy()[0]
                
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
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        self.q_network.train()
        current_q_values = self.q_network(states_tensor)
        current_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)
            # Get max Q-value for next state (mask invalid actions if needed)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_value': current_q_values.mean().item()
        }

