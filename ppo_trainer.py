"""PPO (Proximal Policy Optimization) trainer for poker."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
from collections import deque
from config import PokerConfig as cfg


class ExperienceBuffer:
    """Buffer for storing PPO rollout experiences."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def add(self, state, action, reward, log_prob, value, done):
        """Add a single experience."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_advantages_and_returns(self, gamma=0.99, gae_lambda=0.95, last_value=0.0):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            last_value: Value estimate for terminal state
        """
        advantages = []
        returns = []
        
        # Compute advantages using GAE
        gae = 0
        next_value = last_value
        
        for step in reversed(range(len(self.rewards))):
            if self.dones[step]:
                delta = self.rewards[step] - self.values[step]
                gae = delta
            else:
                delta = self.rewards[step] + gamma * next_value - self.values[step]
                gae = delta + gamma * gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[step])
            next_value = self.values[step]
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batch(self, indices):
        """Get a batch of experiences by indices."""
        batch_states = [self.states[i] for i in indices]
        batch_actions = [self.actions[i] for i in indices]
        batch_log_probs = [self.log_probs[i] for i in indices]
        batch_advantages = [self.advantages[i] for i in indices]
        batch_returns = [self.returns[i] for i in indices]
        batch_old_values = [self.values[i] for i in indices]
        
        return {
            'states': torch.FloatTensor(np.array(batch_states)),
            'actions': torch.LongTensor(batch_actions),
            'old_log_probs': torch.FloatTensor(batch_log_probs),
            'advantages': torch.FloatTensor(batch_advantages),
            'returns': torch.FloatTensor(batch_returns),
            'old_values': torch.FloatTensor(batch_old_values),
        }
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
    
    def __len__(self):
        return len(self.states)


class PPOTrainer:
    """PPO trainer for poker policy optimization."""
    
    def __init__(
        self,
        policy_network,
        learning_rate=3e-4,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        gamma=0.99,
        gae_lambda=0.95,
        ppo_epochs=4,
        batch_size=64,
        device=None
    ):
        """
        Initialize PPO trainer.
        
        Args:
            policy_network: Policy network (PokerValuePolicyNetwork)
            learning_rate: Learning rate
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping norm
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            ppo_epochs: Number of PPO update epochs per batch
            batch_size: Batch size for updates
            device: Device to use
        """
        self.policy_network = policy_network
        self.device = device or policy_network.device
        self.policy_network.to(self.device)
        
        # Hyperparameters
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate,
            eps=1e-5
        )
        
        # Experience buffer
        self.buffer = ExperienceBuffer()
    
    def select_action(self, state: np.ndarray, valid_actions: np.ndarray, deterministic: bool = False):
        """
        Select an action using the policy network.
        
        Args:
            state: State vector
            valid_actions: Binary mask of valid actions
            deterministic: If True, select greedily; else sample from policy
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        self.policy_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy, value = self.policy_network(state_tensor)
            
            # Mask invalid actions
            policy = policy.cpu().numpy()[0]
            policy = policy * valid_actions
            if policy.sum() > 0:
                policy = policy / policy.sum()
            else:
                # Fallback to uniform over valid actions
                policy = valid_actions / valid_actions.sum()
            
            if deterministic:
                action = np.argmax(policy)
            else:
                action = np.random.choice(len(policy), p=policy)
            
            log_prob = np.log(policy[action] + 1e-8)
            value = value.cpu().numpy()[0][0]
        
        return action, log_prob, value
    
    def update(self):
        """
        Update the policy network using PPO.
        
        Returns:
            Dictionary of training statistics
        """
        if len(self.buffer) == 0:
            return {}
        
        # Compute advantages and returns
        self.buffer.compute_advantages_and_returns(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        # Normalize advantages
        advantages = np.array(self.buffer.advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages.tolist()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        advantages = torch.FloatTensor(self.buffer.advantages).to(self.device)
        returns = torch.FloatTensor(self.buffer.returns).to(self.device)
        old_values = torch.FloatTensor(self.buffer.values).to(self.device)
        
        # Training statistics
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0,
            'clip_fraction': 0.0
        }
        
        # PPO update epochs
        indices = np.arange(len(self.buffer))
        num_batches = (len(self.buffer) + self.batch_size - 1) // self.batch_size
        
        num_updates = 0  # Track actual number of updates (excluding skipped batches)
        
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.buffer))
                batch_indices = indices[start_idx:end_idx]
                
                # Skip batches with only 1 sample (BatchNorm requires batch_size > 1)
                if len(batch_indices) < 2:
                    continue
                
                num_updates += 1
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Forward pass
                self.policy_network.train()
                policies, values = self.policy_network(batch_states)
                
                # Get action probabilities
                action_probs = policies.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                new_log_probs = torch.log(action_probs + 1e-8)
                
                # Compute policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss (clipped)
                value_clipped = batch_old_values + torch.clamp(
                    values.squeeze() - batch_old_values,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss1 = (values.squeeze() - batch_returns).pow(2)
                value_loss2 = (value_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # Entropy bonus
                entropy = -(policies * torch.log(policies + 1e-8)).sum(dim=1).mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Statistics
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss.item()
                stats['entropy'] += entropy.item()
                stats['total_loss'] += total_loss.item()
                
                # Clip fraction
                clip_fraction = ((ratio < 1.0 - self.clip_epsilon) | (ratio > 1.0 + self.clip_epsilon)).float().mean()
                stats['clip_fraction'] += clip_fraction.item()
        
        # Average statistics (only count actual updates, not skipped batches)
        if num_updates > 0:
            for key in stats:
                stats[key] /= num_updates
        
        # Clear buffer
        self.buffer.clear()
        
        return stats

