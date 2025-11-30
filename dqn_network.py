"""Deep Q-Network (DQN) for poker."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import PokerConfig as cfg


class DQNNetwork(nn.Module):
    """Deep Q-Network that outputs Q-values for each action."""

    def __init__(self, state_size=None, action_size=None, dropout_rate=0.3, behavior_embedding_dim=0):
        """
        Initialize DQN network.
        
        Args:
            state_size: Size of state vector
            action_size: Number of actions
            dropout_rate: Dropout rate
            behavior_embedding_dim: Dimension of behavior embedding (0 = disabled)
        """
        super(DQNNetwork, self).__init__()

        if state_size is None:
            state_size = cfg.STATE_SIZE
        if action_size is None:
            action_size = cfg.NUM_ACTIONS
        
        self.behavior_embedding_dim = behavior_embedding_dim
        self.base_state_size = state_size
        
        # Input size includes behavior embedding if enabled
        input_size = state_size + behavior_embedding_dim

        # Shared layers
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        # Q-value head
        self.q_fc1 = nn.Linear(128, 128)
        self.q_bn1 = nn.BatchNorm1d(128)
        self.q_dropout1 = nn.Dropout(dropout_rate * 0.5)
        self.q_fc2 = nn.Linear(128, 64)
        self.q_fc3 = nn.Linear(64, action_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, z=None):
        """
        Forward pass through the network.
        
        Args:
            x: State vector of shape (batch_size, state_size) or (state_size,)
            z: Optional behavior embedding of shape (batch_size, behavior_embedding_dim) or (behavior_embedding_dim,)
        
        Returns:
            Q-values for each action
        """
        # Handle single sample vs batch
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Concatenate behavior embedding if provided
        if z is not None:
            if z.dim() == 1:
                z = z.unsqueeze(0)
            # Ensure batch sizes match
            if x.size(0) != z.size(0):
                raise ValueError(f"Batch size mismatch: state={x.size(0)}, embedding={z.size(0)}")
            x = torch.cat([x, z], dim=1)
        elif self.behavior_embedding_dim > 0:
            # Use zeros if embedding not provided (backward compatibility)
            batch_size = x.size(0)
            z_zeros = torch.zeros(batch_size, self.behavior_embedding_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, z_zeros], dim=1)
        
        # Shared layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        
        # Q-value head
        q = F.relu(self.q_bn1(self.q_fc1(x)))
        q = self.q_dropout1(q)
        q = F.relu(self.q_fc2(q))
        q = self.q_fc3(q)  # No activation - raw Q-values
        
        # Remove batch dimension if single sample
        if single_sample:
            q = q.squeeze(0)

        return q

    def predict(self, state: np.ndarray, z: np.ndarray = None) -> np.ndarray:
        """
        Predict Q-values for a single state.

        Args:
            state: State vector (numpy array)
            z: Optional behavior embedding (numpy array)

        Returns:
            Q-values for each action
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            z_tensor = torch.FloatTensor(z).to(self.device) if z is not None else None
            q_values = self.forward(state_tensor, z_tensor)
            q_values = q_values.cpu().numpy()
            if q_values.ndim > 1:
                q_values = q_values[0]

        return q_values

    def predict_batch(self, states: np.ndarray, z_batch: np.ndarray = None) -> np.ndarray:
        """
        Predict Q-values for a batch of states.

        Args:
            states: Batch of state vectors (numpy array)
            z_batch: Optional batch of behavior embeddings (numpy array)

        Returns:
            Q-values for each state-action pair
        """
        self.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            z_tensor = torch.FloatTensor(z_batch).to(self.device) if z_batch is not None else None
            q_values = self.forward(states_tensor, z_tensor)
            q_values = q_values.cpu().numpy()

        return q_values


if __name__ == "__main__":
    # Test the network
    print("Testing DQNNetwork...")

    model = DQNNetwork()
    print(f"Model device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test with random state
    state = np.random.randn(cfg.STATE_SIZE).astype(np.float32)
    q_values = model.predict(state)

    print(f"\nTest prediction:")
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}")

    # Test batch prediction
    batch_states = np.random.randn(32, cfg.STATE_SIZE).astype(np.float32)
    batch_q_values = model.predict_batch(batch_states)

    print(f"\nBatch prediction:")
    print(f"Q-values shape: {batch_q_values.shape}")

    print("\nNetwork test complete!")

