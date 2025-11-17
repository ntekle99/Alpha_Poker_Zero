"""Value and policy neural network for poker."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import PokerConfig as cfg


class PokerValuePolicyNetwork(nn.Module):
    """Neural network that outputs both policy and value for poker states."""

    def __init__(self, state_size=None, action_size=None, dropout_rate=0.3):
        super(PokerValuePolicyNetwork, self).__init__()

        if state_size is None:
            state_size = cfg.STATE_SIZE
        if action_size is None:
            action_size = cfg.NUM_ACTIONS

        # Shared layers - deeper architecture
        self.fc1 = nn.Linear(state_size, 1024)
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

        # Policy head - deeper
        self.policy_fc1 = nn.Linear(128, 128)
        self.policy_bn1 = nn.BatchNorm1d(128)
        self.policy_dropout1 = nn.Dropout(dropout_rate * 0.5)
        self.policy_fc2 = nn.Linear(128, 64)
        self.policy_fc3 = nn.Linear(64, action_size)

        # Value head - deeper
        self.value_fc1 = nn.Linear(128, 128)
        self.value_bn1 = nn.BatchNorm1d(128)
        self.value_dropout1 = nn.Dropout(dropout_rate * 0.5)
        self.value_fc2 = nn.Linear(128, 64)
        self.value_fc3 = nn.Linear(64, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        """Forward pass through the network."""
        # Shared layers with residual-like connections
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        
        # Policy head
        policy = F.relu(self.policy_bn1(self.policy_fc1(x)))
        policy = self.policy_dropout1(policy)
        policy = F.relu(self.policy_fc2(policy))
        policy = self.policy_fc3(policy)
        policy = F.softmax(policy, dim=-1)

        # Value head
        value = F.relu(self.value_bn1(self.value_fc1(x)))
        value = self.value_dropout1(value)
        value = F.relu(self.value_fc2(value))
        value = torch.tanh(self.value_fc3(value))

        return policy, value

    def predict(self, state: np.ndarray) -> tuple:
        """
        Predict policy and value for a single state.

        Args:
            state: State vector (numpy array)

        Returns:
            Tuple of (value, policy) where:
            - value: scalar value estimate
            - policy: probability distribution over actions
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy, value = self.forward(state_tensor)
            policy = policy.cpu().numpy()[0]
            value = value.cpu().numpy()[0][0]

        return value, policy

    def predict_batch(self, states: np.ndarray) -> tuple:
        """
        Predict policy and value for a batch of states.

        Args:
            states: Batch of state vectors (numpy array)

        Returns:
            Tuple of (values, policies)
        """
        self.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            policies, values = self.forward(states_tensor)
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()

        return values, policies


class ValuePolicyNetworkWrapper:
    """Wrapper for the neural network to provide a simple interface for MCTS."""

    def __init__(self, model_path=None):
        self.model = PokerValuePolicyNetwork()

        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.model.device))
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Could not load model from {model_path}: {e}")
                print("Starting with random weights")

    def get_vp(self, state: np.ndarray, player: int) -> tuple:
        """
        Get value and policy for a state.

        Args:
            state: State vector (numpy array)
            player: Current player (not used for poker since state is already canonical)

        Returns:
            Tuple of (value, policy)
        """
        return self.model.predict(state)

    def save(self, path: str):
        """Save model to disk."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk."""
        self.model.load_state_dict(torch.load(path, map_location=self.model.device))
        self.model.eval()
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Test the network
    print("Testing PokerValuePolicyNetwork...")

    model = PokerValuePolicyNetwork()
    print(f"Model device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test with random state
    state = np.random.randn(cfg.STATE_SIZE).astype(np.float32)
    value, policy = model.predict(state)

    print(f"\nTest prediction:")
    print(f"Value: {value}")
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum()}")
    print(f"Policy: {policy}")

    # Test batch prediction
    batch_states = np.random.randn(32, cfg.STATE_SIZE).astype(np.float32)
    values, policies = model.predict_batch(batch_states)

    print(f"\nBatch prediction:")
    print(f"Values shape: {values.shape}")
    print(f"Policies shape: {policies.shape}")

    print("\nNetwork test complete!")
