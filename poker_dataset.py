"""Dataset handler for poker training data."""

import pickle
import numpy as np
from collections import deque
from config import PokerConfig as cfg


class PokerTrainingDataset:
    """Dataset for poker self-play training data."""

    def __init__(self, max_size=None):
        """
        Initialize the dataset.

        Args:
            max_size: Maximum number of samples to keep (uses deque for efficiency)
        """
        if max_size is None:
            max_size = cfg.DATASET_QUEUE_SIZE

        self.max_size = max_size
        self.states = deque(maxlen=max_size)
        self.action_probs = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)

    def add_game_to_training_dataset(self, game_data: list, final_reward: float):
        """
        Add a game's data to the training dataset.

        Args:
            game_data: List of (state, action_probs, player) tuples
            final_reward: Final reward for the game
        """
        for state, action_probs, player in game_data:
            # Store state, action probabilities, and reward from that player's perspective
            # The reward is the final game outcome from that player's perspective
            self.states.append(state)
            self.action_probs.append(action_probs)
            # Reward from perspective of player who made the action
            self.rewards.append(final_reward * player)

    def get_batch(self, batch_size: int = None):
        """
        Get a random batch of training data.

        Args:
            batch_size: Size of batch to return

        Returns:
            Tuple of (states, action_probs, rewards) as numpy arrays
        """
        if batch_size is None:
            batch_size = cfg.BATCH_SIZE

        if len(self.states) < batch_size:
            batch_size = len(self.states)

        indices = np.random.choice(len(self.states), batch_size, replace=False)

        batch_states = np.array([self.states[i] for i in indices])
        batch_action_probs = np.array([self.action_probs[i] for i in indices])
        batch_rewards = np.array([self.rewards[i] for i in indices])

        return batch_states, batch_action_probs, batch_rewards

    def save(self, filepath: str):
        """Save the dataset to disk."""
        data = {
            'states': list(self.states),
            'action_probs': list(self.action_probs),
            'rewards': list(self.rewards),
            'max_size': self.max_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Dataset saved to {filepath} ({len(self.states)} samples)")

    def load(self, filepath: str):
        """Load the dataset from disk."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.max_size = data['max_size']
            self.states = deque(data['states'], maxlen=self.max_size)
            self.action_probs = deque(data['action_probs'], maxlen=self.max_size)
            self.rewards = deque(data['rewards'], maxlen=self.max_size)

            print(f"Dataset loaded from {filepath} ({len(self.states)} samples)")
        except FileNotFoundError:
            print(f"No dataset found at {filepath}, starting fresh")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.states)

    def get_stats(self):
        """Get statistics about the dataset."""
        if len(self.rewards) == 0:
            return {
                'size': 0,
                'mean_reward': 0,
                'std_reward': 0,
                'max_reward': 0,
                'min_reward': 0
            }

        rewards_array = np.array(list(self.rewards))
        return {
            'size': len(self.states),
            'mean_reward': np.mean(rewards_array),
            'std_reward': np.std(rewards_array),
            'max_reward': np.max(rewards_array),
            'min_reward': np.min(rewards_array)
        }


if __name__ == "__main__":
    # Test the dataset
    print("Testing PokerTrainingDataset...")

    dataset = PokerTrainingDataset(max_size=1000)

    # Add some dummy data
    for i in range(100):
        game_data = []
        for j in range(10):
            state = np.random.randn(cfg.STATE_SIZE)
            action_probs = np.random.rand(cfg.NUM_ACTIONS)
            action_probs = action_probs / action_probs.sum()
            player = 1 if j % 2 == 0 else -1
            game_data.append((state, action_probs, player))

        final_reward = np.random.choice([1, -1, 0])
        dataset.add_game_to_training_dataset(game_data, final_reward)

    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset stats: {dataset.get_stats()}")

    # Test batch retrieval
    batch_states, batch_action_probs, batch_rewards = dataset.get_batch(32)
    print(f"\nBatch shapes:")
    print(f"States: {batch_states.shape}")
    print(f"Action probs: {batch_action_probs.shape}")
    print(f"Rewards: {batch_rewards.shape}")

    # Test save/load
    test_path = "/tmp/test_poker_dataset.pkl"
    dataset.save(test_path)

    dataset2 = PokerTrainingDataset()
    dataset2.load(test_path)
    print(f"\nLoaded dataset size: {len(dataset2)}")

    print("\nDataset test complete!")
