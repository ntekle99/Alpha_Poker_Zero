"""Dataset generation for pretraining psychology network."""

import numpy as np
import random
from typing import List, Tuple, Dict
from collections import defaultdict
from config import PokerConfig as cfg
try:
    from .opponent_history import ActionFeatureEncoder, OpponentHistoryManager
except ImportError:
    from opponent_history import ActionFeatureEncoder, OpponentHistoryManager


class OpponentDatasetGenerator:
    """
    Generates synthetic opponent action histories and computes frequency labels.
    
    Used for Phase 1 supervised pretraining of psychology network.
    """
    
    def __init__(self, max_history_length: int = 30):
        """
        Initialize dataset generator.
        
        Args:
            max_history_length: Maximum history length (K)
        """
        self.max_history_length = max_history_length
    
    def generate_synthetic_history(
        self,
        num_actions: int,
        player_type: str = "random",
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generate a synthetic action history for an opponent.
        
        Args:
            num_actions: Number of actions in history
            player_type: Type of player ("random", "tight", "loose", "aggressive", "passive")
            seed: Random seed for reproducibility
        
        Returns:
            history: Array of shape (num_actions, 7) with action features
            frequencies: Dict with keys 'fold_pct', 'call_pct', 'raise_pct', 'big_bet_pct'
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        history = []
        action_counts = defaultdict(int)
        big_bet_count = 0
        
        # Define player behavior profiles
        profiles = {
            "random": {
                "fold_prob": 0.2,
                "call_prob": 0.4,
                "bet_prob": 0.3,
                "raise_prob": 0.1,
                "big_bet_prob": 0.3
            },
            "tight": {
                "fold_prob": 0.4,
                "call_prob": 0.4,
                "bet_prob": 0.15,
                "raise_prob": 0.05,
                "big_bet_prob": 0.2
            },
            "loose": {
                "fold_prob": 0.1,
                "call_prob": 0.5,
                "bet_prob": 0.3,
                "raise_prob": 0.1,
                "big_bet_prob": 0.25
            },
            "aggressive": {
                "fold_prob": 0.15,
                "call_prob": 0.25,
                "bet_prob": 0.35,
                "raise_prob": 0.25,
                "big_bet_prob": 0.5
            },
            "passive": {
                "fold_prob": 0.25,
                "call_prob": 0.55,
                "bet_prob": 0.15,
                "raise_prob": 0.05,
                "big_bet_prob": 0.2
            }
        }
        
        profile = profiles.get(player_type, profiles["random"])
        
        for i in range(num_actions):
            # Random game context
            pot_size = random.uniform(50, 500)
            stage = random.randint(0, 3)
            position = random.randint(0, 5)
            num_players = random.choice([2, 3, 4, 6])
            to_call = random.uniform(0, pot_size * 0.5)
            raise_count = random.randint(0, 3)
            had_initiative = random.choice([True, False])
            
            # Sample action based on profile
            rand = random.random()
            if rand < profile["fold_prob"]:
                action = cfg.ACTION_FOLD
                bet_size = 0.0
                action_type = ActionFeatureEncoder.ACTION_FOLD
            elif rand < profile["fold_prob"] + profile["call_prob"]:
                action = cfg.ACTION_CHECK_CALL
                bet_size = 0.0
                action_type = ActionFeatureEncoder.ACTION_CALL if to_call > 0 else ActionFeatureEncoder.ACTION_CHECK
            elif rand < profile["fold_prob"] + profile["call_prob"] + profile["bet_prob"]:
                # Bet
                if random.random() < profile["big_bet_prob"]:
                    bet_size = pot_size * random.uniform(1.5, 2.5)  # Large bet
                    action = cfg.ACTION_BET_LARGE
                    big_bet_count += 1
                else:
                    bet_size = pot_size * random.uniform(0.3, 0.7)  # Small/medium bet
                    action = cfg.ACTION_BET_SMALL
                action_type = ActionFeatureEncoder.ACTION_BET
            else:
                # Raise
                bet_size = pot_size * random.uniform(0.5, 1.5)
                action = cfg.ACTION_BET_MEDIUM
                action_type = ActionFeatureEncoder.ACTION_RAISE
            
            # Encode features
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
                is_all_in=(action == cfg.ACTION_ALL_IN)
            )
            
            # Override action_type with sampled value
            features[0] = action_type
            
            history.append(features)
            
            # Track action counts
            if action_type == ActionFeatureEncoder.ACTION_FOLD:
                action_counts['fold'] += 1
            elif action_type == ActionFeatureEncoder.ACTION_CALL:
                action_counts['call'] += 1
            elif action_type == ActionFeatureEncoder.ACTION_RAISE:
                action_counts['raise'] += 1
        
        # Compute frequencies
        total = len(history)
        frequencies = {
            'fold_pct': action_counts['fold'] / total if total > 0 else 0.0,
            'call_pct': action_counts['call'] / total if total > 0 else 0.0,
            'raise_pct': action_counts['raise'] / total if total > 0 else 0.0,
            'big_bet_pct': big_bet_count / total if total > 0 else 0.0
        }
        
        return np.array(history, dtype=np.float32), frequencies
    
    def generate_dataset(
        self,
        num_samples: int,
        min_history_length: int = 10,
        max_history_length: int = 40,
        player_types: Optional[List[str]] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate a dataset of opponent histories and frequency labels.
        
        Args:
            num_samples: Number of samples to generate
            min_history_length: Minimum history length per sample
            max_history_length: Maximum history length per sample
            player_types: List of player types to sample from
        
        Returns:
            histories: List of arrays, each of shape (K, 7) where K varies
            labels: List of arrays, each of shape (4,) with [fold_pct, call_pct, raise_pct, big_bet_pct]
        """
        if player_types is None:
            player_types = ["random", "tight", "loose", "aggressive", "passive"]
        
        histories = []
        labels = []
        
        for i in range(num_samples):
            # Random history length
            history_length = random.randint(min_history_length, max_history_length)
            
            # Random player type
            player_type = random.choice(player_types)
            
            # Generate history
            history, frequencies = self.generate_synthetic_history(
                num_actions=history_length,
                player_type=player_type,
                seed=None  # Don't fix seed for diversity
            )
            
            # Pad or truncate to max_history_length
            if len(history) < self.max_history_length:
                padding = np.zeros(
                    (self.max_history_length - len(history), 7),
                    dtype=np.float32
                )
                history = np.vstack([padding, history])
            elif len(history) > self.max_history_length:
                history = history[-self.max_history_length:]
            
            # Create label vector
            label = np.array([
                frequencies['fold_pct'],
                frequencies['call_pct'],
                frequencies['raise_pct'],
                frequencies['big_bet_pct']
            ], dtype=np.float32)
            
            histories.append(history)
            labels.append(label)
        
        return histories, labels
    
    def generate_from_selfplay(
        self,
        game_logs: List[Dict],
        opponent_id: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate history and labels from actual self-play game logs.
        
        Args:
            game_logs: List of game state/action dictionaries
            opponent_id: ID of opponent to extract history for
        
        Returns:
            history: Array of shape (K, 7)
            label: Array of shape (4,) with frequencies
        """
        # TODO: Implement extraction from actual game logs
        # This would parse self-play data and extract opponent actions
        # For now, return synthetic data
        return self.generate_synthetic_history(
            num_actions=random.randint(10, 40),
            player_type="random"
        )


class OpponentDataset:
    """
    PyTorch-style dataset for psychology network training.
    """
    
    def __init__(self, histories: List[np.ndarray], labels: List[np.ndarray]):
        """
        Initialize dataset.
        
        Args:
            histories: List of history arrays, each (K, 7)
            labels: List of label arrays, each (4,)
        """
        self.histories = histories
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.histories)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a sample from the dataset.
        
        Returns:
            history: Array of shape (K, 7)
            label: Array of shape (4,)
        """
        return self.histories[idx], self.labels[idx]
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of samples.
        
        Args:
            indices: List of sample indices
        
        Returns:
            histories: Array of shape (batch_size, K, 7)
            labels: Array of shape (batch_size, 4)
        """
        batch_histories = np.stack([self.histories[i] for i in indices])
        batch_labels = np.stack([self.labels[i] for i in indices])
        return batch_histories, batch_labels


if __name__ == "__main__":
    # Test dataset generation
    print("Testing OpponentDatasetGenerator...")
    
    generator = OpponentDatasetGenerator(max_history_length=30)
    
    # Generate single history
    history, frequencies = generator.generate_synthetic_history(
        num_actions=25,
        player_type="aggressive"
    )
    
    print(f"Generated history shape: {history.shape}")
    print(f"Frequencies: {frequencies}")
    
    # Generate dataset
    histories, labels = generator.generate_dataset(
        num_samples=100,
        min_history_length=15,
        max_history_length=35
    )
    
    print(f"\nGenerated dataset:")
    print(f"Number of samples: {len(histories)}")
    print(f"History shape (first): {histories[0].shape}")
    print(f"Label shape (first): {labels[0].shape}")
    print(f"Label sample: {labels[0]}")
    
    # Test dataset class
    dataset = OpponentDataset(histories, labels)
    print(f"\nDataset length: {len(dataset)}")
    
    sample_history, sample_label = dataset[0]
    print(f"Sample history shape: {sample_history.shape}")
    print(f"Sample label: {sample_label}")
    
    batch_histories, batch_labels = dataset.get_batch([0, 1, 2])
    print(f"\nBatch histories shape: {batch_histories.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    
    print("\nTest complete!")

