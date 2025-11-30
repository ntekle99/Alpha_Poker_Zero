"""Opponent history management and feature encoding for psychology network."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque
from config import PokerConfig as cfg


class ActionFeatureEncoder:
    """
    Encodes poker actions into feature vectors for psychology network.
    
    Features:
    - action_type: 0=fold, 1=check, 2=call, 3=bet, 4=raise, 5=all-in
    - bet_size_bucket: 0=none, 1=small (<0.5 pot), 2=medium (0.5-1 pot),
      3=large (1-2 pot), 4=overbet (>2 pot), 5=all-in
    - street: 0=preflop, 1=flop, 2=turn, 3=river
    - position: 0=UTG, 1=MP, 2=CO, 3=BTN, 4=SB, 5=BB
    - num_players_bucket: 0=HU, 1=3-way, 2=4-way, 3=5+
    - initiative: 0=no, 1=yes
    - facing_action_bucket: 0=none, 1=small bet, 2=medium bet, 3=large bet,
      4=overbet, 5=raise, 6=3bet, 7=4bet+
    """
    
    # Action type mapping
    ACTION_FOLD = 0
    ACTION_CHECK = 1
    ACTION_CALL = 2
    ACTION_BET = 3
    ACTION_RAISE = 4
    ACTION_ALL_IN = 5
    
    # Street mapping
    STREET_PREFLOP = 0
    STREET_FLOP = 1
    STREET_TURN = 2
    STREET_RIVER = 3
    
    # Position mapping
    POSITION_UTG = 0
    POSITION_MP = 1
    POSITION_CO = 2
    POSITION_BTN = 3
    POSITION_SB = 4
    POSITION_BB = 5
    
    @staticmethod
    def encode_action_type(action: int) -> int:
        """
        Encode action from config to psychology network action type.
        
        Args:
            action: Action index from config (ACTION_FOLD, ACTION_CHECK_CALL, etc.)
        
        Returns:
            Encoded action type (0-5)
        """
        if action == cfg.ACTION_FOLD:
            return ActionFeatureEncoder.ACTION_FOLD
        elif action == cfg.ACTION_CHECK_CALL:
            # Need context to distinguish check vs call
            # This will be handled by the caller based on to_call amount
            return ActionFeatureEncoder.ACTION_CHECK  # Default, caller should override
        elif action == cfg.ACTION_BET_SMALL:
            return ActionFeatureEncoder.ACTION_BET
        elif action == cfg.ACTION_BET_MEDIUM:
            return ActionFeatureEncoder.ACTION_BET
        elif action == cfg.ACTION_BET_LARGE:
            return ActionFeatureEncoder.ACTION_BET
        elif action == cfg.ACTION_ALL_IN:
            return ActionFeatureEncoder.ACTION_ALL_IN
        else:
            return ActionFeatureEncoder.ACTION_CHECK  # Default
    
    @staticmethod
    def encode_bet_size_bucket(bet_size: float, pot_size: float, is_all_in: bool) -> int:
        """
        Encode bet size into bucket.
        
        Args:
            bet_size: Bet size in chips
            pot_size: Pot size in chips
            is_all_in: Whether this is an all-in bet
        
        Returns:
            Bet size bucket (0-5)
        """
        if is_all_in:
            return 5
        
        if bet_size == 0:
            return 0
        
        ratio = bet_size / pot_size if pot_size > 0 else 0
        
        if ratio < 0.5:
            return 1  # Small bet
        elif ratio < 1.0:
            return 2  # Medium bet
        elif ratio < 2.0:
            return 3  # Large bet
        else:
            return 4  # Overbet
    
    @staticmethod
    def encode_street(stage: int) -> int:
        """
        Encode game stage to street.
        
        Args:
            stage: Stage from game state (0=preflop, 1=flop, 2=turn, 3=river)
        
        Returns:
            Street encoding (0-3)
        """
        return stage
    
    @staticmethod
    def encode_position(position: int, num_players: int) -> int:
        """
        Encode player position.
        
        Args:
            position: Position index (0-based, 0=UTG, etc.)
            num_players: Total number of players
        
        Returns:
            Position encoding (0-5)
        """
        # For heads-up, map to SB/BB
        if num_players == 2:
            return ActionFeatureEncoder.POSITION_SB if position == 0 else ActionFeatureEncoder.POSITION_BB
        
        # For 6-max, map positions
        if num_players <= 6:
            if position == 0:
                return ActionFeatureEncoder.POSITION_UTG
            elif position == num_players - 3:
                return ActionFeatureEncoder.POSITION_CO
            elif position == num_players - 2:
                return ActionFeatureEncoder.POSITION_BTN
            elif position == num_players - 1:
                return ActionFeatureEncoder.POSITION_SB
            else:
                return ActionFeatureEncoder.POSITION_MP
        
        # Default to MP for other cases
        return ActionFeatureEncoder.POSITION_MP
    
    @staticmethod
    def encode_num_players_bucket(num_players: int) -> int:
        """
        Encode number of players into bucket.
        
        Args:
            num_players: Number of players in hand
        
        Returns:
            Bucket (0=HU, 1=3-way, 2=4-way, 3=5+)
        """
        if num_players == 2:
            return 0  # HU
        elif num_players == 3:
            return 1  # 3-way
        elif num_players == 4:
            return 2  # 4-way
        else:
            return 3  # 5+
    
    @staticmethod
    def encode_facing_action_bucket(
        to_call: float,
        pot_size: float,
        raise_count: int
    ) -> int:
        """
        Encode facing action into bucket.
        
        Args:
            to_call: Amount to call
            pot_size: Pot size
            raise_count: Number of raises in current betting round
        
        Returns:
            Facing action bucket (0-7)
        """
        if to_call == 0:
            return 0  # None
        
        if raise_count >= 4:
            return 7  # 4bet+
        elif raise_count == 3:
            return 6  # 3bet
        elif raise_count >= 1:
            return 5  # Raise
        
        # Bet sizing
        ratio = to_call / pot_size if pot_size > 0 else 0
        
        if ratio < 0.5:
            return 1  # Small bet
        elif ratio < 1.0:
            return 2  # Medium bet
        elif ratio < 2.0:
            return 3  # Large bet
        else:
            return 4  # Overbet
    
    @staticmethod
    def encode_action_features(
        action: int,
        bet_size: float,
        pot_size: float,
        stage: int,
        position: int,
        num_players: int,
        had_initiative: bool,
        to_call: float,
        raise_count: int,
        is_all_in: bool = False
    ) -> np.ndarray:
        """
        Encode a complete action into feature vector.
        
        Args:
            action: Action index from config
            bet_size: Bet size in chips
            pot_size: Pot size in chips
            stage: Game stage (0-3)
            position: Player position
            num_players: Number of players
            had_initiative: Whether player had initiative (acted first)
            to_call: Amount to call
            raise_count: Number of raises in betting round
            is_all_in: Whether action was all-in
        
        Returns:
            Feature vector of shape (7,)
        """
        # Determine action type (check vs call)
        if action == cfg.ACTION_CHECK_CALL:
            if to_call > 0:
                action_type = ActionFeatureEncoder.ACTION_CALL
            else:
                action_type = ActionFeatureEncoder.ACTION_CHECK
        else:
            action_type = ActionFeatureEncoder.encode_action_type(action)
        
        # Encode all features
        bet_size_bucket = ActionFeatureEncoder.encode_bet_size_bucket(
            bet_size, pot_size, is_all_in
        )
        street = ActionFeatureEncoder.encode_street(stage)
        position_enc = ActionFeatureEncoder.encode_position(position, num_players)
        num_players_bucket = ActionFeatureEncoder.encode_num_players_bucket(num_players)
        initiative = 1 if had_initiative else 0
        facing_action = ActionFeatureEncoder.encode_facing_action_bucket(
            to_call, pot_size, raise_count
        )
        
        return np.array([
            action_type,
            bet_size_bucket,
            street,
            position_enc,
            num_players_bucket,
            initiative,
            facing_action
        ], dtype=np.float32)


class OpponentHistoryManager:
    """
    Manages rolling history of opponent actions for psychology network.
    
    Maintains a deque of action feature vectors for each opponent,
    with a maximum history length K (typically 20-40).
    """
    
    def __init__(self, max_history_length: int = 30):
        """
        Initialize history manager.
        
        Args:
            max_history_length: Maximum number of actions to keep in history (K)
        """
        self.max_history_length = max_history_length
        self.histories: Dict[int, deque] = {}  # opponent_id -> deque of feature vectors
        self.raise_counts: Dict[int, int] = {}  # Track raise counts per betting round
    
    def reset_opponent(self, opponent_id: int):
        """Reset history for a specific opponent."""
        if opponent_id in self.histories:
            self.histories[opponent_id].clear()
        if opponent_id in self.raise_counts:
            self.raise_counts[opponent_id] = 0
    
    def reset_all(self):
        """Reset all opponent histories."""
        self.histories.clear()
        self.raise_counts.clear()
    
    def add_action(
        self,
        opponent_id: int,
        action_features: np.ndarray,
        is_new_betting_round: bool = False
    ):
        """
        Add an action to opponent's history.
        
        Args:
            opponent_id: Unique identifier for opponent
            action_features: Feature vector of shape (7,)
            is_new_betting_round: Whether this starts a new betting round
        """
        if opponent_id not in self.histories:
            self.histories[opponent_id] = deque(maxlen=self.max_history_length)
            self.raise_counts[opponent_id] = 0
        
        if is_new_betting_round:
            self.raise_counts[opponent_id] = 0
        
        # Increment raise count if action is a raise
        if action_features[0] == ActionFeatureEncoder.ACTION_RAISE:
            self.raise_counts[opponent_id] += 1
        
        self.histories[opponent_id].append(action_features.copy())
    
    def get_history(self, opponent_id: int) -> np.ndarray:
        """
        Get action history for opponent as array.
        
        Args:
            opponent_id: Opponent identifier
        
        Returns:
            Array of shape (K, 7) where K is current history length
        """
        if opponent_id not in self.histories or len(self.histories[opponent_id]) == 0:
            # Return zero-padded array if no history
            return np.zeros((self.max_history_length, 7), dtype=np.float32)
        
        history = np.array(list(self.histories[opponent_id]), dtype=np.float32)
        
        # Pad or truncate to max_history_length
        if len(history) < self.max_history_length:
            padding = np.zeros(
                (self.max_history_length - len(history), 7),
                dtype=np.float32
            )
            history = np.vstack([padding, history])
        elif len(history) > self.max_history_length:
            history = history[-self.max_history_length:]
        
        return history
    
    def get_history_length(self, opponent_id: int) -> int:
        """Get current history length for opponent."""
        if opponent_id not in self.histories:
            return 0
        return len(self.histories[opponent_id])
    
    def has_history(self, opponent_id: int) -> bool:
        """Check if opponent has any history."""
        return opponent_id in self.histories and len(self.histories[opponent_id]) > 0


if __name__ == "__main__":
    # Test ActionFeatureEncoder
    print("Testing ActionFeatureEncoder...")
    
    # Test encoding
    features = ActionFeatureEncoder.encode_action_features(
        action=cfg.ACTION_BET_MEDIUM,
        bet_size=100.0,
        pot_size=100.0,
        stage=1,  # Flop
        position=3,  # BTN
        num_players=2,  # HU
        had_initiative=True,
        to_call=0.0,
        raise_count=0,
        is_all_in=False
    )
    
    print(f"Encoded features: {features}")
    print(f"Feature shape: {features.shape}")
    print(f"Action type: {features[0]}, Bet size bucket: {features[1]}, Street: {features[2]}")
    
    # Test OpponentHistoryManager
    print("\nTesting OpponentHistoryManager...")
    
    manager = OpponentHistoryManager(max_history_length=20)
    
    # Add some actions
    for i in range(5):
        features = ActionFeatureEncoder.encode_action_features(
            action=cfg.ACTION_CHECK_CALL,
            bet_size=0.0,
            pot_size=100.0,
            stage=i % 4,
            position=i % 6,
            num_players=2,
            had_initiative=(i % 2 == 0),
            to_call=0.0,
            raise_count=0
        )
        manager.add_action(opponent_id=1, action_features=features)
    
    history = manager.get_history(opponent_id=1)
    print(f"History shape: {history.shape}")
    print(f"History length: {manager.get_history_length(opponent_id=1)}")
    print(f"First action features: {history[-5:][0]}")
    
    print("\nTest complete!")

