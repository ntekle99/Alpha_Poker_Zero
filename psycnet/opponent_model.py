"""Psychology network for opponent modeling in poker."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PsychologyNetwork(nn.Module):
    """
    Psychology network that learns opponent behavior embeddings.
    
    Architecture:
    - Embedding layers for categorical features
    - LSTM to process sequence of actions
    - Linear head to produce behavior embedding z
    - Optional supervised head for pretraining (predicts action frequencies)
    """
    
    def __init__(
        self,
        embedding_dim: int = 32,
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        behavior_embedding_dim: int = 16,
        dropout: float = 0.2,
        use_supervised_head: bool = True
    ):
        """
        Initialize psychology network.
        
        Args:
            embedding_dim: Dimension for embedding categorical features
            lstm_hidden_dim: Hidden dimension of LSTM
            lstm_num_layers: Number of LSTM layers
            behavior_embedding_dim: Dimension of output behavior embedding z
            dropout: Dropout rate
            use_supervised_head: Whether to include supervised pretraining head
        """
        super(PsychologyNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.behavior_embedding_dim = behavior_embedding_dim
        self.use_supervised_head = use_supervised_head
        
        # Feature dimensions (from ActionFeatureEncoder)
        # action_type: 6 (fold, check, call, bet, raise, all-in)
        # bet_size_bucket: 6 (0-5)
        # street: 4 (preflop, flop, turn, river)
        # position: 6 (UTG, MP, CO, BTN, SB, BB)
        # num_players_bucket: 4 (HU, 3-way, 4-way, 5+)
        # initiative: 2 (binary)
        # facing_action_bucket: 8 (0-7)
        
        # Embedding layers for categorical features
        self.action_type_embed = nn.Embedding(6, embedding_dim)
        self.bet_size_embed = nn.Embedding(6, embedding_dim)
        self.street_embed = nn.Embedding(4, embedding_dim)
        self.position_embed = nn.Embedding(6, embedding_dim)
        self.num_players_embed = nn.Embedding(4, embedding_dim)
        self.initiative_embed = nn.Embedding(2, embedding_dim)
        self.facing_action_embed = nn.Embedding(8, embedding_dim)
        
        # Total embedded feature dimension
        embedded_dim = 7 * embedding_dim
        
        # LSTM to process sequence
        self.lstm = nn.LSTM(
            input_size=embedded_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Behavior embedding head
        self.behavior_fc1 = nn.Linear(lstm_hidden_dim, 64)
        self.behavior_dropout1 = nn.Dropout(dropout)
        self.behavior_fc2 = nn.Linear(64, behavior_embedding_dim)
        
        # Supervised head for pretraining (predicts action frequencies)
        if use_supervised_head:
            self.supervised_fc1 = nn.Linear(lstm_hidden_dim, 64)
            self.supervised_dropout1 = nn.Dropout(dropout)
            self.supervised_fc2 = nn.Linear(64, 4)  # fold%, call%, raise%, big_bet%
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(
        self,
        action_sequences: torch.Tensor,
        return_supervised: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through psychology network.
        
        Args:
            action_sequences: Tensor of shape (batch_size, seq_len, feature_dim)
                where feature_dim = 7 (action_type, bet_size, street, position,
                num_players, initiative, facing_action)
            return_supervised: Whether to return supervised head predictions
        
        Returns:
            behavior_embedding: Tensor of shape (batch_size, behavior_embedding_dim)
            supervised_pred: Optional tensor of shape (batch_size, 4) with frequency predictions
        """
        batch_size, seq_len, feature_dim = action_sequences.shape
        
        # Extract features (assuming they are integer indices)
        action_type = action_sequences[:, :, 0].long()
        bet_size = action_sequences[:, :, 1].long()
        street = action_sequences[:, :, 2].long()
        position = action_sequences[:, :, 3].long()
        num_players = action_sequences[:, :, 4].long()
        initiative = action_sequences[:, :, 5].long()
        facing_action = action_sequences[:, :, 6].long()
        
        # Embed categorical features
        action_emb = self.action_type_embed(action_type)  # (B, T, E)
        bet_size_emb = self.bet_size_embed(bet_size)
        street_emb = self.street_embed(street)
        position_emb = self.position_embed(position)
        num_players_emb = self.num_players_embed(num_players)
        initiative_emb = self.initiative_embed(initiative)
        facing_action_emb = self.facing_action_embed(facing_action)
        
        # Concatenate all embeddings
        embedded = torch.cat([
            action_emb, bet_size_emb, street_emb, position_emb,
            num_players_emb, initiative_emb, facing_action_emb
        ], dim=-1)  # (B, T, 7*E)
        
        # Process through LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (B, T, H), hidden: (num_layers, B, H)
        
        # Use final hidden state (from last layer)
        final_hidden = hidden[-1]  # (B, H)
        
        # Behavior embedding head
        z = F.relu(self.behavior_fc1(final_hidden))
        z = self.behavior_dropout1(z)
        z = self.behavior_fc2(z)  # (B, behavior_embedding_dim)
        
        # Supervised head (if requested and available)
        supervised_pred = None
        if return_supervised and self.use_supervised_head:
            supervised_h = F.relu(self.supervised_fc1(final_hidden))
            supervised_h = self.supervised_dropout1(supervised_h)
            supervised_pred = torch.sigmoid(self.supervised_fc2(supervised_h))  # (B, 4)
        
        return z, supervised_pred
    
    def encode_opponent(
        self,
        action_sequence: np.ndarray,
        return_supervised: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Encode an opponent's action sequence into behavior embedding.
        
        Args:
            action_sequence: Array of shape (seq_len, 7) with action features
            return_supervised: Whether to return supervised predictions
        
        Returns:
            behavior_embedding: Array of shape (behavior_embedding_dim,)
            supervised_pred: Optional array of shape (4,) with frequency predictions
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            if action_sequence.ndim == 2:
                action_sequence = action_sequence[np.newaxis, :, :]
            
            action_tensor = torch.FloatTensor(action_sequence).to(self.device)
            z, supervised = self.forward(action_tensor, return_supervised=return_supervised)
            
            z_np = z.cpu().numpy()[0]
            supervised_np = supervised.cpu().numpy()[0] if supervised is not None else None
            
            return z_np, supervised_np


if __name__ == "__main__":
    # Test the network
    print("Testing PsychologyNetwork...")
    
    model = PsychologyNetwork(
        embedding_dim=32,
        lstm_hidden_dim=128,
        behavior_embedding_dim=16,
        use_supervised_head=True
    )
    
    print(f"Model device: {model.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test with random sequence
    batch_size = 4
    seq_len = 20
    feature_dim = 7
    
    # Random integer features
    action_sequences = torch.randint(0, 6, (batch_size, seq_len, feature_dim)).float()
    action_sequences[:, :, 1] = torch.randint(0, 6, (batch_size, seq_len))  # bet_size
    action_sequences[:, :, 2] = torch.randint(0, 4, (batch_size, seq_len))  # street
    action_sequences[:, :, 3] = torch.randint(0, 6, (batch_size, seq_len))  # position
    action_sequences[:, :, 4] = torch.randint(0, 4, (batch_size, seq_len))  # num_players
    action_sequences[:, :, 5] = torch.randint(0, 2, (batch_size, seq_len))  # initiative
    action_sequences[:, :, 6] = torch.randint(0, 8, (batch_size, seq_len))  # facing_action
    
    z, supervised = model.forward(action_sequences, return_supervised=True)
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {action_sequences.shape}")
    print(f"Behavior embedding shape: {z.shape}")
    print(f"Supervised prediction shape: {supervised.shape if supervised is not None else None}")
    print(f"Behavior embedding sample: {z[0]}")
    if supervised is not None:
        print(f"Supervised prediction sample: {supervised[0]}")
    
    # Test encoding
    test_sequence = np.random.randint(0, 6, (seq_len, feature_dim)).astype(np.float32)
    z_np, supervised_np = model.encode_opponent(test_sequence, return_supervised=True)
    
    print(f"\nTest encoding:")
    print(f"Input sequence shape: {test_sequence.shape}")
    print(f"Behavior embedding shape: {z_np.shape}")
    print(f"Supervised prediction shape: {supervised_np.shape if supervised_np is not None else None}")
    
    print("\nNetwork test complete!")

