"""Phase 1: Supervised pretraining script for psychology network."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import argparse
from typing import Tuple, Optional

try:
    from .opponent_model import PsychologyNetwork
    from .opponent_dataset import OpponentDatasetGenerator, OpponentDataset
except ImportError:
    # For standalone script execution
    from opponent_model import PsychologyNetwork
    from opponent_dataset import OpponentDatasetGenerator, OpponentDataset


def train_epoch(
    model: PsychologyNetwork,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Average loss, average accuracy (if applicable)
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for histories, labels in dataloader:
        histories = histories.to(device)
        labels = labels.to(device)
        
        # Forward pass
        _, supervised_pred = model.forward(histories, return_supervised=True)
        
        # Compute loss
        loss = criterion(supervised_pred, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, 0.0  # Accuracy not applicable for regression


def validate(
    model: PsychologyNetwork,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Returns:
        Average loss, average accuracy
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for histories, labels in dataloader:
            histories = histories.to(device)
            labels = labels.to(device)
            
            # Forward pass
            _, supervised_pred = model.forward(histories, return_supervised=True)
            
            # Compute loss
            loss = criterion(supervised_pred, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, 0.0


def train_psychology_network(
    num_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    train_samples: int = 10000,
    val_samples: int = 2000,
    max_history_length: int = 30,
    embedding_dim: int = 32,
    lstm_hidden_dim: int = 128,
    behavior_embedding_dim: int = 16,
    save_path: str = "poker-rl/output/models/psychology_network.pt",
    load_path: Optional[str] = None
):
    """
    Train psychology network with supervised learning (Phase 1).
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        train_samples: Number of training samples to generate
        val_samples: Number of validation samples to generate
        max_history_length: Maximum history length (K)
        embedding_dim: Embedding dimension
        lstm_hidden_dim: LSTM hidden dimension
        behavior_embedding_dim: Behavior embedding dimension
        save_path: Path to save trained model
        load_path: Path to load pretrained model (optional)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize model
    model = PsychologyNetwork(
        embedding_dim=embedding_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        behavior_embedding_dim=behavior_embedding_dim,
        use_supervised_head=True
    )
    model.to(device)
    
    # Load pretrained weights if specified
    if load_path and os.path.exists(load_path):
        print(f"Loading pretrained model from {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
    
    # Generate dataset
    print("Generating training dataset...")
    generator = OpponentDatasetGenerator(max_history_length=max_history_length)
    
    train_histories, train_labels = generator.generate_dataset(
        num_samples=train_samples,
        min_history_length=10,
        max_history_length=40
    )
    
    print("Generating validation dataset...")
    val_histories, val_labels = generator.generate_dataset(
        num_samples=val_samples,
        min_history_length=10,
        max_history_length=40
    )
    
    # Convert to tensors
    train_histories_tensor = torch.FloatTensor(np.stack(train_histories))
    train_labels_tensor = torch.FloatTensor(np.stack(train_labels))
    val_histories_tensor = torch.FloatTensor(np.stack(val_histories))
    val_labels_tensor = torch.FloatTensor(np.stack(val_labels))
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_histories_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_histories_tensor, val_labels_tensor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        train_loss, _ = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, _ = validate(model, val_loader, criterion, device)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        
        print()
    
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Model saved to: {save_path}")
    
    # Test encoding
    print("\nTesting model encoding...")
    model.eval()
    test_history = train_histories[0]
    z, supervised = model.encode_opponent(test_history, return_supervised=True)
    print(f"Test history shape: {test_history.shape}")
    print(f"Behavior embedding shape: {z.shape}")
    print(f"Behavior embedding: {z}")
    print(f"Supervised prediction: {supervised}")
    print(f"True label: {train_labels[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train psychology network (Phase 1)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--train-samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--val-samples", type=int, default=2000, help="Validation samples")
    parser.add_argument("--max-history", type=int, default=30, help="Max history length")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--lstm-hidden", type=int, default=128, help="LSTM hidden dimension")
    parser.add_argument("--behavior-dim", type=int, default=16, help="Behavior embedding dimension")
    parser.add_argument("--save-path", type=str, default="poker-rl/output/models/psychology_network.pt",
                        help="Path to save model")
    parser.add_argument("--load-path", type=str, default=None, help="Path to load pretrained model")
    
    args = parser.parse_args()
    
    train_psychology_network(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        max_history_length=args.max_history,
        embedding_dim=args.embedding_dim,
        lstm_hidden_dim=args.lstm_hidden,
        behavior_embedding_dim=args.behavior_dim,
        save_path=args.save_path,
        load_path=args.load_path
    )

