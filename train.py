"""Training script for poker neural network using self-play data."""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import argparse
from config import PokerConfig as cfg
from value_policy_network import PokerValuePolicyNetwork, ValuePolicyNetworkWrapper
from poker_dataset import PokerTrainingDataset


class PokerLoss(nn.Module):
    """Combined loss function for policy and value."""
    
    def __init__(self, value_weight=1.0, policy_weight=1.0, l2_weight=1e-4):
        super(PokerLoss, self).__init__()
        self.value_weight = value_weight
        self.policy_weight = policy_weight
        self.l2_weight = l2_weight
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    
    def forward(self, policy_pred, value_pred, policy_target, value_target):
        """
        Compute combined loss.
        
        Args:
            policy_pred: Predicted policy logits (batch_size, num_actions)
            value_pred: Predicted values (batch_size, 1)
            policy_target: Target policy probabilities (batch_size, num_actions)
            value_target: Target values (batch_size, 1)
        """
        # Policy loss: cross-entropy between predicted and target distributions
        # Convert target probabilities to log probabilities for numerical stability
        policy_loss = -torch.sum(policy_target * torch.log(policy_pred + 1e-8), dim=1).mean()
        
        # Value loss: MSE between predicted and actual values
        value_loss = self.mse_loss(value_pred.squeeze(), value_target.squeeze())
        
        # Combined loss
        total_loss = self.policy_weight * policy_loss + self.value_weight * value_loss
        
        return total_loss, policy_loss, value_loss


def create_data_loader(dataset: PokerTrainingDataset, batch_size: int, shuffle: bool = True):
    """Create a PyTorch DataLoader from the dataset."""
    if len(dataset) == 0:
        return None
    
    # Convert to numpy arrays
    states = np.array(list(dataset.states))
    action_probs = np.array(list(dataset.action_probs))
    rewards = np.array(list(dataset.rewards))
    
    # Convert to tensors
    states_tensor = torch.FloatTensor(states)
    action_probs_tensor = torch.FloatTensor(action_probs)
    rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
    
    # Create dataset and dataloader
    tensor_dataset = TensorDataset(states_tensor, action_probs_tensor, rewards_tensor)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    return dataloader


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    
    for states, action_probs, rewards in tqdm(dataloader, desc="Training", leave=False):
        states = states.to(device)
        action_probs = action_probs.to(device)
        rewards = rewards.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        policy_pred, value_pred = model(states)
        
        # Compute loss
        loss, policy_loss, value_loss = criterion(
            policy_pred, value_pred, action_probs, rewards
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
    avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_policy_loss, avg_value_loss


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for states, action_probs, rewards in dataloader:
            states = states.to(device)
            action_probs = action_probs.to(device)
            rewards = rewards.to(device)
            
            # Forward pass
            policy_pred, value_pred = model(states)
            
            # Compute loss
            loss, policy_loss, value_loss = criterion(
                policy_pred, value_pred, action_probs, rewards
            )
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
    avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_policy_loss, avg_value_loss


def train_model(
    dataset_path: str = None,
    model_path: str = None,
    output_path: str = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    weight_decay: float = None,
    train_split: float = 0.9,
    save_interval: int = 10,
):
    """
    Train the poker neural network.
    
    Args:
        dataset_path: Path to training dataset pickle file
        model_path: Path to pre-trained model (optional)
        output_path: Path to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        weight_decay: L2 regularization weight
        train_split: Fraction of data to use for training (rest for validation)
        save_interval: Save model every N epochs
    """
    # Use config defaults if not specified
    if dataset_path is None:
        dataset_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)
    if output_path is None:
        output_path = os.path.join(cfg.SAVE_MODEL_PATH, "poker_model.pt")
    if epochs is None:
        epochs = cfg.EPOCHS
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    if learning_rate is None:
        learning_rate = cfg.LEARNING_RATE
    if weight_decay is None:
        weight_decay = cfg.WEIGHT_DECAY if hasattr(cfg, 'WEIGHT_DECAY') else 1e-4
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = PokerTrainingDataset()
    if os.path.exists(dataset_path):
        dataset.load(dataset_path)
        print(f"Loaded dataset with {len(dataset)} samples")
    else:
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return
    
    # Split into train and validation
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    # Create separate datasets (simple approach - split indices)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))
    
    # Create data loaders
    print("Creating data loaders...")
    train_states = np.array([dataset.states[i] for i in train_indices])
    train_action_probs = np.array([dataset.action_probs[i] for i in train_indices])
    train_rewards = np.array([dataset.rewards[i] for i in train_indices])
    
    val_states = np.array([dataset.states[i] for i in val_indices])
    val_action_probs = np.array([dataset.action_probs[i] for i in val_indices])
    val_rewards = np.array([dataset.rewards[i] for i in val_indices])
    
    train_dataset = TensorDataset(
        torch.FloatTensor(train_states),
        torch.FloatTensor(train_action_probs),
        torch.FloatTensor(train_rewards).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_states),
        torch.FloatTensor(val_action_probs),
        torch.FloatTensor(val_rewards).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    print("Initializing model...")
    model = PokerValuePolicyNetwork()
    device = model.device
    print(f"Using device: {device}")
    
    # Load pre-trained model if provided
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded pre-trained model from {model_path}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting with random weights")
    
    # Loss function and optimizer
    criterion = PokerLoss(
        value_weight=cfg.VALUE_LOSS_WEIGHT if hasattr(cfg, 'VALUE_LOSS_WEIGHT') else 1.0,
        policy_weight=cfg.POLICY_LOSS_WEIGHT if hasattr(cfg, 'POLICY_LOSS_WEIGHT') else 1.0,
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    # Set minimum learning rate to 1% of initial LR to prevent it from going to zero
    min_lr = learning_rate * 0.1
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=min_lr
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Minimum learning rate: {min_lr:.8f}")
    print(f"Weight decay: {weight_decay}")
    print("=" * 60 + "\n")
    
    best_val_loss = float('inf')
    best_model_path = output_path.replace('.pt', '_best.pt')
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_policy_loss, train_value_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_policy_loss, val_value_loss = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"Train Loss: {train_loss:.6f} (Policy: {train_policy_loss:.6f}, Value: {train_value_loss:.6f})")
        print(f"Val Loss:   {val_loss:.6f} (Policy: {val_policy_loss:.6f}, Value: {val_value_loss:.6f})")
        print(f"Learning Rate: {current_lr:.6f}", end="")
        if current_lr < old_lr:
            print(f" (reduced from {old_lr:.6f})")
        else:
            print()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! (Val Loss: {val_loss:.6f})")
        
        # Periodic saves
        if epoch % save_interval == 0:
            checkpoint_path = output_path.replace('.pt', f'_epoch_{epoch}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    torch.save(model.state_dict(), output_path)
    print(f"\nTraining complete!")
    print(f"Final model saved to: {output_path}")
    print(f"Best model saved to: {best_model_path} (Val Loss: {best_val_loss:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train poker neural network")
    parser.add_argument("--dataset", type=str, default=None,
                        help=f"Path to dataset (default: {cfg.SAVE_PICKLES}/{cfg.DATASET_PATH})")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to pre-trained model (optional)")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Output path for trained model (default: {cfg.SAVE_MODEL_PATH}/poker_model.pt)")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Number of epochs (default: {cfg.EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Batch size (default: {cfg.BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=None,
                        help=f"Learning rate (default: {cfg.LEARNING_RATE})")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay for L2 regularization (default: 1e-4)")
    parser.add_argument("--train-split", type=float, default=0.9,
                        help="Fraction of data for training (default: 0.9)")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")
    
    args = parser.parse_args()
    
    train_model(
        dataset_path=args.dataset,
        model_path=args.model,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
        save_interval=args.save_interval,
    )

