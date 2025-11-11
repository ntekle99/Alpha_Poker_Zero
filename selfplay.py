"""Self-play script for generating poker training data."""

import os
import numpy as np
from tqdm import tqdm
from config import PokerConfig as cfg
from poker_game import PokerGame, PokerGameState
from poker_mcts import PokerMCTS
from poker_dataset import PokerTrainingDataset
from value_policy_network import ValuePolicyNetworkWrapper


def play_single_hand(game: PokerGame, mcts: PokerMCTS, temperature: float = 1.0) -> tuple:
    """
    Play a single hand of poker using MCTS for both players.

    Args:
        game: PokerGame instance
        mcts: PokerMCTS instance
        temperature: Temperature for action selection

    Returns:
        Tuple of (hand_data, final_reward) where hand_data is a list of
        (state_vector, action_probs, player) tuples
    """
    # Initialize new hand
    state = game.init_new_hand()
    hand_data = []

    action_count = 0
    max_actions = 50  # Prevent infinite loops

    # Play until hand is terminal or max actions reached
    while not game.is_terminal(state) and action_count < max_actions:
        current_player = state.current_player

        # Get canonical state representation
        state_vector = game.get_canonical_state(state, current_player)

        # Run MCTS from this position
        root_node = mcts.init_root_node(state, current_player)
        root_node = mcts.run_simulation(root_node)

        # Select action
        action, action_probs = mcts.select_action(root_node, temperature=temperature)

        # Store data for training
        hand_data.append((state_vector, action_probs, current_player))

        # Apply action to state
        state = game.apply_action(state, action, current_player)

        action_count += 1

    # Get final reward from player 1's perspective
    if game.is_terminal(state):
        final_reward = game.get_reward(state, player=1)
    else:
        # Max actions reached without terminal state - treat as draw
        final_reward = 0.0

    return hand_data, final_reward


def generate_selfplay_data(num_games: int = None, save_interval: int = 100, model_path: str = None):
    """
    Generate self-play data for training.

    Args:
        num_games: Number of games to play
        save_interval: Save dataset every N games
        model_path: Path to pre-trained model (optional)
    """
    if num_games is None:
        num_games = cfg.SELFPLAY_GAMES

    # Create output directories
    os.makedirs(cfg.SAVE_PICKLES, exist_ok=True)
    os.makedirs(cfg.SAVE_MODEL_PATH, exist_ok=True)

    save_path = os.path.join(cfg.SAVE_PICKLES, cfg.DATASET_PATH)

    # Initialize components
    print("Initializing components...")
    game = PokerGame()
    vpn = ValuePolicyNetworkWrapper(model_path=model_path)
    mcts = PokerMCTS(game, vpn.get_vp)
    dataset = PokerTrainingDataset()

    # Try to load existing dataset
    if os.path.exists(save_path):
        dataset.load(save_path)
        print(f"Loaded existing dataset with {len(dataset)} samples")

    print(f"\nStarting self-play for {num_games} games...")
    print(f"MCTS simulations per action: {cfg.NUM_SIMULATIONS}")
    print(f"Dataset will be saved to: {save_path}")
    print()

    # Statistics
    wins_p1 = 0
    wins_p2 = 0
    draws = 0
    total_actions = 0

    # Play games
    for game_num in tqdm(range(num_games), desc="Playing games", total=num_games):
        try:
            # Play a single hand
            hand_data, final_reward = play_single_hand(game, mcts, temperature=cfg.TEMPERATURE)

            # Add to dataset
            dataset.add_game_to_training_dataset(hand_data, final_reward)

            # Update statistics
            total_actions += len(hand_data)
            if final_reward > 0:
                wins_p1 += 1
            elif final_reward < 0:
                wins_p2 += 1
            else:
                draws += 1

            # Save periodically
            if (game_num + 1) % save_interval == 0:
                dataset.save(save_path)
                stats = dataset.get_stats()
                print(f"\n[Game {game_num + 1}/{num_games}]")
                print(f"  Dataset size: {stats['size']}")
                print(f"  Mean reward: {stats['mean_reward']:.3f}")
                print(f"  P1 wins: {wins_p1}, P2 wins: {wins_p2}, Draws: {draws}")
                print(f"  Avg actions per hand: {total_actions / (game_num + 1):.1f}")

        except Exception as e:
            print(f"\nError in game {game_num}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final save
    print("\nSaving final dataset...")
    dataset.save(save_path)

    # Print final statistics
    print("\n" + "=" * 60)
    print("SELF-PLAY COMPLETE")
    print("=" * 60)
    stats = dataset.get_stats()
    print(f"Total games played: {num_games}")
    print(f"Total samples collected: {stats['size']}")
    print(f"Average actions per hand: {total_actions / num_games:.1f}")
    print(f"\nGame outcomes:")
    print(f"  Player 1 wins: {wins_p1} ({100 * wins_p1 / num_games:.1f}%)")
    print(f"  Player 2 wins: {wins_p2} ({100 * wins_p2 / num_games:.1f}%)")
    print(f"  Draws: {draws} ({100 * draws / num_games:.1f}%)")
    print(f"\nDataset statistics:")
    print(f"  Mean reward: {stats['mean_reward']:.3f}")
    print(f"  Std reward: {stats['std_reward']:.3f}")
    print(f"  Max reward: {stats['max_reward']:.3f}")
    print(f"  Min reward: {stats['min_reward']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate poker self-play data")
    parser.add_argument("--games", type=int, default=None,
                        help=f"Number of games to play (default: {cfg.SELFPLAY_GAMES})")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Save dataset every N games (default: 100)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to pre-trained model (optional)")
    parser.add_argument("--simulations", type=int, default=None,
                        help=f"MCTS simulations per action (default: {cfg.NUM_SIMULATIONS})")

    args = parser.parse_args()

    # Update config if specified
    if args.simulations is not None:
        cfg.NUM_SIMULATIONS = args.simulations

    # Run self-play
    generate_selfplay_data(
        num_games=args.games,
        save_interval=args.save_interval,
        model_path=args.model
    )
