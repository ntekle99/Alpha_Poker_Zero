"""Evaluation script for DQN-trained poker models."""

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from config import PokerConfig as cfg
from poker_game import PokerGame
from dqn_network import DQNNetwork
from evaluate import get_random_action


def evaluate_dqn_model(
    model_path: str,
    num_games: int = 100,
    hands_per_game: int = None,
    vs_random: bool = True,
    deterministic: bool = True
):
    """
    Evaluate a DQN-trained model.
    
    A "game" consists of multiple hands. The winner is determined by
    who has more chips at the end of all hands.
    
    Args:
        model_path: Path to trained model
        num_games: Number of games (matches) to play
        hands_per_game: Number of hands per game (default: cfg.HANDS_PER_GAME)
        vs_random: If True, play against random; else self-play
        deterministic: If True, use greedy policy; else sample
    """
    if hands_per_game is None:
        hands_per_game = cfg.HANDS_PER_GAME
    
    print("=" * 60)
    print("DQN MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Games (matches): {num_games}")
    print(f"Hands per game: {hands_per_game}")
    print(f"Total hands: {num_games * hands_per_game}")
    print(f"Mode: {'vs Random' if vs_random else 'Self-play'}")
    print(f"Policy: {'Deterministic' if deterministic else 'Stochastic'}")
    print("=" * 60 + "\n")
    
    # Load model
    model = DQNNetwork()
    device = model.device
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        error_msg = str(e)
        if "size mismatch" in error_msg and "fc1.weight" in error_msg:
            print(f"\nModel incompatible: State size mismatch")
            print(f"   The saved model was trained with a different state representation.")
            print(f"   Current state size: {model.fc1.in_features}")
            if "326" in error_msg:
                print(f"   Saved model state size: 326 (old version with enhanced features)")
                print(f"   Current state size: 320 (simplified version)")
            print(f"\n   Solution: Retrain the model with the current state representation:")
            print(f"   python3 train_dqn.py --episodes 10000")
        else:
            print(f"Error loading model: {e}")
        return
    
    model.eval()
    
    # Initialize game
    game = PokerGame()
    
    # Statistics
    model_game_wins = 0
    opponent_game_wins = 0
    draws = 0
    model_total_final_chips = 0  # Sum of actual final chips across all games
    opponent_total_final_chips = 0  # Sum of actual final chips across all games
    total_actions = 0
    total_pot_won = 0
    total_pot_contributed = 0
    
    print("Playing evaluation games...")
    for game_num in tqdm(range(num_games), desc="Evaluating"):
        try:
            # Initialize chip stacks for this game
            p1_chips = cfg.STARTING_CHIPS
            p2_chips = cfg.STARTING_CHIPS
            game_total_chips = p1_chips + p2_chips  # Track total chips for this game
            
            # Track pot statistics for this game
            game_pot_won = 0
            game_pot_contributed = 0
            
            # Play multiple hands in this game
            for hand_num in range(hands_per_game):
                state = game.init_new_hand()
                
                # Set chip stacks (carry over from previous hands, minus blinds)
                # The pot already has the blinds from init_new_hand(), so we just update stacks
                state.player1_stack = p1_chips - cfg.SMALL_BLIND
                state.player2_stack = p2_chips - cfg.BIG_BLIND
                
                # Ensure stacks don't go negative
                if state.player1_stack < 0:
                    state.player1_stack = 0
                if state.player2_stack < 0:
                    state.player2_stack = 0
                
                # Reset pot to match the blinds we subtracted (ensures chip conservation)
                state.pot = cfg.SMALL_BLIND + cfg.BIG_BLIND
                state.player1_bet = cfg.SMALL_BLIND
                state.player2_bet = cfg.BIG_BLIND
                state.current_bet = cfg.BIG_BLIND
                
                # Verify chip conservation: stacks + pot should equal total chips
                total_in_state = state.player1_stack + state.player2_stack + state.pot
                if abs(game_total_chips - total_in_state) > 0.01:
                    # Fix: adjust pot to ensure conservation
                    state.pot = game_total_chips - state.player1_stack - state.player2_stack
                    state.pot = max(cfg.SMALL_BLIND + cfg.BIG_BLIND, state.pot)  # Pot can't be less than blinds
                
                # Track starting stack for this hand (to calculate contribution)
                p1_start_hand = state.player1_stack
                
                action_count = 0
                max_actions = 50
                
                while not game.is_terminal(state) and action_count < max_actions:
                    current_player = state.current_player
                    state_vector = game.get_canonical_state(state, current_player)
                    valid_actions = game.get_valid_moves(state, current_player)
                    
                    # Select action
                    if vs_random and current_player == -1:
                        # Random player
                        action = get_random_action(state, current_player)
                    else:
                        # Model player
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
                            q_values = model(state_tensor)
                            q_values = q_values.cpu().numpy()[0]
                            
                            # Mask invalid actions
                            q_values = q_values * valid_actions
                            q_values = np.where(valid_actions == 1, q_values, -np.inf)
                            
                            if deterministic:
                                action = np.argmax(q_values)
                            else:
                                # Sample from valid actions based on Q-values
                                valid_indices = np.where(valid_actions == 1)[0]
                                valid_q = q_values[valid_indices]
                                probs = np.exp(valid_q - np.max(valid_q))  # Softmax
                                probs = probs / probs.sum()
                                action = np.random.choice(valid_indices, p=probs)
                    
                    # Apply action
                    state = game.apply_action(state, action, current_player)
                    action_count += 1
                    total_actions += 1
                
                # Update chip stacks after hand
                if game.is_terminal(state):
                    # Verify chip conservation before updating
                    total_in_state = state.player1_stack + state.player2_stack + state.pot
                    
                    # Calculate pot contribution for this hand
                    p1_contributed = p1_start_hand - state.player1_stack
                    game_pot_contributed += p1_contributed
                    
                    # When a hand ends, the pot is separate from stacks
                    # We need to add the pot to the winner's stack
                    if state.winner == 1:
                        # Player 1 won the pot
                        p1_chips = state.player1_stack + state.pot
                        p2_chips = state.player2_stack
                        game_pot_won += state.pot
                    elif state.winner == -1:
                        # Player 2 won the pot
                        p1_chips = state.player1_stack
                        p2_chips = state.player2_stack + state.pot
                    else:
                        # Draw - split pot
                        p1_chips = state.player1_stack + state.pot // 2
                        p2_chips = state.player2_stack + (state.pot - state.pot // 2)
                        game_pot_won += state.pot // 2
                    
                    # Ensure chips are conserved: total should equal game_total_chips
                    total_after_hand = p1_chips + p2_chips
                    if abs(total_after_hand - game_total_chips) > 0.01:
                        # Fix: scale chips to preserve ratio while conserving total
                        if total_after_hand > 0:
                            scale = game_total_chips / total_after_hand
                            p1_chips = int(p1_chips * scale)
                            p2_chips = game_total_chips - p1_chips
                        else:
                            # Both players busted, split evenly
                            p1_chips = game_total_chips // 2
                            p2_chips = game_total_chips - p1_chips
                    
                    # Ensure stacks are non-negative
                    p1_chips = max(0, p1_chips)
                    p2_chips = max(0, p2_chips)
                    
                    # Final verification: ensure total is exactly game_total_chips
                    p1_chips = game_total_chips - p2_chips
            
            # Accumulate pot statistics
            total_pot_won += game_pot_won
            total_pot_contributed += game_pot_contributed
            
            # Determine game winner based on final chip count
            # Track actual final chips (not net gains)
            model_total_final_chips += p1_chips
            opponent_total_final_chips += p2_chips
            
            if p1_chips > p2_chips:
                model_game_wins += 1
            elif p2_chips > p1_chips:
                opponent_game_wins += 1
            else:
                draws += 1
                
        except Exception as e:
            print(f"\nError in game {game_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Games (matches) played: {num_games}")
    print(f"Hands per game: {hands_per_game}")
    print(f"Total hands: {num_games * hands_per_game}")
    print(f"\nGame win rates:")
    if vs_random:
        print(f"  DQN Model: {model_game_wins} ({100 * model_game_wins / num_games:.1f}%)")
        print(f"  Random Player: {opponent_game_wins} ({100 * opponent_game_wins / num_games:.1f}%)")
    else:
        print(f"  Player 1: {model_game_wins} ({100 * model_game_wins / num_games:.1f}%)")
        print(f"  Player 2: {opponent_game_wins} ({100 * opponent_game_wins / num_games:.1f}%)")
    print(f"  Draws: {draws} ({100 * draws / num_games:.1f}%)")
    
    # Chip statistics
    total_starting_chips = cfg.STARTING_CHIPS * 2 * num_games  # 2 players * num_games
    total_final_chips = model_total_final_chips + opponent_total_final_chips
    
    # Calculate net gains/losses
    model_net_gain = model_total_final_chips - (cfg.STARTING_CHIPS * num_games)
    opponent_net_loss = (cfg.STARTING_CHIPS * num_games) - opponent_total_final_chips
    
    avg_model_chips = model_total_final_chips / num_games if num_games > 0 else 0
    avg_opponent_chips = opponent_total_final_chips / num_games if num_games > 0 else 0
    avg_model_net = model_net_gain / num_games if num_games > 0 else 0
    avg_opponent_net = opponent_net_loss / num_games if num_games > 0 else 0
    
    model_percentage = (model_total_final_chips / total_final_chips) * 100 if total_final_chips > 0 else 0
    opponent_percentage = (opponent_total_final_chips / total_final_chips) * 100 if total_final_chips > 0 else 0
    
    print(f"\nChip statistics:")
    print(f"  Starting chips per game: {cfg.STARTING_CHIPS * 2} ({cfg.STARTING_CHIPS} per player)")
    print(f"  Total starting chips ({num_games} games): {total_starting_chips:,}")
    print(f"  Total final chips: {total_final_chips:,}")
    print(f"")
    if vs_random:
        print(f"  Model final chips: {model_total_final_chips:,} ({model_percentage:.1f}% of final total)")
        print(f"    Net gain: {model_net_gain:+,d} ({avg_model_net:+.1f} per game)")
        print(f"    Average per game: {avg_model_chips:.1f} chips")
        print(f"  Random Player final chips: {opponent_total_final_chips:,} ({opponent_percentage:.1f}% of final total)")
        print(f"    Net loss: {opponent_net_loss:+,d} ({avg_opponent_net:+.1f} per game)")
        print(f"    Average per game: {avg_opponent_chips:.1f} chips")
    else:
        print(f"  Player 1 final chips: {model_total_final_chips:,} ({model_percentage:.1f}% of final total)")
        print(f"    Net gain: {model_net_gain:+,d} ({avg_model_net:+.1f} per game)")
        print(f"    Average per game: {avg_model_chips:.1f} chips")
        print(f"  Player 2 final chips: {opponent_total_final_chips:,} ({opponent_percentage:.1f}% of final total)")
        print(f"    Net loss: {opponent_net_loss:+,d} ({avg_opponent_net:+.1f} per game)")
        print(f"    Average per game: {avg_opponent_chips:.1f} chips")
    
    # Pot statistics (if available)
    if total_pot_contributed > 0:
        pot_win_rate = (total_pot_won / total_pot_contributed) * 100 if total_pot_contributed > 0 else 0
        print(f"\nPot statistics:")
        print(f"  Total pot won: {total_pot_won}")
        print(f"  Total pot contributed: {total_pot_contributed}")
        print(f"  Pot win rate: {pot_win_rate:.1f}% (won {pot_win_rate:.1f}% of pots contributed to)")
        if pot_win_rate > 100:
            print(f"  Excellent! Winning more than contributed (selective play)")
        elif pot_win_rate > 50:
            print(f"  Good! Winning majority of pots played")
        else:
            print(f"  Contributing to pots but not winning enough")
    
    win_rate = model_game_wins / num_games if num_games > 0 else 0.0
    print(f"\nPerformance Analysis:")
    if win_rate > 0.55:
        print(f"  Model is STRONG! ({win_rate*100:.1f}% win rate)")
    elif win_rate > 0.50:
        print(f"  Model is learning ({win_rate*100:.1f}% win rate)")
    elif win_rate > 0.45:
        print(f"  Model needs more training ({win_rate*100:.1f}% win rate)")
    else:
        print(f"  Model needs significant improvement ({win_rate*100:.1f}% win rate)")
    
    if avg_model_net > 0:
        print(f"  Model is winning chips on average (+{avg_model_net:.1f} per game)")
    else:
        print(f"  Model is losing chips on average ({avg_model_net:.1f} per game)")
    
    print(f"\nAverage actions per hand: {total_actions / (num_games * hands_per_game):.1f}")
    print("=" * 60)
    
    return win_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DQN-trained poker models")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to DQN model to evaluate")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games (matches) for evaluation (default: 100)")
    parser.add_argument("--hands-per-game", type=int, default=None,
                        help=f"Number of hands per game (default: {cfg.HANDS_PER_GAME})")
    parser.add_argument("--vs-random", action="store_true", default=True,
                        help="Evaluate against random player (default: True)")
    parser.add_argument("--self-play", action="store_true",
                        help="Evaluate with self-play instead of vs random")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of deterministic")
    
    args = parser.parse_args()
    
    evaluate_dqn_model(
        model_path=args.model,
        num_games=args.games,
        hands_per_game=args.hands_per_game,
        vs_random=not args.self_play if args.self_play else args.vs_random,
        deterministic=not args.stochastic
    )

