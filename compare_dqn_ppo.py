"""Compare DQN bot vs PPO bot."""

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from config import PokerConfig as cfg
from poker_game import PokerGame, PokerGameState
from value_policy_network import PokerValuePolicyNetwork
from dqn_network import DQNNetwork


def ppo_select_action(
    model: PokerValuePolicyNetwork,
    game: PokerGame,
    state: PokerGameState,
    player: int,
    deterministic: bool = True
) -> int:
    """
    Select action using PPO model (direct policy, no MCTS).
    
    Args:
        model: PPO-trained policy network
        game: PokerGame instance (for valid moves)
        state: Current game state
        player: Current player (1 or -1)
        deterministic: If True, use greedy; else sample
    
    Returns:
        Action index
    """
    state_vector = state.to_vector(player)
    valid_actions = game.get_valid_moves(state, player)
    
    model.eval()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(model.device)
        policy, _ = model(state_tensor)
        policy = policy.cpu().numpy()[0]
        policy = policy * valid_actions
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy = valid_actions / valid_actions.sum()
        
        if deterministic:
            action = np.argmax(policy)
        else:
            action = np.random.choice(len(policy), p=policy)
    
    return action


def dqn_select_action(
    model: DQNNetwork,
    game: PokerGame,
    state: PokerGameState,
    player: int,
    deterministic: bool = True
) -> int:
    """
    Select action using DQN model.
    
    Args:
        model: DQN-trained Q-network
        game: PokerGame instance (for valid moves)
        state: Current game state
        player: Current player (1 or -1)
        deterministic: If True, use greedy; else sample
    
    Returns:
        Action index
    """
    state_vector = state.to_vector(player)
    valid_actions = game.get_valid_moves(state, player)
    
    model.eval()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(model.device)
        q_values = model(state_tensor)
        q_values = q_values.cpu().numpy()[0]
        
        # Mask invalid actions
        masked_q_values = q_values + (1 - valid_actions) * -1e9
        
        if deterministic:
            action = np.argmax(masked_q_values)
        else:
            # Sample from valid actions based on Q-values
            valid_indices = np.where(valid_actions == 1)[0]
            if len(valid_indices) > 0:
                valid_q = q_values[valid_indices]
                probs = np.exp(valid_q - np.max(valid_q))  # Softmax
                probs = probs / probs.sum()
                action = np.random.choice(valid_indices, p=probs)
            else:
                action = np.random.choice(cfg.NUM_ACTIONS)
    
    return action


def compare_dqn_vs_ppo(
    dqn_model_path: str,
    ppo_model_path: str,
    num_games: int = 100,
    hands_per_game: int = None,
    dqn_is_player1: bool = True,
    dqn_deterministic: bool = True,
    ppo_deterministic: bool = True
):
    """
    Compare DQN bot vs PPO bot.
    
    A "game" consists of multiple hands. The winner is determined by
    who has more chips at the end of all hands.
    
    Args:
        dqn_model_path: Path to DQN-trained model
        ppo_model_path: Path to PPO-trained model
        num_games: Number of games (matches) to play
        hands_per_game: Number of hands per game (default: cfg.HANDS_PER_GAME)
        dqn_is_player1: If True, DQN plays as Player 1; else Player 2
        dqn_deterministic: If True, DQN uses greedy policy; else samples
        ppo_deterministic: If True, PPO uses greedy policy; else samples
    """
    if hands_per_game is None:
        hands_per_game = cfg.HANDS_PER_GAME
    
    print("=" * 60)
    print("DQN vs PPO COMPARISON")
    print("=" * 60)
    print(f"DQN Model: {dqn_model_path}")
    print(f"PPO Model: {ppo_model_path}")
    print(f"DQN plays as: {'Player 1' if dqn_is_player1 else 'Player 2'}")
    print(f"DQN policy: {'Deterministic' if dqn_deterministic else 'Stochastic'}")
    print(f"PPO policy: {'Deterministic' if ppo_deterministic else 'Stochastic'}")
    print(f"Games (matches): {num_games}")
    print(f"Hands per game: {hands_per_game}")
    print(f"Total hands: {num_games * hands_per_game}")
    print("=" * 60 + "\n")
    
    # Load DQN model
    dqn_model = DQNNetwork()
    device = dqn_model.device
    
    try:
        dqn_model.load_state_dict(torch.load(dqn_model_path, map_location=device))
        print(f"Loaded DQN model from {dqn_model_path}")
    except Exception as e:
        error_msg = str(e)
        if "size mismatch" in error_msg:
            print(f"\nDQN Model incompatible: State size mismatch")
            print(f"   The saved model was trained with a different state representation.")
            print(f"   Current state size: {dqn_model.fc1.in_features}")
            print(f"\n   Solution: Retrain the DQN model with the current state representation:")
            print(f"   python3 train_dqn.py --episodes 10000")
            return
        else:
            print(f"Error loading DQN model: {e}")
            return
    
    dqn_model.eval()
    
    # Load PPO model
    ppo_model = PokerValuePolicyNetwork()
    
    try:
        ppo_model.load_state_dict(torch.load(ppo_model_path, map_location=device))
        print(f"Loaded PPO model from {ppo_model_path}")
    except Exception as e:
        error_msg = str(e)
        if "size mismatch" in error_msg:
            print(f"\nPPO Model incompatible: State size mismatch")
            print(f"   The saved model was trained with a different state representation.")
            print(f"   Current state size: {ppo_model.fc1.in_features}")
            print(f"\n   Solution: Retrain the PPO model with the current state representation:")
            print(f"   python3 train_ppo.py --updates 1000")
            return
        else:
            print(f"Error loading PPO model: {e}")
            return
    
    ppo_model.eval()
    
    # Initialize game
    game = PokerGame()
    
    # Statistics
    dqn_game_wins = 0
    ppo_game_wins = 0
    draws = 0
    dqn_total_final_chips = 0  # Sum of actual final chips across all games
    ppo_total_final_chips = 0  # Sum of actual final chips across all games
    total_actions = 0
    
    print("Playing games...")
    for game_num in tqdm(range(num_games), desc="Comparing"):
        try:
            # Initialize chip stacks for this game
            p1_chips = cfg.STARTING_CHIPS
            p2_chips = cfg.STARTING_CHIPS
            game_total_chips = p1_chips + p2_chips  # Track total chips for this game
            
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
                
                action_count = 0
                max_actions = 50
                
                while not game.is_terminal(state) and action_count < max_actions:
                    current_player = state.current_player
                    is_dqn_turn = (current_player == 1 and dqn_is_player1) or (current_player == -1 and not dqn_is_player1)
                    
                    if is_dqn_turn:
                        # DQN bot's turn
                        action = dqn_select_action(dqn_model, game, state, current_player, deterministic=dqn_deterministic)
                    else:
                        # PPO bot's turn
                        action = ppo_select_action(ppo_model, game, state, current_player, deterministic=ppo_deterministic)
                    
                    # Apply action
                    state = game.apply_action(state, action, current_player)
                    action_count += 1
                    total_actions += 1
                
                # Update chip stacks after hand
                if game.is_terminal(state):
                    # Verify chip conservation before updating
                    total_in_state = state.player1_stack + state.player2_stack + state.pot
                    
                    # When a hand ends, the pot is separate from stacks
                    # We need to add the pot to the winner's stack
                    if state.winner == 1:
                        # Player 1 won the pot
                        p1_chips = state.player1_stack + state.pot
                        p2_chips = state.player2_stack
                    elif state.winner == -1:
                        # Player 2 won the pot
                        p1_chips = state.player1_stack
                        p2_chips = state.player2_stack + state.pot
                    else:
                        # Draw - split pot
                        p1_chips = state.player1_stack + state.pot // 2
                        p2_chips = state.player2_stack + (state.pot - state.pot // 2)
                    
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
            
            # Determine game winner based on final chip count
            if dqn_is_player1:
                dqn_chips = p1_chips
                ppo_chips = p2_chips
            else:
                dqn_chips = p2_chips
                ppo_chips = p1_chips
            
            # Track actual final chips (not net gains)
            dqn_total_final_chips += dqn_chips
            ppo_total_final_chips += ppo_chips
            
            if dqn_chips > ppo_chips:
                dqn_game_wins += 1
            elif ppo_chips > dqn_chips:
                ppo_game_wins += 1
            else:
                draws += 1
                
        except Exception as e:
            print(f"\nError in game {game_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Games (matches) played: {num_games}")
    print(f"Hands per game: {hands_per_game}")
    print(f"Total hands: {num_games * hands_per_game}")
    
    print(f"\nGame win rates:")
    print(f"  DQN Bot: {dqn_game_wins} ({100 * dqn_game_wins / num_games:.1f}%)")
    print(f"  PPO Bot: {ppo_game_wins} ({100 * ppo_game_wins / num_games:.1f}%)")
    print(f"  Draws: {draws} ({100 * draws / num_games:.1f}%)")
    
    # Chip statistics
    total_starting_chips = cfg.STARTING_CHIPS * 2 * num_games  # 2 players * num_games
    total_final_chips = dqn_total_final_chips + ppo_total_final_chips
    
    # Calculate net gains/losses
    dqn_net_gain = dqn_total_final_chips - (cfg.STARTING_CHIPS * num_games)
    ppo_net_loss = (cfg.STARTING_CHIPS * num_games) - ppo_total_final_chips
    
    avg_dqn_chips = dqn_total_final_chips / num_games if num_games > 0 else 0
    avg_ppo_chips = ppo_total_final_chips / num_games if num_games > 0 else 0
    avg_dqn_net = dqn_net_gain / num_games if num_games > 0 else 0
    avg_ppo_net = ppo_net_loss / num_games if num_games > 0 else 0
    
    dqn_percentage = (dqn_total_final_chips / total_final_chips) * 100 if total_final_chips > 0 else 0
    ppo_percentage = (ppo_total_final_chips / total_final_chips) * 100 if total_final_chips > 0 else 0
    
    print(f"\nChip statistics:")
    print(f"  Starting chips per game: {cfg.STARTING_CHIPS * 2} ({cfg.STARTING_CHIPS} per player)")
    print(f"  Total starting chips ({num_games} games): {total_starting_chips:,}")
    print(f"  Total final chips: {total_final_chips:,}")
    print(f"")
    print(f"  DQN Bot final chips: {dqn_total_final_chips:,} ({dqn_percentage:.1f}% of final total)")
    print(f"    Net gain: {dqn_net_gain:+,d} ({avg_dqn_net:+.1f} per game)")
    print(f"    Average per game: {avg_dqn_chips:.1f} chips")
    print(f"  PPO Bot final chips: {ppo_total_final_chips:,} ({ppo_percentage:.1f}% of final total)")
    print(f"    Net loss: {ppo_net_loss:+,d} ({avg_ppo_net:+.1f} per game)")
    print(f"    Average per game: {avg_ppo_chips:.1f} chips")
    
    # Analysis
    dqn_win_rate = dqn_game_wins / num_games if num_games > 0 else 0.0
    print(f"\nAnalysis:")
    if dqn_win_rate > 0.55:
        print(f"  DQN Bot is STRONGER! ({dqn_win_rate*100:.1f}% win rate)")
        print(f"     DQN's Q-learning approach outperforms PPO's policy optimization")
    elif dqn_win_rate > 0.50:
        print(f"  DQN Bot is slightly stronger ({dqn_win_rate*100:.1f}% win rate)")
    elif dqn_win_rate > 0.45:
        print(f"  PPO Bot is slightly stronger ({(1-dqn_win_rate)*100:.1f}% win rate)")
    else:
        print(f"  PPO Bot is STRONGER! ({(1-dqn_win_rate)*100:.1f}% win rate)")
        print(f"     PPO's policy optimization outperforms DQN's Q-learning")
    
    if avg_dqn_net > 0:
        print(f"  DQN is winning chips on average (+{avg_dqn_net:.1f} per game)")
    else:
        print(f"  DQN is losing chips on average ({avg_dqn_net:.1f} per game)")
    
    print(f"\nAverage actions per hand: {total_actions / (num_games * hands_per_game):.1f}")
    print("=" * 60)
    
    return dqn_win_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DQN bot vs PPO bot")
    parser.add_argument("--dqn-model", type=str, required=True,
                        help="Path to DQN model")
    parser.add_argument("--ppo-model", type=str, required=True,
                        help="Path to PPO model")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games (matches) to play (default: 100)")
    parser.add_argument("--hands-per-game", type=int, default=None,
                        help=f"Number of hands per game (default: {cfg.HANDS_PER_GAME})")
    parser.add_argument("--dqn-player1", action="store_true", default=True,
                        help="DQN plays as Player 1 (default: True)")
    parser.add_argument("--dqn-player2", action="store_true",
                        help="DQN plays as Player 2")
    parser.add_argument("--dqn-stochastic", action="store_true",
                        help="Use stochastic policy for DQN")
    parser.add_argument("--ppo-stochastic", action="store_true",
                        help="Use stochastic policy for PPO")
    
    args = parser.parse_args()
    
    dqn_is_player1 = not args.dqn_player2 if args.dqn_player2 else args.dqn_player1
    
    compare_dqn_vs_ppo(
        dqn_model_path=args.dqn_model,
        ppo_model_path=args.ppo_model,
        num_games=args.games,
        hands_per_game=args.hands_per_game,
        dqn_is_player1=dqn_is_player1,
        dqn_deterministic=not args.dqn_stochastic,
        ppo_deterministic=not args.ppo_stochastic
    )

