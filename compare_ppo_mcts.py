"""Compare PPO bot vs MCTS bot."""

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from config import PokerConfig as cfg
from poker_game import PokerGame, PokerGameState
from value_policy_network import PokerValuePolicyNetwork, ValuePolicyNetworkWrapper
from poker_mcts import PokerMCTS


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


def compare_ppo_vs_mcts(
    ppo_model_path: str,
    mcts_model_path: str,
    num_games: int = 100,
    hands_per_game: int = None,
    mcts_simulations: int = None,
    ppo_is_player1: bool = True,
    ppo_deterministic: bool = True
):
    """
    Compare PPO bot vs MCTS bot.
    
    A "game" consists of multiple hands. The winner is determined by
    who has more chips at the end of all hands.
    
    Args:
        ppo_model_path: Path to PPO-trained model
        mcts_model_path: Path to MCTS-trained model (or same as PPO)
        num_games: Number of games (matches) to play
        hands_per_game: Number of hands per game (default: cfg.HANDS_PER_GAME)
        mcts_simulations: MCTS simulations per action
        ppo_is_player1: If True, PPO plays as Player 1; else Player 2
        ppo_deterministic: If True, PPO uses greedy policy; else samples
    """
    if mcts_simulations is None:
        mcts_simulations = cfg.NUM_SIMULATIONS
    if hands_per_game is None:
        hands_per_game = cfg.HANDS_PER_GAME
    
    print("=" * 60)
    print("PPO vs MCTS COMPARISON")
    print("=" * 60)
    print(f"PPO Model: {ppo_model_path}")
    print(f"MCTS Model: {mcts_model_path}")
    print(f"PPO plays as: {'Player 1' if ppo_is_player1 else 'Player 2'}")
    print(f"PPO policy: {'Deterministic' if ppo_deterministic else 'Stochastic'}")
    print(f"Games (matches): {num_games}")
    print(f"Hands per game: {hands_per_game}")
    print(f"Total hands: {num_games * hands_per_game}")
    print(f"MCTS Simulations: {mcts_simulations}")
    print("=" * 60 + "\n")
    
    # Load PPO model
    ppo_model = PokerValuePolicyNetwork()
    device = ppo_model.device
    
    try:
        ppo_model.load_state_dict(torch.load(ppo_model_path, map_location=device))
        print(f"Loaded PPO model from {ppo_model_path}")
    except Exception as e:
        print(f"Error loading PPO model: {e}")
        return
    
    # Load MCTS model (can be same or different)
    try:
        vpn_wrapper = ValuePolicyNetworkWrapper(model_path=mcts_model_path)
        print(f"Loaded MCTS model from {mcts_model_path}")
    except Exception as e:
        print(f"Error loading MCTS model: {e}")
        return
    
    # Initialize game and MCTS
    game = PokerGame()
    mcts = PokerMCTS(game, vpn_wrapper.get_vp)
    
    # Temporarily override simulations
    original_simulations = cfg.NUM_SIMULATIONS
    cfg.NUM_SIMULATIONS = mcts_simulations
    
    # Statistics
    ppo_game_wins = 0
    mcts_game_wins = 0
    draws = 0
    ppo_total_final_chips = 0  # Sum of actual final chips across all games
    mcts_total_final_chips = 0  # Sum of actual final chips across all games
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
                    is_ppo_turn = (current_player == 1 and ppo_is_player1) or (current_player == -1 and not ppo_is_player1)
                    
                    if is_ppo_turn:
                        # PPO bot's turn - direct policy
                        action = ppo_select_action(ppo_model, game, state, current_player, deterministic=ppo_deterministic)
                    else:
                        # MCTS bot's turn
                        root_node = mcts.init_root_node(state, current_player)
                        root_node = mcts.run_simulation(root_node, num_simulations=mcts_simulations)
                        action, _ = mcts.select_action(root_node, temperature=0.0)
                    
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
            if ppo_is_player1:
                ppo_chips = p1_chips
                mcts_chips = p2_chips
            else:
                ppo_chips = p2_chips
                mcts_chips = p1_chips
            
            # Track actual final chips (not net gains)
            ppo_total_final_chips += ppo_chips
            mcts_total_final_chips += mcts_chips
            
            if ppo_chips > mcts_chips:
                ppo_game_wins += 1
            elif mcts_chips > ppo_chips:
                mcts_game_wins += 1
            else:
                draws += 1
                
        except Exception as e:
            print(f"\nError in game {game_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Restore original simulations
    cfg.NUM_SIMULATIONS = original_simulations
    
    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Games (matches) played: {num_games}")
    print(f"Hands per game: {hands_per_game}")
    print(f"Total hands: {num_games * hands_per_game}")
    print(f"\nGame win rates:")
    print(f"  PPO Bot: {ppo_game_wins} ({100 * ppo_game_wins / num_games:.1f}%)")
    print(f"  MCTS Bot: {mcts_game_wins} ({100 * mcts_game_wins / num_games:.1f}%)")
    print(f"  Draws: {draws} ({100 * draws / num_games:.1f}%)")
    
    # Chip statistics
    total_starting_chips = cfg.STARTING_CHIPS * 2 * num_games  # 2 players * num_games
    total_final_chips = ppo_total_final_chips + mcts_total_final_chips
    
    # Calculate net gains/losses
    ppo_net_gain = ppo_total_final_chips - (cfg.STARTING_CHIPS * num_games)
    mcts_net_loss = (cfg.STARTING_CHIPS * num_games) - mcts_total_final_chips
    
    avg_ppo_chips = ppo_total_final_chips / num_games if num_games > 0 else 0
    avg_mcts_chips = mcts_total_final_chips / num_games if num_games > 0 else 0
    avg_ppo_net = ppo_net_gain / num_games if num_games > 0 else 0
    avg_mcts_net = mcts_net_loss / num_games if num_games > 0 else 0
    
    ppo_percentage = (ppo_total_final_chips / total_final_chips) * 100 if total_final_chips > 0 else 0
    mcts_percentage = (mcts_total_final_chips / total_final_chips) * 100 if total_final_chips > 0 else 0
    
    print(f"\nChip statistics:")
    print(f"  Starting chips per game: {cfg.STARTING_CHIPS * 2} ({cfg.STARTING_CHIPS} per player)")
    print(f"  Total starting chips ({num_games} games): {total_starting_chips:,}")
    print(f"  Total final chips: {total_final_chips:,}")
    print(f"")
    print(f"  PPO Bot final chips: {ppo_total_final_chips:,} ({ppo_percentage:.1f}% of final total)")
    print(f"    Net gain: {ppo_net_gain:+,d} ({avg_ppo_net:+.1f} per game)")
    print(f"    Average per game: {avg_ppo_chips:.1f} chips")
    print(f"  MCTS Bot final chips: {mcts_total_final_chips:,} ({mcts_percentage:.1f}% of final total)")
    print(f"    Net loss: {mcts_net_loss:+,d} ({avg_mcts_net:+.1f} per game)")
    print(f"    Average per game: {avg_mcts_chips:.1f} chips")
    
    # Analysis
    ppo_win_rate = ppo_game_wins / num_games if num_games > 0 else 0.0
    print(f"\nAnalysis:")
    if ppo_win_rate > 0.55:
        print(f"  PPO Bot is STRONGER! ({ppo_win_rate*100:.1f}% win rate)")
        print(f"     PPO's direct policy learning outperforms MCTS tree search")
        if avg_ppo_chips > 0:
            print(f"     PPO is also winning chips on average (+{avg_ppo_chips:.1f} per game)")
    elif ppo_win_rate > 0.50:
        print(f"  PPO Bot has slight advantage ({ppo_win_rate*100:.1f}% win rate)")
        if avg_ppo_chips > avg_mcts_chips:
            print(f"     PPO is winning more chips on average")
    elif ppo_win_rate > 0.45:
        print(f"  MCTS Bot has slight advantage ({ppo_win_rate*100:.1f}% PPO win rate)")
        print(f"     MCTS tree search is helping the MCTS bot")
        if avg_mcts_chips > avg_ppo_chips:
            print(f"     MCTS is winning more chips on average")
    else:
        print(f"  MCTS Bot is STRONGER! ({ppo_win_rate*100:.1f}% PPO win rate)")
        print(f"     MCTS tree search significantly outperforms direct policy")
        if avg_mcts_chips > 0:
            print(f"     MCTS is also winning chips on average (+{avg_mcts_chips:.1f} per game)")
    
    print(f"\nAverage actions per hand: {total_actions / (num_games * hands_per_game):.1f}")
    print("=" * 60)
    
    return ppo_win_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PPO bot vs MCTS bot")
    parser.add_argument("--ppo-model", type=str, required=True,
                        help="Path to PPO-trained model")
    parser.add_argument("--mcts-model", type=str, default=None,
                        help="Path to MCTS-trained model (default: same as PPO)")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games (matches) to play (default: 100)")
    parser.add_argument("--hands-per-game", type=int, default=None,
                        help=f"Number of hands per game (default: {cfg.HANDS_PER_GAME})")
    parser.add_argument("--simulations", type=int, default=None,
                        help=f"MCTS simulations per action (default: {cfg.NUM_SIMULATIONS})")
    parser.add_argument("--ppo-player", type=int, choices=[1, 2], default=1,
                        help="Which player PPO plays as (1 or 2, default: 1)")
    parser.add_argument("--ppo-stochastic", action="store_true",
                        help="Use stochastic policy for PPO (default: deterministic)")
    
    args = parser.parse_args()
    
    # If MCTS model not specified, use same as PPO
    if args.mcts_model is None:
        args.mcts_model = args.ppo_model
    
    compare_ppo_vs_mcts(
        ppo_model_path=args.ppo_model,
        mcts_model_path=args.mcts_model,
        num_games=args.games,
        hands_per_game=args.hands_per_game,
        mcts_simulations=args.simulations,
        ppo_is_player1=(args.ppo_player == 1),
        ppo_deterministic=not args.ppo_stochastic
    )

