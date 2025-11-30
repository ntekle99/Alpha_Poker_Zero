"""Compare DQN bot vs MCTS bot."""

import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
from config import PokerConfig as cfg
from poker_game import PokerGame, PokerGameState
from dqn_network import DQNNetwork
from value_policy_network import ValuePolicyNetworkWrapper
from poker_mcts import PokerMCTS


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


def compare_dqn_vs_mcts(
    dqn_model_path: str,
    mcts_model_path: str,
    num_games: int = 100,
    hands_per_game: int = None,
    mcts_simulations: int = None,
    dqn_is_player1: bool = True,
    dqn_deterministic: bool = True
):
    """
    Compare DQN bot vs MCTS bot.
    
    A "game" consists of multiple hands. The winner is determined by
    who has more chips at the end of all hands.
    
    Args:
        dqn_model_path: Path to DQN-trained model
        mcts_model_path: Path to MCTS-trained model
        num_games: Number of games (matches) to play
        hands_per_game: Number of hands per game (default: cfg.HANDS_PER_GAME)
        mcts_simulations: MCTS simulations per action
        dqn_is_player1: If True, DQN plays as Player 1; else Player 2
        dqn_deterministic: If True, DQN uses greedy policy; else samples
    """
    if mcts_simulations is None:
        mcts_simulations = cfg.NUM_SIMULATIONS
    if hands_per_game is None:
        hands_per_game = cfg.HANDS_PER_GAME
    
    print("=" * 60)
    print("DQN vs MCTS COMPARISON")
    print("=" * 60)
    print(f"DQN Model: {dqn_model_path}")
    print(f"MCTS Model: {mcts_model_path}")
    print(f"DQN plays as: {'Player 1' if dqn_is_player1 else 'Player 2'}")
    print(f"DQN policy: {'Deterministic' if dqn_deterministic else 'Stochastic'}")
    print(f"Games (matches): {num_games}")
    print(f"Hands per game: {hands_per_game}")
    print(f"Total hands: {num_games * hands_per_game}")
    print(f"MCTS Simulations: {mcts_simulations}")
    print("=" * 60 + "\n")
    
    # Load DQN model - try to detect behavior_embedding_dim from saved weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(dqn_model_path, map_location=device)
    
    # Try to detect behavior_embedding_dim from model weights
    behavior_dim = 0
    if "fc1.weight" in checkpoint:
        input_size = checkpoint["fc1.weight"].shape[1]
        base_state_size = cfg.STATE_SIZE
        if input_size > base_state_size:
            behavior_dim = input_size - base_state_size
            print(f"Detected behavior_embedding_dim: {behavior_dim}")
    
    dqn_model = DQNNetwork(behavior_embedding_dim=behavior_dim)
    dqn_model.to(device)
    
    try:
        dqn_model.load_state_dict(checkpoint)
        print(f"Loaded DQN model from {dqn_model_path}")
        if behavior_dim > 0:
            print(f"  Model uses behavior embeddings (dim: {behavior_dim})")
    except Exception as e:
        error_msg = str(e)
        if "size mismatch" in error_msg:
            print(f"\nDQN Model incompatible: State size mismatch")
            print(f"   The saved model was trained with a different state representation.")
            print(f"   Detected input size: {checkpoint.get('fc1.weight', torch.tensor(0)).shape[1] if 'fc1.weight' in checkpoint else 'unknown'}")
            print(f"   Current state size: {cfg.STATE_SIZE}")
            print(f"\n   Try with: --use-opponent-modeling --behavior-embedding-dim 16")
            return
        else:
            print(f"Error loading DQN model: {e}")
            return
    
    dqn_model.eval()
    
    # Load MCTS model
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
    dqn_game_wins = 0
    mcts_game_wins = 0
    draws = 0
    dqn_total_final_chips = 0
    mcts_total_final_chips = 0
    total_actions = 0
    
    print("Playing games...")
    for game_num in tqdm(range(num_games), desc="Comparing"):
        try:
            # Initialize chip stacks for this game
            p1_chips = cfg.STARTING_CHIPS
            p2_chips = cfg.STARTING_CHIPS
            game_total_chips = p1_chips + p2_chips
            
            # Play multiple hands in this game
            for hand_num in range(hands_per_game):
                state = game.init_new_hand()
                
                # Set chip stacks (carry over from previous hands, minus blinds)
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
                
                # Verify chip conservation
                total_in_state = state.player1_stack + state.player2_stack + state.pot
                if abs(game_total_chips - total_in_state) > 0.01:
                    state.pot = game_total_chips - state.player1_stack - state.player2_stack
                    state.pot = max(cfg.SMALL_BLIND + cfg.BIG_BLIND, state.pot)
                
                action_count = 0
                max_actions = 50
                
                while not game.is_terminal(state) and action_count < max_actions:
                    current_player = state.current_player
                    is_dqn_turn = (current_player == 1 and dqn_is_player1) or (current_player == -1 and not dqn_is_player1)
                    
                    if is_dqn_turn:
                        # DQN bot's turn
                        action = dqn_select_action(dqn_model, game, state, current_player, deterministic=dqn_deterministic)
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
                    total_in_state = state.player1_stack + state.player2_stack + state.pot
                    
                    if state.winner == 1:
                        p1_chips = state.player1_stack + state.pot
                        p2_chips = state.player2_stack
                    elif state.winner == -1:
                        p1_chips = state.player1_stack
                        p2_chips = state.player2_stack + state.pot
                    else:
                        p1_chips = state.player1_stack + state.pot // 2
                        p2_chips = state.player2_stack + (state.pot - state.pot // 2)
                    
                    # Ensure chips are conserved
                    total_after_hand = p1_chips + p2_chips
                    if abs(total_after_hand - game_total_chips) > 0.01:
                        if total_after_hand > 0:
                            scale = game_total_chips / total_after_hand
                            p1_chips = int(p1_chips * scale)
                            p2_chips = game_total_chips - p1_chips
                        else:
                            p1_chips = game_total_chips // 2
                            p2_chips = game_total_chips - p1_chips
                    
                    p1_chips = max(0, p1_chips)
                    p2_chips = max(0, p2_chips)
                    p1_chips = game_total_chips - p2_chips
            
            # Determine game winner
            if dqn_is_player1:
                dqn_chips = p1_chips
                mcts_chips = p2_chips
            else:
                dqn_chips = p2_chips
                mcts_chips = p1_chips
            
            dqn_total_final_chips += dqn_chips
            mcts_total_final_chips += mcts_chips
            
            if dqn_chips > mcts_chips:
                dqn_game_wins += 1
            elif mcts_chips > dqn_chips:
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
    print(f"  DQN Bot: {dqn_game_wins} ({100 * dqn_game_wins / num_games:.1f}%)")
    print(f"  MCTS Bot: {mcts_game_wins} ({100 * mcts_game_wins / num_games:.1f}%)")
    print(f"  Draws: {draws} ({100 * draws / num_games:.1f}%)")
    
    # Chip statistics
    total_starting_chips = cfg.STARTING_CHIPS * 2 * num_games
    total_final_chips = dqn_total_final_chips + mcts_total_final_chips
    
    dqn_net_gain = dqn_total_final_chips - (cfg.STARTING_CHIPS * num_games)
    mcts_net_loss = (cfg.STARTING_CHIPS * num_games) - mcts_total_final_chips
    
    avg_dqn_chips = dqn_total_final_chips / num_games if num_games > 0 else 0
    avg_mcts_chips = mcts_total_final_chips / num_games if num_games > 0 else 0
    avg_dqn_net = dqn_net_gain / num_games if num_games > 0 else 0
    avg_mcts_net = mcts_net_loss / num_games if num_games > 0 else 0
    
    dqn_percentage = (dqn_total_final_chips / total_final_chips) * 100 if total_final_chips > 0 else 0
    mcts_percentage = (mcts_total_final_chips / total_final_chips) * 100 if total_final_chips > 0 else 0
    
    print(f"\nChip statistics:")
    print(f"  Starting chips per game: {cfg.STARTING_CHIPS * 2} ({cfg.STARTING_CHIPS} per player)")
    print(f"  Total starting chips ({num_games} games): {total_starting_chips:,}")
    print(f"  Total final chips: {total_final_chips:,}")
    print(f"")
    print(f"  DQN Bot final chips: {dqn_total_final_chips:,} ({dqn_percentage:.1f}% of final total)")
    print(f"    Net gain: {dqn_net_gain:+,d} ({avg_dqn_net:+.1f} per game)")
    print(f"    Average per game: {avg_dqn_chips:.1f} chips")
    print(f"  MCTS Bot final chips: {mcts_total_final_chips:,} ({mcts_percentage:.1f}% of final total)")
    print(f"    Net loss: {mcts_net_loss:+,d} ({avg_mcts_net:+.1f} per game)")
    print(f"    Average per game: {avg_mcts_chips:.1f} chips")
    
    # Analysis
    dqn_win_rate = dqn_game_wins / num_games if num_games > 0 else 0.0
    print(f"\nAnalysis:")
    if dqn_win_rate > 0.55:
        print(f"  DQN Bot is STRONGER! ({dqn_win_rate*100:.1f}% win rate)")
        print(f"     DQN's Q-learning approach outperforms MCTS tree search")
    elif dqn_win_rate > 0.50:
        print(f"  DQN Bot is slightly stronger ({dqn_win_rate*100:.1f}% win rate)")
    elif dqn_win_rate > 0.45:
        print(f"  MCTS Bot is slightly stronger ({(1-dqn_win_rate)*100:.1f}% win rate)")
    else:
        print(f"  MCTS Bot is STRONGER! ({(1-dqn_win_rate)*100:.1f}% win rate)")
        print(f"     MCTS tree search outperforms DQN's Q-learning")
    
    if avg_dqn_net > 0:
        print(f"  DQN is winning chips on average (+{avg_dqn_net:.1f} per game)")
    else:
        print(f"  DQN is losing chips on average ({avg_dqn_net:.1f} per game)")
    
    print(f"\nAverage actions per hand: {total_actions / (num_games * hands_per_game):.1f}")
    print("=" * 60)
    
    return dqn_win_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DQN bot vs MCTS bot")
    parser.add_argument("--dqn-model", type=str, required=True,
                        help="Path to DQN model")
    parser.add_argument("--mcts-model", type=str, required=True,
                        help="Path to MCTS model")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games (matches) to play (default: 100)")
    parser.add_argument("--hands-per-game", type=int, default=None,
                        help=f"Number of hands per game (default: {cfg.HANDS_PER_GAME})")
    parser.add_argument("--simulations", type=int, default=None,
                        help=f"MCTS simulations per action (default: {cfg.NUM_SIMULATIONS})")
    parser.add_argument("--dqn-player", type=int, choices=[1, 2], default=1,
                        help="Which player DQN plays as (1 or 2, default: 1)")
    parser.add_argument("--dqn-stochastic", action="store_true",
                        help="Use stochastic policy for DQN")
    
    args = parser.parse_args()
    
    compare_dqn_vs_mcts(
        dqn_model_path=args.dqn_model,
        mcts_model_path=args.mcts_model,
        num_games=args.games,
        hands_per_game=args.hands_per_game,
        mcts_simulations=args.simulations,
        dqn_is_player1=(args.dqn_player == 1),
        dqn_deterministic=not args.dqn_stochastic
    )

