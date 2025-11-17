"""Evaluation script to test trained poker models."""

import os
import numpy as np
from tqdm import tqdm
import argparse
import random
from config import PokerConfig as cfg
from poker_game import PokerGame, PokerGameState
from poker_mcts import PokerMCTS
from value_policy_network import ValuePolicyNetworkWrapper


def evaluate_model(model_path: str, num_games: int = 100, num_simulations: int = None):
    """
    Evaluate a trained model by playing games.
    
    Args:
        model_path: Path to trained model
        num_games: Number of games to play for evaluation
        num_simulations: MCTS simulations per action (default: config value)
    """
    if num_simulations is None:
        num_simulations = cfg.NUM_SIMULATIONS
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Games: {num_games}")
    print(f"MCTS Simulations: {num_simulations}")
    print("=" * 60 + "\n")
    
    # Initialize components
    game = PokerGame()
    vpn = ValuePolicyNetworkWrapper(model_path=model_path)
    mcts = PokerMCTS(game, vpn.get_vp)
    
    # Temporarily override simulations for evaluation
    original_simulations = cfg.NUM_SIMULATIONS
    cfg.NUM_SIMULATIONS = num_simulations
    
    # Statistics
    wins_p1 = 0
    wins_p2 = 0
    draws = 0
    total_actions = 0
    rewards_p1 = []
    rewards_p2 = []
    
    # Play games
    print("Playing evaluation games...")
    for game_num in tqdm(range(num_games), desc="Evaluating"):
        try:
            # Play a single hand
            state = game.init_new_hand()
            action_count = 0
            max_actions = 50
            
            while not game.is_terminal(state) and action_count < max_actions:
                current_player = state.current_player
                
                # Run MCTS
                root_node = mcts.init_root_node(state, current_player)
                root_node = mcts.run_simulation(root_node, num_simulations=num_simulations)
                
                # Select action (use temperature=0 for evaluation = greedy)
                action, _ = mcts.select_action(root_node, temperature=0.0)
                
                # Apply action
                state = game.apply_action(state, action, current_player)
                action_count += 1
            
            # Get rewards
            if game.is_terminal(state):
                reward_p1 = game.get_reward(state, player=1)
                reward_p2 = game.get_reward(state, player=-1)
                
                rewards_p1.append(reward_p1)
                rewards_p2.append(reward_p2)
                
                if reward_p1 > 0:
                    wins_p1 += 1
                elif reward_p1 < 0:
                    wins_p2 += 1
                else:
                    draws += 1
                
                total_actions += action_count
        except Exception as e:
            print(f"\nError in game {game_num}: {e}")
            continue
    
    # Restore original simulations
    cfg.NUM_SIMULATIONS = original_simulations
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Games played: {num_games}")
    print(f"\nWin rates:")
    print(f"  Player 1: {wins_p1} ({100 * wins_p1 / num_games:.1f}%)")
    print(f"  Player 2: {wins_p2} ({100 * wins_p2 / num_games:.1f}%)")
    print(f"  Draws: {draws} ({100 * draws / num_games:.1f}%)")
    
    if rewards_p1:
        print(f"\nReward statistics (Player 1):")
        print(f"  Mean: {np.mean(rewards_p1):.4f}")
        print(f"  Std: {np.std(rewards_p1):.4f}")
        print(f"  Min: {np.min(rewards_p1):.4f}")
        print(f"  Max: {np.max(rewards_p1):.4f}")
    
    print(f"\nAverage actions per hand: {total_actions / num_games:.1f}")
    print("=" * 60)
    
    # Return win rate as a metric
    win_rate = wins_p1 / num_games if num_games > 0 else 0.0
    return win_rate


def random_action_to_index(state: PokerGameState, player: int, action_str: str, amount: int) -> int:
    """
    Convert RandomPlayer action to action index for PokerGame.
    
    Args:
        state: Current game state
        player: Current player (1 or -1)
        action_str: Action string ('fold', 'call', 'check', 'raise')
        amount: Bet/raise amount (if applicable)
    
    Returns:
        Action index (0-5)
    """
    if action_str == 'fold':
        return cfg.ACTION_FOLD
    elif action_str == 'check' or action_str == 'call':
        return cfg.ACTION_CHECK_CALL
    elif action_str == 'raise':
        # Convert raise amount to appropriate bet size
        my_bet = state.player1_bet if player == 1 else state.player2_bet
        my_stack = state.player1_stack if player == 1 else state.player2_stack
        to_call = state.current_bet - my_bet
        total_bet = to_call + amount
        
        # Determine bet size relative to pot
        if total_bet >= my_stack:
            return cfg.ACTION_ALL_IN
        elif total_bet >= state.pot * 1.5:  # Large bet (2x pot)
            return cfg.ACTION_BET_LARGE
        elif total_bet >= state.pot * 0.75:  # Medium bet (1x pot)
            return cfg.ACTION_BET_MEDIUM
        else:  # Small bet (0.5x pot)
            return cfg.ACTION_BET_SMALL
    else:
        # Default to check/call
        return cfg.ACTION_CHECK_CALL


def get_random_action(state: PokerGameState, player: int) -> int:
    """
    Get a random action for a player (simulating RandomPlayer behavior).
    
    Args:
        state: Current game state
        player: Current player (1 or -1)
    
    Returns:
        Action index (0-5)
    """
    my_bet = state.player1_bet if player == 1 else state.player2_bet
    my_stack = state.player1_stack if player == 1 else state.player2_stack
    to_call = state.current_bet - my_bet
    
    # Get valid actions
    valid_actions = []
    
    if to_call > 0:
        valid_actions.append(cfg.ACTION_FOLD)
    
    valid_actions.append(cfg.ACTION_CHECK_CALL)
    
    if my_stack > to_call:
        remaining_after_call = my_stack - to_call
        
        if remaining_after_call >= state.pot * 0.5:
            valid_actions.append(cfg.ACTION_BET_SMALL)
        if remaining_after_call >= state.pot:
            valid_actions.append(cfg.ACTION_BET_MEDIUM)
        if remaining_after_call >= state.pot * 2:
            valid_actions.append(cfg.ACTION_BET_LARGE)
        valid_actions.append(cfg.ACTION_ALL_IN)
    
    # Randomly select from valid actions (weighted towards check/call)
    if len(valid_actions) == 0:
        return cfg.ACTION_CHECK_CALL
    
    # Weight actions: check/call more likely, fold less likely
    weights = []
    for action in valid_actions:
        if action == cfg.ACTION_CHECK_CALL:
            weights.append(3.0)  # More likely to check/call
        elif action == cfg.ACTION_FOLD:
            weights.append(1.0)  # Less likely to fold
        else:
            weights.append(1.5)  # Moderate chance for bets
    
    # Normalize weights
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    
    return np.random.choice(valid_actions, p=probabilities)


def evaluate_vs_random(model_path: str, num_games: int = 100, num_simulations: int = None, model_is_player1: bool = True, hands_per_game: int = None):
    """
    Evaluate a trained model against a random player.
    
    A "game" consists of multiple hands. The winner is determined by
    who has more chips at the end of all hands.
    
    Args:
        model_path: Path to trained model
        num_games: Number of games (matches) to play
        num_simulations: MCTS simulations per action (default: config value)
        model_is_player1: If True, model plays as Player 1, else Player 2
        hands_per_game: Number of hands per game (default: cfg.HANDS_PER_GAME)
    """
    if num_simulations is None:
        num_simulations = cfg.NUM_SIMULATIONS
    if hands_per_game is None:
        hands_per_game = cfg.HANDS_PER_GAME
    
    print("=" * 60)
    print("MODEL vs RANDOM PLAYER EVALUATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Model plays as: {'Player 1' if model_is_player1 else 'Player 2'}")
    print(f"Games (matches): {num_games}")
    print(f"Hands per game: {hands_per_game}")
    print(f"Total hands: {num_games * hands_per_game}")
    print(f"MCTS Simulations: {num_simulations}")
    print("=" * 60 + "\n")
    
    # Initialize components
    game = PokerGame()
    vpn = ValuePolicyNetworkWrapper(model_path=model_path)
    mcts = PokerMCTS(game, vpn.get_vp)
    
    # Temporarily override simulations for evaluation
    original_simulations = cfg.NUM_SIMULATIONS
    cfg.NUM_SIMULATIONS = num_simulations
    
    # Statistics
    model_game_wins = 0
    random_game_wins = 0
    draws = 0
    model_total_final_chips = 0
    random_total_final_chips = 0
    total_actions = 0
    
    # Play games
    print("Playing games (Model vs Random)...")
    for game_num in tqdm(range(num_games), desc="Evaluating"):
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
                    is_model_turn = (current_player == 1 and model_is_player1) or (current_player == -1 and not model_is_player1)
                    
                    if is_model_turn:
                        # Model's turn - use MCTS
                        root_node = mcts.init_root_node(state, current_player)
                        root_node = mcts.run_simulation(root_node, num_simulations=num_simulations)
                        action, _ = mcts.select_action(root_node, temperature=0.0)
                    else:
                        # Random player's turn
                        action = get_random_action(state, current_player)
                    
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
            
            # Determine game winner based on final chip count
            if model_is_player1:
                model_chips = p1_chips
                random_chips = p2_chips
            else:
                model_chips = p2_chips
                random_chips = p1_chips
            
            model_total_final_chips += model_chips
            random_total_final_chips += random_chips
            
            if model_chips > random_chips:
                model_game_wins += 1
            elif random_chips > model_chips:
                random_game_wins += 1
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
    print("EVALUATION RESULTS (Model vs Random)")
    print("=" * 60)
    print(f"Games (matches) played: {num_games}")
    print(f"Hands per game: {hands_per_game}")
    print(f"Total hands: {num_games * hands_per_game}")
    
    print(f"\nGame win rates:")
    print(f"  Trained Model: {model_game_wins} ({100 * model_game_wins / num_games:.1f}%)")
    print(f"  Random Player: {random_game_wins} ({100 * random_game_wins / num_games:.1f}%)")
    print(f"  Draws: {draws} ({100 * draws / num_games:.1f}%)")
    
    # Chip statistics
    total_starting_chips = cfg.STARTING_CHIPS * 2 * num_games
    total_final_chips = model_total_final_chips + random_total_final_chips
    
    model_net_gain = model_total_final_chips - (cfg.STARTING_CHIPS * num_games)
    random_net_loss = (cfg.STARTING_CHIPS * num_games) - random_total_final_chips
    
    avg_model_chips = model_total_final_chips / num_games if num_games > 0 else 0
    avg_random_chips = random_total_final_chips / num_games if num_games > 0 else 0
    avg_model_net = model_net_gain / num_games if num_games > 0 else 0
    avg_random_net = random_net_loss / num_games if num_games > 0 else 0
    
    model_percentage = (model_total_final_chips / total_final_chips) * 100 if total_final_chips > 0 else 0
    random_percentage = (random_total_final_chips / total_final_chips) * 100 if total_final_chips > 0 else 0
    
    print(f"\nChip statistics:")
    print(f"  Starting chips per game: {cfg.STARTING_CHIPS * 2} ({cfg.STARTING_CHIPS} per player)")
    print(f"  Total starting chips ({num_games} games): {total_starting_chips:,}")
    print(f"  Total final chips: {total_final_chips:,}")
    print(f"")
    print(f"  Model final chips: {model_total_final_chips:,} ({model_percentage:.1f}% of final total)")
    print(f"    Net gain: {model_net_gain:+,d} ({avg_model_net:+.1f} per game)")
    print(f"    Average per game: {avg_model_chips:.1f} chips")
    print(f"  Random Player final chips: {random_total_final_chips:,} ({random_percentage:.1f}% of final total)")
    print(f"    Net loss: {random_net_loss:+,d} ({avg_random_net:+.1f} per game)")
    print(f"    Average per game: {avg_random_chips:.1f} chips")
    
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


def compare_models(model1_path: str, model2_path: str, num_games: int = 100):
    """
    Compare two models by having them play against each other.
    
    Args:
        model1_path: Path to first model
        model2_path: Path to second model
        num_games: Number of games to play
    """
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"Model 1: {model1_path}")
    print(f"Model 2: {model2_path}")
    print(f"Games: {num_games}")
    print("=" * 60 + "\n")
    
    # Initialize components
    game = PokerGame()
    vpn1 = ValuePolicyNetworkWrapper(model_path=model1_path)
    vpn2 = ValuePolicyNetworkWrapper(model_path=model2_path)
    
    mcts1 = PokerMCTS(game, vpn1.get_vp)
    mcts2 = PokerMCTS(game, vpn2.get_vp)
    
    # Statistics
    wins_model1 = 0
    wins_model2 = 0
    draws = 0
    
    # Play games
    print("Playing comparison games...")
    for game_num in tqdm(range(num_games), desc="Comparing"):
        try:
            state = game.init_new_hand()
            action_count = 0
            max_actions = 50
            
            while not game.is_terminal(state) and action_count < max_actions:
                current_player = state.current_player
                
                # Use appropriate MCTS based on player
                if current_player == 1:
                    mcts = mcts1
                else:
                    mcts = mcts2
                
                # Run MCTS
                root_node = mcts.init_root_node(state, current_player)
                root_node = mcts.run_simulation(root_node)
                
                # Select action (greedy for evaluation)
                action, _ = mcts.select_action(root_node, temperature=0.0)
                
                # Apply action
                state = game.apply_action(state, action, current_player)
                action_count += 1
            
            # Get winner
            if game.is_terminal(state):
                reward_p1 = game.get_reward(state, player=1)
                if reward_p1 > 0:
                    wins_model1 += 1
                elif reward_p1 < 0:
                    wins_model2 += 1
                else:
                    draws += 1
        except Exception as e:
            print(f"\nError in game {game_num}: {e}")
            continue
    
    # Print results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"Model 1 wins: {wins_model1} ({100 * wins_model1 / num_games:.1f}%)")
    print(f"Model 2 wins: {wins_model2} ({100 * wins_model2 / num_games:.1f}%)")
    print(f"Draws: {draws} ({100 * draws / num_games:.1f}%)")
    print("=" * 60)
    
    return wins_model1 / num_games if num_games > 0 else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate poker models")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model to evaluate")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games for evaluation (default: 100)")
    parser.add_argument("--simulations", type=int, default=None,
                        help=f"MCTS simulations per action (default: {cfg.NUM_SIMULATIONS})")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to second model for comparison")
    parser.add_argument("--vs-random", action="store_true",
                        help="Evaluate model against random player")
    parser.add_argument("--model-player", type=int, choices=[1, 2], default=1,
                        help="Which player the model plays as (1 or 2, default: 1)")
    parser.add_argument("--hands-per-game", type=int, default=None,
                        help=f"Number of hands per game (default: {cfg.HANDS_PER_GAME})")
    
    args = parser.parse_args()
    
    if args.vs_random:
        evaluate_vs_random(args.model, args.games, args.simulations, 
                          model_is_player1=(args.model_player == 1),
                          hands_per_game=args.hands_per_game)
    elif args.compare:
        compare_models(args.model, args.compare, args.games)
    else:
        evaluate_model(args.model, args.games, args.simulations)

