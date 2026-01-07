"""Flask backend server for Alpha Poker Zero."""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import torch
import os
from poker_game import PokerGame, PokerGameState
from evaluate import get_random_action
from config import PokerConfig as cfg
from card import Card, Suit, Rank

# Try to import AI models
try:
    from dqn_network import DQNNetwork
    from psycnet import PsychologyNetwork, OpponentHistoryManager, ActionFeatureEncoder
    AI_MODELS_AVAILABLE = True
except ImportError:
    AI_MODELS_AVAILABLE = False
    print("Warning: AI models not available. Advice feature will be limited.")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global game state
game = PokerGame()
current_state = None

# AI models for advice
dqn_model = None
psychology_model = None
opponent_history = None

def load_ai_models():
    """Load DQN and psychology models if available."""
    global dqn_model, psychology_model, opponent_history
    
    if not AI_MODELS_AVAILABLE:
        return
    
    try:
        # Try to load DQN model - check multiple possible filenames
        dqn_paths = [
            "poker-rl/output/models/dqn_finetuned_zproj.pt",
            "poker-rl/output/models/dqn_best.pth",
            "poker-rl/output/models/dqn_poker_model_best.pt"
        ]
        for dqn_path in dqn_paths:
            if os.path.exists(dqn_path):
                try:
                    print(f"Attempting to load DQN from {dqn_path}...")
                    state_dict = torch.load(dqn_path, map_location='cpu')
                    
                    # Check the input size from the saved model
                    # fc1.weight shape tells us the input size: [hidden_size, input_size]
                    if 'fc1.weight' in state_dict:
                        saved_input_size = state_dict['fc1.weight'].shape[1]
                        print(f"Saved model input size: {saved_input_size}")
                        
                        # If saved model has 320 input size, it doesn't have behavior embedding
                        if saved_input_size == 320:
                            print("Model was trained without behavior embedding, loading with behavior_embedding_dim=0")
                            dqn_model = DQNNetwork(behavior_embedding_dim=0)
                        else:
                            # Model has behavior embedding
                            behavior_dim = saved_input_size - 320
                            print(f"Model has behavior embedding dimension: {behavior_dim}")
                            dqn_model = DQNNetwork(behavior_embedding_dim=behavior_dim)
                    else:
                        # Default: try without behavior embedding first
                        dqn_model = DQNNetwork(behavior_embedding_dim=0)
                    
                    # Load the state dict
                    # The saved model may have z_proj keys that don't exist in current architecture
                    # So we'll use strict=False to ignore those
                    try:
                        dqn_model.load_state_dict(state_dict, strict=True)
                        print("✓ Loaded with strict=True")
                    except RuntimeError as e:
                        print(f"⚠ Strict load failed (expected if model has z_proj), trying strict=False: {e}")
                        # Filter out z_proj keys if they exist
                        filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith('z_proj')}
                        dqn_model.load_state_dict(filtered_dict, strict=False)
                        print("✓ Loaded with strict=False (ignored z_proj keys from saved model)")
                    
                    dqn_model.eval()
                    print(f"✓ Loaded DQN model from {dqn_path}")
                    break
                except Exception as e:
                    print(f"⚠ Error loading DQN from {dqn_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    dqn_model = None
                    continue
        if dqn_model is None:
            print(f"⚠ DQN model not found. Checked: {', '.join(dqn_paths)}")
        
        # Try to load psychology model - check multiple possible filenames
        psych_paths = [
            "poker-rl/output/models/psychology_network.pt",
            "poker-rl/output/models/psychology_network.pth"
        ]
        for psych_path in psych_paths:
            if os.path.exists(psych_path):
                try:
                    psychology_model = PsychologyNetwork(behavior_embedding_dim=16)
                    psychology_model.load_state_dict(torch.load(psych_path, map_location='cpu'))
                    psychology_model.eval()
                    print(f"✓ Loaded Psychology model from {psych_path}")
                    break
                except Exception as e:
                    print(f"⚠ Error loading Psychology model from {psych_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        if psychology_model is None:
            print(f"⚠ Psychology model not found. Checked: {', '.join(psych_paths)}")
        
        # Initialize opponent history manager
        opponent_history = OpponentHistoryManager(max_history_length=30)
        print("✓ Initialized opponent history manager")
        
    except Exception as e:
        print(f"⚠ Error loading AI models: {e}")
        dqn_model = None
        psychology_model = None
        opponent_history = None

# Load models on startup
load_ai_models()


def card_to_dict(card: Card) -> dict:
    """Convert Card object to dictionary."""
    suit_names = {Suit.CLUBS: 'clubs', Suit.DIAMONDS: 'diamonds', 
                  Suit.HEARTS: 'hearts', Suit.SPADES: 'spades'}
    rank_names = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                  9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    
    return {
        'rank': rank_names[card.rank],
        'suit': suit_names[card.suit]
    }


def to_python_int(value):
    """Convert numpy int64/int32 to Python int."""
    if hasattr(value, 'item'):
        return int(value.item())
    return int(value)

def to_python_float(value):
    """Convert numpy float64/float32 to Python float."""
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)

def state_to_json(state: PokerGameState, player_perspective: int = 1) -> dict:
    """Convert game state to JSON format for frontend."""
    # Determine which player is the user (player 1) and which is the bot (player 2)
    user_cards = state.hole_cards_p1 if player_perspective == 1 else state.hole_cards_p2
    bot_cards = state.hole_cards_p2 if player_perspective == 1 else state.hole_cards_p1
    
    # Convert cards
    user_cards_dict = [card_to_dict(card) for card in user_cards]
    bot_cards_dict = [card_to_dict(card) for card in bot_cards]
    community_cards_dict = [card_to_dict(card) for card in state.community_cards]
    
    # Get stacks
    user_stack = state.player1_stack if player_perspective == 1 else state.player2_stack
    bot_stack = state.player2_stack if player_perspective == 1 else state.player1_stack
    
    # Get bets
    user_bet = state.player1_bet if player_perspective == 1 else state.player2_bet
    bot_bet = state.player2_bet if player_perspective == 1 else state.player1_bet
    
    # Determine current player from user's perspective
    is_user_turn = (state.current_player == player_perspective)
    
    # Stage names
    stage_names = ['preflop', 'flop', 'turn', 'river']
    stage_name = stage_names[state.stage] if state.stage < len(stage_names) else 'showdown'
    
    return {
        'pot': to_python_int(state.pot),
        'smallBlind': cfg.SMALL_BLIND,
        'bigBlind': cfg.BIG_BLIND,
        'currentBet': to_python_int(state.current_bet),
        'gameStage': stage_name,
        'isTerminal': state.is_terminal,
        'players': [
            {
                'id': 1,
                'name': 'You',
                'chips': to_python_int(user_stack),
                'cards': user_cards_dict,
                'isActive': True,
                'isCurrentPlayer': is_user_turn,
                'position': 'right',
                'currentBet': to_python_int(user_bet)
            },
            {
                'id': 2,
                'name': 'Bot',
                'chips': to_python_int(bot_stack),
                'cards': bot_cards_dict if state.is_terminal else [{'suit': 'back', 'rank': 'back'}, {'suit': 'back', 'rank': 'back'}],
                'isActive': True,
                'isCurrentPlayer': not is_user_turn,
                'position': 'left',
                'currentBet': to_python_int(bot_bet)
            }
        ],
        'communityCards': community_cards_dict,
        'dealerPosition': 0,  # Player 1 is dealer for now
        'lastAction': state.last_action,
        'winner': to_python_int(state.winner) if state.is_terminal and state.winner is not None else None,
        'winnerName': 'You' if (state.is_terminal and state.winner == player_perspective) else ('Bot' if (state.is_terminal and state.winner == -player_perspective) else None)
    }


@app.route('/api/game/start', methods=['POST'])
def start_game():
    """Start a new hand."""
    global current_state, game
    
    game = PokerGame()
    current_state = game.init_new_hand()
    
    return jsonify({
        'success': True,
        'state': state_to_json(current_state)
    })


@app.route('/api/game/state', methods=['GET'])
def get_state():
    """Get current game state."""
    global current_state
    
    if current_state is None:
        return jsonify({
            'success': False,
            'error': 'No game in progress. Start a new game first.'
        }), 400
    
    return jsonify({
        'success': True,
        'state': state_to_json(current_state)
    })


@app.route('/api/game/action', methods=['POST'])
def make_action():
    """User makes an action."""
    global current_state, game
    
    if current_state is None:
        return jsonify({
            'success': False,
            'error': 'No game in progress.'
        }), 400
    
    if current_state.is_terminal:
        return jsonify({
            'success': False,
            'error': 'Hand is over. Start a new hand.'
        }), 400
    
    if current_state.current_player != 1:
        return jsonify({
            'success': False,
            'error': 'Not your turn.'
        }), 400
    
    data = request.json
    action = data.get('action')
    raise_amount = data.get('amount')
    
    # Handle raise with custom amount
    if action == 'raise' and raise_amount is not None:
        # raise_amount is the total amount to bet (including call)
        # We need to calculate the additional raise beyond the call
        my_bet = current_state.player1_bet
        my_stack = current_state.player1_stack
        to_call = current_state.current_bet - my_bet
        
        # Ensure raise_amount is valid
        if raise_amount < to_call:
            raise_amount = to_call  # Minimum is to call
        
        if raise_amount >= my_stack:
            action_index = cfg.ACTION_ALL_IN
        elif raise_amount >= current_state.pot * 2:
            action_index = cfg.ACTION_BET_LARGE
        elif raise_amount >= current_state.pot:
            action_index = cfg.ACTION_BET_MEDIUM
        elif raise_amount >= current_state.pot * 0.5:
            action_index = cfg.ACTION_BET_SMALL
        else:
            action_index = cfg.ACTION_BET_SMALL
    else:
        # Map frontend actions to backend action indices
        action_map = {
            'fold': cfg.ACTION_FOLD,
            'call': cfg.ACTION_CHECK_CALL,
            'check': cfg.ACTION_CHECK_CALL,
            'raise': cfg.ACTION_BET_MEDIUM,  # Default to medium bet
            'bet_small': cfg.ACTION_BET_SMALL,
            'bet_medium': cfg.ACTION_BET_MEDIUM,
            'bet_large': cfg.ACTION_BET_LARGE,
            'all_in': cfg.ACTION_ALL_IN
        }
        
        if action not in action_map:
            return jsonify({
                'success': False,
                'error': f'Invalid action: {action}'
            }), 400
        
        action_index = action_map[action]
    
    # Apply user action
    try:
        current_state = game.apply_action(current_state, action_index, 1)
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error applying action: {error_msg}'
        }), 500
    
    # If hand is over, return state
    if current_state.is_terminal:
        return jsonify({
            'success': True,
            'state': state_to_json(current_state),
            'handOver': True
        })
    
    # Bot's turn - get random action
    if current_state.current_player == -1:
        try:
            bot_bet_before = current_state.player2_bet
            current_bet_before = current_state.current_bet
            bot_action = get_random_action(current_state, -1)
            
            # Track bot action for psychology network
            if opponent_history is not None and AI_MODELS_AVAILABLE:
                try:
                    from psycnet import ActionFeatureEncoder
                    to_call = current_bet_before - bot_bet_before
                    bet_size = 0
                    if bot_action == cfg.ACTION_BET_SMALL:
                        bet_size = int(current_state.pot * 0.5)
                    elif bot_action == cfg.ACTION_BET_MEDIUM:
                        bet_size = current_state.pot
                    elif bot_action == cfg.ACTION_BET_LARGE:
                        bet_size = int(current_state.pot * 2)
                    elif bot_action == cfg.ACTION_ALL_IN:
                        bet_size = current_state.player2_stack
                    
                    action_features = ActionFeatureEncoder.encode_action_features(
                        action=bot_action,
                        bet_size=float(bet_size),
                        pot_size=float(current_state.pot),
                        stage=current_state.stage,
                        position=1,  # Bot is player 2 (BB position)
                        num_players=2,
                        had_initiative=(current_state.current_player == -1),
                        to_call=float(to_call),
                        raise_count=0,  # TODO: track raise count
                        is_all_in=(bot_action == cfg.ACTION_ALL_IN)
                    )
                    is_new_round = (current_state.stage == 0 and bot_bet_before == 0)
                    opponent_history.add_action(-1, action_features, is_new_betting_round=is_new_round)
                except Exception as e:
                    print(f"Error tracking bot action: {e}")
            
            current_state = game.apply_action(current_state, bot_action, -1)
            bot_bet_after = current_state.player2_bet
            current_bet_after = current_state.current_bet
            
            # Calculate raise amount if bot raised
            bot_raise_amount = None
            if current_state.last_action == 'raise':
                # Raise amount is the increase in current_bet
                bot_raise_amount = current_bet_after - current_bet_before
            
            # Check if hand is over after bot action
            if current_state.is_terminal:
                return jsonify({
                    'success': True,
                    'state': state_to_json(current_state),
                    'handOver': True,
                    'botAction': to_python_int(bot_action),
                    'botRaiseAmount': to_python_int(bot_raise_amount) if bot_raise_amount is not None else None
                })
            
            # Return state with bot action info
            return jsonify({
                'success': True,
                'state': state_to_json(current_state),
                'handOver': False,
                'botRaiseAmount': to_python_int(bot_raise_amount) if bot_raise_amount is not None else None
            })
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Error with bot action: {error_msg}'
            }), 500
    
    return jsonify({
        'success': True,
        'state': state_to_json(current_state),
        'handOver': False
    })


@app.route('/api/game/bot-action', methods=['POST'])
def bot_action():
    """Get bot's action (for manual bot turn handling if needed)."""
    global current_state, game
    
    if current_state is None:
        return jsonify({
            'success': False,
            'error': 'No game in progress.'
        }), 400
    
    if current_state.is_terminal:
        return jsonify({
            'success': False,
            'error': 'Hand is over.'
        }), 400
    
    if current_state.current_player != -1:
        return jsonify({
            'success': False,
            'error': 'Not bot\'s turn.'
        }), 400
    
    # Get random action for bot
    bot_action = get_random_action(current_state, -1)
    current_state = game.apply_action(current_state, bot_action, -1)
    
    return jsonify({
        'success': True,
        'state': state_to_json(current_state),
        'action': to_python_int(bot_action),
        'handOver': current_state.is_terminal
    })


@app.route('/api/game/valid-actions', methods=['GET'])
def get_valid_actions():
    """Get valid actions for current player."""
    global current_state, game
    
    if current_state is None:
        return jsonify({
            'success': False,
            'error': 'No game in progress.'
        }), 400
    
    if current_state.is_terminal:
        return jsonify({
            'success': False,
            'error': 'Hand is over.'
        }), 400
    
    valid_actions_mask = game.get_valid_moves(current_state, current_state.current_player)
    valid_actions = []
    
    action_names = ['fold', 'check_call', 'bet_small', 'bet_medium', 'bet_large', 'all_in']
    for i, is_valid in enumerate(valid_actions_mask):
        if is_valid:
            valid_actions.append(action_names[i])
    
    return jsonify({
        'success': True,
        'validActions': valid_actions,
        'currentPlayer': current_state.current_player
    })


def get_current_position_info(state: PokerGameState) -> dict:
    """Get current position information."""
    rank_map = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
    suit_map = {Suit.HEARTS: '♥', Suit.DIAMONDS: '♦', Suit.CLUBS: '♣', Suit.SPADES: '♠'}
    
    my_cards = state.hole_cards_p1
    my_cards_str = ' '.join([
        f"{rank_map.get(card.rank, card.rank)}{suit_map.get(card.suit, '')}"
        for card in my_cards
    ])
    
    stage_names = ['preflop', 'flop', 'turn', 'river']
    stage_name = stage_names[state.stage] if state.stage < len(stage_names) else 'showdown'
    
    community_cards_str = 'None'
    if state.community_cards:
        community_cards_str = ' '.join([
            f"{rank_map.get(card.rank, card.rank)}{suit_map.get(card.suit, '')}"
            for card in state.community_cards
        ])
    
    return {
        'myCards': my_cards_str,
        'myChips': int(state.player1_stack),
        'pot': int(state.pot),
        'currentBet': int(state.current_bet),
        'stage': stage_name,
        'communityCards': community_cards_str,
        'opponentChips': int(state.player2_stack),
        'blinds': f"{cfg.SMALL_BLIND}/{cfg.BIG_BLIND}"
    }


def generate_combined_advice(position_info, dqn_advice, dqn_probs, opponent_type, opponent_analysis, dqn_error=None, psych_error=None) -> str:
    """Generate combined advice from all sources with specific recommendations."""
    advice_parts = []
    
    # Position info (condensed)
    advice_parts.append("📊 Current Position:")
    advice_parts.append(f"Cards: {position_info['myCards']} | Pot: {position_info['pot']} | Bet: {position_info['currentBet']} | Stage: {position_info['stage']}")
    advice_parts.append("")
    
    # Opponent analysis - Psychology Network judgment
    if opponent_type and opponent_analysis:
        advice_parts.append(f"🎭 Opponent Profile (Psychology Network Analysis):")
        advice_parts.append(f"Type: {opponent_type}")
        advice_parts.append(f"Behavior Patterns:")
        advice_parts.append(f"  • Folds: {opponent_analysis['fold_frequency']:.1%} | Calls: {opponent_analysis['call_frequency']:.1%} | Raises: {opponent_analysis['raise_frequency']:.1%}")
        advice_parts.append("")
    elif psych_error:
        advice_parts.append(f"🎭 Opponent Profile:")
        if "Insufficient" in psych_error or "No opponent history" in psych_error:
            advice_parts.append("Waiting for opponent actions to analyze...")
            advice_parts.append("(Play a few hands to build opponent profile)")
        else:
            advice_parts.append(f"Analysis unavailable: {psych_error}")
        advice_parts.append("")
    
    # DQN output - Action recommendations
    if dqn_advice and dqn_probs:
        advice_parts.append(f"🤖 DQN AI Recommendation: {dqn_advice}")
        top_actions = sorted(
            [(k, v) for k, v in dqn_probs.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        advice_parts.append("Action Confidence:")
        for action, prob in top_actions:
            action_name = action.replace('_', ' ').title()
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            advice_parts.append(f"  {action_name:15} {bar} {prob:.1%}")
        advice_parts.append("")
    elif dqn_error:
        advice_parts.append(f"🤖 DQN AI Analysis:")
        if "not loaded" in dqn_error or "not found" in dqn_error:
            advice_parts.append("DQN model not available (model file missing)")
            advice_parts.append("To enable: Train DQN and save to poker-rl/output/models/dqn_best.pth")
        else:
            advice_parts.append(f"Error: {dqn_error}")
        advice_parts.append("")
    
    # Combined strategic decision
    advice_parts.append("💡 My Recommendation:")
    
    # Get best action from DQN
    best_action = None
    best_prob = 0
    if dqn_probs:
        for action, prob in dqn_probs.items():
            if prob > best_prob:
                best_prob = prob
                best_action = action
    
    # Generate specific advice based on opponent type + DQN
    if opponent_type == "Tight/Passive":
        pot = position_info.get('pot', 0)
        current_bet = position_info.get('currentBet', 0)
        my_chips = position_info.get('myChips', 1000)
        
        if best_action == "fold":
            advice_parts.append(f"→ Action: FOLD - Tight/passive opponent. Weak hand, save chips.")
        elif best_action in ["bet_small", "bet_medium", "bet_large", "all_in"]:
            if my_chips < pot * 0.1:
                advice_parts.append(f"→ Action: BET SMALL - Low on chips ({my_chips}), bet small. They fold often, steal the pot.")
            else:
                advice_parts.append(f"→ Action: {dqn_advice.upper()} - STEAL OPPORTUNITY! They fold often, bet into {pot} pot to win it.")
        elif best_action == "check_call":
            if current_bet == 0:
                advice_parts.append(f"→ Action: CHECK - Take free card. Passive player won't bet.")
            elif my_chips < current_bet:
                advice_parts.append(f"→ Action: ALL IN - Only {my_chips} chips left, must go all-in to call {current_bet}.")
            else:
                advice_parts.append(f"→ Action: CALL {current_bet} - Cheap call against passive player.")
        else:
            advice_parts.append(f"→ Action: {dqn_advice.upper() if dqn_advice else 'BET'} - Bet to steal from tight/passive player.")
            
    elif opponent_type == "Aggressive":
        pot = position_info.get('pot', 0)
        current_bet = position_info.get('currentBet', 0)
        stage = position_info.get('stage', 'preflop')
        my_chips = position_info.get('myChips', 1000)
        
        if best_action == "fold":
            advice_parts.append(f"→ Action: FOLD - Aggressive opponent bet {current_bet}. Don't call without a strong hand.")
        elif best_action in ["bet_small", "bet_medium", "bet_large", "all_in"]:
            if current_bet > 0:
                if my_chips <= current_bet:
                    advice_parts.append(f"→ Action: ALL IN - Only {my_chips} chips left, can't re-raise {current_bet}.")
                elif my_chips < current_bet * 2:
                    advice_parts.append(f"→ Action: ALL IN - Low on chips ({my_chips}), re-raise would be all-in.")
                else:
                    advice_parts.append(f"→ Action: {dqn_advice.upper()} - Re-raise aggressive opponent's {current_bet} bet. Show strength.")
            else:
                advice_parts.append(f"→ Action: {dqn_advice.upper()} - Bet into {pot} pot. Aggressive player will likely raise, so bet strong hands.")
        elif best_action == "check_call":
            if current_bet == 0:
                advice_parts.append(f"→ Action: CHECK - Let aggressive opponent bet on {stage}.")
            elif my_chips < current_bet:
                advice_parts.append(f"→ Action: ALL IN - Only {my_chips} chips left, must go all-in to call {current_bet}.")
            else:
                advice_parts.append(f"→ Action: CALL {current_bet} - Call with strong hand. Be ready to fold if they raise again.")
        else:
            advice_parts.append(f"→ Action: {dqn_advice.upper() if dqn_advice else 'FOLD'} - Against aggressive player, need strong hand.")
            
    elif opponent_type == "Calling Station":
        pot = position_info.get('pot', 0)
        current_bet = position_info.get('currentBet', 0)
        my_chips = position_info.get('myChips', 1000)
        
        if best_action == "fold":
            advice_parts.append(f"→ Action: FOLD - Calling station bet {current_bet}. If DQN says fold, hand is too weak.")
        elif best_action in ["bet_small", "bet_medium", "bet_large", "all_in"]:
            if my_chips < pot * 0.1:
                advice_parts.append(f"→ Action: BET SMALL - Low on chips ({my_chips}), bet small. They call everything, value bet.")
            else:
                advice_parts.append(f"→ Action: {dqn_advice.upper()} - VALUE BET! They call everything, so bet strong hands for value.")
        elif best_action == "check_call":
            if current_bet == 0:
                advice_parts.append(f"→ Action: CHECK - Take free card. They'll call any bet anyway.")
            elif my_chips < current_bet:
                advice_parts.append(f"→ Action: ALL IN - Only {my_chips} chips left, must go all-in to call {current_bet}.")
            else:
                advice_parts.append(f"→ Action: CALL {current_bet} - They call often, so call with decent hand.")
        else:
            advice_parts.append(f"→ Action: {dqn_advice.upper() if dqn_advice else 'BET'} - Value bet strong hands, avoid bluffing.")
            
    elif opponent_type == "Loose/Aggressive":
        # Get context for more specific advice
        pot = position_info.get('pot', 0)
        current_bet = position_info.get('currentBet', 0)
        stage = position_info.get('stage', 'preflop')
        my_chips = position_info.get('myChips', 1000)
        pot_odds = current_bet / (pot + current_bet) if (pot + current_bet) > 0 else 0
        
        if best_action == "fold":
            if current_bet > my_chips * 0.3:
                advice_parts.append(f"→ Action: FOLD - LAG opponent raised {current_bet} into {pot} pot. Too expensive with weak hand.")
            else:
                advice_parts.append(f"→ Action: FOLD - Weak hand against aggressive opponent. Save chips for better spots.")
        elif best_action in ["bet_small", "bet_medium", "bet_large", "all_in"]:
            bet_size = "small" if "small" in best_action else ("medium" if "medium" in best_action else ("large" if "large" in best_action else "all-in"))
            if current_bet > 0:
                # Check if we can actually re-raise
                if my_chips <= current_bet:
                    advice_parts.append(f"→ Action: ALL IN - Only {my_chips} chips left, can't re-raise {current_bet}. Go all-in.")
                elif my_chips < current_bet * 2:
                    advice_parts.append(f"→ Action: ALL IN - Low on chips ({my_chips}), re-raise would be all-in. Strong hand against LAG.")
                else:
                    advice_parts.append(f"→ Action: {dqn_advice.upper()} - Re-raise LAG's {current_bet} bet. You have {my_chips} chips, strong hand.")
            else:
                if my_chips < pot * 0.1:
                    advice_parts.append(f"→ Action: BET SMALL - Low on chips ({my_chips}), bet small into {pot} pot. LAG will call with wide range.")
                else:
                    advice_parts.append(f"→ Action: {dqn_advice.upper()} - Bet {bet_size} into {pot} pot. LAG will call with wide range, value bet your strong hand.")
        elif best_action == "check_call":
            if current_bet == 0:
                advice_parts.append(f"→ Action: CHECK - Take free card on {stage}. LAG will likely bet next street if you check.")
            elif my_chips < current_bet:
                advice_parts.append(f"→ Action: ALL IN - Only {my_chips} chips left, must go all-in to call {current_bet}.")
            elif pot_odds < 0.25:
                advice_parts.append(f"→ Action: CALL {current_bet} - Good pot odds ({pot_odds:.0%}). LAG bluffs often, call with decent hand.")
            else:
                advice_parts.append(f"→ Action: CALL {current_bet} - Expensive but you have a hand that can beat LAG's wide range.")
        else:
            advice_parts.append(f"→ Action: {dqn_advice.upper() if dqn_advice else 'FOLD'} - LAG opponent requires strong hands to continue.")
            
    else:  # Balanced or unknown
        pot = position_info.get('pot', 0)
        current_bet = position_info.get('currentBet', 0)
        my_chips = position_info.get('myChips', 1000)
        stage = position_info.get('stage', 'preflop')
        
        if best_action and dqn_advice:
            if best_action == "fold":
                advice_parts.append(f"→ Action: FOLD - Weak hand, save chips for better spots.")
            elif best_action in ["bet_small", "bet_medium", "bet_large", "all_in"]:
                if my_chips < current_bet * 2:
                    advice_parts.append(f"→ Action: {dqn_advice.upper()} - Low on chips ({my_chips}), but DQN recommends betting.")
                else:
                    advice_parts.append(f"→ Action: {dqn_advice.upper()} - Bet into {pot} pot on {stage}.")
            elif best_action == "check_call":
                if current_bet == 0:
                    advice_parts.append(f"→ Action: CHECK - Take free card on {stage}.")
                elif my_chips < current_bet:
                    advice_parts.append(f"→ Action: ALL IN - Only {my_chips} chips left, must go all-in to call {current_bet}.")
                else:
                    advice_parts.append(f"→ Action: CALL {current_bet} - Call with decent hand.")
            else:
                advice_parts.append(f"→ Action: {dqn_advice.upper() if dqn_advice else 'CHECK'} - DQN recommendation.")
        else:
            advice_parts.append("→ Action: CHECK - Evaluate your hand strength and pot odds.")
    
    return "\n".join(advice_parts)


def _ensure_tensor(x):
    """Helper function to ensure input is a torch.Tensor."""
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    else:
        return torch.tensor(x, dtype=torch.float32)

@app.route('/api/game/advice', methods=['POST'])
def get_advice():
    """Get AI advice for current position."""
    global current_state, game, dqn_model, psychology_model, opponent_history
    
    if current_state is None:
        return jsonify({
            'success': False,
            'error': 'No game in progress.'
        }), 400
    
    if current_state.is_terminal:
        return jsonify({
            'success': False,
            'error': 'Hand is over.'
        }), 400
    
    try:
        # Get current position info
        position_info = get_current_position_info(current_state)
        
        # Get DQN recommendation
        dqn_advice = None
        dqn_action_probs = None
        dqn_error = None
        
        if dqn_model is None:
            dqn_error = "DQN model not loaded"
            print(f"⚠ DQN model not available for advice. Global dqn_model = {dqn_model}")
            # Check if file exists but model didn't load
            if os.path.exists("poker-rl/output/models/dqn_finetuned_zproj.pt"):
                dqn_error += " (model file exists but failed to load - check backend console for errors)"
        else:
            try:
                state_vector = current_state.to_vector(1)  # Player 1 perspective
                
                # Get behavior embedding if available
                z = None
                if psychology_model is not None and opponent_history is not None:
                    bot_history = opponent_history.get_history(opponent_id=-1)
                    if opponent_history.has_history(-1):
                        z_tensor, _ = psychology_model.encode_opponent(
                            torch.from_numpy(bot_history).unsqueeze(0).float(),
                            return_supervised=False
                        )
                        # Use helper function to ensure it's a tensor
                        z_tensor = _ensure_tensor(z_tensor)
                        if z_tensor.dim() > 1:
                            z = z_tensor[0].detach().cpu().numpy()
                        else:
                            z = z_tensor.detach().cpu().numpy()
                
                # Get Q-values
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0)
                    # Only pass behavior embedding if model was trained with it
                    if z is not None and dqn_model.behavior_embedding_dim > 0:
                        z_tensor = torch.from_numpy(z).float().unsqueeze(0)
                        q_values = dqn_model(state_tensor, z_tensor)
                    else:
                        # Model doesn't support behavior embedding, or z is None
                        q_values = dqn_model(state_tensor)
                    
                    # Handle different output shapes - convert to numpy first
                    # Check if it's a tensor or numpy array
                    if isinstance(q_values, torch.Tensor):
                        print(f"DEBUG: q_values (tensor) shape: {q_values.shape}, dim: {q_values.dim()}")
                        q_values_np = q_values.cpu().numpy()
                    else:
                        print(f"DEBUG: q_values (numpy) shape: {q_values.shape}, ndim: {q_values.ndim}")
                        q_values_np = np.array(q_values)  # Ensure it's numpy
                    
                    # Handle different numpy array shapes - use indexing, not squeeze
                    if q_values_np.ndim == 2:
                        # 2D array - take first row (removes batch dimension)
                        q_values = q_values_np[0]
                    elif q_values_np.ndim == 1:
                        # Already 1D, use as is
                        q_values = q_values_np
                    else:
                        # Higher dimensional - flatten
                        q_values = q_values_np.flatten()
                    
                    # Ensure it's 1D array
                    if q_values.ndim > 1:
                        q_values = q_values.flatten()
                    
                    print(f"DEBUG: Final q_values shape: {q_values.shape}, len: {len(q_values)}")
                    
                    # Final check - should be 1D with 6 elements (one per action)
                    if len(q_values) != cfg.NUM_ACTIONS:
                        raise ValueError(f"Expected {cfg.NUM_ACTIONS} Q-values, got {len(q_values)}, shape: {q_values.shape}")
                    
                    # Convert to probabilities (softmax) - ensure it's a torch tensor
                    q_tensor = torch.from_numpy(q_values).float()
                    action_probs = torch.softmax(q_tensor, dim=0).numpy()
                    dqn_action_probs = {
                        'fold': float(action_probs[cfg.ACTION_FOLD]),
                        'check_call': float(action_probs[cfg.ACTION_CHECK_CALL]),
                        'bet_small': float(action_probs[cfg.ACTION_BET_SMALL]),
                        'bet_medium': float(action_probs[cfg.ACTION_BET_MEDIUM]),
                        'bet_large': float(action_probs[cfg.ACTION_BET_LARGE]),
                        'all_in': float(action_probs[cfg.ACTION_ALL_IN])
                    }
                    
                    # Get recommended action
                    recommended_action_idx = np.argmax(q_values)
                    action_names = ['Fold', 'Check/Call', 'Bet Small', 'Bet Medium', 'Bet Large', 'All-In']
                    dqn_advice = action_names[recommended_action_idx]
                    print(f"✓ DQN recommendation: {dqn_advice}")
            except Exception as e:
                dqn_error = str(e)
                print(f"Error getting DQN advice: {e}")
                import traceback
                traceback.print_exc()
        
        # Get psychology network judgment
        opponent_type = None
        opponent_analysis = None
        psych_error = None
        
        if psychology_model is None:
            psych_error = "Psychology model not loaded (model file not found)"
            print("⚠ Psychology model not available for advice")
        elif opponent_history is None:
            psych_error = "Opponent history manager not initialized"
            print("⚠ Opponent history not available")
        else:
            try:
                bot_history = opponent_history.get_history(opponent_id=-1)
                if not opponent_history.has_history(-1):
                    psych_error = "Insufficient opponent history (need at least 1 action)"
                    print("⚠ No opponent history yet - need bot to act first")
                else:
                    history_tensor = torch.from_numpy(bot_history).unsqueeze(0).float()
                    
                    with torch.no_grad():
                        z_tensor, supervised_output = psychology_model.encode_opponent(
                            history_tensor, return_supervised=True
                        )
                        
                        # Interpret supervised output (action frequencies)
                        if supervised_output is not None:
                            # Use helper function to ensure it's a tensor
                            supervised_output = _ensure_tensor(supervised_output)
                            
                            # Now it's guaranteed to be a tensor - safe to call .dim()
                            if supervised_output.dim() == 1:
                                supervised_output = supervised_output.unsqueeze(0)
                            # Use dim=1 if 2D, dim=0 if 1D
                            dim_to_use = 1 if supervised_output.dim() == 2 else 0
                            
                            probs = torch.softmax(supervised_output, dim=dim_to_use)
                            # Use indexing instead of squeeze to avoid errors
                            # probs is always a tensor from softmax
                            if probs.dim() > 1:
                                # Always use indexing [0] instead of squeeze
                                probs = probs[0]
                            probs = probs.cpu().numpy()
                            fold_pct = float(probs[0])
                            call_pct = float(probs[1])
                            raise_pct = float(probs[2])
                            big_bet_pct = float(probs[3])
                            
                            # Classify opponent type
                            if fold_pct > 0.4:
                                opponent_type = "Tight/Passive"
                            elif raise_pct > 0.3:
                                opponent_type = "Aggressive"
                            elif call_pct > 0.5:
                                opponent_type = "Calling Station"
                            elif big_bet_pct > 0.2:
                                opponent_type = "Loose/Aggressive"
                            else:
                                opponent_type = "Balanced"
                            
                            opponent_analysis = {
                                'type': opponent_type,
                                'fold_frequency': fold_pct,
                                'call_frequency': call_pct,
                                'raise_frequency': raise_pct,
                                'big_bet_frequency': big_bet_pct
                            }
                            print(f"✓ Psychology analysis: {opponent_type}")
            except Exception as e:
                psych_error = str(e)
                print(f"Error getting psychology analysis: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate combined advice
        advice_text = generate_combined_advice(
            position_info, dqn_advice, dqn_action_probs, opponent_type, opponent_analysis,
            dqn_error, psych_error
        )
        
        return jsonify({
            'success': True,
            'positionInfo': position_info,
            'dqnAdvice': dqn_advice,
            'dqnActionProbs': dqn_action_probs,
            'dqnError': dqn_error,
            'opponentType': opponent_type,
            'opponentAnalysis': opponent_analysis,
            'psychError': psych_error,
            'advice': advice_text
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error generating advice: {str(e)}'
        }), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Starting Alpha Poker Zero backend server...")
    print("\nAPI endpoints:")
    print("  POST /api/game/start - Start a new hand")
    print("  GET  /api/game/state - Get current game state")
    print("  POST /api/game/action - Make an action (user)")
    print("  POST /api/game/bot-action - Get bot's action")
    print("  GET  /api/game/valid-actions - Get valid actions")
    print("  POST /api/game/advice - Get AI advice for current position")
    print("\n" + "=" * 60)
    app.run(debug=True, port=5001, host='0.0.0.0')

