"""Poker game adapter for MCTS self-play."""

import numpy as np
from typing import List, Tuple, Optional
from card import Card, Deck, Rank, Suit
from hand_evaluator import HandEvaluator
from config import PokerConfig as cfg


class PokerGameState:
    """Represents the state of a poker game for MCTS."""

    def __init__(self):
        self.deck = Deck()
        # Player 1 perspective
        self.hole_cards_p1: List[Card] = []
        self.hole_cards_p2: List[Card] = []
        self.community_cards: List[Card] = []

        # Game state
        self.pot = cfg.BIG_BLIND + cfg.SMALL_BLIND
        self.player1_stack = cfg.STARTING_CHIPS - cfg.SMALL_BLIND
        self.player2_stack = cfg.STARTING_CHIPS - cfg.BIG_BLIND
        self.current_bet = cfg.BIG_BLIND
        self.player1_bet = cfg.SMALL_BLIND
        self.player2_bet = cfg.BIG_BLIND

        # Stage: 0=preflop, 1=flop, 2=turn, 3=river
        self.stage = 0

        # Game status
        self.is_terminal = False
        self.winner = None  # 1, -1, or 0 for draw

        # Player to act (1 or -1)
        self.current_player = 1

        # Betting round tracking
        self.last_action = None
        self.betting_round_complete = False  # Track if betting round is complete
        self.players_acted_this_round = set()  # Track which players have acted this betting round

    def copy(self):
        """Create a deep copy of the game state."""
        new_state = PokerGameState.__new__(PokerGameState)
        new_state.deck = Deck()
        new_state.deck.cards = self.deck.cards.copy()

        new_state.hole_cards_p1 = self.hole_cards_p1.copy()
        new_state.hole_cards_p2 = self.hole_cards_p2.copy()
        new_state.community_cards = self.community_cards.copy()

        new_state.pot = self.pot
        new_state.player1_stack = self.player1_stack
        new_state.player2_stack = self.player2_stack
        new_state.current_bet = self.current_bet
        new_state.player1_bet = self.player1_bet
        new_state.player2_bet = self.player2_bet

        new_state.stage = self.stage
        new_state.is_terminal = self.is_terminal
        new_state.winner = self.winner
        new_state.current_player = self.current_player
        new_state.last_action = self.last_action
        new_state.betting_round_complete = self.betting_round_complete
        new_state.players_acted_this_round = self.players_acted_this_round.copy()

        return new_state

    def to_vector(self, player: int) -> np.ndarray:
        """
        Convert game state to feature vector from given player's perspective.

        Returns normalized feature vector.
        """
        features = []

        # Encode hole cards (one-hot encoding for each card)
        # We only know our own hole cards
        hole_cards = self.hole_cards_p1 if player == 1 else self.hole_cards_p2
        card_vector = np.zeros(52)
        for card in hole_cards:
            card_idx = (card.rank - 2) * 4 + card.suit
            card_vector[card_idx] = 1
        features.extend(card_vector)

        # Encode community cards (one-hot for up to 5 cards)
        for _ in range(5):
            card_vector = np.zeros(52)
            if _ < len(self.community_cards):
                card = self.community_cards[_]
                card_idx = (card.rank - 2) * 4 + card.suit
                card_vector[card_idx] = 1
            features.extend(card_vector)

        # Normalize stack sizes and pot (divide by starting chips)
        my_stack = self.player1_stack if player == 1 else self.player2_stack
        opp_stack = self.player2_stack if player == 1 else self.player1_stack
        features.append(self.pot / cfg.STARTING_CHIPS)
        features.append(my_stack / cfg.STARTING_CHIPS)
        features.append(opp_stack / cfg.STARTING_CHIPS)

        # Current bet to call (normalized)
        my_bet = self.player1_bet if player == 1 else self.player2_bet
        to_call = self.current_bet - my_bet
        features.append(to_call / cfg.STARTING_CHIPS)

        # Game stage (one-hot)
        stage_vector = [0, 0, 0, 0]
        stage_vector[self.stage] = 1
        features.extend(stage_vector)

        return np.array(features, dtype=np.float32)


class PokerGame:
    """Poker game interface for MCTS."""

    def __init__(self):
        self.state = PokerGameState()

    def init_new_hand(self) -> PokerGameState:
        """Initialize a new hand."""
        state = PokerGameState()

        # Deal hole cards
        state.hole_cards_p1 = state.deck.deal(2)
        state.hole_cards_p2 = state.deck.deal(2)

        # Player 1 acts first preflop (big blind acts last preflop)
        state.current_player = 1
        state.betting_round_complete = False  # New betting round
        state.players_acted_this_round = set()  # No one has acted yet in preflop betting

        return state

    def get_valid_moves(self, state: PokerGameState, player: int) -> np.ndarray:
        """
        Get valid actions for current player.

        Returns binary mask of valid actions.
        """
        valid_actions = np.zeros(cfg.NUM_ACTIONS)

        my_bet = state.player1_bet if player == 1 else state.player2_bet
        my_stack = state.player1_stack if player == 1 else state.player2_stack
        to_call = state.current_bet - my_bet

        # Can always fold (unless we can check)
        if to_call > 0:
            valid_actions[cfg.ACTION_FOLD] = 1

        # Can check if no bet to call, otherwise can call
        valid_actions[cfg.ACTION_CHECK_CALL] = 1

        # Can bet/raise if we have chips
        if my_stack > to_call:
            remaining_after_call = my_stack - to_call

            # Small bet (0.5x pot)
            if remaining_after_call >= state.pot * 0.5:
                valid_actions[cfg.ACTION_BET_SMALL] = 1

            # Medium bet (1x pot)
            if remaining_after_call >= state.pot:
                valid_actions[cfg.ACTION_BET_MEDIUM] = 1

            # Large bet (2x pot)
            if remaining_after_call >= state.pot * 2:
                valid_actions[cfg.ACTION_BET_LARGE] = 1

            # All-in (always valid if we have chips)
            valid_actions[cfg.ACTION_ALL_IN] = 1

        return valid_actions

    def apply_action(self, state: PokerGameState, action: int, player: int) -> PokerGameState:
        """
        Apply an action to the state and return new state.

        Note: This mutates the state for efficiency.
        """
        my_bet = state.player1_bet if player == 1 else state.player2_bet
        my_stack = state.player1_stack if player == 1 else state.player2_stack
        to_call = state.current_bet - my_bet

        if action == cfg.ACTION_FOLD:
            # Player folds
            state.is_terminal = True
            state.winner = -player  # Opponent wins
            state.last_action = 'fold'

        elif action == cfg.ACTION_CHECK_CALL:
            # Check or call
            if to_call > 0:
                # Call
                amount = min(to_call, my_stack)
                if player == 1:
                    state.player1_stack -= amount
                    state.player1_bet += amount
                else:
                    state.player2_stack -= amount
                    state.player2_bet += amount
                state.pot += amount
                state.last_action = 'call'
            else:
                # Check
                state.last_action = 'check'
            
            # Mark this player as having acted
            state.players_acted_this_round.add(player)

        else:
            # Bet or raise
            bet_size = 0
            if action == cfg.ACTION_BET_SMALL:
                bet_size = int(state.pot * 0.5)
            elif action == cfg.ACTION_BET_MEDIUM:
                bet_size = state.pot
            elif action == cfg.ACTION_BET_LARGE:
                bet_size = int(state.pot * 2)
            elif action == cfg.ACTION_ALL_IN:
                bet_size = my_stack - to_call

            # Apply the bet/raise
            total_to_add = to_call + bet_size
            actual_amount = min(total_to_add, my_stack)

            if player == 1:
                state.player1_stack -= actual_amount
                state.player1_bet += actual_amount
            else:
                state.player2_stack -= actual_amount
                state.player2_bet += actual_amount

            state.pot += actual_amount
            state.current_bet = max(state.player1_bet, state.player2_bet)
            state.last_action = 'raise'
            state.betting_round_complete = False  # Raise starts new betting round
            state.players_acted_this_round = {player}  # Reset - only this player has acted after raise

        # Switch player
        state.current_player = -player
        
        # Check if betting round is complete AFTER switching players
        # Only advance if:
        # 1. Both players have acted in this betting round
        # 2. Both players' bets match (betting round complete)
        # 3. Last action was check/call (not a raise)
        # 4. We haven't already advanced this betting round
        if (not state.betting_round_complete and
            len(state.players_acted_this_round) == 2 and  # Both players have acted
            state.player1_bet == state.player2_bet and 
            state.last_action in ['check', 'call']):
            # Mark betting round as complete to prevent double advancement
            state.betting_round_complete = True
            # Both players have acted and bets match - advance to next stage
            self._advance_stage(state)

        return state

    def _advance_stage(self, state: PokerGameState):
        """Advance to next stage of the hand."""
        # Reset bets for new round
        state.player1_bet = 0
        state.player2_bet = 0
        state.current_bet = 0
        state.betting_round_complete = False  # Reset for new betting round
        state.players_acted_this_round = set()  # Reset for new betting round

        if state.stage == 0:
            # Deal flop
            state.deck.deal(1)  # Burn card
            state.community_cards.extend(state.deck.deal(3))
            state.stage = 1
        elif state.stage == 1:
            # Deal turn
            state.deck.deal(1)  # Burn card
            state.community_cards.extend(state.deck.deal(1))
            state.stage = 2
        elif state.stage == 2:
            # Deal river
            state.deck.deal(1)  # Burn card
            state.community_cards.extend(state.deck.deal(1))
            state.stage = 3
        elif state.stage == 3:
            # Showdown
            self._evaluate_showdown(state)

    def _evaluate_showdown(self, state: PokerGameState):
        """Evaluate showdown and determine winner."""
        # Evaluate both hands
        p1_cards = state.hole_cards_p1 + state.community_cards
        p2_cards = state.hole_cards_p2 + state.community_cards

        p1_rank, p1_tiebreaker, _ = HandEvaluator.best_hand(p1_cards)
        p2_rank, p2_tiebreaker, _ = HandEvaluator.best_hand(p2_cards)

        comparison = HandEvaluator._compare_hands(
            (p1_rank, p1_tiebreaker),
            (p2_rank, p2_tiebreaker)
        )

        state.is_terminal = True
        if comparison > 0:
            state.winner = 1
        elif comparison < 0:
            state.winner = -1
        else:
            state.winner = 0

    def is_terminal(self, state: PokerGameState) -> bool:
        """Check if game is over."""
        return state.is_terminal

    def get_reward(self, state: PokerGameState, player: int) -> Optional[float]:
        """
        Get reward for the given player.

        Returns normalized reward (chips won/lost divided by starting stack).
        Uses symmetric rewards: win = net chips gained, lose = chips lost.
        Simple reward function focused on winning games, not just chips.
        """
        if not state.is_terminal:
            return None

        my_contribution = cfg.STARTING_CHIPS - (
            state.player1_stack if player == 1 else state.player2_stack
        )
        
        if state.winner == player:
            # Won the pot - reward is net chips gained
            net_gain = state.pot - my_contribution
            return net_gain / cfg.STARTING_CHIPS
        elif state.winner == -player:
            # Lost - penalty is chips contributed
            return -my_contribution / cfg.STARTING_CHIPS
        else:
            # Draw - no net change
            return 0.0

    def get_canonical_state(self, state: PokerGameState, player: int) -> np.ndarray:
        """
        Get canonical form of state (always from current player's perspective).
        """
        return state.to_vector(player)
