"""Player classes for poker game."""

from typing import List, Optional
from card import Card
import random


class Player:
    """Base class for a poker player."""

    def __init__(self, name: str, chips: int = 1000):
        """
        Initialize a player.

        Args:
            name: Player's name
            chips: Starting chip count
        """
        self.name = name
        self.chips = chips
        self.hole_cards: List[Card] = []
        self.current_bet = 0
        self.folded = False
        self.all_in = False

    def receive_cards(self, cards: List[Card]):
        """Receive hole cards."""
        self.hole_cards = cards

    def bet(self, amount: int) -> int:
        """
        Make a bet.

        Args:
            amount: Amount to bet

        Returns:
            Actual amount bet (may be less if all-in)
        """
        if amount >= self.chips:
            # All-in
            actual_bet = self.chips
            self.chips = 0
            self.all_in = True
        else:
            actual_bet = amount
            self.chips -= amount

        self.current_bet += actual_bet
        return actual_bet

    def reset_for_new_hand(self):
        """Reset player state for a new hand."""
        self.hole_cards = []
        self.current_bet = 0
        self.folded = False
        self.all_in = False

    def reset_current_bet(self):
        """Reset current bet for a new betting round."""
        self.current_bet = 0

    def fold(self):
        """Fold the hand."""
        self.folded = True

    def get_action(self, current_bet: int, min_raise: int) -> tuple:
        """
        Get the player's action. To be overridden by subclasses.

        Args:
            current_bet: Current bet to match
            min_raise: Minimum raise amount

        Returns:
            Tuple of (action, amount) where action is 'fold', 'call', 'raise', or 'check'
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """String representation of player."""
        return f"{self.name} (${self.chips})"


class HumanPlayer(Player):
    """Human player that gets input from console."""

    def get_action(self, current_bet: int, min_raise: int) -> tuple:
        """
        Get action from human player via console input.

        Args:
            current_bet: Current bet to match
            min_raise: Minimum raise amount

        Returns:
            Tuple of (action, amount)
        """
        amount_to_call = current_bet - self.current_bet

        print(f"\n{self.name}'s turn:")
        print(f"Your cards: {self.hole_cards}")
        print(f"Your chips: ${self.chips}")
        print(f"Current bet: ${self.current_bet}")

        while True:
            if amount_to_call == 0:
                action = input("Action (check/raise/fold): ").lower().strip()
                if action == 'check':
                    return ('check', 0)
                elif action == 'fold':
                    return ('fold', 0)
                elif action == 'raise':
                    try:
                        raise_amount = int(input(f"Raise amount (min ${min_raise}): $"))
                        if raise_amount < min_raise:
                            print(f"Minimum raise is ${min_raise}")
                            continue
                        if raise_amount > self.chips:
                            print(f"You only have ${self.chips}")
                            continue
                        return ('raise', raise_amount)
                    except ValueError:
                        print("Invalid amount")
                        continue
                else:
                    print("Invalid action")
            else:
                print(f"Amount to call: ${amount_to_call}")
                action = input("Action (call/raise/fold): ").lower().strip()
                if action == 'call':
                    return ('call', amount_to_call)
                elif action == 'fold':
                    return ('fold', 0)
                elif action == 'raise':
                    try:
                        raise_amount = int(input(f"Raise amount (min ${min_raise}): $"))
                        if raise_amount < min_raise:
                            print(f"Minimum raise is ${min_raise}")
                            continue
                        if raise_amount + amount_to_call > self.chips:
                            print(f"You only have ${self.chips}")
                            continue
                        return ('raise', raise_amount)
                    except ValueError:
                        print("Invalid amount")
                        continue
                else:
                    print("Invalid action")


class RandomPlayer(Player):
    """AI player that makes random decisions."""

    def get_action(self, current_bet: int, min_raise: int) -> tuple:
        """
        Get random action from AI.

        Args:
            current_bet: Current bet to match
            min_raise: Minimum raise amount

        Returns:
            Tuple of (action, amount)
        """
        amount_to_call = current_bet - self.current_bet

        if amount_to_call == 0:
            # Can check or raise
            action = random.choice(['check', 'check', 'raise', 'fold'])
            if action == 'check':
                return ('check', 0)
            elif action == 'fold':
                return ('fold', 0)
            else:  # raise
                if self.chips >= min_raise:
                    # Random raise between min_raise and 3x min_raise
                    max_raise = min(min_raise * 3, self.chips)
                    raise_amount = random.randint(min_raise, max_raise)
                    return ('raise', raise_amount)
                else:
                    return ('check', 0)
        else:
            # Need to call, raise, or fold
            if amount_to_call >= self.chips:
                # Must go all-in or fold
                action = random.choice(['call', 'fold'])
                if action == 'call':
                    return ('call', amount_to_call)
                else:
                    return ('fold', 0)
            else:
                action = random.choice(['call', 'call', 'raise', 'fold'])
                if action == 'call':
                    return ('call', amount_to_call)
                elif action == 'fold':
                    return ('fold', 0)
                else:  # raise
                    if self.chips >= amount_to_call + min_raise:
                        max_raise = min(min_raise * 3, self.chips - amount_to_call)
                        raise_amount = random.randint(min_raise, max_raise)
                        return ('raise', raise_amount)
                    else:
                        return ('call', amount_to_call)
