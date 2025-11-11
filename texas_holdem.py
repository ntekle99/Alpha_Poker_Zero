"""Texas Hold'em poker game implementation."""

from typing import List, Optional
from card import Deck, Card
from player import Player
from hand_evaluator import HandEvaluator
import time


class TexasHoldem:
    """Texas Hold'em poker game."""

    def __init__(self, players: List[Player], small_blind: int = 5, big_blind: int = 10):
        """
        Initialize the game.

        Args:
            players: List of players (should be 2 for heads-up)
            small_blind: Small blind amount
            big_blind: Big blind amount
        """
        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.dealer_position = 0
        self.deck = Deck()
        self.community_cards: List[Card] = []
        self.pot = 0
        self.current_bet = 0
        self.min_raise = big_blind

    def play_hand(self):
        """Play a single hand of Texas Hold'em."""
        print("\n" + "=" * 60)
        print("NEW HAND")
        print("=" * 60)

        # Reset for new hand
        self._reset_hand()

        # Post blinds
        self._post_blinds()

        # Deal hole cards
        self._deal_hole_cards()

        # Pre-flop betting
        print("\n--- PRE-FLOP ---")
        if not self._betting_round(first_to_act=self._get_next_active_player(self.dealer_position)):
            self._end_hand()
            return

        # Flop
        self._deal_flop()
        print("\n--- FLOP ---")
        print(f"Community cards: {self.community_cards}")
        if not self._betting_round(first_to_act=self._get_next_active_player(self.dealer_position)):
            self._end_hand()
            return

        # Turn
        self._deal_turn()
        print("\n--- TURN ---")
        print(f"Community cards: {self.community_cards}")
        if not self._betting_round(first_to_act=self._get_next_active_player(self.dealer_position)):
            self._end_hand()
            return

        # River
        self._deal_river()
        print("\n--- RIVER ---")
        print(f"Community cards: {self.community_cards}")
        if not self._betting_round(first_to_act=self._get_next_active_player(self.dealer_position)):
            self._end_hand()
            return

        # Showdown
        self._showdown()
        self._end_hand()

    def _reset_hand(self):
        """Reset game state for a new hand."""
        self.deck.reset()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.min_raise = self.big_blind

        for player in self.players:
            player.reset_for_new_hand()

    def _post_blinds(self):
        """Post small and big blinds."""
        # In heads-up, dealer posts small blind
        small_blind_player = self.players[self.dealer_position]
        big_blind_player = self.players[(self.dealer_position + 1) % len(self.players)]

        sb_amount = small_blind_player.bet(self.small_blind)
        bb_amount = big_blind_player.bet(self.big_blind)

        self.pot += sb_amount + bb_amount
        self.current_bet = self.big_blind

        print(f"{small_blind_player.name} posts small blind ${sb_amount}")
        print(f"{big_blind_player.name} posts big blind ${bb_amount}")

    def _deal_hole_cards(self):
        """Deal 2 hole cards to each player."""
        for player in self.players:
            cards = self.deck.deal(2)
            player.receive_cards(cards)
            print(f"{player.name} receives hole cards")

    def _deal_flop(self):
        """Deal the flop (3 community cards)."""
        self.deck.deal(1)  # Burn card
        self.community_cards.extend(self.deck.deal(3))

    def _deal_turn(self):
        """Deal the turn (1 community card)."""
        self.deck.deal(1)  # Burn card
        self.community_cards.extend(self.deck.deal(1))

    def _deal_river(self):
        """Deal the river (1 community card)."""
        self.deck.deal(1)  # Burn card
        self.community_cards.extend(self.deck.deal(1))

    def _betting_round(self, first_to_act: int) -> bool:
        """
        Conduct a betting round.

        Args:
            first_to_act: Index of first player to act

        Returns:
            True if hand should continue, False if only one player remains
        """
        # Reset current bets for new round
        for player in self.players:
            player.reset_current_bet()

        self.current_bet = 0
        last_raiser = None
        current_player_idx = first_to_act

        # Continue until all active players have acted and bets are equal
        players_acted = set()

        while True:
            player = self.players[current_player_idx]

            # Skip if player folded or is all-in
            if player.folded or player.all_in:
                current_player_idx = (current_player_idx + 1) % len(self.players)
                continue

            # Get player action
            amount_to_call = self.current_bet - player.current_bet
            action, amount = player.get_action(self.current_bet, self.min_raise)

            print(f"\n{player.name} {action}s", end="")

            if action == 'fold':
                player.fold()
                print()
            elif action == 'check':
                print()
                players_acted.add(current_player_idx)
            elif action == 'call':
                actual_amount = player.bet(amount)
                self.pot += actual_amount
                print(f" ${actual_amount}")
                players_acted.add(current_player_idx)
            elif action == 'raise':
                # First match current bet, then raise
                call_amount = amount_to_call
                total_amount = call_amount + amount
                actual_amount = player.bet(total_amount)
                self.pot += actual_amount

                self.current_bet = player.current_bet
                self.min_raise = amount
                last_raiser = current_player_idx
                players_acted = {current_player_idx}  # Reset, everyone needs to act again
                print(f" ${amount} (total bet: ${player.current_bet})")

            # Check if only one player remains
            active_players = [p for p in self.players if not p.folded]
            if len(active_players) == 1:
                return False

            # Check if betting round is complete
            all_in_or_folded = all(p.folded or p.all_in for p in self.players)
            if all_in_or_folded:
                break

            active_not_all_in = [i for i, p in enumerate(self.players) if not p.folded and not p.all_in]
            if len(active_not_all_in) > 0:
                # Check if all active players have acted and bets are equal
                all_acted = all(i in players_acted for i in active_not_all_in)
                bets_equal = all(
                    p.current_bet == self.current_bet or p.folded or p.all_in
                    for p in self.players
                )

                if all_acted and bets_equal:
                    break

            current_player_idx = (current_player_idx + 1) % len(self.players)

        return True

    def _showdown(self):
        """Determine the winner at showdown."""
        print("\n--- SHOWDOWN ---")

        active_players = [p for p in self.players if not p.folded]

        best_hand_rank = None
        best_tiebreaker = None
        winners = []

        for player in active_players:
            all_cards = player.hole_cards + self.community_cards
            hand_rank, tiebreaker, best_5 = HandEvaluator.best_hand(all_cards)
            hand_name = HandEvaluator.get_hand_name(hand_rank)

            print(f"\n{player.name} shows: {player.hole_cards}")
            print(f"  Best hand: {hand_name} - {best_5}")

            if best_hand_rank is None:
                best_hand_rank = hand_rank
                best_tiebreaker = tiebreaker
                winners = [player]
            else:
                comparison = HandEvaluator._compare_hands(
                    (hand_rank, tiebreaker),
                    (best_hand_rank, best_tiebreaker)
                )

                if comparison > 0:
                    best_hand_rank = hand_rank
                    best_tiebreaker = tiebreaker
                    winners = [player]
                elif comparison == 0:
                    winners.append(player)

        # Distribute pot
        winnings_per_player = self.pot // len(winners)
        print(f"\n{'=' * 60}")
        if len(winners) == 1:
            print(f"{winners[0].name} wins ${self.pot}!")
            winners[0].chips += self.pot
        else:
            winner_names = ", ".join([w.name for w in winners])
            print(f"Tie! {winner_names} split the pot (${winnings_per_player} each)")
            for winner in winners:
                winner.chips += winnings_per_player

    def _end_hand(self):
        """End the hand and show chip counts."""
        # If only one player remains
        active_players = [p for p in self.players if not p.folded]
        if len(active_players) == 1:
            winner = active_players[0]
            print(f"\n{winner.name} wins ${self.pot} (others folded)")
            winner.chips += self.pot

        print(f"\n{'=' * 60}")
        print("CHIP COUNTS:")
        for player in self.players:
            print(f"  {player.name}: ${player.chips}")
        print("=" * 60)

        # Move dealer button
        self.dealer_position = (self.dealer_position + 1) % len(self.players)

        # Check if any player is out of chips
        for player in self.players:
            if player.chips <= 0:
                print(f"\n{player.name} is out of chips!")

    def _get_next_active_player(self, from_position: int) -> int:
        """Get the next active player from a position."""
        idx = (from_position + 1) % len(self.players)
        while self.players[idx].folded:
            idx = (idx + 1) % len(self.players)
        return idx

    def can_continue(self) -> bool:
        """Check if the game can continue."""
        return all(player.chips > 0 for player in self.players)
