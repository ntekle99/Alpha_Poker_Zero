"""Hand evaluation for Texas Hold'em poker."""

from enum import IntEnum
from typing import List, Tuple
from collections import Counter
from itertools import combinations
from card import Card, Rank


class HandRank(IntEnum):
    """Poker hand rankings."""
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


class HandEvaluator:
    """Evaluates poker hands."""

    HAND_NAMES = {
        HandRank.HIGH_CARD: "High Card",
        HandRank.PAIR: "Pair",
        HandRank.TWO_PAIR: "Two Pair",
        HandRank.THREE_OF_A_KIND: "Three of a Kind",
        HandRank.STRAIGHT: "Straight",
        HandRank.FLUSH: "Flush",
        HandRank.FULL_HOUSE: "Full House",
        HandRank.FOUR_OF_A_KIND: "Four of a Kind",
        HandRank.STRAIGHT_FLUSH: "Straight Flush",
        HandRank.ROYAL_FLUSH: "Royal Flush"
    }

    @staticmethod
    def evaluate_hand(cards: List[Card]) -> Tuple[HandRank, List[int]]:
        """
        Evaluate a poker hand and return its rank and tiebreaker values.

        Args:
            cards: List of 5 cards to evaluate

        Returns:
            Tuple of (hand_rank, tiebreaker_values)
        """
        ranks = sorted([card.rank for card in cards], reverse=True)
        suits = [card.suit for card in cards]
        rank_counts = Counter(ranks)

        is_flush = len(set(suits)) == 1
        is_straight = HandEvaluator._is_straight(ranks)

        # Check for royal flush
        if is_flush and is_straight and ranks[0] == Rank.ACE:
            return (HandRank.ROYAL_FLUSH, ranks)

        # Check for straight flush
        if is_flush and is_straight:
            return (HandRank.STRAIGHT_FLUSH, ranks)

        # Check for four of a kind
        if 4 in rank_counts.values():
            quad_rank = [rank for rank, count in rank_counts.items() if count == 4][0]
            kicker = [rank for rank in ranks if rank != quad_rank][0]
            return (HandRank.FOUR_OF_A_KIND, [quad_rank, kicker])

        # Check for full house
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trip_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            return (HandRank.FULL_HOUSE, [trip_rank, pair_rank])

        # Check for flush
        if is_flush:
            return (HandRank.FLUSH, ranks)

        # Check for straight
        if is_straight:
            return (HandRank.STRAIGHT, ranks)

        # Check for three of a kind
        if 3 in rank_counts.values():
            trip_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            kickers = sorted([rank for rank in ranks if rank != trip_rank], reverse=True)
            return (HandRank.THREE_OF_A_KIND, [trip_rank] + kickers)

        # Check for two pair
        pairs = [rank for rank, count in rank_counts.items() if count == 2]
        if len(pairs) == 2:
            pairs = sorted(pairs, reverse=True)
            kicker = [rank for rank in ranks if rank not in pairs][0]
            return (HandRank.TWO_PAIR, pairs + [kicker])

        # Check for one pair
        if 2 in rank_counts.values():
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            kickers = sorted([rank for rank in ranks if rank != pair_rank], reverse=True)
            return (HandRank.PAIR, [pair_rank] + kickers)

        # High card
        return (HandRank.HIGH_CARD, ranks)

    @staticmethod
    def _is_straight(ranks: List[int]) -> bool:
        """Check if ranks form a straight."""
        # Check normal straight
        if ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5:
            return True

        # Check for A-2-3-4-5 (wheel)
        if ranks == [14, 5, 4, 3, 2]:
            return True

        return False

    @staticmethod
    def best_hand(all_cards: List[Card]) -> Tuple[HandRank, List[int], List[Card]]:
        """
        Find the best 5-card hand from 7 cards.

        Args:
            all_cards: List of 7 cards (2 hole + 5 community)

        Returns:
            Tuple of (hand_rank, tiebreaker_values, best_5_cards)
        """
        best_rank = None
        best_tiebreaker = None
        best_5_cards = None

        # Try all combinations of 5 cards
        for five_cards in combinations(all_cards, 5):
            rank, tiebreaker = HandEvaluator.evaluate_hand(list(five_cards))

            # Compare with current best
            if best_rank is None or HandEvaluator._compare_hands(
                (rank, tiebreaker), (best_rank, best_tiebreaker)
            ) > 0:
                best_rank = rank
                best_tiebreaker = tiebreaker
                best_5_cards = list(five_cards)

        return (best_rank, best_tiebreaker, best_5_cards)

    @staticmethod
    def _compare_hands(hand1: Tuple[HandRank, List[int]],
                      hand2: Tuple[HandRank, List[int]]) -> int:
        """
        Compare two hands.

        Returns:
            1 if hand1 > hand2
            -1 if hand1 < hand2
            0 if equal
        """
        rank1, tiebreaker1 = hand1
        rank2, tiebreaker2 = hand2

        if rank1 > rank2:
            return 1
        elif rank1 < rank2:
            return -1
        else:
            # Compare tiebreakers
            for t1, t2 in zip(tiebreaker1, tiebreaker2):
                if t1 > t2:
                    return 1
                elif t1 < t2:
                    return -1
            return 0

    @staticmethod
    def get_hand_name(hand_rank: HandRank) -> str:
        """Get the name of a hand rank."""
        return HandEvaluator.HAND_NAMES[hand_rank]
