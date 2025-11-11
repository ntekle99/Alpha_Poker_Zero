"""Card representation for poker game."""

from enum import IntEnum
from typing import List
import random


class Suit(IntEnum):
    """Card suits."""
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    """Card ranks (2-14, where 14 is Ace)."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


class Card:
    """Represents a single playing card."""

    SUIT_SYMBOLS = {
        Suit.CLUBS: '♣',
        Suit.DIAMONDS: '♦',
        Suit.HEARTS: '♥',
        Suit.SPADES: '♠'
    }

    RANK_NAMES = {
        Rank.TWO: '2', Rank.THREE: '3', Rank.FOUR: '4', Rank.FIVE: '5',
        Rank.SIX: '6', Rank.SEVEN: '7', Rank.EIGHT: '8', Rank.NINE: '9',
        Rank.TEN: '10', Rank.JACK: 'J', Rank.QUEEN: 'Q',
        Rank.KING: 'K', Rank.ACE: 'A'
    }

    def __init__(self, rank: Rank, suit: Suit):
        """Initialize a card with rank and suit."""
        self.rank = rank
        self.suit = suit

    def __str__(self) -> str:
        """String representation of the card."""
        return f"{self.RANK_NAMES[self.rank]}{self.SUIT_SYMBOLS[self.suit]}"

    def __repr__(self) -> str:
        """Repr of the card."""
        return self.__str__()

    def __eq__(self, other) -> bool:
        """Check equality based on rank and suit."""
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        """Hash for using cards in sets/dicts."""
        return hash((self.rank, self.suit))


class Deck:
    """Represents a deck of playing cards."""

    def __init__(self):
        """Initialize a standard 52-card deck."""
        self.cards: List[Card] = []
        self.reset()

    def reset(self):
        """Reset the deck to a full 52-card deck and shuffle."""
        self.cards = [Card(rank, suit) for suit in Suit for rank in Rank]
        self.shuffle()

    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)

    def deal(self, num_cards: int = 1) -> List[Card]:
        """Deal a specified number of cards from the deck."""
        if num_cards > len(self.cards):
            raise ValueError(f"Not enough cards in deck. Requested {num_cards}, available {len(self.cards)}")

        dealt_cards = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt_cards

    def __len__(self) -> int:
        """Return the number of cards left in the deck."""
        return len(self.cards)
