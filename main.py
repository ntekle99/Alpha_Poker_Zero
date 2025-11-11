"""Main entry point for Texas Hold'em poker game."""

from player import HumanPlayer, RandomPlayer
from texas_holdem import TexasHoldem


def main():
    """Run the poker game."""
    print("=" * 60)
    print("TEXAS HOLD'EM POKER")
    print("=" * 60)

    # Get player name
    player_name = input("\nEnter your name: ").strip()
    if not player_name:
        player_name = "Player"

    # Set starting chips
    try:
        starting_chips = int(input("Enter starting chips (default 1000): ") or "1000")
    except ValueError:
        starting_chips = 1000

    # Set blinds
    try:
        small_blind = int(input("Enter small blind (default 5): ") or "5")
        big_blind = int(input("Enter big blind (default 10): ") or "10")
    except ValueError:
        small_blind = 5
        big_blind = 10

    # Create players
    human = HumanPlayer(player_name, starting_chips)
    ai = RandomPlayer("Random AI", starting_chips)

    players = [human, ai]

    # Create game
    game = TexasHoldem(players, small_blind, big_blind)

    print(f"\n{player_name} vs Random AI")
    print(f"Starting chips: ${starting_chips}")
    print(f"Blinds: ${small_blind}/${big_blind}")
    print("\nLet's play!\n")

    # Game loop
    hand_number = 1
    while game.can_continue():
        print(f"\n{'#' * 60}")
        print(f"HAND #{hand_number}")
        print(f"{'#' * 60}")

        try:
            game.play_hand()
            hand_number += 1

            # Ask if player wants to continue
            if game.can_continue():
                continue_game = input("\nPlay another hand? (y/n): ").lower().strip()
                if continue_game != 'y':
                    break
        except KeyboardInterrupt:
            print("\n\nGame interrupted by user.")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")
            import traceback
            traceback.print_exc()
            break

    # Game over
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)
    print("\nFinal chip counts:")
    for player in players:
        print(f"  {player.name}: ${player.chips}")

    # Determine winner
    winner = max(players, key=lambda p: p.chips)
    if winner.chips == min(p.chips for p in players):
        print("\nIt's a tie!")
    else:
        print(f"\n{winner.name} wins!")

    print("\nThanks for playing!")


if __name__ == "__main__":
    main()
