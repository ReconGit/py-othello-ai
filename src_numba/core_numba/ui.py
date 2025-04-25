from .othello import Othello, Cell, State

ANSI_RESET = "\033[0m"
ANSI_BLACK = "\033[30m"
ANSI_WHITE = "\033[37m"
ANSI_YELLOW = "\033[33m"


def print_board(board: list[list[Cell]]) -> None:
    print("   A B C D E F G H")
    for y in range(8):
        print(y + 1, end=" |")
        for x in range(8):
            if board[y][x] == Cell.EMPTY:
                print(" ", end="|")
            elif board[y][x] == Cell.BLACK:
                print(f"{ANSI_BLACK}●{ANSI_RESET}", end="|")
            elif board[y][x] == Cell.WHITE:
                print(f"{ANSI_WHITE}●{ANSI_RESET}", end="|")
            elif board[y][x] == Cell.VALID:
                print(f"{ANSI_YELLOW}*{ANSI_RESET}", end="|")
        print()


def user_move() -> tuple:
    """Prompt the user to enter a move, and return the move as a tuple of coordinates."""
    user_input = input().strip().upper()
    # check if user wants to quit
    if user_input == "Q":
        print("\nGame quit.")
        exit(0)
    # check if input is valid
    if len(user_input) != 2 or not user_input[1].isdigit():
        raise IndexError("Invalid input.")
    # convert user input to coordinates
    x, y = ord(user_input[0]) - ord("A"), int(user_input[1]) - 1
    if not (0 <= x <= 7 and 0 <= y <= 7):
        raise IndexError("Position out of bounds.")

    return x, y


def print_score(game: Othello) -> None:
    """Print the current score."""
    print(f"{ANSI_BLACK}Black: {game.black_score}{ANSI_RESET} | White: {game.white_score}")


def print_state(game: Othello) -> None:
    """Print whose turn it is."""
    if game.state == State.BLACK_TURN:
        print(f"{ANSI_BLACK}     BLACK turn {ANSI_RESET}")
    elif game.state == State.WHITE_TURN:
        print("     WHITE turn ")
    else:
        print("  Unexpected state.")
