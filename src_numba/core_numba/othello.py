import numpy as np
from numba import jit, njit

# Constants replacing Enums
CELL_EMPTY = 0
CELL_BLACK = 1
CELL_WHITE = 2
CELL_VALID = 3

STATE_BLACK_TURN = 1
STATE_WHITE_TURN = 2
STATE_BLACK_WON = 3
STATE_WHITE_WON = 4
STATE_DRAW = 5

# Directions for checking flipped cells
DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=np.int32)


@njit
def init_game():
    """Initialize the Othello game state."""
    board = np.zeros((8, 8), dtype=np.uint8)
    board[3, 3] = CELL_WHITE
    board[3, 4] = CELL_BLACK
    board[4, 3] = CELL_BLACK
    board[4, 4] = CELL_WHITE
    board[2, 3] = CELL_VALID
    board[3, 2] = CELL_VALID
    board[4, 5] = CELL_VALID
    board[5, 4] = CELL_VALID
    return board, 2, 2, STATE_BLACK_TURN


@njit
def make_move(
    board: np.ndarray,
    black_score: np.int32,
    white_score: np.int32,
    state: np.int32,
    move_x: int,
    move_y: int,
):
    """Make a move and update the game state. Returns (board, black_score, white_score, state, success)."""

    if state not in (STATE_BLACK_TURN, STATE_WHITE_TURN):
        return board, black_score, white_score, state, 0
    if board[move_y, move_x] != CELL_VALID:
        return board, black_score, white_score, state, 0

    player = CELL_BLACK if state == STATE_BLACK_TURN else CELL_WHITE
    opponent = CELL_WHITE if state == STATE_BLACK_TURN else CELL_BLACK
    board[move_y, move_x] = player

    # Flip cells and update scores incrementally
    flipped = get_flipped_cells(board, move_x, move_y, player, opponent)
    num_flipped = flipped.shape[0]
    for x, y in flipped:
        board[y, x] = player

    # Update scores: +1 for the new piece, +num_flipped for flipped pieces
    if player == CELL_BLACK:
        black_score += 1 + num_flipped
        white_score -= num_flipped
    else:
        white_score += 1 + num_flipped
        black_score -= num_flipped

    # Update state
    return update_state(board, black_score, white_score, state)


@njit
def get_valid_moves(board: np.ndarray, state: np.int32):
    """Return valid moves as a NumPy array of [x, y] coordinates."""

    if state not in (STATE_BLACK_TURN, STATE_WHITE_TURN):
        return np.zeros((0, 2), dtype=np.int32)

    valid_moves = np.zeros((64, 2), dtype=np.int32)
    count = 0
    for y in range(8):
        for x in range(8):
            if board[y, x] == CELL_VALID:
                valid_moves[count, 0] = x
                valid_moves[count, 1] = y
                count += 1
    return valid_moves[:count]


@njit
def update_state(
    board: np.ndarray,
    black_score: np.int32,
    white_score: np.int32,
    state: np.int32,
):
    """Update turn, valid cells, and game state. Returns (board, black_score, white_score, state, success)."""
    # Check if game is over
    if is_full(board) or black_score == 0 or white_score == 0:
        if black_score > white_score:
            return board, black_score, white_score, STATE_BLACK_WON, 1
        elif black_score < white_score:
            return board, black_score, white_score, STATE_WHITE_WON, 1
        else:
            return board, black_score, white_score, STATE_DRAW, 1

    # Switch turn and update valid cells
    next_state = STATE_WHITE_TURN if state == STATE_BLACK_TURN else STATE_BLACK_TURN
    board = update_valid_cells(board, next_state)

    # Check if there are valid moves
    valid_moves = get_valid_moves(board, next_state)
    if valid_moves.shape[0] == 0:
        # No valid moves, switch turn again
        next_state = STATE_WHITE_TURN if next_state == STATE_BLACK_TURN else STATE_BLACK_TURN
        board = update_valid_cells(board, next_state)
        valid_moves = get_valid_moves(board, next_state)
        if valid_moves.shape[0] == 0:
            # No valid moves for either player, game over
            if black_score > white_score:
                return board, black_score, white_score, STATE_BLACK_WON, 1
            elif black_score < white_score:
                return board, black_score, white_score, STATE_WHITE_WON, 1
            else:
                return board, black_score, white_score, STATE_DRAW, 1

    return board, black_score, white_score, next_state, 1


@njit
def is_full(board: np.ndarray):
    """Check if the board is full."""
    for y in range(8):
        for x in range(8):
            if board[y, x] in (CELL_EMPTY, CELL_VALID):
                return 0
    return 1


@njit
def update_valid_cells(board: np.ndarray, state: np.int32):
    """Update valid cells for the current player's turn."""

    player = CELL_BLACK if state == STATE_BLACK_TURN else CELL_WHITE
    opponent = CELL_WHITE if state == STATE_BLACK_TURN else CELL_BLACK

    # Clear and set valid cells in one pass
    for y in range(8):
        for x in range(8):
            if board[y, x] == CELL_VALID:
                board[y, x] = CELL_EMPTY
            elif board[y, x] == CELL_EMPTY:
                if len_flipped_cells(board, x, y, player, opponent):
                    board[y, x] = CELL_VALID
    return board


@njit
def get_flipped_cells(
    board: np.ndarray,
    x: int,
    y: int,
    player: np.int32,
    opponent: np.int32,
):
    """Get all cells that would be flipped by a move."""
    flipped = np.zeros((64, 2), dtype=np.int32)
    count = 0

    for dx, dy in DIRECTIONS:
        line_flipped = flipped_cells_in_direction(board, x, y, dx, dy, player, opponent)
        for fx, fy in line_flipped:
            flipped[count, 0] = fx
            flipped[count, 1] = fy
            count += 1

    return flipped[:count]


@njit
def len_flipped_cells(
    board: np.ndarray,
    x: int,
    y: int,
    player: np.int32,
    opponent: np.int32,
):
    """Count flipped cells without storing them."""

    count = 0
    for dx, dy in DIRECTIONS:
        line_flipped = flipped_cells_in_direction(board, x, y, dx, dy, player, opponent)
        count += line_flipped.shape[0]
    return count


@njit
def flipped_cells_in_direction(
    board: np.ndarray,
    x: int,
    y: int,
    dx: int,
    dy: int,
    player: np.int32,
    opponent: np.int32,
):
    """Get flipped cells in a specific direction."""

    flipped = np.zeros((8, 2), dtype=np.int32)
    count = 0
    x, y = x + dx, y + dy

    while 0 <= x < 8 and 0 <= y < 8 and board[y, x] == opponent:
        flipped[count, 0] = x
        flipped[count, 1] = y
        count += 1
        x, y = x + dx, y + dy

    if not (0 <= x < 8 and 0 <= y < 8) or board[y, x] != player:
        return flipped[:0]
    return flipped[:count]


@njit
def make_move_reversible(
    board: np.ndarray,
    black_score: np.int32,
    white_score: np.int32,
    state: np.int32,
    move_x: int,
    move_y: int,
):
    """Make a move and return flipped cells for undoing. Returns (board, black_score, white_score, state, flipped)."""

    if state not in (STATE_BLACK_TURN, STATE_WHITE_TURN) or board[move_y, move_x] != 3:
        return board, black_score, white_score, state, np.zeros((0, 2), dtype=np.int32)

    player = CELL_BLACK if state == STATE_BLACK_TURN else CELL_WHITE
    opponent = CELL_WHITE if state == STATE_BLACK_TURN else CELL_BLACK
    board[move_y, move_x] = player

    flipped = get_flipped_cells(board, move_x, move_y, player, opponent)
    num_flipped = flipped.shape[0]
    for x, y in flipped:
        board[y, x] = player

    # Update scores
    if player == CELL_BLACK:
        black_score += 1 + num_flipped
        white_score -= num_flipped
    else:
        white_score += 1 + num_flipped
        black_score -= num_flipped

    board, black_score, white_score, new_state, _ = update_state(board, black_score, white_score, state)
    return board, black_score, white_score, new_state, flipped


@njit
def undo_move(
    board: np.ndarray,
    move_x: int,
    move_y: int,
    flipped: np.ndarray,
    original_state: np.int32,
):
    """Undo a move by restoring the board."""

    player = CELL_BLACK if original_state == STATE_BLACK_TURN else CELL_WHITE
    opponent = CELL_WHITE if original_state == STATE_BLACK_TURN else CELL_BLACK

    # Restore flipped cells to opponent
    for x, y in flipped:
        board[y, x] = opponent

    # Clear the move position
    board[move_y, move_x] = 0  # Will be updated to VALID later if needed


# Example usage (not Numba-compiled)
def print_board(board: np.ndarray) -> None:
    """Print the board for debugging."""

    for row in board:
        print(
            " ".join(
                ["." if c == CELL_EMPTY else "V" if c == CELL_VALID else "B" if c == CELL_BLACK else "W" for c in row]
            )
        )
    print()


def state_to_str(state: int) -> str:
    """Convert state to string for printing."""

    return {
        STATE_BLACK_TURN: "Black's turn",
        STATE_WHITE_TURN: "White's turn",
        STATE_BLACK_WON: "Black won",
        STATE_WHITE_WON: "White won",
        STATE_DRAW: "Draw",
    }.get(state, "Unknown")


if __name__ == "__main__":
    # Initialize game
    board, black_score, white_score, state = init_game()
    print("Initial board:")
    print_board(board)

    # Test a valid move
    print("Making move (3, 2):")
    board, black_score, white_score, state, success = make_move(board, black_score, white_score, state, 3, 2)
    if success:
        print("Move successful!")
        print_board(board)
        print(f"Black score: {black_score}, White score: {white_score}")
        print(f"State: {state_to_str(state)}")

        # Print valid moves
        valid_moves = get_valid_moves(board, state)
        print("Valid moves:", valid_moves.tolist())
    else:
        print("Move failed!")

    # Test an invalid move
    print("\nMaking invalid move (0, 0):")
    board, black_score, white_score, state, success = make_move(board, black_score, white_score, state, 0, 0)
    if success:
        print("Move successful!")
        print_board(board)
    else:
        print("Move failed!")
