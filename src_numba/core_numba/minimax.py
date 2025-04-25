import random
from typing import Tuple

import numpy as np
from numba import njit

from .othello import (
    CELL_BLACK,
    CELL_WHITE,
    STATE_BLACK_TURN,
    STATE_BLACK_WON,
    STATE_DRAW,
    STATE_WHITE_TURN,
    STATE_WHITE_WON,
    get_valid_moves,
    make_move,
)

# Rewards matrix for board evaluation (NumPy array for Numba)
REWARDS = np.array(
    [
        [80, -20, 20, 10, 10, 20, -20, 80],
        [-20, -40, -10, -10, -10, -10, -40, -20],
        [20, -10, 10, 0, 0, 10, -10, 20],
        [10, -10, 0, 5, 5, 0, -10, 10],
        [10, -10, 0, 5, 5, 0, -10, 10],
        [20, -10, 10, 0, 0, 10, -10, 20],
        [-20, -40, -10, -10, -10, -10, -40, -20],
        [80, -20, 20, 10, 10, 20, -20, 80],
    ],
    dtype=np.int32,
)


def minimax_move(board: np.ndarray, black_score: int, white_score: int, state: int, depth: int) -> Tuple[int, int]:
    """Use minimax to find a good move for the current player. Returns (x, y)."""
    moves = [tuple(move) for move in get_valid_moves(board, state)]  # Convert to list of tuples
    if not moves:
        return (-1, -1)
    if len(moves) == 1:
        return moves[0]

    round_idx = _calculate_round(board)
    if round_idx < 3:
        return moves[random.randint(0, len(moves) - 1)]

    # Adjust depth based on round
    if round_idx >= 50:
        depth += 10  # Endgame solver
    elif round_idx > 40:
        depth += 2
    elif round_idx > 30:
        depth += 1

    _, best_move = _minimax(board, black_score, white_score, state, state, depth, -float("inf"), float("inf"))
    return best_move


def _minimax(
    board: np.ndarray,
    black_score: int,
    white_score: int,
    state: int,
    my_turn: int,
    depth: int,
    alpha: float,
    beta: float,
) -> Tuple[float, Tuple[int, int]]:
    """Minimax with alpha-beta pruning. Returns (value, (x, y))."""
    if depth == 0 or state not in (STATE_BLACK_TURN, STATE_WHITE_TURN):
        return _evaluate_board(board, black_score, white_score, state, my_turn), (-1, -1)

    moves = [tuple(move) for move in get_valid_moves(board, state)]
    if not moves:
        return _evaluate_board(board, black_score, white_score, state, my_turn), (-1, -1)

    best_move = moves[0]
    best_value = float("-inf") if state == my_turn else float("inf")

    for move in moves:
        # Create a new game state
        sim_board = board.copy()
        sim_black_score = black_score
        sim_white_score = white_score
        sim_state = state
        sim_board, sim_black_score, sim_white_score, sim_state, success = make_move(
            sim_board, sim_black_score, sim_white_score, sim_state, move[0], move[1]
        )
        if not success:
            continue  # Skip invalid moves

        value = _minimax(sim_board, sim_black_score, sim_white_score, sim_state, my_turn, depth - 1, alpha, beta)[0]

        if state == my_turn:
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)
        else:
            if value < best_value:
                best_value = value
                best_move = move
            beta = min(beta, best_value)

        if alpha >= beta:
            break

    return best_value, best_move


@njit
def _evaluate_board(board: np.ndarray, black_score: int, white_score: int, state: int, my_turn: int) -> float:
    """Evaluate the board using the REWARDS matrix with Numba-compatible loops."""
    if state == STATE_BLACK_WON:
        return np.float64(np.inf) if my_turn == STATE_BLACK_TURN else np.float64(-np.inf)
    if state == STATE_WHITE_WON:
        return np.float64(np.inf) if my_turn == STATE_WHITE_TURN else np.float64(-np.inf)
    if state == STATE_DRAW:
        return 0.0

    reward = np.float64(0.0)
    for y in range(8):
        for x in range(8):
            if board[y, x] == CELL_BLACK:
                reward += REWARDS[y, x] if my_turn == STATE_BLACK_TURN else -REWARDS[y, x]
            elif board[y, x] == CELL_WHITE:
                reward += REWARDS[y, x] if my_turn == STATE_WHITE_TURN else -REWARDS[y, x]
    return reward


@njit
def _calculate_round(board: np.ndarray) -> int:
    """Calculate the current round based on the number of pieces using NumPy."""
    count = np.sum((board == CELL_BLACK) | (board == CELL_WHITE))
    return int(count - 3)
