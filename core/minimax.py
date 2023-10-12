import copy
import random
import sys
from .othello import Othello, Cell, State


def minimax_move(game: Othello, depth: int) -> tuple[int, int]:
    """Use minimax algorithm to find a good move for the current player."""
    moves = game.get_valid_moves()
    if len(moves) == 1:  # only one move available
        return moves[0]

    round_idx = _calculate_round(game.board)  # random first move
    if round_idx < 3:
        return moves[random.randint(0, len(moves) - 1)]

    # increase depth based on round, later rounds matter more
    if round_idx >= 50:
        depth += 10  # end game solver
    elif round_idx > 40:
        depth += 2
    elif round_idx > 30:
        depth += 1

    return _minimax(game, game.state, depth, -sys.maxsize, sys.maxsize)[1]


def _minimax(game: Othello, my_turn: State, depth: int, alpha: int, beta: int) -> tuple[int, tuple[int, int]]:
    """Minimax tree search algorithm."""
    state = game.state
    if depth == 0 or state != State.BLACK_TURN and state != State.WHITE_TURN:
        return _evaluate_board(game, my_turn), (-1, -1)

    moves = game.get_valid_moves()
    best_move = moves[0]
    best_value = -sys.maxsize if state == my_turn else sys.maxsize

    for move in moves:
        simulation = copy.deepcopy(game)
        simulation.make_move(move)
        value = _minimax(simulation, my_turn, depth - 1, alpha, beta)[0]

        if state == my_turn:
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(best_value, alpha)
        else:
            if value < best_value:
                best_value = value
                best_move = move
            beta = min(
                best_value,
                beta,
            )
        if alpha >= beta:
            break  # prune

    return best_value, best_move


REWARDS = [
    [80, -20, 20, 10, 10, 20, -20, 80],
    [-20, -40, -10, -10, -10, -10, -40, -20],
    [20, -10, 10, 0, 0, 10, -10, 20],
    [10, -10, 0, 5, 5, 0, -10, 10],
    [10, -10, 0, 5, 5, 0, -10, 10],
    [20, -10, 10, 0, 0, 10, -10, 20],
    [-20, -40, -10, -10, -10, -10, -40, -20],
    [80, -20, 20, 10, 10, 20, -20, 80],
]


def _evaluate_board(game: Othello, my_turn: State) -> int:
    """Use a heuristic to evaluate the board."""
    state = game.state
    if state == State.BLACK_WON:
        return sys.maxsize if my_turn == State.BLACK_TURN else -sys.maxsize
    elif state == State.WHITE_WON:
        return sys.maxsize if my_turn == State.WHITE_TURN else -sys.maxsize
    elif state == State.DRAW:
        return 0

    reward = 0
    for y in range(8):
        for x in range(8):
            if game.board[y][x] == Cell.BLACK:
                reward += REWARDS[y][x] if my_turn == State.BLACK_TURN else -REWARDS[y][x]
            elif game.board[y][x] == Cell.WHITE:
                reward += REWARDS[y][x] if my_turn == State.WHITE_TURN else -REWARDS[y][x]
    return reward


def _calculate_round(board: list[list[Cell]]) -> int:
    """Calculate the current round based on the number of pieces on the board."""
    count = -3
    for y in range(8):
        for x in range(8):
            if board[y][x] in (Cell.BLACK, Cell.WHITE):
                count += 1
    return count
