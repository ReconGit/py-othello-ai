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

    # iterate depth based on round, later rounds matter more
    if round_idx >= 50:
        depth += 10  # end game solver
    elif round_idx > 40:
        depth += 2  # later rounds are faster
    elif round_idx > 30:
        depth += 1

    my_turn = game.state
    best_move = moves[0]
    best_value = -sys.maxsize
    for move in moves:
        simulation = copy.deepcopy(game)
        simulation.make_move(move)
        value = _minimax(simulation, my_turn, depth - 1, -sys.maxsize, sys.maxsize)
        # print(f"Move: {move}, value: {value}") # for debug
        if value > best_value:
            best_value = value
            best_move = move
        if value >= 300:
            break  # good enough move

    return best_move


def _minimax(game: Othello, my_turn: State, depth: int, alpha: int, beta: int) -> int:
    """Minimax tree search algorithm."""
    state = game.state
    if state == State.BLACK_WON:
        return 300 if my_turn == State.BLACK_TURN else -300
    elif state == State.WHITE_WON:
        return 300 if my_turn == State.WHITE_TURN else -300
    elif state == State.DRAW:
        return 0

    if depth <= 0:
        return _evaluate_board(game.board, my_turn)

    moves = game.get_valid_moves()
    best_value = -sys.maxsize if state == my_turn else sys.maxsize
    for move in moves:
        simulation = copy.deepcopy(game)
        simulation.make_move(move)
        value = _minimax(simulation, my_turn, depth - 1, alpha, beta)

        if state == my_turn:
            best_value = max(best_value, value)
            alpha = max(best_value, alpha)
        else:
            best_value = min(best_value, value)
            beta = min(
                best_value,
                beta,
            )
        if alpha >= beta:
            break  # prune

    return best_value


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


def _evaluate_board(board: list[list[Cell]], my_turn: State) -> int:
    """Use a heuristic to evaluate the board."""
    reward = 0
    for y in range(8):
        for x in range(8):
            if board[y][x] == Cell.BLACK:
                reward += REWARDS[y][x] if my_turn == State.BLACK_TURN else -REWARDS[y][x]
            elif board[y][x] == Cell.WHITE:
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
