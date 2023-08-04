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

    # iterate depth based on round index, later rounds matter more
    if round_idx >= 50: depth = depth + 10  # end game solver
    elif round_idx > 44: depth = depth + 2  # later rounds are faster
    elif round_idx > 34: depth = depth + 1
    # elif round_idx > 24: depth = depth + 1
    # elif round_idx < 20: depth = depth + 1  # early rounds are fast

    my_turn = game.state
    best_move = moves[0]
    best_value = -sys.maxsize
    for move in moves:
        simulation = copy.deepcopy(game)
        simulation.make_move(move[0], move[1])
        value = _minimax(simulation, my_turn, depth - 1, -sys.maxsize, sys.maxsize)
        # print(f"Move: {move}, value: {value}") # for debug
        if value > best_value:
            best_value = value
            best_move = move
        if value >= 300: break  # good enough move

    return best_move


def _minimax(game: Othello, my_turn: State, depth: int, alpha: int, beta: int) -> int:
    """Minimax tree search algorithm."""
    state = game.state
    if state == State.BLACK_WON: return 300 if State.BLACK_TURN == my_turn else -300
    elif state == State.WHITE_WON: return 300 if State.WHITE_TURN == my_turn else -300
    elif state == State.DRAW: return 0

    if depth <= 0:  # use heuristic when out of depth
        return _evaluate_board(game.board, my_turn)

    moves = game.get_valid_moves()
    best_value = -sys.maxsize if state == my_turn else sys.maxsize
    for move in moves:
        simulation = copy.deepcopy(game)
        simulation.make_move(move[0], move[1])
        value = _minimax(simulation, my_turn, depth - 1, alpha, beta)

        if state == my_turn:
            best_value = max(value, best_value)
            alpha = max(alpha, best_value)
        else:
            best_value = min(value, best_value)
            beta = min(beta, best_value)
        if alpha >= beta: break  # prune

    return best_value


def _evaluate_board(board: list[list[Cell]], ai_turn: State) -> int:
    """Use a heuristic to evaluate the board."""
    rewards = [
        [80, -20, 20, 10, 10, 20, -20, 80],
        [-20, -40, -10, -10, -10, -10, -40, -20],
        [20, -10, 10, 0, 0, 10, -10, 20],
        [10, -10, 0, 5, 5, 0, -10, 10],
        [10, -10, 0, 5, 5, 0, -10, 10],
        [20, -10, 10, 0, 0, 10, -10, 20],
        [-20, -40, -10, -10, -10, -10, -40, -20],
        [80, -20, 20, 10, 10, 20, -20, 80],
    ]
    reward = 0
    for y in range(8):
        for x in range(8):
            if board[y][x] == Cell.BLACK:
                reward += rewards[y][x] if ai_turn == State.BLACK_TURN else -rewards[y][x]
            elif board[y][x] == Cell.WHITE:
                reward += rewards[y][x] if ai_turn == State.WHITE_TURN else -rewards[y][x]
    return reward


def _calculate_round(board: list[list[Cell]]) -> int:
    """Calculate the current round based on the number of pieces on the board."""
    count = -3
    for y in range(8):
        for x in range(8):
            if board[y][x] in (Cell.BLACK, Cell.WHITE):
                count += 1
    return count
