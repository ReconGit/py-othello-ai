import copy
import random
from .othello import Othello, Cell, State


def minimax_move(game: Othello, depth: int) -> tuple[int, int]:
    """Use minimax algorithm to find a good move for the current player."""
    
    moves = game.get_valid_moves()
    if len(moves) == 1: return moves[0]  # only one move available

    round = _round(game.board)  # random first move
    if round < 3: return moves[random.randint(0, len(moves) - 1)]

    # iterate depth based on round, later rounds matter more
    if round >= 50: depth = depth + 10  # end game solver
    elif round > 44: depth = depth + 2  # later rounds are faster
    elif round > 34: depth = depth + 1
    # elif round > 24: depth = depth + 1
    # elif round < 20: depth = depth + 1  # early rounds are fast

    state = game.state
    best_move = moves[0]
    best_value = -10000 if state == State.BLACK_TURN else 10000
    for move in moves:
        simulation = copy.deepcopy(game)
        simulation.make_move(move[0], move[1])
        value = _minimax(simulation, state, depth - 1, -10000, 10000)
        # print(f"Move: {move}, value: {value}")
        if value > best_value:
            best_value = value
            best_move = move
        if value >= 300: break

    return best_move


def _minimax(game: Othello, ai_turn: State, depth: int, alpha: int, beta: int) -> int:
    """Minimax tree search algorithm."""
    state = game.state
    if state == State.BLACK_WON: return 300 if State.BLACK_TURN == ai_turn else -300
    elif state == State.WHITE_WON: return 300 if State.WHITE_TURN == ai_turn else -300
    elif state == State.DRAW: return 0

    if depth == 0: return _evaluate_board(game.board, ai_turn)  # use heuristic when out of depth

    moves = game.get_valid_moves()
    best_value = -10000 if state == ai_turn else 10000
    for move in moves:
        simulation = copy.deepcopy(game)
        simulation.make_move(move[0], move[1])
        value = _minimax(simulation, ai_turn, depth - 1, alpha, beta)

        if state == ai_turn:
            best_value = max(best_value, value)
            alpha = max(alpha, value)
        else:
            best_value = min(best_value, value)
            beta = min(beta, value)
        if alpha >= beta: break

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


def _round(board: list[list[Cell]]) -> int:
    """Return the round based on the number of pieces on the board."""
    count = -3
    for y in range(8):
        for x in range(8):
            if board[y][x] in (Cell.BLACK, Cell.WHITE):
                count += 1
    return count
