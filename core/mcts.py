import random
from .othello import Othello, Cell, State

# TODO: Implement Monte Carlo Tree Search

def _random_move(game: Othello) -> tuple[int, int]:
    moves = game.get_valid_moves()
    x, y = moves[random.randint(0, len(moves) - 1)]
    return x, y