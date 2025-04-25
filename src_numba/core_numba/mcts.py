from __future__ import annotations
import random
import math
import numpy as np
from numba import njit
from typing import List, Tuple
from .othello import (
    make_move,
    get_valid_moves,
    STATE_BLACK_TURN,
    STATE_WHITE_TURN,
    STATE_BLACK_WON,
    STATE_WHITE_WON,
    STATE_DRAW,
)


def mcts_move(board: np.ndarray, black_score: int, white_score: int, state: int, iterations: int) -> Tuple[int, int]:
    """Returns the best move for the current turn using Monte Carlo Tree Search."""
    valid_moves = [tuple(move) for move in get_valid_moves(board, state)]  # Convert to list of tuples
    root = Node(None, (-1, -1), state, valid_moves)

    for _ in range(iterations):
        node = root
        # Create a new game state for simulation
        sim_board = board.copy()
        sim_black_score = black_score
        sim_white_score = white_score
        sim_state = state

        # SELECT promising child node while current node is fully expanded and non-terminal
        while node.unexplored == [] and node.children != []:
            node = node.select_child()
            sim_board, sim_black_score, sim_white_score, sim_state, success = make_move(
                sim_board, sim_black_score, sim_white_score, sim_state, node.move[0], node.move[1]
            )
            if not success:
                break

        # EXPAND one random unexplored move
        if node.unexplored != []:
            explored_move = node.unexplored[random.randint(0, len(node.unexplored) - 1)]
            explored_turn = sim_state
            sim_board, sim_black_score, sim_white_score, sim_state, success = make_move(
                sim_board, sim_black_score, sim_white_score, sim_state, explored_move[0], explored_move[1]
            )
            if success:
                # Remove explored move and add child node
                node.unexplored.remove(explored_move)
                child = Node(
                    node, explored_move, explored_turn, [tuple(move) for move in get_valid_moves(sim_board, sim_state)]
                )
                node.children.append(child)
                node = child

        # SIMULATE while game is not over
        winner = simulate_game(sim_board, sim_black_score, sim_white_score, sim_state)

        # BACKPROPAGATE simulation result
        while node is not None:
            node.visits += 1
            node.wins += compute_win_increment(winner, node.turn)
            node = node.parent

    return root.get_most_visited().move


class Node:
    """Node of the MCTS tree."""

    def __init__(self, parent: Node | None, move: Tuple[int, int], turn: int, unexplored: List[Tuple[int, int]]):
        self.move = move
        self.turn = turn
        self.unexplored = unexplored
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.wins = 0

    def select_child(self) -> Node:
        selected = self.children[0]
        best_uct = float("-inf")
        ln_total = 2 * math.log(self.visits)
        for child in self.children:
            # UCT formula extracted to Numba function
            child_uct = compute_uct(child.wins, child.visits, ln_total)
            if child_uct > best_uct:
                best_uct = child_uct
                selected = child
        return selected

    def get_most_visited(self) -> Node:
        # Use Numba function to find index of most visited child
        visits = np.array([child.visits for child in self.children], dtype=np.int32)
        max_idx = find_most_visited(visits)
        return self.children[max_idx]


@njit
def compute_uct(wins: np.int32, visits: np.int32, ln_total: np.float64) -> np.float64:
    """Compute the UCT value for a node."""
    return (wins / visits) + np.sqrt(ln_total / visits)


@njit
def simulate_game(board: np.ndarray, black_score: np.int32, white_score: np.int32, state: np.int32):
    """Simulate a random game from the given state and return the winner."""
    sim_board = board.copy()
    sim_black_score = black_score
    sim_white_score = white_score
    sim_state = state

    while sim_state in (STATE_BLACK_TURN, STATE_WHITE_TURN):
        moves = get_valid_moves(sim_board, sim_state)
        if moves.shape[0] == 0:
            sim_board, sim_black_score, sim_white_score, sim_state, _ = make_move(
                sim_board, sim_black_score, sim_white_score, sim_state, 0, 0
            )
            continue
        move_idx = np.random.randint(0, moves.shape[0])
        sim_board, sim_black_score, sim_white_score, sim_state, success = make_move(
            sim_board, sim_black_score, sim_white_score, sim_state, moves[move_idx, 0], moves[move_idx, 1]
        )
        if not success:
            break

    return sim_state


@njit
def compute_win_increment(winner: np.int32, turn: np.int32):
    """Compute the win increment for backpropagation."""
    if winner == STATE_DRAW:
        return 0
    if (winner == STATE_BLACK_WON) == (turn == STATE_BLACK_TURN):
        return 1
    return -1


@njit
def find_most_visited(visits: np.ndarray):
    """Return the index of the child with the most visits."""
    max_idx = 0
    max_visits = visits[0]
    for i in range(1, visits.shape[0]):
        if visits[i] > max_visits:
            max_visits = visits[i]
            max_idx = i
    return max_idx
