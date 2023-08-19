from __future__ import annotations
import random
import copy
import math
from .othello import Othello, State


class Node:
    """Node of the MCTS tree."""

    def __init__(self, move: tuple[int, int], turn: State, unexplored: list[tuple[int, int]], parent: Node | None):
        self.move = move
        self.turn = turn
        self.unexplored = unexplored
        self.parent = parent
        self.children: list[Node] = []
        self.visits = 0
        self.wins = 0

    def add_and_get_child(self, move: tuple[int, int], turn: State, unexplored: list[tuple[int, int]]) -> Node:
        child = Node(move, turn, unexplored, self)
        self.unexplored.remove(move)
        self.children.append(child)
        return child

    def select_child(self) -> Node:
        best_uct = float("-inf")
        selected = self.children[0]

        log_total = 2 * math.log(self.visits)
        for child in self.children:
            # UCT formula for selecting promising nodes
            child_uct = child.wins / child.visits + math.sqrt(log_total / child.visits)
            if child_uct > best_uct:
                best_uct = child_uct
                selected = child
        return selected

    def get_most_visited(self) -> Node:  # best move
        return max(self.children, key=lambda child: child.visits)


def mcts_move(game: Othello, iterations: int) -> tuple[int, int]:
    """Returns the best move for the current turn using Monte Carlo Tree Search."""
    root = Node((-1, -1), game.state, game.get_valid_moves(), None)  # root move has no position
    for _ in range(iterations):  # one iteration explores one move
        node = root
        simulation = copy.deepcopy(game)

        # SELECT promising child node while current node is fully expanded and non-terminal
        while node.unexplored == [] and node.children != []:
            node = node.select_child()
            simulation.make_move(node.move)

        # EXPAND one random unexplored move
        if node.unexplored != []:
            move = node.unexplored[random.randint(0, len(node.unexplored) - 1)]
            turn = simulation.state
            simulation.make_move(move)
            node = node.add_and_get_child(move, turn, simulation.get_valid_moves())

        # SIMULATE while game is not over, make a random move
        while simulation.state in (State.BLACK_TURN, State.WHITE_TURN):
            moves = simulation.get_valid_moves()
            move = moves[random.randint(0, len(moves) - 1)]
            simulation.make_move(move)

        # BACKPROPAGATE simulation result
        winner = simulation.state
        while node is not None:
            node.visits += 1
            if winner == State.DRAW:
                pass
            elif (winner == State.BLACK_WON) == (node.turn == State.BLACK_TURN):
                node.wins += 1
            else:
                node.wins -= 1
            node = node.parent

    return root.get_most_visited().move
