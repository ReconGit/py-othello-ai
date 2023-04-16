from __future__ import annotations
import random
import copy
import math
from .othello import Othello, State


class Node:
    """Node of the MCTS tree."""

    def __init__(self, x: int, y: int, turn: State, unexplored: list[tuple[int, int]], parent: Node | None):
        self.position = x, y
        self.turn = turn
        self.unexplored = unexplored
        self.parent = parent
        self.children: list[Node] = []
        self.visits = 0
        self.wins = 0

    def add_child(self, x: int, y: int, player: State, unexplored: list[tuple[int, int]]) -> Node:
        child = Node(x, y, player, unexplored, self)
        self.unexplored.remove((x, y))
        self.children.append(child)
        return child

    def select_child(self) -> Node:
        best_uct = float("-inf")
        selected = self.children[0]
        for child in self.children:  # UCT formula for selecting promising nodes
            child_uct = child.wins / child.visits + math.sqrt(2 * math.log(self.visits) / child.visits)
            if child_uct > best_uct:
                best_uct = child_uct
                selected = child
        return selected

    def get_most_visited(self) -> Node:  # best move
        return max(self.children, key=lambda child: child.visits)


def mcts_move(game: Othello, iterations: int, nsims: int) -> tuple[int, int]:
    """Returns the best move for the current player using Monte Carlo Tree Search."""

    root = Node(-1, -1, game.state, game.get_valid_moves(), None)  # root node has no position
    for _ in range(iterations):
        # one iteration explores one move
        node = root
        simulation = copy.deepcopy(game)

        # select
        while node.unexplored == [] and node.children != []:  # node is fully expanded and non-terminal
            node = node.select_child()
            simulation.make_move(node.position[0], node.position[1])

        # expand
        if simulation.state in (State.BLACK_TURN, State.WHITE_TURN):  # game could be over from the select step
            if node.unexplored != []:
                x, y = node.unexplored[random.randint(0, len(node.unexplored) - 1)]
                turn = simulation.state
                simulation.make_move(x, y)
                node = node.add_child(x, y, turn, simulation.get_valid_moves())

        for _ in range(nsims):
            # simulate
            while simulation.state in (State.BLACK_TURN, State.WHITE_TURN):
                moves = simulation.get_valid_moves()
                x, y = moves[random.randint(0, len(moves) - 1)]
                simulation.make_move(x, y)  # play a random move

            # backpropagate
            winner = simulation.state
            while node is not None:
                node.visits += 1
                if winner == State.BLACK_WON and node.turn == State.BLACK_TURN or winner == State.WHITE_WON and node.turn == State.WHITE_TURN:
                    node.wins += 1
                elif winner == State.WHITE_WON and node.turn == State.BLACK_TURN or winner == State.BLACK_WON and node.turn == State.WHITE_TURN:
                    node.wins -= 1
                node = node.parent

    return root.get_most_visited().position


def _random_move(game: Othello) -> tuple[int, int]:
    moves = game.get_valid_moves()
    x, y = moves[random.randint(0, len(moves) - 1)]
    return x, y
