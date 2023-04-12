import time
import random
from core.othello import Othello, State, Cell
from core.minimax import minimax_move
from core.mcts import _random_move, mcts_move
from core.ui import print_board, print_score, print_turn


def performance_test():
    random.seed(10)
    start = time.time()
    game = Othello()
    games = 0
    limit = 1000
    while True:
        while game.state == State.BLACK_TURN or game.state == State.WHITE_TURN:
            try:
                x, y = _random_move(game)
                game.make_move(x, y)
            except IndexError as e:
                print(e)
        games += 1
        if games >= limit:
            break

    end = time.time()
    print(f"Time: {end - start}")


def minimax_vs_random():
    start = time.time()

    limit = 10
    depth = 1

    games = 0
    black_wins = 0
    white_wins = 0
    draws = 0
    while True:
        game = Othello()
        while game.state == State.BLACK_TURN or game.state == State.WHITE_TURN:
            # print_board(game.board)
            x, y = minimax_move(game, depth) if game.state == State.BLACK_TURN else _random_move(game)
            game.make_move(x, y)

        if game.state == State.BLACK_WON:
            black_wins += 1
            print("AI WON")
        elif game.state == State.WHITE_WON:
            white_wins += 1
            print("AI LOST")
        else:
            draws += 1
            print("DRAW")
        games += 1
        if games >= limit:
            break

    print(f"BLACK Minimax AI: {black_wins}")
    print(f"WHITE Random: {white_wins}")
    print(f"Draws: {draws}\n")

    games = 0
    black_wins = 0
    white_wins = 0
    draws = 0
    while True:
        game = Othello()
        while game.state == State.BLACK_TURN or game.state == State.WHITE_TURN:
            # print_board(game.board)
            x, y = minimax_move(game, depth) if game.state == State.WHITE_TURN else _random_move(game)
            game.make_move(x, y)

        if game.state == State.BLACK_WON:
            black_wins += 1
            print("AI LOST")
        elif game.state == State.WHITE_WON:
            white_wins += 1
            print("AI WON")
        else:
            draws += 1
            print("DRAW")
        games += 1
        if games >= limit:
            break

    print(f"BLACK Random: {black_wins}")
    print(f"WHITE Minimax AI: {white_wins}")
    print(f"Draws: {draws}\n")

    end = time.time()
    print(f"Time: {end - start}")


def mcts_vs_random():
    start = time.time()

    limit = 20
    iterations = 50
    nsims = 50

    games = 0
    black_wins = 0
    white_wins = 0
    draws = 0
    while True:
        game = Othello()
        while game.state == State.BLACK_TURN or game.state == State.WHITE_TURN:
            # print_board(game.board)
            x, y = mcts_move(game, iterations, nsims) if game.state == State.BLACK_TURN else _random_move(game)
            game.make_move(x, y)

        if game.state == State.BLACK_WON:
            black_wins += 1
            # print("AI WON")
        elif game.state == State.WHITE_WON:
            white_wins += 1
            # print("AI LOST")
        else:
            draws += 1
            # print("DRAW")
        games += 1
        if games >= limit:
            break

    print(f"BLACK MCTS AI: {black_wins}")
    print(f"WHITE Random: {white_wins}")
    print(f"Draws: {draws}\n")

    games = 0
    black_wins = 0
    white_wins = 0
    draws = 0
    while True:
        game = Othello()
        while game.state == State.BLACK_TURN or game.state == State.WHITE_TURN:
            # print_board(game.board)
            x, y = mcts_move(game, iterations, nsims) if game.state == State.WHITE_TURN else _random_move(game)
            game.make_move(x, y)

        if game.state == State.BLACK_WON:
            black_wins += 1
            # print("AI LOST")
        elif game.state == State.WHITE_WON:
            white_wins += 1
            # print("AI WON")
        else:
            draws += 1
            # print("DRAW")
        games += 1
        if games >= limit:
            break

    print(f"BLACK Random: {black_wins}")
    print(f"WHITE MCTS AI: {white_wins}")
    print(f"Draws: {draws}\n")

    end = time.time()
    print(f"Time: {end - start}")


def mcts_vs_minimax():
    start = time.time()

    limit = 40
    iterations = 100
    nsims = 50
    depth = 1

    games = 0
    black_wins = 0
    white_wins = 0
    draws = 0
    while True:
        game = Othello()
        while game.state == State.BLACK_TURN or game.state == State.WHITE_TURN:
            # print_board(game.board)
            x, y = mcts_move(game, iterations, nsims) if game.state == State.BLACK_TURN else minimax_move(game, depth)
            game.make_move(x, y)

        if game.state == State.BLACK_WON:
            black_wins += 1
            # print("AI WON")
        elif game.state == State.WHITE_WON:
            white_wins += 1
            # print("AI LOST")
        else:
            draws += 1
            # print("DRAW")
        games += 1
        if games >= limit:
            break

    print(f"BLACK MCTS: {black_wins}")
    print(f"WHITE Minimax: {white_wins}")
    print(f"Draws: {draws}\n")

    games = 0
    black_wins = 0
    white_wins = 0
    draws = 0
    while True:
        game = Othello()
        while game.state == State.BLACK_TURN or game.state == State.WHITE_TURN:
            # print_board(game.board)
            x, y = mcts_move(game, iterations, nsims) if game.state == State.WHITE_TURN else minimax_move(game, depth)
            game.make_move(x, y)

        if game.state == State.BLACK_WON:
            black_wins += 1
            # print("AI LOST")
        elif game.state == State.WHITE_WON:
            white_wins += 1
            # print("AI WON")
        else:
            draws += 1
            # print("DRAW")
        games += 1
        if games >= limit:
            break

    print(f"BLACK Minimax: {black_wins}")
    print(f"WHITE MCTS: {white_wins}")
    print(f"Draws: {draws}\n")

    end = time.time()
    print(f"Time: {end - start}")

print("Running test..")
mcts_vs_minimax()
#mcts_vs_random()
#minimax_vs_random()
#performance_test()
