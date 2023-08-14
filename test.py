import time
import random
from core.othello import Othello, State
from core.minimax import minimax_move
from core.mcts import mcts_move
from colorama import Fore, Style
from typing import Callable


def random_move(game: Othello) -> tuple[int, int]:
    moves = game.get_valid_moves()
    x, y = moves[random.randint(0, len(moves) - 1)]
    return x, y


def benchmark_game(BLACK_AI: Callable[[Othello], tuple[int, int]], WHITE_AI: Callable[[Othello], tuple[int, int]], games_count: int) -> None:
    black_wins = 0
    white_wins = 0
    draws = 0
    start_time = time.time()
    for i in range(games_count):
        current_time = time.time() - start_time
        print(f"  elapsed time: {current_time:.2f}s, game: {i + 1}/{games_count}...", end="\r")

        game = Othello()
        while game.state == State.BLACK_TURN or game.state == State.WHITE_TURN:
            try:
                x, y = BLACK_AI(game) if game.state == State.BLACK_TURN else WHITE_AI(game)
                game.make_move(x, y)
            except IndexError as e:
                print(e)

        if game.state == State.BLACK_WON:
            black_wins += 1
        elif game.state == State.WHITE_WON:
            white_wins += 1
        else:
            draws += 1

    print(f"  elapsed time: {time.time() - start_time:.2f}s                     ")
    print(f"BLACK win: {black_wins}, BLACK winrate: {black_wins / games_count * 100:.1f}%")
    print(f"WHITE win: {white_wins}, WHITE winrate: {white_wins / games_count * 100:.1f}%")
    print(f"draws: {draws}, draw rate: {draws / games_count * 100:.1f}%\n")


def test() -> None:
    GAMES_COUNT = 10
    MINIMAX_DEPTH = 2
    MCTS_SIMULATIONS = 10

    start_time = time.time()
    print()

    print(f"{Fore.BLUE}Random vs. Random:{Style.RESET_ALL}")
    benchmark_game(random_move, random_move, GAMES_COUNT)

    print(f"{Fore.BLUE}BLACK Minimax vs. WHITE Random:{Style.RESET_ALL}")
    benchmark_game(lambda game: minimax_move(game, 2), random_move, GAMES_COUNT)

    print(f"{Fore.BLUE}WHITE Minimax vs. BLACK Random:{Style.RESET_ALL}")
    benchmark_game(random_move, lambda game: minimax_move(game, 2), GAMES_COUNT)

    print(f"{Fore.BLUE}Minimax vs. Minimax:{Style.RESET_ALL}")
    benchmark_game(lambda game: minimax_move(game, MINIMAX_DEPTH), lambda game: minimax_move(game, MINIMAX_DEPTH), GAMES_COUNT)

    print(f"{Fore.BLUE}BLACK MCTS vs. WHITE Random:{Style.RESET_ALL}")
    benchmark_game(lambda game: mcts_move(game, MCTS_SIMULATIONS), random_move, GAMES_COUNT)

    print(f"{Fore.BLUE}WHITE MCTS vs. BLACK Random:{Style.RESET_ALL}")
    benchmark_game(random_move, lambda game: mcts_move(game, MCTS_SIMULATIONS), GAMES_COUNT)

    print(f"{Fore.BLUE}MCTS vs. MCTS:{Style.RESET_ALL}")
    benchmark_game(lambda game: mcts_move(game, MCTS_SIMULATIONS), lambda game: mcts_move(game, MCTS_SIMULATIONS), GAMES_COUNT)

    print(f"{Fore.BLUE}BLACK Minimax vs. WHITE MCTS:{Style.RESET_ALL}")
    benchmark_game(lambda game: minimax_move(game, MINIMAX_DEPTH), lambda game: mcts_move(game, MCTS_SIMULATIONS), GAMES_COUNT)

    print(f"{Fore.BLUE}WHITE Minimax vs. BLACK MCTS:{Style.RESET_ALL}")
    benchmark_game(lambda game: mcts_move(game, MCTS_SIMULATIONS), lambda game: minimax_move(game, MINIMAX_DEPTH), GAMES_COUNT)

    print("Total time elapsed:", time.time() - start_time)


if __name__ == "__main__":
    test()
