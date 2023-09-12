import time
import random
from core.othello import Othello, State
from core.minimax import minimax_move
from core.mcts import mcts_move
from colorama import Fore, Style
from typing import Callable

GAMES_COUNT = 10
MINIMAX_DEPTH = 1
MCTS_SIMULATIONS = 10


def run_benchmarks() -> None:
    print(f"{Fore.MAGENTA}Running benchmarks...{Style.RESET_ALL}\n")
    start_time = time.time()

    print(f"{Fore.BLUE}Random vs Random:{Style.RESET_ALL}")
    benchmark_game(random_move, random_move)

    print(f"{Fore.BLUE}BLACK Minimax vs WHITE Random:{Style.RESET_ALL}")
    benchmark_game(lambda game: minimax_move(game, 2), random_move)

    print(f"{Fore.BLUE}WHITE Minimax vs BLACK Random:{Style.RESET_ALL}")
    benchmark_game(random_move, lambda game: minimax_move(game, 2))

    print(f"{Fore.BLUE}Minimax vs Minimax:{Style.RESET_ALL}")
    benchmark_game(lambda game: minimax_move(game, MINIMAX_DEPTH), lambda game: minimax_move(game, MINIMAX_DEPTH))

    print(f"{Fore.BLUE}BLACK MCTS vs WHITE Random:{Style.RESET_ALL}")
    benchmark_game(lambda game: mcts_move(game, MCTS_SIMULATIONS), random_move)

    print(f"{Fore.BLUE}WHITE MCTS vs BLACK Random:{Style.RESET_ALL}")
    benchmark_game(random_move, lambda game: mcts_move(game, MCTS_SIMULATIONS))

    print(f"{Fore.BLUE}MCTS vs MCTS:{Style.RESET_ALL}")
    benchmark_game(lambda game: mcts_move(game, MCTS_SIMULATIONS), lambda game: mcts_move(game, MCTS_SIMULATIONS))

    print(f"{Fore.BLUE}BLACK Minimax vs WHITE MCTS:{Style.RESET_ALL}")
    benchmark_game(lambda game: minimax_move(game, MINIMAX_DEPTH), lambda game: mcts_move(game, MCTS_SIMULATIONS))

    print(f"{Fore.BLUE}WHITE Minimax vs BLACK MCTS:{Style.RESET_ALL}")
    benchmark_game(lambda game: mcts_move(game, MCTS_SIMULATIONS), lambda game: minimax_move(game, MINIMAX_DEPTH))

    print(f"{Fore.MAGENTA}Total time elapsed: {time.time() - start_time:.2f}{Style.RESET_ALL}")


def benchmark_game(BLACK_AI: Callable[[Othello], tuple[int, int]], WHITE_AI: Callable[[Othello], tuple[int, int]]) -> None:
    black_wins = 0
    white_wins = 0
    draws = 0
    start_time = time.time()
    for i in range(GAMES_COUNT):
        current_time = time.time() - start_time
        print(f"  game: {i + 1}/{GAMES_COUNT}  elapsed time: {current_time:.2f}s", end="\r")

        game = Othello()
        while game.state == State.BLACK_TURN or game.state == State.WHITE_TURN:
            try:
                move = BLACK_AI(game) if game.state == State.BLACK_TURN else WHITE_AI(game)
                game.make_move(move)
            except IndexError as e:
                print(e)

        if game.state == State.BLACK_WON:
            black_wins += 1
        elif game.state == State.WHITE_WON:
            white_wins += 1
        else:
            draws += 1

    print(f"  elapsed time: {time.time() - start_time:.2f}s                     ")
    print(f"    BLACK wins: {black_wins} {black_wins / GAMES_COUNT * 100:.1f}%")
    print(f"    WHITE wins: {white_wins} {white_wins / GAMES_COUNT * 100:.1f}%")
    print(f"         draws: {draws} {draws / GAMES_COUNT * 100:.1f}%\n")


def random_move(game: Othello) -> tuple[int, int]:
    moves = game.get_valid_moves()
    move = moves[random.randint(0, len(moves) - 1)]
    return move


if __name__ == "__main__":
    run_benchmarks()
