import time
from typing import Callable

import numpy as np
from colorama import Fore, Style
from core_numba.mcts import mcts_move
from core_numba.minimax import minimax_move
from core_numba.othello import (
    STATE_BLACK_TURN,
    STATE_BLACK_WON,
    STATE_DRAW,
    STATE_WHITE_TURN,
    STATE_WHITE_WON,
    get_valid_moves,
    init_game,
    make_move,
)
from numba import njit

GAMES_COUNT = 20
MINIMAX_DEPTH = 2
MCTS_SIMULATIONS = 20


def run_benchmarks() -> None:
    print(f"{Fore.MAGENTA}Running benchmarks...{Style.RESET_ALL}\n")
    start_time = time.time()

    print(f"{Fore.BLUE}Random vs Random:{Style.RESET_ALL}")
    benchmark_game(random_move_wrapper, random_move_wrapper)

    print(f"{Fore.BLUE}BLACK Minimax vs WHITE Random:{Style.RESET_ALL}")
    benchmark_game(minimax_move_wrapper, random_move_wrapper)

    print(f"{Fore.BLUE}WHITE Minimax vs BLACK Random:{Style.RESET_ALL}")
    benchmark_game(random_move_wrapper, minimax_move_wrapper)

    print(f"{Fore.BLUE}Minimax vs Minimax:{Style.RESET_ALL}")
    benchmark_game(minimax_move_wrapper, minimax_move_wrapper)

    print(f"{Fore.BLUE}BLACK MCTS vs WHITE Random:{Style.RESET_ALL}")
    benchmark_game(mcts_move_wrapper, random_move_wrapper)

    print(f"{Fore.BLUE}WHITE MCTS vs BLACK Random:{Style.RESET_ALL}")
    benchmark_game(random_move_wrapper, mcts_move_wrapper)

    print(f"{Fore.BLUE}MCTS vs MCTS:{Style.RESET_ALL}")
    benchmark_game(mcts_move_wrapper, mcts_move_wrapper)

    print(f"{Fore.BLUE}BLACK Minimax vs WHITE MCTS:{Style.RESET_ALL}")
    benchmark_game(minimax_move_wrapper, mcts_move_wrapper)

    print(f"{Fore.BLUE}WHITE Minimax vs BLACK MCTS:{Style.RESET_ALL}")
    benchmark_game(mcts_move_wrapper, minimax_move_wrapper)

    print(f"{Fore.MAGENTA}Total time elapsed: {time.time() - start_time:.2f}{Style.RESET_ALL}")


def benchmark_game(
    BLACK_AI: Callable[[np.ndarray, np.int32, np.int32, np.int32], np.ndarray],
    WHITE_AI: Callable[[np.ndarray, np.int32, np.int32, np.int32], np.ndarray],
) -> None:
    black_wins = 0
    white_wins = 0
    draws = 0
    start_time = time.time()

    for i in range(GAMES_COUNT):
        current_time = time.time() - start_time
        print(f"  game: {i + 1}/{GAMES_COUNT}  elapsed time: {current_time:.2f}s", end="\r")

        board, black_score, white_score, state = init_game()
        while state in (STATE_BLACK_TURN, STATE_WHITE_TURN):
            move = (
                BLACK_AI(board, black_score, white_score, state)
                if state == STATE_BLACK_TURN
                else WHITE_AI(board, black_score, white_score, state)
            )
            if move[0] == -1 and move[1] == -1:  # No valid moves
                board, black_score, white_score, state, _ = make_move(board, black_score, white_score, state, 0, 0)
                continue
            board, black_score, white_score, state, success = make_move(
                board, black_score, white_score, state, move[0], move[1]
            )
            if not success:
                print(f"Invalid move attempted: {move.tolist()}")
                break

        if state == STATE_BLACK_WON:
            black_wins += 1
        elif state == STATE_WHITE_WON:
            white_wins += 1
        else:
            draws += 1

    print(f"  elapsed time: {time.time() - start_time:.2f}s                     ")
    print(f"    BLACK wins: {black_wins} {black_wins / GAMES_COUNT * 100:.0f}%")
    print(f"    WHITE wins: {white_wins} {white_wins / GAMES_COUNT * 100:.0f}%")
    print(f"         draws: {draws} {draws / GAMES_COUNT * 100:.0f}%\n")


@njit
def random_move(
    board: np.ndarray,
    black_score: np.int32,
    white_score: np.int32,
    state: np.int32,
):
    moves = get_valid_moves(board, state)
    if moves.shape[0] == 0:
        return np.array([-1, -1], dtype=np.int32)
    move_idx = np.random.randint(0, moves.shape[0])
    return moves[move_idx]


def random_move_wrapper(
    board: np.ndarray,
    black_score: np.int32,
    white_score: np.int32,
    state: np.int32,
):
    return random_move(board, black_score, white_score, state)


def minimax_move_wrapper(
    board: np.ndarray,
    black_score: np.int32,
    white_score: np.int32,
    state: np.int32,
):
    return minimax_move(board, black_score, white_score, state, MINIMAX_DEPTH)


def mcts_move_wrapper(
    board: np.ndarray,
    black_score: np.int32,
    white_score: np.int32,
    state: np.int32,
):
    return mcts_move(board, black_score, white_score, state, MCTS_SIMULATIONS)


if __name__ == "__main__":
    run_benchmarks()
