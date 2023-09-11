from core.othello import Othello, State
from core.ui import print_board, user_move, print_score, print_state
from core.mcts import mcts_move
from core.minimax import minimax_move

def main():
    print("Welcome to Othello!")

    game = Othello()
    round = 0
    while game.state == State.BLACK_TURN or game.state == State.WHITE_TURN:
        round += 1
        print(f"\n      Round {round}")
        print_board(game.board)
        print_score(game)
        try:
            move = mcts_move(game, 10) if game.state == State.BLACK_TURN else minimax_move(game, 1)
            print_state(game)
            print(f"      Move: {chr(ord('A') + move[0])}{str(move[1] + 1)}")
            game.make_move(move)
        except IndexError as e:
            print(e)

    print(f"\n      Game Over!")
    print_board(game.board)
    print_score(game)
    if game.state == State.BLACK_WON:
        print("     BLACK won\n")
    elif game.state == State.WHITE_WON:
        print("     WHITE won\n")
    else:
        print("        DRAW\n")


if __name__ == "__main__":
    main()
