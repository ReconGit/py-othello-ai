from enum import Enum


class Cell(Enum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2
    VALID = 3


class State(Enum):
    BLACK_TURN = 1
    WHITE_TURN = 2
    BLACK_WON = 3
    WHITE_WON = 4
    DRAW = 5


class Othello:
    """Represents a game of Othello."""

    def __init__(self) -> None:
        # initialize the starting board
        self.black_score = 2
        self.white_score = 2
        self.board = [[Cell.EMPTY for _ in range(8)] for _ in range(8)]
        self.board[3][3] = Cell.WHITE
        self.board[3][4] = Cell.BLACK
        self.board[4][3] = Cell.BLACK
        self.board[4][4] = Cell.WHITE
        self.board[2][3] = Cell.VALID
        self.board[3][2] = Cell.VALID
        self.board[4][5] = Cell.VALID
        self.board[5][4] = Cell.VALID
        self.state = State.BLACK_TURN

    def make_move(self, x: int, y: int) -> None:
        # sanity checks
        if self.state != State.BLACK_TURN and self.state != State.WHITE_TURN:
            raise ValueError("Cannot make move because the game is over.")
        if self.board[y][x] != Cell.VALID:
            raise IndexError("Position is not valid.")

        reverse = Cell.BLACK if self.state == State.BLACK_TURN else Cell.WHITE  # color of current player
        self.board[y][x] = reverse
        # reverse cells
        for move in self._flipped_cells(x, y):
            self.board[move[1]][move[0]] = reverse
        self._update_state()

    def get_valid_moves(self) -> list[tuple[int, int]]:
        return [(x, y) for x in range(8) for y in range(8) if self.board[y][x] == Cell.VALID]

    def _is_full(self) -> bool:
        for y in range(8):
            for x in range(8):
                if self.board[y][x] in (Cell.EMPTY, Cell.VALID):
                    return False
        return True

    def _update_state(self) -> None:
        # update score
        self.black_score = sum(row.count(Cell.BLACK) for row in self.board)
        self.white_score = sum(row.count(Cell.WHITE) for row in self.board)

        # check if game is over
        if self._is_full() or self.black_score == 0 or self.white_score == 0:
            if self.black_score > self.white_score:
                self.state = State.BLACK_WON
            elif self.black_score < self.white_score:
                self.state = State.WHITE_WON
            else:
                self.state = State.DRAW
            return

        # switch turn
        self.state = State.WHITE_TURN if self.state == State.BLACK_TURN else State.BLACK_TURN
        self._update_valid_moves()
        if len(self.get_valid_moves()) == 0:
            # print("No valid moves. Skipping turn.")  # for debug
            self.state = State.WHITE_TURN if self.state == State.BLACK_TURN else State.BLACK_TURN
            self._update_valid_moves()
            if len(self.get_valid_moves()) == 0:
                # print("No valid moves even for the other player. Game over.")  # for debug
                if self.black_score > self.white_score:
                    self.state = State.BLACK_WON
                elif self.black_score < self.white_score:
                    self.state = State.WHITE_WON
                else:
                    self.state = State.DRAW

    def _update_valid_moves(self) -> None:
        for y in range(8):
            for x in range(8):
                if self.board[y][x] == Cell.VALID:
                    self.board[y][x] = Cell.EMPTY
                if self.board[y][x] == Cell.EMPTY and self._flipped_cells(x, y) != []:
                    self.board[y][x] = Cell.VALID

    def _flipped_cells(self, x: int, y: int) -> list[tuple[int, int]]:
        player = Cell.BLACK if self.state == State.BLACK_TURN else Cell.WHITE
        opponent = Cell.WHITE if self.state == State.BLACK_TURN else Cell.BLACK
        flipped = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            flipped += self._flipped_cells_in_direction(x, y, dx, dy, player, opponent)
        return flipped

    def _flipped_cells_in_direction(self, x: int, y: int, dx: int, dy: int, player: Cell, opponent: Cell) -> list[tuple[int, int]]:
        flipped = []
        x, y = x + dx, y + dy
        while 0 <= x <= 7 and 0 <= y <= 7 and self.board[y][x] == opponent:
            flipped.append((x, y))
            x, y = x + dx, y + dy
        if not (0 <= x <= 7 and 0 <= y <= 7) or self.board[y][x] != player:
            return []
        return flipped
