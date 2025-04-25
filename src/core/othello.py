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

    def make_move(self, move: tuple[int, int]) -> None:
        """Makes a move at the given position and updates the game state."""
        if self.state not in (State.BLACK_TURN, State.WHITE_TURN):
            raise ValueError("Can't make move: Game is over")
        if self.board[move[1]][move[0]] != Cell.VALID:
            raise IndexError(f"Can't make move: Position invalid {move}")

        # reverse cells and update state
        reverse = Cell.BLACK if self.state == State.BLACK_TURN else Cell.WHITE  # color of current player
        self.board[move[1]][move[0]] = reverse
        for cell in self._flipped_cells(move):
            self.board[cell[1]][cell[0]] = reverse
        self._update_state()

    def get_valid_moves(self) -> list[tuple[int, int]]:
        """Returns a list of valid moves for the current turn."""
        if self.state not in (State.BLACK_TURN, State.WHITE_TURN):
            return []
        return [(x, y) for x in range(8) for y in range(8) if self.board[y][x] == Cell.VALID]

    def _update_state(self) -> None:
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

        # switch turn and update valid cells
        self.state = State.WHITE_TURN if self.state == State.BLACK_TURN else State.BLACK_TURN
        self._update_valid_cells()
        if len(self.get_valid_moves()) == 0:
            # print("No valid moves. Skipping turn.")  # for debug
            self.state = State.WHITE_TURN if self.state == State.BLACK_TURN else State.BLACK_TURN
            self._update_valid_cells()
            if len(self.get_valid_moves()) == 0:
                # print("No valid moves even for the other player. Game over.")  # for debug
                if self.black_score > self.white_score:
                    self.state = State.BLACK_WON
                elif self.black_score < self.white_score:
                    self.state = State.WHITE_WON
                else:
                    self.state = State.DRAW

    def _is_full(self) -> bool:
        for y in range(8):
            for x in range(8):
                if self.board[y][x] in (Cell.EMPTY, Cell.VALID):
                    return False
        return True

    def _update_valid_cells(self) -> None:
        for y in range(8):
            for x in range(8):
                if self.board[y][x] == Cell.VALID:
                    self.board[y][x] = Cell.EMPTY
                if self.board[y][x] == Cell.EMPTY and self._flipped_cells((x, y)) != []:
                    self.board[y][x] = Cell.VALID

    def _flipped_cells(self, move: tuple[int, int]) -> list[tuple[int, int]]:
        player = Cell.BLACK if self.state == State.BLACK_TURN else Cell.WHITE
        opponent = Cell.WHITE if self.state == State.BLACK_TURN else Cell.BLACK
        flipped = []
        x, y = move[0], move[1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            flipped += self._flipped_cells_in_direction(x, y, dx, dy, player, opponent)
        return flipped

    def _flipped_cells_in_direction(self, x: int, y: int, dx: int, dy: int, player: Cell, opponent: Cell) -> list[tuple[int, int]]:
        flipped = []
        x, y = x + dx, y + dy
        while 0 <= x < 8 and 0 <= y < 8 and self.board[y][x] == opponent:
            flipped.append((x, y))
            x, y = x + dx, y + dy
        if not (0 <= x < 8 and 0 <= y < 8) or self.board[y][x] != player:
            return []
        return flipped
