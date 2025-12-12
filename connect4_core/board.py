import numpy as np
from . import config as CFG

"""Module for the Connect 4 game board logic."""


class Board:
    def __init__(self):
        self.rows = CFG.ROW_COUNT
        self.cols = CFG.COLUMN_COUNT
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.rows, self.cols), dtype=int)

    def is_valid_location(self, col):
        return self.grid[self.rows - 1][col] == 0

    def get_next_open_row(self, col):
        for r in range(self.rows):
            if self.grid[r][col] == 0:
                return r
        return None

    def drop_piece(self, row, col, piece):
        self.grid[row][col] = piece

    def winning_move(self, piece):
        return (self.check_horizontal(piece) or
                self.check_vertical(piece) or
                self.check_positive_diagonal(piece) or
                self.check_negative_diagonal(piece))

    def check_horizontal(self, piece):
        for c in range(self.cols - 3):
            for r in range(self.rows):
                if (self.grid[r][c] == piece and
                        self.grid[r][c + 1] == piece and
                        self.grid[r][c + 2] == piece and
                        self.grid[r][c + 3] == piece):
                    return True
        return False

    def check_vertical(self, piece):
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if (self.grid[r][c] == piece and
                        self.grid[r + 1][c] == piece and
                        self.grid[r + 2][c] == piece and
                        self.grid[r + 3][c] == piece):
                    return True
        return False

    def check_positive_diagonal(self, piece):
        for c in range(self.cols - 3):
            for r in range(self.rows - 3):
                if (self.grid[r][c] == piece and
                        self.grid[r + 1][c + 1] == piece and
                        self.grid[r + 2][c + 2] == piece and
                        self.grid[r + 3][c + 3] == piece):
                    return True
        return False

    def check_negative_diagonal(self, piece):
        for c in range(self.cols - 3):
            for r in range(3, self.rows):
                if (self.grid[r][c] == piece and
                        self.grid[r - 1][c + 1] == piece and
                        self.grid[r - 2][c + 2] == piece and
                        self.grid[r - 3][c + 3] == piece):
                    return True
        return False

    def is_full(self):
        return not np.any(self.grid == 0)

    def get_board(self):
        return self.grid

    def get_valid_locations(self):
        return [c for c in range(self.cols) if self.is_valid_location(c)]

    def copy(self):
        new_board = Board()
        new_board.grid = np.copy(self.grid)
        return new_board
