import numpy as np
from . import config as CFG

"""Module for the Connect 4 game board logic."""


class Board:
    """
    Represents the Connect 4 game board and its logic.
    Handles the grid state, move validation, and win detection.
    """

    def __init__(self):
        """Initialize the board.
         Set up the grid dimensions and reset the board state."""
        self.rows = CFG.ROW_COUNT
        self.cols = CFG.COLUMN_COUNT
        self.reset()

    def reset(self):
        """Reset the game grid to an empty state (zeros)."""
        self.grid = np.zeros((self.rows, self.cols), dtype=int)

    def is_valid_location(self, col):
        """Check if a column has at least one empty slot (top row is empty)."""
        return self.grid[self.rows - 1][col] == 0

    def get_next_open_row(self, col):
        """Find the first empty row in the specified column."""
        for r in range(self.rows):

            if self.grid[r][col] == 0:
                return r

        return None

    def drop_piece(self, row, col, piece):
        """Place a piece on the board at the specified location."""
        self.grid[row][col] = piece

    def winning_move(self, piece):
        """Check if the last move created a winning condition (4 in a row)."""
        return (self.check_horizontal(piece) or
                self.check_vertical(piece) or
                self.check_positive_diagonal(piece) or
                self.check_negative_diagonal(piece))

    def check_horizontal(self, piece):
        """Check for 4 consecutive pieces horizontally."""
        for c in range(self.cols - 3):
            for r in range(self.rows):

                if (self.grid[r][c] == piece and
                        self.grid[r][c + 1] == piece and
                        self.grid[r][c + 2] == piece and
                        self.grid[r][c + 3] == piece):
                    return True

        return False

    def check_vertical(self, piece):
        """Check for 4 consecutive pieces vertically."""
        for c in range(self.cols):
            for r in range(self.rows - 3):

                if (self.grid[r][c] == piece and
                        self.grid[r + 1][c] == piece and
                        self.grid[r + 2][c] == piece and
                        self.grid[r + 3][c] == piece):
                    return True

        return False

    def check_positive_diagonal(self, piece):
        """Check for 4 consecutive pieces on positive diagonals."""
        for c in range(self.cols - 3):
            for r in range(self.rows - 3):

                if (self.grid[r][c] == piece and
                        self.grid[r + 1][c + 1] == piece and
                        self.grid[r + 2][c + 2] == piece and
                        self.grid[r + 3][c + 3] == piece):
                    return True

        return False

    def check_negative_diagonal(self, piece):
        """Check for 4 consecutive pieces on negative diagonals."""
        for c in range(self.cols - 3):
            for r in range(3, self.rows):

                if (self.grid[r][c] == piece and
                        self.grid[r - 1][c + 1] == piece and
                        self.grid[r - 2][c + 2] == piece and
                        self.grid[r - 3][c + 3] == piece):
                    return True

        return False

    def is_full(self):
        """Check if the board is completely full (draw)."""
        return not np.any(self.grid == 0)

    def get_board(self):
        """Return the current grid state."""
        return self.grid

    def get_valid_locations(self):
        """Return a list of all columns where a move is possible."""
        return [c for c in range(self.cols) if self.is_valid_location(c)]

    def copy(self):
        """Create a deep copy of the board (useful for AI simulations)."""
        new_board = Board()
        new_board.grid = np.copy(self.grid)
        return new_board
