"""
Constants for the Connect 4 game UI.
Includes colors, dimensions, and re-exports game logic constants.
"""

from connect4_core.config import (
    ROW_COUNT, COLUMN_COUNT, PLAYER_1, PLAYER_2, EMPTY
)

# Colors
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)

# UI Dimensions
SQUARESIZE = 100
WIDTH = COLUMN_COUNT * SQUARESIZE
HEIGHT = (ROW_COUNT + 1) * SQUARESIZE
SIZE = (WIDTH, HEIGHT)
RADIUS = int(SQUARESIZE / 2 - 5)

__all__ = [
    'ROW_COUNT', 'COLUMN_COUNT', 'PLAYER_1', 'PLAYER_2', 'EMPTY',
    'BLUE', 'BLACK', 'RED', 'YELLOW', 'WHITE', 'GRAY',
    'SQUARESIZE', 'WIDTH', 'HEIGHT', 'SIZE', 'RADIUS'
]
