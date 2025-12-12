import pygame
import sys
from . import constants as CTS


"""UI module for Connect Four game using Pygame."""


class Connect4UI:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(CTS.SIZE)
        pygame.display.set_caption("Connect Four")
        self.font = pygame.font.SysFont("monospace", 75)

    def draw_board(self, board):
        self.draw_grid()
        self.draw_pieces(board)
        pygame.display.update()

    def draw_grid(self):
        for c in range(CTS.COLUMN_COUNT):
            for r in range(CTS.ROW_COUNT):
                pygame.draw.rect(
                    self.screen,
                    CTS.BLUE,
                    (c * CTS.SQUARESIZE,
                     r * CTS.SQUARESIZE + CTS.SQUARESIZE,
                     CTS.SQUARESIZE, CTS.SQUARESIZE))
                pygame.draw.circle(
                    self.screen,
                    CTS.BLACK,
                    (int(c * CTS.SQUARESIZE + CTS.SQUARESIZE / 2),
                     int(r * CTS.SQUARESIZE +
                         CTS.SQUARESIZE + CTS.SQUARESIZE / 2)),
                    CTS.RADIUS)

    def draw_pieces(self, board):
        grid = board.get_board()
        for c in range(CTS.COLUMN_COUNT):
            for r in range(CTS.ROW_COUNT):
                if grid[r][c] == CTS.PLAYER_1:
                    pygame.draw.circle(
                        self.screen,
                        CTS.RED,
                        (int(c * CTS.SQUARESIZE + CTS.SQUARESIZE / 2),
                         CTS.HEIGHT -
                         int(r * CTS.SQUARESIZE + CTS.SQUARESIZE / 2)),
                        CTS.RADIUS)
                elif grid[r][c] == CTS.PLAYER_2:
                    pygame.draw.circle(
                        self.screen,
                        CTS.YELLOW,
                        (int(c * CTS.SQUARESIZE + CTS.SQUARESIZE / 2),
                         CTS.HEIGHT -
                         int(r * CTS.SQUARESIZE + CTS.SQUARESIZE / 2)),
                        CTS.RADIUS)

    def draw_hover_piece(self, x_pos, turn):
        # Clear the top row area
        pygame.draw.rect(self.screen, CTS.BLACK,
                         (0, 0, CTS.SIZE[0], CTS.SQUARESIZE))

        color = CTS.RED if turn == CTS.PLAYER_1 else CTS.YELLOW
        pygame.draw.circle(self.screen, color,
                           (x_pos, int(CTS.SQUARESIZE / 2)), CTS.RADIUS)
        pygame.display.update()

    def show_message(self, message, color=CTS.RED):
        label = self.font.render(message, 1, color)
        # Center the message
        text_rect = label.get_rect(
            center=(CTS.SIZE[0] / 2,
                    CTS.SQUARESIZE / 2))
        # Clear top area first
        pygame.draw.rect(self.screen, CTS.BLACK,
                         (0, 0, CTS.SIZE[0], CTS.SQUARESIZE))
        self.screen.blit(label, text_rect)
        pygame.display.update()

    def wait_for_click(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return

    def quit(self):
        pygame.quit()
