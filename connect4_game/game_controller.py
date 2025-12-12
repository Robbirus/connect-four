import pygame
import sys
import math
from . import constants as CTS
from connect4_core.board import Board
from .ui import Connect4UI

"""Controller module for Connect Four game."""


class GameController:
    def __init__(self):
        self.board = Board()
        self.ui = Connect4UI()
        self.game_over = False
        self.turn = CTS.PLAYER_1

    def run(self):
        self.ui.draw_board(self.board)

        while not self.game_over:
            for event in pygame.event.get():
                if self.handle_event(event):
                    # If a move was made, ignore other events in this batch
                    # to prevent buffered clicks from triggering multiple moves
                    break

        if self.game_over:
            self.ui.wait_for_click()

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEMOTION:
            self.handle_mouse_motion(event)

        if event.type == pygame.MOUSEBUTTONDOWN:
            return self.handle_mouse_click(event)

        return False

    def handle_mouse_motion(self, event):
        pygame.draw.rect(
            self.ui.screen,
            (0, 0, 0),
            (0, 0, self.ui.screen.get_width(), CTS.SQUARESIZE)
        )

        posx = event.pos[0]
        col = int(math.floor(posx / CTS.SQUARESIZE))
        col = max(0, min(col, CTS.COLUMN_COUNT - 1))
        snapped_x = int(col * CTS.SQUARESIZE + CTS.SQUARESIZE / 2)

        self.ui.draw_hover_piece(snapped_x, self.turn)

    def handle_mouse_click(self, event):
        pygame.draw.rect(
            self.ui.screen,
            (0, 0, 0),
            (0, 0, self.ui.screen.get_width(), CTS.SQUARESIZE)
        )

        posx = event.pos[0]
        col = int(math.floor(posx / CTS.SQUARESIZE))

        if self.board.is_valid_location(col):
            self.process_move(col)

            # Update hover piece after move if game continues
            if not self.game_over:
                # Recalculate snapped position for the new turn
                col = int(math.floor(posx / CTS.SQUARESIZE))
                col = max(0, min(col, CTS.COLUMN_COUNT - 1))
                snapped_x = int(col * CTS.SQUARESIZE + CTS.SQUARESIZE / 2)
                self.ui.draw_hover_piece(snapped_x, self.turn)

            return True

        return False

    def process_move(self, col):
        row = self.board.get_next_open_row(col)
        self.ui.animate_drop(self.board, row, col, self.turn)
        self.board.drop_piece(row, col, self.turn)

        if self.board.winning_move(self.turn):
            self.handle_win()
        elif self.board.is_full():
            self.handle_draw()
        else:
            self.switch_turn()

    def handle_win(self):
        self.ui.draw_board(self.board)
        msg = f"Player {self.turn} wins!!"
        color = CTS.RED if self.turn == CTS.PLAYER_1 else CTS.YELLOW
        self.ui.show_message(msg, color)
        self.game_over = True

    def handle_draw(self):
        self.ui.draw_board(self.board)
        self.ui.show_message("Draw!", (255, 255, 255))
        self.game_over = True

    def switch_turn(self):
        self.turn = CTS.PLAYER_2 if self.turn == CTS.PLAYER_1 else CTS.PLAYER_1
        self.ui.draw_board(self.board)
