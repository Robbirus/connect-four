"""
Controller for Connect Four game with AI opponent.
Allows playing Human vs AI.
"""

import pygame
import sys
import math
import time
from . import constants as CTS
from connect4_core.board import Board
from .ui import Connect4UI


class AIGameController:
    """
    Controller for Human vs AI Connect Four game.
    Handles event processing, AI moves, and game state.
    """

    def __init__(self, ai_agent, human_player=1):
        """
        Initialize the game with an AI opponent.

        Args:
            ai_agent: Connect4Agent instance
            human_player: Which player is human (1 or 2)
        """
        self.board = Board()
        self.ui = Connect4UI()
        self.game_over = False
        self.turn = CTS.PLAYER_1

        self.ai_agent = ai_agent
        self.ai_agent.set_eval_mode()  # Disable exploration
        self.human_player = human_player
        self.ai_player = CTS.PLAYER_2 if human_player == CTS.PLAYER_1 else CTS.PLAYER_1

        # AI thinking delay (milliseconds)
        self.ai_delay = 500

    def run(self):
        """
        Main game loop.
        Handles human moves and AI responses.
        Restarts automatically after each game.
        """
        running = True
        while running:
            self._reset_game()
            self.ui.draw_board(self.board)

            # If AI goes first
            if self.turn == self.ai_player:
                self._ai_move()

            while not self.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        self.game_over = True
                        break
                    if self.handle_event(event):
                        # Human made a move, now AI responds
                        if not self.game_over and self.turn == self.ai_player:
                            pygame.time.wait(self.ai_delay)
                            self._ai_move()
                        break

            if running and self.game_over:
                # Wait for click to restart
                self.ui.show_message("Click to play again", CTS.WHITE)
                if not self._wait_for_restart():
                    running = False

    def _reset_game(self):
        """Reset the game state for a new game."""
        self.board.reset()
        self.game_over = False
        self.turn = CTS.PLAYER_1

    def _wait_for_restart(self):
        """Wait for click to restart or quit. Returns False if quit."""
        pygame.time.wait(1000)  # Small delay before accepting click
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return True

    def handle_event(self, event):
        """
        Process a single Pygame event.
        Returns True if a move was successfully made.
        """
        if event.type == pygame.QUIT:
            sys.exit()

        # Only allow human input on human's turn
        if self.turn != self.human_player:
            return False

        if event.type == pygame.MOUSEMOTION:
            self.handle_mouse_motion(event)

        if event.type == pygame.MOUSEBUTTONDOWN:
            return self.handle_mouse_click(event)

        return False

    def handle_mouse_motion(self, event):
        """Update the UI to show the piece hovering over the current column."""
        if self.turn != self.human_player:
            return

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
        """
        Handle a mouse click to drop a piece.
        """
        pygame.draw.rect(
            self.ui.screen,
            (0, 0, 0),
            (0, 0, self.ui.screen.get_width(), CTS.SQUARESIZE)
        )

        posx = event.pos[0]
        col = int(math.floor(posx / CTS.SQUARESIZE))

        if not self.board.is_valid_location(col):
            return False

        self.process_move(col)
        return True

    def _ai_move(self):
        """
        Let the AI make a move.
        """
        # Show "AI thinking" message
        self.ui.show_message("AI thinking...", CTS.YELLOW)
        pygame.display.update()

        # Get AI's move
        valid_moves = self.board.get_valid_locations()
        col = self.ai_agent.select_move(
            self.board,
            self.ai_player,
            valid_moves,
            training=False
        )

        # Small delay for visual effect
        time.sleep(0.3)

        # Process the move
        self.process_move(col)

    def process_move(self, col):
        """
        Execute a move in the specified column.
        """
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
        """Handle the game state when a player wins."""
        self.ui.draw_board(self.board)

        if self.turn == self.human_player:
            msg = "You win! 🎉"
            color = CTS.RED if self.turn == CTS.PLAYER_1 else CTS.YELLOW
        else:
            msg = "AI wins!"
            color = CTS.RED if self.turn == CTS.PLAYER_1 else CTS.YELLOW

        self.ui.show_message(msg, color)
        self.game_over = True

    def handle_draw(self):
        """Handle the game state when the board is full (draw)."""
        self.ui.draw_board(self.board)
        self.ui.show_message("Draw!", CTS.WHITE)
        self.game_over = True

    def switch_turn(self):
        """Switch the active player."""
        self.turn = CTS.PLAYER_2 if self.turn == CTS.PLAYER_1 else CTS.PLAYER_1
        self.ui.draw_board(self.board)
