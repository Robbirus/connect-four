from connect4_core.board import Board
import connect4_game.constants as CTS

class GameEngine:
    def __init__(self):
        self.board = Board()
        self.game_over = False
        self.turn = CTS.PLAYER_1
        self.winner = None

    def get_valid_moves(self):
        moves = []
        for col in range(7):
            if self.board.is_valid_location(col):
                moves.append(col)
        return moves

    def play_move(self, col):
        row = self.board.get_next_open_row(col)
        self.board.drop_piece(row, col, self.turn)

        if self.board.winning_move(self.turn):
            self.game_over = True
            self.winner = self.turn

        elif self.board.is_full():
            self.game_over = True
            self.winner = 0

        else:
            self.turn = CTS.PLAYER_2 if self.turn == CTS.PLAYER_1 else CTS.PLAYER_1
