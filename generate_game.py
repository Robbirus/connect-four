# generate_games.py
import sqlite3
import random
import json
from datetime import datetime

ROWS, COLS = 6, 7

def create_schema(db_path="games.db"):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.executescript(open("schema.sql").read())
        # insert two default players
        cur.execute("INSERT INTO players(name) VALUES(?)", ("player1",))
        cur.execute("INSERT INTO players(name) VALUES(?)", ("player2",))
        conn.commit()

def new_board():
    return [[0]*COLS for _ in range(ROWS)]

def can_play(board, col):
    return board[0][col] == 0

def play_move(board, col, player_id):
    for r in range(ROWS-1, -1, -1):
        if board[r][col] == 0:
            board[r][col] = player_id
            return True, r, col
    return False, None, None

def is_win(board, player):
    # horizontal, vertical, diag checks
    for r in range(ROWS):
        for c in range(COLS-3):
            if all(board[r][c+i] == player for i in range(4)): return True
    for c in range(COLS):
        for r in range(ROWS-3):
            if all(board[r+i][c] == player for i in range(4)): return True
    for r in range(ROWS-3):
        for c in range(COLS-3):
            if all(board[r+i][c+i] == player for i in range(4)): return True
    for r in range(3, ROWS):
        for c in range(COLS-3):
            if all(board[r-i][c+i] == player for i in range(4)): return True
    return False

def available_moves(board):
    return [c for c in range(COLS) if can_play(board, c)]

def random_agent(board, player_id):
    return random.choice(available_moves(board))

def simulate_one_game(db_path="games.db"):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        board = new_board()
        moves_list = []
        cur.execute("INSERT INTO games(date_played, winner, num_moves) VALUES(?,?,?)", (datetime.utcnow().isoformat(), None, 0))
        game_id = cur.lastrowid
        current_player = 1  # 1 or 2 corresponds to player ids in players table (simple mapping)
        move_index = 0
        winner = None

        while True:
            moves = available_moves(board)
            if not moves:
                break
            col = random_agent(board, current_player)
            play_move(board, col, current_player)
            # save move with previous state
            state_json = json.dumps(board)  # simple - in training you prefer np arrays
            cur.execute("INSERT INTO moves(game_id, move_index, player_id, column, board_state) VALUES(?,?,?,?,?)",
                        (game_id, move_index, current_player, col, state_json))
            move_index += 1
            if is_win(board, current_player):
                winner = current_player
                break
            current_player = 2 if current_player == 1 else 1

        cur.execute("UPDATE games SET winner=?, num_moves=? WHERE id=?", (winner, move_index, game_id))
        conn.commit()

if __name__ == "__main__":
    db = "games.db"
    create_schema(db)
    for _ in range(5000):  # génère 5000 parties
        simulate_one_game(db)
    print("Done generating games")
