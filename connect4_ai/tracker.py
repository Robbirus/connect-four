"""
Training Tracker for Connect 4 AI.
Handles all database operations for traceability:
sessions, snapshots, games, moves, evaluations.
"""

import json
import os
import sqlite3
import numpy as np


DB_NAME = "games.db"
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "schema.sql")


class Tracker:
    """
    Exposes an API to record training data into SQLite.

    Usage:
        tracker = Tracker()
        sid = tracker.start_session(hyperparameters={...})
        gid = tracker.start_game(sid, game_type='self_play')
        tracker.record_move(gid, move_index=0, player_id=1, column=3, ...)
        tracker.end_game(gid, winner=1, num_moves=12, reward_p1=1.0, reward_p2=-1.0)
        tracker.record_snapshot(sid, game_number=100, epsilon=0.5, ...)
        tracker.end_session(sid, num_games=1000)
        tracker.close()
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_NAME
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_db()
        self._ensure_ai_players()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_db(self):
        """Create tables if they don't exist."""
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = f.read()
        # Use executescript so CREATE TABLE IF NOT EXISTS isn't needed;
        # we catch the "already exists" error gracefully.
        statements = schema.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
        self._conn.executescript(statements)

    def _ensure_ai_players(self):
        """Make sure player 1 (AI-P1) and player 2 (AI-P2) exist."""
        cur = self._conn.cursor()
        for pid, name in [(1, "AI-P1"), (2, "AI-P2")]:
            cur.execute("SELECT id FROM players WHERE id = ?", (pid,))
            if cur.fetchone() is None:
                cur.execute("INSERT INTO players (id, name) VALUES (?, ?)", (pid, name))
        self._conn.commit()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def start_session(self, hyperparameters=None):
        """Start a new training session. Returns the session id."""
        hp_json = json.dumps(hyperparameters) if hyperparameters else None
        cur = self._conn.execute(
            "INSERT INTO training_sessions (hyperparameters) VALUES (?)",
            (hp_json,),
        )
        self._conn.commit()
        return cur.lastrowid

    def end_session(self, session_id, num_games):
        """Mark a training session as finished."""
        self._conn.execute(
            "UPDATE training_sessions SET ended_at = datetime('now'), num_games = ? WHERE id = ?",
            (num_games, session_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Snapshots (courbes d'apprentissage)
    # ------------------------------------------------------------------

    def record_snapshot(self, session_id, game_number, *,
                        epsilon=None, avg_loss=None,
                        p1_win_rate=None, p2_win_rate=None, draw_rate=None,
                        avg_moves_per_game=None, avg_reward=None):
        """Insert one training snapshot row."""
        self._conn.execute(
            """INSERT INTO training_snapshots
               (session_id, game_number, epsilon, avg_loss,
                p1_win_rate, p2_win_rate, draw_rate,
                avg_moves_per_game, avg_reward)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, game_number, epsilon, avg_loss,
             p1_win_rate, p2_win_rate, draw_rate,
             avg_moves_per_game, avg_reward),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Evaluations
    # ------------------------------------------------------------------

    def record_evaluation(self, session_id, game_number,
                          opponent_type, num_games, wins, losses, draws):
        """Record an evaluation run against a reference opponent."""
        self._conn.execute(
            """INSERT INTO evaluations
               (session_id, game_number, opponent_type, num_games, wins, losses, draws)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (session_id, game_number, opponent_type,
             num_games, wins, losses, draws),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Games
    # ------------------------------------------------------------------

    def start_game(self, session_id=None, game_type="self_play"):
        """Create a new game row. Returns the game id."""
        cur = self._conn.execute(
            "INSERT INTO games (session_id, game_type) VALUES (?, ?)",
            (session_id, game_type),
        )
        self._conn.commit()
        return cur.lastrowid

    def end_game(self, game_id, winner, num_moves,
                 reward_p1=None, reward_p2=None):
        """Finalize a game with its result."""
        self._conn.execute(
            """UPDATE games
               SET winner = ?, num_moves = ?,
                   total_reward_p1 = ?, total_reward_p2 = ?
               WHERE id = ?""",
            (winner if winner != 0 else None,
             num_moves, reward_p1, reward_p2, game_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Moves
    # ------------------------------------------------------------------

    def record_move(self, game_id, move_index, player_id, column, *,
                    board_state=None, reward=None,
                    chosen_q_value=None, best_q_value=None,
                    was_exploration=False):
        """Record a single move."""
        state_json = None
        if board_state is not None:
            if isinstance(board_state, np.ndarray):
                state_json = json.dumps(board_state.tolist())
            else:
                state_json = json.dumps(board_state)

        self._conn.execute(
            """INSERT INTO moves
               (game_id, move_index, player_id, column_played,
                board_state, reward, chosen_q_value, best_q_value, was_exploration)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (game_id, move_index, player_id, column,
             state_json, reward, chosen_q_value, best_q_value,
             1 if was_exploration else 0),
        )
        # Commit is batched at end_game for performance

    # ------------------------------------------------------------------
    # Queries (API de lecture)
    # ------------------------------------------------------------------

    def get_session_snapshots(self, session_id):
        """Return all snapshots for a session as list of dicts."""
        cur = self._conn.execute(
            "SELECT * FROM training_snapshots WHERE session_id = ? ORDER BY game_number",
            (session_id,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_session_evaluations(self, session_id):
        """Return all evaluations for a session."""
        cur = self._conn.execute(
            "SELECT * FROM evaluations WHERE session_id = ? ORDER BY game_number",
            (session_id,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_game_moves(self, game_id):
        """Return all moves for a game."""
        cur = self._conn.execute(
            "SELECT * FROM moves WHERE game_id = ? ORDER BY move_index",
            (game_id,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_all_sessions(self):
        """Return all training sessions."""
        cur = self._conn.execute(
            "SELECT * FROM training_sessions ORDER BY started_at DESC"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
