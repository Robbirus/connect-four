"""
Self-Play Trainer for Connect 4 AI.
Trains the agent by playing games against itself.
"""

import time
import os
import numpy as np
from connect4_core.board import Board
from .agent import Connect4Agent
from .evaluator import compute_reward, analyze_move
from .tracker import Tracker


class SelfPlayTrainer:
    """
    Trains Connect 4 AI through self-play.

    The agent plays against itself, learning from both winning
    and losing positions to improve its strategy.
    """

    def __init__(
        self,
        model_path=None,
        db_path=None,
        batch_size=None,
        replay_updates_per_game=4,
        use_amp=None,
        enable_tracking=True,
    ):
        """
        Initialize the trainer.

        Args:
            model_path: Path to load/save the model
            db_path: Path to the SQLite database (default: games.db)
        """
        self.agent = Connect4Agent(
            model_path=model_path,
            batch_size=batch_size,
            use_amp=use_amp,
        )
        self.model_path = model_path or "connect4_dqn.pth"
        self.tracker = Tracker(db_path=db_path) if enable_tracking else None
        self.session_id = None
        self.replay_updates_per_game = max(1, int(replay_updates_per_game))
        self._replay_started_logged = False

        # Statistics
        self.games_played = 0
        self.player1_wins = 0
        self.player2_wins = 0
        self.draws = 0
        self.total_moves = 0
        self.losses = []
        self.recent_rewards = []  # rewards over recent window

    def play_game(self, verbose=False):
        """
        Play a single self-play game and learn from it.

        Args:
            verbose: Print game progress

        Returns:
            Winner (1, 2, or 0 for draw)
        """
        board = Board()
        current_player = 1
        move_count = 0
        game_history = []  # Store (board_state, player, action, move_meta)

        # Track game in DB
        game_id = None
        if self.tracker is not None:
            game_id = self.tracker.start_game(
                session_id=self.session_id, game_type="self_play")

        while True:
            # Get valid moves
            valid_moves = board.get_valid_locations()

            if not valid_moves:
                # Draw
                self._process_game_end(
                    game_history, winner=0, move_count=move_count,
                    game_id=game_id)
                self.draws += 1
                return 0

            # Store state before move
            state_before = np.copy(board.grid)

            # Select move (with Q-value metadata)
            action, move_meta = self.agent.select_move_tracked(
                board, current_player, valid_moves, training=True)

            # Make move
            row = board.get_next_open_row(action)
            board.drop_piece(row, action, current_player)

            # Store in history
            game_history.append((state_before, current_player, action, move_meta))
            move_count += 1
            self.total_moves += 1

            if verbose:
                print(f"\nPlayer {current_player} plays column {action}")
                self._print_board(board)

            # Check for win
            if board.winning_move(current_player):
                self._process_game_end(
                    game_history, winner=current_player,
                    move_count=move_count, game_id=game_id)
                if current_player == 1:
                    self.player1_wins += 1
                else:
                    self.player2_wins += 1
                return current_player

            # Switch player
            current_player = 2 if current_player == 1 else 1

    def _process_game_end(self, game_history, winner, move_count, game_id):
        """
        Process end of game - assign rewards, train, and record to DB.

        Args:
            game_history: List of (state, player, action, move_meta) tuples
            winner: Winner (1, 2, or 0 for draw)
            move_count: Total moves in game
            game_id: Database game id
        """
        # Rebuild board states for analysis
        boards = [np.zeros((6, 7), dtype=int)]
        current_board = np.zeros((6, 7), dtype=int)

        for state, player, action, _ in game_history:
            row = 0
            for r in range(6):
                if current_board[r, action] == 0:
                    row = r
                    break
            current_board[row, action] = player
            boards.append(current_board.copy())

        reward_p1 = 0.0
        reward_p2 = 0.0

        # Process each move and assign rewards
        for i, (state, player, action, move_meta) in enumerate(game_history):
            is_final_move = (i >= len(game_history) - 2)
            board_after = boards[i + 1]

            # Analyze the move for special conditions
            move_analysis = analyze_move(state, board_after, player, action)

            # Compute reward
            if winner == 0:
                reward = compute_reward(
                    board_after, player, False, False, True, move_count)
            elif winner == player:
                if is_final_move:
                    reward = compute_reward(
                        board_after, player, True, False, False, move_count)
                else:
                    reward = compute_reward(
                        board_after, player, False, False, False, move_count,
                        blocked_win=move_analysis['blocked_win'],
                        created_trap=move_analysis['created_trap']
                    )
                    reward += 0.03
            else:
                if is_final_move:
                    reward = compute_reward(
                        board_after, player, False, True, False, move_count,
                        missed_block=move_analysis['missed_block']
                    )
                else:
                    reward = compute_reward(
                        board_after, player, False, False, False, move_count,
                        blocked_win=move_analysis['blocked_win'],
                        missed_block=move_analysis['missed_block']
                    )
                    reward -= 0.03

            # Accumulate rewards per player
            if player == 1:
                reward_p1 += reward
            else:
                reward_p2 += reward

            # Record move to DB
            if self.tracker is not None:
                self.tracker.record_move(
                    game_id, move_index=i, player_id=player, column=action,
                    reward=reward,
                    chosen_q_value=move_meta.get("chosen_q"),
                    best_q_value=move_meta.get("best_q"),
                    was_exploration=move_meta.get("was_exploration", False),
                )

            # Get next state
            if i + 1 < len(game_history):
                next_state = game_history[i + 1][0]
                done = False
            else:
                next_state = None
                done = True

            # Store experience
            self.agent.remember(state, action, reward,
                                next_state, done, player)

        # Finalize game in DB
        if self.tracker is not None:
            self.tracker.end_game(
                game_id, winner=winner, num_moves=move_count,
                reward_p1=reward_p1, reward_p2=reward_p2)

        self.recent_rewards.append((reward_p1 + reward_p2) / 2)

        # Ramp optimization steps with replay-buffer fill to avoid sudden stalls.
        if len(self.agent.memory) < self.agent.batch_size:
            return

        max_updates_from_buffer = max(1, len(self.agent.memory) // self.agent.batch_size)
        updates_to_run = min(self.replay_updates_per_game, max_updates_from_buffer)

        for _ in range(updates_to_run):
            loss = self.agent.replay()
            if loss is not None:
                self.losses.append(loss)

    def train(self, num_games=10000, save_every=1000, verbose_every=100):
        """
        Train the agent through self-play.

        Args:
            num_games: Number of games to play
            save_every: Save model every N games
            verbose_every: Print stats every N games
        """
        print(f"Starting training for {num_games} games...")
        print(f"Device: {self.agent.device}")
        print(f"Batch size: {self.agent.batch_size}")
        print(f"Replay updates/game: {self.replay_updates_per_game}")
        print(f"AMP enabled: {self.agent.use_amp}")
        print(f"DB tracking: {'on' if self.tracker is not None else 'off'}")
        print("-" * 50)

        # Start tracking session
        if self.tracker is not None:
            self.session_id = self.tracker.start_session(
                hyperparameters={
                    "lr": self.agent.learning_rate,
                    "gamma": self.agent.gamma,
                    "batch_size": self.agent.batch_size,
                    "epsilon_start": self.agent.epsilon,
                    "epsilon_min": self.agent.epsilon_min,
                    "epsilon_decay": self.agent.epsilon_decay,
                    "replay_updates_per_game": self.replay_updates_per_game,
                    "use_amp": self.agent.use_amp,
                    "seed": os.environ.get("PYTHONHASHSEED", "unknown"),
                    "seed_mode": "best-effort"
                }
            )

        start_time = time.time()

        for game_num in range(1, num_games + 1):
            self.play_game(verbose=False)
            self.games_played += 1

            if (not self._replay_started_logged
                    and len(self.agent.memory) >= self.agent.batch_size):
                self._replay_started_logged = True
                print(
                    f"Replay activated at game {game_num} "
                    f"(memory={len(self.agent.memory)}, batch_size={self.agent.batch_size}, "
                    f"max_updates/game={self.replay_updates_per_game})"
                )

            # Record snapshot + print stats periodically
            if game_num % verbose_every == 0:
                self._print_stats(game_num, start_time)
                self._record_snapshot(game_num)

            # Save model periodically
            if game_num % save_every == 0:
                self.agent.save(self.model_path)

        # Final save
        self.agent.save(self.model_path)
        if self.tracker is not None and self.session_id is not None:
            self.tracker.end_session(self.session_id, num_games=num_games)
        print("\nTraining complete!")
        self._print_stats(num_games, start_time)

    def _print_stats(self, game_num, start_time):
        """Print training statistics."""
        elapsed = time.time() - start_time
        games_per_sec = game_num / elapsed if elapsed > 0 else 0

        total = self.player1_wins + self.player2_wins + self.draws
        p1_rate = self.player1_wins / total * 100 if total > 0 else 0
        p2_rate = self.player2_wins / total * 100 if total > 0 else 0
        draw_rate = self.draws / total * 100 if total > 0 else 0

        avg_loss = np.mean(self.losses[-100:]) if self.losses else 0

        print(f"Game {game_num:6d} | "
              f"P1: {p1_rate:5.1f}% | P2: {p2_rate:5.1f}% | Draw: {draw_rate:5.1f}% | "
              f"ε: {self.agent.epsilon:.3f} | Loss: {avg_loss:.4f} | "
              f"{games_per_sec:.1f} games/s")

    def _record_snapshot(self, game_number):
        """Record a training snapshot to the database."""
        if self.session_id is None:
            return

        total = self.player1_wins + self.player2_wins + self.draws
        avg_loss = float(np.mean(self.losses[-100:])) if self.losses else None
        avg_reward = (
            float(np.mean(self.recent_rewards[-100:]))
            if self.recent_rewards else None
        )
        avg_moves = (
            self.total_moves / total if total > 0 else None
        )

        self.tracker.record_snapshot(
            self.session_id,
            game_number=game_number,
            epsilon=self.agent.epsilon,
            avg_loss=avg_loss,
            p1_win_rate=self.player1_wins / total if total > 0 else None,
            p2_win_rate=self.player2_wins / total if total > 0 else None,
            draw_rate=self.draws / total if total > 0 else None,
            avg_moves_per_game=avg_moves,
            avg_reward=avg_reward,
        )

    def _print_board(self, board):
        """Print board state to console."""
        grid = board.grid
        symbols = {0: '.', 1: 'X', 2: 'O'}
        print("\n 0 1 2 3 4 5 6")
        for r in range(5, -1, -1):
            row_str = " ".join(symbols[grid[r, c]] for c in range(7))
            print(f" {row_str}")
        print()


def train_ai(num_games=10000, model_path="connect4_dqn.pth"):
    """
    Convenience function to train the AI.

    Args:
        num_games: Number of self-play games
        model_path: Path to save the model
    """
    trainer = SelfPlayTrainer(model_path=model_path)
    trainer.train(num_games=num_games)
    return trainer


if __name__ == "__main__":
    train_ai(num_games=10000)
