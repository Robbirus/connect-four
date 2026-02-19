"""
Position Evaluator for Connect 4.
Provides heuristic evaluation of board positions for reward shaping.
"""

import numpy as np
from connect4_core import config as CFG


class PositionEvaluator:
    """
    Evaluates Connect 4 board positions using heuristics.
    Used for reward shaping during training.
    """

    # Center column weights (middle columns are more valuable)
    CENTER_WEIGHTS = np.array([0, 1, 2, 3, 2, 1, 0])

    def __init__(self):
        self.rows = CFG.ROW_COUNT
        self.cols = CFG.COLUMN_COUNT

    def evaluate(self, board, player):
        """
        Evaluate the board position from a player's perspective.

        Args:
            board: Board object or numpy array (6x7)
            player: Player ID (1 or 2)

        Returns:
            float: Score (positive = good for player)
        """
        if hasattr(board, 'grid'):
            grid = board.grid
        else:
            grid = board

        opponent = 2 if player == 1 else 1

        score = 0.0

        # 1. Center control bonus
        score += self._center_control(grid, player, opponent)

        # 2. Threat evaluation (2 and 3 in a row)
        score += self._threat_score(grid, player, opponent)

        # 3. Blocking value
        score += self._blocking_value(grid, player, opponent)

        # 4. Double threat detection (trap)
        score += self._double_threat_score(grid, player, opponent)

        # 5. Immediate winning/losing move detection
        score += self._immediate_threat_score(grid, player, opponent)

        return score

    def _center_control(self, grid, player, opponent):
        """Reward for pieces closer to center columns."""
        score = 0.0
        for col in range(self.cols):
            player_pieces = np.sum(grid[:, col] == player)
            opponent_pieces = np.sum(grid[:, col] == opponent)
            weight = self.CENTER_WEIGHTS[col]
            score += (player_pieces - opponent_pieces) * weight * 0.03
        return score

    def _threat_score(self, grid, player, opponent):
        """Evaluate threats (potential winning positions)."""
        score = 0.0

        # Count sequences of 2 and 3
        player_twos = self._count_sequences(grid, player, 2)
        player_threes = self._count_sequences(grid, player, 3)
        opponent_twos = self._count_sequences(grid, opponent, 2)
        opponent_threes = self._count_sequences(grid, opponent, 3)

        # Reward our threats, penalize opponent threats
        score += player_twos * 0.02
        score += player_threes * 0.08
        score -= opponent_twos * 0.02
        score -= opponent_threes * 0.10  # Opponent 3s are dangerous!

        return score

    def _count_sequences(self, grid, player, length):
        """Count sequences of a given length for a player."""
        count = 0

        # Horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                window = grid[r, c:c + 4]
                if self._is_promising_window(window, player, length):
                    count += 1

        # Vertical
        for c in range(self.cols):
            for r in range(self.rows - 3):
                window = grid[r:r + 4, c]
                if self._is_promising_window(window, player, length):
                    count += 1

        # Positive diagonal
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [grid[r + i, c + i] for i in range(4)]
                if self._is_promising_window(np.array(window), player, length):
                    count += 1

        # Negative diagonal
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                window = [grid[r - i, c + i] for i in range(4)]
                if self._is_promising_window(np.array(window), player, length):
                    count += 1

        return count

    def _is_promising_window(self, window, player, required_pieces):
        """Check if a window of 4 has the required pieces and no opponent pieces."""
        player_count = np.sum(window == player)
        empty_count = np.sum(window == 0)
        return player_count == required_pieces \
            and empty_count == (4 - required_pieces)

    def _blocking_value(self, grid, player, opponent):
        """Reward for blocking opponent's winning threats."""
        opponent_winning_threats = self._count_sequences(grid, opponent, 3)
        return -opponent_winning_threats * 0.05

    def _double_threat_score(self, grid, player, opponent):
        """
        Detect double threats (traps) - having 2+ ways to win.
        This is a key tactical pattern in Connect 4.
        """
        score = 0.0

        # Count playable winning threats for player
        player_playable_threats = self._count_playable_threats(grid, player)
        opponent_playable_threats = self._count_playable_threats(
            grid, opponent)

        # Double threat is very valuable (almost guaranteed win)
        if player_playable_threats >= 2:
            score += 0.4
        elif player_playable_threats == 1:
            score += 0.1

        # Opponent double threat is very dangerous
        if opponent_playable_threats >= 2:
            score -= 0.5
        elif opponent_playable_threats == 1:
            score -= 0.15

        return score

    def _count_playable_threats(self, grid, player):
        """
        Count threats where the winning move is immediately playable
        (the empty cell is at the bottom or has a piece below it).
        """
        threats = 0
        opponent = 2 if player == 1 else 1

        # Check all windows of 4
        windows = self._get_all_windows_with_positions(grid)

        for window, positions in windows:
            player_count = sum(1 for p in window if p == player)
            empty_count = sum(1 for p in window if p == 0)
            opponent_count = sum(1 for p in window if p == opponent)

            # 3 player pieces + 1 empty = potential win
            if player_count == 3 and empty_count == 1 and opponent_count == 0:
                # Find the empty position
                for i, (val, (r, c)) in enumerate(zip(window, positions)):
                    if val == 0:
                        # Check if this position is playable (bottom row or has piece below)
                        if r == 0 or grid[r - 1, c] != 0:
                            threats += 1
                        break

        return threats

    def _get_all_windows_with_positions(self, grid):
        """Get all windows of 4 with their positions."""
        windows = []

        # Horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                window = [grid[r, c + i] for i in range(4)]
                positions = [(r, c + i) for i in range(4)]
                windows.append((window, positions))

        # Vertical
        for c in range(self.cols):
            for r in range(self.rows - 3):
                window = [grid[r + i, c] for i in range(4)]
                positions = [(r + i, c) for i in range(4)]
                windows.append((window, positions))

        # Positive diagonal
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                window = [grid[r + i, c + i] for i in range(4)]
                positions = [(r + i, c + i) for i in range(4)]
                windows.append((window, positions))

        # Negative diagonal
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                window = [grid[r - i, c + i] for i in range(4)]
                positions = [(r - i, c + i) for i in range(4)]
                windows.append((window, positions))

        return windows

    def _immediate_threat_score(self, grid, player, opponent):
        """
        Large bonus/penalty for immediate win/loss situations.
        Helps the AI recognize when it must block or can win.
        """
        score = 0.0

        # Check if player can win immediately
        if self._can_win_next_move(grid, player):
            score += 0.5

        # Check if opponent can win immediately (must block!)
        if self._can_win_next_move(grid, opponent):
            score -= 0.6

        return score

    def _can_win_next_move(self, grid, player):
        """Check if player can win on their next move."""
        for col in range(self.cols):
            # Find the row where a piece would land
            row = None
            for r in range(self.rows):
                if grid[r, col] == 0:
                    row = r
                    break

            if row is not None:
                # Simulate the move
                grid[row, col] = player
                wins = self._check_win(grid, player, row, col)
                grid[row, col] = 0  # Undo

                if wins:
                    return True

        return False

    def _check_win(self, grid, player, row, col):
        """Check if the last move at (row, col) wins the game."""
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal /
            (1, -1),  # Diagonal \
        ]

        for dr, dc in directions:
            count = 1
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols \
                    and grid[r, c] == player:
                count += 1
                r, c = r + dr, c + dc
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.cols \
                    and grid[r, c] == player:
                count += 1
                r, c = r - dr, c - dc

            if count >= 4:
                return True

        return False


def compute_reward(board, player, is_win, is_loss, is_draw, move_count,
                   blocked_win=False, missed_block=False, created_trap=False):
    """
    Compute the reward for a state transition.

    Args:
        board: Current board state
        player: Current player
        is_win: Whether the player won
        is_loss: Whether the player lost
        is_draw: Whether it's a draw
        move_count: Total moves played in the game
        blocked_win: Whether this move blocked opponent's winning move
        missed_block: Whether this move failed to block a winning threat
        created_trap: Whether this move created a double threat

    Returns:
        float: Reward value
    """
    evaluator = PositionEvaluator()

    if is_win:
        # Base win reward + bonus for quick wins
        quick_win_bonus = max(0, (21 - move_count) / 42) * \
            0.5  # Bonus for winning fast
        return 1.0 + quick_win_bonus

    if is_loss:
        # Extra penalty if we missed an obvious block
        penalty = -0.3 if missed_block else 0
        return -1.0 + penalty

    if is_draw:
        return 0.1  # Small positive - draw is better than losing

    # Intermediate rewards
    reward = 0.0

    # Reward for blocking opponent's win
    if blocked_win:
        reward += 0.15

    # Reward for creating a trap (double threat)
    if created_trap:
        reward += 0.2

    # Position-based reward
    reward += evaluator.evaluate(board, player) * 0.1

    return reward


def analyze_move(board_before, board_after, player, col):
    """
    Analyze a move to determine special conditions for rewards.

    Args:
        board_before: Board state before the move
        board_after: Board state after the move
        player: Player who made the move
        col: Column where the piece was dropped

    Returns:
        dict with keys: blocked_win, missed_block, created_trap
    """
    evaluator = PositionEvaluator()
    opponent = 2 if player == 1 else 1

    result = {
        'blocked_win': False,
        'missed_block': False,
        'created_trap': False
    }

    # Check if opponent could win before this move
    if hasattr(board_before, 'grid'):
        grid_before = board_before.grid.copy()
    else:
        grid_before = board_before.copy()

    opponent_could_win = evaluator._can_win_next_move(grid_before, opponent)

    # Check if we blocked it
    if opponent_could_win:
        if hasattr(board_after, 'grid'):
            grid_after = board_after.grid.copy()
        else:
            grid_after = board_after.copy()

        opponent_can_still_win = evaluator._can_win_next_move(
            grid_after, opponent)

        if not opponent_can_still_win:
            result['blocked_win'] = True
        else:
            result['missed_block'] = True

    # Check if we created a double threat
    if hasattr(board_after, 'grid'):
        grid_after = board_after.grid
    else:
        grid_after = board_after

    playable_threats = evaluator._count_playable_threats(grid_after, player)
    if playable_threats >= 2:
        result['created_trap'] = True

    return result
