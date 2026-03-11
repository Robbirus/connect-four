"""
Generate static training graphs from the SQLite database.

Usage:
    python generate_graphs.py [--db PATH] [--session ID] [--out DIR]

Generates PNG files in the output directory:
    1. loss_curve.png         - Loss over training
    2. win_rates.png          - P1/P2/Draw rates over time
    3. epsilon_decay.png      - Exploration rate decay
    4. avg_reward.png         - Average reward over time
    5. game_length.png        - Average moves per game
    6. move_quality.png       - Q-value gap (chosen vs best)
    7. exploration_ratio.png  - % exploration vs exploitation
    8. reward_distribution.png - Distribution of game rewards
"""

import argparse
import os
import sqlite3
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend


def get_connection(db_path):
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        sys.exit(1)
    return sqlite3.connect(db_path)


def pick_session(conn, session_id=None):
    """Return the session id to use (latest if not specified)."""
    cur = conn.cursor()
    if session_id is not None:
        cur.execute("SELECT id FROM training_sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        if row is None:
            print(f"Session {session_id} not found.")
            sys.exit(1)
        return session_id

    cur.execute("SELECT id, started_at, num_games FROM training_sessions ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    if row is None:
        print("No training sessions in the database.")
        sys.exit(1)
    print(f"Using latest session: #{row[0]}  ({row[1]}, {row[2]} games)")
    return row[0]


# ------------------------------------------------------------------ charts


def plot_loss_curve(conn, sid, out_dir):
    """1. Loss curve over training."""
    rows = conn.execute(
        "SELECT game_number, avg_loss FROM training_snapshots WHERE session_id = ? ORDER BY game_number",
        (sid,),
    ).fetchall()
    if not rows:
        return
    x, y = zip(*[(r[0], r[1]) for r in rows if r[1] is not None])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, linewidth=1.2, color="#2196F3")
    ax.set_xlabel("Game")
    ax.set_ylabel("Average Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)
    print("  -> loss_curve.png")


def plot_win_rates(conn, sid, out_dir):
    """2. Win rates over time."""
    rows = conn.execute(
        "SELECT game_number, p1_win_rate, p2_win_rate, draw_rate "
        "FROM training_snapshots WHERE session_id = ? ORDER BY game_number",
        (sid,),
    ).fetchall()
    if not rows:
        return
    x = [r[0] for r in rows]
    p1 = [r[1] * 100 if r[1] is not None else 0 for r in rows]
    p2 = [r[2] * 100 if r[2] is not None else 0 for r in rows]
    dr = [r[3] * 100 if r[3] is not None else 0 for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, p1, label="P1 Win %", color="#F44336", linewidth=1.2)
    ax.plot(x, p2, label="P2 Win %", color="#FFC107", linewidth=1.2)
    ax.plot(x, dr, label="Draw %", color="#9E9E9E", linewidth=1.2)
    ax.set_xlabel("Game")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Win / Draw Rates")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "win_rates.png"), dpi=150)
    plt.close(fig)
    print("  -> win_rates.png")


def plot_epsilon(conn, sid, out_dir):
    """3. Epsilon decay."""
    rows = conn.execute(
        "SELECT game_number, epsilon FROM training_snapshots WHERE session_id = ? ORDER BY game_number",
        (sid,),
    ).fetchall()
    if not rows:
        return
    x, y = zip(*[(r[0], r[1]) for r in rows if r[1] is not None])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, linewidth=1.2, color="#4CAF50")
    ax.set_xlabel("Game")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate (Epsilon Decay)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "epsilon_decay.png"), dpi=150)
    plt.close(fig)
    print("  -> epsilon_decay.png")


def plot_avg_reward(conn, sid, out_dir):
    """4. Average reward over time."""
    rows = conn.execute(
        "SELECT game_number, avg_reward FROM training_snapshots WHERE session_id = ? ORDER BY game_number",
        (sid,),
    ).fetchall()
    if not rows:
        return
    data = [(r[0], r[1]) for r in rows if r[1] is not None]
    if not data:
        return
    x, y = zip(*data)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, linewidth=1.2, color="#9C27B0")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Game")
    ax.set_ylabel("Average Reward")
    ax.set_title("Average Reward per Game")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "avg_reward.png"), dpi=150)
    plt.close(fig)
    print("  -> avg_reward.png")


def plot_game_length(conn, sid, out_dir):
    """5. Average game length over time."""
    # Use actual games table – compute rolling average per window of 100 games
    rows = conn.execute(
        "SELECT id, num_moves FROM games WHERE session_id = ? AND num_moves IS NOT NULL ORDER BY id",
        (sid,),
    ).fetchall()
    if len(rows) < 10:
        return

    moves = [r[1] for r in rows]
    window = min(100, len(moves) // 2)
    avg = np.convolve(moves, np.ones(window) / window, mode="valid")
    x = np.arange(window, len(moves) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, avg, linewidth=1.2, color="#FF9800")
    ax.set_xlabel("Game")
    ax.set_ylabel("Moves per Game")
    ax.set_title(f"Game Length (rolling avg, window={window})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "game_length.png"), dpi=150)
    plt.close(fig)
    print("  -> game_length.png")


def plot_move_quality(conn, sid, out_dir):
    """6. Move quality – gap between chosen Q and best Q."""
    rows = conn.execute(
        """SELECT m.chosen_q_value, m.best_q_value, g.id
           FROM moves m JOIN games g ON m.game_id = g.id
           WHERE g.session_id = ?
             AND m.chosen_q_value IS NOT NULL
             AND m.best_q_value IS NOT NULL
           ORDER BY g.id, m.move_index""",
        (sid,),
    ).fetchall()
    if len(rows) < 100:
        return

    # Compute per-game average gap
    from collections import defaultdict
    game_gaps = defaultdict(list)
    for chosen, best, gid in rows:
        game_gaps[gid].append(best - chosen)

    game_ids = sorted(game_gaps.keys())
    avg_gaps = [np.mean(game_gaps[gid]) for gid in game_ids]

    # Smooth
    window = min(100, len(avg_gaps) // 2)
    if window < 2:
        return
    smoothed = np.convolve(avg_gaps, np.ones(window) / window, mode="valid")
    x = np.arange(window, len(avg_gaps) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, smoothed, linewidth=1.2, color="#E91E63")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Game")
    ax.set_ylabel("Avg Q-gap (best - chosen)")
    ax.set_title("Move Quality (lower = better decisions)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "move_quality.png"), dpi=150)
    plt.close(fig)
    print("  -> move_quality.png")


def plot_exploration_ratio(conn, sid, out_dir):
    """7. Exploration vs exploitation ratio over time."""
    rows = conn.execute(
        """SELECT g.id, m.was_exploration
           FROM moves m JOIN games g ON m.game_id = g.id
           WHERE g.session_id = ?
           ORDER BY g.id, m.move_index""",
        (sid,),
    ).fetchall()
    if len(rows) < 100:
        return

    from collections import defaultdict
    game_explore = defaultdict(list)
    for gid, expl in rows:
        game_explore[gid].append(expl)

    game_ids = sorted(game_explore.keys())
    ratios = [np.mean(game_explore[gid]) * 100 for gid in game_ids]

    window = min(100, len(ratios) // 2)
    if window < 2:
        return
    smoothed = np.convolve(ratios, np.ones(window) / window, mode="valid")
    x = np.arange(window, len(ratios) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, smoothed, linewidth=1.2, color="#00BCD4")
    ax.set_xlabel("Game")
    ax.set_ylabel("Exploration %")
    ax.set_title("Exploration vs Exploitation")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "exploration_ratio.png"), dpi=150)
    plt.close(fig)
    print("  -> exploration_ratio.png")


def plot_reward_distribution(conn, sid, out_dir):
    """8. Distribution of total rewards per game."""
    rows = conn.execute(
        "SELECT total_reward_p1, total_reward_p2 FROM games WHERE session_id = ? "
        "AND total_reward_p1 IS NOT NULL",
        (sid,),
    ).fetchall()
    if len(rows) < 10:
        return

    r1 = [r[0] for r in rows]
    r2 = [r[1] for r in rows if r[1] is not None]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(r1, bins=50, alpha=0.6, label="P1 reward", color="#F44336")
    if r2:
        ax.hist(r2, bins=50, alpha=0.6, label="P2 reward", color="#FFC107")
    ax.set_xlabel("Total Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution per Game")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reward_distribution.png"), dpi=150)
    plt.close(fig)
    print("  -> reward_distribution.png")


# ------------------------------------------------------------------ main


def main():
    parser = argparse.ArgumentParser(description="Generate training graphs")
    parser.add_argument("--db", default="games.db", help="Path to SQLite database")
    parser.add_argument("--session", type=int, default=None, help="Session ID (default: latest)")
    parser.add_argument("--out", default="graphs", help="Output directory (default: graphs/)")
    args = parser.parse_args()

    conn = get_connection(args.db)
    sid = pick_session(conn, args.session)
    os.makedirs(args.out, exist_ok=True)

    print(f"\nGenerating graphs for session #{sid} -> {args.out}/\n")

    plot_loss_curve(conn, sid, args.out)
    plot_win_rates(conn, sid, args.out)
    plot_epsilon(conn, sid, args.out)
    plot_avg_reward(conn, sid, args.out)
    plot_game_length(conn, sid, args.out)
    plot_move_quality(conn, sid, args.out)
    plot_exploration_ratio(conn, sid, args.out)
    plot_reward_distribution(conn, sid, args.out)

    conn.close()
    print(f"\nDone! Graphs saved in {args.out}/")


if __name__ == "__main__":
    main()
