"""
Play Connect 4 against the trained AI.

Usage:
    python play_vs_ai.py [--model PATH] [--player 1|2]

Example:
    python play_vs_ai.py --model connect4_dqn.pth --player 1
"""

import argparse
import os
from connect4_ai.agent import Connect4Agent
from connect4_game.ai_controller import AIGameController


def main():
    parser = argparse.ArgumentParser(description='Play Connect 4 vs AI')
    parser.add_argument('--model', type=str, default='connect4_dqn.pth',
                        help='Path to trained model')
    parser.add_argument('--player', type=int, choices=[1, 2], default=1,
                        help='Play as player 1 (red) or 2 (yellow)')

    args = parser.parse_args()

    print("=" * 50)
    print("🎮 Connect 4 - Human vs AI")
    print("=" * 50)

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"⚠️  Model not found at {args.model}")
        print("   Training a new model first...")
        print("   Run: python train_ai.py")
        print()
        print("   Or start with an untrained AI? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            return

    print(f"🤖 Loading AI from: {args.model}")
    agent = Connect4Agent(model_path=args.model)

    player_color = "Red (first)" if args.player == 1 else "Yellow (second)"
    print(f"👤 You are playing as: {player_color}")
    print("=" * 50)
    print()
    print("Click on a column to drop your piece!")
    print("Close the window or click after game ends to exit.")
    print()

    # Start game
    game = AIGameController(ai_agent=agent, human_player=args.player)
    game.run()


if __name__ == "__main__":
    main()
