"""
Train the Connect 4 AI using self-play with Deep Q-Learning.

Usage:
    python train_ai.py [--games NUM] [--model PATH]

Example:
    python train_ai.py --games 20000 --model connect4_dqn.pth
"""

import argparse
from connect4_ai.trainer import SelfPlayTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Connect 4 AI')
    parser.add_argument('--games', type=int, default=10000,
                        help='Number of self-play games (default: 10000)')
    parser.add_argument('--model', type=str, default='connect4_dqn.pth',
                        help='Model save path (default: connect4_dqn.pth)')
    parser.add_argument('--save-every', type=int, default=1000,
                        help='Save model every N games (default: 1000)')
    parser.add_argument('--verbose-every', type=int, default=100,
                        help='Print stats every N games (default: 100)')

    args = parser.parse_args()

    print("=" * 60)
    print("🎮 Connect 4 AI Training - Self-Play DQN")
    print("=" * 60)
    print(f"📊 Games to play: {args.games}")
    print(f"💾 Model path: {args.model}")
    print(f"📈 Save every: {args.save_every} games")
    print("=" * 60)
    print()

    trainer = SelfPlayTrainer(model_path=args.model)
    trainer.train(
        num_games=args.games,
        save_every=args.save_every,
        verbose_every=args.verbose_every
    )

    print()
    print("=" * 60)
    print("✅ Training complete!")
    print(f"📦 Model saved to: {args.model}")
    print("=" * 60)


if __name__ == "__main__":
    main()
