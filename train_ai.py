"""
Train the Connect 4 AI using self-play with Deep Q-Learning.

Usage:
    python train_ai.py [--games NUM] [--model PATH]

Example:
    python train_ai.py --games 20000 --model connect4_dqn.pth
"""

import argparse
from connect4_ai.trainer import SelfPlayTrainer
from utils.seed import seed_everything

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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for best-effort reproducibility (default: 42)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Replay batch size (default: 256)')
    parser.add_argument('--replay-updates', type=int, default=4,
                        help='Gradient updates per completed game (default: 4)')
    parser.add_argument('--amp', action='store_true',
                        help='Enable mixed precision training on CUDA')
    parser.add_argument('--no-tracking', action='store_true',
                        help='Disable SQLite tracking for faster training')

    args = parser.parse_args()

    # Seed first (before creating any torch modules)
    seed_everything(args.seed)

    print("=" * 60)
    print("🎮 Connect 4 AI Training - Self-Play DQN")
    print("=" * 60)
    print(f"📊 Games to play: {args.games}")
    print(f"💾 Model path: {args.model}")
    print(f"📈 Save every: {args.save_every} games")
    print(f"🧮 Batch size: {args.batch_size}")
    print(f"⚙️  Replay updates/game: {args.replay_updates}")
    print(f"🚀 AMP: {'on' if args.amp else 'off'}")
    print(f"🗃️  DB tracking: {'off' if args.no_tracking else 'on'}")
    print("=" * 60)
    print()

    trainer = SelfPlayTrainer(
        model_path=args.model,
        batch_size=args.batch_size,
        replay_updates_per_game=args.replay_updates,
        use_amp=args.amp,
        enable_tracking=not args.no_tracking,
    )
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
