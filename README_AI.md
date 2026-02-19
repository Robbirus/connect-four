# Connect 4 AI

A Connect 4 game with a Deep Q-Network (DQN) AI trained through self-play.

## Project Structure

```
connect-four/
├── connect4_core/           # Game logic (no UI dependencies)
│   ├── board.py            # Board class with game rules
│   └── config.py           # Game constants
│
├── connect4_game/           # Pygame UI
│   ├── ui.py               # Graphics and animations
│   ├── constants.py        # UI constants
│   ├── game_controller.py  # Human vs Human game
│   └── ai_controller.py    # Human vs AI game
│
├── connect4_ai/             # AI module
│   ├── network.py          # CNN architecture
│   ├── agent.py            # DQN agent with experience replay
│   ├── evaluator.py        # Position evaluation & rewards
│   └── trainer.py          # Self-play training
│
├── main.py                  # Play Human vs Human
├── train_ai.py              # Train the AI
└── play_vs_ai.py            # Play against the AI
```

## Installation

```bash
pip install torch numpy pygame
```

## Usage

### 1. Train the AI

```bash
# Train for 10,000 games (default)
python train_ai.py

# Train for more games
python train_ai.py --games 50000

# Specify model path
python train_ai.py --games 20000 --model my_model.pth
```

Training output:
```
Game   100 | P1: 48.0% | P2: 46.0% | Draw:  6.0% | ε: 0.951 | Loss: 0.1234 | 45.2 games/s
Game   200 | P1: 49.5% | P2: 45.0% | Draw:  5.5% | ε: 0.904 | Loss: 0.0987 | 44.8 games/s
...
```

### 2. Play Against the AI

```bash
# Play as Player 1 (Red, goes first)
python play_vs_ai.py

# Play as Player 2 (Yellow, goes second)
python play_vs_ai.py --player 2

# Use a specific model
python play_vs_ai.py --model my_model.pth
```

### 3. Human vs Human

```bash
python main.py
```

## How the AI Works

### Neural Network Architecture

- **Input**: 2 channels × 6 rows × 7 columns
  - Channel 1: Current player's pieces
  - Channel 2: Opponent's pieces
- **Hidden**: 4 convolutional layers with batch normalization
- **Output**: Q-values for each of 7 columns

### Training Method: Deep Q-Learning with Self-Play

1. **Self-Play**: The AI plays against itself
2. **Experience Replay**: Stores game experiences in memory
3. **Target Network**: Stabilizes learning with periodic updates
4. **Epsilon-Greedy**: Balances exploration vs exploitation

### Reward System

| Event | Reward |
|-------|--------|
| Win | +1.0 to +1.3 (bonus for quick wins) |
| Loss | -1.0 |
| Draw | 0.0 |
| Center control | +0.02 per piece × column weight |
| Threats (3 in a row) | +0.05 |
| Opponent threats | -0.05 |

## Tips for Training

- **10,000 games**: Basic competency
- **50,000 games**: Good play
- **100,000+ games**: Strong play

The AI learns:
- Center column control is important
- Block opponent's winning threats
- Create multiple threats simultaneously
- Avoid moves that enable opponent wins

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pygame
