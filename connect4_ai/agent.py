"""
Connect 4 AI Agent with DQN (Deep Q-Network).
Handles move selection, learning, and model management.
"""

import os
import random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F

from .network import Connect4Network
from .evaluator import PositionEvaluator


class Connect4Agent:
    """
    DQN-based agent for playing Connect 4.

    Features:
    - Experience replay for stable learning
    - Epsilon-greedy exploration
    - Target network for stable Q-learning
    - Position evaluation for reward shaping
    """

    def __init__(self, model_path=None, device=None):
        """
        Initialize the agent.

        Args:
            model_path: Path to load a pre-trained model
            device: torch device (auto-detected if None)
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Main network and target network
        self.policy_net = Connect4Network().to(self.device)
        self.target_net = Connect4Network().to(self.device)

        # Experience replay buffer
        self.memory = deque(maxlen=100000)

        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.target_update_freq = 1000

        # Optimizer (must be created before load)
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )

        # Training stats
        self.steps = 0

        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            self.load(model_path)

        # Copy weights to target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Position evaluator for reward shaping
        self.evaluator = PositionEvaluator()

    def board_to_tensor(self, board, player):
        """
        Convert board state to tensor input for the network.

        Args:
            board: Board object or numpy array (6x7)
            player: Current player (1 or 2)

        Returns:
            Tensor of shape (1, 2, 6, 7)
        """
        if hasattr(board, 'grid'):
            grid = board.grid
        else:
            grid = board

        opponent = 2 if player == 1 else 1

        # Channel 0: current player's pieces
        current_channel = (grid == player).astype(np.float32)
        # Channel 1: opponent's pieces
        opponent_channel = (grid == opponent).astype(np.float32)

        # Stack channels and add batch dimension
        state = np.stack([current_channel, opponent_channel], axis=0)
        tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        return tensor

    def select_move(self, board, player, valid_moves, training=True):
        """
        Select a move using epsilon-greedy policy.

        Args:
            board: Current board state
            player: Current player
            valid_moves: List of valid column indices
            training: Whether in training mode (enables exploration)

        Returns:
            Selected column index
        """
        action, _ = self.select_move_tracked(board, player, valid_moves, training)
        return action

    def select_move_tracked(self, board, player, valid_moves, training=True):
        """
        Select a move and return metadata for tracking.

        Returns:
            (action, meta) where meta is a dict with chosen_q, best_q, was_exploration
        """
        if not valid_moves:
            return None, {}

        # Epsilon-greedy exploration during training
        was_exploration = training and random.random() < self.epsilon

        # Always compute Q-values for tracking
        state_tensor = self.board_to_tensor(board, player)
        with torch.no_grad():
            self.policy_net.eval()
            q_values = self.policy_net(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]

            masked_q = np.full(7, float('-inf'))
            for col in valid_moves:
                masked_q[col] = q_values_np[col]

            best_action = int(np.argmax(masked_q))
            best_q = float(masked_q[best_action])

        if was_exploration:
            action = random.choice(valid_moves)
        else:
            action = best_action

        chosen_q = float(q_values_np[action])

        meta = {
            "chosen_q": chosen_q,
            "best_q": best_q,
            "was_exploration": was_exploration,
        }
        return action, meta

    def remember(self, state, action, reward, next_state, done, player):
        """
        Store experience in replay buffer.

        Args:
            state: Board state before action
            action: Column selected
            reward: Reward received
            next_state: Board state after action
            done: Whether game ended
            player: Current player
        """
        state_tensor = self.board_to_tensor(state, player)

        if next_state is not None:
            # Opponent's perspective for next state
            opponent = 2 if player == 1 else 1
            next_tensor = self.board_to_tensor(next_state, opponent)
        else:
            next_tensor = None

        self.memory.append((
            state_tensor.cpu(),
            action,
            reward,
            next_tensor.cpu() if next_tensor is not None else None,
            done
        ))

    def replay(self):
        """
        Train on a batch of experiences from replay buffer.

        Returns:
            Average loss, or None if not enough samples
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)

        states = torch.cat([exp[0] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp[1] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp[2] for exp in batch]).to(self.device)
        dones = torch.FloatTensor([exp[4] for exp in batch]).to(self.device)

        # Get next states (handle None for terminal states)
        non_final_mask = torch.tensor(
            [exp[3] is not None for exp in batch],
            dtype=torch.bool
        ).to(self.device)

        non_final_next_states = torch.cat(
            [exp[3] for exp in batch if exp[3] is not None]
        ).to(self.device) if any(non_final_mask) else None

        # Compute current Q values
        self.policy_net.train()
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        next_q = torch.zeros(self.batch_size).to(self.device)
        if non_final_next_states is not None \
           and len(non_final_next_states) > 0:
            with torch.no_grad():
                # Use negative of opponent's max Q (zero-sum game)
                next_q[non_final_mask] = - \
                    self.target_net(non_final_next_states).max(1)[0]

        target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Compute loss and update
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, path):
        """Save model weights to file."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """Load model weights from file."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'epsilon' in checkpoint:
                self.epsilon = checkpoint['epsilon']
            if 'steps' in checkpoint:
                self.steps = checkpoint['steps']
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}, starting fresh")

    def set_eval_mode(self):
        """Set network to evaluation mode (no exploration)."""
        self.policy_net.eval()
        self.epsilon = 0

    def set_train_mode(self):
        """Set network to training mode."""
        self.policy_net.train()
