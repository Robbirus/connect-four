"""
Neural Network architecture for Connect 4 AI.
Uses a CNN to evaluate board positions and predict moves.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4Network(nn.Module):
    """
    Deep Q-Network for Connect 4.

    Architecture:
    - Input: 2 channels (current player, opponent) x 6 rows x 7 cols
    - Several convolutional layers to detect patterns
    - Fully connected layers for move prediction
    - Output: Q-values for each of 7 columns
    """

    def __init__(self):
        super().__init__()

        # Convolutional layers for pattern recognition
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 6 * 7, 256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)

        # Output: Q-value for each column (7 actions)
        self.fc_out = nn.Linear(128, 7)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 2, 6, 7)
               Channel 0: current player (1 where piece exists)
               Channel 1: opponent (1 where piece exists)

        Returns:
            Q-values for each of 7 columns
        """
        # Convolutional layers with batch norm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # Output Q-values
        return self.fc_out(x)

    def predict_move(self, x, valid_moves, temperature=1.0):
        """
        Predict the best move given valid moves.

        Args:
            x: Board state tensor
            valid_moves: List of valid column indices
            temperature: For exploration (higher = more random)

        Returns:
            Selected column index
        """
        with torch.no_grad():
            q_values = self.forward(x)

            # Mask invalid moves with very negative values
            mask = torch.full_like(q_values, float('-inf'))
            for col in valid_moves:
                mask[0, col] = 0

            masked_q = q_values + mask

            if temperature == 0:
                # Greedy selection
                return int(torch.argmax(masked_q, dim=1).item())
            else:
                # Softmax with temperature for exploration
                probs = F.softmax(masked_q / temperature, dim=1)
                return int(torch.multinomial(probs, 1).item())
