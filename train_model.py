# train_model.py
import sqlite3
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Connect4Dataset(Dataset):
    def __init__(self, db_path="games.db"):
        self.samples = []
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT board_state, player_id, column, game_id FROM moves")
            for row in cur.fetchall():
                board_state = np.array(json.loads(row[0]), dtype=np.float32)  # shape (6,7)
                player = row[1]
                move = row[2]
                # convert state to input channels (player perspective)
                # channel 0 = current player pieces, channel 1 = opponent pieces
                current = (board_state == player).astype(np.float32)
                opponent = (board_state == (1 if player == 2 else 2)).astype(np.float32)
                x = np.stack([current, opponent], axis=0)  # shape (2,6,7)
                y = move  # target column 0..6
                self.samples.append((x, y))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(128*6*7, 256),
            nn.ReLU(),
            nn.Linear(256, 7)  # 7 columns
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)

def train(db_path="games.db", epochs=5):
    ds = Connect4Dataset(db_path)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = SimpleCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss=0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device).long()
            out = model(xb)
            loss = criterion(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} loss {total_loss/len(loader)}")
    torch.save(model.state_dict(), "connect4_model.pth")

if __name__ == "__main__":
    train()
