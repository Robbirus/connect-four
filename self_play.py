from connect4_game.game_engine import GameEngine
from simple_ai import random_ai

def generate_games(n=1000):
    for i in range(n):
        game = GameEngine()

        while not game.game_over:
            move = random_ai(game)
            game.play_move(move)

        if i % 50 == 0:
            print("Game", i, "winner:", game.winner)

if __name__ == "__main__":
    generate_games(5000)
