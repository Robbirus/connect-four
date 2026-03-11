PRAGMA foreign_keys = ON;

-- ========================================
-- Joueurs
-- ========================================
CREATE TABLE players (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL
);

-- ========================================
-- Sessions d'entraînement (1 par lancement de train_ai.py)
-- ========================================
CREATE TABLE training_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT NOT NULL DEFAULT (datetime('now')),
  ended_at TEXT,
  num_games INTEGER,
  hyperparameters TEXT -- JSON: {"lr", "gamma", "batch_size", "epsilon_start", "epsilon_min", "epsilon_decay"}
);

-- ========================================
-- Snapshots périodiques (courbes d'apprentissage)
-- Enregistré toutes les N parties pendant l'entraînement
-- ========================================
CREATE TABLE training_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  game_number INTEGER NOT NULL,
  epsilon REAL,
  avg_loss REAL,
  p1_win_rate REAL,
  p2_win_rate REAL,
  draw_rate REAL,
  avg_moves_per_game REAL,
  avg_reward REAL,
  FOREIGN KEY(session_id) REFERENCES training_sessions(id)
);

-- ========================================
-- Évaluations contre adversaires de référence
-- Pour mesurer la progression réelle du bot
-- ========================================
CREATE TABLE evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER NOT NULL,
  game_number INTEGER NOT NULL,      -- à quel point de l'entraînement
  opponent_type TEXT NOT NULL,        -- 'random', 'minimax_depth3', etc.
  num_games INTEGER NOT NULL,
  wins INTEGER NOT NULL,
  losses INTEGER NOT NULL,
  draws INTEGER NOT NULL,
  FOREIGN KEY(session_id) REFERENCES training_sessions(id)
);

-- ========================================
-- Parties jouées
-- ========================================
CREATE TABLE games (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER,                -- NULL si partie hors entraînement
  game_type TEXT NOT NULL DEFAULT 'self_play', -- 'self_play', 'vs_human', 'vs_random', 'evaluation'
  date_played TEXT NOT NULL DEFAULT (datetime('now')),
  winner INTEGER,                    -- NULL si match nul
  num_moves INTEGER,
  total_reward_p1 REAL,
  total_reward_p2 REAL,
  FOREIGN KEY(session_id) REFERENCES training_sessions(id),
  FOREIGN KEY(winner) REFERENCES players(id)
);

-- ========================================
-- Coups joués
-- ========================================
CREATE TABLE moves (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id INTEGER NOT NULL,
  move_index INTEGER NOT NULL,       -- 0-based
  player_id INTEGER NOT NULL,
  column_played INTEGER NOT NULL,    -- 0..6
  board_state TEXT,                  -- JSON du plateau AVANT le coup (optionnel)
  reward REAL,
  chosen_q_value REAL,               -- Q-value du coup joué
  best_q_value REAL,                 -- Q-value max parmi les coups valides
  was_exploration INTEGER DEFAULT 0, -- 1 si coup aléatoire (epsilon)
  FOREIGN KEY(game_id) REFERENCES games(id),
  FOREIGN KEY(player_id) REFERENCES players(id)
);
