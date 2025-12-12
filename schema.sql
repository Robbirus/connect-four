PRAGMA foreign_keys = ON;

CREATE TABLE players (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT
);

CREATE TABLE games (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date_played TEXT,
  winner INTEGER, -- NULL if draw, otherwise player id
  num_moves INTEGER,
  FOREIGN KEY(winner) REFERENCES players(id)
);

CREATE TABLE moves (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  game_id INTEGER,
  move_index INTEGER, -- 0-based
  player_id INTEGER,
  column INTEGER, -- move (0..6 for Connect4)
  board_state BLOB, -- serialized board BEFORE move (optional)
  FOREIGN KEY(game_id) REFERENCES games(id),
  FOREIGN KEY(player_id) REFERENCES players(id)
);
