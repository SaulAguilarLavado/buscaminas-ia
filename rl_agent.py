"""Primer agente Q-Learning con estados locales observables del Buscaminas."""

import json
from collections import defaultdict
from pathlib import Path
from random import Random

from adaptive_game import Minesweeper


def available(game: Minesweeper) -> list[tuple[int, int]]:
  """Devuelve casillas cerradas sin consultar la ubicación de las minas."""
  return [
    (row, column)
    for row in range(game.size)
    for column in range(game.size)
    if (row, column) not in game.opened
  ]


def state(game: Minesweeper, cell: tuple[int, int]) -> str:
  """Resume pistas visibles alrededor de una casilla candidata."""
  clues = []
  for neighbor in game.neighbors(cell):
    if neighbor not in game.opened:
      continue
    hidden = sum(near not in game.opened for near in game.neighbors(neighbor))
    clues.append((game.number(neighbor), hidden))
  if not clues:
    return "sin_pistas"
  estimates = [round(4 * number / hidden) for number, hidden in clues]
  return f"{min(4, min(estimates))}:{min(4, max(estimates))}:{min(3, len(clues))}"


def choose(game: Minesweeper, values: dict[str, float], rng: Random,
           epsilon: float) -> tuple[int, int]:
  """Selecciona una acción válida con exploración epsilon-greedy."""
  cells = available(game)
  if rng.random() < epsilon:
    return rng.choice(cells)
  best = max(values.get(state(game, cell), 0.0) for cell in cells)
  return rng.choice([
    cell for cell in cells if values.get(state(game, cell), 0.0) == best
  ])


def episode(values: dict[str, float], seed: int, rng: Random,
            epsilon: float, learn: bool) -> bool:
  """Juega una partida y actualiza Q después de cada acción si corresponde."""
  game = Minesweeper(5, 3, seed)
  game.reveal(2, 2)
  while not game.finished:
    cell = choose(game, values, rng, epsilon)
    key = state(game, cell)
    before = len(game.opened)
    game.reveal(*cell)
    reward = 10.0 if game.won else -10.0 if game.finished else 0.1 * (
      len(game.opened) - before
    )
    if learn:
      next_best = 0.0 if game.finished else max(
        values.get(state(game, other), 0.0) for other in available(game)
      )
      values[key] += 0.15 * (reward + 0.95 * next_best - values[key])
  return game.won


def train_agent(root: Path, episodes: int = 10000, evaluation: int = 1000,
                seed: int = 42) -> dict:
  """Entrena Q-Learning y compara con una política aleatoria en mismas semillas."""
  if not 1 <= episodes <= 1_000_000 or not 1 <= evaluation <= 100_000:
    raise ValueError("episodes y evaluation deben ser positivos y razonables")
  values: dict[str, float] = defaultdict(float)
  rng = Random(seed)
  training_wins = 0
  for index in range(episodes):
    epsilon = max(0.05, 1.0 - 0.95 * index / max(1, episodes - 1))
    training_wins += episode(values, seed + index, rng, epsilon, True)
  learned_wins = 0
  random_wins = 0
  for index in range(evaluation):
    board_seed = seed + episodes + index
    learned_wins += episode(values, board_seed, Random(board_seed), 0.0, False)
    random_wins += episode(values, board_seed, Random(board_seed), 1.0, False)
  report = {
    "algoritmo": "Q-Learning tabular con estado local compartido",
    "tablero": "5x5, 3 minas; primera casilla central segura",
    "partidas_entrenamiento": episodes,
    "victorias_entrenamiento": training_wins,
    "partidas_evaluacion": evaluation,
    "victorias_agente": learned_wins,
    "tasa_victorias_agente": round(learned_wins / evaluation, 4),
    "victorias_aleatorio": random_wins,
    "tasa_victorias_aleatorio": round(random_wins / evaluation, 4),
    "estados_aprendidos": len(values),
    "semilla": seed,
  }
  artifacts = root / "artifacts"
  artifacts.mkdir(exist_ok=True)
  (artifacts / "rl_q_values.json").write_text(json.dumps(dict(values), indent=2))
  (artifacts / "rl_metrics.json").write_text(json.dumps(report, indent=2))
  return report


def demo_agent(root: Path, seed: int = 42) -> str:
  """Muestra una partida del agente entrenado sin revelar minas durante el juego."""
  model_path = root / "artifacts" / "rl_q_values.json"
  if not model_path.is_file():
    raise FileNotFoundError("Entrena primero con: python main.py train-agent")
  values = json.loads(model_path.read_text())
  game = Minesweeper(5, 3, seed)
  game.reveal(2, 2)
  lines = ["Inicio: primera casilla central abierta", board_text(game)]
  rng = Random(seed)
  turn = 0
  while not game.finished:
    cell = choose(game, values, rng, 0.0)
    clue = state(game, cell)
    game.reveal(*cell)
    turn += 1
    lines.extend([
      f"Jugada {turn}: abrir ({cell[0] + 1}, {cell[1] + 1}), estado {clue}",
      board_text(game),
    ])
  lines.append("Resultado: victoria" if game.won else "Resultado: derrota")
  return "\n".join(lines)


def board_text(game: Minesweeper) -> str:
  """Formatea el tablero observable para la demostración."""
  rows = game.visible_board()
  return "\n".join(" ".join(row) for row in rows)
