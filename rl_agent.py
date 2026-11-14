"""Agente Q-Learning con estado basado en la probabilidad de mina por casilla.

La probabilidad se calcula solo con información visible (números abiertos, número
total de minas y regla de inicio seguro); nunca consulta ``game.mines`` salvo mediante
``game.number`` de casillas ya abiertas. La política de ejecución solo usa ese estado.
"""

import json
from functools import lru_cache
from math import ceil
from pathlib import Path
from random import Random

from adaptive_game import Minesweeper
from board_rules import (
  CELLS, FIRST, MINES, SIZE, all_layouts, game_for_layout, game_for_seed, neighbors,
)

ALPHA, GAMMA = 0.15, 0.95
EPSILON_START, EPSILON_END = 1.0, 0.05
REWARD_WIN, REWARD_LOSS, REWARD_PER_CELL = 10.0, -10.0, 0.1


@lru_cache(maxsize=1)
def _tables() -> tuple[dict, dict, tuple[int, ...]]:
  """Máscaras de bits compartidas: casillas, vecinas y las 560 distribuciones."""
  bit = {cell: 1 << (cell[0] * SIZE + cell[1]) for cell in CELLS}
  neighbor_mask = {
    cell: sum(bit[near] for near in neighbors(cell)) for cell in CELLS
  }
  layouts = tuple(sum(bit[cell] for cell in layout) for layout in all_layouts())
  return bit, neighbor_mask, layouts


class Belief:
  """Conjunto de distribuciones de minas compatibles con lo observado."""

  def __init__(self, game: Minesweeper):
    if (game.size, game.mine_count) != (SIZE, MINES) or not game.safe_start:
      raise ValueError("Belief solo admite el tablero de referencia con inicio seguro")
    self.game = game
    self.bit, self.neighbor_mask, masks = _tables()
    self.layouts = list(masks)
    self.seen: set[tuple[int, int]] = set()
    self.update()

  def update(self) -> None:
    """Filtra distribuciones con las casillas abiertas desde la última vez."""
    new = [cell for cell in self.game.opened if cell not in self.seen]
    for cell in new:
      mask = self.neighbor_mask[cell]
      number = self.game.number(cell)  # número visible de una casilla abierta
      cell_bit = self.bit[cell]
      self.layouts = [
        layout for layout in self.layouts
        if not layout & cell_bit and (layout & mask).bit_count() == number
      ]
    self.seen.update(new)

  def probabilities(self) -> dict[tuple[int, int], float]:
    """Probabilidad de mina de cada casilla cerrada."""
    total = len(self.layouts)
    return {
      cell: sum(1 for layout in self.layouts if layout & bit) / total
      for cell, bit in self.bit.items() if cell not in self.game.opened
    }


def available(game: Minesweeper) -> list[tuple[int, int]]:
  """Devuelve casillas cerradas sin consultar la ubicación de las minas."""
  return [cell for cell in CELLS if cell not in game.opened]


def state_key(game: Minesweeper, cell: tuple[int, int], probability: float) -> str:
  """Estado: probabilidad discretizada y casillas vecinas aún cerradas."""
  bucket = 0 if probability == 0 else min(21, ceil(probability * 20))
  hidden = min(3, sum(n not in game.opened for n in game.neighbors(cell)))
  return f"{bucket}:{hidden}"


def keys(game: Minesweeper, belief: Belief) -> dict[tuple[int, int], str]:
  """Estado de todas las casillas cerradas."""
  return {
    cell: state_key(game, cell, p) for cell, p in belief.probabilities().items()
  }


def choose(states: dict[tuple[int, int], str], values: dict[str, float],
           rng: Random, epsilon: float) -> tuple[int, int]:
  """Selecciona una acción válida con exploración epsilon-greedy."""
  if rng.random() < epsilon:
    return rng.choice(list(states))
  best = max(values.get(key, 0.0) for key in states.values())
  return rng.choice([c for c, k in states.items() if values.get(k, 0.0) == best])


def epsilon_at(index: int, episodes: int) -> float:
  """Exploración lineal de EPSILON_START a EPSILON_END."""
  progress = index / max(1, episodes - 1)
  return max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * progress)


def train_episode(values: dict[str, float], layout: frozenset, rng: Random,
                  epsilon: float) -> bool:
  """Juega una partida sobre una distribución dada y actualiza Q tras cada acción."""
  game = game_for_layout(layout)
  belief = Belief(game)
  while not game.finished:
    states = keys(game, belief)
    cell = choose(states, values, rng, epsilon)
    key = states[cell]
    before = len(game.opened)
    game.reveal(*cell)
    belief.update()
    reward = REWARD_WIN if game.won else REWARD_LOSS if game.finished else (
      REWARD_PER_CELL * (len(game.opened) - before)
    )
    next_best = 0.0 if game.finished else max(
      values.get(k, 0.0) for k in keys(game, belief).values()
    )
    values[key] = values.get(key, 0.0) + ALPHA * (
      reward + GAMMA * next_best - values.get(key, 0.0)
    )
  return game.won


def new_game(seed: int) -> tuple[Minesweeper, Belief]:
  """Partida de demostración: minas por semilla y casilla central abierta."""
  game = game_for_seed(seed)
  return game, Belief(game)


def demo_agent(root: Path, seed: int = 42) -> str:
  """Muestra una partida del agente entrenado sin revelar minas durante el juego."""
  model_path = root / "artifacts" / "rl_q_values.json"
  if not model_path.is_file():
    raise FileNotFoundError("Entrena primero con: python main.py train-agent")
  values = json.loads(model_path.read_text())
  game, belief = new_game(seed)
  lines = ["Inicio: casilla central abierta (inicio seguro)", board_text(game)]
  rng = Random(seed)
  turn = 0
  while not game.finished:
    states = keys(game, belief)
    cell = choose(states, values, rng, 0.0)
    clue = states[cell]
    game.reveal(*cell)
    belief.update()
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
