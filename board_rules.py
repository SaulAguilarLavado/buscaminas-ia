"""Reglas fijas del tablero de referencia: 5x5, 3 minas, inicio seguro central.

Con el centro y sus ocho vecinas protegidas quedan C(16, 3) = 560 distribuciones de
minas posibles. Este módulo es la única fuente de esas reglas para el solucionador
exacto, el agente y la evaluación.
"""

from functools import lru_cache
from itertools import combinations
from random import Random

from adaptive_game import Minesweeper

SIZE, MINES, FIRST = 5, 3, (2, 2)
SAFE_CELLS = SIZE * SIZE - MINES
SPLIT_SEED = 2024
CELLS = [(row, column) for row in range(SIZE) for column in range(SIZE)]

_PROBE = Minesweeper(SIZE, MINES, 0, safe_start=True)
PROTECTED = frozenset({FIRST, *_PROBE.neighbors(FIRST)})


def neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
  """Casillas adyacentes válidas, con las reglas de Minesweeper."""
  return _PROBE.neighbors(cell)


@lru_cache(maxsize=1)
def all_layouts() -> tuple[frozenset, ...]:
  """Las 560 distribuciones de minas, en orden determinista."""
  free = [cell for cell in CELLS if cell not in PROTECTED]
  return tuple(frozenset(combo) for combo in combinations(free, MINES))


def split_layouts(seed: int = SPLIT_SEED,
                  heldout_fraction: float = 0.2) -> tuple[tuple, tuple]:
  """Separa distribuciones de entrenamiento (448) y retenidas (112), sin solape."""
  layouts = list(all_layouts())
  Random(seed).shuffle(layouts)
  heldout = round(len(layouts) * heldout_fraction)
  return tuple(layouts[heldout:]), tuple(layouts[:heldout])


def game_for_layout(layout: frozenset) -> Minesweeper:
  """Partida con una distribución dada y la casilla central ya abierta."""
  game = Minesweeper(SIZE, MINES, 0, safe_start=True)
  game.mines = set(layout)
  game.started = True
  game.reveal(*FIRST)
  return game


def game_for_seed(seed: int) -> Minesweeper:
  """Partida de referencia con minas por semilla y la casilla central abierta."""
  game = Minesweeper(SIZE, MINES, seed, safe_start=True)
  game.reveal(*FIRST)
  return game
