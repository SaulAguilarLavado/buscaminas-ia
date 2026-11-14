"""Búsqueda exacta del máximo de victorias posible en el tablero de referencia.

Método (expectimax sobre estados de información, sin ninguna política fija):

* El jugador ve solo las casillas abiertas y sus números. Su estado de información es
  el conjunto de distribuciones de minas aún compatibles más la región abierta.
* En cada estado se evalúa CADA casilla cerrada. Para cada una, las distribuciones
  compatibles que no la contienen se agrupan por lo que el jugador observaría tras
  abrirla (región abierta y números nuevos); las que la contienen pierden.
* El valor del estado es el máximo, sobre todas las casillas, de las victorias
  alcanzables. Se cuentan distribuciones (enteros), no probabilidades en coma flotante.

Como el valor es un máximo sobre todas las acciones, ninguna política cuya jugada
dependa solo del tablero visible (determinista, o con desempate fijado por ese tablero)
puede superarlo. Una política aleatoria lo respeta en esperanza; una sola pasada con una
secuencia aleatoria compartida entre partidas puede superarlo por azar, porque la
aleatoriedad queda correlacionada con la distribución concreta. Un desempate solo elige
entre acciones; no puede elevar el valor esperado por encima de este máximo.

Supuestos: las 560 distribuciones son equiprobables, la primera jugada es la casilla
central y ganar es abrir las 22 casillas seguras. Las banderas no cambian la partida.
"""

import json
import time
from functools import lru_cache
from pathlib import Path
from random import Random

from board_rules import (
  CELLS, FIRST, MINES, SAFE_CELLS, SIZE, all_layouts, game_for_layout, neighbors,
)


def number(layout: frozenset, cell: tuple[int, int]) -> int:
  """Minas adyacentes a una casilla (implementación independiente de Minesweeper)."""
  return sum(near in layout for near in neighbors(cell))


def flood(layout: frozenset, opened: frozenset, cell: tuple[int, int]) -> frozenset:
  """Región abierta tras abrir una casilla segura, propagando los ceros."""
  reached = set(opened)
  pending = [cell]
  while pending:
    current = pending.pop()
    if current in reached:
      continue
    reached.add(current)
    if number(layout, current) == 0:
      pending.extend(near for near in neighbors(current) if near not in layout)
  return frozenset(reached)


def observation(layout: frozenset, opened: frozenset, reached: frozenset) -> tuple:
  """Lo que ve el jugador tras la jugada: región abierta y números nuevos."""
  return reached, tuple(number(layout, cell) for cell in sorted(reached - opened))


@lru_cache(maxsize=None)
def best_wins(layouts: frozenset, opened: frozenset) -> int:
  """Máximo de distribuciones ganadas entre `layouts` con la mejor jugada posible."""
  best = 0
  for cell in CELLS:
    if cell in opened:
      continue
    groups: dict[tuple, list] = {}
    for layout in layouts:
      if cell in layout:
        continue  # abrir una mina pierde: no suma victorias
      reached = flood(layout, opened, cell)
      groups.setdefault(observation(layout, opened, reached), []).append(layout)
    total = 0
    for (reached, _), group in groups.items():
      total += len(group) if len(reached) == SAFE_CELLS else best_wins(
        frozenset(group), reached
      )
    best = max(best, total)
  return best


def optimal_win_count() -> tuple[int, int]:
  """Devuelve (victorias óptimas, distribuciones) para el tablero de referencia."""
  best_wins.cache_clear()
  groups: dict[tuple, list] = {}
  for layout in all_layouts():
    reached = flood(layout, frozenset(), FIRST)
    groups.setdefault(observation(layout, frozenset(), reached), []).append(layout)
  wins = 0
  for (reached, _), group in groups.items():
    wins += len(group) if len(reached) == SAFE_CELLS else best_wins(
      frozenset(group), reached
    )
  return wins, len(all_layouts())


def validate_transitions(sequences_per_layout: int = 5, seed: int = 0) -> dict:
  """Compara `flood` y las reglas de fin con la clase Minesweeper real."""
  rng = Random(seed)
  moves = 0
  mismatches = 0
  for layout in all_layouts():
    for _ in range(sequences_per_layout):
      game = game_for_layout(layout)
      opened = flood(layout, frozenset(), FIRST)
      mismatches += set(game.opened) != set(opened)
      mismatches += game.won != (len(opened) == SAFE_CELLS)
      while not game.finished:
        cell = rng.choice([c for c in CELLS if c not in game.opened])
        game.reveal(*cell)
        moves += 1
        if cell in layout:
          mismatches += not game.finished or game.won
          break
        opened = flood(layout, opened, cell)
        mismatches += set(game.opened) != set(opened)
        mismatches += game.won != (len(opened) == SAFE_CELLS)
        mismatches += game.finished != game.won
  return {"movimientos": moves, "discrepancias": int(mismatches)}


def verify_optimal(root: Path) -> dict:
  """Calcula y guarda el óptimo exacto junto con su validación."""
  start = time.perf_counter()
  validation = validate_transitions()
  wins, total = optimal_win_count()
  report = {
    "tablero": f"{SIZE}x{SIZE}, {MINES} minas; inicio seguro central",
    "victorias_optimas": wins,
    "distribuciones": total,
    "tasa_optima": round(wins / total, 6),
    "metodo": "expectimax exacto sobre estados de información (optimal_solver.py)",
    "supuestos": [
      "560 distribuciones equiprobables",
      "primera jugada fija en la casilla central",
      "ganar = abrir las 22 casillas seguras; las banderas no cambian la partida",
    ],
    "validacion_transiciones": validation,
    "segundos": round(time.perf_counter() - start, 1),
  }
  artifacts = root / "artifacts"
  artifacts.mkdir(exist_ok=True)
  (artifacts / "optimal_solution.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2)
  )
  return report
