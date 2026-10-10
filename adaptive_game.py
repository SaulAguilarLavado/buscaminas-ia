"""Buscaminas de consola y adaptación explicable de dificultad."""

from collections import deque
from dataclasses import dataclass, field
from random import Random


LEVELS = {1: (5, 3), 2: (6, 6), 3: (8, 10)}


@dataclass
class AdaptiveSession:
  """Ajusta un nivel según una ventana de cinco partidas completas."""

  level: int = 1
  history: list[tuple[bool, int]] = field(default_factory=list)

  def record(self, won: bool, hints: int) -> str:
    """Registra una partida y ajusta un nivel cuando hay cinco nuevas."""
    self.history.append((won, hints))
    if len(self.history) < 5:
      return f"Partidas registradas: {len(self.history)}/5; nivel {self.level}."
    wins = sum(result for result, _ in self.history)
    mean_hints = sum(used for _, used in self.history) / 5
    if wins >= 4 and mean_hints <= 1:
      change = 1
    elif wins <= 1 or mean_hints >= 3:
      change = -1
    else:
      change = 0
    self.level = max(1, min(3, self.level + change))
    self.history.clear()
    return f"Ventana cerrada: {wins}/5 victorias; siguiente nivel {self.level}."


class Minesweeper:
  """Tablero con primer movimiento seguro y pistas sin revelar la solución."""

  def __init__(self, size: int, mine_count: int, seed: int | None = None):
    if size < 2 or not 0 < mine_count < size * size:
      raise ValueError("Dimensiones o número de minas inválidos")
    self.size = size
    self.mine_count = mine_count
    self.random = Random(seed)
    self.mines: set[tuple[int, int]] = set()
    self.opened: set[tuple[int, int]] = set()
    self.flags: set[tuple[int, int]] = set()
    self.started = False
    self.finished = False
    self.won = False
    self.hints_used = 0

  @classmethod
  def for_level(cls, level: int, seed: int | None = None) -> "Minesweeper":
    """Crea un tablero para un nivel válido."""
    if level not in LEVELS:
      raise ValueError("Nivel desconocido")
    return cls(*LEVELS[level], seed=seed)

  def valid(self, row: int, column: int) -> bool:
    """Comprueba si una posición pertenece al tablero."""
    return 0 <= row < self.size and 0 <= column < self.size

  def neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
    """Devuelve las casillas adyacentes válidas."""
    row, column = cell
    return [
      (row + dr, column + dc)
      for dr in (-1, 0, 1)
      for dc in (-1, 0, 1)
      if (dr or dc) and self.valid(row + dr, column + dc)
    ]

  def place_mines(self, first: tuple[int, int]) -> None:
    """Distribuye minas después del primer movimiento."""
    choices = [
      (row, column)
      for row in range(self.size)
      for column in range(self.size)
      if (row, column) != first
    ]
    self.mines = set(self.random.sample(choices, self.mine_count))
    self.started = True

  def number(self, cell: tuple[int, int]) -> int:
    """Cuenta minas adyacentes a una casilla."""
    return sum(neighbor in self.mines for neighbor in self.neighbors(cell))

  def reveal(self, row: int, column: int) -> str:
    """Abre una casilla y propaga espacios vacíos."""
    cell = (row, column)
    if not self.valid(row, column) or self.finished:
      return "Casilla fuera de rango o partida finalizada."
    if cell in self.flags or cell in self.opened:
      return "Esa casilla ya está marcada o abierta."
    if not self.started:
      self.place_mines(cell)
    if cell in self.mines:
      self.finished = True
      return "Había una mina."
    self._flood(cell)
    self.won = len(self.opened) == self.size * self.size - self.mine_count
    self.finished = self.won
    return "Casilla abierta." if not self.won else "Tablero completado."

  def _flood(self, first: tuple[int, int]) -> None:
    pending = deque([first])
    while pending:
      cell = pending.popleft()
      if cell in self.opened or cell in self.flags:
        continue
      self.opened.add(cell)
      if self.number(cell) == 0:
        pending.extend(neighbor for neighbor in self.neighbors(cell) if neighbor not in self.mines)

  def toggle_flag(self, row: int, column: int) -> str:
    """Marca o desmarca una casilla cerrada."""
    cell = (row, column)
    if not self.valid(row, column) or self.finished or cell in self.opened:
      return "No se puede marcar esa casilla."
    if cell in self.flags:
      self.flags.remove(cell)
      return "Bandera retirada."
    self.flags.add(cell)
    return "Bandera colocada."

  def hint(self) -> str:
    """Sugiere un número observable para razonar, sin consultar minas ocultas."""
    if self.finished:
      return "La partida ya terminó."
    self.hints_used += 1
    clues = [
      (cell, self.number(cell), len(set(self.neighbors(cell)) - self.opened))
      for cell in self.opened
      if self.number(cell) > 0
    ]
    clues = [item for item in clues if item[2] > 0]
    if not clues:
      return "Abre una casilla y observa los números que aparezcan."
    cell, number, hidden = min(clues, key=lambda item: item[2])
    return (
      f"Revisa el {number} en ({cell[0] + 1}, {cell[1] + 1}): "
      f"tiene {hidden} vecinas cerradas."
    )

  def visible_board(self) -> list[list[str]]:
    """Construye la vista del jugador sin revelar minas ocultas."""
    return [
      [self._visible_cell((row, column)) for column in range(self.size)]
      for row in range(self.size)
    ]

  def _visible_cell(self, cell: tuple[int, int]) -> str:
    if cell in self.flags:
      return "F"
    if cell in self.opened:
      return str(self.number(cell))
    if self.finished and cell in self.mines:
      return "*"
    return "·"
