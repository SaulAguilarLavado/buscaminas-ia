"""Punto de entrada para el análisis y el prototipo adaptativo de Buscaminas."""

import argparse
import json
from pathlib import Path

from adaptive_game import AdaptiveSession, Minesweeper
from dataset_pipeline import prepare_dataset, train_classifier


ROOT = Path(__file__).resolve().parent


def show_board(game: Minesweeper) -> None:
  """Muestra únicamente el tablero visible para el jugador."""
  print("   " + " ".join(f"{column:2}" for column in range(1, game.size + 1)))
  for row, cells in enumerate(game.visible_board(), start=1):
    print(f"{row:2} " + " ".join(f"{cell:>2}" for cell in cells))


def run_game(session: AdaptiveSession, seed: int | None) -> None:
  """Ejecuta una partida y registra sus resultados para la adaptación."""
  game = Minesweeper.for_level(session.level, seed)
  print(f"\nNivel {session.level}: {game.size}x{game.size}, {game.mine_count} minas")
  print("Comandos: abrir fila columna | bandera fila columna | pista | salir")
  while not game.finished:
    show_board(game)
    command = input("> ").strip().lower().split()
    if command == ["salir"]:
      return
    print(handle_command(game, command))
  show_board(game)
  print("¡Ganaste!" if game.won else "Partida terminada: encontraste una mina.")
  print(session.record(game.won, game.hints_used))


def handle_command(game: Minesweeper, command: list[str]) -> str:
  """Valida y ejecuta una acción ingresada por el jugador."""
  if command == ["pista"]:
    return game.hint()
  if len(command) != 3 or command[0] not in {"abrir", "bandera"}:
    return "Comando inválido. Usa abrir, bandera, pista o salir."
  if not all(value.isdigit() for value in command[1:]):
    return "Fila y columna deben ser números positivos."
  row, column = (int(value) - 1 for value in command[1:])
  if command[0] == "abrir":
    return game.reveal(row, column)
  return game.toggle_flag(row, column)


def play(seed: int | None) -> None:
  """Inicia partidas sucesivas con dificultad adaptativa."""
  session = AdaptiveSession()
  while True:
    run_game(session, seed)
    if input("¿Jugar otra partida? [s/N]: ").strip().lower() != "s":
      break


def main() -> None:
  """Procesa argumentos y ejecuta la función elegida."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("command", choices=["prepare", "train", "play"])
  parser.add_argument("--dataset", type=Path, default=None)
  parser.add_argument("--seed", type=int, default=None)
  args = parser.parse_args()
  if args.command == "play":
    play(args.seed)
    return
  if args.command == "prepare":
    result = prepare_dataset(ROOT, args.dataset)
  else:
    result = train_classifier(ROOT, args.dataset)
  print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
