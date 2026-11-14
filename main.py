"""Punto de entrada para el análisis y el prototipo adaptativo de Buscaminas."""

import argparse
import json
from pathlib import Path

from adaptive_game import AdaptiveSession, Minesweeper
from dataset_pipeline import prepare_dataset, train_classifier
from evaluate import compare_policies, run_early_probe, run_training
from optimal_solver import verify_optimal
from rl_agent import demo_agent


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
  parser.add_argument(
    "command", choices=["prepare", "train", "train-agent", "demo-agent", "play",
                        "report-charts", "verify-optimal", "compare-policies",
                        "probe-early"]
  )
  parser.add_argument("--dataset", type=Path, default=None)
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--episodes", type=int, default=10000)
  parser.add_argument("--evaluation", type=int, default=None,
                      help="obsoleto: train-agent ahora evalúa todas las distribuciones")
  parser.add_argument("--runs", type=int, default=10,
                      help="train-agent: corridas de entrenamiento independientes")
  parser.add_argument("--checkpoint-every", type=int, default=500,
                      help="train-agent: episodios entre evaluaciones voraces")
  parser.add_argument("--early-every", type=int, default=25,
                      help="train-agent: episodios entre checkpoints hasta el 500 (0 = sin)")
  parser.add_argument("--probe-episodes", type=int, default=25,
                      help="probe-early: primeros episodios a repetir uno a uno")
  parser.add_argument("--tie-seeds", type=int, default=None,
                      help="semillas de desempate (train-agent: 5; compare-policies: 30)")
  args = parser.parse_args()
  if args.command == "play":
    play(args.seed)
    return
  if args.command == "demo-agent":
    print(demo_agent(ROOT, args.seed or 42))
    return
  if args.command == "report-charts":
    from reports.plots import generate_charts

    print(json.dumps(generate_charts(ROOT), ensure_ascii=False, indent=2))
    return
  if args.command == "verify-optimal":
    print(json.dumps(verify_optimal(ROOT), ensure_ascii=False, indent=2))
    return
  if args.command == "probe-early":
    print(json.dumps(run_early_probe(ROOT, args.probe_episodes),
                     ensure_ascii=False, indent=2))
    return
  if args.command == "compare-policies":
    result = compare_policies(ROOT, args.tie_seeds or 30)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return
  if args.command == "train-agent":
    if args.evaluation is not None:
      print("Aviso: --evaluation está obsoleto y se ignora.")
    result = run_training(ROOT, args.episodes, args.runs, args.checkpoint_every,
                          args.tie_seeds or 5, args.seed or 42,
                          early_every=args.early_every)
  elif args.command == "prepare":
    result = prepare_dataset(ROOT, args.dataset)
  else:
    result = train_classifier(ROOT, args.dataset)
  print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
  main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
