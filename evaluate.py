"""Comparación reproducible de políticas y curva de aprendizaje.

Todas las políticas se evalúan con las mismas reglas (5x5, 3 minas, inicio seguro
central) sobre las 560 distribuciones posibles y, por separado, sobre las 112
distribuciones retenidas que el entrenamiento nunca usa.

Cada distribución se juega con una secuencia aleatoria nueva `Random(semilla)`, igual para
todas las distribuciones de una pasada. Así, para una semilla dada, la jugada depende solo
del tablero visible (el desempate es una función fija de lo observado) y el óptimo exacto
acota cada pasada. La variabilidad entre semillas refleja distintas reglas de desempate,
no el ruido de partidas independientes; las políticas deterministas dan variabilidad cero.
"""

import csv
import json
import shutil
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Callable, Iterable

from board_rules import (
  FIRST, MINES, SIZE, SPLIT_SEED, all_layouts, game_for_layout, split_layouts,
)
from optimal_solver import optimal_win_count
from rl_agent import (
  ALPHA, EPSILON_END, EPSILON_START, GAMMA, REWARD_LOSS, REWARD_PER_CELL,
  REWARD_WIN, Belief, available, choose, epsilon_at, keys, train_episode,
)

SPLITS = ("retenidas_112", "todas_560")
CURVE_FIELDS = ["corrida_semilla", "episodio", "conjunto", "n_distribuciones",
                "semillas_desempate", "tasa_media", "tasa_min", "tasa_max"]


@dataclass(frozen=True)
class Policy:
  """Política evaluable: elige una casilla cerrada con la información visible."""

  name: str
  pick: Callable
  needs_belief: bool = True


def _random(game, belief, rng):
  return rng.choice(available(game))


def _min_probability_random_ties(game, belief, rng):
  probabilities = belief.probabilities()
  lowest = min(probabilities.values())
  return rng.choice([c for c, p in probabilities.items() if p == lowest])


def _min_probability_fixed_ties(game, belief, rng):
  probabilities = belief.probabilities()
  lowest = min(probabilities.values())
  closed = lambda cell: sum(n not in game.opened for n in game.neighbors(cell))
  return max((c for c, p in probabilities.items() if p == lowest),
             key=lambda cell: (closed(cell), cell))


def q_policy(name: str, values: dict[str, float]) -> Policy:
  """Política voraz (epsilon = 0) sobre una tabla Q."""
  return Policy(name, lambda game, belief, rng: choose(
    keys(game, belief), values, rng, 0.0))


RANDOM = Policy("aleatoria", _random, needs_belief=False)
MIN_PROBABILITY = Policy("probabilidad_minima_desempate_aleatorio",
                         _min_probability_random_ties)
MIN_PROBABILITY_FIXED = Policy("probabilidad_minima_desempate_fijo",
                               _min_probability_fixed_ties)


def play_layout(layout: frozenset, policy: Policy, rng: Random) -> bool:
  """Juega una distribución completa y devuelve si se ganó."""
  game = game_for_layout(layout)
  belief = Belief(game) if policy.needs_belief else None
  while not game.finished:
    game.reveal(*policy.pick(game, belief, rng))
    if belief:
      belief.update()
  return game.won


def summarize(wins_per_seed: list[int], layouts: int) -> dict:
  """Resume victorias por semilla de desempate."""
  rates = [wins / layouts for wins in wins_per_seed]
  return {
    "distribuciones": layouts,
    "semillas_desempate": len(wins_per_seed),
    "victorias_por_semilla": wins_per_seed,
    "tasa_media": round(statistics.fmean(rates), 6),
    "desviacion": round(statistics.stdev(rates), 6) if len(rates) > 1 else 0.0,
    "tasa_min": round(min(rates), 6),
    "tasa_max": round(max(rates), 6),
  }


def evaluate_policy(policy: Policy, layouts: Iterable, tie_seeds: Iterable[int]) -> dict:
  """Evalúa una política en cada distribución, una pasada por semilla de desempate.

  Cada partida reinicia `Random(seed)`: no se comparte aleatoriedad entre distribuciones.
  """
  layouts = list(layouts)
  wins = []
  for seed in tie_seeds:
    wins.append(sum(play_layout(layout, policy, Random(seed)) for layout in layouts))
  return summarize(wins, len(layouts))


def archive_previous(artifacts: Path, names: list[str]) -> str | None:
  """Copia a artifacts/history/<fecha> los resultados que se van a reemplazar."""
  existing = [artifacts / name for name in names if (artifacts / name).is_file()]
  if not existing:
    return None
  stamp = datetime.fromtimestamp(min(p.stat().st_mtime for p in existing))
  destination = artifacts / "history" / stamp.strftime("%Y%m%d-%H%M%S")
  suffix = 1
  while destination.exists():
    destination = destination.with_name(f"{destination.name}-{suffix}")
    suffix += 1
  destination.mkdir(parents=True)
  for path in existing:
    shutil.copy2(path, destination / path.name)
  return str(destination)


def is_checkpoint(episode: int, episodes: int, checkpoint_every: int,
                  early_every: int, early_until: int) -> bool:
  """Checkpoints densos hasta `early_until` y luego cada `checkpoint_every`."""
  early = early_every > 0 and episode <= early_until and episode % early_every == 0
  return early or episode % checkpoint_every == 0 or episode == episodes


def _config(episodes: int, runs: int, checkpoint_every: int, tie_seeds: int,
            seed: int, split_seed: int, early_every: int, early_until: int) -> dict:
  train, heldout = split_layouts(split_seed)
  return {
    "tablero": f"{SIZE}x{SIZE}, {MINES} minas; inicio seguro central {FIRST}",
    "algoritmo": "Q-Learning tabular; estado = cubeta de probabilidad de mina + "
                 "vecinas cerradas (la probabilidad se calcula con Belief)",
    "alpha": ALPHA, "gamma": GAMMA,
    "epsilon_inicial": EPSILON_START, "epsilon_final": EPSILON_END,
    "recompensas": {"victoria": REWARD_WIN, "derrota": REWARD_LOSS,
                    "por_casilla_abierta": REWARD_PER_CELL},
    "episodios_por_corrida": episodes,
    "corridas": runs,
    "semillas_corridas": [seed + k for k in range(runs)],
    "semilla_division": split_seed,
    "distribuciones_entrenamiento": len(train),
    "distribuciones_retenidas": len(heldout),
    "checkpoint_cada": checkpoint_every,
    "checkpoint_temprano_cada": early_every,
    "checkpoint_temprano_hasta": early_until,
    "semillas_desempate_por_checkpoint": tie_seeds,
    "modelo_guardado": "primera corrida (semilla base)",
  }


def run_training(root: Path, episodes: int = 10000, runs: int = 10,
                 checkpoint_every: int = 500, tie_seeds: int = 5, seed: int = 42,
                 split_seed: int = SPLIT_SEED, early_every: int = 25,
                 early_until: int = 500) -> dict:
  """Entrena varias corridas con evaluación voraz periódica (curva de aprendizaje)."""
  if not (1 <= episodes <= 1_000_000 and 1 <= runs <= 100
          and 1 <= checkpoint_every <= episodes and 1 <= tie_seeds <= 1000
          and 0 <= early_every <= episodes and 0 <= early_until <= episodes):
    raise ValueError("Parámetros fuera de rango")
  artifacts = root / "artifacts"
  artifacts.mkdir(exist_ok=True)
  archived = archive_previous(artifacts, [
    "rl_metrics.json", "rl_q_values.json", "learning_curve.csv",
    "policy_comparison.json"])
  train, heldout = split_layouts(split_seed)
  evaluation_sets = {"retenidas_112": heldout, "todas_560": all_layouts()}
  rows: list[dict] = []
  run_reports = []
  primary_values: dict[str, float] = {}
  for k in range(runs):
    run_seed = seed + k
    values: dict[str, float] = defaultdict(float)
    rng = Random(run_seed)
    training_wins = 0

    def checkpoint(episode: int) -> None:
      policy = q_policy("q", values)
      for split, layouts in evaluation_sets.items():
        result = evaluate_policy(policy, layouts, range(tie_seeds))
        rows.append({
          "corrida_semilla": run_seed, "episodio": episode, "conjunto": split,
          "n_distribuciones": result["distribuciones"],
          "semillas_desempate": tie_seeds, "tasa_media": result["tasa_media"],
          "tasa_min": result["tasa_min"], "tasa_max": result["tasa_max"]})

    checkpoint(0)
    for index in range(episodes):
      training_wins += train_episode(
        values, rng.choice(train), rng, epsilon_at(index, episodes))
      if is_checkpoint(index + 1, episodes, checkpoint_every, early_every,
                       early_until):
        checkpoint(index + 1)
    final = {r["conjunto"]: r["tasa_media"] for r in rows
             if r["corrida_semilla"] == run_seed and r["episodio"] == episodes}
    run_reports.append({
      "semilla": run_seed, "victorias_entrenamiento": training_wins,
      "estados_aprendidos": len(values), "tasa_final": final})
    if k == 0:
      primary_values = dict(values)
    print(f"corrida {k + 1}/{runs} (semilla {run_seed}): final {final}", flush=True)
  (artifacts / "rl_q_values.json").write_text(json.dumps(primary_values, indent=2))
  with (artifacts / "learning_curve.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
  summary = {}
  for split in SPLITS:
    rates = [report["tasa_final"][split] for report in run_reports]
    summary[split] = {
      "media": round(statistics.fmean(rates), 6),
      "desviacion": round(statistics.stdev(rates), 6) if len(rates) > 1 else 0.0,
      "minimo": min(rates), "maximo": max(rates)}
  report = {
    "configuracion": _config(episodes, runs, checkpoint_every, tie_seeds, seed,
                             split_seed, early_every, early_until),
    "resultados_por_corrida": run_reports,
    "resumen_final_entre_corridas": summary,
    "curva": "learning_curve.csv",
    "respaldo_anterior": archived,
  }
  (artifacts / "rl_metrics.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2))
  return report


def compare_policies(root: Path, tie_seeds: int = 30) -> dict:
  """Compara cuatro políticas con las mismas reglas y distribuciones."""
  artifacts = root / "artifacts"
  metrics_path = artifacts / "rl_metrics.json"
  values_path = artifacts / "rl_q_values.json"
  if not (metrics_path.is_file() and values_path.is_file()):
    raise FileNotFoundError("Entrena primero con: python main.py train-agent")
  config = json.loads(metrics_path.read_text())["configuracion"]
  values = json.loads(values_path.read_text())
  _, heldout = split_layouts(config["semilla_division"])
  optimum_path = artifacts / "optimal_solution.json"
  if optimum_path.is_file():
    optimum = json.loads(optimum_path.read_text())["victorias_optimas"]
  else:
    optimum = optimal_win_count()[0]
  policies = [
    RANDOM, MIN_PROBABILITY, MIN_PROBABILITY_FIXED,
    q_policy("q_sin_entrenar", {}), q_policy("q_entrenado", values),
  ]
  sets = {"todas_560": all_layouts(), "retenidas_112": heldout}
  results = {}
  for policy in policies:
    results[policy.name] = {
      split: evaluate_policy(policy, layouts, range(tie_seeds))
      for split, layouts in sets.items()
    }
  best_pass = max(max(r["todas_560"]["victorias_por_semilla"]) for r in results.values())
  report = {
    "configuracion": {
      "tablero": config["tablero"], "semillas_desempate": tie_seeds,
      "semilla_division": config["semilla_division"],
      "distribuciones_retenidas": len(heldout),
      "modelo_q": "artifacts/rl_q_values.json (primera corrida de train-agent)",
      "q_sin_entrenar": "tabla vacía: todas las casillas empatan",
    },
    "optimo_exacto_560": {"victorias": optimum, "distribuciones": len(all_layouts()),
                          "tasa": round(optimum / len(all_layouts()), 6)},
    "aclaracion_semillas": (
      "Cada distribucion reinicia Random(semilla): para una semilla, la jugada depende "
      "solo del tablero visible, por lo que el optimo exacto acota cada pasada. La "
      "variabilidad entre semillas son reglas de desempate distintas, no ruido de "
      "partidas independientes."),
    "mejor_pasada_560": best_pass,
    "ninguna_pasada_supera_el_optimo": best_pass <= optimum,
    "politicas": results,
  }
  archive_previous(artifacts, ["policy_comparison.json"])
  (artifacts / "policy_comparison.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2))
  return report


def run_early_probe(root: Path, probe_episodes: int = 25) -> dict:
  """Repite los primeros episodios de cada corrida, evaluando tras cada uno.

  Usa los mismos ajustes que `train-agent` (guardados en rl_metrics.json), incluida la
  exploración del calendario completo de `episodios_por_corrida`: solo se reproducen los
  primeros `probe_episodes` episodios, pero con epsilon calculado sobre el total original.
  Antes de escribir, exige que los episodios 0 y `probe_episodes` coincidan exactamente
  con learning_curve.csv.
  """
  artifacts = root / "artifacts"
  metrics_path = artifacts / "rl_metrics.json"
  curve_path = artifacts / "learning_curve.csv"
  if not (metrics_path.is_file() and curve_path.is_file()):
    raise FileNotFoundError("Entrena primero con: python main.py train-agent")
  config = json.loads(metrics_path.read_text())["configuracion"]
  total = config["episodios_por_corrida"]
  tie_seeds = config["semillas_desempate_por_checkpoint"]
  if not 1 <= probe_episodes <= total:
    raise ValueError("probe_episodes debe estar entre 1 y los episodios de la corrida")
  train, heldout = split_layouts(config["semilla_division"])
  evaluation_sets = {"retenidas_112": heldout, "todas_560": all_layouts()}
  rows: list[dict] = []
  for run_seed in config["semillas_corridas"]:
    values: dict[str, float] = defaultdict(float)
    rng = Random(run_seed)

    def checkpoint(episode: int) -> None:
      policy = q_policy("q", values)
      for split, layouts in evaluation_sets.items():
        result = evaluate_policy(policy, layouts, range(tie_seeds))
        rows.append({
          "corrida_semilla": run_seed, "episodio": episode, "conjunto": split,
          "n_distribuciones": result["distribuciones"],
          "semillas_desempate": tie_seeds, "tasa_media": result["tasa_media"],
          "tasa_min": result["tasa_min"], "tasa_max": result["tasa_max"]})

    checkpoint(0)
    for index in range(probe_episodes):
      train_episode(values, rng.choice(train), rng, epsilon_at(index, total))
      checkpoint(index + 1)
  verification = verify_against_curve(rows, curve_path, {0, probe_episodes})
  if verification["discrepancias"]:
    raise ValueError(f"El sondeo no reproduce learning_curve.csv: {verification}")
  with (artifacts / "learning_curve_early.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
  report = {
    "descripcion": "Repeticion episodio a episodio de los primeros episodios de cada "
                   "corrida; epsilon calculado con el calendario completo",
    "episodios_repetidos": probe_episodes,
    "episodios_calendario_epsilon": total,
    "corridas": config["semillas_corridas"],
    "semillas_desempate": tie_seeds,
    "semilla_division": config["semilla_division"],
    "verificacion_contra_learning_curve_csv": verification,
    "archivo": "learning_curve_early.csv",
  }
  (artifacts / "learning_curve_early.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2))
  return report


def verify_against_curve(rows: list[dict], curve_path: Path,
                         episodes: set[int]) -> dict:
  """Compara filas del sondeo con learning_curve.csv en los episodios indicados."""
  with curve_path.open(newline="") as handle:
    reference = {(r["corrida_semilla"], int(r["episodio"]), r["conjunto"]): r
                 for r in csv.DictReader(handle)}
  checked = 0
  mismatches = []
  for row in rows:
    if row["episodio"] not in episodes:
      continue
    key = (str(row["corrida_semilla"]), row["episodio"], row["conjunto"])
    other = reference.get(key)
    checked += 1
    same = other is not None and all(
      float(other[field]) == float(row[field])
      for field in ("tasa_media", "tasa_min", "tasa_max", "n_distribuciones"))
    if not same:
      mismatches.append(list(key))
  return {"filas_comparadas": checked, "episodios": sorted(episodes),
          "discrepancias": len(mismatches), "ejemplos": mismatches[:3]}
