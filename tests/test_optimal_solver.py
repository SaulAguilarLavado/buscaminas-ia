"""Pruebas del solucionador exacto, la separación de distribuciones y la política.

Ejecutar desde la raíz del proyecto:  python -m unittest discover -s tests -t .
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from random import Random

from board_rules import FIRST, all_layouts, game_for_layout, split_layouts
from evaluate import (
  MIN_PROBABILITY_FIXED, MIN_PROBABILITY, RANDOM, evaluate_policy, play_layout,
  q_policy,
)
from evaluate import is_checkpoint, run_early_probe, run_training, verify_against_curve
from optimal_solver import optimal_win_count, validate_transitions
from rl_agent import Belief, available, keys


class BoardRulesTest(unittest.TestCase):
  def test_there_are_560_layouts_with_protected_center(self):
    layouts = all_layouts()
    self.assertEqual(len(layouts), 560)
    self.assertEqual(len(set(layouts)), 560)
    protected = {FIRST, *game_for_layout(layouts[0]).neighbors(FIRST)}
    self.assertTrue(all(not (layout & protected) for layout in layouts))

  def test_split_is_disjoint_and_complete(self):
    train, heldout = split_layouts()
    self.assertEqual((len(train), len(heldout)), (448, 112))
    self.assertFalse(set(train) & set(heldout))
    self.assertEqual(set(train) | set(heldout), set(all_layouts()))


class SolverTest(unittest.TestCase):
  def test_transitions_match_real_game(self):
    result = validate_transitions(sequences_per_layout=3)
    self.assertGreater(result["movimientos"], 1000)
    self.assertEqual(result["discrepancias"], 0)

  def test_exact_optimum_is_512_of_560(self):
    self.assertEqual(optimal_win_count(), (512, 560))

  def test_visible_board_policies_never_exceed_the_optimum(self):
    """Con una semilla fresca por partida, el desempate depende solo del tablero."""
    for policy in (RANDOM, MIN_PROBABILITY, MIN_PROBABILITY_FIXED, q_policy("q0", {})):
      for seed in range(3):
        wins = sum(play_layout(layout, policy, Random(seed))
                   for layout in all_layouts())
        self.assertLessEqual(wins, 512, f"{policy.name} semilla {seed}")

  def test_evaluate_policy_never_exceeds_the_optimum(self):
    """evaluate_policy reinicia la semilla por partida, así que 512 acota cada pasada."""
    for policy in (RANDOM, MIN_PROBABILITY, MIN_PROBABILITY_FIXED, q_policy("q0", {})):
      result = evaluate_policy(policy, all_layouts(), range(10))
      self.assertLessEqual(max(result["victorias_por_semilla"]), 512, policy.name)

  def test_evaluate_policy_matches_manual_fresh_seed_loop(self):
    manual = [sum(play_layout(layout, MIN_PROBABILITY, Random(seed))
                  for layout in all_layouts()) for seed in range(3)]
    result = evaluate_policy(MIN_PROBABILITY, all_layouts(), range(3))
    self.assertEqual(result["victorias_por_semilla"], manual)


class CheckpointScheduleTest(unittest.TestCase):
  def test_early_then_coarse_checkpoints(self):
    picks = [e for e in range(1, 10001) if is_checkpoint(e, 10000, 500, 25, 500)]
    self.assertEqual(picks[:20], list(range(25, 501, 25)))
    self.assertEqual(picks[20:], list(range(1000, 10001, 500)))
    self.assertEqual(
      [e for e in range(1, 10001) if is_checkpoint(e, 10000, 500, 0, 500)],
      list(range(500, 10001, 500)))


class EarlyProbeTest(unittest.TestCase):
  def test_probe_replays_run_training_checkpoints_exactly(self):
    """Con epsilon del calendario completo (60), el sondeo coincide episodio a episodio."""
    with tempfile.TemporaryDirectory() as folder:
      root = Path(folder)
      run_training(root, episodes=60, runs=2, checkpoint_every=5, tie_seeds=1,
                   seed=7, early_every=1, early_until=5)
      report = run_early_probe(root, probe_episodes=5)
      self.assertEqual(report["verificacion_contra_learning_curve_csv"]["discrepancias"], 0)
      read = lambda name: {(r["corrida_semilla"], r["episodio"], r["conjunto"]): r
                           for r in csv.DictReader(
                             (root / "artifacts" / name).read_text().splitlines())}
      full, early = read("learning_curve.csv"), read("learning_curve_early.csv")
      self.assertEqual(len(early), 2 * 6 * 2)  # corridas x episodios 0..5 x conjuntos
      for key, row in early.items():
        for field in ("tasa_media", "tasa_min", "tasa_max"):
          self.assertEqual(row[field], full[key][field], key)

  def test_verification_detects_a_mismatch(self):
    with tempfile.TemporaryDirectory() as folder:
      path = Path(folder) / "curve.csv"
      path.write_text("corrida_semilla,episodio,conjunto,n_distribuciones,"
                      "semillas_desempate,tasa_media,tasa_min,tasa_max\n"
                      "1,5,todas_560,560,1,0.5,0.5,0.5\n")
      row = {"corrida_semilla": 1, "episodio": 5, "conjunto": "todas_560",
             "n_distribuciones": 560, "tasa_media": 0.6, "tasa_min": 0.5,
             "tasa_max": 0.5}
      self.assertEqual(verify_against_curve([row], path, {5})["discrepancias"], 1)
      row["tasa_media"] = 0.5
      self.assertEqual(verify_against_curve([row], path, {5})["discrepancias"], 0)


class InformationOnlyTest(unittest.TestCase):
  def test_choice_depends_only_on_visible_board(self):
    """Dos distribuciones con el mismo tablero visible dan la misma jugada."""
    checked = 0
    for index in range(0, 560, 14):
      layout = all_layouts()[index]
      game = game_for_layout(layout)
      belief = Belief(game)
      if game.finished or len(belief.layouts) < 2:
        continue
      other = next(
        frozenset(c for c, bit in belief.bit.items() if mask & bit)
        for mask in belief.layouts if mask != sum(belief.bit[c] for c in layout))
      twin = game_for_layout(other)
      self.assertEqual(game.visible_board(), twin.visible_board())
      for policy in (MIN_PROBABILITY, MIN_PROBABILITY_FIXED, q_policy("q0", {})):
        picks = [policy.pick(g, Belief(g), Random(7)) for g in (game, twin)]
        self.assertEqual(picks[0], picks[1], policy.name)
      self.assertEqual(keys(game, belief), keys(twin, Belief(twin)))
      self.assertEqual(available(game), available(twin))
      checked += 1
    self.assertGreater(checked, 5)

  def test_min_probability_wins_the_optimal_count(self):
    rng = Random(0)
    wins = sum(play_layout(layout, MIN_PROBABILITY_FIXED, rng)
               for layout in all_layouts())
    self.assertEqual(wins, 512)


if __name__ == "__main__":
  unittest.main()
