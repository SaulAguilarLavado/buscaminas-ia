"""Genera gráficas PNG con Matplotlib a partir de métricas guardadas."""

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "artifacts" / ".matplotlib"
CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CONFIG))
os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


NAVY = "#17233b"
TEAL = "#087e83"
ORANGE = "#d86a35"


def _read_metrics(path: Path) -> dict:
  if not path.is_file():
    raise FileNotFoundError(f"Falta {path.name}. Ejecuta train-agent, compare-policies y train primero.")
  return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, path: Path) -> None:
  fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
  plt.close(fig)


def _bar_chart(path: Path, title: str, subtitle: str, labels: list[str],
               values: list[float], colors: list[str], note: str) -> None:
  fig, ax = plt.subplots(figsize=(10, 5.5))
  bars = ax.barh(labels, values, color=colors, height=0.55)
  ax.invert_yaxis()
  ax.set_xlim(0, 1.05)
  ax.xaxis.set_major_formatter(PercentFormatter(1))
  ax.grid(axis="x", alpha=0.18)
  ax.set_axisbelow(True)
  ax.set_title(title, loc="left", fontsize=19, fontweight="bold", color=NAVY, pad=28)
  ax.text(0, 1.04, subtitle, transform=ax.transAxes, color="#66758c", fontsize=11)
  for bar, value in zip(bars, values):
    ax.text(value + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{value:.1%}".replace(".", ","), va="center", fontweight="bold", color=NAVY)
  for edge in ("top", "right", "left"):
    ax.spines[edge].set_visible(False)
  fig.text(0.12, 0.04, note, fontsize=10, color="#66758c")
  fig.subplots_adjust(left=0.25, right=0.94, top=0.78, bottom=0.19)
  _save(fig, path)


GREY = "#9aa5b8"
POLICIES = [
  ("aleatoria", "Aleatoria", GREY),
  ("probabilidad_minima_desempate_aleatorio",
   "Prob. mínima\n(desempate aleatorio)", NAVY),
  ("probabilidad_minima_desempate_fijo", "Prob. mínima\n(desempate fijo)", NAVY),
  ("q_sin_entrenar", "Q sin entrenar", ORANGE),
  ("q_entrenado", "Q entrenado", TEAL),
]
PANELS = [("todas_560", "Las 560 distribuciones"),
          ("retenidas_112", "112 distribuciones retenidas")]


def _pct(value: float) -> str:
  return f"{value:.1%}".replace(".", ",")


def _policies(path: Path, comparison: dict, manifest: dict) -> None:
  """Barras por política: media entre semillas y rango mínimo-máximo."""
  optimum = comparison["optimo_exacto_560"]
  seeds = comparison["configuracion"]["semillas_desempate"]
  fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharey=True)
  series = []
  for ax, (split, title) in zip(axes, PANELS):
    data = [comparison["politicas"][key][split] for key, _, _ in POLICIES]
    means = [item["tasa_media"] for item in data]
    low = [item["tasa_media"] - item["tasa_min"] for item in data]
    high = [item["tasa_max"] - item["tasa_media"] for item in data]
    ax.barh([label for _, label, _ in POLICIES], means, xerr=[low, high],
            color=[color for _, _, color in POLICIES], height=0.55,
            error_kw={"ecolor": "#444", "capsize": 4})
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    ax.axvline(0.90, color=NAVY, linestyle=":", label="Meta del proyecto: 90 %")
    if split == "todas_560":
      ax.axvline(optimum["tasa"], color=ORANGE, linestyle="--",
                 label=f"Óptimo exacto: {optimum['victorias']}/{optimum['distribuciones']}"
                       f" ({_pct(optimum['tasa'])})")
    ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, loc="left")
    for y, item in enumerate(data):
      ax.text(item["tasa_max"] + 0.012, y, _pct(item["tasa_media"]), va="center",
              fontweight="bold", color=NAVY)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.09), fontsize=9.5, frameon=False)
    for edge in ("top", "right"):
      ax.spines[edge].set_visible(False)
    for (key, _, _), item in zip(POLICIES, data):
      series.append({"conjunto": split, "politica": key, "media": item["tasa_media"],
                     "minimo": item["tasa_min"], "maximo": item["tasa_max"]})
  fig.suptitle("Victorias por política (mismas reglas y distribuciones)",
               x=0.01, ha="left", fontsize=18, fontweight="bold", color=NAVY)
  fig.text(0.01, 0.015,
           f"Barra = media de {seeds} semillas de desempate; barra de error = mínimo y "
           "máximo (distintos desempates). El óptimo exacto aplica solo a las 560 "
           "distribuciones.",
           fontsize=9.5, color="#66758c")
  fig.subplots_adjust(left=0.17, right=0.97, top=0.85, bottom=0.24, wspace=0.08)
  _save(fig, path)
  manifest[path.name] = {
    "series": series,
    "lineas": {"meta": 0.90, "optimo_exacto_560": optimum["tasa"]},
    "semillas_desempate": seeds,
  }


def _read_curve(path: Path) -> list[dict]:
  if not path.is_file():
    raise FileNotFoundError(f"Falta {path.name}. Ejecuta train-agent primero.")
  with path.open(newline="", encoding="utf-8") as handle:
    return [{**row, "episodio": int(row["episodio"]),
             "tasa_media": float(row["tasa_media"]),
             "tasa_min": float(row["tasa_min"]), "tasa_max": float(row["tasa_max"])}
            for row in csv.DictReader(handle)]


def _merge_curve(main: list[dict], early: list[dict]) -> list[dict]:
  """Une el CSV principal con el sondeo episodio a episodio.

  Las filas que ambos archivos comparten (episodios 0 y 25) deben ser idénticas; si no,
  se aborta. En esos episodios se conserva la fila del CSV principal.
  """
  index = {(r["corrida_semilla"], r["episodio"], r["conjunto"]): r for r in main}
  extra = []
  shared = set()
  for row in early:
    key = (row["corrida_semilla"], row["episodio"], row["conjunto"])
    other = index.get(key)
    if other is None:
      extra.append({**row, "fuente": "learning_curve_early.csv", "en_sondeo": True})
    elif any(other[f] != row[f] for f in ("tasa_media", "tasa_min", "tasa_max")):
      raise ValueError(f"learning_curve_early.csv no coincide con learning_curve.csv: {key}")
    else:
      shared.add(key)
  merged = [{**row, "fuente": "learning_curve.csv",
             "en_sondeo": (row["corrida_semilla"], row["episodio"],
                           row["conjunto"]) in shared} for row in main]
  return merged + extra


def _curve(path: Path, rows: list[dict], comparison: dict, manifest: dict) -> None:
  """Curva en tres paneles: sondeo episodio a episodio, tramo inicial y rango completo."""
  main_file = "learning_curve.csv"
  runs = len({row["corrida_semilla"] for row in rows})
  ties = rows[0]["semillas_desempate"]
  optimum = comparison["optimo_exacto_560"]["tasa"]
  random_mean = comparison["politicas"]["aleatoria"]["todas_560"]["tasa_media"]
  main_episodes = sorted({r["episodio"] for r in rows if r["fuente"] == main_file})
  episodes_all = sorted({row["episodio"] for row in rows})
  gaps = [b - a for a, b in zip(main_episodes, main_episodes[1:])]
  first_step = min(gaps)
  mid_limit = main_episodes[-1]
  for index, gap in enumerate(gaps):
    if gap != first_step:
      mid_limit = main_episodes[index]
      break
  later_step = max(gaps)
  early_limit = max((r["episodio"] for r in rows if r["en_sondeo"]), default=None)
  series = []
  points = []
  for split, label, color in (("retenidas_112", "112 retenidas", TEAL),
                              ("todas_560", "Las 560 distribuciones", NAVY)):
    stats = []
    for episode in episodes_all:
      selected = [row for row in rows
                  if row["conjunto"] == split and row["episodio"] == episode]
      rates = [row["tasa_media"] for row in selected]
      stats.append((episode, sum(rates) / len(rates), min(rates), max(rates),
                    len(rates), selected[0]["fuente"]))
    series.append((label, color, stats))
    points.extend({"conjunto": split, "episodio": e, "media": m, "minimo": lo,
                   "maximo": hi, "corridas": n, "fuente": f}
                  for e, m, lo, hi, n, f in stats)
  panels = []
  if early_limit:
    panels.append((early_limit, f"Episodios 0 a {early_limit} (cada episodio)"))
  mid_title = f"Episodios 0 a {mid_limit} (cada {first_step}"
  mid_title += f"; cada episodio hasta el {early_limit})" if early_limit else ")"
  panels.append((mid_limit, mid_title))
  panels.append((episodes_all[-1],
                 f"Episodios 0 a {episodes_all[-1]} (cada {later_step} tras el {mid_limit})"))
  fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 6.6), sharey=True)
  for position, (ax, (limit, title)) in enumerate(zip(axes, panels)):
    dense = position == 0
    for label, color, stats in series:
      shown = [item for item in stats if item[0] <= limit]
      x = [item[0] for item in shown]
      ax.plot(x, [item[1] for item in shown], color=color, linewidth=2, marker="o",
              markersize=4 if dense else 2.5, label=label)
      ax.fill_between(x, [item[2] for item in shown], [item[3] for item in shown],
                      color=color, alpha=0.15)
    ax.axhline(optimum, color=ORANGE, linestyle="--",
               label=f"Óptimo exacto en las 560: {_pct(optimum)}")
    ax.axhline(0.90, color=NAVY, linestyle=":", label="Meta del proyecto: 90 %")
    ax.axhline(random_mean, color=GREY, linestyle="-.",
               label=f"Aleatoria (media, 560): {_pct(random_mean)}")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-limit * 0.02, limit * 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Episodios de entrenamiento")
    ax.grid(alpha=0.18)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=NAVY)
    for edge in ("top", "right"):
      ax.spines[edge].set_visible(False)
  axes[0].set_ylabel("Tasa de victorias (política voraz, sin exploración)")
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10,
             bbox_to_anchor=(0.5, 0.055))
  fig.suptitle("Curva de aprendizaje del agente Q", x=0.01, ha="left", fontsize=18,
               fontweight="bold", color=NAVY)
  note = (f"{runs} corridas independientes; cada punto = media de {ties} semillas de "
          "desempate; línea = media entre corridas; banda = mínimo-máximo entre corridas.")
  if early_limit:
    note += (f" Episodios 1 a {early_limit - 1}: repetición de los primeros episodios con "
             f"el calendario de exploración original; el episodio {early_limit} coincide "
             "exactamente con learning_curve.csv.")
  fig.text(0.01, 0.012, note, fontsize=8.8, color="#66758c", wrap=True)
  fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.27, wspace=0.05)
  _save(fig, path)
  manifest[path.name] = {
    "puntos": points, "corridas": runs, "semillas_desempate": ties,
    "zoom_hasta_episodio": mid_limit, "episodio_por_episodio_hasta": early_limit,
    "lineas": {"optimo_exacto_560": optimum, "meta": 0.90,
               "aleatoria_media_560": random_mean}}


def _classifier(path: Path, metrics: dict) -> None:
  test = metrics["metricas_prueba"]
  _bar_chart(
    path, "Clasificador de casillas: conjunto de prueba",
    "Métricas para la clase mina (safe = 0)",
    ["Precisión", "Recall", "F1"],
    [float(test["precision_minas"]), float(test["recall_minas"]),
     float(test["f1_minas"])],
    [TEAL, ORANGE, NAVY],
    "El recall bajo indica que muchas minas no son detectadas por el clasificador.",
  )


def _confusion(path: Path, metrics: dict) -> None:
  matrix = metrics["metricas_prueba"]["matriz_confusion_0_1"]
  proportions = [[value / sum(row) for value in row] for row in matrix]
  fig, ax = plt.subplots(figsize=(9, 6))
  image = ax.imshow(proportions, cmap="Blues", vmin=0, vmax=1)
  ax.set_xticks([0, 1], ["Mina (0)", "Segura (1)"])
  ax.set_yticks([0, 1], ["Mina (0)", "Segura (1)"])
  ax.set_xlabel("Predicción")
  ax.set_ylabel("Clase real")
  ax.set_title("Matriz de confusión: conjunto de prueba", fontweight="bold", pad=20)
  for row in range(2):
    for col in range(2):
      color = "white" if proportions[row][col] > 0.5 else NAVY
      count = f"{matrix[row][col]:,}".replace(",", " ")
      rate = f"{proportions[row][col]:.1%}".replace(".", ",")
      ax.text(col, row, f"{count}\n({rate})",
              ha="center", va="center", fontsize=16, fontweight="bold", color=color)
  fig.colorbar(image, ax=ax, format=PercentFormatter(1), label="Porcentaje por clase real")
  fig.text(0.12, 0.03, "Mina clasificada como segura: error peligroso para una pista.",
           fontsize=10, color=ORANGE)
  fig.subplots_adjust(bottom=0.17)
  _save(fig, path)


def _archive_figure(path: Path, history: Path) -> None:
  """Conserva una gráfica previa antes de reemplazarla."""
  if not path.is_file():
    return
  stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
  destination = history / f"{stamp}-figuras"
  destination.mkdir(parents=True, exist_ok=True)
  shutil.copy2(path, destination / path.name)


def generate_charts(root: Path) -> list[str]:
  """Guarda las gráficas PNG en artifacts/figures y su manifiesto de datos."""
  source = root / "artifacts"
  comparison = _read_metrics(source / "policy_comparison.json")
  curve = _read_curve(source / "learning_curve.csv")
  early_path = source / "learning_curve_early.csv"
  curve = _merge_curve(curve, _read_curve(early_path) if early_path.is_file() else [])
  classifier = _read_metrics(source / "model_metrics.json")
  destination = source / "figures"
  destination.mkdir(parents=True, exist_ok=True)
  for name in ("victorias_agente.png", "curva_aprendizaje.png"):
    _archive_figure(destination / name, source / "history")
  manifest: dict = {}
  _policies(destination / "victorias_agente.png", comparison, manifest)
  _curve(destination / "curva_aprendizaje.png", curve, comparison, manifest)
  _classifier(destination / "metricas_clasificador.png", classifier)
  _confusion(destination / "matriz_confusion.png", classifier)
  (destination / "charts_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2))
  return [str(destination / name) for name in (
    "victorias_agente.png", "curva_aprendizaje.png",
    "metricas_clasificador.png", "matriz_confusion.png")]
