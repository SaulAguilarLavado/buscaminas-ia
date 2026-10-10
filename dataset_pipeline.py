"""Carga, limpia y evalúa el dataset público de Buscaminas."""

import json
from pathlib import Path

import joblib
import kagglehub
import pandas as pd
from kagglehub import KaggleDatasetAdapter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


DATASET_HANDLE = "michelechierchia/dataset-minesweeper-game"
DATASET_FILE = "minesweeper_dataset.csv"
CELL_COLUMNS = [
  f"cell_{row}_{column}"
  for row in range(-2, 3)
  for column in range(-2, 3)
  if (row, column) != (0, 0)
]
EXPECTED_COLUMNS = set(CELL_COLUMNS + ["global_density", "safe"])


def load_source(supplied: Path | None) -> pd.DataFrame:
  """Carga las 26 columnas mediante KaggleHub o desde un CSV local."""
  dtypes = {column: "int8" for column in CELL_COLUMNS + ["safe"]}
  dtypes["global_density"] = "float64"
  if supplied:
    if not supplied.is_file() or supplied.suffix.lower() != ".csv":
      raise ValueError("--dataset debe apuntar a un archivo CSV existente")
    return pd.read_csv(supplied, dtype=dtypes)
  return kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    DATASET_HANDLE,
    DATASET_FILE,
    pandas_kwargs={"dtype": dtypes},
  )


def validate(frame: pd.DataFrame) -> None:
  """Comprueba esquema y rangos antes de preparar el dataset."""
  if set(frame.columns) != EXPECTED_COLUMNS:
    raise ValueError("El CSV no tiene las 26 columnas esperadas")
  if not frame["safe"].isin([0, 1]).all():
    raise ValueError("La columna safe solo debe contener 0 o 1")
  if not frame[CELL_COLUMNS].isin(range(-2, 9)).all().all():
    raise ValueError("Las pistas deben estar entre -2 y 8")
  if not frame["global_density"].between(0, 1).all():
    raise ValueError("global_density debe estar entre 0 y 1")


def load_clean(supplied: Path | None) -> tuple[pd.DataFrame, dict]:
  """Valida estructura, cuenta problemas y elimina filas idénticas."""
  frame = load_source(supplied)
  validate(frame)
  summary = {
    "registros_originales": int(len(frame)),
    "columnas": int(len(frame.columns)),
    "nulos": int(frame.isna().sum().sum()),
    "duplicados": int(frame.duplicated().sum()),
  }
  frame = frame.drop_duplicates().reset_index(drop=True)
  summary["registros_limpios"] = int(len(frame))
  summary["clases_limpias"] = {
    str(key): int(value) for key, value in frame["safe"].value_counts().items()
  }
  return frame, summary


def save_json(root: Path, name: str, values: dict) -> None:
  """Guarda un reporte legible en la carpeta artifacts."""
  destination = root / "artifacts"
  destination.mkdir(exist_ok=True)
  (destination / name).write_text(json.dumps(values, ensure_ascii=False, indent=2))


def prepare_dataset(root: Path, supplied: Path | None) -> dict:
  """Revisa y limpia los datos sin duplicar el CSV en el proyecto."""
  _, summary = load_clean(supplied)
  save_json(root, "data_summary.json", summary)
  return summary


def split_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Separa 70/15/15 con semilla y estratificación."""
  train_val, test = train_test_split(
    frame, test_size=0.15, stratify=frame["safe"], random_state=42
  )
  train, validation = train_test_split(
    train_val, test_size=3 / 17, stratify=train_val["safe"], random_state=42
  )
  return train, validation, test


def evaluate(model: HistGradientBoostingClassifier, data: pd.DataFrame) -> dict:
  """Calcula métricas de minas, la clase de mayor riesgo."""
  truth = data["safe"]
  predicted = model.predict(data[CELL_COLUMNS])
  precision, recall, f1, _ = precision_recall_fscore_support(
    truth, predicted, labels=[0], zero_division=0
  )
  return {
    "exactitud_base_siempre_segura": round(float((truth == 1).mean()), 4),
    "exactitud_modelo": round(float(accuracy_score(truth, predicted)), 4),
    "precision_minas": round(float(precision[0]), 4),
    "recall_minas": round(float(recall[0]), 4),
    "f1_minas": round(float(f1[0]), 4),
    "matriz_confusion_0_1": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
  }


def train_classifier(root: Path, supplied: Path | None) -> dict:
  """Entrena un clasificador exploratorio sobre 24 pistas visibles."""
  frame, summary = load_clean(supplied)
  train, validation, test = split_data(frame)
  sample = train.sample(n=min(200_000, len(train)), random_state=42)
  model = HistGradientBoostingClassifier(
    max_iter=80, max_leaf_nodes=31, max_depth=6,
    learning_rate=0.08, l2_regularization=1.0,
    early_stopping=True, random_state=42,
  )
  model.fit(sample[CELL_COLUMNS], sample["safe"])
  summary.update({
    "entrenamiento": int(len(train)),
    "validacion": int(len(validation)),
    "prueba": int(len(test)),
    "muestra_entrenada": int(len(sample)),
    "variables_visibles": len(CELL_COLUMNS),
    "metricas_validacion": evaluate(model, validation),
    "metricas_prueba": evaluate(model, test),
  })
  destination = root / "artifacts"
  destination.mkdir(exist_ok=True)
  joblib.dump(model, destination / "cell_classifier.joblib")
  save_json(root, "model_metrics.json", summary)
  return summary
