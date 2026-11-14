# Buscaminas adaptativo

Prototipo de consola para practicar Buscaminas con pistas explicativas y dificultad
adaptativa. Incluye preparación del dataset público y un clasificador exploratorio
de casillas. El clasificador todavía no se usa para recomendar jugadas al estudiante:
su sensibilidad a las minas debe mejorar antes de integrarlo.

## Instalación y prueba

El repositorio es privado: cada integrante necesita acceso en GitHub antes de clonarlo.
Se requiere Python 3.12 o posterior. En macOS o Linux:

```bash
git clone https://github.com/SaulAguilarLavado/buscaminas-ia.git
cd buscaminas-ia
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python main.py demo-agent --seed 42
```

En Windows, crea el entorno con `py -m venv .venv` y usa
`.venv\Scripts\python.exe` en lugar de `.venv/bin/python`. Puedes abrir la carpeta
clonada en PyCharm y configurar `.venv` como intérprete del proyecto.
El modelo y las métricas finales ya están en el repositorio: `demo-agent` no requiere
reentrenar ni descargar el dataset. Para jugar manualmente, ejecuta
`.venv/bin/python main.py play` (o el equivalente de Windows); ese modo está separado
del agente entrenado.

La carga sigue la opción **Code → kagglehub - pandas DataFrame** de Kaggle. Se usa
`minesweeper_dataset.csv` como `file_path`; KaggleHub gestiona su caché. No necesitas
copiar el CSV al proyecto. También puedes pasar un CSV existente con `--dataset`.

## Comandos

```bash
python main.py prepare
python main.py train
python main.py verify-optimal
python main.py train-agent --episodes 10000 --runs 10 --checkpoint-every 500 --early-every 25
python main.py probe-early --probe-episodes 25
python main.py compare-policies --tie-seeds 30
python main.py demo-agent --seed 42
python main.py report-charts
python main.py play
```

`prepare` valida las 26 columnas, cuenta filas nulas y duplicadas, y guarda solo un
resumen en `artifacts/data_summary.json`. La vista previa del portal Kaggle muestra
10 de 26 columnas; esa limitación visual no reduce el dataset que lee el programa.
`train` usa 24 variables visibles, divide 70/15/15 y guarda un modelo exploratorio
y sus métricas en `artifacts/`. `play` inicia el juego de consola; después de cada
cinco partidas ajusta como máximo un nivel según victorias y pistas solicitadas.
Las pistas señalan números del tablero para razonar y no consultan las minas ocultas.

## Experimento del agente (5x5, 3 minas, inicio seguro)

Reglas de referencia: la casilla central se abre primero y ella y sus ocho vecinas
nunca tienen mina, así que existen exactamente C(16, 3) = 560 distribuciones de minas.
`play` no usa estas reglas ni el agente; sigue siendo el juego de consola de siempre.

- `verify-optimal` calcula con una búsqueda exacta (expectimax en `optimal_solver.py`)
  el máximo de distribuciones que cualquier jugador que solo ve el tablero puede ganar,
  y lo valida contra la clase `Minesweeper` real. Resultado en
  `artifacts/optimal_solution.json`. Pruebas: `python -m unittest discover -s tests -t .`.
- `train-agent` entrena `--runs` corridas independientes de Q-Learning tabular con
  448 distribuciones de entrenamiento y deja 112 retenidas que nunca se usan para
  entrenar. Cada `--early-every` episodios hasta el 500 y luego cada `--checkpoint-every` evalúa la política voraz (sin
  exploración) en las 112 retenidas y, por separado, en las 560, y lo guarda en
  `artifacts/learning_curve.csv`. Guarda `artifacts/rl_metrics.json` (ajustes y
  resultados por corrida) y `artifacts/rl_q_values.json` (modelo de la primera corrida).
- `probe-early` repite solo los primeros `--probe-episodes` episodios de cada corrida,
  evaluando tras cada episodio, con el mismo entrenamiento y semillas pero con epsilon
  calculado sobre el calendario completo de `train-agent`. Antes de guardar
  `artifacts/learning_curve_early.csv` y `.json`, exige que los episodios 0 y 25 coincidan
  exactamente con `learning_curve.csv`; la gráfica de la curva usa ambos archivos.
- `compare-policies` evalúa con las mismas reglas la política aleatoria, la de
  probabilidad mínima (sin aprendizaje; con desempate aleatorio y con desempate fijo), la
  Q sin entrenar y la Q entrenada, en las 560 y en las 112 retenidas, con `--tie-seeds`
  semillas de desempate. Resultado en `artifacts/policy_comparison.json`.
- El estado del agente incluye la probabilidad de mina calculada con las pistas visibles
  (`Belief` en `rl_agent.py`). Qué aporta el aprendizaje respecto a ese cálculo es lo que
  mide la comparación; no se asume.
- Los resultados finales seleccionados de `artifacts/` están versionados. La carpeta
  `artifacts/history/` conserva versiones anteriores solo en el equipo local; tampoco
  se versiona la caché `artifacts/.matplotlib/`.
- `demo-agent` reproduce una partida automática con el modelo guardado.
- `report-charts` genera con Matplotlib, en `artifacts/figures/`, las gráficas de
  políticas, curva de aprendizaje, clasificador y matriz de confusión, y un
  `charts_manifest.json` con los valores exactos graficados. Ejecuta antes
  `verify-optimal`, `train-agent`, `compare-policies` y `train` si faltan sus archivos.
- El agente no controla la partida interactiva; es un experimento separado.

`prepare`, `train` y `play`: ver arriba. `train-agent` ya no admite `--evaluation`.

## Datos y alcance

- Fuente: [Dataset minesweeper game de Michele Chierchia](https://www.kaggle.com/datasets/michelechierchia/dataset-minesweeper-game).
- No se recogen nombres, correos ni datos de menores.
- Las métricas del modelo miden clasificación de casillas, no aprendizaje escolar.
- La adaptación educativa con estudiantes aún requiere una prueba supervisada.
