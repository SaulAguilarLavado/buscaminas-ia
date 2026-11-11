# Buscaminas adaptativo

Prototipo de consola para practicar Buscaminas con pistas explicativas y dificultad
adaptativa. Incluye preparación del dataset público y un clasificador exploratorio
de casillas. El clasificador todavía no se usa para recomendar jugadas al estudiante:
su sensibilidad a las minas debe mejorar antes de integrarlo.

## Configuración en PyCharm

1. Abre esta carpeta como proyecto.
2. Configura un intérprete Python 3.12 o posterior.
3. En la terminal de PyCharm ejecuta `python -m pip install -r requirements.txt`.

La carga sigue la opción **Code → kagglehub - pandas DataFrame** de Kaggle. Se usa
`minesweeper_dataset.csv` como `file_path`; KaggleHub gestiona su caché. No necesitas
copiar el CSV al proyecto. También puedes pasar un CSV existente con `--dataset`.

## Comandos

```bash
python main.py prepare
python main.py train
python main.py train-agent --episodes 10000 --evaluation 1000
python main.py demo-agent --seed 42
python main.py play
```

`prepare` valida las 26 columnas, cuenta filas nulas y duplicadas, y guarda solo un
resumen en `artifacts/data_summary.json`. La vista previa del portal Kaggle muestra
10 de 26 columnas; esa limitación visual no reduce el dataset que lee el programa.
`train` usa 24 variables visibles, divide 70/15/15 y guarda un modelo exploratorio
y sus métricas en `artifacts/`. `play` inicia el juego de consola; después de cada
cinco partidas ajusta como máximo un nivel según victorias y pistas solicitadas.
Las pistas señalan números del tablero para razonar y no consultan las minas ocultas.

`train-agent` entrena un primer agente Q-Learning en tableros 5x5 con 3 minas.
Compara sus victorias con una política aleatoria usando las mismas semillas y
guarda `artifacts/rl_q_values.json` y `artifacts/rl_metrics.json`. El agente
entrenado todavía no controla la partida interactiva; es un experimento separado.
`demo-agent` reproduce una partida automática usando los valores aprendidos.

## Datos y alcance

- Fuente: [Dataset minesweeper game de Michele Chierchia](https://www.kaggle.com/datasets/michelechierchia/dataset-minesweeper-game).
- No se recogen nombres, correos ni datos de menores.
- Las métricas del modelo miden clasificación de casillas, no aprendizaje escolar.
- La adaptación educativa con estudiantes aún requiere una prueba supervisada.
