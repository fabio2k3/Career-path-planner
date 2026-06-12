# Career Path Planner

Sistema que recibe un objetivo profesional en lenguaje natural y genera una secuencia óptima de cursos respetando prerrequisitos, usando A* sobre un grafo de dependencias y un LLM para interpretar el objetivo y evaluar la trayectoria resultante.

---

## Estructura del proyecto

```
proyecto_trayectoria/
├── .streamlit/
│   └── config.toml
├── data/
│   ├── dataset.json              # Catálogo de cursos, habilidades y perfiles
│   └── instances/
│       └── instances.json        # Instancias de prueba
├── experiments/
│   ├── run_experiments.py        # Experimento A* vs Greedy
│   ├── monte_carlo.py            # Simulación Monte Carlo
│   ├── generate_visualizations.py
│   ├── generate_report.py
│   └── results/                  # CSVs y figuras generados
├── scripts/
│   ├── generate_synthetic_dataset.py
│   ├── validate_dataset.py
│   └── fix_dag.py
├── src/
│   ├── graph.py                  # Representación del DAG
│   ├── search.py                 # A* y Greedy
│   └── llm_integration.py        # Parseador y evaluador LLM
├── tests/
├── .env                          # API key (crearla y NO subir al repositorio)
├── .gitignore
├── app.py                        # Interfaz visual (Streamlit)
├── LICENSE
├── main.py                       # Pipeline principal
├── README_APP.md
├── README.md
└── requirements.txt
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Configuración del LLM

El sistema usa la API de Hugging Face. Para obtener una clave gratuita:

1. Crear cuenta en [https://huggingface.co](https://huggingface.co)
2. Ir a **Settings → Access Tokens → New token** (tipo *Read*)
3. Copiar el token generado

Crear el archivo `.env` en la raíz del proyecto:

```
HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
---

## Uso

### Modo interactivo

```bash
python src/main.py
```

El sistema solicita el objetivo por teclado y muestra la trayectoria con su evaluación LLM.

### Modo directo con objetivo

```bash
python src/main.py --objetivo "Quiero trabajar como Data Scientist"
```

### Opciones disponibles

| Opción | Valores | Descripción |
|---|---|---|
| `--objetivo` | texto libre | Objetivo profesional en lenguaje natural |
| `--algoritmo` | `astar` \| `greedy` | Algoritmo de búsqueda (por defecto: `astar`) |
| `--sin-llm` | — | Ejecuta sin LLM, seleccionando el perfil manualmente |
| `--habilidades` | `hab1,hab2,...` | Habilidades iniciales del usuario |

### Ejemplos de prueba

```bash
# Data Scientist desde cero
python src/main.py --objetivo "Quiero ser Data Scientist"

# ML Engineer con habilidades iniciales
python src/main.py --objetivo "Quiero llevar modelos a producción como ML Engineer" \
                   --habilidades "python_basico,estadistica_basica,algebra_lineal"

# Backend Developer con Greedy
python src/main.py --objetivo "Quiero desarrollar APIs y microservicios" \
                   --algoritmo greedy

# Sin LLM, selección manual de perfil
python src/main.py --sin-llm --algoritmo astar
```

---

## Experimentos

```bash
# Experimento principal: A* vs Greedy en todas las instancias
python experiments/run_experiments.py

# Simulación Monte Carlo (200 corridas)
python experiments/monte_carlo.py --runs 200 --seed 42

# Generar visualizaciones
python experiments/generate_visualizations.py

# Generar informe ejecutivo en Markdown
python experiments/generate_report.py
```

Los resultados se guardan en `experiments/results/`.


> Para usar la interfaz visual del proyecto consulta el [`README_APP.md`](README_APP.md).
