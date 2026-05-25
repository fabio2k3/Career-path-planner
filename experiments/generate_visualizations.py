"""
generate_visualizations.py

Genera las visualizaciones del análisis experimental.

Gráficos:
  1. Nodos expandidos A* vs Greedy (escala log) — eficiencia computacional
  2. Puntuaciones LLM por instancia y perfil
  3. Cursos y semanas por instancia — calidad de solución
  4. Tiempo de búsqueda A* vs Greedy
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR  = Path(__file__).parent / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# *** Paleta de colores ***
C_ASTAR  = "#2563EB"   # azul
C_GREEDY = "#F59E0B"   # ambar
C_LLM    = "#10B981"   # verde
C_DS     = "#6366F1"   # indigo (data scientist)
C_BE     = "#EC4899"   # rosa   (backend)
C_ML     = "#F97316"   # naranja (ml engineer)

PERFIL_COLOR = {
    "data_scientist":   C_DS,
    "backend_developer": C_BE,
    "ml_engineer":      C_ML,
}

PERFIL_LABEL = {
    "data_scientist":   "Data Scientist",
    "backend_developer": "Backend Dev",
    "ml_engineer":      "ML Engineer",
}

def leer_csv(nombre):
    path = RESULTS_DIR / nombre
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))

# *** Gráfico 1: Nodos expandidos A* vs Greedy ***

def grafico_nodos(sin_llm):
    astar_rows  = [r for r in sin_llm if "A*" in r["algoritmo"]]
    greedy_rows = [r for r in sin_llm if "Greedy" in r["algoritmo"]]

    ids      = [r["instancia_id"] for r in astar_rows]
    n_astar  = [int(r["nodos_expandidos"]) for r in astar_rows]
    n_greedy = [int(r["nodos_expandidos"]) for r in greedy_rows]
    perfiles = [r["perfil_objetivo"] for r in astar_rows]

    x = np.arange(len(ids))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    bars_a = ax.bar(x - w/2, n_astar,  w, color=C_ASTAR,  alpha=0.9, label="A* (cursos)", zorder=3)
    bars_g = ax.bar(x + w/2, n_greedy, w, color=C_GREEDY, alpha=0.9, label="Greedy",      zorder=3)

    # Colorear por perfil el fondo de cada grupo
    for i, pid in enumerate(perfiles):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.05,
                   color=PERFIL_COLOR[pid], zorder=0)

    # Etiquetas sobre las barras
    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.05,
                f"{h:,.0f}", ha="center", va="bottom",
                fontsize=7.5, color="white", fontweight="bold")
    for bar in bars_g:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.05,
                f"{h:,.0f}", ha="center", va="bottom",
                fontsize=7.5, color=C_GREEDY, fontweight="bold")

    # Ratios A*/Greedy encima
    for i, (na, ng) in enumerate(zip(n_astar, n_greedy)):
        ratio = na / max(ng, 1)
        ax.text(i, max(na, ng) * 1.35,
                f"{ratio:.1f}×", ha="center", va="bottom",
                fontsize=8, color="#94A3B8", style="italic")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=30, ha="right", color="#CBD5E1", fontsize=9)
    ax.set_ylabel("Nodos expandidos (escala log)", color="#CBD5E1", fontsize=10)
    ax.set_title("Nodos Expandidos: A* vs Greedy\n(el ratio muestra cuántas veces más nodos expande A*)",
                 color="white", fontsize=12, fontweight="bold", pad=15)
    ax.tick_params(colors="#94A3B8")
    ax.yaxis.label.set_color("#CBD5E1")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(axis="y", color="#334155", alpha=0.5, zorder=0)

    legend_extra = [
        mpatches.Patch(color=C_DS,  label="Data Scientist"),
        mpatches.Patch(color=C_BE,  label="Backend Dev"),
        mpatches.Patch(color=C_ML,  label="ML Engineer"),
    ]
    l1 = ax.legend(loc="upper left", framealpha=0.2,
                   labelcolor="white", facecolor="#1E293B", fontsize=9)
    l2 = ax.legend(handles=legend_extra, loc="upper right", framealpha=0.2,
                   labelcolor="white", facecolor="#1E293B", fontsize=9,
                   title="Perfil objetivo", title_fontsize=8)
    ax.add_artist(l1)

    plt.tight_layout()
    out = FIGURES_DIR / "01_nodos_expandidos.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {out}")

# *** Gráfico 2: Puntuaciones LLM ***

def grafico_llm(con_llm):
    ids        = [r["instancia_id"] for r in con_llm]
    puntuaciones = [float(r["puntuacion_llm"]) for r in con_llm]
    perfiles   = [r["perfil_objetivo"] for r in con_llm]
    colores    = [PERFIL_COLOR[p] for p in perfiles]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    bars = ax.bar(ids, puntuaciones, color=colores, alpha=0.85, zorder=3, width=0.55)

    # Línea de promedio
    avg = sum(puntuaciones) / len(puntuaciones)
    ax.axhline(avg, color="#F8FAFC", linestyle="--", linewidth=1.2,
               alpha=0.6, label=f"Promedio: {avg:.2f}/10", zorder=4)

    # Zona de calidad
    ax.axhspan(9, 10, alpha=0.06, color=C_LLM,  label="Excelente (9-10)")
    ax.axhspan(7,  9, alpha=0.06, color=C_ASTAR, label="Bueno (7-8)")

    # Etiquetas
    for bar, p in zip(bars, puntuaciones):
        ax.text(bar.get_x() + bar.get_width()/2, p + 0.1,
                f"{p:.0f}/10", ha="center", va="bottom",
                fontsize=10, color="white", fontweight="bold")

    ax.set_ylim(0, 11)
    ax.set_yticks(range(0, 11))
    ax.set_ylabel("Puntuación LLM (0-10)", color="#CBD5E1", fontsize=10)
    ax.set_title("Evaluación LLM de Trayectorias por Instancia",
                 color="white", fontsize=12, fontweight="bold", pad=15)
    ax.tick_params(colors="#94A3B8")
    ax.set_xticklabels(ids, rotation=30, ha="right", color="#CBD5E1", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(axis="y", color="#334155", alpha=0.4, zorder=0)

    legend_perfiles = [
        mpatches.Patch(color=C_DS, label="Data Scientist"),
        mpatches.Patch(color=C_BE, label="Backend Dev"),
        mpatches.Patch(color=C_ML, label="ML Engineer"),
    ]
    l1 = ax.legend(loc="lower right", framealpha=0.2,
                   labelcolor="white", facecolor="#1E293B", fontsize=9)
    l2 = ax.legend(handles=legend_perfiles, loc="upper left", framealpha=0.2,
                   labelcolor="white", facecolor="#1E293B", fontsize=9)
    ax.add_artist(l1)

    plt.tight_layout()
    out = FIGURES_DIR / "02_puntuaciones_llm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {out}")

# *** Gráfico 3: Cursos y semanas por instancia ***

def grafico_calidad(sin_llm):
    astar_rows = [r for r in sin_llm if "A*" in r["algoritmo"]]
    ids     = [r["instancia_id"] for r in astar_rows]
    cursos  = [int(r["num_cursos"]) for r in astar_rows]
    semanas = [int(r["costo_total_semanas"]) for r in astar_rows]
    perfiles = [r["perfil_objetivo"] for r in astar_rows]
    habs_ini = [int(r["habs_iniciales"]) for r in astar_rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("#0F172A")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1E293B")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.tick_params(colors="#94A3B8")
        ax.grid(axis="y", color="#334155", alpha=0.4, zorder=0)

    colores = [PERFIL_COLOR[p] for p in perfiles]

    # Subplot 1: cursos
    b1 = ax1.bar(ids, cursos, color=colores, alpha=0.85, zorder=3, width=0.55)
    for bar, c, hi in zip(b1, cursos, habs_ini):
        ax1.text(bar.get_x() + bar.get_width()/2, c + 0.15,
                 f"{c}", ha="center", va="bottom",
                 fontsize=10, color="white", fontweight="bold")
        ax1.text(bar.get_x() + bar.get_width()/2, 0.3,
                 f"ini:{hi}", ha="center", va="bottom",
                 fontsize=7, color="#94A3B8")
    ax1.set_ylabel("Número de cursos", color="#CBD5E1", fontsize=10)
    ax1.set_title("Calidad de Solución: Cursos y Semanas por Instancia (A*)",
                  color="white", fontsize=12, fontweight="bold", pad=12)

    # Subplot 2: semanas
    b2 = ax2.bar(ids, semanas, color=colores, alpha=0.85, zorder=3, width=0.55)
    for bar, s in zip(b2, semanas):
        ax2.text(bar.get_x() + bar.get_width()/2, s + 0.5,
                 f"{s}s", ha="center", va="bottom",
                 fontsize=10, color="white", fontweight="bold")
    ax2.set_ylabel("Semanas totales", color="#CBD5E1", fontsize=10)
    ax2.set_xticklabels(ids, rotation=30, ha="right", color="#CBD5E1", fontsize=9)

    legend = [
        mpatches.Patch(color=C_DS, label="Data Scientist"),
        mpatches.Patch(color=C_BE, label="Backend Dev"),
        mpatches.Patch(color=C_ML, label="ML Engineer"),
    ]
    ax1.legend(handles=legend, loc="upper right", framealpha=0.2,
               labelcolor="white", facecolor="#1E293B", fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "03_calidad_solucion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {out}")

# *** Gráfico 4: Tiempo de búsqueda ***

def grafico_tiempo(sin_llm):
    astar_rows  = [r for r in sin_llm if "A*" in r["algoritmo"]]
    greedy_rows = [r for r in sin_llm if "Greedy" in r["algoritmo"]]

    ids     = [r["instancia_id"] for r in astar_rows]
    t_astar = [float(r["tiempo_segundos"]) * 1000 for r in astar_rows]   # ms
    t_greedy= [float(r["tiempo_segundos"]) * 1000 for r in greedy_rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    x = np.arange(len(ids))
    w = 0.35
    ax.bar(x - w/2, t_astar,  w, color=C_ASTAR,  alpha=0.85, label="A* (cursos)", zorder=3)
    ax.bar(x + w/2, t_greedy, w, color=C_GREEDY, alpha=0.85, label="Greedy",      zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=30, ha="right", color="#CBD5E1", fontsize=9)
    ax.set_ylabel("Tiempo de búsqueda (ms, escala log)", color="#CBD5E1", fontsize=10)
    ax.set_title("Tiempo de Búsqueda: A* vs Greedy (milisegundos)",
                 color="white", fontsize=12, fontweight="bold", pad=15)
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(axis="y", color="#334155", alpha=0.4, zorder=0)
    ax.legend(framealpha=0.2, labelcolor="white", facecolor="#1E293B", fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "04_tiempo_busqueda.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {out}")

# *** Main ***

def main():
    print("=" * 56)
    print("  GENERANDO VISUALIZACIONES — Career Path Planner")
    print("=" * 56)

    sin_llm = leer_csv("resultados_sin_llm.csv")
    con_llm = leer_csv("resultados_con_llm.csv")

    print(f"\n  Datos cargados:")
    print(f"    Sin LLM : {len(sin_llm)} filas")
    print(f"    Con LLM : {len(con_llm)} filas")
    print(f"\n  Generando gráficos...\n")

    grafico_nodos(sin_llm)
    grafico_llm(con_llm)
    grafico_calidad(sin_llm)
    grafico_tiempo(sin_llm)

    print(f"\n  ✓ 4 visualizaciones guardadas en: {FIGURES_DIR}")

if __name__ == "__main__":
    main()