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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Paleta base
C_ASTAR = "#2563EB"   # azul
C_GREEDY = "#F59E0B"  # ámbar
C_LLM = "#10B981"     # verde
C_DS = "#6366F1"      # índigo
C_BE = "#EC4899"      # rosa
C_ML = "#F97316"      # naranja

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


def _safe_int(valor, default=0):
    try:
        if valor is None or valor == "":
            return default
        return int(float(valor))
    except Exception:
        return default


def _safe_float(valor, default=np.nan):
    try:
        if valor is None or valor == "":
            return default
        return float(valor)
    except Exception:
        return default


def _score_style(score):
    if score >= 9:
        return "Excelente", "#22C55E"
    if score >= 7:
        return "Bueno", "#3B82F6"
    if score >= 5:
        return "Aceptable", "#F59E0B"
    return "Deficiente", "#EF4444"


# Gráfico 1: Nodos expandidos A* vs Greedy
def grafico_nodos(sin_llm):
    astar_rows = [r for r in sin_llm if "A*" in r["algoritmo"]]
    greedy_rows = [r for r in sin_llm if "Greedy" in r["algoritmo"]]

    ids = [r["instancia_id"] for r in astar_rows]
    n_astar = [_safe_int(r["nodos_expandidos"]) for r in astar_rows]
    n_greedy = [_safe_int(r["nodos_expandidos"]) for r in greedy_rows]
    perfiles = [r["perfil_objetivo"] for r in astar_rows]

    x = np.arange(len(ids))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    bars_a = ax.bar(
        x - w / 2, n_astar, w, color=C_ASTAR, alpha=0.9,
        label="A* (cursos)", zorder=3
    )
    bars_g = ax.bar(
        x + w / 2, n_greedy, w, color=C_GREEDY, alpha=0.9,
        label="Greedy", zorder=3
    )

    for i, pid in enumerate(perfiles):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.05, color=PERFIL_COLOR[pid], zorder=0)

    for bar in bars_a:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h * 1.05, f"{h:,.0f}",
            ha="center", va="bottom", fontsize=7.5, color="white", fontweight="bold"
        )
    for bar in bars_g:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h * 1.05, f"{h:,.0f}",
            ha="center", va="bottom", fontsize=7.5, color=C_GREEDY, fontweight="bold"
        )

    for i, (na, ng) in enumerate(zip(n_astar, n_greedy)):
        ratio = na / max(ng, 1)
        ax.text(
            i, max(na, ng) * 1.35, f"{ratio:.1f}×",
            ha="center", va="bottom", fontsize=8, color="#94A3B8", style="italic"
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=30, ha="right", color="#CBD5E1", fontsize=9)
    ax.set_ylabel("Nodos expandidos (escala log)", color="#CBD5E1", fontsize=10)
    ax.set_title(
        "Nodos Expandidos: A* vs Greedy\n(el ratio muestra cuántas veces más nodos expande A*)",
        color="white", fontsize=12, fontweight="bold", pad=15
    )
    ax.tick_params(colors="#94A3B8")
    ax.yaxis.label.set_color("#CBD5E1")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(axis="y", color="#334155", alpha=0.5, zorder=0)

    legend_extra = [
        mpatches.Patch(color=C_DS, label="Data Scientist"),
        mpatches.Patch(color=C_BE, label="Backend Dev"),
        mpatches.Patch(color=C_ML, label="ML Engineer"),
    ]
    l1 = ax.legend(
        loc="upper left", framealpha=0.2,
        labelcolor="white", facecolor="#1E293B", fontsize=9
    )
    l2 = ax.legend(
        handles=legend_extra, loc="upper right", framealpha=0.2,
        labelcolor="white", facecolor="#1E293B", fontsize=9,
        title="Perfil objetivo", title_fontsize=8
    )
    ax.add_artist(l1)

    plt.tight_layout()
    out = FIGURES_DIR / "01_nodos_expandidos.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {out}")


# Gráfico 2: Puntuaciones LLM
def grafico_llm(con_llm):
    filas = []
    for r in con_llm:
        puntuacion = _safe_float(r.get("puntuacion_llm"), np.nan)
        if np.isnan(puntuacion):
            continue
        cursos = _safe_int(r.get("num_cursos"))
        semanas = _safe_int(r.get("costo_total_semanas"))
        filas.append({
            "instancia_id": r.get("instancia_id", ""),
            "perfil_objetivo": r.get("perfil_objetivo", ""),
            "puntuacion": puntuacion,
            "cursos": cursos,
            "semanas": semanas,
            "larga": (cursos > 12) or (semanas > 50),
            "nivel": r.get("nivel_calidad_llm", ""),
        })

    if not filas:
        print("  ⚠ No hay puntuaciones LLM válidas para graficar.")
        return

    ids = [r["instancia_id"] for r in filas]
    puntuaciones = [r["puntuacion"] for r in filas]
    perfiles = [r["perfil_objetivo"] for r in filas]
    cursos = [r["cursos"] for r in filas]
    semanas = [r["semanas"] for r in filas]
    larga = [r["larga"] for r in filas]

    # Color por banda de calidad, borde por perfil
    face_colors = []
    edge_colors = []
    markers = []
    for i, p in enumerate(puntuaciones):
        _, color = _score_style(p)
        face_colors.append(color)
        edge_colors.append(PERFIL_COLOR.get(perfiles[i], "#E2E8F0"))
        markers.append("X" if larga[i] else "o")

    x = np.arange(len(ids))

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    # Zonas de interpretación
    ax.axhspan(9, 10, alpha=0.07, color="#22C55E")
    ax.axhspan(7, 9, alpha=0.07, color="#3B82F6")
    ax.axhspan(5, 7, alpha=0.06, color="#F59E0B")
    ax.axhspan(0, 5, alpha=0.05, color="#EF4444")

    # Barras más sobrias + dispersión encima
    bars = ax.bar(
        x, puntuaciones, width=0.62,
        color=face_colors, alpha=0.72, zorder=2
    )

    # Línea sutil para mostrar la variación entre instancias
    ax.plot(
        x, puntuaciones, color="#CBD5E1",
        linewidth=1.2, alpha=0.55, zorder=3
    )

    # Puntos con marcador especial para trayectorias largas
    for i, (score, ec, marker, is_long, sem, cur) in enumerate(
        zip(puntuaciones, edge_colors, markers, larga, semanas, cursos)
    ):
        size = 105 if is_long else 82
        ax.scatter(
            i, score, s=size, marker=marker,
            color=face_colors[i], edgecolor=ec,
            linewidth=1.8, zorder=4
        )
        ax.text(
            i, score + 0.18, f"{score:.1f}/10",
            ha="center", va="bottom",
            fontsize=9.5, color="white", fontweight="bold", zorder=5
        )
        ax.text(
            i, 0.22, f"{cur}c · {sem}s",
            ha="center", va="bottom",
            fontsize=7.5, color="#94A3B8", zorder=5
        )

    avg = float(np.mean(puntuaciones))
    med = float(np.median(puntuaciones))
    ax.axhline(
        avg, color="#F8FAFC", linestyle="--", linewidth=1.6,
        alpha=0.85, label=f"Promedio general: {avg:.2f}/10", zorder=1
    )
    ax.axhline(
        med, color="#A78BFA", linestyle=":", linewidth=1.6,
        alpha=0.95, label=f"Mediana: {med:.2f}/10", zorder=1
    )

    # Comparativa concisas vs largas
    long_scores = [p for p, is_long in zip(puntuaciones, larga) if is_long]
    short_scores = [p for p, is_long in zip(puntuaciones, larga) if not is_long]

    long_avg = float(np.mean(long_scores)) if long_scores else None
    short_avg = float(np.mean(short_scores)) if short_scores else None

    summary_lines = [
        f"Promedio general: {avg:.2f}/10",
        f"Mediana: {med:.2f}/10",
    ]
    if short_avg is not None:
        summary_lines.append(f"Trayectorias concisas (n={len(short_scores)}): {short_avg:.2f}/10")
    if long_avg is not None:
        summary_lines.append(f"Trayectorias largas (n={len(long_scores)}): {long_avg:.2f}/10")
    summary_lines.append("X = trayectoria larga (>12 cursos o >50 semanas)")

    ax.text(
        0.985, 0.965, "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9.2, color="#E2E8F0",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#0F172A", edgecolor="#334155", alpha=0.88),
        zorder=10
    )

    ax.set_ylim(0, 10.45)
    ax.set_yticks(np.arange(0, 11, 1))
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=28, ha="right", color="#CBD5E1", fontsize=9)
    ax.set_ylabel("Puntuación LLM (0-10)", color="#CBD5E1", fontsize=10)
    ax.set_title(
        "Evaluación LLM de Trayectorias por Instancia\n"
        "La varianza visual resalta la penalización sobre rutas extensas",
        color="white", fontsize=12, fontweight="bold", pad=15
    )
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(axis="y", color="#334155", alpha=0.35, zorder=0)

    # Leyenda compacta
    legend_quality = [
        mpatches.Patch(color="#22C55E", label="Excelente (9-10)"),
        mpatches.Patch(color="#3B82F6", label="Bueno (7-8)"),
        mpatches.Patch(color="#F59E0B", label="Aceptable (5-6)"),
        mpatches.Patch(color="#EF4444", label="Deficiente (0-4)"),
    ]
    legend_profiles = [
        mpatches.Patch(color=C_DS, label="Data Scientist"),
        mpatches.Patch(color=C_BE, label="Backend Dev"),
        mpatches.Patch(color=C_ML, label="ML Engineer"),
    ]

    l1 = ax.legend(
        handles=legend_quality, loc="lower left",
        framealpha=0.22, labelcolor="white",
        facecolor="#1E293B", fontsize=8.7
    )
    l2 = ax.legend(
        handles=legend_profiles, loc="upper left",
        framealpha=0.22, labelcolor="white",
        facecolor="#1E293B", fontsize=8.7,
        title="Perfil objetivo", title_fontsize=8
    )
    ax.add_artist(l1)

    plt.tight_layout()
    out = FIGURES_DIR / "02_puntuaciones_llm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {out}")


# Gráfico 3: Cursos y semanas por instancia
def grafico_calidad(sin_llm):
    astar_rows = [r for r in sin_llm if "A*" in r["algoritmo"]]
    ids = [r["instancia_id"] for r in astar_rows]
    cursos = [_safe_int(r["num_cursos"]) for r in astar_rows]
    semanas = [_safe_int(r["costo_total_semanas"]) for r in astar_rows]
    perfiles = [r["perfil_objetivo"] for r in astar_rows]
    habs_ini = [_safe_int(r["habs_iniciales"]) for r in astar_rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("#0F172A")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1E293B")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.tick_params(colors="#94A3B8")
        ax.grid(axis="y", color="#334155", alpha=0.4, zorder=0)

    colores = [PERFIL_COLOR[p] for p in perfiles]

    b1 = ax1.bar(ids, cursos, color=colores, alpha=0.85, zorder=3, width=0.55)
    for bar, c, hi in zip(b1, cursos, habs_ini):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, c + 0.15, f"{c}",
            ha="center", va="bottom", fontsize=10, color="white", fontweight="bold"
        )
        ax1.text(
            bar.get_x() + bar.get_width() / 2, 0.3, f"ini:{hi}",
            ha="center", va="bottom", fontsize=7, color="#94A3B8"
        )
    ax1.set_ylabel("Número de cursos", color="#CBD5E1", fontsize=10)
    ax1.set_title(
        "Calidad de Solución: Cursos y Semanas por Instancia (A*)",
        color="white", fontsize=12, fontweight="bold", pad=12
    )

    b2 = ax2.bar(ids, semanas, color=colores, alpha=0.85, zorder=3, width=0.55)
    for bar, s in zip(b2, semanas):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, s + 0.5, f"{s}s",
            ha="center", va="bottom", fontsize=10, color="white", fontweight="bold"
        )
    ax2.set_ylabel("Semanas totales", color="#CBD5E1", fontsize=10)
    ax2.set_xticks(np.arange(len(ids)))
    ax2.set_xticklabels(ids, rotation=30, ha="right", color="#CBD5E1", fontsize=9)

    legend = [
        mpatches.Patch(color=C_DS, label="Data Scientist"),
        mpatches.Patch(color=C_BE, label="Backend Dev"),
        mpatches.Patch(color=C_ML, label="ML Engineer"),
    ]
    ax1.legend(
        handles=legend, loc="upper right", framealpha=0.2,
        labelcolor="white", facecolor="#1E293B", fontsize=9
    )

    plt.tight_layout()
    out = FIGURES_DIR / "03_calidad_solucion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {out}")


# Gráfico 4: Tiempo de búsqueda
def grafico_tiempo(sin_llm):
    astar_rows = [r for r in sin_llm if "A*" in r["algoritmo"]]
    greedy_rows = [r for r in sin_llm if "Greedy" in r["algoritmo"]]

    ids = [r["instancia_id"] for r in astar_rows]
    t_astar = [float(r["tiempo_segundos"]) * 1000 for r in astar_rows]
    t_greedy = [float(r["tiempo_segundos"]) * 1000 for r in greedy_rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    x = np.arange(len(ids))
    w = 0.35
    ax.bar(x - w / 2, t_astar, w, color=C_ASTAR, alpha=0.85, label="A* (cursos)", zorder=3)
    ax.bar(x + w / 2, t_greedy, w, color=C_GREEDY, alpha=0.85, label="Greedy", zorder=3)

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


def main():
    print("=" * 56)
    print("  GENERANDO VISUALIZACIONES — Career Path Planner")
    print("=" * 56)

    sin_llm = leer_csv("resultados_sin_llm.csv")
    con_llm = leer_csv("resultados_con_llm.csv")

    print("\n  Datos cargados:")
    print(f"    Sin LLM : {len(sin_llm)} filas")
    print(f"    Con LLM : {len(con_llm)} filas")
    print("\n  Generando gráficos...\n")

    grafico_nodos(sin_llm)
    grafico_llm(con_llm)
    grafico_calidad(sin_llm)
    grafico_tiempo(sin_llm)

    print(f"\n  ✓ 4 visualizaciones guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()