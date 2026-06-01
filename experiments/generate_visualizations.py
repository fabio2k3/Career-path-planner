"""
generate_visualizations.py
--------------------------
Genera las visualizaciones del análisis experimental.

Gráficos producidos:
  1. Nodos expandidos A*(semanas) vs A*(cursos) vs Greedy — eficiencia computacional
  2. Puntuaciones LLM por instancia y perfil
  3. Cursos y semanas por instancia — calidad de solución (A* semanas vs cursos vs Greedy)
  4. Tiempo de búsqueda comparativo

Salida: experiments/results/figures/
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Paleta de colores ────────────────────────────────────────────────────────
C_ASTAR_SEM = "#1D4ED8"   # azul oscuro  — A* semanas
C_ASTAR_CUR = "#0EA5E9"   # azul claro   — A* cursos
C_GREEDY    = "#F59E0B"   # ámbar        — Greedy
C_LLM       = "#10B981"   # verde        — LLM
C_DS        = "#6366F1"   # índigo       — Data Scientist
C_BE        = "#EC4899"   # rosa         — Backend Dev
C_ML        = "#F97316"   # naranja      — ML Engineer

PERFIL_COLOR = {
    "data_scientist":    C_DS,
    "backend_developer": C_BE,
    "ml_engineer":       C_ML,
}
PERFIL_LABEL = {
    "data_scientist":    "Data Scientist",
    "backend_developer": "Backend Dev",
    "ml_engineer":       "ML Engineer",
}

# Fondo claro para imprimir bien en el informe
BG_FIGURE = "#FFFFFF"
BG_AXES   = "#F8FAFC"
TEXT_COLOR = "#1E293B"
GRID_COLOR = "#CBD5E1"


# ─── Utilidades ───────────────────────────────────────────────────────────────

def leer_csv(path: Path) -> list:
    if not path.exists():
        print(f"  ⚠ Archivo no encontrado: {path}", file=sys.stderr)
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"  ⚠ Error leyendo {path}: {e}", file=sys.stderr)
        return []


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except (ValueError, TypeError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (ValueError, TypeError):
        return default


def _aplicar_estilo_base(fig, ax):
    """Aplica estilo claro consistente a una figura."""
    fig.patch.set_facecolor(BG_FIGURE)
    ax.set_facecolor(BG_AXES)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.6, zorder=0)


# ─── Gráfico 1: Nodos expandidos ──────────────────────────────────────────────

def grafico_nodos(sin_llm: list):
    if not sin_llm:
        return

    astar_sem_rows = [r for r in sin_llm if r.get("algoritmo") == "A* (semanas)"]
    astar_cur_rows = [r for r in sin_llm if r.get("algoritmo") == "A* (cursos)"]
    greedy_rows    = [r for r in sin_llm if r.get("algoritmo") == "Greedy"]

    if not astar_sem_rows:
        print("  ⚠ No hay filas de A*(semanas) en resultados_sin_llm.csv")
        return

    ids      = [r["instancia_id"] for r in astar_sem_rows]
    n_as     = [_safe_int(r["nodos_expandidos"]) for r in astar_sem_rows]
    n_ac     = [_safe_int(r["nodos_expandidos"]) for r in astar_cur_rows] if astar_cur_rows else [0]*len(ids)
    n_g      = [_safe_int(r["nodos_expandidos"]) for r in greedy_rows]    if greedy_rows    else [0]*len(ids)
    perfiles = [r.get("perfil_objetivo", "") for r in astar_sem_rows]

    x = np.arange(len(ids))
    w = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    _aplicar_estilo_base(fig, ax)

    ax.bar(x - w,     n_as, w, color=C_ASTAR_SEM, alpha=0.88, label="A* (semanas)", zorder=3)
    ax.bar(x,         n_ac, w, color=C_ASTAR_CUR, alpha=0.88, label="A* (cursos)",  zorder=3)
    ax.bar(x + w,     n_g,  w, color=C_GREEDY,    alpha=0.88, label="Greedy",       zorder=3)

    # Bandas de perfil
    for i, pid in enumerate(perfiles):
        col = PERFIL_COLOR.get(pid, "#94A3B8")
        ax.axvspan(i - 0.45, i + 0.45, alpha=0.06, color=col, zorder=0)

    # Ratio A*(semanas)/Greedy
    for i, (na, ng) in enumerate(zip(n_as, n_g)):
        if ng > 0:
            ratio = na / ng
            ax.text(i, max(na, _safe_int(n_ac[i] if n_ac else 0), ng) * 1.12,
                    f"{ratio:.0f}×", ha="center", va="bottom",
                    fontsize=8, color="#64748B", style="italic")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Nodos expandidos (escala log)", fontsize=10)
    ax.set_title(
        "Nodos Expandidos por Algoritmo\n"
        "(el ratio ×N indica cuántas veces más nodos expande A*(semanas) vs Greedy)",
        fontsize=12, fontweight="bold", pad=14
    )

    legend_alg = ax.legend(loc="upper left", framealpha=0.8, fontsize=9)
    legend_prof = ax.legend(
        handles=[mpatches.Patch(color=PERFIL_COLOR[p], label=PERFIL_LABEL[p])
                 for p in PERFIL_COLOR],
        loc="upper right", framealpha=0.8, fontsize=9,
        title="Perfil objetivo", title_fontsize=8
    )
    ax.add_artist(legend_alg)

    plt.tight_layout()
    out = FIGURES_DIR / "01_nodos_expandidos.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_FIGURE)
    plt.close()
    print(f"  ✓ {out}")


# ─── Gráfico 2: Puntuaciones LLM ──────────────────────────────────────────────

def grafico_llm(con_llm: list):
    if not con_llm:
        return

    filas = []
    for r in con_llm:
        p = _safe_float(r.get("puntuacion_llm"), -1.0)
        if p < 0:
            continue
        filas.append({
            "id":      r.get("instancia_id", ""),
            "perfil":  r.get("perfil_objetivo", ""),
            "p":       p,
            "cursos":  _safe_int(r.get("num_cursos")),
            "semanas": _safe_int(r.get("costo_total_semanas")),
            "larga":   (_safe_int(r.get("num_cursos")) > 12
                        or _safe_int(r.get("costo_total_semanas")) > 50),
            "modo":    r.get("modo_evaluacion", ""),
        })

    if not filas:
        print("  ⚠ No hay puntuaciones LLM válidas para graficar.")
        return

    ids      = [f["id"]  for f in filas]
    scores   = [f["p"]   for f in filas]
    perfiles = [f["perfil"] for f in filas]
    cursos   = [f["cursos"] for f in filas]
    semanas  = [f["semanas"] for f in filas]
    largas   = [f["larga"]  for f in filas]

    def _color_score(s):
        if s >= 9:   return "#16A34A"
        if s >= 7:   return "#2563EB"
        if s >= 5:   return "#D97706"
        return "#DC2626"

    face_colors  = [_color_score(s) for s in scores]
    edge_colors  = [PERFIL_COLOR.get(p, "#94A3B8") for p in perfiles]

    x = np.arange(len(ids))
    fig, ax = plt.subplots(figsize=(13, 6))
    _aplicar_estilo_base(fig, ax)

    # Zonas de calidad
    ax.axhspan(9, 10,  alpha=0.06, color="#16A34A")
    ax.axhspan(7,  9,  alpha=0.06, color="#2563EB")
    ax.axhspan(5,  7,  alpha=0.05, color="#D97706")
    ax.axhspan(0,  5,  alpha=0.05, color="#DC2626")

    ax.bar(x, scores, width=0.62, color=face_colors, alpha=0.75, zorder=2)
    ax.plot(x, scores, color="#475569", linewidth=1.2, alpha=0.6, zorder=3)

    for i, (s, ec, is_long, c, sem) in enumerate(
            zip(scores, edge_colors, largas, cursos, semanas)):
        marker = "X" if is_long else "o"
        ax.scatter(i, s, s=100, marker=marker,
                   color=face_colors[i], edgecolor=ec, linewidth=1.8, zorder=4)
        ax.text(i, s + 0.18, f"{s:.1f}", ha="center", va="bottom",
                fontsize=9.5, color=TEXT_COLOR, fontweight="bold", zorder=5)
        ax.text(i, 0.2, f"{c}c·{sem}s", ha="center", va="bottom",
                fontsize=7.5, color="#64748B", zorder=5)

    avg = float(np.mean(scores))
    med = float(np.median(scores))
    ax.axhline(avg, color="#1E293B", linestyle="--", linewidth=1.6,
               alpha=0.7, label=f"Media: {avg:.2f}/10")
    ax.axhline(med, color="#7C3AED", linestyle=":",  linewidth=1.6,
               alpha=0.9, label=f"Mediana: {med:.2f}/10")

    # Resumen en caja de texto
    long_scores  = [s for s, l in zip(scores, largas) if l]
    short_scores = [s for s, l in zip(scores, largas) if not l]
    summary = [f"Media: {avg:.2f}/10", f"Mediana: {med:.2f}/10"]
    if short_scores:
        summary.append(f"Trayectorias concisas (n={len(short_scores)}): {np.mean(short_scores):.2f}/10")
    if long_scores:
        summary.append(f"Trayectorias largas (n={len(long_scores)}): {np.mean(long_scores):.2f}/10")
    summary.append("X = larga (>12 cursos o >50 semanas)")
    ax.text(0.985, 0.97, "\n".join(summary),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=GRID_COLOR, alpha=0.9),
            zorder=10)

    ax.set_ylim(0, 10.6)
    ax.set_yticks(range(0, 11))
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("Puntuación LLM (0-10)", fontsize=10)
    ax.set_title(
        "Evaluación LLM de Trayectorias\n"
        "(puntuaciones más bajas en trayectorias largas reflejan penalización por eficiencia)",
        fontsize=12, fontweight="bold", pad=14
    )

    legend_quality = [
        mpatches.Patch(color="#16A34A", label="Excelente (9-10)"),
        mpatches.Patch(color="#2563EB", label="Bueno (7-8)"),
        mpatches.Patch(color="#D97706", label="Aceptable (5-6)"),
        mpatches.Patch(color="#DC2626", label="Deficiente (0-4)"),
    ]
    legend_prof = [mpatches.Patch(color=PERFIL_COLOR[p], label=PERFIL_LABEL[p])
                   for p in PERFIL_COLOR]
    l1 = ax.legend(handles=legend_quality, loc="lower left",  framealpha=0.85, fontsize=8.5)
    l2 = ax.legend(handles=legend_prof,    loc="upper left",  framealpha=0.85, fontsize=8.5,
                   title="Perfil objetivo", title_fontsize=8)
    ax.add_artist(l1)
    ax.legend(loc="lower right", framealpha=0.85, fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "02_puntuaciones_llm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_FIGURE)
    plt.close()
    print(f"  ✓ {out}")


# ─── Gráfico 3: Calidad de solución ───────────────────────────────────────────

def grafico_calidad(sin_llm: list):
    if not sin_llm:
        return

    # Comparar los tres algoritmos en cursos y semanas
    configs = [
        ("A* (semanas)", C_ASTAR_SEM, "A*(sem)"),
        ("A* (cursos)",  C_ASTAR_CUR, "A*(cur)"),
        ("Greedy",       C_GREEDY,    "Greedy"),
    ]

    # Tomar A*(semanas) como referencia para los IDs
    ref_rows  = [r for r in sin_llm if r.get("algoritmo") == "A* (semanas)"]
    if not ref_rows:
        ref_rows = [r for r in sin_llm if r.get("algoritmo") == "A* (cursos)"]
    if not ref_rows:
        print("  ⚠ No hay datos de A* en resultados_sin_llm.csv")
        return

    ids      = [r["instancia_id"]     for r in ref_rows]
    perfiles = [r.get("perfil_objetivo", "") for r in ref_rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.patch.set_facecolor(BG_FIGURE)
    for ax in (ax1, ax2):
        _aplicar_estilo_base(fig, ax)

    x = np.arange(len(ids))
    n_configs = len(configs)
    total_w   = 0.7
    w         = total_w / n_configs

    for ci, (alg_name, color, label) in enumerate(configs):
        alg_rows_map = {r["instancia_id"]: r
                        for r in sin_llm if r.get("algoritmo") == alg_name}
        cursos  = [_safe_int(alg_rows_map.get(iid, {}).get("num_cursos",  0)) for iid in ids]
        semanas = [_safe_int(alg_rows_map.get(iid, {}).get("costo_total_semanas", 0)) for iid in ids]

        offset = (ci - n_configs / 2 + 0.5) * w
        ax1.bar(x + offset, cursos,  w * 0.9, color=color, alpha=0.82, label=label, zorder=3)
        ax2.bar(x + offset, semanas, w * 0.9, color=color, alpha=0.82,             zorder=3)

    # Etiquetas de perfil en la parte inferior
    colores_fondo = [PERFIL_COLOR.get(p, "#CBD5E1") for p in perfiles]
    for i, col in enumerate(colores_fondo):
        ax1.axvspan(i - 0.45, i + 0.45, alpha=0.05, color=col, zorder=0)
        ax2.axvspan(i - 0.45, i + 0.45, alpha=0.05, color=col, zorder=0)

    ax1.set_ylabel("Número de cursos", fontsize=10)
    ax1.set_title("Calidad de Solución: Cursos y Semanas por Instancia y Algoritmo",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", framealpha=0.85, fontsize=9)

    ax2.set_ylabel("Semanas totales", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ids, rotation=30, ha="right", fontsize=9)

    legend_prof = [mpatches.Patch(color=PERFIL_COLOR[p], label=PERFIL_LABEL[p])
                   for p in PERFIL_COLOR]
    ax1.legend(
        handles=ax1.get_legend_handles_labels()[0] + legend_prof,
        labels =ax1.get_legend_handles_labels()[1] + [PERFIL_LABEL[p] for p in PERFIL_COLOR],
        loc="upper right", framealpha=0.85, fontsize=8.5,
        ncol=2
    )

    plt.tight_layout()
    out = FIGURES_DIR / "03_calidad_solucion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_FIGURE)
    plt.close()
    print(f"  ✓ {out}")


# ─── Gráfico 4: Tiempo de búsqueda ────────────────────────────────────────────

def grafico_tiempo(sin_llm: list):
    if not sin_llm:
        return

    configs = [
        ("A* (semanas)", C_ASTAR_SEM, "A*(sem)"),
        ("A* (cursos)",  C_ASTAR_CUR, "A*(cur)"),
        ("Greedy",       C_GREEDY,    "Greedy"),
    ]

    ref_rows = [r for r in sin_llm if r.get("algoritmo") == "A* (semanas)"]
    if not ref_rows:
        ref_rows = [r for r in sin_llm if r.get("algoritmo") == "A* (cursos)"]
    if not ref_rows:
        return

    ids = [r["instancia_id"] for r in ref_rows]
    x   = np.arange(len(ids))
    n   = len(configs)
    w   = 0.22

    fig, ax = plt.subplots(figsize=(13, 5))
    _aplicar_estilo_base(fig, ax)

    for ci, (alg_name, color, label) in enumerate(configs):
        rows_map = {r["instancia_id"]: r for r in sin_llm if r.get("algoritmo") == alg_name}
        tiempos_ms = [_safe_float(rows_map.get(iid, {}).get("tiempo_segundos", 0)) * 1000
                      for iid in ids]
        offset = (ci - n / 2 + 0.5) * w
        ax.bar(x + offset, tiempos_ms, w * 0.9, color=color, alpha=0.85, label=label, zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Tiempo de búsqueda (ms, escala log)", fontsize=10)
    ax.set_title("Tiempo de Búsqueda por Algoritmo (milisegundos)",
                 fontsize=12, fontweight="bold", pad=14)
    ax.legend(framealpha=0.85, fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "04_tiempo_busqueda.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG_FIGURE)
    plt.close()
    print(f"  ✓ {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 56)
    print("  GENERANDO VISUALIZACIONES — Career Path Planner")
    print("=" * 56)

    sin_llm = leer_csv(RESULTS_DIR / "resultados_sin_llm.csv")
    con_llm = leer_csv(RESULTS_DIR / "resultados_con_llm.csv")

    if not sin_llm and not con_llm:
        print("  ✗ No se encontraron archivos de resultados.")
        print("  Ejecuta primero: python experiments/run_experiments.py")
        return

    print(f"\n  Datos cargados:")
    print(f"    Sin LLM : {len(sin_llm)} filas")
    print(f"    Con LLM : {len(con_llm)} filas")
    print(f"\n  Generando gráficos...\n")

    grafico_nodos(sin_llm)
    grafico_llm(con_llm)
    grafico_calidad(sin_llm)
    grafico_tiempo(sin_llm)

    print(f"\n  ✓ Visualizaciones guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()