"""
generate_visualizations.py

Genera las 4 visualizaciones del análisis experimental.

Gráficos producidos:
  01_nodos_expandidos.png  — Nodos expandidos A* vs Greedy (escala log)
  02_puntuaciones_llm.png  — Puntuaciones LLM por instancia y perfil
  03_calidad_solucion.png  — Cursos y semanas por instancia (A*)
  04_tiempo_busqueda.png   — Tiempo de búsqueda A* vs Greedy (ms, escala log)

Correcciones v2:
  - Emparejamiento por instancia_id en lugar de zip() para evitar
    desincronización si alguna instancia falla.
  - Filtro por exito=="True" antes de graficar.
  - _color_perfil() con color por defecto para perfiles no catalogados.
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

# ── Paleta ────────────────────────────────────────────────────────────────────

C_ASTAR  = "#2563EB"
C_GREEDY = "#F59E0B"

PERFIL_COLOR = {
    "data_scientist":    "#6366F1",
    "backend_developer": "#EC4899",
    "ml_engineer":       "#F97316",
}

PERFIL_LABEL = {
    "data_scientist":    "Data Scientist",
    "backend_developer": "Backend Dev",
    "ml_engineer":       "ML Engineer",
}

_COLOR_DEFAULT = "#64748B"


def _color_perfil(perfil: str) -> str:
    return PERFIL_COLOR.get(perfil, _COLOR_DEFAULT)


# ── Lectura de CSV ─────────────────────────────────────────────────────────────

def leer_csv(nombre: str) -> list[dict]:
    path = RESULTS_DIR / nombre
    if not path.exists():
        print(f"  ⚠ No existe {path}. Se omite.")
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_int(valor, default: int = 0) -> int:
    try:
        return int(float(valor)) if valor not in (None, "") else default
    except Exception:
        return default


def _safe_float(valor, default: float = np.nan) -> float:
    try:
        return float(valor) if valor not in (None, "") else default
    except Exception:
        return default


# ── Gráfico 1: Nodos expandidos ───────────────────────────────────────────────

def grafico_nodos(sin_llm: list[dict]) -> None:
    astar_dict  = {r["instancia_id"]: r for r in sin_llm
                   if "A*" in r.get("algoritmo", "") and r.get("exito") == "True"}
    greedy_dict = {r["instancia_id"]: r for r in sin_llm
                   if "Greedy" in r.get("algoritmo", "") and r.get("exito") == "True"}
    ids = sorted(set(astar_dict) & set(greedy_dict))

    if not ids:
        print("  ⚠ Sin datos para gráfico de nodos.")
        return

    n_astar  = [_safe_int(astar_dict[i]["nodos_expandidos"])  for i in ids]
    n_greedy = [_safe_int(greedy_dict[i]["nodos_expandidos"]) for i in ids]
    perfiles = [astar_dict[i]["perfil_objetivo"]               for i in ids]

    x = np.arange(len(ids))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    ax.bar(x - w/2, n_astar,  w, color=C_ASTAR,  alpha=0.9, label="A*",     zorder=3)
    ax.bar(x + w/2, n_greedy, w, color=C_GREEDY, alpha=0.9, label="Greedy", zorder=3)

    for i, pid in enumerate(perfiles):
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.05, color=_color_perfil(pid), zorder=0)

    for i, (na, ng) in enumerate(zip(n_astar, n_greedy)):
        ratio = na / max(ng, 1)
        ax.text(i, max(na, ng) * 1.35, f"{ratio:.1f}×",
                ha="center", va="bottom", fontsize=8,
                color="#94A3B8", style="italic")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=30, ha="right", color="#CBD5E1", fontsize=9)
    ax.set_ylabel("Nodos expandidos (escala log)", color="#CBD5E1", fontsize=10)
    ax.set_title(
        "Nodos Expandidos: A* vs Greedy\n"
        "(ratio = cuántas veces más nodos expande A* respecto a Greedy)",
        color="white", fontsize=12, fontweight="bold", pad=15,
    )
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(axis="y", color="#334155", alpha=0.5, zorder=0)

    leg_alg = ax.legend(
        loc="upper left", framealpha=0.2,
        labelcolor="white", facecolor="#1E293B", fontsize=9,
    )
    ax.legend(
        handles=[mpatches.Patch(color=v, label=PERFIL_LABEL.get(k, k))
                 for k, v in PERFIL_COLOR.items()],
        loc="upper right", framealpha=0.2,
        labelcolor="white", facecolor="#1E293B", fontsize=9,
        title="Perfil objetivo", title_fontsize=8,
    )
    ax.add_artist(leg_alg)

    plt.tight_layout()
    out = FIGURES_DIR / "01_nodos_expandidos.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {out}")


# ── Gráfico 2: Puntuaciones LLM ───────────────────────────────────────────────

def _score_style(score: float) -> tuple[str, str]:
    if score >= 9:  return "Excelente", "#22C55E"
    if score >= 7:  return "Bueno",     "#3B82F6"
    if score >= 5:  return "Aceptable", "#F59E0B"
    return "Deficiente", "#EF4444"


def grafico_llm(con_llm: list[dict]) -> None:
    filas = []
    for r in con_llm:
        p = _safe_float(r.get("puntuacion_llm"))
        if np.isnan(p):
            continue
        cursos  = _safe_int(r.get("num_cursos"))
        semanas = _safe_int(r.get("costo_total_semanas"))
        filas.append({
            "id":     r.get("instancia_id", ""),
            "perfil": r.get("perfil_objetivo", ""),
            "p":      p,
            "cursos": cursos,
            "sem":    semanas,
            "larga":  cursos > 12 or semanas > 50,
        })

    if not filas:
        print("  ⚠ Sin puntuaciones LLM válidas para graficar.")
        return

    ids  = [f["id"] for f in filas]
    puns = [f["p"]  for f in filas]
    x    = np.arange(len(ids))

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    # Zonas de calidad
    ax.axhspan(9, 10, alpha=0.07, color="#22C55E")
    ax.axhspan(7, 9,  alpha=0.07, color="#3B82F6")
    ax.axhspan(5, 7,  alpha=0.06, color="#F59E0B")
    ax.axhspan(0, 5,  alpha=0.05, color="#EF4444")

    face_colors = [_score_style(p)[1] for p in puns]
    ax.bar(x, puns, width=0.62, color=face_colors, alpha=0.72, zorder=2)
    ax.plot(x, puns, color="#CBD5E1", linewidth=1.2, alpha=0.55, zorder=3)

    for i, f in enumerate(filas):
        marker = "X" if f["larga"] else "o"
        ax.scatter(i, f["p"], s=105 if f["larga"] else 82, marker=marker,
                   color=face_colors[i], edgecolor=_color_perfil(f["perfil"]),
                   linewidth=1.8, zorder=4)
        ax.text(i, f["p"] + 0.18, f"{f['p']:.1f}/10",
                ha="center", va="bottom", fontsize=9.5,
                color="white", fontweight="bold", zorder=5)
        ax.text(i, 0.22, f"{f['cursos']}c·{f['sem']}s",
                ha="center", va="bottom", fontsize=7.5,
                color="#94A3B8", zorder=5)

    avg = float(np.mean(puns))
    med = float(np.median(puns))
    ax.axhline(avg, color="#F8FAFC", linestyle="--", linewidth=1.6,
               alpha=0.85, label=f"Promedio: {avg:.2f}/10", zorder=1)
    ax.axhline(med, color="#A78BFA", linestyle=":",  linewidth=1.6,
               alpha=0.95, label=f"Mediana:  {med:.2f}/10", zorder=1)

    long_s  = [f["p"] for f in filas if f["larga"]]
    short_s = [f["p"] for f in filas if not f["larga"]]
    summ = [f"Promedio: {avg:.2f}/10", f"Mediana: {med:.2f}/10"]
    if short_s:
        summ.append(f"Concisas (n={len(short_s)}): {np.mean(short_s):.2f}/10")
    if long_s:
        summ.append(f"Largas   (n={len(long_s)}): {np.mean(long_s):.2f}/10")
    summ.append("X = trayectoria larga (>12 cursos o >50 sem)")

    ax.text(0.985, 0.965, "\n".join(summ), transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#E2E8F0",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#0F172A",
                      edgecolor="#334155", alpha=0.88), zorder=10)

    ax.set_ylim(0, 10.45)
    ax.set_yticks(np.arange(0, 11, 1))
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=28, ha="right", color="#CBD5E1", fontsize=9)
    ax.set_ylabel("Puntuación LLM (0-10)", color="#CBD5E1", fontsize=10)
    ax.set_title(
        "Evaluación LLM de Trayectorias por Instancia\n"
        "Penalización visible sobre rutas extensas",
        color="white", fontsize=12, fontweight="bold", pad=15,
    )
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(axis="y", color="#334155", alpha=0.35, zorder=0)

    l1 = ax.legend(
        handles=[mpatches.Patch(color="#22C55E", label="Excelente (9-10)"),
                 mpatches.Patch(color="#3B82F6", label="Bueno (7-8)"),
                 mpatches.Patch(color="#F59E0B", label="Aceptable (5-6)"),
                 mpatches.Patch(color="#EF4444", label="Deficiente (0-4)")],
        loc="lower left", framealpha=0.22,
        labelcolor="white", facecolor="#1E293B", fontsize=8.7,
    )
    l2 = ax.legend(
        handles=[mpatches.Patch(color=v, label=PERFIL_LABEL.get(k, k))
                 for k, v in PERFIL_COLOR.items()],
        loc="upper left", framealpha=0.22,
        labelcolor="white", facecolor="#1E293B", fontsize=8.7,
        title="Perfil objetivo", title_fontsize=8,
    )
    ax.add_artist(l1)

    plt.tight_layout()
    out = FIGURES_DIR / "02_puntuaciones_llm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {out}")


# ── Gráfico 3: Calidad de solución ────────────────────────────────────────────

def grafico_calidad(sin_llm: list[dict]) -> None:
    astar_rows = [r for r in sin_llm
                  if "A*" in r.get("algoritmo", "") and r.get("exito") == "True"]
    if not astar_rows:
        print("  ⚠ Sin datos A* para gráfico de calidad.")
        return

    ids      = [r["instancia_id"]                          for r in astar_rows]
    cursos   = [_safe_int(r["num_cursos"])                  for r in astar_rows]
    semanas  = [_safe_int(r["costo_total_semanas"])         for r in astar_rows]
    perfiles = [r["perfil_objetivo"]                        for r in astar_rows]
    habs_ini = [_safe_int(r.get("habs_iniciales", 0))       for r in astar_rows]
    colores  = [_color_perfil(p)                            for p in perfiles]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("#0F172A")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1E293B")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.tick_params(colors="#94A3B8")
        ax.grid(axis="y", color="#334155", alpha=0.4, zorder=0)

    b1 = ax1.bar(ids, cursos, color=colores, alpha=0.85, zorder=3, width=0.55)
    for bar, c, hi in zip(b1, cursos, habs_ini):
        ax1.text(bar.get_x() + bar.get_width() / 2, c + 0.15, str(c),
                 ha="center", va="bottom", fontsize=10,
                 color="white", fontweight="bold")
        ax1.text(bar.get_x() + bar.get_width() / 2, 0.3, f"ini:{hi}",
                 ha="center", va="bottom", fontsize=7, color="#94A3B8")
    ax1.set_ylabel("Número de cursos", color="#CBD5E1", fontsize=10)
    ax1.set_title(
        "Calidad de Solución: Cursos y Semanas por Instancia (A*)",
        color="white", fontsize=12, fontweight="bold", pad=12,
    )

    b2 = ax2.bar(ids, semanas, color=colores, alpha=0.85, zorder=3, width=0.55)
    for bar, s in zip(b2, semanas):
        ax2.text(bar.get_x() + bar.get_width() / 2, s + 0.5, f"{s}s",
                 ha="center", va="bottom", fontsize=10,
                 color="white", fontweight="bold")
    ax2.set_ylabel("Semanas totales", color="#CBD5E1", fontsize=10)
    ax2.set_xticks(np.arange(len(ids)))
    ax2.set_xticklabels(ids, rotation=30, ha="right", color="#CBD5E1", fontsize=9)

    ax1.legend(
        handles=[mpatches.Patch(color=v, label=PERFIL_LABEL.get(k, k))
                 for k, v in PERFIL_COLOR.items()],
        loc="upper right", framealpha=0.2,
        labelcolor="white", facecolor="#1E293B", fontsize=9,
    )

    plt.tight_layout()
    out = FIGURES_DIR / "03_calidad_solucion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {out}")


# ── Gráfico 4: Tiempo de búsqueda ─────────────────────────────────────────────

def grafico_tiempo(sin_llm: list[dict]) -> None:
    astar_dict  = {r["instancia_id"]: r for r in sin_llm
                   if "A*" in r.get("algoritmo", "") and r.get("exito") == "True"}
    greedy_dict = {r["instancia_id"]: r for r in sin_llm
                   if "Greedy" in r.get("algoritmo", "") and r.get("exito") == "True"}
    ids = sorted(set(astar_dict) & set(greedy_dict))

    if not ids:
        print("  ⚠ Sin datos para gráfico de tiempos.")
        return

    t_a = [_safe_float(astar_dict[i]["tiempo_segundos"],  0.0) * 1000 for i in ids]
    t_g = [_safe_float(greedy_dict[i]["tiempo_segundos"], 0.0) * 1000 for i in ids]

    x = np.arange(len(ids))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    ax.bar(x - w/2, t_a, w, color=C_ASTAR,  alpha=0.85, label="A*",     zorder=3)
    ax.bar(x + w/2, t_g, w, color=C_GREEDY, alpha=0.85, label="Greedy", zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=30, ha="right", color="#CBD5E1", fontsize=9)
    ax.set_ylabel("Tiempo de búsqueda (ms, escala log)", color="#CBD5E1", fontsize=10)
    ax.set_title(
        "Tiempo de Búsqueda: A* vs Greedy (milisegundos)",
        color="white", fontsize=12, fontweight="bold", pad=15,
    )
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
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