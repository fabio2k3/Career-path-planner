"""
monte_carlo.py
--------------
Simulación Monte Carlo del sistema Career Path Planner.

Objetivo:
  Evaluar el comportamiento estadístico de A* y Greedy sobre instancias
  generadas aleatoriamente, introduciendo incertidumbre en:
    - Habilidades iniciales del usuario (subconjunto aleatorio del catálogo)
    - Perfil objetivo (seleccionado aleatoriamente entre los disponibles)
    - Duración de los cursos (perturbación ±20% via distribución triangular)

Fundamento teórico (contenidos del curso de Simulación):
  - Generación de variables aleatorias mediante Transformada Inversa (TTI)
  - Distribución triangular para modelar incertidumbre en duración de cursos
  - Diseño experimental de simulación: múltiples corridas independientes
  - Análisis estadístico: media, desviación estándar, IC 95%, histogramas

Uso:
    python experiments/monte_carlo.py
    python experiments/monte_carlo.py --runs 500
    python experiments/monte_carlo.py --runs 500 --seed 42
    python experiments/monte_carlo.py --runs 500 --sin-graficos

Salida:
    experiments/results/monte_carlo_resultados.csv
    experiments/results/figures/mc_histogramas.png
    experiments/results/figures/mc_comparativa.png
    experiments/results/figures/mc_perturbacion.png
"""

import sys
import csv
import math
import random
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph import GrafoCursos, Curso
from search import astar, greedy

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

CSV_MC = RESULTS_DIR / "monte_carlo_resultados.csv"
SEP    = "═" * 64


# ── Generación de variables aleatorias via TTI ────────────────────────────────

def triangular_tti(a: float, b: float, c: float) -> float:
    """
    Genera una variable aleatoria con distribución triangular
    mediante la Transformada de la Inversa (TTI).

    Parámetros:
      a : mínimo
      b : máximo
      c : moda (valor más probable), con a ≤ c ≤ b

    La FDA de la triangular es:
      F(x) = (x-a)^2 / [(b-a)(c-a)]        si a ≤ x ≤ c
      F(x) = 1 - (b-x)^2 / [(b-a)(b-c)]    si c < x ≤ b

    Invirtiendo F(U) = x para U ~ Uniforme(0,1):
      Si U ≤ (c-a)/(b-a):  x = a + sqrt(U · (b-a) · (c-a))
      Si U >  (c-a)/(b-a): x = b - sqrt((1-U) · (b-a) · (b-c))
    """
    u  = random.random()
    fc = (c - a) / (b - a)

    if u <= fc:
        return a + math.sqrt(u * (b - a) * (c - a))
    else:
        return b - math.sqrt((1.0 - u) * (b - a) * (b - c))


def perturbar_duracion(duracion_original: int, factor: float = 0.20) -> int:
    """
    Aplica una perturbación triangular a la duración de un curso.

    Modela la incertidumbre en el tiempo real de completar un curso:
    un estudiante puede tardar entre (1-factor) y (1+factor) veces
    la duración nominal, con mayor probabilidad en la duración nominal (moda).

    Retorna la duración perturbada redondeada (mínimo 1 semana).
    """
    a = duracion_original * (1.0 - factor)
    b = duracion_original * (1.0 + factor)
    c = float(duracion_original)
    return max(1, round(triangular_tti(a, b, c)))


# ── Grafo con duraciones perturbadas ─────────────────────────────────────────

class _GrafoPerturbado:
    """
    Proxy de GrafoCursos con duraciones de cursos perturbadas
    mediante distribución triangular. No modifica el grafo original.
    """

    def __init__(self, grafo: GrafoCursos, factor: float = 0.20):
        self.perfiles    = grafo.perfiles
        self.habilidades = grafo.habilidades
        self.cursos: dict[str, Curso] = {}

        for cid, curso in grafo.cursos.items():
            nueva_dur = perturbar_duracion(curso.duracion_semanas, factor)
            self.cursos[cid] = Curso(
                id=curso.id,
                nombre=curso.nombre,
                descripcion=curso.descripcion,
                prerrequisitos=curso.prerrequisitos,
                habilidades=curso.habilidades,
                duracion_semanas=nueva_dur,
                nivel=curso.nivel,
            )

    def cursos_disponibles(self, hab, tomados):
        return [
            c for c in self.cursos.values()
            if c.id not in tomados
            and c.prerrequisitos.issubset(hab)
            and not c.habilidades.issubset(hab)
        ]

    def aplicar_curso(self, hab, curso):
        return hab | curso.habilidades

    def es_objetivo(self, hab, perfil_id):
        return self.perfiles[perfil_id].habilidades_requeridas.issubset(hab)

    def habilidades_faltantes(self, hab, perfil_id):
        return self.perfiles[perfil_id].habilidades_requeridas - hab


# ── Generación de instancias aleatorias ──────────────────────────────────────

def generar_instancia_aleatoria(
    grafo: GrafoCursos,
    max_habs_iniciales: int = 5,
) -> tuple[frozenset, str]:
    """
    Genera una instancia aleatoria:
      - Perfil objetivo: uniforme entre todos los disponibles
      - Habilidades iniciales: subconjunto aleatorio de 0 a max_habs_iniciales
        (garantiza que no satisfagan ya el perfil completo)
    """
    perfil_id = random.choice(list(grafo.perfiles.keys()))
    perfil    = grafo.perfiles[perfil_id]
    todas     = list(grafo.habilidades)
    n         = random.randint(0, min(max_habs_iniciales, len(todas) - 1))
    habs_ini  = frozenset(random.sample(todas, n))

    # Asegurar que no satisfaga ya el objetivo
    while perfil.habilidades_requeridas.issubset(habs_ini) and habs_ini:
        habs_ini = frozenset(list(habs_ini)[:-1])

    return habs_ini, perfil_id


# ── Estructura de resultado ───────────────────────────────────────────────────

@dataclass
class ResultadoMC:
    corrida:             int
    perfil_id:           str
    n_habs_iniciales:    int
    algoritmo:           str
    exito:               bool
    num_cursos:          int
    semanas_perturbadas: int
    semanas_nominales:   int
    nodos_expandidos:    int
    tiempo_segundos:     float


# ── Simulación ────────────────────────────────────────────────────────────────

def ejecutar_simulacion(
    grafo:          GrafoCursos,
    n_runs:         int   = 500,
    factor_perturb: float = 0.20,
    max_habs_ini:   int   = 5,
) -> list[ResultadoMC]:
    """
    Ejecuta n_runs corridas Monte Carlo independientes.
    Cada corrida genera una instancia aleatoria y corre A* y Greedy
    sobre un grafo con duraciones perturbadas.
    """
    resultados: list[ResultadoMC] = []
    exitos_a = exitos_g = 0
    t0 = time.perf_counter()

    print(f"\n  Ejecutando {n_runs} corridas Monte Carlo...")
    print(f"  Perturbación: ±{int(factor_perturb*100)}% (distribución triangular TTI)")
    print(f"  Habilidades iniciales por corrida: 0–{max_habs_ini}\n")

    for i in range(1, n_runs + 1):
        habs_ini, perfil_id = generar_instancia_aleatoria(grafo, max_habs_ini)
        grafo_pert          = _GrafoPerturbado(grafo, factor_perturb)

        for alg_fn, alg_nombre in [(astar, "A*"), (greedy, "Greedy")]:
            try:
                if alg_fn is astar:
                    r = astar(grafo_pert, habs_ini, perfil_id,
                              f"mc_{i}", criterio="cursos", max_nodos=5_000)
                else:
                    r = greedy(grafo_pert, habs_ini, perfil_id, f"mc_{i}")
            except Exception:
                resultados.append(ResultadoMC(
                    corrida=i, perfil_id=perfil_id,
                    n_habs_iniciales=len(habs_ini),
                    algoritmo=alg_nombre, exito=False,
                    num_cursos=0, semanas_perturbadas=0,
                    semanas_nominales=0, nodos_expandidos=0,
                    tiempo_segundos=0.0,
                ))
                continue

            sem_nom = sum(
                grafo.cursos[c.id].duracion_semanas for c in r.trayectoria
            ) if r.exito else 0

            resultados.append(ResultadoMC(
                corrida=i, perfil_id=perfil_id,
                n_habs_iniciales=len(habs_ini),
                algoritmo=alg_nombre, exito=r.exito,
                num_cursos=r.num_cursos,
                semanas_perturbadas=r.costo_total_semanas,
                semanas_nominales=sem_nom,
                nodos_expandidos=r.nodos_expandidos,
                tiempo_segundos=r.tiempo_segundos,
            ))

            if r.exito:
                if alg_nombre == "A*":     exitos_a += 1
                if alg_nombre == "Greedy": exitos_g += 1

        if i % 50 == 0 or i == n_runs:
            elapsed = time.perf_counter() - t0
            print(f"  [{i/n_runs*100:5.1f}%] Corrida {i:>4}/{n_runs}  "
                  f"A*={exitos_a:>4} éxitos  Greedy={exitos_g:>4} éxitos  "
                  f"t={elapsed:.1f}s")

    print()
    return resultados


# ── Análisis estadístico ──────────────────────────────────────────────────────

def ic_95(valores: list[float]) -> tuple[float, float]:
    """IC 95% para la media usando z=1.96 (válido para n≥30)."""
    if len(valores) < 2:
        return (0.0, 0.0)
    n   = len(valores)
    mu  = mean(valores)
    s   = stdev(valores)
    err = 1.96 * s / math.sqrt(n)
    return (mu - err, mu + err)


def analizar(resultados: list[ResultadoMC]) -> dict:
    stats = {}
    for alg in ["A*", "Greedy"]:
        exit_rows  = [r for r in resultados if r.algoritmo == alg and r.exito]
        total_rows = [r for r in resultados if r.algoritmo == alg]

        if not exit_rows:
            stats[alg] = None
            continue

        cursos   = [r.num_cursos           for r in exit_rows]
        sem_pert = [r.semanas_perturbadas   for r in exit_rows]
        sem_nom  = [r.semanas_nominales     for r in exit_rows]
        nodos    = [r.nodos_expandidos      for r in exit_rows]
        tiempos  = [r.tiempo_segundos       for r in exit_rows]

        stats[alg] = {
            "n_total":        len(total_rows),
            "n_exito":        len(exit_rows),
            "tasa_exito":     len(exit_rows) / len(total_rows) * 100,
            "cursos_media":   mean(cursos),
            "cursos_std":     stdev(cursos)   if len(cursos)   > 1 else 0.0,
            "cursos_min":     min(cursos),
            "cursos_max":     max(cursos),
            "cursos_ic95":    ic_95([float(x) for x in cursos]),
            "sem_pert_media": mean(sem_pert),
            "sem_pert_std":   stdev(sem_pert) if len(sem_pert) > 1 else 0.0,
            "sem_nom_media":  mean(sem_nom),
            "sem_nom_std":    stdev(sem_nom)  if len(sem_nom)  > 1 else 0.0,
            "sem_ic95":       ic_95([float(x) for x in sem_pert]),
            "nodos_media":    mean(nodos),
            "nodos_std":      stdev(nodos)    if len(nodos)    > 1 else 0.0,
            "tiempo_media":   mean(tiempos),
            "tiempo_std":     stdev(tiempos)  if len(tiempos)  > 1 else 0.0,
            "cursos_raw":     cursos,
            "sem_raw":        sem_pert,
            "nodos_raw":      nodos,
        }
    return stats


def imprimir_estadisticas(stats: dict, n_runs: int) -> None:
    print(f"\n{SEP}")
    print(f"  RESULTADOS MONTE CARLO — {n_runs} corridas")
    print(SEP)

    for alg in ["A*", "Greedy"]:
        s = stats.get(alg)
        if not s:
            continue
        ic_c = s["cursos_ic95"]
        ic_s = s["sem_ic95"]

        print(f"\n  {'─'*60}")
        print(f"  Algoritmo : {alg}")
        print(f"  {'─'*60}")
        print(f"  Corridas totales   : {s['n_total']}")
        print(f"  Corridas exitosas  : {s['n_exito']} ({s['tasa_exito']:.1f}%)")
        print(f"\n  Número de cursos:")
        print(f"    Media ± Desv.std : {s['cursos_media']:.2f} ± {s['cursos_std']:.2f}")
        print(f"    IC 95%           : [{ic_c[0]:.2f}, {ic_c[1]:.2f}]")
        print(f"    Rango            : [{s['cursos_min']}, {s['cursos_max']}]")
        print(f"\n  Semanas (perturbadas con TTI triangular):")
        print(f"    Media ± Desv.std : {s['sem_pert_media']:.2f} ± {s['sem_pert_std']:.2f}")
        print(f"    IC 95%           : [{ic_s[0]:.2f}, {ic_s[1]:.2f}]")
        print(f"\n  Semanas (nominales deterministas):")
        print(f"    Media ± Desv.std : {s['sem_nom_media']:.2f} ± {s['sem_nom_std']:.2f}")
        print(f"\n  Nodos expandidos:")
        print(f"    Media ± Desv.std : {s['nodos_media']:.1f} ± {s['nodos_std']:.1f}")
        print(f"\n  Tiempo de búsqueda:")
        print(f"    Media ± Desv.std : {s['tiempo_media']:.4f}s ± {s['tiempo_std']:.4f}s")

    sa = stats.get("A*")
    sg = stats.get("Greedy")
    if sa and sg:
        print(f"\n  {'─'*60}")
        print(f"  COMPARATIVA A* vs Greedy")
        print(f"  {'─'*60}")
        dc = sa["cursos_media"]   - sg["cursos_media"]
        ds = sa["sem_pert_media"] - sg["sem_pert_media"]
        rn = sa["nodos_media"]    / max(sg["nodos_media"], 1)
        print(f"  Δ cursos (A* - Greedy)   : {dc:+.2f} "
              f"({'A* usa más' if dc > 0 else 'A* usa menos o igual'})")
        print(f"  Δ semanas (A* - Greedy)  : {ds:+.2f} "
              f"({'A* más largo' if ds > 0 else 'A* más corto o igual'})")
        ratio_inv = 1/rn if rn > 0 else 0
        print(f"  Ratio nodos A*/Greedy    : {rn:.3f}x")
        print(f"  Ratio nodos Greedy/A*    : {ratio_inv:.1f}x (Greedy expande más nodos)")

# ── CSV ───────────────────────────────────────────────────────────────────────

def guardar_csv(resultados: list[ResultadoMC]) -> None:
    with open(CSV_MC, "w", newline="", encoding="utf-8") as f:
        campos = ["corrida", "perfil_id", "n_habs_iniciales", "algoritmo",
                  "exito", "num_cursos", "semanas_perturbadas",
                  "semanas_nominales", "nodos_expandidos", "tiempo_segundos"]
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in resultados:
            w.writerow({
                "corrida":             r.corrida,
                "perfil_id":           r.perfil_id,
                "n_habs_iniciales":    r.n_habs_iniciales,
                "algoritmo":           r.algoritmo,
                "exito":               r.exito,
                "num_cursos":          r.num_cursos,
                "semanas_perturbadas": r.semanas_perturbadas,
                "semanas_nominales":   r.semanas_nominales,
                "nodos_expandidos":    r.nodos_expandidos,
                "tiempo_segundos":     round(r.tiempo_segundos, 6),
            })
    print(f"  ✓ CSV guardado en: {CSV_MC}")


# ── Visualizaciones ───────────────────────────────────────────────────────────

def generar_graficos(stats: dict, resultados: list[ResultadoMC]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  ⚠ matplotlib no disponible. Omitiendo gráficos.")
        return

    C_A  = "#2563EB"
    C_G  = "#F59E0B"
    FG   = "#0F172A"
    BG   = "#1E293B"

    def _estilo(fig, axes):
        fig.patch.set_facecolor(FG)
        for ax in (axes if hasattr(axes, "__iter__") else [axes]):
            ax.set_facecolor(BG)
            ax.tick_params(colors="#94A3B8")
            ax.yaxis.label.set_color("#CBD5E1")
            ax.xaxis.label.set_color("#CBD5E1")
            ax.title.set_color("white")
            for sp in ax.spines.values():
                sp.set_color("#334155")
            ax.grid(axis="y", color="#334155", alpha=0.4, zorder=0)

    sa = stats.get("A*")
    sg = stats.get("Greedy")
    if not sa or not sg:
        print("  ⚠ Datos insuficientes para gráficos.")
        return

    # ── Gráfico 1: Histogramas de distribución ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _estilo(fig, axes)

    pares = [
        (sa["cursos_raw"],  sg["cursos_raw"],  "Número de cursos",            axes[0]),
        (sa["sem_raw"],     sg["sem_raw"],      "Semanas (perturbadas)",       axes[1]),
        (sa["nodos_raw"],   sg["nodos_raw"],    "Nodos expandidos",            axes[2]),
    ]
    for da, dg, titulo, ax in pares:
        todos = da + dg
        bins  = min(30, max(10, len(set(todos))))
        rng   = (min(todos), max(todos))
        ax.hist(da, bins=bins, range=rng, color=C_A, alpha=0.65, label="A*",     zorder=3)
        ax.hist(dg, bins=bins, range=rng, color=C_G, alpha=0.65, label="Greedy", zorder=3)
        ax.axvline(mean(da), color=C_A, linestyle="--", linewidth=1.5, alpha=0.9,
                   label=f"μ A* = {mean(da):.1f}")
        ax.axvline(mean(dg), color=C_G, linestyle="--", linewidth=1.5, alpha=0.9,
                   label=f"μ Greedy = {mean(dg):.1f}")
        ax.set_title(titulo, fontsize=10, fontweight="bold", pad=8)
        ax.set_ylabel("Frecuencia", fontsize=9)
        ax.legend(fontsize=7.5, framealpha=0.2, labelcolor="white", facecolor=BG)

    fig.suptitle("Distribuciones Monte Carlo — A* vs Greedy",
                 color="white", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "mc_histogramas.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {out}")

    # ── Gráfico 2: Medias con IC 95% ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    _estilo(fig, axes)

    metricas = [
        ("cursos_media",   "cursos_ic95",   "cursos_std",   "Número de cursos",      axes[0]),
        ("sem_pert_media", "sem_ic95",      "sem_pert_std", "Semanas (perturbadas)", axes[1]),
    ]
    algs    = ["A*", "Greedy"]
    colores = [C_A, C_G]

    for m_key, ic_key, std_key, titulo, ax in metricas:
        medias = [stats[a][m_key]   for a in algs]
        ics    = [stats[a][ic_key]  for a in algs]
        x      = np.arange(len(algs))
        bars   = ax.bar(x, medias, width=0.45, color=colores, alpha=0.85, zorder=3)

        for j, (bar, ic) in enumerate(zip(bars, ics)):
            ax.errorbar(
                bar.get_x() + bar.get_width() / 2, medias[j],
                yerr=[[medias[j] - ic[0]], [ic[1] - medias[j]]],
                fmt="none", color="white", capsize=5,
                linewidth=1.5, capthick=1.5, zorder=4,
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ic[1] + 0.3, f"{medias[j]:.2f}",
                ha="center", va="bottom",
                fontsize=9, color="white", fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(algs, color="#CBD5E1", fontsize=10)
        ax.set_ylabel(titulo, fontsize=9)
        ax.set_title(f"{titulo}\nmedia ± IC 95%",
                     fontsize=10, fontweight="bold", pad=8)

    fig.suptitle("Comparativa Estadística Monte Carlo",
                 color="white", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "mc_comparativa.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {out}")

    # ── Gráfico 3: Efecto de la perturbación (nominal vs perturbado) ──────────
    fig, ax = plt.subplots(figsize=(9, 5))
    _estilo(fig, [ax])

    for alg, color in [("A*", C_A), ("Greedy", C_G)]:
        filas = [r for r in resultados if r.algoritmo == alg and r.exito]
        nom   = [r.semanas_nominales   for r in filas]
        pert  = [r.semanas_perturbadas for r in filas]
        muestra = min(200, len(nom))
        idx = random.sample(range(len(nom)), muestra)
        ax.scatter([nom[i]  for i in idx], [pert[i] for i in idx],
                   color=color, alpha=0.35, s=10, label=alg, zorder=3)

    lim = max(
        max(r.semanas_nominales   for r in resultados if r.exito),
        max(r.semanas_perturbadas for r in resultados if r.exito),
    ) + 3
    ax.plot([0, lim], [0, lim], color="#475569", linestyle="--",
            linewidth=1, alpha=0.7, label="Sin perturbación", zorder=2)

    ax.set_xlabel("Semanas nominales (deterministas)",    fontsize=9)
    ax.set_ylabel("Semanas perturbadas (estocásticas)",   fontsize=9)
    ax.set_title("Efecto de la perturbación triangular ±20%\n"
                 "Semanas nominales vs semanas estocásticas",
                 fontsize=10, fontweight="bold", pad=8)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="white", facecolor=BG)

    plt.tight_layout()
    out = FIGURES_DIR / "mc_perturbacion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulación Monte Carlo del Career Path Planner"
    )
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--factor", type=float, default=0.20)
    parser.add_argument("--max-habs-ini", type=int, default=5)
    parser.add_argument("--sin-graficos", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    print(SEP)
    print("  SIMULACIÓN MONTE CARLO — Career Path Planner")
    print("  Generación de variables via Transformada de la Inversa (TTI)")
    print(SEP)
    print(f"\n  Parámetros:")
    print(f"    Corridas             : {args.runs}")
    print(f"    Semilla aleatoria    : {args.seed}")
    print(f"    Perturbación         : ±{int(args.factor*100)}% triangular")
    print(f"    Máx habs iniciales   : {args.max_habs_ini}")

    try:
        grafo = GrafoCursos()
    except FileNotFoundError as e:
        print(f"\n  ✗ {e}")
        sys.exit(1)

    print(f"\n  Grafo: {len(grafo.cursos)} cursos | "
          f"{len(grafo.habilidades)} habilidades | "
          f"{len(grafo.perfiles)} perfiles")

    t0         = time.perf_counter()
    resultados = ejecutar_simulacion(
        grafo,
        n_runs=args.runs,
        factor_perturb=args.factor,
        max_habs_ini=args.max_habs_ini,
    )
    t_total = time.perf_counter() - t0

    stats = analizar(resultados)
    imprimir_estadisticas(stats, args.runs)

    print(f"\n  Tiempo total      : {t_total:.2f}s")
    print(f"  Tiempo por corrida : {t_total / args.runs * 1000:.1f}ms")

    print()
    guardar_csv(resultados)

    if not args.sin_graficos:
        print()
        generar_graficos(stats, resultados)

    print(f"\n{SEP}")
    print("  ✓ Simulación Monte Carlo completada.")
    print(SEP)


if __name__ == "__main__":
    main()
