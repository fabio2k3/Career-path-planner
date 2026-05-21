"""
run_experiments.py
------------------
Diseño experimental del sistema Career Path Planner.

Configuraciones comparadas:
  (A) A* sin LLM  — criterio semanas
  (B) A* sin LLM  — criterio cursos
  (C) Greedy sin LLM
  (D) A* + evaluación LLM — criterio cursos

Instancias: las 8 instancias del dataset.

Métricas registradas:
  - num_cursos          : longitud de la trayectoria
  - costo_total_semanas : duración total
  - nodos_expandidos    : esfuerzo computacional
  - tiempo_segundos     : tiempo de cómputo
  - puntuacion_llm      : calidad evaluada por LLM (solo config D)
  - nivel_calidad_llm   : etiqueta de calidad (solo config D)

Resultados guardados en: experiments/results/results.csv
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph import GrafoCursos
from search import astar, greedy, validar_trayectoria
from llm_integration import evaluar_trayectoria

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
RESULTS_DIR  = BASE_DIR / "results"
RESULTS_CSV  = RESULTS_DIR / "results.csv"
RESULTS_JSON = RESULTS_DIR / "results_full.json"

RESULTS_DIR.mkdir(exist_ok=True)

SEP  = "═" * 64
SEP2 = "─" * 64

# ── Objetivos en lenguaje natural por instancia (para la evaluación LLM) ─────
OBJETIVOS_NL = {
    "inst_01": "Quiero convertirme en Data Scientist desde cero",
    "inst_02": "Tengo base matematica y quiero ser Data Scientist",
    "inst_03": "Se Python basico y quiero especializarme en Data Science",
    "inst_04": "Quiero ser desarrollador backend desde cero",
    "inst_05": "Soy frontend y quiero pasarme al backend con Python",
    "inst_06": "Soy backend developer y quiero transicionarme a ML Engineering",
    "inst_07": "Soy Data Scientist y quiero llevar modelos a produccion como ML Engineer",
    "inst_08": "Quiero ser ML Engineer, aun no tengo conocimientos tecnicos",
}


def correr_configuracion(
    grafo, instancia, config_nombre, algoritmo_fn, criterio,
    con_llm=False, objetivo_nl=None,
):
    """Ejecuta una configuración sobre una instancia y devuelve los métricas."""
    # Greedy no acepta el parámetro criterio
    if algoritmo_fn.__name__ == "greedy":
        result = algoritmo_fn(
            grafo,
            instancia.habilidades_iniciales,
            instancia.perfil_objetivo,
            instancia.id,
        )
    else:
        result = algoritmo_fn(
            grafo,
            instancia.habilidades_iniciales,
            instancia.perfil_objetivo,
            instancia.id,
            criterio=criterio,
        )

    fila = {
        "configuracion":       config_nombre,
        "instancia_id":        instancia.id,
        "perfil_objetivo":     instancia.perfil_objetivo,
        "habilidades_iniciales": len(instancia.habilidades_iniciales),
        "exito":               result.exito,
        "num_cursos":          result.num_cursos if result.exito else None,
        "costo_total_semanas": result.costo_total_semanas if result.exito else None,
        "nodos_expandidos":    result.nodos_expandidos,
        "tiempo_segundos":     result.tiempo_segundos,
        "puntuacion_llm":      None,
        "nivel_calidad_llm":   None,
        "trayectoria_ids":     [c.id for c in result.trayectoria] if result.exito else [],
    }

    # Configuración D: evaluar con LLM
    if con_llm and result.exito and objetivo_nl:
        try:
            evaluacion = evaluar_trayectoria(
                objetivo_nl,
                instancia.perfil_objetivo,
                result.trayectoria,
                instancia.habilidades_iniciales,
            )
            fila["puntuacion_llm"]    = evaluacion.get("puntuacion")
            fila["nivel_calidad_llm"] = evaluacion.get("nivel_calidad")
            fila["tiempo_segundos"]  += evaluacion.get("tiempo_segundos", 0)
        except Exception as e:
            print(f"    ⚠ LLM error en {instancia.id}: {e}")

    return fila, result


def imprimir_progreso(fila, instancia_desc):
    estado = "✓" if fila["exito"] else "✗"
    llm    = f"  LLM: {fila['puntuacion_llm']}/10" if fila["puntuacion_llm"] else ""
    print(f"    {estado} [{fila['configuracion']:<16}] {instancia_desc:<45} "
          f"cursos={fila['num_cursos'] or '-':>3}  "
          f"sem={fila['costo_total_semanas'] or '-':>3}  "
          f"nodos={fila['nodos_expandidos']:>6}  "
          f"t={fila['tiempo_segundos']:.3f}s{llm}")


def guardar_resultados(filas: list):
    """Guarda los resultados en CSV y JSON."""
    campos_csv = [
        "configuracion", "instancia_id", "perfil_objetivo",
        "habilidades_iniciales", "exito",
        "num_cursos", "costo_total_semanas",
        "nodos_expandidos", "tiempo_segundos",
        "puntuacion_llm", "nivel_calidad_llm",
    ]

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos_csv, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filas)

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(filas, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ CSV  guardado: {RESULTS_CSV}")
    print(f"  ✓ JSON guardado: {RESULTS_JSON}")


def imprimir_resumen(filas: list):
    """Imprime tabla resumen comparando las 4 configuraciones."""
    from collections import defaultdict

    print(f"\n{SEP}")
    print("  RESUMEN COMPARATIVO DE CONFIGURACIONES")
    print(SEP)

    configs = ["A* (semanas)", "A* (cursos)", "Greedy", "A* + LLM"]
    metricas = defaultdict(lambda: {
        "cursos": [], "semanas": [], "nodos": [], "tiempo": [], "llm": []
    })

    for f in filas:
        if not f["exito"]:
            continue
        c = f["configuracion"]
        metricas[c]["cursos"].append(f["num_cursos"])
        metricas[c]["semanas"].append(f["costo_total_semanas"])
        metricas[c]["nodos"].append(f["nodos_expandidos"])
        metricas[c]["tiempo"].append(f["tiempo_segundos"])
        if f["puntuacion_llm"] is not None:
            metricas[c]["llm"].append(f["puntuacion_llm"])

    def avg(lst): return round(sum(lst) / len(lst), 2) if lst else "-"

    print(f"\n  {'Configuración':<18} {'Cursos':>7} {'Semanas':>8} "
          f"{'Nodos':>9} {'Tiempo(s)':>10} {'LLM':>6}")
    print(f"  {SEP2}")

    for config in configs:
        m = metricas[config]
        llm_str = str(avg(m["llm"])) if m["llm"] else "  -"
        print(f"  {config:<18} {str(avg(m['cursos'])):>7} "
              f"{str(avg(m['semanas'])):>8} "
              f"{str(avg(m['nodos'])):>9} "
              f"{str(avg(m['tiempo'])):>10} "
              f"{llm_str:>6}")

    print(f"\n  Instancias por configuración: {len(filas)//4} instancias × 4 configs")

    # Análisis de diferencias
    print(f"\n  Análisis:")
    astar_sem  = metricas["A* (semanas)"]
    astar_cur  = metricas["A* (cursos)"]
    greedy_m   = metricas["Greedy"]
    astar_llm  = metricas["A* + LLM"]

    if astar_sem["nodos"] and astar_cur["nodos"]:
        ratio = avg(astar_sem["nodos"]) / avg(astar_cur["nodos"])
        print(f"  → A*(semanas) expande {ratio:.1f}x más nodos que A*(cursos).")

    avg_greedy_t = avg(greedy_m["tiempo"])
    avg_astar_t  = avg(astar_cur["tiempo"])
    if astar_cur["tiempo"] and greedy_m["tiempo"] and avg_greedy_t > 0:
        ratio_t = avg_astar_t / avg_greedy_t
        print(f"  → A*(cursos) es {ratio_t:.1f}x más lento que Greedy, "
              f"pero garantiza optimalidad.")
    elif astar_cur["tiempo"] and greedy_m["tiempo"]:
        print(f"  → Greedy es extremadamente rápido (tiempo ≈ 0s); "
              f"A*(cursos) tarda {avg_astar_t:.4f}s promedio.")

    if astar_llm["llm"]:
        print(f"  → Puntuación promedio LLM (A* + LLM): "
              f"{avg(astar_llm['llm'])}/10")


def main():
    print(SEP)
    print("  EXPERIMENTOS — Career Path Planner")
    print("  3 configuraciones × 8 instancias")
    print(SEP)

    grafo     = GrafoCursos()
    instancias = grafo.cargar_instancias()

    print(f"\n  Grafo: {len(grafo.cursos)} cursos | "
          f"{len(grafo.habilidades)} habilidades | "
          f"{len(grafo.perfiles)} perfiles")
    print(f"  Instancias: {len(instancias)}")
    print(f"\n  Configuraciones:")
    print(f"    A  → A* criterio=semanas  (sin LLM)")
    print(f"    B  → A* criterio=cursos   (sin LLM)")
    print(f"    C  → Greedy               (sin LLM)")
    print(f"    D  → A* criterio=cursos + evaluación LLM")

    todas_filas = []
    t_inicio = time.perf_counter()

    # ── Configuración A: A* semanas ───────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  CONFIGURACIÓN A — A* (criterio=semanas)")
    print(SEP2)
    for inst in instancias:
        fila, _ = correr_configuracion(
            grafo, inst, "A* (semanas)", astar, "semanas"
        )
        todas_filas.append(fila)
        imprimir_progreso(fila, inst.descripcion)

    # ── Configuración B: A* cursos ────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  CONFIGURACIÓN B — A* (criterio=cursos)")
    print(SEP2)
    for inst in instancias:
        fila, _ = correr_configuracion(
            grafo, inst, "A* (cursos)", astar, "cursos"
        )
        todas_filas.append(fila)
        imprimir_progreso(fila, inst.descripcion)

    # ── Configuración C: Greedy ───────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  CONFIGURACIÓN C — Greedy")
    print(SEP2)
    for inst in instancias:
        fila, _ = correr_configuracion(
            grafo, inst, "Greedy", greedy, "semanas"
        )
        todas_filas.append(fila)
        imprimir_progreso(fila, inst.descripcion)

    # ── Configuración D: A* + LLM ─────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  CONFIGURACIÓN D — A* + evaluación LLM (criterio=cursos)")
    print(f"  (Incluye llamadas al LLM — puede tardar ~3s por instancia)")
    print(SEP2)
    for inst in instancias:
        obj_nl = OBJETIVOS_NL.get(inst.id, inst.objetivo_texto)
        fila, _ = correr_configuracion(
            grafo, inst, "A* + LLM", astar, "cursos",
            con_llm=True, objetivo_nl=obj_nl,
        )
        todas_filas.append(fila)
        imprimir_progreso(fila, inst.descripcion)

    t_total = time.perf_counter() - t_inicio
    print(f"\n  Tiempo total de experimentos: {t_total:.2f}s")

    # ── Guardar y resumir ─────────────────────────────────────────────────────
    guardar_resultados(todas_filas)
    imprimir_resumen(todas_filas)

    print(f"\n{SEP}")
    print(f"  ✓ Experimentos completados: {len(todas_filas)} registros totales.")
    print(SEP)


if __name__ == "__main__":
    main()