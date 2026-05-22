"""
run_experiments.py
------------------
Diseño experimental del proyecto — ejecuta las 3 configuraciones
sobre las 8 instancias del dataset y guarda los resultados en CSV.

Configuraciones:
  (a) A* sin LLM   (criterio=cursos)
  (b) Greedy sin LLM
  (c) A* con evaluación LLM (criterio=cursos)

Métricas registradas por instancia:
  - num_cursos, costo_total_semanas
  - nodos_expandidos, tiempo_segundos
  - puntuacion_llm, nivel_calidad_llm   (solo config c)
  - trayectoria_valida
"""

import sys
import csv
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph import GrafoCursos
from search import astar, greedy, validar_trayectoria

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_SIN_LLM  = RESULTS_DIR / "resultados_sin_llm.csv"
CSV_CON_LLM  = RESULTS_DIR / "resultados_con_llm.csv"
JSON_DETALLE = RESULTS_DIR / "detalle_completo.json"

SEP = "═" * 64


# ── Experimento A y B: A* y Greedy sin LLM ────────────────────────────────────

def experimento_sin_llm(grafo, instancias):
    print(f"\n{SEP}")
    print(f"  EXPERIMENTO A+B — A* y Greedy sin LLM")
    print(f"  {len(instancias)} instancias × 2 algoritmos = "
          f"{len(instancias)*2} ejecuciones")
    print(SEP)

    filas = []

    for inst in instancias:
        print(f"\n  [{inst.id}] {inst.descripcion}")

        for algoritmo_fn, criterio in [(astar, "cursos"), (greedy, None)]:
            if algoritmo_fn == astar:
                r = astar(grafo, inst.habilidades_iniciales,
                          inst.perfil_objetivo, inst.id,
                          criterio=criterio)
                nombre_alg = f"A* ({criterio})"
            else:
                r = greedy(grafo, inst.habilidades_iniciales,
                           inst.perfil_objetivo, inst.id)
                nombre_alg = "Greedy"

            # Validar trayectoria
            valido = False
            if r.exito:
                valido, _ = validar_trayectoria(
                    grafo, r.trayectoria,
                    inst.habilidades_iniciales, inst.perfil_objetivo
                )

            fila = {
                "instancia_id":        inst.id,
                "descripcion":         inst.descripcion,
                "perfil_objetivo":     inst.perfil_objetivo,
                "habs_iniciales":      len(inst.habilidades_iniciales),
                "algoritmo":           nombre_alg,
                "exito":               r.exito,
                "num_cursos":          r.num_cursos,
                "costo_total_semanas": r.costo_total_semanas,
                "nodos_expandidos":    r.nodos_expandidos,
                "tiempo_segundos":     round(r.tiempo_segundos, 6),
                "trayectoria_valida":  valido,
                "trayectoria_ids":     "|".join(c.id for c in r.trayectoria),
            }
            filas.append(fila)

            status = "✓" if r.exito else "✗"
            print(f"    {status} {nombre_alg:<15} "
                  f"cursos={r.num_cursos:>3}  "
                  f"semanas={r.costo_total_semanas:>3}  "
                  f"nodos={r.nodos_expandidos:>7}  "
                  f"t={r.tiempo_segundos:.4f}s")

    # Guardar CSV
    with open(CSV_SIN_LLM, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=filas[0].keys())
        writer.writeheader()
        writer.writerows(filas)

    print(f"\n  ✓ Resultados guardados en: {CSV_SIN_LLM}")
    return filas


# ── Experimento C: A* con evaluación LLM ─────────────────────────────────────

def experimento_con_llm(grafo, instancias):
    print(f"\n{SEP}")
    print(f"  EXPERIMENTO C — A* con evaluación LLM")
    print(f"  {len(instancias)} instancias × 1 configuración")
    print(f"  (Nota: incluye llamadas al LLM — puede tardar ~2-5s por instancia)")
    print(SEP)

    try:
        from llm_integration import evaluar_trayectoria
    except ImportError as e:
        print(f"  ✗ No se pudo importar llm_integration: {e}")
        return []

    filas = []

    for inst in instancias:
        print(f"\n  [{inst.id}] {inst.descripcion}")

        # Generar trayectoria con A*
        r = astar(grafo, inst.habilidades_iniciales,
                  inst.perfil_objetivo, inst.id, criterio="cursos")

        if not r.exito:
            print(f"    ✗ A* no encontró trayectoria.")
            continue

        valido, _ = validar_trayectoria(
            grafo, r.trayectoria,
            inst.habilidades_iniciales, inst.perfil_objetivo
        )

        # Evaluar con LLM
        puntuacion_llm = None
        nivel_llm      = None
        tiempo_llm     = None
        resumen_llm    = None

        try:
            time.sleep(10)  # delay para evitar rate limit del tier gratuito
            ev = evaluar_trayectoria(
                inst.objetivo_texto,
                inst.perfil_objetivo,
                r.trayectoria,
                inst.habilidades_iniciales,
            )
            puntuacion_llm = ev.get("puntuacion")
            nivel_llm      = ev.get("nivel_calidad")
            tiempo_llm     = ev.get("tiempo_segundos")
            resumen_llm    = ev.get("resumen", "")
            print(f"    ✓ A*(cursos)  cursos={r.num_cursos}  "
                  f"semanas={r.costo_total_semanas}  "
                  f"LLM={puntuacion_llm}/10 ({nivel_llm})")
        except Exception as e:
            print(f"    ⚠ A*(cursos)  cursos={r.num_cursos}  "
                  f"semanas={r.costo_total_semanas}  "
                  f"LLM=error ({e})")

        fila = {
            "instancia_id":        inst.id,
            "descripcion":         inst.descripcion,
            "perfil_objetivo":     inst.perfil_objetivo,
            "habs_iniciales":      len(inst.habilidades_iniciales),
            "algoritmo":           "A* (cursos) + LLM",
            "exito":               r.exito,
            "num_cursos":          r.num_cursos,
            "costo_total_semanas": r.costo_total_semanas,
            "nodos_expandidos":    r.nodos_expandidos,
            "tiempo_busqueda_s":   round(r.tiempo_segundos, 6),
            "trayectoria_valida":  valido,
            "puntuacion_llm":      puntuacion_llm,
            "nivel_calidad_llm":   nivel_llm,
            "tiempo_llm_s":        tiempo_llm,
            "resumen_llm":         resumen_llm,
            "trayectoria_ids":     "|".join(c.id for c in r.trayectoria),
        }
        filas.append(fila)

    if filas:
        with open(CSV_CON_LLM, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=filas[0].keys())
            writer.writeheader()
            writer.writerows(filas)
        print(f"\n  ✓ Resultados guardados en: {CSV_CON_LLM}")

    return filas


# ── Análisis comparativo ──────────────────────────────────────────────────────

def analisis_comparativo(filas_sin_llm, filas_con_llm):
    print(f"\n{SEP}")
    print(f"  ANÁLISIS COMPARATIVO")
    print(SEP)

    # Separar A* y Greedy
    astar_rows  = [f for f in filas_sin_llm if "A*" in f["algoritmo"]]
    greedy_rows = [f for f in filas_sin_llm if "Greedy" in f["algoritmo"]]

    print(f"\n  1. A* vs Greedy — Calidad de solución")
    print(f"  {'Instancia':<12} {'A* Cursos':>10} {'Greedy C':>10} "
          f"{'A* Sem':>8} {'Greedy S':>9} {'Δ Sem':>7}")
    print(f"  {'─'*60}")

    for r_a, r_g in zip(astar_rows, greedy_rows):
        delta = r_g["costo_total_semanas"] - r_a["costo_total_semanas"]
        signo = "+" if delta > 0 else ""
        print(f"  {r_a['instancia_id']:<12} "
              f"{r_a['num_cursos']:>10} {r_g['num_cursos']:>10} "
              f"{r_a['costo_total_semanas']:>8} {r_g['costo_total_semanas']:>9} "
              f"{signo}{delta:>6}")

    # Tiempos
    print(f"\n  2. A* vs Greedy — Eficiencia computacional")
    print(f"  {'Instancia':<12} {'A* Nodos':>10} {'Greedy N':>10} "
          f"{'A* t(s)':>10} {'Greedy t':>10} {'Ratio':>8}")
    print(f"  {'─'*62}")

    for r_a, r_g in zip(astar_rows, greedy_rows):
        ratio = (r_a["nodos_expandidos"] / r_g["nodos_expandidos"]
                 if r_g["nodos_expandidos"] > 0 else float("inf"))
        print(f"  {r_a['instancia_id']:<12} "
              f"{r_a['nodos_expandidos']:>10} {r_g['nodos_expandidos']:>10} "
              f"{r_a['tiempo_segundos']:>10.4f} {r_g['tiempo_segundos']:>10.4f} "
              f"{ratio:>7.1f}x")

    # Puntuaciones LLM
    if filas_con_llm:
        puntuaciones = [f["puntuacion_llm"] for f in filas_con_llm
                        if f.get("puntuacion_llm") is not None]
        if puntuaciones:
            print(f"\n  3. Evaluaciones LLM — Puntuaciones")
            print(f"  {'Instancia':<12} {'Perfil':<22} "
                  f"{'Puntuación':>12} {'Calidad':<12}")
            print(f"  {'─'*62}")
            for f in filas_con_llm:
                if f.get("puntuacion_llm") is not None:
                    print(f"  {f['instancia_id']:<12} {f['perfil_objetivo']:<22} "
                          f"{str(f['puntuacion_llm'])+'/10':>12} "
                          f"{str(f.get('nivel_calidad_llm', '?')):<12}")
            avg = sum(puntuaciones) / len(puntuaciones)
            print(f"\n  Puntuación promedio LLM: {avg:.2f}/10")
            print(f"  Puntuación mínima      : {min(puntuaciones)}/10")
            print(f"  Puntuación máxima      : {max(puntuaciones)}/10")

    # Guardar detalle completo en JSON
    detalle = {
        "sin_llm": filas_sin_llm,
        "con_llm": filas_con_llm,
    }
    with open(JSON_DETALLE, "w", encoding="utf-8") as f:
        json.dump(detalle, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Detalle completo guardado en: {JSON_DETALLE}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  DISEÑO EXPERIMENTAL — Career Path Planner")
    print("  3 configuraciones × 8 instancias")
    print(SEP)

    grafo     = GrafoCursos()
    instancias = grafo.cargar_instancias()

    print(f"\n  Grafo: {len(grafo.cursos)} cursos | "
          f"{len(grafo.habilidades)} habilidades | "
          f"{len(grafo.perfiles)} perfiles")
    print(f"  Instancias: {len(instancias)}")

    # Experimentos A y B: sin LLM
    filas_sin_llm = experimento_sin_llm(grafo, instancias)

    # Experimento C: con LLM
    filas_con_llm = experimento_con_llm(grafo, instancias)

    # Análisis comparativo
    if filas_sin_llm:
        analisis_comparativo(filas_sin_llm, filas_con_llm)

    print(f"\n{SEP}")
    print(f"  ✓ Experimentos completados.")
    print(f"  Archivos generados:")
    print(f"    · {CSV_SIN_LLM}")
    print(f"    · {CSV_CON_LLM}")
    print(f"    · {JSON_DETALLE}")
    print(SEP)


if __name__ == "__main__":
    main()