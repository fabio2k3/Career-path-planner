"""
run_experiments.py
------------------
Diseño experimental del proyecto — ejecuta las 3 configuraciones
sobre las instancias del dataset y guarda los resultados en CSV.

Configuraciones:
  (a) A* criterio=semanas    sin LLM   → minimiza semanas totales
  (b) A* criterio=cursos     sin LLM   → minimiza número de cursos
  (c) Greedy                 sin LLM
  (d) A* criterio=cursos     con LLM   → misma configuración + evaluación semántica

Métricas registradas por instancia:
  - num_cursos, costo_total_semanas
  - nodos_expandidos, tiempo_segundos
  - trayectoria_valida, trayectoria_ids
  - puntuacion_llm, nivel_calidad_llm   (solo config d)
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fila_base(inst, nombre_alg: str, r, valido: bool) -> dict:
    """Construye la fila de resultados base para el CSV sin LLM."""
    return {
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
        "trayectoria_ids":     ";".join(c.id for c in r.trayectoria),
    }


# ─── Experimento sin LLM: A*(semanas), A*(cursos) y Greedy ───────────────────

def experimento_sin_llm(grafo: GrafoCursos, instancias: list) -> list:
    """
    Ejecuta tres algoritmos sobre todas las instancias:
      · A* con criterio 'semanas'  (minimiza duración total)
      · A* con criterio 'cursos'   (minimiza número de pasos)
      · Greedy                     (referencia no óptima)
    """
    configs = [
        (lambda g, hi, po, iid: astar(g, hi, po, iid, criterio="semanas"), "A* (semanas)"),
        (lambda g, hi, po, iid: astar(g, hi, po, iid, criterio="cursos"),  "A* (cursos)"),
        (greedy, "Greedy"),
    ]

    print(f"\n{SEP}")
    print(f"  EXPERIMENTO — A*(semanas) + A*(cursos) + Greedy sin LLM")
    print(f"  {len(instancias)} instancias × {len(configs)} configuraciones = "
          f"{len(instancias) * len(configs)} ejecuciones")
    print(SEP)

    filas = []

    for inst in instancias:
        print(f"\n  [{inst.id}] {inst.descripcion}")

        for alg_fn, nombre_alg in configs:
            r = alg_fn(grafo, inst.habilidades_iniciales, inst.perfil_objetivo, inst.id)

            valido = False
            if r.exito:
                valido, _ = validar_trayectoria(
                    grafo, r.trayectoria,
                    inst.habilidades_iniciales, inst.perfil_objetivo
                )

            filas.append(_fila_base(inst, nombre_alg, r, valido))

            status = "✓" if r.exito else "✗"
            print(f"    {status} {nombre_alg:<16} "
                  f"cursos={r.num_cursos:>3}  "
                  f"semanas={r.costo_total_semanas:>3}  "
                  f"nodos={r.nodos_expandidos:>7}  "
                  f"t={r.tiempo_segundos:.4f}s  "
                  f"válida={'✓' if valido else '✗'}")

    with open(CSV_SIN_LLM, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=filas[0].keys())
        writer.writeheader()
        writer.writerows(filas)

    print(f"\n  ✓ Resultados guardados en: {CSV_SIN_LLM}")
    return filas


# ─── Experimento con LLM: A*(cursos) + evaluación semántica ──────────────────

def experimento_con_llm(grafo: GrafoCursos, instancias: list) -> list:
    """
    Ejecuta A*(cursos) sobre todas las instancias y evalúa cada trayectoria
    con el LLM. Usa fallback simulado si la API no está disponible.
    """
    print(f"\n{SEP}")
    print(f"  EXPERIMENTO C — A*(cursos) con evaluación LLM")
    print(f"  {len(instancias)} instancias")
    print(f"  (Incluye llamadas al LLM — puede tardar ~2-5s por instancia)")
    print(SEP)

    try:
        from llm_integration import evaluar_trayectoria_con_fallback
    except ImportError as e:
        print(f"  ✗ No se pudo importar llm_integration: {e}")
        return []

    filas = []

    for inst in instancias:
        print(f"\n  [{inst.id}] {inst.descripcion}")

        r = astar(grafo, inst.habilidades_iniciales,
                  inst.perfil_objetivo, inst.id, criterio="cursos")

        if not r.exito:
            print(f"    ✗ A*(cursos) no encontró trayectoria.")
            # Registrar el fallo igualmente para que el CSV tenga todas las instancias
            filas.append({
                "instancia_id":        inst.id,
                "descripcion":         inst.descripcion,
                "perfil_objetivo":     inst.perfil_objetivo,
                "habs_iniciales":      len(inst.habilidades_iniciales),
                "algoritmo":           "A* (cursos) + LLM",
                "exito":               False,
                "num_cursos":          0,
                "costo_total_semanas": 0,
                "nodos_expandidos":    r.nodos_expandidos,
                "tiempo_busqueda_s":   round(r.tiempo_segundos, 6),
                "trayectoria_valida":  False,
                "puntuacion_llm":      None,
                "nivel_calidad_llm":   None,
                "tiempo_llm_s":        None,
                "resumen_llm":         None,
                "modo_evaluacion":     None,
                "trayectoria_ids":     "",
            })
            continue

        valido, _ = validar_trayectoria(
            grafo, r.trayectoria, inst.habilidades_iniciales, inst.perfil_objetivo
        )

        # Evaluar con LLM (con fallback automático)
        puntuacion_llm = None
        nivel_llm      = None
        tiempo_llm     = None
        resumen_llm    = None
        modo_eval      = None

        try:
            ev = evaluar_trayectoria_con_fallback(
                inst.objetivo_texto,
                inst.perfil_objetivo,
                r.trayectoria,
                inst.habilidades_iniciales,
            )
            puntuacion_llm = ev.get("puntuacion")
            nivel_llm      = ev.get("nivel_calidad")
            tiempo_llm     = ev.get("tiempo_segundos")
            resumen_llm    = ev.get("resumen", "")
            modo_eval      = ev.get("modo", "desconocido")

            print(f"    ✓ A*(cursos)  cursos={r.num_cursos}  "
                  f"semanas={r.costo_total_semanas}  "
                  f"LLM={puntuacion_llm}/10 ({nivel_llm})  [{modo_eval}]")
        except Exception as e:
            print(f"    ⚠ Evaluación LLM fallida para {inst.id}: {e}")

        filas.append({
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
            "modo_evaluacion":     modo_eval,
            "trayectoria_ids":     ";".join(c.id for c in r.trayectoria),
        })

    if filas:
        with open(CSV_CON_LLM, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=filas[0].keys())
            writer.writeheader()
            writer.writerows(filas)
        print(f"\n  ✓ Resultados guardados en: {CSV_CON_LLM}")

    return filas


# ─── Análisis comparativo ─────────────────────────────────────────────────────

def analisis_comparativo(filas_sin_llm: list, filas_con_llm: list):
    print(f"\n{SEP}")
    print(f"  ANÁLISIS COMPARATIVO")
    print(SEP)

    # Indexar por instancia para comparación segura (evita problemas de orden)
    astar_sem  = {f["instancia_id"]: f for f in filas_sin_llm if f["algoritmo"] == "A* (semanas)"}
    astar_cur  = {f["instancia_id"]: f for f in filas_sin_llm if f["algoritmo"] == "A* (cursos)"}
    greedy_d   = {f["instancia_id"]: f for f in filas_sin_llm if f["algoritmo"] == "Greedy"}

    instancias_ids = sorted(astar_sem.keys())

    # 1. Calidad de solución
    print(f"\n  1. Calidad de solución — Cursos y Semanas")
    print(f"  {'Instancia':<12} {'A*(sem)C':>9} {'A*(cur)C':>9} {'GreedyC':>8} "
          f"{'A*(sem)S':>9} {'A*(cur)S':>9} {'GreedyS':>8}")
    print(f"  {'─'*68}")

    for iid in instancias_ids:
        ra_s = astar_sem.get(iid, {})
        ra_c = astar_cur.get(iid, {})
        rg   = greedy_d.get(iid, {})
        print(f"  {iid:<12} "
              f"{str(ra_s.get('num_cursos','—')):>9} "
              f"{str(ra_c.get('num_cursos','—')):>9} "
              f"{str(rg.get('num_cursos','—')):>8} "
              f"{str(ra_s.get('costo_total_semanas','—')):>9} "
              f"{str(ra_c.get('costo_total_semanas','—')):>9} "
              f"{str(rg.get('costo_total_semanas','—')):>8}")

    # 2. Eficiencia computacional
    print(f"\n  2. Eficiencia computacional — Nodos expandidos")
    print(f"  {'Instancia':<12} {'A*(sem)N':>10} {'A*(cur)N':>10} {'GreedyN':>8} "
          f"{'Ratio A*/G':>12}")
    print(f"  {'─'*56}")

    for iid in instancias_ids:
        ra_s = astar_sem.get(iid, {})
        rg   = greedy_d.get(iid, {})
        n_a = ra_s.get("nodos_expandidos", 0) or 0
        n_g = rg.get("nodos_expandidos", 1) or 1
        ratio = n_a / n_g
        ra_c = astar_cur.get(iid, {})
        print(f"  {iid:<12} {n_a:>10} "
              f"{str(ra_c.get('nodos_expandidos','—')):>10} "
              f"{n_g:>8} {ratio:>11.1f}x")

    # 3. Evaluaciones LLM
    if filas_con_llm:
        puntuaciones = [(f["instancia_id"], f.get("puntuacion_llm"))
                        for f in filas_con_llm if f.get("puntuacion_llm") is not None]
        if puntuaciones:
            print(f"\n  3. Evaluaciones LLM — Puntuaciones")
            print(f"  {'Instancia':<12} {'Perfil':<22} {'Puntuación':>12} {'Calidad':<12} {'Modo':<12}")
            print(f"  {'─'*72}")
            for f in filas_con_llm:
                if f.get("puntuacion_llm") is not None:
                    print(f"  {f['instancia_id']:<12} {f['perfil_objetivo']:<22} "
                          f"{str(f['puntuacion_llm'])+'/10':>12} "
                          f"{str(f.get('nivel_calidad_llm', '?')):<12} "
                          f"{str(f.get('modo_evaluacion', '?')):<12}")
            valores = [p for _, p in puntuaciones]
            print(f"\n  Puntuación media   : {sum(valores)/len(valores):.2f}/10")
            print(f"  Puntuación mínima  : {min(valores)}/10")
            print(f"  Puntuación máxima  : {max(valores)}/10")

    # Guardar detalle completo en JSON
    detalle = {"sin_llm": filas_sin_llm, "con_llm": filas_con_llm}
    with open(JSON_DETALLE, "w", encoding="utf-8") as f:
        json.dump(detalle, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Detalle completo guardado en: {JSON_DETALLE}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  DISEÑO EXPERIMENTAL — Career Path Planner")
    print("  Configuraciones: A*(semanas) | A*(cursos) | Greedy | A*(cursos)+LLM")
    print(SEP)

    grafo     = GrafoCursos()
    instancias = grafo.cargar_instancias()

    print(f"\n  Grafo: {len(grafo.cursos)} cursos | "
          f"{len(grafo.habilidades)} habilidades | "
          f"{len(grafo.perfiles)} perfiles")
    print(f"  Instancias: {len(instancias)}")

    # Experimentos sin LLM
    filas_sin_llm = experimento_sin_llm(grafo, instancias)

    # Experimento con LLM
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