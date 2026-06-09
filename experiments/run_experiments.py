"""
run_experiments.py

Diseño experimental del proyecto — ejecuta las 3 configuraciones
sobre las instancias del dataset y guarda los resultados en CSV.

Métricas registradas por instancia:
- num_cursos, costo_total_semanas
- nodos_expandidos, tiempo_segundos
- trayectoria_valida, habs_iniciales
- puntuacion_llm, nivel_calidad_llm   (solo config c)
"""

import sys
import csv
import json
from pathlib import Path

# Se agrega la carpeta src al path para poder importar los módulos del proyecto.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph import GrafoCursos
from search import astar, greedy, validar_trayectoria

# Directorio donde se almacenan los resultados experimentales.
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_SIN_LLM = RESULTS_DIR / "resultados_sin_llm.csv"
CSV_CON_LLM = RESULTS_DIR / "resultados_con_llm.csv"
JSON_DETALLE = RESULTS_DIR / "detalle_completo.json"

# Separador visual para la salida por consola.
SEP = "═" * 64


# *** Experimento A y B: A* y Greedy sin LLM ***

def experimento_sin_llm(grafo: GrafoCursos, instancias: list) -> list:
    """
    Ejecuta A* y Greedy sobre todas las instancias sin intervención del LLM.

    Registra métricas de búsqueda, validez de la trayectoria y coste total.
    """
    print(f"\n{SEP}")
    print(f"  EXPERIMENTO A+B — A* y Greedy sin LLM")
    print(f"  {len(instancias)} instancias × 2 algoritmos = "
          f"{len(instancias) * 2} ejecuciones")
    print(SEP)

    filas = []

    for inst in instancias:
        print(f"\n  [{inst.id}] {inst.descripcion}")

        for algoritmo_fn, criterio, nombre_alg in [
            (astar, "cursos", "A* (cursos)"),
            (greedy, None, "Greedy"),
        ]:
            try:
                if algoritmo_fn is astar:
                    r = astar(
                        grafo,
                        inst.habilidades_iniciales,
                        inst.perfil_objetivo,
                        inst.id,
                        criterio=criterio,
                    )
                else:
                    r = greedy(
                        grafo,
                        inst.habilidades_iniciales,
                        inst.perfil_objetivo,
                        inst.id,
                    )
            except Exception as e:
                print(f"    ✗ {nombre_alg}: error — {e}")
                filas.append({
                    "instancia_id": inst.id,
                    "descripcion": inst.descripcion,
                    "perfil_objetivo": inst.perfil_objetivo,
                    "habs_iniciales": len(inst.habilidades_iniciales),
                    "algoritmo": nombre_alg,
                    "exito": False,
                    "num_cursos": 0,
                    "costo_total_semanas": 0,
                    "nodos_expandidos": 0,
                    "tiempo_segundos": 0.0,
                    "trayectoria_valida": False,
                    "trayectoria_ids": "",
                })
                continue

            # La trayectoria solo puede validarse si el algoritmo encontró solución.
            valido = False
            if r.exito:
                valido, _ = validar_trayectoria(
                    grafo,
                    r.trayectoria,
                    inst.habilidades_iniciales,
                    inst.perfil_objetivo,
                )

            fila = {
                "instancia_id": inst.id,
                "descripcion": inst.descripcion,
                "perfil_objetivo": inst.perfil_objetivo,
                "habs_iniciales": len(inst.habilidades_iniciales),
                "algoritmo": nombre_alg,
                "exito": r.exito,
                "num_cursos": r.num_cursos,
                "costo_total_semanas": r.costo_total_semanas,
                "nodos_expandidos": r.nodos_expandidos,
                "tiempo_segundos": round(r.tiempo_segundos, 6),
                "trayectoria_valida": valido,
                "trayectoria_ids": "|".join(c.id for c in r.trayectoria),
            }
            filas.append(fila)

            status = "✓" if r.exito else "✗"
            print(f"    {status} {nombre_alg:<15} "
                  f"cursos={r.num_cursos:>3}  "
                  f"semanas={r.costo_total_semanas:>3}  "
                  f"nodos={r.nodos_expandidos:>7}  "
                  f"t={r.tiempo_segundos:.4f}s")

    if filas:
        with open(CSV_SIN_LLM, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=filas[0].keys())
            writer.writeheader()
            writer.writerows(filas)
        print(f"\n  ✓ Resultados guardados en: {CSV_SIN_LLM}")

    return filas


# *** Experimento C: A* con evaluación LLM ***

def experimento_con_llm(grafo: GrafoCursos, instancias: list) -> list:
    """
    Ejecuta A* y luego evalúa la trayectoria con el componente LLM.

    Esta fase añade una capa semántica de análisis sobre la solución obtenida.
    """
    print(f"\n{SEP}")
    print(f"  EXPERIMENTO C — A* con evaluación LLM")
    print(f"  {len(instancias)} instancias | ~2-5s por instancia")
    print(SEP)

    try:
        from llm_integration import evaluar_trayectoria_con_fallback
    except ImportError as e:
        print(f"  ✗ No se pudo importar llm_integration: {e}")
        return []

    filas = []

    for inst in instancias:
        print(f"\n  [{inst.id}] {inst.descripcion}")

        try:
            r = astar(
                grafo,
                inst.habilidades_iniciales,
                inst.perfil_objetivo,
                inst.id,
                criterio="cursos",
            )
        except Exception as e:
            print(f"    ✗ A* falló: {e}")
            continue

        if not r.exito:
            print(f"    ✗ A* no encontró trayectoria.")
            continue

        valido, _ = validar_trayectoria(
            grafo,
            r.trayectoria,
            inst.habilidades_iniciales,
            inst.perfil_objetivo,
        )

        objetivo_texto = (
            inst.objetivo_texto.strip()
            or f"Alcanzar el perfil: {inst.perfil_objetivo}"
        )

        # Valores del LLM: se rellenan solo si la evaluación tiene éxito.
        puntuacion_llm = None
        nivel_llm = None
        tiempo_llm = None
        resumen_llm = None
        modo_llm = None

        try:
            ev = evaluar_trayectoria_con_fallback(
                objetivo_texto,
                inst.perfil_objetivo,
                r.trayectoria,
                inst.habilidades_iniciales,
            )
            puntuacion_llm = ev.get("puntuacion")
            nivel_llm = ev.get("nivel_calidad")
            tiempo_llm = ev.get("tiempo_segundos")
            resumen_llm = ev.get("resumen", "")
            modo_llm = ev.get("modo", "llm_real")
            print(f"    ✓ cursos={r.num_cursos}  semanas={r.costo_total_semanas}  "
                  f"LLM={puntuacion_llm}/10 ({nivel_llm}) [{modo_llm}]")
        except Exception as e:
            print(f"    ⚠ cursos={r.num_cursos}  semanas={r.costo_total_semanas}  "
                  f"LLM=error ({e})")

        filas.append({
            "instancia_id": inst.id,
            "descripcion": inst.descripcion,
            "perfil_objetivo": inst.perfil_objetivo,
            "habs_iniciales": len(inst.habilidades_iniciales),
            "algoritmo": "A* (cursos) + LLM",
            "exito": r.exito,
            "num_cursos": r.num_cursos,
            "costo_total_semanas": r.costo_total_semanas,
            "nodos_expandidos": r.nodos_expandidos,
            "tiempo_busqueda_s": round(r.tiempo_segundos, 6),
            "trayectoria_valida": valido,
            "puntuacion_llm": puntuacion_llm,
            "nivel_calidad_llm": nivel_llm,
            "tiempo_llm_s": tiempo_llm,
            "resumen_llm": resumen_llm,
            "trayectoria_ids": "|".join(c.id for c in r.trayectoria),
        })

    if filas:
        with open(CSV_CON_LLM, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=filas[0].keys())
            writer.writeheader()
            writer.writerows(filas)
        print(f"\n  ✓ Resultados guardados en: {CSV_CON_LLM}")

    return filas


# *** Análisis comparativo ***

def analisis_comparativo(filas_sin_llm: list, filas_con_llm: list) -> None:
    """
    Imprime un análisis comparativo entre A* y Greedy, y resume las evaluaciones LLM.
    """
    print(f"\n{SEP}")
    print(f"  ANÁLISIS COMPARATIVO")
    print(SEP)

    # Índices por instancia para comparar ambas estrategias sobre el mismo caso.
    astar_por_inst = {
        r["instancia_id"]: r for r in filas_sin_llm
        if "A*" in r["algoritmo"] and r["exito"]
    }
    greedy_por_inst = {
        r["instancia_id"]: r for r in filas_sin_llm
        if "Greedy" in r["algoritmo"] and r["exito"]
    }
    ids_comunes = sorted(set(astar_por_inst) & set(greedy_por_inst))

    if ids_comunes:
        print(f"\n  1. A* vs Greedy — Calidad de solución")
        print(f"  {'Instancia':<12} {'A* Cursos':>10} {'Greedy C':>10} "
              f"{'A* Sem':>8} {'Greedy S':>9} {'Δ Sem':>7}")
        print(f"  {'─' * 60}")

        for iid in ids_comunes:
            r_a = astar_por_inst[iid]
            r_g = greedy_por_inst[iid]
            # Conversión explícita desde el CSV para operar numéricamente.
            delta = int(r_g["costo_total_semanas"]) - int(r_a["costo_total_semanas"])
            signo = "+" if delta > 0 else ""
            print(f"  {iid:<12} "
                  f"{int(r_a['num_cursos']):>10} {int(r_g['num_cursos']):>10} "
                  f"{int(r_a['costo_total_semanas']):>8} "
                  f"{int(r_g['costo_total_semanas']):>9} "
                  f"{signo}{delta:>6}")

        print(f"\n  2. A* vs Greedy — Eficiencia computacional")
        print(f"  {'Instancia':<12} {'A* Nodos':>10} {'Greedy N':>10} "
              f"{'A* t(s)':>10} {'Greedy t':>10} {'Ratio N':>8}")
        print(f"  {'─' * 62}")

        for iid in ids_comunes:
            r_a = astar_por_inst[iid]
            r_g = greedy_por_inst[iid]
            nodos_a = int(r_a["nodos_expandidos"])
            nodos_g = int(r_g["nodos_expandidos"])
            tiempo_a = float(r_a["tiempo_segundos"])
            tiempo_g = float(r_g["tiempo_segundos"])
            ratio = nodos_a / max(nodos_g, 1)

            print(f"  {iid:<12} "
                  f"{nodos_a:>10} {nodos_g:>10} "
                  f"{tiempo_a:>10.4f} {tiempo_g:>10.4f} "
                  f"{ratio:>7.1f}x")

    # Resumen de puntuaciones LLM cuando existen resultados válidos.
    if filas_con_llm:
        puntuaciones = [
            f["puntuacion_llm"] for f in filas_con_llm
            if f.get("puntuacion_llm") is not None
        ]
        if puntuaciones:
            print(f"\n  3. Evaluaciones LLM")
            print(f"  {'Instancia':<12} {'Perfil':<22} "
                  f"{'Puntuación':>12} {'Calidad':<12}")
            print(f"  {'─' * 62}")

            for f in filas_con_llm:
                if f.get("puntuacion_llm") is not None:
                    print(f"  {f['instancia_id']:<12} {f['perfil_objetivo']:<22} "
                          f"{str(f['puntuacion_llm']) + '/10':>12} "
                          f"{str(f.get('nivel_calidad_llm', '?')):<12}")

            avg = sum(float(p) for p in puntuaciones) / len(puntuaciones)
            print(f"\n  Puntuación promedio LLM : {avg:.2f}/10")
            print(f"  Puntuación mínima       : {min(float(p) for p in puntuaciones)}/10")
            print(f"  Puntuación máxima       : {max(float(p) for p in puntuaciones)}/10")

    # Guardado del detalle completo en JSON.
    detalle = {"sin_llm": filas_sin_llm, "con_llm": filas_con_llm}
    with open(JSON_DETALLE, "w", encoding="utf-8") as f:
        json.dump(detalle, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Detalle completo guardado en: {JSON_DETALLE}")


# **** Main ****

def main() -> None:
    """Punto de entrada del diseño experimental."""
    print(SEP)
    print("  DISEÑO EXPERIMENTAL — Career Path Planner")
    print("  3 configuraciones × N instancias")
    print(SEP)

    try:
        grafo = GrafoCursos()
    except FileNotFoundError as e:
        print(f"\n  ✗ {e}")
        sys.exit(1)

    instancias = grafo.cargar_instancias()

    print(f"\n  Grafo     : {len(grafo.cursos)} cursos | "
          f"{len(grafo.habilidades)} habilidades | "
          f"{len(grafo.perfiles)} perfiles")
    print(f"  Instancias: {len(instancias)}")

    try:
        filas_sin_llm = experimento_sin_llm(grafo, instancias)
        filas_con_llm = experimento_con_llm(grafo, instancias)
        if filas_sin_llm:
            analisis_comparativo(filas_sin_llm, filas_con_llm)
    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrumpido. Los resultados parciales han sido guardados.")
        sys.exit(0)

    print(f"\n{SEP}")
    print(f"  ✓ Experimentos completados.")
    print(f"  Archivos generados:")
    print(f"    · {CSV_SIN_LLM}")
    print(f"    · {CSV_CON_LLM}")
    print(f"    · {JSON_DETALLE}")
    print(SEP)


if __name__ == "__main__":
    main()