"""
test_multicreterio.py

Testea las nuevas funcionalidades de search:
  1. A* con criterio "cursos"   (minimiza número de cursos)
  2. A* con criterio "balance"  (α·semanas + β·cursos)
  3. Comparación de los 3 criterios de A* + Greedy en inst_01 e inst_05
  4. K-mejores trayectorias (k=3) en inst_03
  5. Prueba de rendimiento en inst_08 (desde cero → ML Engineer, el más difícil)
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from graph import GrafoCursos
from search import astar, greedy, k_mejores, validar_trayectoria


SEP  = "═" * 60
SEP2 = "─" * 60


def seccion(titulo: str):
    print(f"\n{SEP}")
    print(f"  {titulo}")
    print(SEP)


def comparar_criterios(grafo, instancia):
    """Compara los 3 criterios de A* + Greedy sobre una instancia."""
    seccion(f"COMPARACIÓN DE CRITERIOS — {instancia.id}")
    print(f"  {instancia.descripcion}")
    print(f"  Objetivo: {instancia.objetivo_texto}\n")

    resultados = {}

    # A* con los 3 criterios
    for criterio in ["semanas", "cursos", "balance"]:
        r = astar(grafo, instancia.habilidades_iniciales,
                  instancia.perfil_objetivo, instancia.id,
                  criterio=criterio, alpha=0.5)
        resultados[f"A* ({criterio})"] = r

    # Greedy
    r = greedy(grafo, instancia.habilidades_iniciales,
               instancia.perfil_objetivo, instancia.id)
    resultados["Greedy"] = r

    # Tabla comparativa
    print(f"  {'Algoritmo':<22} {'Cursos':>7} {'Semanas':>8} {'Nodos':>8} {'Tiempo':>10}")
    print(f"  {SEP2}")
    for nombre, r in resultados.items():
        if r.exito:
            print(f"  {nombre:<22} {r.num_cursos:>7} {r.costo_total_semanas:>8} "
                  f"{r.nodos_expandidos:>8} {r.tiempo_segundos:>9.4f}s")
        else:
            print(f"  {nombre:<22} {'—':>7} {'—':>8} {'—':>8} {'—':>10}")

    # Análisis de diferencias
    print(f"\n  Análisis:")
    r_sem = resultados["A* (semanas)"]
    r_cur = resultados["A* (cursos)"]
    r_bal = resultados["A* (balance)"]
    r_gre = resultados["Greedy"]

    if r_sem.exito and r_cur.exito:
        diff_sem = r_sem.costo_total_semanas - r_cur.costo_total_semanas
        diff_cur = r_cur.num_cursos - r_sem.num_cursos
        if diff_sem != 0 or diff_cur != 0:
            print(f"  → A*(semanas) vs A*(cursos):")
            print(f"    A*(semanas) usa {abs(diff_sem)} semana(s) "
                  f"{'menos' if diff_sem < 0 else 'más'} pero "
                  f"{abs(diff_cur)} curso(s) {'más' if diff_cur > 0 else 'menos'}.")
        else:
            print(f"  → A*(semanas) y A*(cursos) encuentran la misma trayectoria.")

    if r_sem.exito and r_gre.exito:
        if r_sem.costo_total_semanas != r_gre.costo_total_semanas:
            print(f"  → A*(semanas) encuentra {r_gre.costo_total_semanas - r_sem.costo_total_semanas}"
                  f" semana(s) menos que Greedy. ✓ Ventaja de optimalidad.")
        else:
            print(f"  → A*(semanas) y Greedy coinciden en costo (dataset bien estructurado).")

    # Validar todas las trayectorias
    print(f"\n  Validación de trayectorias:")
    for nombre, r in resultados.items():
        if r.exito:
            ok, msg = validar_trayectoria(grafo, r.trayectoria,
                                          instancia.habilidades_iniciales,
                                          instancia.perfil_objetivo)
            status = "✓ VÁLIDA" if ok else f"✗ INVÁLIDA: {msg}"
            print(f"    {nombre:<22}: {status}")


def probar_k_mejores(grafo, instancia, k=3, criterio="cursos"):
    """Genera y muestra k trayectorias alternativas."""
    seccion(f"K-MEJORES TRAYECTORIAS (k={k}) — {instancia.id}")
    print(f"  {instancia.descripcion}")
    print(f"  Objetivo: {instancia.objetivo_texto}\n")

    soluciones = k_mejores(
        grafo,
        instancia.habilidades_iniciales,
        instancia.perfil_objetivo,
        instancia.id,
        k=k,
        criterio=criterio,
    )

    if not soluciones:
        print("  No se encontraron trayectorias.")
        return

    print(f"  Se encontraron {len(soluciones)} trayectoria(s) alternativa(s):\n")
    for idx, sol in enumerate(soluciones, 1):
        print(f"  ── Trayectoria #{idx} "
              f"({sol.num_cursos} cursos, {sol.costo_total_semanas} semanas) ──")
        for i, c in enumerate(sol.trayectoria, 1):
            print(f"    {i:>2}. [{c.nivel[:3].upper()}] {c.nombre} ({c.duracion_semanas}s)")

        ok, msg = validar_trayectoria(grafo, sol.trayectoria,
                                      instancia.habilidades_iniciales,
                                      instancia.perfil_objetivo)
        print(f"       Validación: {'✓ VÁLIDA' if ok else f'✗ {msg}'}\n")

    # Mostrar diferencias entre trayectorias
    if len(soluciones) > 1:
        ids = [[c.id for c in s.trayectoria] for s in soluciones]
        comunes = set(ids[0])
        for traj in ids[1:]:
            comunes &= set(traj)
        print(f"  Cursos comunes en todas las trayectorias : {len(comunes)}")
        print(f"  Cursos comunes: {sorted(comunes)}")


def probar_rendimiento_inst08(grafo, instancia):
    """Prueba de rendimiento en la instancia más difícil."""
    seccion(f"PRUEBA DE RENDIMIENTO — {instancia.id} (CASO MÁS DIFÍCIL)")
    print(f"  {instancia.descripcion}")
    print(f"  Habilidades iniciales: {len(instancia.habilidades_iniciales)}\n")

    for criterio in ["cursos", "balance"]:  # 'semanas' omitido en inst_08 por alto costo computacional
        print(f"  Ejecutando A* criterio='{criterio}'...")
        t0 = time.perf_counter()
        r = astar(grafo, instancia.habilidades_iniciales,
                  instancia.perfil_objetivo, instancia.id,
                  criterio=criterio)
        elapsed = time.perf_counter() - t0

        if r.exito:
            print(f"  ✓ Solución: {r.num_cursos} cursos, "
                  f"{r.costo_total_semanas} semanas, "
                  f"{r.nodos_expandidos} nodos, {elapsed:.4f}s")
            ok, _ = validar_trayectoria(grafo, r.trayectoria,
                                         instancia.habilidades_iniciales,
                                         instancia.perfil_objetivo)
            print(f"    Trayectoria válida: {'✓' if ok else '✗'}")
            print(f"    Cursos:")
            for i, c in enumerate(r.trayectoria, 1):
                print(f"      {i:>2}. {c.nombre} ({c.duracion_semanas}s)")
        else:
            print(f"  ✗ Sin solución tras {elapsed:.4f}s")
        print()

    print(f"  Ejecutando Greedy...")
    r = greedy(grafo, instancia.habilidades_iniciales,
               instancia.perfil_objetivo, instancia.id)
    if r.exito:
        print(f"  ✓ Greedy: {r.num_cursos} cursos, "
              f"{r.costo_total_semanas} semanas, "
              f"{r.nodos_expandidos} nodos, {r.tiempo_segundos:.4f}s")
        ok, _ = validar_trayectoria(grafo, r.trayectoria,
                                     instancia.habilidades_iniciales,
                                     instancia.perfil_objetivo)
        print(f"    Trayectoria válida: {'✓' if ok else '✗'}")


def main():
    print(SEP)
    print("  TEST DÍA 4 — Multi-criterio y K-Mejores")
    print("  Career Path Planner")
    print(SEP)

    grafo = GrafoCursos()
    instancias = {i.id: i for i in grafo.cargar_instancias()}

    print(f"\n  Grafo: {len(grafo.cursos)} cursos | "
          f"{len(grafo.habilidades)} habilidades | "
          f"{len(grafo.perfiles)} perfiles")

    # 1. Comparación de criterios en inst_01 y inst_05
    comparar_criterios(grafo, instancias["inst_01"])
    comparar_criterios(grafo, instancias["inst_05"])

    # 2. K-mejores en inst_03 (Python básico → Data Scientist)
    probar_k_mejores(grafo, instancias["inst_03"], k=3, criterio="cursos")

    # 3. Rendimiento en inst_08 (el caso más difícil)
    probar_rendimiento_inst08(grafo, instancias["inst_08"])

    print(f"\n{SEP}")
    print("  ✓ Todas las pruebas del día 4 completadas.")
    print(SEP)


if __name__ == "__main__":
    main()