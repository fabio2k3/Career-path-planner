"""
test_multicriterio.py

Prueba las funcionalidades avanzadas de search.py:
  1. Comparación de los 3 criterios de A* + Greedy en inst_01 e inst_05.
  2. K-mejores trayectorias (k=3) en inst_03.
  3. Rendimiento en inst_08 (caso más difícil) con timeout de seguridad.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from graph import GrafoCursos
from search import astar, greedy, k_mejores, validar_trayectoria


SEP  = "═" * 62
SEP2 = "─" * 62


# ── Helpers ───────────────────────────────────────────────────────────────────

def seccion(titulo: str) -> None:
    print(f"\n{SEP}")
    print(f"  {titulo}")
    print(SEP)


def _get_instancia(instancias: list, inst_id: str, fallback_idx: int = 0):
    """Devuelve la instancia por ID o la del índice fallback si no existe."""
    match = next((i for i in instancias if i.id == inst_id), None)
    if match is None:
        print(f"  ⚠ Instancia '{inst_id}' no encontrada. "
              f"Usando instancia en posición {fallback_idx}.")
        return instancias[fallback_idx] if fallback_idx < len(instancias) else None
    return match


# ── Test 1: Comparación de criterios ─────────────────────────────────────────

def comparar_criterios(grafo: GrafoCursos, instancia) -> None:
    seccion(f"COMPARACIÓN DE CRITERIOS — {instancia.id}")
    print(f"  {instancia.descripcion}")
    print(f"  Objetivo: {instancia.objetivo_texto}\n")

    resultados = {}

    for criterio in ["semanas", "cursos", "balance"]:
        r = astar(
            grafo,
            instancia.habilidades_iniciales,
            instancia.perfil_objetivo,
            instancia.id,
            criterio=criterio,
            alpha=0.5,
        )
        resultados[f"A* ({criterio})"] = r

    r_gre = greedy(
        grafo,
        instancia.habilidades_iniciales,
        instancia.perfil_objetivo,
        instancia.id,
    )
    resultados["Greedy"] = r_gre

    # Tabla comparativa
    print(f"  {'Algoritmo':<22} {'Cursos':>7} {'Semanas':>8} "
          f"{'Nodos':>8} {'Tiempo(s)':>10}")
    print(f"  {SEP2}")
    for nombre, r in resultados.items():
        if r.exito:
            print(f"  {nombre:<22} {r.num_cursos:>7} "
                  f"{r.costo_total_semanas:>8} "
                  f"{r.nodos_expandidos:>8} "
                  f"{r.tiempo_segundos:>10.4f}")
        else:
            print(f"  {nombre:<22} {'—':>7} {'—':>8} {'—':>8} {'—':>10}")

    # Análisis diferencias
    r_sem = resultados["A* (semanas)"]
    r_cur = resultados["A* (cursos)"]
    r_bal = resultados["A* (balance)"]

    print(f"\n  Análisis:")

    if r_sem.exito and r_cur.exito:
        d_sem = r_sem.costo_total_semanas - r_cur.costo_total_semanas
        d_cur = r_cur.num_cursos - r_sem.num_cursos
        if d_sem != 0 or d_cur != 0:
            print(f"  → A*(semanas) vs A*(cursos): "
                  f"{abs(d_sem)} sem {'menos' if d_sem < 0 else 'más'}, "
                  f"{abs(d_cur)} curso(s) {'más' if d_cur > 0 else 'menos'}.")
        else:
            print(f"  → A*(semanas) y A*(cursos) encuentran la misma trayectoria.")

    if r_sem.exito and r_gre.exito:
        diff = r_gre.costo_total_semanas - r_sem.costo_total_semanas
        if diff > 0:
            print(f"  → A*(semanas) es {diff} semana(s) más corto que Greedy. "
                  f"✓ Optimalidad confirmada.")
        elif diff == 0:
            print(f"  → A*(semanas) y Greedy coinciden en costo total.")
        else:
            print(f"  → Greedy encontró {-diff} semana(s) menos (caso inusual).")

    # Validación
    print(f"\n  Validación:")
    for nombre, r in resultados.items():
        if r.exito:
            ok, msg = validar_trayectoria(
                grafo, r.trayectoria,
                instancia.habilidades_iniciales,
                instancia.perfil_objetivo,
            )
            print(f"    {nombre:<22}: {'✓ VÁLIDA' if ok else f'✗ {msg}'}")


# ── Test 2: K mejores trayectorias ────────────────────────────────────────────

def probar_k_mejores(
    grafo: GrafoCursos, instancia, k: int = 3, criterio: str = "cursos"
) -> None:
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
        print("  ✗ No se encontraron trayectorias.")
        return

    print(f"  Se encontraron {len(soluciones)} trayectoria(s):\n")

    for idx, sol in enumerate(soluciones, 1):
        print(f"  ── Trayectoria #{idx} "
              f"({sol.num_cursos} cursos, {sol.costo_total_semanas} semanas) ──")
        for i, c in enumerate(sol.trayectoria, 1):
            print(f"    {i:>2}. [{c.nivel[:3].upper()}] "
                  f"{c.nombre} ({c.duracion_semanas}s)")
        ok, msg = validar_trayectoria(
            grafo, sol.trayectoria,
            instancia.habilidades_iniciales,
            instancia.perfil_objetivo,
        )
        print(f"       Validación: {'✓ VÁLIDA' if ok else f'✗ {msg}'}\n")

    if len(soluciones) > 1:
        ids_listas = [frozenset(c.id for c in s.trayectoria) for s in soluciones]
        comunes    = ids_listas[0]
        for ids in ids_listas[1:]:
            comunes &= ids
        print(f"  Cursos comunes en todas las trayectorias: {len(comunes)}")
        if comunes:
            nombres = [grafo.cursos[cid].nombre for cid in sorted(comunes)]
            for n in nombres:
                print(f"    · {n}")


# ── Test 3: Rendimiento en instancia difícil ──────────────────────────────────

def probar_rendimiento(grafo: GrafoCursos, instancia) -> None:
    seccion(f"PRUEBA DE RENDIMIENTO — {instancia.id}")
    print(f"  {instancia.descripcion}")
    print(f"  Habilidades iniciales: {len(instancia.habilidades_iniciales)}\n")

    # Criterio "semanas" con max_nodos reducido como salvaguarda
    # (en datasets grandes puede ser costoso; se limita para reproducibilidad)
    MAX_NODOS_SEMANAS = 50_000

    for criterio, max_nodos in [
        ("semanas", MAX_NODOS_SEMANAS),
        ("cursos",  500_000),
        ("balance", 500_000),
    ]:
        limite_txt = f" (max_nodos={max_nodos:,})" if max_nodos < 500_000 else ""
        print(f"  Ejecutando A* criterio='{criterio}'{limite_txt}...")
        t0 = time.perf_counter()
        r  = astar(
            grafo,
            instancia.habilidades_iniciales,
            instancia.perfil_objetivo,
            instancia.id,
            criterio=criterio,
            max_nodos=max_nodos,
        )
        elapsed = time.perf_counter() - t0

        if r.exito:
            ok, _ = validar_trayectoria(
                grafo, r.trayectoria,
                instancia.habilidades_iniciales,
                instancia.perfil_objetivo,
            )
            print(f"  ✓ {r.num_cursos} cursos | {r.costo_total_semanas} semanas | "
                  f"{r.nodos_expandidos:,} nodos | {elapsed:.4f}s | "
                  f"válida={'✓' if ok else '✗'}")
            for i, c in enumerate(r.trayectoria, 1):
                print(f"    {i:>2}. {c.nombre} ({c.duracion_semanas}s)")
        else:
            if r.nodos_expandidos >= max_nodos:
                print(f"  ⚠ Límite de nodos alcanzado ({max_nodos:,}). "
                      f"Sin solución en {elapsed:.4f}s.")
            else:
                print(f"  ✗ Sin solución en {elapsed:.4f}s.")
        print()

    # Greedy como referencia de velocidad
    print(f"  Ejecutando Greedy (sin límite)...")
    r_gre = greedy(
        grafo,
        instancia.habilidades_iniciales,
        instancia.perfil_objetivo,
        instancia.id,
    )
    if r_gre.exito:
        ok, _ = validar_trayectoria(
            grafo, r_gre.trayectoria,
            instancia.habilidades_iniciales,
            instancia.perfil_objetivo,
        )
        print(f"  ✓ Greedy: {r_gre.num_cursos} cursos | "
              f"{r_gre.costo_total_semanas} semanas | "
              f"{r_gre.nodos_expandidos:,} nodos | "
              f"{r_gre.tiempo_segundos:.4f}s | "
              f"válida={'✓' if ok else '✗'}")
    else:
        print(f"  ✗ Greedy: sin solución.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP)
    print("  TEST MULTICRITERIO Y K-MEJORES — Career Path Planner")
    print(SEP)

    grafo      = GrafoCursos()
    instancias = grafo.cargar_instancias()

    print(f"\n  Grafo: {len(grafo.cursos)} cursos | "
          f"{len(grafo.habilidades)} habilidades | "
          f"{len(grafo.perfiles)} perfiles")

    # Test 1: comparación de criterios en 2 instancias
    for inst_id, fallback in [("inst_01", 0), ("inst_05", 4)]:
        inst = _get_instancia(instancias, inst_id, fallback)
        if inst:
            comparar_criterios(grafo, inst)

    # Test 2: k-mejores en inst_03
    inst = _get_instancia(instancias, "inst_03", 2)
    if inst:
        probar_k_mejores(grafo, inst, k=3, criterio="cursos")

    # Test 3: rendimiento en inst_08
    inst = _get_instancia(instancias, "inst_08", 7)
    if inst:
        probar_rendimiento(grafo, inst)

    print(f"\n{SEP}")
    print("  ✓ Todas las pruebas completadas.")
    print(SEP)


if __name__ == "__main__":
    main()