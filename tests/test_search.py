"""
test_search.py

Prueba los algoritmos A* y Greedy sobre instancias del dataset.

Para cada instancia:
1. Corre A* (criterio=semanas) y A* (criterio=cursos).
2. Corre Greedy.
3. Imprime la trayectoria encontrada.
4. Valida que la trayectoria es correcta.
5. Compara los tres algoritmos.
"""

import sys
from pathlib import Path

# Se agrega la carpeta src al path para poder importar los módulos del proyecto.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from graph import GrafoCursos
from search import astar, greedy, validar_trayectoria

# Separadores visuales para organizar la salida por consola.
SEP = "═" * 62
SEP2 = "─" * 62


# *** Helpers *********

def correr_instancia(grafo: GrafoCursos, instancia) -> None:
    """
    Ejecuta A*, Greedy y validaciones sobre una sola instancia del dataset.

    La función imprime:
    - estado inicial,
    - resultados de cada algoritmo,
    - validación de trayectorias,
    - comparación de métricas.
    """
    print(f"\n{SEP}")
    print(f"  INSTANCIA : {instancia.id}")
    print(f"  {instancia.descripcion}")
    print(f"  Objetivo  : {instancia.objetivo_texto}")
    print(SEP)

    print(f"  Habilidades iniciales: {len(instancia.habilidades_iniciales)}")
    if instancia.habilidades_iniciales:
        for h in sorted(instancia.habilidades_iniciales):
            print(f"    · {h}")

    print(f"\n{grafo.resumen_estado(instancia.habilidades_iniciales, instancia.perfil_objetivo)}")

    resultados = {}

    # A* con criterio de semanas: prioriza minimizar duración total.
    r_sem = astar(
        grafo,
        instancia.habilidades_iniciales,
        instancia.perfil_objetivo,
        instancia.id,
        criterio="semanas",
    )
    r_sem.imprimir()
    resultados["A* (semanas)"] = r_sem

    # A* con criterio de cursos: prioriza minimizar el número de cursos.
    r_cur = astar(
        grafo,
        instancia.habilidades_iniciales,
        instancia.perfil_objetivo,
        instancia.id,
        criterio="cursos",
    )
    r_cur.imprimir()
    resultados["A* (cursos)"] = r_cur

    # Greedy: usa solo la heurística como guía.
    r_gre = greedy(
        grafo,
        instancia.habilidades_iniciales,
        instancia.perfil_objetivo,
        instancia.id,
    )
    r_gre.imprimir()
    resultados["Greedy"] = r_gre

    # Validación formal de cada trayectoria encontrada.
    print(f"\n  Validación de trayectorias:")
    for nombre, r in resultados.items():
        if r.exito:
            ok, msg = validar_trayectoria(
                grafo, r.trayectoria,
                instancia.habilidades_iniciales,
                instancia.perfil_objetivo,
            )
            status = "✓ VÁLIDA" if ok else f"✗ INVÁLIDA: {msg}"
        else:
            status = "— Sin solución"
        print(f"    {nombre:<18}: {status}")

    # Tabla comparativa de métricas principales.
    print(f"\n  Comparativa:")
    print(f"  {'Algoritmo':<18} {'Cursos':>7} {'Semanas':>8} "
          f"{'Nodos':>8} {'Tiempo(s)':>10}")
    print(f"  {SEP2}")
    for nombre, r in resultados.items():
        if r.exito:
            print(f"  {nombre:<18} {r.num_cursos:>7} "
                  f"{r.costo_total_semanas:>8} "
                  f"{r.nodos_expandidos:>8} "
                  f"{r.tiempo_segundos:>10.4f}")
        else:
            print(f"  {nombre:<18} {'—':>7} {'—':>8} {'—':>8} {'—':>10}")

    # Comparación A*(semanas) vs Greedy.
    if r_sem.exito and r_gre.exito:
        diff = r_gre.costo_total_semanas - r_sem.costo_total_semanas
        if diff > 0:
            print(f"\n  → A*(semanas) es {diff} semana(s) más corto que Greedy. "
                  f"✓ Ventaja de optimalidad confirmada.")
        elif diff < 0:
            print(f"\n  → Greedy encontró {-diff} semana(s) menos (caso inusual).")
        else:
            print(f"\n  → A*(semanas) y Greedy coinciden en costo total.")

    # Comparación A*(semanas) vs A*(cursos).
    if r_sem.exito and r_cur.exito:
        d_sem = r_sem.costo_total_semanas - r_cur.costo_total_semanas
        d_cur = r_cur.num_cursos - r_sem.num_cursos
        if d_sem != 0 or d_cur != 0:
            print(f"  → A*(semanas) vs A*(cursos): "
                  f"{abs(d_sem)} sem {'menos' if d_sem < 0 else 'más'}, "
                  f"{abs(d_cur)} curso(s) {'más' if d_cur > 0 else 'menos'}.")
        else:
            print(f"  → A*(semanas) y A*(cursos) encuentran la misma trayectoria.")


def test_sin_solucion(grafo: GrafoCursos) -> None:
    """
    Verifica que A* y Greedy manejan correctamente un perfil inexistente.

    La prueba espera que los algoritmos fallen de forma controlada:
    - devolviendo exito=False, o
    - lanzando ValueError con mensaje explicativo.
    """
    print(f"\n{SEP}")
    print(f"  TEST CASO SIN SOLUCIÓN — perfil inexistente")
    print(SEP)

    perfil_falso = "perfil_que_no_existe_xyz"
    habs = frozenset()

    for nombre, fn in [
        ("A*", lambda: astar(grafo, habs, perfil_falso, "test")),
        ("Greedy", lambda: greedy(grafo, habs, perfil_falso, "test")),
    ]:
        try:
            r = fn()
            status = "✗ INCORRECTO: debería haber fallado" if r.exito else "✓ exito=False correctamente"
        except ValueError as e:
            status = f"✓ ValueError capturado: {str(e)[:60]}"
        except Exception as e:
            status = f"⚠ Excepción inesperada: {e}"
        print(f"  {nombre:<8}: {status}")


# *** Main ***

def main() -> None:
    """
    Punto de entrada del script de pruebas de búsqueda.
    """
    print(SEP)
    print("  TEST DE BÚSQUEDA — Career Path Planner")
    print("  Algoritmos: A* (semanas), A* (cursos), Greedy")
    print(SEP)

    grafo = GrafoCursos()
    instancias = grafo.cargar_instancias()

    print(f"\n  Grafo cargado: {len(grafo.cursos)} cursos | "
          f"{len(grafo.habilidades)} habilidades | "
          f"{len(grafo.perfiles)} perfiles")

    # Selección de instancias representativas para las pruebas.
    ids_test = {"inst_01", "inst_05", "inst_07"}
    instancias_test = [i for i in instancias if i.id in ids_test]

    if not instancias_test:
        print(f"\n  ⚠ No se encontraron las instancias {ids_test}.")
        print(f"  Usando las 3 primeras disponibles.")
        instancias_test = instancias[:3]

    print(f"  Ejecutando {len(instancias_test)} instancia(s) de prueba...\n")

    for instancia in instancias_test:
        correr_instancia(grafo, instancia)

    # Prueba adicional para casos sin solución.
    test_sin_solucion(grafo)

    print(f"\n{SEP}")
    print("  ✓ Todas las pruebas completadas.")
    print(SEP)


if __name__ == "__main__":
    main()