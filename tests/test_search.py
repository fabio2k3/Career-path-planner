"""
test_search.py

Prueba los algoritmos A* y Greedy sobre 3 instancias del dataset:
  - inst_01: desde cero → Data Scientist
  - inst_05: frontend   → Backend Developer
  - inst_07: DS         → ML Engineer

Para cada instancia:
  1. Corre A* y Greedy.
  2. Imprime la trayectoria encontrada.
  3. Valida que la trayectoria es correcta (prerrequisitos respetados + objetivo alcanzado).
  4. Compara ambos algoritmos.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from graph import GrafoCursos
from search import astar, greedy, validar_trayectoria


def correr_instancia(grafo, instancia):
    print(f"\n{'═'*56}")
    print(f"  INSTANCIA: {instancia.id}")
    print(f"  {instancia.descripcion}")
    print(f"  Objetivo : {instancia.objetivo_texto}")
    print(f"{'═'*56}")
    print(f"  Habilidades iniciales: {len(instancia.habilidades_iniciales)}")
    if instancia.habilidades_iniciales:
        for h in sorted(instancia.habilidades_iniciales):
            print(f"    · {h}")
    print(f"\n{grafo.resumen_estado(instancia.habilidades_iniciales, instancia.perfil_objetivo)}")

    resultados = {}

    for nombre, algoritmo in [("A*", astar), ("Greedy", greedy)]:
        resultado = algoritmo(
            grafo,
            instancia.habilidades_iniciales,
            instancia.perfil_objetivo,
            instancia.id,
        )
        resultado.imprimir()
        resultados[nombre] = resultado

        # Validación de la trayectoria
        if resultado.exito:
            valido, msg = validar_trayectoria(
                grafo,
                resultado.trayectoria,
                instancia.habilidades_iniciales,
                instancia.perfil_objetivo,
            )
            status = "✓ Trayectoria VÁLIDA" if valido else f"✗ INVÁLIDA: {msg}"
            print(f"  Validación: {status}\n")

    # Comparación
    r_astar  = resultados["A*"]
    r_greedy = resultados["Greedy"]

    if r_astar.exito and r_greedy.exito:
        print(f"  {'─'*54}")
        print(f"  COMPARACIÓN A* vs Greedy")
        print(f"  {'─'*54}")
        print(f"  {'Métrica':<35} {'A*':>8} {'Greedy':>8}")
        print(f"  {'─'*54}")
        print(f"  {'Cursos en trayectoria':<35} {len(r_astar.trayectoria):>8} {len(r_greedy.trayectoria):>8}")
        print(f"  {'Costo total (semanas)':<35} {r_astar.costo_total_semanas:>8} {r_greedy.costo_total_semanas:>8}")
        print(f"  {'Nodos expandidos':<35} {r_astar.nodos_expandidos:>8} {r_greedy.nodos_expandidos:>8}")
        print(f"  {'Tiempo (s)':<35} {r_astar.tiempo_segundos:>8.4f} {r_greedy.tiempo_segundos:>8.4f}")

        diff = r_greedy.costo_total_semanas - r_astar.costo_total_semanas
        if diff > 0:
            print(f"\n  → A* encontró una trayectoria {diff} semana(s) más corta que Greedy.")
        elif diff < 0:
            print(f"\n  → Greedy encontró una trayectoria {-diff} semana(s) más corta (caso inusual).")
        else:
            print(f"\n  → Ambos algoritmos encontraron trayectorias de igual costo.")
        print(f"  {'═'*56}\n")


def main():
    print("=" * 56)
    print("  TEST DE BÚSQUEDA — Career Path Planner")
    print("  Algoritmos: A* y Greedy")
    print("=" * 56)

    grafo = GrafoCursos()
    instancias = grafo.cargar_instancias()

    # Seleccionar 3 instancias representativas para el test del día 3
    ids_test = {"inst_01", "inst_05", "inst_07"}
    instancias_test = [i for i in instancias if i.id in ids_test]

    print(f"\n  Grafo cargado: {len(grafo.cursos)} cursos, "
          f"{len(grafo.habilidades)} habilidades, "
          f"{len(grafo.perfiles)} perfiles.")
    print(f"  Ejecutando {len(instancias_test)} instancias de prueba...\n")

    for instancia in instancias_test:
        correr_instancia(grafo, instancia)

    print("\n  ✓ Todas las pruebas completadas.")


if __name__ == "__main__":
    main()