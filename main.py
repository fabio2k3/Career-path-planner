"""
main.py
-------
Pipeline principal del sistema Career Path Planner.

Uso:
    python src/main.py
    python src/main.py --objetivo "Quiero ser Data Scientist"
    python src/main.py --objetivo "Quiero ser ML Engineer" --algoritmo greedy
    python src/main.py --objetivo "..." --sin-llm

Modos:
    Interactivo (sin argumentos): solicita el objetivo por teclado.
    Automático  (con --objetivo) : ejecuta directamente con el texto dado.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from graph import GrafoCursos
from search import astar, greedy, validar_trayectoria
from llm_integration import pipeline_completo, parsear_objetivo, evaluar_trayectoria


SEP  = "═" * 62
SEP2 = "─" * 62


def imprimir_bienvenida():
    print(f"\n{SEP}")
    print("  CAREER PATH PLANNER")
    print("  Sistema de Planificación de Trayectoria Profesional")
    print("  Proyecto Final — Inteligencia Artificial y Simulación")
    print(SEP)


def imprimir_trayectoria(result, titulo="Trayectoria generada"):
    print(f"\n  {titulo}:")
    print(f"  {SEP2}")
    for i, c in enumerate(result.trayectoria, 1):
        print(f"    {i:>2}. [{c.nivel[:3].upper()}] {c.nombre} "
              f"({c.duracion_semanas} semanas)")
    print(f"  {SEP2}")
    print(f"  Total: {result.num_cursos} cursos | "
          f"{result.costo_total_semanas} semanas | "
          f"{result.nodos_expandidos} nodos expandidos | "
          f"{result.tiempo_segundos:.3f}s")


def imprimir_evaluacion(evaluacion):
    puntuacion = evaluacion.get("puntuacion", "?")
    calidad    = evaluacion.get("nivel_calidad", "?").upper()
    if isinstance(puntuacion, (int, float)):
        barra = "█" * int(puntuacion) + "░" * (10 - int(puntuacion))
        print(f"\n  Evaluación LLM: {puntuacion}/10  [{barra}]  {calidad}")
    else:
        print(f"\n  Evaluación LLM: {puntuacion}/10  {calidad}")

    print(f"\n  Fortalezas:")
    for f in evaluacion.get("fortalezas", []):
        print(f"    ✓ {f}")
    print(f"\n  Debilidades:")
    for d in evaluacion.get("debilidades", []):
        print(f"    ✗ {d}")
    print(f"\n  Sugerencias:")
    for s in evaluacion.get("sugerencias", []):
        print(f"    → {s}")
    print(f"\n  Resumen: \"{evaluacion.get('resumen', '?')}\"")


def modo_con_llm(objetivo_texto: str, algoritmo: str, grafo: GrafoCursos):
    """Pipeline completo con LLM: parseo → búsqueda → evaluación."""
    print(f"\n  Objetivo: \"{objetivo_texto}\"")
    print(f"  Algoritmo: {algoritmo.upper()} | Modo: con LLM\n")

    algoritmo_fn = astar if algoritmo == "astar" else greedy

    resultado = pipeline_completo(
        objetivo_texto=objetivo_texto,
        grafo=grafo,
        algoritmo_fn=algoritmo_fn,
        habilidades_iniciales=frozenset(),
        instancia_id="main_pipeline",
    )

    if not resultado["exito_total"]:
        print("\n  ✗ No se pudo completar el pipeline.")
        return

    # Mostrar trayectoria
    busqueda = resultado["paso2_busqueda"]
    print(f"\n{SEP2}")
    print(f"  TRAYECTORIA ÓPTIMA — {busqueda['num_cursos']} cursos, "
          f"{busqueda['costo_total_semanas']} semanas")
    print(SEP2)
    for i, nombre in enumerate(busqueda["trayectoria_nombres"], 1):
        print(f"    {i:>2}. {nombre}")

    # Mostrar evaluación
    imprimir_evaluacion(resultado["paso3_evaluacion"])


def modo_sin_llm(objetivo_texto: str, algoritmo: str, grafo: GrafoCursos):
    """Búsqueda directa sin LLM: el usuario elige perfil del catálogo."""
    print(f"\n  Objetivo: \"{objetivo_texto}\"")
    print(f"  Algoritmo: {algoritmo.upper()} | Modo: sin LLM\n")

    # Mostrar perfiles disponibles
    print("  Perfiles disponibles:")
    perfiles = list(grafo.perfiles.keys())
    for i, pid in enumerate(perfiles, 1):
        perfil = grafo.perfiles[pid]
        print(f"    {i}. {perfil.nombre} ({pid})")

    print()
    try:
        opcion = int(input("  Selecciona un perfil (número): ").strip())
        perfil_id = perfiles[opcion - 1]
    except (ValueError, IndexError):
        print("  ✗ Opción inválida.")
        return

    algoritmo_fn = astar if algoritmo == "astar" else greedy
    result = algoritmo_fn(grafo, frozenset(), perfil_id, "main",
                          criterio="cursos")

    if not result.exito:
        print("  ✗ No se encontró trayectoria.")
        return

    imprimir_trayectoria(result, f"Trayectoria hacia {perfil_id}")

    # Validar
    ok, msg = validar_trayectoria(grafo, result.trayectoria,
                                   frozenset(), perfil_id)
    print(f"\n  Validación: {'✓ Trayectoria válida' if ok else f'✗ {msg}'}")


def main():
    parser = argparse.ArgumentParser(
        description="Career Path Planner — Planificación de trayectoria profesional"
    )
    parser.add_argument("--objetivo", type=str, default=None,
                        help="Objetivo profesional en lenguaje natural")
    parser.add_argument("--algoritmo", type=str, default="astar",
                        choices=["astar", "greedy"],
                        help="Algoritmo de búsqueda (default: astar)")
    parser.add_argument("--sin-llm", action="store_true",
                        help="Ejecutar sin usar el LLM")
    args = parser.parse_args()

    imprimir_bienvenida()

    grafo = GrafoCursos()
    print(f"\n  Sistema cargado: {len(grafo.cursos)} cursos | "
          f"{len(grafo.habilidades)} habilidades | "
          f"{len(grafo.perfiles)} perfiles\n")

    # Obtener objetivo
    if args.objetivo:
        objetivo_texto = args.objetivo
    else:
        print("  Escribe tu objetivo profesional (o 'salir' para terminar):")
        objetivo_texto = input("  > ").strip()
        if objetivo_texto.lower() in ("salir", "exit", "q"):
            print("  Hasta luego.")
            return

    if not objetivo_texto:
        print("  ✗ Objetivo vacío.")
        return

    # Ejecutar pipeline
    if args.sin_llm:
        modo_sin_llm(objetivo_texto, args.algoritmo, grafo)
    else:
        try:
            modo_con_llm(objetivo_texto, args.algoritmo, grafo)
        except EnvironmentError as e:
            print(f"\n  ✗ {e}")
            print("\n  Ejecuta con --sin-llm para usar el sistema sin API key.")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()