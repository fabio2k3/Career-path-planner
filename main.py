"""
main.py

Pipeline completo interactivo del sistema Career Path Planner.

Modos de uso:
  1. Interactivo : el usuario escribe su objetivo en lenguaje natural
  2. Demo        : ejecuta casos predefinidos sin necesidad de input

Flujo completo:
  Objetivo NL → LLM Parseador → A* → LLM Evaluador → Resultado
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from graph import GrafoCursos
from search import astar, greedy, validar_trayectoria
from llm_integration import pipeline_completo, parsear_objetivo, evaluar_trayectoria

SEP  = "═" * 62
SEP2 = "─" * 62


def imprimir_resultado(resultado: dict):
    """Imprime el resultado completo del pipeline de forma legible."""
    print(f"\n{SEP}")
    print(f"  RESULTADO DEL SISTEMA")
    print(SEP)

    # Paso 1: Parseo
    parseo = resultado.get("paso1_parseo", {})
    if parseo:
        print(f"\n  [PARSEO LLM]")
        print(f"  Perfil detectado  : {parseo.get('perfil_detectado', '?')}")
        print(f"  Confianza         : {parseo.get('confianza', '?')}")
        print(f"  Habilidades ({len(parseo.get('habilidades_validas', []))}) : "
              f"{', '.join(parseo.get('habilidades_validas', []))}")

    # Paso 2: Trayectoria
    busqueda = resultado.get("paso2_busqueda", {})
    if busqueda and busqueda.get("exito"):
        print(f"\n  [TRAYECTORIA GENERADA]")
        print(f"  Cursos  : {busqueda['num_cursos']}")
        print(f"  Semanas : {busqueda['costo_total_semanas']}")
        print(f"  Orden:")
        for i, nombre in enumerate(busqueda.get("trayectoria_nombres", []), 1):
            print(f"    {i:>2}. {nombre}")

    # Paso 3: Evaluación
    evaluacion = resultado.get("paso3_evaluacion", {})
    if evaluacion:
        puntuacion = evaluacion.get("puntuacion", "?")
        calidad    = evaluacion.get("nivel_calidad", "?").upper()
        barra = "█" * int(puntuacion) + "░" * (10 - int(puntuacion)) \
            if isinstance(puntuacion, int) else ""
        print(f"\n  [EVALUACIÓN LLM]")
        print(f"  Puntuación : {puntuacion}/10  [{barra}]  {calidad}")
        print(f"\n  Fortalezas:")
        for f in evaluacion.get("fortalezas", []):
            print(f"    ✓ {f}")
        print(f"\n  Debilidades:")
        for d in evaluacion.get("debilidades", []):
            print(f"    ✗ {d}")
        print(f"\n  Sugerencias:")
        for s in evaluacion.get("sugerencias", []):
            print(f"    → {s}")
        print(f"\n  Resumen:")
        print(f"    \"{evaluacion.get('resumen', '')}\"")

    print(f"\n{SEP}")


def modo_interactivo(grafo: GrafoCursos):
    """Modo interactivo: el usuario escribe su objetivo."""
    print(f"\n{SEP}")
    print(f"  CAREER PATH PLANNER — Modo Interactivo")
    print(SEP)
    print(f"\n  Describe tu objetivo profesional en lenguaje natural.")
    print(f"  Ejemplos:")
    print(f"    - Quiero ser Data Scientist trabajando con Python y ML")
    print(f"    - Me interesa el backend y construir APIs escalables")
    print(f"    - Quiero llevar modelos de IA a producción como ML Engineer")
    print(f"\n  (escribe 'salir' para terminar)\n")

    while True:
        try:
            objetivo = input("  Tu objetivo: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Saliendo...")
            break

        if objetivo.lower() in ("salir", "exit", "quit", ""):
            print("  ¡Hasta luego!")
            break

        print()
        resultado = pipeline_completo(
            objetivo_texto=objetivo,
            grafo=grafo,
            algoritmo_fn=astar,
            habilidades_iniciales=frozenset(),
            instancia_id="interactivo",
        )
        imprimir_resultado(resultado)


def modo_demo(grafo: GrafoCursos):
    """Modo demo: ejecuta 3 casos predefinidos."""
    print(f"\n{SEP}")
    print(f"  CAREER PATH PLANNER — Modo Demo")
    print(SEP)

    casos = [
        {
            "objetivo": "Quiero convertirme en Data Scientist analizando datos con Python",
            "habilidades_iniciales": frozenset(),
            "id": "demo_01",
        },
        {
            "objetivo": "Soy desarrollador frontend y quiero pasarme al backend con Python",
            "habilidades_iniciales": frozenset(["html_css", "javascript", "logica_programacion"]),
            "id": "demo_02",
        },
        {
            "objetivo": "Quiero llevar modelos de machine learning a producción como ML Engineer",
            "habilidades_iniciales": frozenset(["python_avanzado", "machine_learning"]),
            "id": "demo_03",
        },
    ]

    for i, caso in enumerate(casos, 1):
        print(f"\n  {'─'*60}")
        print(f"  Demo {i}/3 — {caso['objetivo']}")
        if caso["habilidades_iniciales"]:
            print(f"  Habilidades iniciales: {sorted(caso['habilidades_iniciales'])}")
        print(f"  {'─'*60}")

        resultado = pipeline_completo(
            objetivo_texto=caso["objetivo"],
            grafo=grafo,
            algoritmo_fn=astar,
            habilidades_iniciales=caso["habilidades_iniciales"],
            instancia_id=caso["id"],
        )
        imprimir_resultado(resultado)


def main():
    print(SEP)
    print(f"  CAREER PATH PLANNER")
    print(f"  Sistema de Planificación de Trayectorias Profesionales")
    print(SEP)

    grafo = GrafoCursos()
    print(f"\n  Catálogo cargado: {len(grafo.cursos)} cursos | "
          f"{len(grafo.habilidades)} habilidades | "
          f"{len(grafo.perfiles)} perfiles")

    # Determinar modo según argumentos
    args = sys.argv[1:]
    if "--demo" in args:
        modo_demo(grafo)
    else:
        modo_interactivo(grafo)


if __name__ == "__main__":
    main()