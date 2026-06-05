"""
main.py

Pipeline principal del sistema Career Path Planner.

Uso:
    python src/main.py
    python src/main.py --objetivo "Quiero ser Data Scientist"
    python src/main.py --objetivo "Quiero ser ML Engineer" --algoritmo greedy
    python src/main.py --objetivo "..." --sin-llm
    python src/main.py --objetivo "..." --habilidades python_basico,estadistica

Modos:
    Interactivo (sin --objetivo) : solicita el objetivo por teclado.
    Automático  (con --objetivo) : ejecuta directamente con el texto dado.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from graph import GrafoCursos
from search import astar, greedy, validar_trayectoria
from llm_integration import pipeline_completo


SEP  = "═" * 62
SEP2 = "─" * 62


def imprimir_bienvenida() -> None:
    print(f"\n{SEP}")
    print("  CAREER PATH PLANNER")
    print("  Sistema de Planificación de Trayectoria Profesional")
    print("  Proyecto Final — Inteligencia Artificial y Simulación")
    print(SEP)


def imprimir_trayectoria(result, titulo: str = "Trayectoria generada") -> None:
    print(f"\n  {titulo}:")
    print(f"  {SEP2}")
    for i, c in enumerate(result.trayectoria, 1):
        print(f"    {i:>2}. [{c.nivel[:3].upper()}] {c.nombre} ({c.duracion_semanas} semanas)")
    print(f"  {SEP2}")
    print(f"  Total : {result.num_cursos} cursos | "
          f"{result.costo_total_semanas} semanas | "
          f"{result.nodos_expandidos} nodos expandidos | "
          f"{result.tiempo_segundos:.3f}s")


def imprimir_evaluacion(evaluacion: dict) -> None:
    puntuacion = evaluacion.get("puntuacion", "?")
    calidad    = str(evaluacion.get("nivel_calidad", "?")).upper()
    modo       = evaluacion.get("modo", "llm_real")
    sufijo     = " [simulado]" if modo == "simulado" else ""

    if isinstance(puntuacion, (int, float)):
        barra = "█" * int(puntuacion) + "░" * (10 - int(puntuacion))
        print(f"\n  Evaluación LLM: {puntuacion}/10  [{barra}]  {calidad}{sufijo}")
    else:
        print(f"\n  Evaluación LLM: {puntuacion}/10  {calidad}{sufijo}")

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


def _parsear_habilidades(texto: str, grafo: GrafoCursos) -> frozenset:
    """Convierte 'hab1,hab2,hab3' en frozenset filtrando contra el catálogo."""
    if not texto:
        return frozenset()
    partes    = [h.strip() for h in texto.split(",") if h.strip()]
    validas   = [h for h in partes if h in grafo.habilidades]
    invalidas = [h for h in partes if h not in grafo.habilidades]
    if invalidas:
        print(f"  ⚠ Habilidades no encontradas en el catálogo: {invalidas}")
    return frozenset(validas)


def modo_con_llm(
    objetivo_texto: str,
    algoritmo: str,
    grafo: GrafoCursos,
    habilidades_iniciales: frozenset,
) -> None:
    """Pipeline completo: LN → LLM parseador → A*/Greedy → LLM evaluador."""
    print(f"\n  Objetivo  : \"{objetivo_texto}\"")
    print(f"  Algoritmo : {algoritmo.upper()} | Modo: con LLM")
    if habilidades_iniciales:
        print(f"  Habilidades iniciales: {sorted(habilidades_iniciales)}")

    algoritmo_fn = astar if algoritmo == "astar" else greedy

    resultado = pipeline_completo(
        objetivo_texto=objetivo_texto,
        grafo=grafo,
        algoritmo_fn=algoritmo_fn,
        habilidades_iniciales=habilidades_iniciales,
        instancia_id="main_pipeline",
    )

    if not resultado["exito_total"]:
        print("\n  ✗ No se pudo completar el pipeline.")
        return

    busqueda = resultado["paso2_busqueda"]
    print(f"\n{SEP2}")
    print(f"  TRAYECTORIA ÓPTIMA — {busqueda['num_cursos']} cursos, "
          f"{busqueda['costo_total_semanas']} semanas")
    print(SEP2)
    for i, nombre in enumerate(busqueda["trayectoria_nombres"], 1):
        print(f"    {i:>2}. {nombre}")

    imprimir_evaluacion(resultado["paso3_evaluacion"])


def modo_sin_llm(
    objetivo_texto: str,
    algoritmo: str,
    grafo: GrafoCursos,
    habilidades_iniciales: frozenset,
) -> None:
    """Búsqueda directa sin LLM: el usuario elige el perfil del catálogo."""
    print(f"\n  Objetivo  : \"{objetivo_texto}\"")
    print(f"  Algoritmo : {algoritmo.upper()} | Modo: sin LLM")

    print("\n  Perfiles disponibles:")
    perfiles = list(grafo.perfiles.keys())
    for i, pid in enumerate(perfiles, 1):
        print(f"    {i}. {grafo.perfiles[pid].nombre} ({pid})")

    print()
    try:
        opcion    = int(input("  Selecciona un perfil (número): ").strip())
        perfil_id = perfiles[opcion - 1]
    except (ValueError, IndexError):
        print("  ✗ Opción inválida.")
        return

    algoritmo_fn = astar if algoritmo == "astar" else greedy
    result       = algoritmo_fn(
        grafo, habilidades_iniciales, perfil_id,
        "main", criterio="cursos",
    )

    if not result.exito:
        print("  ✗ No se encontró trayectoria.")
        return

    imprimir_trayectoria(result, f"Trayectoria hacia {grafo.perfiles[perfil_id].nombre}")
    ok, msg = validar_trayectoria(
        grafo, result.trayectoria, habilidades_iniciales, perfil_id
    )
    print(f"\n  Validación: {'✓ Trayectoria válida' if ok else f'✗ {msg}'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Career Path Planner — Planificación de trayectoria profesional"
    )
    parser.add_argument(
        "--objetivo", type=str, default=None,
        help="Objetivo profesional en lenguaje natural",
    )
    parser.add_argument(
        "--algoritmo", type=str, default="astar",
        choices=["astar", "greedy"],
        help="Algoritmo de búsqueda (default: astar)",
    )
    parser.add_argument(
        "--sin-llm", action="store_true",
        help="Ejecutar sin usar el LLM",
    )
    parser.add_argument(
        "--habilidades", type=str, default="",
        help="Habilidades iniciales separadas por comas (ej: python_basico,estadistica)",
    )
    args = parser.parse_args()

    imprimir_bienvenida()

    try:
        grafo = GrafoCursos()
    except FileNotFoundError as e:
        print(f"\n  ✗ {e}")
        sys.exit(1)

    print(f"\n  Sistema cargado: {len(grafo.cursos)} cursos | "
          f"{len(grafo.habilidades)} habilidades | "
          f"{len(grafo.perfiles)} perfiles\n")

    habilidades_iniciales = _parsear_habilidades(args.habilidades, grafo)

    # Obtener objetivo
    if args.objetivo:
        objetivo_texto = args.objetivo.strip()
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
    try:
        if args.sin_llm:
            modo_sin_llm(objetivo_texto, args.algoritmo, grafo, habilidades_iniciales)
        else:
            modo_con_llm(objetivo_texto, args.algoritmo, grafo, habilidades_iniciales)
    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrumpido por el usuario.")
    except EnvironmentError as e:
        print(f"\n  ✗ {e}")
        print("  Ejecuta con --sin-llm para usar el sistema sin API key.")
    except Exception as e:
        print(f"\n  ✗ Error inesperado: {e}")
        raise

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()