"""
test_llm.py

Prueba la integración del LLM:
  1. Parseador: 5 objetivos en lenguaje natural distintos.
  2. Evaluador: evalúa trayectorias generadas por A*.
  3. Pipeline completo: objetivo NL → A* → evaluación LLM.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from dotenv import load_dotenv
load_dotenv(dotenv_path=ROOT / ".env")

from graph import GrafoCursos
from search import astar
from llm_integration import (
    parsear_objetivo,
    evaluar_trayectoria_con_fallback,
    pipeline_completo,
)


SEP  = "═" * 62
SEP2 = "─" * 62

# ── Comprobación de API key ───────────────────────────────────────────────────

_API_DISPONIBLE = bool(os.getenv("HF_API_KEY"))


def _skip_si_sin_api(nombre_test: str) -> bool:
    """Imprime aviso y devuelve True si se debe saltar el test."""
    if not _API_DISPONIBLE:
        print(f"\n  ⚠ {nombre_test} omitido: HF_API_KEY no encontrada.")
        print(f"    Configura el .env para ejecutar este test con LLM real.")
        print(f"    El evaluador simulado seguirá funcionando como fallback.")
        return True
    return False


# ── Helpers de impresión ──────────────────────────────────────────────────────

def _get_instancia(instancias: list, inst_id: str, fallback_idx: int = 0):
    match = next((i for i in instancias if i.id == inst_id), None)
    if match is None:
        print(f"  ⚠ Instancia '{inst_id}' no encontrada. "
              f"Usando posición {fallback_idx}.")
        return instancias[fallback_idx] if fallback_idx < len(instancias) else None
    return match


def imprimir_parseo(resultado: dict, objetivo: str) -> None:
    print(f"\n{SEP2}")
    print(f"  Objetivo : \"{objetivo}\"")
    print(SEP2)
    print(f"  Perfil detectado  : {resultado.get('perfil_detectado', '?')}")
    print(f"  Confianza         : {resultado.get('confianza', '?')}")
    print(f"  Razón             : {resultado.get('razon', '?')}")
    print(f"  Habilidades válidas ({len(resultado.get('habilidades_validas', []))}):")
    for h in sorted(resultado.get("habilidades_validas", [])):
        print(f"    · {h}")
    if resultado.get("habilidades_invalidas"):
        print(f"  ⚠ Fuera del catálogo: {resultado['habilidades_invalidas']}")
    print(f"  Tiempo LLM: {resultado.get('tiempo_segundos', '?')}s")


def imprimir_evaluacion(evaluacion: dict, contexto: str) -> None:
    print(f"\n{SEP2}")
    print(f"  Evaluación — {contexto}")
    print(SEP2)
    puntuacion = evaluacion.get("puntuacion", "?")
    calidad    = str(evaluacion.get("nivel_calidad", "?")).upper()
    modo       = evaluacion.get("modo", "llm_real")
    sufijo     = " [simulado]" if modo == "simulado" else " [LLM real]"

    if isinstance(puntuacion, (int, float)):
        barra = "█" * int(puntuacion) + "░" * (10 - int(puntuacion))
        print(f"  Puntuación : {puntuacion}/10  [{barra}]  {calidad}{sufijo}")
    else:
        print(f"  Puntuación : {puntuacion}/10  {calidad}{sufijo}")

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
    print(f"  Tiempo  : {evaluacion.get('tiempo_segundos', '?')}s")


# ── Test 1: Parseador ─────────────────────────────────────────────────────────

def test_parseador(grafo: GrafoCursos) -> None:
    print(f"\n{SEP}")
    print(f"  TEST 1 — PARSEADOR DE OBJETIVOS (5 casos)")
    print(SEP)

    if _skip_si_sin_api("TEST 1 — Parseador"):
        return

    objetivos = [
        "Quiero convertirme en Data Scientist y trabajar analizando datos con Python",
        "Me interesa el desarrollo backend, construir APIs y manejar bases de datos",
        "Quiero llevar modelos de machine learning a producción como ML Engineer",
        "Soy frontend y quiero aprender a hacer APIs con Python y desplegarlas en la nube",
        "Quiero especializarme en deep learning y trabajar con redes neuronales",
    ]

    exitos = 0
    for i, objetivo in enumerate(objetivos, 1):
        print(f"\n  Caso {i}/{len(objetivos)}...")
        try:
            resultado = parsear_objetivo(objetivo, grafo)
            imprimir_parseo(resultado, objetivo)
            exitos += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n  ✓ Parseador: {exitos}/{len(objetivos)} casos exitosos.")


# ── Test 2: Evaluador ─────────────────────────────────────────────────────────

def test_evaluador(grafo: GrafoCursos, instancias: list) -> None:
    print(f"\n{SEP}")
    print(f"  TEST 2 — EVALUADOR DE TRAYECTORIAS")
    print(SEP)

    casos = [
        ("inst_01", 0, "Quiero convertirme en Data Scientist desde cero"),
        ("inst_05", 4, "Soy desarrollador frontend, quiero pasarme al backend"),
        ("inst_07", 6, "Soy Data Scientist y quiero llevar modelos a producción"),
    ]

    for inst_id, fallback, objetivo in casos:
        inst = _get_instancia(instancias, inst_id, fallback)
        if inst is None:
            continue

        print(f"\n  [{inst.id}] {inst.descripcion}")

        r = astar(
            grafo,
            inst.habilidades_iniciales,
            inst.perfil_objetivo,
            inst.id,
            criterio="cursos",
        )

        if not r.exito:
            print(f"  ✗ A* no encontró trayectoria.")
            continue

        print(f"  Trayectoria A*: {r.num_cursos} cursos, "
              f"{r.costo_total_semanas} semanas")

        # Siempre usa con_fallback: si no hay API usa evaluador simulado
        try:
            evaluacion = evaluar_trayectoria_con_fallback(
                objetivo,
                inst.perfil_objetivo,
                r.trayectoria,
                inst.habilidades_iniciales,
            )
            imprimir_evaluacion(
                evaluacion,
                f"A*(cursos) — {r.num_cursos} cursos, {r.costo_total_semanas} sem",
            )
        except Exception as e:
            print(f"  ✗ Error en evaluación: {e}")


# ── Test 3: Pipeline completo ─────────────────────────────────────────────────

def test_pipeline_completo(grafo: GrafoCursos) -> None:
    print(f"\n{SEP}")
    print(f"  TEST 3 — PIPELINE COMPLETO (NL → búsqueda → evaluación)")
    print(SEP)

    if _skip_si_sin_api("TEST 3 — Pipeline completo"):
        return

    objetivo = (
        "Quiero trabajar como ML Engineer construyendo y desplegando "
        "modelos de inteligencia artificial en producción"
    )
    print(f"\n  Objetivo: \"{objetivo}\"")

    try:
        resultado = pipeline_completo(
            objetivo_texto=objetivo,
            grafo=grafo,
            algoritmo_fn=astar,
            instancia_id="pipeline_test",
        )

        if resultado["exito_total"]:
            busqueda   = resultado["paso2_busqueda"]
            evaluacion = resultado["paso3_evaluacion"]
            print(f"\n  {SEP2}")
            print(f"  RESULTADO FINAL DEL PIPELINE")
            print(f"  {SEP2}")
            print(f"  Cursos en trayectoria : {busqueda['num_cursos']}")
            print(f"  Semanas totales       : {busqueda['costo_total_semanas']}")
            print(f"  Puntuación LLM        : {evaluacion.get('puntuacion', '?')}/10")
            print(f"  Calidad               : {evaluacion.get('nivel_calidad', '?')}")
            print(f"  Resumen: \"{evaluacion.get('resumen', '?')}\"")
        else:
            print(f"\n  ✗ Pipeline no completado.")

    except Exception as e:
        print(f"  ✗ Error en pipeline: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(SEP)
    print("  TEST LLM — Career Path Planner")
    print(SEP)

    if _API_DISPONIBLE:
        print("\n  ✓ HF_API_KEY detectada. Usando LLM real.")
    else:
        print("\n  ⚠ HF_API_KEY no encontrada.")
        print("    Tests de parseo y pipeline se omitirán.")
        print("    El evaluador (test 2) usará el modo simulado.\n")

    grafo      = GrafoCursos()
    instancias = grafo.cargar_instancias()

    print(f"\n  Grafo: {len(grafo.cursos)} cursos | "
          f"{len(grafo.habilidades)} habilidades | "
          f"{len(grafo.perfiles)} perfiles")

    test_parseador(grafo)
    test_evaluador(grafo, instancias)
    test_pipeline_completo(grafo)

    print(f"\n{SEP}")
    print("  ✓ Tests completados.")
    print(SEP)


if __name__ == "__main__":
    main()