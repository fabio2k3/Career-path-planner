"""
test_day5.py
------------
Prueba la integración del LLM:
 
  1. Función 1 — Parseador: 5 objetivos en lenguaje natural distintos
  2. Función 2 — Evaluador: evalúa trayectorias generadas por A*
  3. Pipeline completo: objetivo NL → A* → evaluación LLM
 
Requiere OPENROUTER_API_KEY en el archivo .env
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from graph import GrafoCursos
from search import astar, greedy
from llm_integration import (
    parsear_objetivo,
    evaluar_trayectoria,
    pipeline_completo,
)
 
SEP  = "═" * 62
SEP2 = "─" * 62
 
 
def imprimir_parseo(resultado: dict, objetivo: str):
    print(f"\n{SEP2}")
    print(f"  Objetivo : \"{objetivo}\"")
    print(SEP2)
    print(f"  Perfil detectado  : {resultado.get('perfil_detectado', '?')}")
    print(f"  Confianza         : {resultado.get('confianza', '?')}")
    print(f"  Razón             : {resultado.get('razon', '?')}")
    print(f"  Habilidades válidas ({len(resultado['habilidades_validas'])}):")
    for h in sorted(resultado["habilidades_validas"]):
        print(f"    · {h}")
    if resultado.get("habilidades_invalidas"):
        print(f"  ⚠ Habilidades fuera del catálogo: {resultado['habilidades_invalidas']}")
    print(f"  Tiempo LLM: {resultado['tiempo_segundos']}s")
 
 
def imprimir_evaluacion(evaluacion: dict, algoritmo: str, num_cursos: int, semanas: int):
    print(f"\n{SEP2}")
    print(f"  Evaluación LLM — {algoritmo} ({num_cursos} cursos, {semanas} semanas)")
    print(SEP2)
    puntuacion = evaluacion.get("puntuacion", "?")
    calidad    = evaluacion.get("nivel_calidad", "?")
 
    # Barra visual de puntuación
    if isinstance(puntuacion, (int, float)):
        barra = "█" * int(puntuacion) + "░" * (10 - int(puntuacion))
        print(f"  Puntuación : {puntuacion}/10  [{barra}]  {calidad.upper()}")
    else:
        print(f"  Puntuación : {puntuacion}/10  {calidad.upper()}")
 
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
    print(f"    \"{evaluacion.get('resumen', '?')}\"")
    print(f"\n  Tiempo LLM: {evaluacion.get('tiempo_segundos', '?')}s")
 
 
def test_parseador(grafo):
    """Prueba el parseador con 5 objetivos en lenguaje natural distintos."""
    print(f"\n{SEP}")
    print(f"  TEST 1 — PARSEADOR DE OBJETIVOS (5 casos)")
    print(SEP)
 
    objetivos = [
        "Quiero convertirme en Data Scientist y trabajar analizando datos con Python",
        "Me interesa el desarrollo backend, construir APIs y manejar bases de datos",
        "Quiero llevar modelos de machine learning a producción como ML Engineer",
        "Soy frontend y quiero aprender a hacer APIs con Python y desplegarlas en la nube",
        "Quiero especializarme en deep learning y trabajar con redes neuronales",
    ]
 
    resultados = []
    for i, objetivo in enumerate(objetivos, 1):
        print(f"\n  Caso {i}/{len(objetivos)}...")
        try:
            resultado = parsear_objetivo(objetivo, grafo)
            imprimir_parseo(resultado, objetivo)
            resultados.append({"objetivo": objetivo, "resultado": resultado})
        except Exception as e:
            print(f"  ✗ Error: {e}")
 
    print(f"\n  ✓ Parseador probado con {len(resultados)}/{len(objetivos)} objetivos.")
    return resultados
 
 
def test_evaluador(grafo):
    """Prueba el evaluador con trayectorias generadas por A* y Greedy."""
    print(f"\n{SEP}")
    print(f"  TEST 2 — EVALUADOR DE TRAYECTORIAS")
    print(SEP)
 
    instancias = {i.id: i for i in grafo.cargar_instancias()}
    casos = [
        ("inst_01", "Quiero convertirme en Data Scientist desde cero"),
        ("inst_05", "Soy desarrollador frontend y quiero pasarme al backend con Python"),
        ("inst_07", "Soy Data Scientist y quiero llevar modelos a producción como ML Engineer"),
    ]
 
    for inst_id, objetivo in casos:
        inst = instancias[inst_id]
        print(f"\n  Instancia: {inst_id} — {inst.descripcion}")
 
        # Generar trayectoria con A* (criterio cursos, más rápido)
        r_astar = astar(
            grafo,
            inst.habilidades_iniciales,
            inst.perfil_objetivo,
            inst_id,
            criterio="cursos",
        )
 
        if not r_astar.exito:
            print(f"  ✗ A* no encontró trayectoria.")
            continue
 
        print(f"  Trayectoria A*: {r_astar.num_cursos} cursos, "
              f"{r_astar.costo_total_semanas} semanas")
        print(f"  Enviando al LLM para evaluación...")
 
        try:
            evaluacion = evaluar_trayectoria(
                objetivo,
                inst.perfil_objetivo,
                r_astar.trayectoria,
                inst.habilidades_iniciales,
            )
            imprimir_evaluacion(
                evaluacion, "A*(cursos)",
                r_astar.num_cursos, r_astar.costo_total_semanas,
            )
        except Exception as e:
            print(f"  ✗ Error en evaluación: {e}")
 
 
def test_pipeline_completo(grafo):
    """Prueba el pipeline end-to-end: LN → A* → evaluación."""
    print(f"\n{SEP}")
    print(f"  TEST 3 — PIPELINE COMPLETO (objetivo NL → trayectoria → evaluación)")
    print(SEP)
 
    objetivo = "Quiero trabajar como ML Engineer construyendo y desplegando modelos de inteligencia artificial en producción"
 
    print(f"\n  Objetivo: \"{objetivo}\"")
 
    try:
        resultado = pipeline_completo(
            objetivo_texto=objetivo,
            grafo=grafo,
            algoritmo_fn=astar,
            instancia_id="pipeline_test",
        )
 
        if resultado["exito_total"]:
            print(f"\n  {'─'*60}")
            print(f"  RESULTADO FINAL DEL PIPELINE")
            print(f"  {'─'*60}")
            busqueda = resultado["paso2_busqueda"]
            evaluacion = resultado["paso3_evaluacion"]
            print(f"  Cursos en trayectoria : {busqueda['num_cursos']}")
            print(f"  Semanas totales       : {busqueda['costo_total_semanas']}")
            print(f"  Puntuación LLM        : {evaluacion.get('puntuacion', '?')}/10")
            print(f"  Calidad               : {evaluacion.get('nivel_calidad', '?')}")
            print(f"  Resumen LLM:")
            print(f"    \"{evaluacion.get('resumen', '?')}\"")
        else:
            print(f"\n  ✗ Pipeline no completado.")
 
    except Exception as e:
        print(f"  ✗ Error en pipeline: {e}")
 
 
def main():
    print(SEP)
    print("  TEST DÍA 5 — Integración LLM")
    print("  Career Path Planner")
    print(SEP)
 
    grafo = GrafoCursos()
    print(f"\n  Grafo: {len(grafo.cursos)} cursos | "
          f"{len(grafo.habilidades)} habilidades | "
          f"{len(grafo.perfiles)} perfiles")
 
    # Verificar API key antes de empezar
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n  ✗ ERROR: OPENROUTER_API_KEY no encontrada.")
        print("  Crea un archivo .env en la raíz del proyecto con:")
        print("  OPENROUTER_API_KEY=tu_clave_aqui")
        return
 
    print("\n  ✓ API key detectada. Iniciando tests...\n")
 
    # Test 1: Parseador
    test_parseador(grafo)
 
    # Test 2: Evaluador
    test_evaluador(grafo)
 
    # Test 3: Pipeline completo
    test_pipeline_completo(grafo)
 
    print(f"\n{SEP}")
    print("  ✓ Todos los tests del día 5 completados.")
    print(SEP)
 
 
if __name__ == "__main__":
    main()