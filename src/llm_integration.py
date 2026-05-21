"""
llm_integration.py
------------------
Módulo de integración con el LLM (Groq API - 100% gratuito).

  1. parsear_objetivo(texto, grafo)
     Transforma un objetivo en lenguaje natural en habilidades del dataset.

  2. evaluar_trayectoria(objetivo, perfil_id, trayectoria, habs_ini)
     Evalúa semánticamente la calidad de una trayectoria.

  3. pipeline_completo(objetivo, grafo, algoritmo_fn)
     Pipeline end-to-end: NL → búsqueda → evaluación.

API gratuita: Groq — https://console.groq.com
Modelo: llama-3.1-8b-instant
"""

import json
import os
import re
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

from graph import GrafoCursos, Curso

# ── Configuración ─────────────────────────────────────────────────────────────

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

MODEL      = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024


def _get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY no encontrada.\n"
            "1. Ve a https://console.groq.com\n"
            "2. Regístrate gratis\n"
            "3. API Keys → Create API Key\n"
            "4. Agrega al archivo .env:  GROQ_API_KEY=tu_clave"
        )
    return Groq(api_key=api_key)


# ── Utilidades internas ───────────────────────────────────────────────────────

def _llamar_llm(prompt: str, system: str, reintentos: int = 3) -> str:
    """Llama a la API de Groq con reintentos automáticos en caso de rate limit."""
    client = _get_client()
    for intento in range(reintentos):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "rate_limit" in err_str.lower()) \
                    and intento < reintentos - 1:
                match = re.search(r"(\d+\.?\d*)\s*second", err_str)
                espera = float(match.group(1)) + 2 if match else 20
                print(f"    ⏳ Rate limit (intento {intento+1}/{reintentos}). "
                      f"Esperando {espera:.0f}s...")
                time.sleep(espera)
                continue
            raise


def _parsear_json_respuesta(texto: str) -> dict:
    """
    Extrae el JSON de la respuesta del LLM de forma robusta.
    Estrategia:
      1. Limpiar bloques markdown
      2. Reemplazar comillas tipograficas Unicode
      3. Parsear con strict=False (tolera caracteres de control)
      4. Fallback: extraer bloque JSON con regex
    """
    texto = texto.strip()

    # 1. Eliminar bloques markdown ```json ... ```
    if texto.startswith("```"):
        lineas = texto.split("\n")
        texto = "\n".join(
            lineas[1:-1] if lineas[-1].strip() == "```" else lineas[1:]
        ).strip()

    # 2. Reemplazar comillas tipograficas Unicode por comillas ASCII
    texto = (texto
             .replace("\u201c", '"').replace("\u201d", '"')
             .replace("\u2018", "'").replace("\u2019", "'"))

    # 3. Intentar con strict=False (tolera \n y \t dentro de strings)
    try:
        return json.loads(texto, strict=False)
    except json.JSONDecodeError:
        pass

    # 4. Extraer primer bloque { ... } del texto y reintentar
    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if match:
        bloque = match.group(0)
        bloque = (bloque
                  .replace("\u201c", '"').replace("\u201d", '"')
                  .replace("\u2018", "'").replace("\u2019", "'"))
        try:
            return json.loads(bloque, strict=False)
        except json.JSONDecodeError:
            pass

        # 5. Ultimo recurso: sanitizar comillas dobles dentro de valores string
    # Patron: reemplazar " que aparecen dentro de valores JSON por comilla simple
    try:
        sanitizado = re.sub(
            r'(?<=[:\[,]\s)"((?:[^"\\]|\\.)*?)"(?=[,\]}])',
            lambda m: '"' + m.group(1).replace('"', "'") + '"',
            texto
        )
        return json.loads(sanitizado, strict=False)
    except Exception:
        pass

    raise ValueError(f"No se encontro JSON valido en la respuesta:\n{texto}")

# ── FUNCIÓN 1: Parseador de objetivos ─────────────────────────────────────────

def _construir_prompt_parseador(objetivo_texto: str, habilidades_disponibles: list) -> str:
    habs_str = ", ".join(sorted(habilidades_disponibles))
    return f"""Analiza este objetivo profesional e identifica las habilidades necesarias del catálogo.

CATÁLOGO DE HABILIDADES DISPONIBLES:
{habs_str}

REGLAS:
- Solo selecciona habilidades que estén EXACTAMENTE en el catálogo.
- Selecciona entre 5 y 8 habilidades directamente necesarias para el objetivo.
- Responde ÚNICAMENTE con JSON válido, sin texto adicional ni markdown.

EJEMPLOS:

Objetivo: "Quiero trabajar como ingeniero de datos procesando pipelines en la nube"
{{"habilidades_requeridas": ["python_avanzado", "sql_avanzado", "big_data", "cloud_aws", "docker"], "perfil_detectado": "Data Engineer", "confianza": "alta", "razon": "El rol requiere SQL avanzado, Big Data para pipelines y cloud para despliegue."}}

Objetivo: "Me gustaria especializarme en vision computacional"
{{"habilidades_requeridas": ["python_avanzado", "machine_learning", "deep_learning", "computer_vision", "algebra_lineal"], "perfil_detectado": "Computer Vision Engineer", "confianza": "alta", "razon": "Vision computacional requiere redes convolucionales y bases matematicas solidas."}}

Objetivo: "{objetivo_texto}"
"""


def parsear_objetivo(objetivo_texto: str, grafo: GrafoCursos) -> dict:
    """
    Transforma un objetivo profesional en lenguaje natural en una lista
    estructurada de habilidades requeridas del dataset.
    """
    inicio = time.perf_counter()
    prompt = _construir_prompt_parseador(objetivo_texto, sorted(grafo.habilidades))
    system = (
        "Eres un experto en planificacion de carreras tecnologicas. "
        "Respondes SIEMPRE con JSON valido y nada mas, sin markdown."
    )
    texto = _llamar_llm(prompt, system)
    try:
        resultado = _parsear_json_respuesta(texto)
    except Exception as e:
        raise ValueError(f"Error parseando respuesta del LLM: {e}\nRespuesta: {texto}")

    habs_validas   = [h for h in resultado.get("habilidades_requeridas", [])
                      if h in grafo.habilidades]
    habs_invalidas = [h for h in resultado.get("habilidades_requeridas", [])
                      if h not in grafo.habilidades]

    resultado["habilidades_validas"]   = habs_validas
    resultado["habilidades_invalidas"] = habs_invalidas
    resultado["tiempo_segundos"]       = round(time.perf_counter() - inicio, 3)
    return resultado


# ── FUNCIÓN 2: Evaluador de trayectorias ──────────────────────────────────────

def _construir_prompt_evaluador(
    objetivo_texto: str,
    perfil_id: str,
    trayectoria: list,
    habilidades_iniciales: frozenset,
) -> str:
    tray_str = "\n".join(
        f"  {i+1}. {c.nombre} ({c.duracion_semanas} semanas, nivel {c.nivel})"
        f" - ensena: {', '.join(sorted(c.habilidades))}"
        for i, c in enumerate(trayectoria)
    )
    habs_ini_str = (
        ", ".join(sorted(habilidades_iniciales))
        if habilidades_iniciales else "ninguna (parte desde cero)"
    )
    total_semanas = sum(c.duracion_semanas for c in trayectoria)

    return f"""Evalua esta trayectoria de aprendizaje para el objetivo dado.

OBJETIVO: {objetivo_texto}
PERFIL OBJETIVO: {perfil_id}
HABILIDADES INICIALES: {habs_ini_str}
TRAYECTORIA ({len(trayectoria)} cursos, {total_semanas} semanas):
{tray_str}

Evalua: relevancia, coherencia pedagogica, completitud, eficiencia y progresion.

Responde UNICAMENTE con JSON valido (sin markdown, sin comillas especiales):
{{"puntuacion": <0-10>, "nivel_calidad": <"excelente" o "bueno" o "aceptable" o "deficiente">, "fortalezas": ["...", "..."], "debilidades": ["...", "..."], "sugerencias": ["...", "..."], "resumen": "..."}}"""


def evaluar_trayectoria(
    objetivo_texto: str,
    perfil_id: str,
    trayectoria: list,
    habilidades_iniciales: frozenset,
) -> dict:
    """
    Evalúa semánticamente la calidad de una trayectoria generada por A*/Greedy.
    """
    inicio = time.perf_counter()
    prompt = _construir_prompt_evaluador(
        objetivo_texto, perfil_id, trayectoria, habilidades_iniciales
    )
    system = (
        "Eres un experto en educacion tecnologica y planificacion de carreras. "
        "Evaluas trayectorias de forma critica y constructiva. "
        "Respondes SIEMPRE con JSON valido y nada mas, sin markdown, "
        "sin comillas tipograficas, solo comillas ASCII estandar."
    )
    texto = _llamar_llm(prompt, system)
    try:
        resultado = _parsear_json_respuesta(texto)
    except Exception as e:
        raise ValueError(f"Error parseando respuesta del LLM: {e}\nRespuesta: {texto}")

    resultado["tiempo_segundos"] = round(time.perf_counter() - inicio, 3)
    return resultado


# ── Pipeline completo ─────────────────────────────────────────────────────────

def _detectar_perfil_cercano(habs_requeridas: frozenset, grafo: GrafoCursos) -> str:
    return max(
        grafo.perfiles.keys(),
        key=lambda pid: len(habs_requeridas & grafo.perfiles[pid].habilidades_requeridas),
    )


def pipeline_completo(
    objetivo_texto: str,
    grafo: GrafoCursos,
    algoritmo_fn,
    habilidades_iniciales: frozenset = frozenset(),
    instancia_id: str = "llm_pipeline",
) -> dict:
    """Pipeline end-to-end: objetivo NL → trayectoria → evaluación LLM."""
    resultado = {
        "objetivo_texto":   objetivo_texto,
        "instancia_id":     instancia_id,
        "paso1_parseo":     None,
        "paso2_busqueda":   None,
        "paso3_evaluacion": None,
        "exito_total":      False,
    }

    print(f"\n  [1/3] Parseando objetivo con LLM...")
    parseo = parsear_objetivo(objetivo_texto, grafo)
    resultado["paso1_parseo"] = parseo

    habs_req = frozenset(parseo["habilidades_validas"])
    if not habs_req:
        print("  ✗ El LLM no identificó habilidades válidas.")
        return resultado

    print(f"  ✓ Habilidades: {sorted(habs_req)}")
    print(f"  ✓ Perfil detectado: {parseo.get('perfil_detectado')} "
          f"(confianza: {parseo.get('confianza')})")

    perfil_id = _detectar_perfil_cercano(habs_req, grafo)
    print(f"  ✓ Perfil del catálogo: {perfil_id}")

    print(f"\n  [2/3] Buscando trayectoria con {algoritmo_fn.__name__}...")
    r = algoritmo_fn(grafo, habilidades_iniciales, perfil_id, instancia_id,
                     criterio="cursos")
    resultado["paso2_busqueda"] = r.to_dict()

    if not r.exito:
        print("  ✗ No se encontró trayectoria válida.")
        return resultado

    print(f"  ✓ Trayectoria: {r.num_cursos} cursos, {r.costo_total_semanas} semanas")

    print(f"\n  [3/3] Evaluando trayectoria con LLM...")
    evaluacion = evaluar_trayectoria(
        objetivo_texto, perfil_id, r.trayectoria, habilidades_iniciales
    )
    resultado["paso3_evaluacion"] = evaluacion
    resultado["exito_total"] = True

    print(f"  ✓ Puntuación: {evaluacion.get('puntuacion')}/10 "
          f"({evaluacion.get('nivel_calidad')})")
    return resultado