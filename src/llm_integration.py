"""
llm_integration.py
------------------
Integración LLM via OpenRouter (sin restricciones geográficas, gratuito).

OpenRouter: https://openrouter.ai
Modelo usado: meta-llama/llama-3.1-8b-instruct:free (100% gratuito)

Configuración:
  1. Crea cuenta en https://openrouter.ai
  2. Ve a Keys → Create Key
  3. En .env agrega: OPENROUTER_API_KEY=tu_clave

Expone:
  parsear_objetivo(texto, grafo)    → habilidades del dataset
  evaluar_trayectoria(...)          → puntuación + análisis narrativo
  pipeline_completo(...)            → flujo end-to-end
"""

import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from graph import GrafoCursos, Curso

# ── Configuración ─────────────────────────────────────────────────────────────

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# Modelos gratuitos disponibles en OpenRouter:
#   meta-llama/llama-3.1-8b-instruct:free   → rápido, bueno para JSON
#   meta-llama/llama-3.2-3b-instruct:free   → más ligero
#   mistralai/mistral-7b-instruct:free       → muy bueno para instrucciones
MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash:free")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _get_client() -> OpenAI:
    """Crea cliente OpenAI apuntando a OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY no encontrada.\n"
            "1. Ve a https://openrouter.ai\n"
            "2. Crea cuenta → Keys → Create Key\n"
            "3. En .env agrega: OPENROUTER_API_KEY=tu_clave"
        )
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


# ── Llamada al LLM ────────────────────────────────────────────────────────────

def _llamar_llm(prompt: str, system: str, reintentos: int = 3) -> str:
    """Llama a OpenRouter con reintentos automáticos."""
    client = _get_client()

    for intento in range(reintentos):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=1024,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                extra_headers={
                    "HTTP-Referer": "https://github.com/career-path-planner",
                    "X-Title": "Career Path Planner",
                },
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            err_str = str(e)
            if ("429" in err_str or "rate" in err_str.lower()) \
                    and intento < reintentos - 1:
                # Usar retry_after_seconds de la respuesta si está disponible
                match = re.search(r"retry_after_seconds[^:]+:\s*(\d+)", err_str)
                if not match:
                    match = re.search(r"(\d+)\s*second", err_str)
                espera = int(match.group(1)) + 5 if match else 35
                print(f"    ⏳ Rate limit (intento {intento+1}/{reintentos}). "
                      f"Esperando {espera}s...")
                time.sleep(espera)
                continue
            raise


# ── Parser JSON robusto ───────────────────────────────────────────────────────

def _parsear_json_respuesta(texto: str) -> dict:
    """
    Extrae JSON de la respuesta del LLM de forma robusta.
    Maneja markdown, comillas tipográficas y JSON con texto extra.
    """
    texto = texto.strip()

    # 1. Eliminar bloques markdown
    if texto.startswith("```"):
        lineas = texto.split("\n")
        texto = "\n".join(
            lineas[1:-1] if lineas[-1].strip() == "```" else lineas[1:]
        ).strip()

    # 2. Reemplazar comillas tipográficas Unicode
    texto = (texto
             .replace("\u201c", '"').replace("\u201d", '"')
             .replace("\u2018", "'").replace("\u2019", "'"))

    # 3. Parsear directamente
    try:
        return json.loads(texto, strict=False)
    except json.JSONDecodeError:
        pass

    # 4. Extraer bloque { ... } con regex
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

    # 5. Sanitizar comillas dobles dentro de valores string
    try:
        sanitizado = re.sub(
            r'(:\s*")((?:[^"\\]|\\.)*)(")',
            lambda m: m.group(1) + m.group(2).replace('"', "'") + m.group(3),
            texto
        )
        return json.loads(sanitizado, strict=False)
    except Exception:
        pass

    raise ValueError(f"No se encontró JSON válido en:\n{texto}")


# ── FUNCIÓN 1: Parseador de objetivos ─────────────────────────────────────────

def _construir_prompt_parseador(objetivo_texto: str, habilidades: list) -> str:
    habs_str = ", ".join(sorted(habilidades))
    return f"""Analiza este objetivo profesional e identifica habilidades del catalogo.

CATALOGO:
{habs_str}

REGLAS:
- Solo habilidades EXACTAS del catalogo.
- Entre 5 y 8 habilidades relevantes.
- Responde UNICAMENTE con JSON, sin texto extra ni markdown.

EJEMPLOS:
Objetivo: "Quiero ser ingeniero de datos en la nube"
{{"habilidades_requeridas": ["python_avanzado", "sql_avanzado", "big_data", "cloud_aws", "docker"], "perfil_detectado": "Data Engineer", "confianza": "alta", "razon": "Requiere manejo de datos, SQL y cloud."}}

Objetivo: "Quiero especializarme en vision computacional"
{{"habilidades_requeridas": ["python_avanzado", "machine_learning", "deep_learning", "computer_vision", "algebra_lineal"], "perfil_detectado": "Computer Vision Engineer", "confianza": "alta", "razon": "Vision computacional requiere deep learning y matematicas."}}

Objetivo: "{objetivo_texto}"
"""


def parsear_objetivo(objetivo_texto: str, grafo: GrafoCursos) -> dict:
    """
    Transforma un objetivo profesional en lenguaje natural
    en habilidades estructuradas del dataset.
    """
    inicio = time.perf_counter()
    prompt = _construir_prompt_parseador(objetivo_texto, sorted(grafo.habilidades))
    system = (
        "Eres experto en carreras tecnologicas. "
        "Respondes SOLO con JSON valido, sin markdown ni texto adicional."
    )
    texto = _llamar_llm(prompt, system)
    try:
        resultado = _parsear_json_respuesta(texto)
    except Exception as e:
        raise ValueError(f"Error parseando respuesta LLM: {e}\nRespuesta: {texto}")

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
        f"  {i+1}. {c.nombre} ({c.duracion_semanas}s, {c.nivel})"
        f" - ensena: {', '.join(sorted(c.habilidades))}"
        for i, c in enumerate(trayectoria)
    )
    habs_ini = (", ".join(sorted(habilidades_iniciales))
                if habilidades_iniciales else "ninguna")
    total    = sum(c.duracion_semanas for c in trayectoria)

    return f"""Evalua esta trayectoria de aprendizaje.

OBJETIVO: {objetivo_texto}
PERFIL: {perfil_id}
HABILIDADES INICIALES: {habs_ini}
TRAYECTORIA ({len(trayectoria)} cursos, {total} semanas):
{tray_str}

Responde SOLO con este JSON (sin markdown, sin comillas especiales):
{{"puntuacion": <0-10>, "nivel_calidad": <"excelente"|"bueno"|"aceptable"|"deficiente">, "fortalezas": ["..."], "debilidades": ["..."], "sugerencias": ["..."], "resumen": "..."}}"""


def evaluar_trayectoria(
    objetivo_texto: str,
    perfil_id: str,
    trayectoria: list,
    habilidades_iniciales: frozenset,
    grafo: GrafoCursos = None,
) -> dict:
    """Evalúa semánticamente la calidad de una trayectoria."""
    inicio = time.perf_counter()
    prompt = _construir_prompt_evaluador(
        objetivo_texto, perfil_id, trayectoria, habilidades_iniciales
    )
    system = (
        "Eres experto en educacion tecnologica. "
        "Evaluas trayectorias de forma critica. "
        "Respondes SOLO con JSON valido, sin markdown."
    )
    texto = _llamar_llm(prompt, system)
    try:
        resultado = _parsear_json_respuesta(texto)
    except Exception as e:
        raise ValueError(f"Error parseando respuesta LLM: {e}\nRespuesta: {texto}")

    resultado["tiempo_segundos"] = round(time.perf_counter() - inicio, 3)
    return resultado


# ── Pipeline completo ─────────────────────────────────────────────────────────

def _detectar_perfil_cercano(habs: frozenset, grafo: GrafoCursos) -> str:
    return max(
        grafo.perfiles.keys(),
        key=lambda pid: len(habs & grafo.perfiles[pid].habilidades_requeridas),
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

    print(f"\n  [1/3] Parseando objetivo con LLM ({MODEL})...")
    parseo = parsear_objetivo(objetivo_texto, grafo)
    resultado["paso1_parseo"] = parseo

    habs_req = frozenset(parseo["habilidades_validas"])
    if not habs_req:
        print("  ✗ No se identificaron habilidades válidas.")
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

    print(f"  ✓ Trayectoria: {r.num_cursos} cursos, "
          f"{r.costo_total_semanas} semanas")

    print(f"\n  [3/3] Evaluando trayectoria con LLM ({MODEL})...")
    evaluacion = evaluar_trayectoria(
        objetivo_texto, perfil_id, r.trayectoria,
        habilidades_iniciales, grafo
    )
    resultado["paso3_evaluacion"] = evaluacion
    resultado["exito_total"] = True

    print(f"  ✓ Puntuación: {evaluacion.get('puntuacion')}/10 "
          f"({evaluacion.get('nivel_calidad')})")
    return resultado