"""
llm_integration.py
------------------
Integración LLM vía Hugging Face Inference API (gratuito, sin créditos).

Hugging Face: https://huggingface.co
- Gratis, sin créditos que se agoten
- Solo rate limit por minuto (se resetea automáticamente)

Configuración:
  1. Crea cuenta en https://huggingface.co
  2. Settings → Access Tokens → New token (tipo Read)
  3. En .env agrega: HF_API_KEY=hf_xxxxxxxxxxxxxxxx

Expone:
  parsear_objetivo(texto, grafo)     → habilidades del dataset
  evaluar_trayectoria(...)           → puntuación + análisis narrativo
  evaluar_trayectoria_con_fallback() → igual pero con fallback simulado
  pipeline_completo(...)             → flujo end-to-end
"""

import json
import os
import re
import time
from pathlib import Path

from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from graph import GrafoCursos, Curso

# ─── Configuración ─────────────────────────────────────────────────────────────

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")


def _get_client() -> InferenceClient:
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "HF_API_KEY no encontrada.\n"
            "1. Ve a https://huggingface.co y crea cuenta gratis\n"
            "2. Settings → Access Tokens → New token (tipo Read)\n"
            "3. Agrega al .env:  HF_API_KEY=hf_xxxxxxxxxxxxxxxx"
        )
    return InferenceClient(api_key=api_key)


# ─── Llamada al LLM con reintentos ─────────────────────────────────────────────

def _llamar_llm(prompt: str, system: str, reintentos: int = 4) -> str:
    """
    Llama a HuggingFace Inference API con reintentos automáticos.
    Maneja rate limits (429), modelos cargando (503) y errores genéricos.
    """
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
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            err_str = str(e).lower()

            # Rate limit → esperar tiempo indicado y reintentar
            if ("429" in err_str or "rate" in err_str or "too many" in err_str):
                if intento < reintentos - 1:
                    match = re.search(r"(\d+)\s*second", str(e))
                    espera = int(match.group(1)) + 5 if match else 30
                    print(f"    ⏳ Rate limit (intento {intento+1}/{reintentos}). "
                          f"Esperando {espera}s...")
                    time.sleep(espera)
                    continue

            # Modelo cargando → esperar y reintentar
            if "503" in err_str or "loading" in err_str:
                if intento < reintentos - 1:
                    print(f"    ⏳ Modelo cargando (intento {intento+1}/{reintentos}). "
                          f"Esperando 20s...")
                    time.sleep(20)
                    continue

            # Otros errores transitorios → reintentar con espera corta
            if intento < reintentos - 1:
                time.sleep(5)
                continue

            raise   # Re-lanzar el último error tras agotar reintentos


# ─── Parser JSON robusto ──────────────────────────────────────────────────────

def _parsear_json_respuesta(texto: str) -> dict:
    """Extrae JSON válido de la respuesta del LLM de forma robusta."""
    texto = texto.strip()

    # Eliminar bloques markdown (```json ... ```)
    if texto.startswith("```"):
        lineas = texto.split("\n")
        texto = "\n".join(
            lineas[1:-1] if lineas[-1].strip() == "```" else lineas[1:]
        ).strip()

    # Normalizar comillas tipográficas
    texto = (texto
             .replace("\u201c", '"').replace("\u201d", '"')
             .replace("\u2018", "'").replace("\u2019", "'"))

    # Intento directo
    try:
        return json.loads(texto, strict=False)
    except json.JSONDecodeError:
        pass

    # Extraer primer bloque JSON del texto
    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if match:
        bloque = (match.group(0)
                  .replace("\u201c", '"').replace("\u201d", '"')
                  .replace("\u2018", "'").replace("\u2019", "'"))
        try:
            return json.loads(bloque, strict=False)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No se encontró JSON válido en la respuesta del LLM:\n{texto[:500]}")


# ─── FUNCIÓN 1: Parseador de objetivos ───────────────────────────────────────

def _construir_prompt_parseador(objetivo_texto: str, habilidades: list) -> str:
    habs_str = ", ".join(sorted(habilidades))
    return f"""Analiza este objetivo profesional e identifica habilidades del catalogo.

CATALOGO:
{habs_str}

REGLAS:
- Solo habilidades EXACTAS del catalogo (copia el texto exacto).
- Entre 5 y 8 habilidades relevantes.
- Responde UNICAMENTE con JSON, sin texto extra ni markdown.

EJEMPLOS:
Objetivo: "Quiero ser ingeniero de datos en la nube"
{{"habilidades_requeridas": ["python_avanzado", "sql_avanzado", "big_data", "cloud_aws", "docker"], "perfil_detectado": "Data Engineer", "confianza": "alta", "razon": "Requiere manejo de datos, SQL y cloud."}}

Objetivo: "Quiero especializarme en vision computacional"
{{"habilidades_requeridas": ["python_avanzado", "machine_learning", "deep_learning", "algebra_lineal"], "perfil_detectado": "Computer Vision Engineer", "confianza": "alta", "razon": "Vision computacional requiere deep learning y matematicas."}}

Objetivo: "{objetivo_texto}"
"""


def parsear_objetivo(objetivo_texto: str, grafo: GrafoCursos) -> dict:
    """
    Transforma un objetivo profesional en lenguaje natural
    en una lista estructurada de habilidades del dataset.

    Retorna dict con:
      habilidades_requeridas : lista original del LLM
      habilidades_validas    : habilidades que existen en el catálogo
      habilidades_invalidas  : habilidades que NO existen en el catálogo
      perfil_detectado       : perfil profesional inferido
      confianza              : 'alta' | 'media' | 'baja'
      razon                  : justificación del LLM
      tiempo_segundos        : latencia de la llamada
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
        raise ValueError(f"Error parseando respuesta del LLM: {e}\nRespuesta bruta: {texto}")

    habs_propuestas = resultado.get("habilidades_requeridas", [])
    habs_validas   = [h for h in habs_propuestas if h in grafo.habilidades]
    habs_invalidas = [h for h in habs_propuestas if h not in grafo.habilidades]

    resultado["habilidades_validas"]   = habs_validas
    resultado["habilidades_invalidas"] = habs_invalidas
    resultado["tiempo_segundos"]       = round(time.perf_counter() - inicio, 3)
    return resultado


# ─── FUNCIÓN 2: Evaluador de trayectorias ─────────────────────────────────────

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
    total_semanas = sum(c.duracion_semanas for c in trayectoria)
    num_cursos = len(trayectoria)

    return f"""Evalua esta trayectoria de aprendizaje de forma critica y discriminante.

OBJETIVO: {objetivo_texto}
PERFIL: {perfil_id}
HABILIDADES INICIALES: {habs_ini}
TRAYECTORIA ({num_cursos} cursos, {total_semanas} semanas):
{tray_str}

CRITERIOS DE EVALUACION:
- Penaliza con fuerza trayectorias largas aunque cumplan el objetivo.
- Recompensa trayectorias concisas y bien orientadas.
- Evalua la coherencia pedagogica de la secuencia (progresion logica).
- Considera si todos los cursos son necesarios para el objetivo.

REGLAS DE PENALIZACION:
- Si tiene mas de 12 cursos O mas de 50 semanas: puntuacion maxima 7.
- Si es excesivamente larga (>16 cursos o >80 semanas): puntuacion preferentemente 4-6.
- Mencionar explicitamente la sobreextension en debilidades.

REGLAS DE RECOMPENSA:
- 5 a 7 cursos y menos de 35 semanas: puntuacion 8-10 si esta bien orientada.
- Destacar concision y eficiencia en fortalezas.

RESPONDE SOLO con este JSON (sin markdown ni texto extra):
{{"puntuacion": <0-10>, "nivel_calidad": <"excelente"|"bueno"|"aceptable"|"deficiente">, "fortalezas": ["..."], "debilidades": ["..."], "sugerencias": ["..."], "resumen": "..."}}"""


def evaluar_trayectoria(
    objetivo_texto: str,
    perfil_id: str,
    trayectoria: list,
    habilidades_iniciales: frozenset,
) -> dict:
    """
    Evalúa semánticamente la calidad de una trayectoria de cursos.

    Retorna dict con:
      puntuacion     : float 0-10
      nivel_calidad  : 'excelente' | 'bueno' | 'aceptable' | 'deficiente'
      fortalezas     : lista de puntos positivos
      debilidades    : lista de puntos negativos
      sugerencias    : lista de mejoras propuestas
      resumen        : texto narrativo del evaluador
      tiempo_segundos: latencia de la llamada
    """
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
        raise ValueError(f"Error parseando evaluación del LLM: {e}\nRespuesta bruta: {texto}")

    resultado["tiempo_segundos"] = round(time.perf_counter() - inicio, 3)
    return resultado


# ─── Evaluador simulado (fallback sin API) ────────────────────────────────────

def _evaluar_simulado(
    perfil_id: str,
    trayectoria: list,
    habilidades_iniciales: frozenset,
) -> dict:
    """
    Fallback sin LLM: calcula una puntuación basada en métricas objetivas.
    Se activa automáticamente cuando la API no está disponible.
    """
    num_cursos    = len(trayectoria)
    total_semanas = sum(c.duracion_semanas for c in trayectoria)
    niveles       = [c.nivel for c in trayectoria]
    n_avanzado    = niveles.count("avanzado")
    n_principiante = niveles.count("principiante")

    # Puntuación base con ajustes por eficiencia
    puntuacion = 7.0

    if habilidades_iniciales:
        puntuacion += 0.5          # Bonus por partir con conocimiento previo

    if num_cursos <= 6:            puntuacion += 1.5
    elif num_cursos <= 8:          puntuacion += 1.0
    elif num_cursos <= 12:         puntuacion += 0.5
    elif num_cursos > 16:          puntuacion -= 1.0

    if total_semanas <= 30:        puntuacion += 1.0
    elif total_semanas <= 45:      puntuacion += 0.5
    elif total_semanas > 70:       puntuacion -= 1.0

    puntuacion = round(max(3.0, min(10.0, puntuacion)), 1)

    if puntuacion >= 9:    nivel = "excelente"
    elif puntuacion >= 7:  nivel = "bueno"
    elif puntuacion >= 5:  nivel = "aceptable"
    else:                  nivel = "deficiente"

    fortalezas = [f"Trayectoria de {num_cursos} cursos con progresión coherente."]
    if n_avanzado > 0:
        fortalezas.append(f"Incluye {n_avanzado} curso(s) avanzado(s) para especialización.")
    if num_cursos <= 8:
        fortalezas.append("Ruta eficiente con pocos pasos hasta el objetivo.")

    debilidades = []
    if total_semanas > 50:
        debilidades.append(f"Duración total de {total_semanas} semanas es considerable.")
    if num_cursos > 12:
        debilidades.append(f"Con {num_cursos} cursos, la trayectoria puede resultar extensa.")
    if not debilidades:
        debilidades.append("Podría complementarse con proyectos prácticos.")

    return {
        "puntuacion":    puntuacion,
        "nivel_calidad": nivel,
        "fortalezas":    fortalezas,
        "debilidades":   debilidades,
        "sugerencias":   ["Complementar con proyectos reales para consolidar el aprendizaje."],
        "resumen": (
            f"Trayectoria {nivel} de {num_cursos} cursos y {total_semanas} semanas "
            f"hacia {perfil_id.replace('_', ' ')}. [Evaluación simulada — sin LLM]"
        ),
        "modo": "simulado",
        "tiempo_segundos": 0.0,
    }


def evaluar_trayectoria_con_fallback(
    objetivo_texto: str,
    perfil_id: str,
    trayectoria: list,
    habilidades_iniciales: frozenset,
) -> dict:
    """
    Intenta evaluar con el LLM real. Si falla por cualquier causa de API
    (rate limit, credenciales, modelo no disponible, timeout), activa el
    evaluador simulado basado en métricas para garantizar continuidad.
    """
    try:
        resultado = evaluar_trayectoria(
            objetivo_texto, perfil_id, trayectoria, habilidades_iniciales
        )
        resultado["modo"] = "llm_real"
        return resultado
    except Exception as e:
        err_str = str(e).lower()
        # Fallback para errores de API (autenticación, cuota, disponibilidad)
        if any(c in err_str for c in ["402", "429", "quota", "rate", "403", "401",
                                       "503", "timeout", "connection", "hf_api_key"]):
            print(f"    ⚠ API no disponible ({type(e).__name__}). Usando evaluador simulado.")
            return _evaluar_simulado(perfil_id, trayectoria, habilidades_iniciales)
        raise   # Re-lanzar errores no relacionados con la API


# ─── Detección de perfil más cercano ─────────────────────────────────────────

def _detectar_perfil_cercano(habs: frozenset, grafo: GrafoCursos) -> str:
    """
    Selecciona el perfil del catálogo cuyas habilidades requeridas tienen
    mayor intersección con las habilidades identificadas por el LLM.

    Si no hay intersección con ningún perfil, retorna el primero disponible.
    """
    mejor_perfil = None
    mejor_score = -1

    for pid, perfil in grafo.perfiles.items():
        score = len(habs & perfil.habilidades_requeridas)
        if score > mejor_score:
            mejor_score = score
            mejor_perfil = pid

    return mejor_perfil or next(iter(grafo.perfiles))


# ─── Pipeline completo ────────────────────────────────────────────────────────

def pipeline_completo(
    objetivo_texto: str,
    grafo: GrafoCursos,
    algoritmo_fn,
    habilidades_iniciales: frozenset = frozenset(),
    instancia_id: str = "llm_pipeline",
) -> dict:
    """
    Pipeline end-to-end: objetivo en lenguaje natural → trayectoria → evaluación LLM.

    Pasos:
      1. LLM parsea el objetivo y extrae habilidades del catálogo.
      2. A* (u otro algoritmo) encuentra la trayectoria óptima.
      3. LLM evalúa la trayectoria y genera justificación narrativa.

    Retorna dict con claves:
      objetivo_texto, instancia_id, paso1_parseo, paso2_busqueda,
      paso3_evaluacion, exito_total.
    """
    resultado = {
        "objetivo_texto":   objetivo_texto,
        "instancia_id":     instancia_id,
        "paso1_parseo":     None,
        "paso2_busqueda":   None,
        "paso3_evaluacion": None,
        "exito_total":      False,
    }

    # Paso 1: Parsear objetivo con LLM
    print(f"\n  [1/3] Parseando objetivo con LLM ({MODEL})...")
    parseo = parsear_objetivo(objetivo_texto, grafo)
    resultado["paso1_parseo"] = parseo

    habs_req = frozenset(parseo["habilidades_validas"])
    if not habs_req:
        invalidas = parseo.get("habilidades_invalidas", [])
        print(f"  ✗ No se identificaron habilidades válidas del catálogo.")
        if invalidas:
            print(f"    El LLM propuso estas habilidades que no están en el catálogo: {invalidas}")
            print(f"    Revisa si el dataset contiene las habilidades adecuadas para este objetivo.")
        return resultado

    print(f"  ✓ Habilidades válidas ({len(habs_req)}): {sorted(habs_req)}")
    if parseo.get("habilidades_invalidas"):
        print(f"  ⚠ Habilidades ignoradas (fuera del catálogo): {parseo['habilidades_invalidas']}")
    print(f"  ✓ Perfil detectado: {parseo.get('perfil_detectado')} "
          f"(confianza: {parseo.get('confianza')})")

    # Mapear al perfil más cercano del catálogo
    perfil_id = _detectar_perfil_cercano(habs_req, grafo)
    print(f"  ✓ Perfil del catálogo asignado: {perfil_id}")

    # Paso 2: Búsqueda de trayectoria
    print(f"\n  [2/3] Buscando trayectoria con {algoritmo_fn.__name__} (criterio: cursos)...")
    r = algoritmo_fn(grafo, habilidades_iniciales, perfil_id, instancia_id, criterio="cursos")
    resultado["paso2_busqueda"] = r.to_dict()

    if not r.exito:
        print("  ✗ No se encontró trayectoria válida para el perfil detectado.")
        return resultado

    print(f"  ✓ Trayectoria encontrada: {r.num_cursos} cursos, {r.costo_total_semanas} semanas")

    # Paso 3: Evaluación LLM
    print(f"\n  [3/3] Evaluando trayectoria con LLM ({MODEL})...")
    evaluacion = evaluar_trayectoria(
        objetivo_texto, perfil_id, r.trayectoria, habilidades_iniciales
    )
    resultado["paso3_evaluacion"] = evaluacion
    resultado["exito_total"] = True

    print(f"  ✓ Puntuación: {evaluacion.get('puntuacion')}/10 "
          f"({evaluacion.get('nivel_calidad')})")
    return resultado