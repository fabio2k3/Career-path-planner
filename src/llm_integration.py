"""
llm_integration.py

Integración LLM via Hugging Face Inference API (gratuito).

Configuración:
  1. Crea cuenta en https://huggingface.co
  2. Settings → Access Tokens → New token (tipo Read)
  3. En .env: HF_API_KEY=hf_xxxxxxxxxxxxxxxx

Expone:
  parsear_objetivo(texto, grafo, habilidades_iniciales)
      → dict con habilidades requeridas del dataset
  evaluar_trayectoria(...)
      → dict con puntuación 0-10 + análisis narrativo
  evaluar_trayectoria_con_fallback(...)
      → igual, pero cae al evaluador simulado si la API falla
  pipeline_completo(...)
      → flujo end-to-end: LN → búsqueda → evaluación
"""

import json
import os
import re
import time
from pathlib import Path
from typing import FrozenSet

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from graph import GrafoCursos, Curso

# ── Configuración ─────────────────────────────────────────────────────────────

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# Lista de modelos con fallback automático si el principal falla
_MODELOS_PREFERIDOS = [
    os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]


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


# ── Llamada al LLM con reintentos y fallback de modelo ────────────────────────

def _llamar_llm(prompt: str, system: str, reintentos: int = 4) -> str:
    """
    Llama a HuggingFace Inference API con reintentos automáticos.
    Prueba los modelos en _MODELOS_PREFERIDOS en orden hasta obtener respuesta.
    """
    client = _get_client()

    for modelo in _MODELOS_PREFERIDOS:
        for intento in range(reintentos):
            try:
                response = client.chat.completions.create(
                    model=modelo,
                    max_tokens=1024,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                err_str = str(e)

                # Rate limit → esperar y reintentar
                if ("429" in err_str or "rate" in err_str.lower() or
                        "too many" in err_str.lower()) and intento < reintentos - 1:
                    match  = re.search(r"(\d+)\s*second", err_str)
                    espera = int(match.group(1)) + 5 if match else 30
                    print(f"    ⏳ Rate limit [{modelo}] intento {intento+1}/{reintentos}. "
                          f"Esperando {espera}s...")
                    time.sleep(espera)
                    continue

                # Modelo cargando → esperar y reintentar
                if ("503" in err_str or "loading" in err_str.lower()) and intento < reintentos - 1:
                    print(f"    ⏳ Modelo cargando [{modelo}]. Esperando 20s...")
                    time.sleep(20)
                    continue

                # Error no recuperable en este modelo → probar el siguiente
                print(f"    ⚠ Modelo [{modelo}] falló: {err_str[:80]}. Probando siguiente...")
                break

    raise RuntimeError(
        "Todos los modelos fallaron. Verifica tu HF_API_KEY y conexión a internet."
    )


# ── Parser JSON robusto ───────────────────────────────────────────────────────

def _parsear_json_respuesta(texto: str) -> dict:
    """Extrae JSON válido de la respuesta del LLM de forma robusta."""
    texto = texto.strip()

    # 1. Eliminar bloques markdown ```json ... ```
    if texto.startswith("```"):
        lineas = texto.split("\n")
        fin    = len(lineas) - 1 if lineas[-1].strip() == "```" else len(lineas)
        texto  = "\n".join(lineas[1:fin]).strip()

    # 2. Normalizar comillas tipográficas
    texto = (texto
             .replace("\u201c", '"').replace("\u201d", '"')
             .replace("\u2018", "'").replace("\u2019", "'"))

    # 3. Parseo directo
    try:
        return json.loads(texto, strict=False)
    except json.JSONDecodeError:
        pass

    # 4. Extraer primer bloque { ... }
    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if match:
        bloque = (match.group(0)
                  .replace("\u201c", '"').replace("\u201d", '"'))
        try:
            return json.loads(bloque, strict=False)
        except json.JSONDecodeError:
            pass

    # 5. Sanitizar comillas internas en valores de string
    try:
        sanitizado = re.sub(
            r'(:\s*")((?:[^"\\]|\\.)*?)(")',
            lambda m: m.group(1) + m.group(2).replace('"', "'") + m.group(3),
            texto,
        )
        return json.loads(sanitizado, strict=False)
    except Exception:
        pass

    raise ValueError(
        f"No se encontró JSON válido en la respuesta del LLM:\n{texto[:300]}"
    )


# ── FUNCIÓN 1: Parseador de objetivos ─────────────────────────────────────────

def _construir_prompt_parseador(
    objetivo_texto: str,
    habilidades: list[str],
    habilidades_iniciales: FrozenSet[str],
) -> str:
    habs_str = ", ".join(sorted(habilidades))
    ini_str  = (", ".join(sorted(habilidades_iniciales))
                if habilidades_iniciales else "ninguna")
    return f"""Analiza este objetivo profesional e identifica las habilidades necesarias del catalogo.

OBJETIVO: "{objetivo_texto}"
HABILIDADES YA ADQUIRIDAS POR EL USUARIO: {ini_str}

CATALOGO COMPLETO:
{habs_str}

REGLAS:
- Selecciona SOLO habilidades que existan EXACTAMENTE en el catalogo.
- No incluyas habilidades que el usuario ya tiene.
- Selecciona entre 5 y 8 habilidades que sean mas relevantes para el objetivo.
- Responde UNICAMENTE con un objeto JSON valido, sin texto extra ni markdown.

FORMATO DE RESPUESTA:
{{"habilidades_requeridas": ["hab1", "hab2", ...], "perfil_detectado": "nombre del perfil", "confianza": "alta|media|baja", "razon": "explicacion breve"}}

EJEMPLO:
Objetivo: "Quiero ser ingeniero de datos en la nube"
Habilidades ya adquiridas: python_basico
{{"habilidades_requeridas": ["python_avanzado", "sql_avanzado", "big_data", "cloud_aws", "docker"], "perfil_detectado": "Data Engineer", "confianza": "alta", "razon": "Requiere manejo de datos, SQL y cloud. Python basico ya dominado."}}
"""


def parsear_objetivo(
    objetivo_texto: str,
    grafo: GrafoCursos,
    habilidades_iniciales: FrozenSet[str] = frozenset(),
) -> dict:
    """
    Transforma un objetivo profesional en lenguaje natural
    en habilidades estructuradas del dataset.

    Parámetros:
      objetivo_texto       : texto libre del usuario
      grafo                : grafo del proyecto (para acceder al catálogo)
      habilidades_iniciales: habilidades que el usuario ya posee (se excluyen)

    Retorna dict con:
      habilidades_requeridas : lista original del LLM
      habilidades_validas    : filtradas contra el catálogo
      habilidades_invalidas  : las que no existen en el catálogo
      perfil_detectado       : nombre del perfil según el LLM
      confianza              : alta | media | baja
      razon                  : justificación del LLM
      tiempo_segundos        : latencia de la llamada
    """
    inicio = time.perf_counter()
    prompt = _construir_prompt_parseador(
        objetivo_texto, sorted(grafo.habilidades), habilidades_iniciales
    )
    system = (
        "Eres experto en carreras tecnologicas. "
        "Respondes SOLO con JSON valido, sin markdown ni texto adicional."
    )
    texto = _llamar_llm(prompt, system)

    try:
        resultado = _parsear_json_respuesta(texto)
    except Exception as e:
        raise ValueError(f"Error parseando respuesta LLM: {e}\nRespuesta: {texto}")

    # Filtrar contra el catálogo y excluir las ya adquiridas
    requeridas     = resultado.get("habilidades_requeridas", [])
    habs_validas   = [
        h for h in requeridas
        if h in grafo.habilidades and h not in habilidades_iniciales
    ]
    habs_invalidas = [h for h in requeridas if h not in grafo.habilidades]

    resultado["habilidades_validas"]   = habs_validas
    resultado["habilidades_invalidas"] = habs_invalidas
    resultado["tiempo_segundos"]       = round(time.perf_counter() - inicio, 3)
    return resultado


# ── FUNCIÓN 2: Evaluador de trayectorias ──────────────────────────────────────

def _construir_prompt_evaluador(
    objetivo_texto: str,
    perfil_id: str,
    trayectoria: list[Curso],
    habilidades_iniciales: FrozenSet[str],
) -> str:
    tray_str = "\n".join(
        f"  {i+1}. {c.nombre} ({c.duracion_semanas}s, {c.nivel})"
        f" — ensena: {', '.join(sorted(c.habilidades))}"
        for i, c in enumerate(trayectoria)
    )
    habs_ini  = (", ".join(sorted(habilidades_iniciales))
                 if habilidades_iniciales else "ninguna")
    total     = sum(c.duracion_semanas for c in trayectoria)
    n_cursos  = len(trayectoria)

    return f"""Evalua esta trayectoria de aprendizaje de forma CRITICA y DISCRIMINANTE.

OBJETIVO: {objetivo_texto}
PERFIL: {perfil_id}
HABILIDADES INICIALES DEL USUARIO: {habs_ini}
TRAYECTORIA ({n_cursos} cursos, {total} semanas en total):
{tray_str}

CRITERIOS DE EVALUACION:
- Penaliza fuerte trayectorias largas: si tiene mas de 12 cursos O mas de 50 semanas, puntuacion maxima = 7.
- Recompensa trayectorias concisas: si logra el objetivo en 5-7 cursos y menos de 35 semanas, puntuacion = 8-10.
- Evalua si la secuencia pedagogica es coherente (de menor a mayor nivel).
- Detecta cursos redundantes o irrelevantes para el objetivo.
- Considera las habilidades iniciales del usuario al evaluar si la ruta es adecuada.

REGLAS DE JUSTIFICACION:
- Menciona explicitamente en "debilidades" si la trayectoria es excesivamente larga.
- Menciona explicitamente en "fortalezas" si la trayectoria es eficiente y concisa.

Responde UNICAMENTE con este JSON (sin markdown ni texto extra):
{{"puntuacion": <0-10>, "nivel_calidad": <"excelente"|"bueno"|"aceptable"|"deficiente">, "fortalezas": ["..."], "debilidades": ["..."], "sugerencias": ["..."], "resumen": "..."}}"""


def evaluar_trayectoria(
    objetivo_texto: str,
    perfil_id: str,
    trayectoria: list[Curso],
    habilidades_iniciales: FrozenSet[str],
) -> dict:
    """
    Evalúa semánticamente la calidad de una trayectoria con el LLM.

    Retorna dict con: puntuacion, nivel_calidad, fortalezas,
    debilidades, sugerencias, resumen, tiempo_segundos.
    """
    if not objetivo_texto:
        objetivo_texto = f"Alcanzar el perfil: {perfil_id.replace('_', ' ')}"

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


# ── Evaluador simulado (fallback sin API) ─────────────────────────────────────

def _evaluar_simulado(
    perfil_id: str,
    trayectoria: list[Curso],
    habilidades_iniciales: FrozenSet[str],
) -> dict:
    """
    Evaluador determinista sin LLM basado en métricas calculadas.
    Se activa automáticamente si la API no está disponible.
    """
    n_cursos      = len(trayectoria)
    total_semanas = sum(c.duracion_semanas for c in trayectoria)
    n_avanzado    = sum(1 for c in trayectoria if c.nivel == "avanzado")

    puntuacion = 7.0
    if n_cursos <= 7 and total_semanas <= 35:
        puntuacion += 2.0
    elif n_cursos <= 10 and total_semanas <= 45:
        puntuacion += 1.0
    elif n_cursos > 12 or total_semanas > 50:
        puntuacion -= 1.0
    if n_cursos > 16 or total_semanas > 70:
        puntuacion -= 1.0

    puntuacion = round(max(4.0, min(10.0, puntuacion)))

    if puntuacion >= 9:    nivel = "excelente"
    elif puntuacion >= 7:  nivel = "bueno"
    elif puntuacion >= 5:  nivel = "aceptable"
    else:                  nivel = "deficiente"

    fortalezas  = [f"Trayectoria de {n_cursos} cursos con progresión coherente."]
    debilidades = []

    if n_cursos > 12 or total_semanas > 50:
        debilidades.append(
            f"Trayectoria extensa ({n_cursos} cursos, {total_semanas} semanas). "
            "Considera reducir cursos redundantes."
        )
    else:
        fortalezas.append(
            f"Ruta eficiente: {n_cursos} cursos completados en {total_semanas} semanas."
        )
    if n_avanzado > 0:
        fortalezas.append(
            f"Incluye {n_avanzado} curso(s) avanzado(s) para especialización."
        )

    return {
        "puntuacion":    puntuacion,
        "nivel_calidad": nivel,
        "fortalezas":    fortalezas,
        "debilidades":   debilidades or ["Podría complementarse con proyectos prácticos."],
        "sugerencias":   ["Complementar con proyectos reales para aplicar lo aprendido."],
        "resumen": (
            f"Trayectoria {nivel} de {n_cursos} cursos y {total_semanas} "
            f"semanas para {perfil_id.replace('_', ' ')}. [Evaluación simulada]"
        ),
        "modo": "simulado",
    }


def evaluar_trayectoria_con_fallback(
    objetivo_texto: str,
    perfil_id: str,
    trayectoria: list[Curso],
    habilidades_iniciales: FrozenSet[str],
) -> dict:
    """
    Intenta evaluar con el LLM real.
    Si falla por cuota, rate limit o falta de API key,
    cae automáticamente al evaluador simulado.
    """
    try:
        resultado = evaluar_trayectoria(
            objetivo_texto, perfil_id, trayectoria, habilidades_iniciales
        )
        resultado["modo"] = "llm_real"
        return resultado
    except Exception as e:
        err_str = str(e)
        if any(c in err_str for c in
               ["402", "429", "quota", "rate", "403", "401", "EnvironmentError"]):
            print(f"    ⚠ API no disponible ({err_str[:60]}). Usando evaluador simulado.")
            r = _evaluar_simulado(perfil_id, trayectoria, habilidades_iniciales)
            r["tiempo_segundos"] = 0.0
            return r
        raise


# ── Pipeline completo ─────────────────────────────────────────────────────────

def _detectar_perfil_cercano(habs: FrozenSet[str], grafo: GrafoCursos) -> str:
    """Selecciona el perfil del catálogo con mayor solapamiento de habilidades."""
    return max(
        grafo.perfiles.keys(),
        key=lambda pid: len(habs & grafo.perfiles[pid].habilidades_requeridas),
    )


def pipeline_completo(
    objetivo_texto: str,
    grafo: GrafoCursos,
    algoritmo_fn,
    habilidades_iniciales: FrozenSet[str] = frozenset(),
    instancia_id: str = "llm_pipeline",
) -> dict:
    """
    Pipeline end-to-end:
      [1] Parsear objetivo en lenguaje natural → habilidades estructuradas
      [2] Buscar trayectoria óptima con A* o Greedy
      [3] Evaluar trayectoria con LLM

    Retorna dict con los tres pasos y flag exito_total.
    """
    resultado = {
        "objetivo_texto":   objetivo_texto,
        "instancia_id":     instancia_id,
        "paso1_parseo":     None,
        "paso2_busqueda":   None,
        "paso3_evaluacion": None,
        "exito_total":      False,
    }

    # Paso 1: parseo
    print(f"\n  [1/3] Parseando objetivo con LLM...")
    parseo = parsear_objetivo(objetivo_texto, grafo, habilidades_iniciales)
    resultado["paso1_parseo"] = parseo

    habs_req = frozenset(parseo["habilidades_validas"])
    if not habs_req:
        print("  ✗ No se identificaron habilidades válidas en el catálogo.")
        return resultado

    print(f"  ✓ Habilidades identificadas: {sorted(habs_req)}")
    print(f"  ✓ Perfil detectado: {parseo.get('perfil_detectado')} "
          f"(confianza: {parseo.get('confianza')})")

    perfil_id = _detectar_perfil_cercano(habs_req, grafo)
    print(f"  ✓ Perfil del catálogo asignado: {perfil_id}")

    # Paso 2: búsqueda
    print(f"\n  [2/3] Buscando trayectoria con {algoritmo_fn.__name__}...")
    r = algoritmo_fn(
        grafo, habilidades_iniciales, perfil_id,
        instancia_id, criterio="cursos",
    )
    resultado["paso2_busqueda"] = r.to_dict()

    if not r.exito:
        print("  ✗ No se encontró trayectoria válida.")
        return resultado

    print(f"  ✓ Trayectoria: {r.num_cursos} cursos, {r.costo_total_semanas} semanas")

    # Paso 3: evaluación
    print(f"\n  [3/3] Evaluando trayectoria con LLM...")
    evaluacion = evaluar_trayectoria_con_fallback(
        objetivo_texto, perfil_id, r.trayectoria, habilidades_iniciales
    )
    resultado["paso3_evaluacion"] = evaluacion
    resultado["exito_total"]      = True

    print(f"  ✓ Puntuación: {evaluacion.get('puntuacion')}/10 "
          f"({evaluacion.get('nivel_calidad')})")
    return resultado