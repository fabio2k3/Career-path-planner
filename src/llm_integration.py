"""
llm_integration.py

Integración con LLM mediante Hugging Face Inference API.

Este módulo ofrece tres piezas principales:
- parsear_objetivo(): transforma texto libre en habilidades del catálogo.
- evaluar_trayectoria(): puntúa una ruta de aprendizaje con ayuda del LLM.
- evaluar_trayectoria_con_fallback(): usa un evaluador simulado si la API falla.
- pipeline_completo(): une parseo, búsqueda y evaluación en un flujo end-to-end.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import FrozenSet

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from graph import Curso, GrafoCursos
from search import astar as _astar

# *** Configuración *******

# Se carga el archivo .env desde la raíz del proyecto para tomar credenciales.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# Modelos que se intentan en orden si el primero falla.
_MODELOS_PREFERIDOS = [
    os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]


def _get_client() -> InferenceClient:
    """
    Construye el cliente de Hugging Face usando la API key configurada.

    Raises
    ------
    EnvironmentError
        Si la variable HF_API_KEY no está definida.
    """
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
    Llama a Hugging Face Inference API con reintentos automáticos.

    Si un modelo falla por rate limit, carga lenta o error transitorio,
    se prueba de nuevo y, si es necesario, se pasa al siguiente modelo.
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
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                err_str = str(e)

                # Rate limit: esperar y reintentar.
                if (
                    ("429" in err_str or "rate" in err_str.lower() or "too many" in err_str.lower())
                    and intento < reintentos - 1
                ):
                    match = re.search(r"(\d+)\s*second", err_str)
                    espera = int(match.group(1)) + 5 if match else 30
                    time.sleep(espera)
                    continue

                # Modelo cargando: esperar y reintentar.
                if ("503" in err_str or "loading" in err_str.lower()) and intento < reintentos - 1:
                    time.sleep(20)
                    continue

                # Error no recuperable para este modelo: probar el siguiente.
                break

    raise RuntimeError(
        "Todos los modelos fallaron. Verifica tu HF_API_KEY y conexión a internet."
    )


# *** Parser JSON robusto *******

def _parsear_json_respuesta(texto: str) -> dict:
    """
    Extrae un JSON válido de la respuesta del LLM de forma robusta.

    Se contemplan varios casos habituales:
    - respuesta envuelta en bloque markdown;
    - comillas tipográficas;
    - JSON embebido dentro de texto extra;
    - pequeñas inconsistencias en comillas internas.
    """
    texto = texto.strip()

    # 1-) Eliminar bloques markdown tipo ```json ... ```
    if texto.startswith("```"):
        lineas = texto.split("\n")
        fin = len(lineas) - 1 if lineas[-1].strip() == "```" else len(lineas)
        texto = "\n".join(lineas[1:fin]).strip()

    # 2-) Normalizar comillas tipográficas.
    texto = (
        texto.replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2018", "'").replace("\u2019", "'")
    )

    # 3-) Intento directo.
    try:
        return json.loads(texto, strict=False)
    except json.JSONDecodeError:
        pass

    # 4-) Extraer el primer bloque {...} que aparezca.
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if match:
        bloque = (
            match.group(0)
            .replace("\u201c", '"')
            .replace("\u201d", '"')
        )
        try:
            return json.loads(bloque, strict=False)
        except json.JSONDecodeError:
            pass

    # 5-) Sanitizar comillas internas en valores tipo string.
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


# *** FUNCIÓN 1: Parseador de objetivos ******

def _construir_prompt_parseador(objetivo_texto: str, habilidades: list[str], habilidades_iniciales: FrozenSet[str],) -> str:
    """
    Construye el prompt que se envía al modelo para interpretar el objetivo.

    Se incluye el catálogo completo y las habilidades ya adquiridas para que
    el LLM devuelva solo las habilidades nuevas y relevantes.
    """
    habs_str = ", ".join(sorted(habilidades))
    ini_str = ", ".join(sorted(habilidades_iniciales)) if habilidades_iniciales else "ninguna"

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


def parsear_objetivo(objetivo_texto: str, grafo: GrafoCursos, habilidades_iniciales: FrozenSet[str] = frozenset(),) -> dict:
    """
    Convierte un objetivo en lenguaje natural en habilidades estructuradas.

    El resultado del LLM se filtra contra el catálogo del grafo para separar:
    - habilidades válidas;
    - habilidades inválidas;
    - habilidades que el usuario ya poseía.
    """
    inicio = time.perf_counter()
    prompt = _construir_prompt_parseador(
        objetivo_texto,
        sorted(grafo.habilidades),
        habilidades_iniciales,
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

    # Filtrar contra el catálogo y excluir habilidades ya adquiridas.
    requeridas = resultado.get("habilidades_requeridas", [])
    habs_validas = [
        h for h in requeridas
        if h in grafo.habilidades and h not in habilidades_iniciales
    ]
    habs_invalidas = [h for h in requeridas if h not in grafo.habilidades]

    resultado["habilidades_validas"] = habs_validas
    resultado["habilidades_invalidas"] = habs_invalidas
    resultado["tiempo_segundos"] = round(time.perf_counter() - inicio, 3)
    return resultado


# *** FUNCIÓN 2: Evaluador de trayectorias *******

def _construir_prompt_evaluador(objetivo_texto: str, perfil_id: str, trayectoria: list[Curso], habilidades_iniciales: FrozenSet[str],) -> str:
    """
    Construye el prompt que pide al LLM valorar la trayectoria encontrada.

    La evaluación se enfoca en calidad pedagógica, longitud, coherencia y
    relación con las habilidades iniciales del usuario.
    """
    tray_str = "\n".join(
        f"  {i + 1}. {c.nombre} ({c.duracion_semanas}s, {c.nivel})"
        f" — ensena: {', '.join(sorted(c.habilidades))}"
        for i, c in enumerate(trayectoria)
    )
    habs_ini = ", ".join(sorted(habilidades_iniciales)) if habilidades_iniciales else "ninguna"
    total = sum(c.duracion_semanas for c in trayectoria)
    n_cursos = len(trayectoria)

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
{{"puntuacion": <0-10>, "nivel_calidad": <"excelente"|"bueno"|"aceptable"|"deficiente">, "fortalezas": ["..."], "debilidades": ["..."], "sugerencias": ["..."], "resumen": "..."}}
"""


def evaluar_trayectoria(objetivo_texto: str, perfil_id: str, trayectoria: list[Curso], habilidades_iniciales: FrozenSet[str],) -> dict:
    """
    Evalúa semánticamente la trayectoria encontrada usando el LLM.

    Devuelve un diccionario con la puntuación, el nivel de calidad y una
    explicación textual de fortalezas, debilidades y sugerencias.
    """
    if not objetivo_texto:
        objetivo_texto = f"Alcanzar el perfil: {perfil_id.replace('_', ' ')}"

    inicio = time.perf_counter()
    prompt = _construir_prompt_evaluador(
        objetivo_texto,
        perfil_id,
        trayectoria,
        habilidades_iniciales,
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



# *** Evaluador simulado (fallback sin API) ******

def _evaluar_simulado(perfil_id: str, trayectoria: list[Curso], habilidades_iniciales: FrozenSet[str],) -> dict:
    """
    Evaluador determinista sin LLM basado en métricas simples.

    Se usa como alternativa cuando la API no está disponible, preservando una
    salida estructurada para no romper el flujo principal.
    """
    n_cursos = len(trayectoria)
    total_semanas = sum(c.duracion_semanas for c in trayectoria)
    n_avanzado = sum(1 for c in trayectoria if c.nivel == "avanzado")

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

    if puntuacion >= 9:
        nivel = "excelente"
    elif puntuacion >= 7:
        nivel = "bueno"
    elif puntuacion >= 5:
        nivel = "aceptable"
    else:
        nivel = "deficiente"

    fortalezas = [f"Trayectoria de {n_cursos} cursos con progresión coherente."]
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
        "puntuacion": puntuacion,
        "nivel_calidad": nivel,
        "fortalezas": fortalezas,
        "debilidades": debilidades or ["Podría complementarse con proyectos prácticos."],
        "sugerencias": ["Complementar con proyectos reales para aplicar lo aprendido."],
        "resumen": (
            f"Trayectoria {nivel} de {n_cursos} cursos y {total_semanas} "
            f"semanas para {perfil_id.replace('_', ' ')}. [Evaluación simulada]"
        ),
        "modo": "simulado",
    }


def evaluar_trayectoria_con_fallback(objetivo_texto: str, perfil_id: str, trayectoria: list[Curso], habilidades_iniciales: FrozenSet[str],) -> dict:
    """
    Intenta evaluar con el LLM real y, si falla, usa el evaluador simulado.

    El fallback se activa ante errores típicos de autenticación, cuota o
    disponibilidad del servicio.
    """
    try:
        resultado = evaluar_trayectoria(
            objetivo_texto,
            perfil_id,
            trayectoria,
            habilidades_iniciales,
        )
        resultado["modo"] = "llm_real"
        return resultado
    except Exception as e:
        err_str = str(e)
        if any(
            c in err_str
            for c in ["402", "429", "quota", "rate", "403", "401", "EnvironmentError"]
        ):
            r = _evaluar_simulado(perfil_id, trayectoria, habilidades_iniciales)
            r["tiempo_segundos"] = 0.0
            return r
        raise


# *** Pipeline completo ********

def _detectar_perfil_cercano(habs: FrozenSet[str], grafo: GrafoCursos) -> str:
    """
    Selecciona el perfil del catálogo con mayor solapamiento de habilidades.

    Se usa como asignación práctica cuando el LLM detecta una intención, pero
    el sistema necesita mapearla a un perfil real del dataset.
    """
    return max(
        grafo.perfiles.keys(),
        key=lambda pid: len(habs & grafo.perfiles[pid].habilidades_requeridas),
    )


def pipeline_completo(objetivo_texto: str, grafo: GrafoCursos, algoritmo_fn,
    habilidades_iniciales: FrozenSet[str] = frozenset(), instancia_id: str = "llm_pipeline",
) -> dict:
    """
    Ejecuta el flujo completo:
    1. Interpretar el objetivo en lenguaje natural.
    2. Buscar una trayectoria con A* o Greedy.
    3. Evaluar la trayectoria con el LLM o con fallback.

    Returns
    -------
    dict
        Resultado estructurado de los tres pasos y una bandera final de éxito.
    """
    resultado = {
        "objetivo_texto": objetivo_texto,
        "instancia_id": instancia_id,
        "paso1_parseo": None,
        "paso2_busqueda": None,
        "paso3_evaluacion": None,
        "exito_total": False,
    }

    # Paso 1: parseo del objetivo.
    parseo = parsear_objetivo(objetivo_texto, grafo, habilidades_iniciales)
    resultado["paso1_parseo"] = parseo

    habs_req = frozenset(parseo["habilidades_validas"])
    if not habs_req:
        return resultado

    perfil_id = _detectar_perfil_cercano(habs_req, grafo)

    # Paso 2: búsqueda de trayectoria.
    if algoritmo_fn is _astar:
        r = algoritmo_fn(
            grafo,
            habilidades_iniciales,
            perfil_id,
            instancia_id,
            criterio="cursos",
        )
    else:
        r = algoritmo_fn(
            grafo,
            habilidades_iniciales,
            perfil_id,
            instancia_id,
        )

    resultado["paso2_busqueda"] = r.to_dict()

    if not r.exito:
        return resultado

    # Paso 3: evaluación de la ruta encontrada.
    evaluacion = evaluar_trayectoria_con_fallback(
        objetivo_texto,
        perfil_id,
        r.trayectoria,
        habilidades_iniciales,
    )
    resultado["paso3_evaluacion"] = evaluacion
    resultado["exito_total"] = True

    return resultado