"""
generate_synthetic_dataset.py
-----------------------------
Genera una versión ampliada y coherente del dataset de cursos usando LLM.

Objetivo:
  - Expandir el catálogo de cursos de forma procedimental.
  - Mantener el grafo acíclico y validable (DAG).
  - Crear un archivo final compatible con el resto del proyecto.

Uso:
    python scripts/generate_synthetic_dataset.py
    python scripts/generate_synthetic_dataset.py --courses 150 --profiles 15
    python scripts/generate_synthetic_dataset.py --input data/dataset.json --output data/dataset_sintetico.json
    python scripts/generate_synthetic_dataset.py --courses 150 --validate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ── Configuración ─────────────────────────────────────────────────────────────

_MODELOS_PREFERIDOS = [
    os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]

DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_RETRIES = 4

LEVELS = ["principiante", "intermedio", "avanzado"]

DOMAIN_HINTS = [
    "datos y analitica",
    "backend y APIs",
    "machine learning",
    "deep learning",
    "infraestructura y cloud",
    "calidad de software",
    "sistemas distribuidos",
    "ciberseguridad",
    "ingenieria de datos",
    "MLOps",
]


# ── Entorno ───────────────────────────────────────────────────────────────────

def _load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path if env_path.exists() else None)


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


# ── Llamada al LLM con fallback de modelo ─────────────────────────────────────

def _call_llm(
    prompt: str,
    system: str,
    retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    """
    Llama a HuggingFace Inference API con reintentos automáticos.
    Prueba los modelos en _MODELOS_PREFERIDOS en orden.
    """
    client = _get_client()

    for modelo in _MODELOS_PREFERIDOS:
        for intento in range(retries):
            try:
                response = client.chat.completions.create(
                    model=modelo,
                    max_tokens=2048,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                err = str(e).lower()

                if ("429" in err or "rate" in err or "too many" in err) \
                        and intento < retries - 1:
                    match  = re.search(r"(\d+)\s*second", err)
                    espera = int(match.group(1)) + 5 if match else 30
                    print(f"    ⏳ Rate limit [{modelo}] intento {intento+1}/{retries}. "
                          f"Esperando {espera}s...")
                    time.sleep(espera)
                    continue

                if ("503" in err or "loading" in err) and intento < retries - 1:
                    print(f"    ⏳ Modelo cargando [{modelo}]. Esperando 20s...")
                    time.sleep(20)
                    continue

                print(f"    ⚠ Modelo [{modelo}] falló: {str(e)[:80]}. "
                      f"Probando siguiente...")
                break

    raise RuntimeError(
        "Todos los modelos fallaron. Verifica tu HF_API_KEY y conexión."
    )


# ── Parser JSON robusto ───────────────────────────────────────────────────────

def _extract_json(text: str) -> Any:
    text = text.strip()

    # Eliminar bloques markdown
    if text.startswith("```"):
        lines = text.splitlines()
        fin   = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text  = "\n".join(lines[1:fin]).strip()

    # Normalizar comillas tipográficas
    text = (text
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2018", "'").replace("\u2019", "'"))

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    raise ValueError(f"No se pudo extraer JSON válido:\n{text[:300]}")


# ── Utilidades de dataset ─────────────────────────────────────────────────────

def _next_course_id(existing: list[dict]) -> str:
    """Genera el siguiente ID de curso en formato c01, c02, ..."""
    max_num = 0
    for c in existing:
        m = re.fullmatch(r"c(\d+)", str(c.get("id", "")))
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"c{max_num + 1:02d}"


def _normalize_course(
    course: dict,
    known_skills: list[str],
    used_names: set[str],
    existing_courses: list[dict],
) -> dict | None:
    """
    Normaliza y valida un curso generado por el LLM.
    Devuelve el curso limpio o None si no es válido.
    """
    name = str(course.get("nombre", "")).strip()
    if not name or name in used_names:
        return None

    # Habilidades enseñadas
    raw_skills = course.get("habilidades", [])
    if isinstance(raw_skills, str):
        raw_skills = [raw_skills]
    skills = [str(s).strip() for s in raw_skills if str(s).strip()]
    skills = list(dict.fromkeys(skills))  # deduplicar preservando orden

    # Prerrequisitos (solo si existen en el catálogo)
    prereq = course.get("prerrequisitos", [])
    if isinstance(prereq, str):
        prereq = [prereq]
    prereq = [str(p).strip() for p in prereq if str(p).strip() in known_skills]

    # Evitar que un curso se requiera a sí mismo
    skills = [s for s in skills if s not in prereq]
    if not skills:
        return None

    # Duración
    dur = course.get("duracion_semanas", 0)
    try:
        dur = int(dur)
    except Exception:
        dur = 0
    dur = max(2, min(12, dur))

    # Nivel
    nivel = str(course.get("nivel", "")).strip().lower()
    if nivel not in LEVELS:
        nivel = "principiante" if dur <= 4 else ("intermedio" if dur <= 7 else "avanzado")

    # Descripción
    descripcion = str(course.get("descripcion", "")).strip()
    if not descripcion:
        descripcion = f"Curso de {name.lower()} con enfoque práctico y progresivo."

    cid = _next_course_id(existing_courses)
    used_names.add(name)

    return {
        "id":                cid,
        "nombre":            name,
        "descripcion":       descripcion,
        "prerrequisitos":    prereq,
        "habilidades":       skills,
        "duracion_semanas":  dur,
        "nivel":             nivel,
    }


# ── Validación de DAG ─────────────────────────────────────────────────────────

def _es_dag(dataset: dict) -> tuple[bool, list[str]]:
    """
    Verifica que el grafo de dependencias entre habilidades sea un DAG.
    Devuelve (True, []) si es válido, (False, nodos_en_ciclo) si no.
    """
    from collections import defaultdict

    habilidades = set(dataset.get("habilidades", []))
    grafo       = defaultdict(set)
    in_degree   = {h: 0 for h in habilidades}

    for curso in dataset.get("cursos", []):
        for hab_ens in curso.get("habilidades", []):
            for pre in curso.get("prerrequisitos", []):
                if hab_ens not in grafo[pre]:
                    grafo[pre].add(hab_ens)
                    in_degree[hab_ens] = in_degree.get(hab_ens, 0) + 1

    cola       = deque(h for h in habilidades if in_degree.get(h, 0) == 0)
    procesados = 0
    while cola:
        nodo = cola.popleft()
        procesados += 1
        for vecino in grafo[nodo]:
            in_degree[vecino] -= 1
            if in_degree[vecino] == 0:
                cola.append(vecino)

    if procesados < len(habilidades):
        en_ciclo = [h for h in habilidades if in_degree.get(h, 0) > 0]
        return False, en_ciclo
    return True, []


# ── Generación de cursos ──────────────────────────────────────────────────────

def _build_course_prompt(
    existing_skills: list[str],
    existing_courses: list[dict],
    batch_size: int,
    level: str,
    domain: str,
    new_skill_budget: int,
) -> str:
    sample = [
        {
            "nombre":           c["nombre"],
            "habilidades":      c["habilidades"],
            "prerrequisitos":   c["prerrequisitos"],
            "duracion_semanas": c["duracion_semanas"],
            "nivel":            c["nivel"],
        }
        for c in existing_courses[-8:]
    ]
    return f"""
Genera exactamente {batch_size} cursos nuevos para ampliar un dataset académico.

DOMINIO: {domain}
NIVEL: {level}

REGLAS:
- Devuelve SOLO un array JSON válido.
- Cada objeto: nombre, descripcion, prerrequisitos, habilidades, duracion_semanas, nivel.
- prerrequisitos: SOLO habilidades ya existentes en el catálogo.
- Ningún curso puede requerirse a sí mismo.
- No dupliques nombres existentes.
- Puedes introducir 1-2 habilidades nuevas en snake_case si son plausibles.
- Duración: principiante 2-4 sem, intermedio 4-7 sem, avanzado 6-12 sem.
- Introduce al menos {new_skill_budget} habilidades nuevas si encajan.

CATÁLOGO DE HABILIDADES:
{json.dumps(existing_skills, ensure_ascii=False)}

EJEMPLOS DE ESTILO:
{json.dumps(sample, ensure_ascii=False, indent=2)}

Devuelve únicamente el JSON.
""".strip()


def generate_courses(
    base_data: dict,
    target_total: int,
    batch_size: int,
    output_path: Path,
) -> dict:
    """
    Genera cursos hasta alcanzar target_total.
    Guarda el progreso parcial en output_path tras cada lote.
    """
    data             = deepcopy(base_data)
    existing_courses = data.get("cursos", [])
    existing_skills  = list(dict.fromkeys(data.get("habilidades", [])))
    used_names       = {c["nombre"] for c in existing_courses}
    current_total    = len(existing_courses)

    if current_total >= target_total:
        print(f"  Ya hay {current_total} cursos. No se generan más.")
        return data

    print(f"  Base: {current_total} cursos | {len(existing_skills)} habilidades")
    print(f"  Objetivo: {target_total} cursos")

    remaining = target_total - current_total
    thirds    = [remaining // 3, remaining // 3, remaining - 2 * (remaining // 3)]
    phases    = list(zip(LEVELS, thirds))

    system = (
        "Eres un diseñador curricular experto. "
        "Generas cursos coherentes y estrictamente en JSON."
    )

    for level, amount in phases:
        while amount > 0 and len(existing_courses) < target_total:
            take   = min(batch_size, amount, target_total - len(existing_courses))
            domain = random.choice(DOMAIN_HINTS)
            prompt = _build_course_prompt(
                existing_skills, existing_courses,
                take, level, domain,
                max(1, take // 2),
            )

            try:
                raw    = _call_llm(prompt, system)
                parsed = _extract_json(raw)
            except Exception as e:
                print(f"  ⚠ Error generando lote ({level}): {e}. Reintentando...")
                continue

            if not isinstance(parsed, list):
                print(f"  ⚠ LLM no devolvió lista para nivel {level}. Saltando.")
                continue

            accepted = 0
            for course in parsed:
                if not isinstance(course, dict):
                    continue
                normalized = _normalize_course(
                    course, existing_skills, used_names, existing_courses
                )
                if normalized is None:
                    continue
                existing_courses.append(normalized)
                for s in normalized["habilidades"]:
                    if s not in existing_skills:
                        existing_skills.append(s)
                accepted += 1
                if len(existing_courses) >= target_total:
                    break

            if accepted == 0:
                print(f"  ⚠ Ningún curso válido en el lote ({level}). Continuando.")
                amount -= 1
                continue

            amount -= accepted
            print(f"  ✓ Lote {level}: +{accepted} cursos | "
                  f"total={len(existing_courses)}")

            # Guardado incremental tras cada lote
            data["cursos"]      = existing_courses
            data["habilidades"] = existing_skills
            _update_metadata(data)
            _guardar_json(data, output_path)

    data["cursos"]      = existing_courses
    data["habilidades"] = existing_skills
    return data


# ── Generación de perfiles ────────────────────────────────────────────────────

def _build_profile_prompt(
    skills: list[str],
    existing_profiles: dict,
    count: int,
) -> str:
    return f"""
Genera exactamente {count} nuevos perfiles profesionales para un dataset académico.

REGLAS:
- Devuelve SOLO un array JSON válido.
- Cada objeto: id, nombre, descripcion, habilidades_requeridas.
- habilidades_requeridas: entre 6 y 10 habilidades del catálogo.
- No repitas nombres de perfiles existentes.
- Los perfiles deben ser variados y útiles para rutas distintas.

PERFILES EXISTENTES:
{json.dumps(existing_profiles, ensure_ascii=False, indent=2)}

CATÁLOGO DE HABILIDADES:
{json.dumps(skills, ensure_ascii=False)}

Devuelve únicamente el JSON.
""".strip()


def generate_profiles(data: dict, target_profiles: int) -> dict:
    """Genera perfiles profesionales hasta alcanzar target_profiles."""
    profiles = data.get("perfiles_profesionales", {})
    skills   = data.get("habilidades", [])

    existing_names = {v["nombre"] for v in profiles.values()}
    needed         = max(0, target_profiles - len(profiles))

    if needed == 0:
        print(f"  Ya hay {len(profiles)} perfiles. No se generan más.")
        return data

    print(f"  Generando {needed} perfil(es) adicional(es)...")

    prompt = _build_profile_prompt(skills, profiles, needed)
    system = (
        "Eres un arquitecto curricular experto en perfiles profesionales. "
        "Respondes únicamente con JSON."
    )

    try:
        raw    = _call_llm(prompt, system)
        parsed = _extract_json(raw)
    except Exception as e:
        print(f"  ⚠ Error generando perfiles: {e}. Se mantienen los existentes.")
        return data

    if not isinstance(parsed, list):
        print(f"  ⚠ LLM no devolvió lista de perfiles. Se mantienen los existentes.")
        return data

    next_idx = len(profiles) + 1
    added    = 0

    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = str(item.get("nombre", "")).strip()
        if not name or name in existing_names:
            continue

        req = item.get("habilidades_requeridas", [])
        if isinstance(req, str):
            req = [req]
        req = [str(s).strip() for s in req if str(s).strip() in skills]

        if len(req) < 6:
            continue
        req = req[:10]

        pid = str(item.get("id", "")).strip()
        if not pid:
            pid = f"perfil_{next_idx:02d}"
        while pid in profiles:
            next_idx += 1
            pid = f"perfil_{next_idx:02d}"

        profiles[pid] = {
            "nombre":      name,
            "descripcion": str(item.get("descripcion", "")).strip()
                           or f"Perfil profesional en {name}.",
            "habilidades_requeridas": req,
        }
        existing_names.add(name)
        next_idx += 1
        added    += 1

        if len(profiles) >= target_profiles:
            break

    print(f"  ✓ {added} perfil(es) añadido(s). Total: {len(profiles)}")
    data["perfiles_profesionales"] = profiles
    return data


# ── Utilidades de archivo ─────────────────────────────────────────────────────

def _update_metadata(data: dict) -> None:
    meta = data.setdefault("metadata", {})
    meta["total_cursos"]      = len(data.get("cursos", []))
    meta["total_habilidades"] = len(data.get("habilidades", []))
    meta["version"]           = str(meta.get("version", "1.0")) + "-synthetic"
    meta["descripcion"]       = (
        "Dataset ampliado procedimentalmente con apoyo de LLM "
        "para simulación de trayectorias profesionales."
    )


def _guardar_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Validación opcional post-generación ──────────────────────────────────────

def _validar_dataset(output_path: Path) -> None:
    """
    Ejecuta el validador de scripts/ si existe.
    Busca automáticamente sin depender de rutas relativas frágiles.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    candidates = [
        BASE_DIR / "scripts" / "validate_dataset.py",
        BASE_DIR / "src"     / "validate_dataset.py",
    ]
    validator = next((p for p in candidates if p.exists()), None)

    if validator is None:
        print("  ⚠ Validador no encontrado. Omitiendo validación externa.")
        return

    import subprocess
    print(f"\n  Ejecutando validador: {validator.name}")
    result = subprocess.run(
        [sys.executable, str(validator), "--input", str(output_path)],
        capture_output=False,
    )
    if result.returncode != 0:
        print("  ⚠ El validador reportó errores. Revisa el dataset generado.")
    else:
        print("  ✓ Dataset validado correctamente.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _load_env()

    BASE_DIR = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Genera un dataset sintético ampliado con apoyo de LLM."
    )
    parser.add_argument(
        "--input",  type=Path,
        default=BASE_DIR / "data" / "dataset.json",
        help="Dataset de origen (default: data/dataset.json)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=BASE_DIR / "data" / "dataset_sintetico.json",
        help="Ruta de salida (default: data/dataset_sintetico.json)",
    )
    parser.add_argument(
        "--courses",    type=int, default=150,
        help="Número total de cursos a alcanzar (default: 150)",
    )
    parser.add_argument(
        "--profiles",   type=int, default=15,
        help="Número total de perfiles a alcanzar (default: 15)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Cursos por llamada al LLM (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--seed",       type=int, default=42,
        help="Semilla aleatoria para reproducibilidad (default: 42)",
    )
    parser.add_argument(
        "--validate",   action="store_true",
        help="Ejecutar el validador al finalizar.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("  GENERACIÓN DE DATASET SINTÉTICO — Career Path Planner")
    print("=" * 60)
    print(f"  Input   : {args.input}")
    print(f"  Output  : {args.output}")
    print(f"  Cursos  : {args.courses}")
    print(f"  Perfiles: {args.profiles}")
    print(f"  Semilla : {args.seed}")

    if not args.input.exists():
        print(f"\n  ✗ No existe el dataset de origen: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Generación de cursos (con guardado incremental)
    print(f"\n  [1/3] Generando cursos...")
    data = generate_courses(
        data,
        args.courses,
        max(1, args.batch_size),
        args.output,
    )

    # Generación de perfiles
    print(f"\n  [2/3] Generando perfiles...")
    data = generate_profiles(data, args.profiles)

    # Validación interna de DAG antes de guardar versión final
    print(f"\n  [3/3] Validando DAG interno...")
    es_valido, ciclos = _es_dag(data)
    if not es_valido:
        print(f"  ⚠ Se detectaron ciclos en el DAG: {ciclos[:5]}")
        print(f"    El dataset se guarda igualmente pero requiere revisión.")
    else:
        print(f"  ✓ El grafo es un DAG válido.")

    # Guardado final
    _update_metadata(data)
    _guardar_json(data, args.output)

    print(f"\n  ✓ Dataset guardado en: {args.output}")
    print(f"    Cursos     : {len(data['cursos'])}")
    print(f"    Habilidades: {len(data['habilidades'])}")
    print(f"    Perfiles   : {len(data['perfiles_profesionales'])}")

    # Validación externa opcional
    if args.validate:
        _validar_dataset(args.output)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()