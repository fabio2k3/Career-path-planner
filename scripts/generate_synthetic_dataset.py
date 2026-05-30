"""
generate_synthetic_dataset.py
-----------------------------

Genera una versión ampliada y coherente del dataset de cursos usando LLM.
Objetivo:
  - Expandir el catálogo de cursos de forma procedimental.
  - Mantener el grafo acíclico y validable.
  - Crear un archivo final compatible con el proyecto.

Entrada por defecto:
  - dataset.json

Salida por defecto:
  - dataset_sintetico.json

Uso:
  python generate_synthetic_dataset.py --input dataset.json --output dataset_sintetico.json
  python generate_synthetic_dataset.py --input dataset.json --output dataset_sintetico.json --courses 150 --profiles 15 --validate

Requisitos:
  - HF_API_KEY en .env o variable de entorno.
  - huggingface_hub y python-dotenv instalados.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
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


def load_env():
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()


def get_client() -> InferenceClient:
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "HF_API_KEY no encontrada. Crea un token de lectura en Hugging Face "
            "y añádelo a tu .env como HF_API_KEY=hf_xxx"
        )
    return InferenceClient(api_key=api_key)


def call_llm(prompt: str, system: str, retries: int = DEFAULT_MAX_RETRIES) -> str:
    client = get_client()
    last_error = None

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=2048,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_error = e
            err = str(e).lower()
            if any(x in err for x in ["429", "rate", "too many"]) and attempt < retries - 1:
                time.sleep(20 + 5 * attempt)
                continue
            if any(x in err for x in ["503", "loading"]) and attempt < retries - 1:
                time.sleep(15 + 5 * attempt)
                continue
            if attempt < retries - 1:
                time.sleep(5)
                continue

    raise RuntimeError(f"LLM falló tras {retries} intentos: {last_error}")


def extract_json(text: str) -> Any:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[-1].strip().startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
        else:
            text = "\n".join(lines[1:]).strip()

    text = (
        text.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
    )

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError(f"No se pudo extraer JSON válido:\n{text}")


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9áéíóúñü]+", "_", text, flags=re.IGNORECASE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "curso"


def next_course_id(existing_courses: List[Dict[str, Any]]) -> str:
    max_num = 0
    for c in existing_courses:
        cid = str(c.get("id", ""))
        m = re.fullmatch(r"c(\d+)", cid)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"c{max_num + 1:02d}"


def normalize_course(course: Dict[str, Any],
                     known_skills: List[str],
                     used_names: set,
                     existing_courses: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    name = str(course.get("nombre", "")).strip()
    if not name or name in used_names:
        return None

    raw_skills = course.get("habilidades", [])
    if isinstance(raw_skills, str):
        raw_skills = [raw_skills]
    skills = []
    for s in raw_skills:
        s = str(s).strip()
        if s and s not in skills:
            skills.append(s)

    prereq = course.get("prerrequisitos", [])
    if isinstance(prereq, str):
        prereq = [prereq]
    prereq = [str(p).strip() for p in prereq if str(p).strip() in known_skills]

    # Evitar auto-dependencia
    skills = [s for s in skills if s not in prereq]

    if not skills:
        return None

    dur = course.get("duracion_semanas", 0)
    try:
        dur = int(dur)
    except Exception:
        dur = 0
    dur = max(2, min(12, dur))

    nivel = str(course.get("nivel", "")).strip().lower()
    if nivel not in LEVELS:
        if dur <= 4:
            nivel = "principiante"
        elif dur <= 7:
            nivel = "intermedio"
        else:
            nivel = "avanzado"

    descripcion = str(course.get("descripcion", "")).strip()
    if not descripcion:
        descripcion = f"Curso de {name.lower()} con enfoque práctico y progresivo."

    # Asignar id final
    cid = next_course_id(existing_courses)

    used_names.add(name)
    return {
        "id": cid,
        "nombre": name,
        "descripcion": descripcion,
        "prerrequisitos": prereq,
        "habilidades": skills,
        "duracion_semanas": dur,
        "nivel": nivel,
    }


def build_course_prompt(existing_skills: List[str],
                        existing_courses: List[Dict[str, Any]],
                        batch_size: int,
                        level: str,
                        domain: str,
                        new_skill_budget: int) -> str:
    sample_courses = [
        {
            "nombre": c["nombre"],
            "habilidades": c["habilidades"],
            "prerrequisitos": c["prerrequisitos"],
            "duracion_semanas": c["duracion_semanas"],
            "nivel": c["nivel"],
        }
        for c in existing_courses[-8:]
    ]

    return f"""
Genera exactamente {batch_size} cursos nuevos para ampliar un dataset académico de planificación de trayectorias.

DOMINIO TEMÁTICO:
- {domain}

NIVEL OBJETIVO:
- {level}

REGLAS ESTRICTAS:
- Responde SOLO con JSON válido.
- Debes devolver un ARRAY de objetos.
- Cada objeto debe tener: nombre, descripcion, prerrequisitos, habilidades, duracion_semanas, nivel.
- Los prerrequisitos deben ser SOLO habilidades ya existentes en el catálogo dado.
- Ningún curso puede requerirse a sí mismo.
- Evita duplicar nombres ya presentes.
- Mantén coherencia académica y progresión realista.
- Los cursos pueden introducir 1 o 2 habilidades nuevas si son plausibles.
- Si introduces habilidades nuevas, usa nombres estilo snake_case, semánticos y reutilizables.
- No inventes prerequisitos que no existan.
- Duración:
  * principiante: 2 a 4 semanas
  * intermedio: 4 a 7 semanas
  * avanzado: 6 a 12 semanas
- Intenta que al menos {new_skill_budget} habilidades nuevas aparezcan en este lote si encajan naturalmente.

CATÁLOGO DE HABILIDADES DISPONIBLES:
{json.dumps(existing_skills, ensure_ascii=False, indent=2)}

EJEMPLOS DE ESTILO:
{json.dumps(sample_courses, ensure_ascii=False, indent=2)}

Devuelve únicamente el JSON.
""".strip()


def build_profile_prompt(skills: List[str], existing_profiles: Dict[str, Any], count: int) -> str:
    return f"""
Genera exactamente {count} nuevos perfiles profesionales para un grafo académico de planificación de trayectorias.

REGLAS:
- Responde SOLO con JSON válido.
- Devuelve un ARRAY de objetos.
- Cada objeto debe tener: id, nombre, descripcion, habilidades_requeridas.
- habilidades_requeridas debe contener entre 6 y 10 habilidades del catálogo.
- No repitas nombres de perfiles ya existentes.
- Los perfiles deben ser variados, realistas y útiles para rutas distintas.
- Prioriza perfiles que creen caminos alternativos y densos.

PERFILES EXISTENTES:
{json.dumps(existing_profiles, ensure_ascii=False, indent=2)}

CATÁLOGO DE HABILIDADES:
{json.dumps(skills, ensure_ascii=False, indent=2)}

Devuelve únicamente el JSON.
""".strip()


def generate_courses(base_data: Dict[str, Any],
                     target_total: int,
                     batch_size: int = DEFAULT_BATCH_SIZE) -> Dict[str, Any]:
    data = deepcopy(base_data)
    existing_courses = data.get("cursos", [])
    existing_skills = list(dict.fromkeys(data.get("habilidades", [])))
    used_names = {c["nombre"] for c in existing_courses}

    current_total = len(existing_courses)
    target_total = max(target_total, current_total)

    print(f"Base: {current_total} cursos | {len(existing_skills)} habilidades")
    print(f"Objetivo: {target_total} cursos")

    # Fase por niveles para favorecer un grafo denso pero acíclico
    phase_plan = []
    remaining = target_total - current_total
    if remaining <= 0:
        phase_plan = []
    else:
        thirds = [remaining // 3, remaining // 3, remaining - 2 * (remaining // 3)]
        phase_plan = list(zip(LEVELS, thirds))

    for level, amount in phase_plan:
        while amount > 0 and len(existing_courses) < target_total:
            take = min(batch_size, amount, target_total - len(existing_courses))
            domain = random.choice(DOMAIN_HINTS)
            prompt = build_course_prompt(
                existing_skills=existing_skills,
                existing_courses=existing_courses,
                batch_size=take,
                level=level,
                domain=domain,
                new_skill_budget=max(1, take // 2),
            )
            system = (
                "Eres un diseñador curricular experto. "
                "Generas cursos coherentes, realistas y estrictamente en JSON."
            )

            raw = call_llm(prompt, system)
            parsed = extract_json(raw)
            if not isinstance(parsed, list):
                raise ValueError("El LLM no devolvió una lista de cursos.")

            accepted = 0
            for course in parsed:
                if not isinstance(course, dict):
                    continue
                normalized = normalize_course(course, existing_skills, used_names, existing_courses)
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
                raise ValueError("No se aceptó ningún curso válido en este lote.")

            amount -= accepted
            print(f"  ✓ Lote {level}: +{accepted} cursos | total={len(existing_courses)}")

    data["cursos"] = existing_courses
    data["habilidades"] = existing_skills
    data["metadata"]["total_cursos"] = len(existing_courses)
    data["metadata"]["total_habilidades"] = len(existing_skills)
    data["metadata"]["version"] = str(data["metadata"].get("version", "1.0")) + "-synthetic"

    return data


def generate_profiles(data: Dict[str, Any], target_profiles: int) -> Dict[str, Any]:
    profiles = data.get("perfiles_profesionales", {})
    skills = data.get("habilidades", [])

    existing_profile_names = {v["nombre"] for v in profiles.values()}
    needed = max(0, target_profiles - len(profiles))
    if needed <= 0:
        return data

    prompt = build_profile_prompt(skills, profiles, needed)
    system = (
        "Eres un arquitecto curricular experto en perfiles profesionales. "
        "Respondes únicamente con JSON."
    )
    raw = call_llm(prompt, system)
    parsed = extract_json(raw)
    if not isinstance(parsed, list):
        raise ValueError("El LLM no devolvió una lista de perfiles.")

    next_idx = len(profiles) + 1
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = str(item.get("nombre", "")).strip()
        if not name or name in existing_profile_names:
            continue

        req = item.get("habilidades_requeridas", [])
        if isinstance(req, str):
            req = [req]
        req = [str(s).strip() for s in req if str(s).strip() in skills]
        if len(req) < 6:
            continue
        if len(req) > 10:
            req = req[:10]

        pid = str(item.get("id", "")).strip()
        if not pid:
            pid = f"perfil_{next_idx:02d}"
        while pid in profiles:
            next_idx += 1
            pid = f"perfil_{next_idx:02d}"

        profiles[pid] = {
            "nombre": name,
            "descripcion": str(item.get("descripcion", "")).strip() or f"Perfil profesional en {name}.",
            "habilidades_requeridas": req,
        }
        existing_profile_names.add(name)
        next_idx += 1
        if len(profiles) >= target_profiles:
            break

    if len(profiles) < target_profiles:
        print(
            f"⚠ Solo se generaron {len(profiles)} perfiles "
            f"de {target_profiles} solicitados."
        )

    data["perfiles_profesionales"] = profiles
    return data


def ensure_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    data["metadata"] = data.get("metadata", {})
    data["metadata"]["version"] = data["metadata"].get("version", "1.0")
    data["metadata"]["total_cursos"] = len(data.get("cursos", []))
    data["metadata"]["total_habilidades"] = len(data.get("habilidades", []))
    data["metadata"]["descripcion"] = (
        "Dataset ampliado procedimentalmente con apoyo de LLM para "
        "simulación de trayectorias profesionales"
    )
    return data


def optional_validate(dataset_path: Path, validator_path: Path | None):
    if validator_path is None or not validator_path.exists():
        return
    print(f"Validando con: {validator_path}")
    subprocess.run([sys.executable, str(validator_path), str(dataset_path)], check=True)


def main():
    load_env()

    BASE_DIR = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Genera un dataset sintético ampliado."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=BASE_DIR / "data" / "dataset.json"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "data" / "dataset_sintetico.json"
    )

    parser.add_argument("--courses", type=int, default=150)
    parser.add_argument("--profiles", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validator", type=Path, default=None)

    args = parser.parse_args()

    random.seed(args.seed)

    print(f"\nDataset origen : {args.input}")
    print(f"Dataset salida : {args.output}")

    if not args.input.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrada: {args.input}"
        )

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = generate_courses(
        data,
        args.courses,
        batch_size=max(1, args.batch_size)
    )

    data = generate_profiles(
        data,
        args.profiles
    )

    data = ensure_metadata(data)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Dataset generado: {args.output}")
    print(f"  - Cursos: {len(data['cursos'])}")
    print(f"  - Habilidades: {len(data['habilidades'])}")
    print(f"  - Perfiles: {len(data['perfiles_profesionales'])}")

    if args.validate:
        validator = args.validator

        if validator is None:
            candidates = [
                BASE_DIR / "scripts" / "validate_dataset.py",
                Path(__file__).resolve().parent / "validate_dataset.py",
                BASE_DIR / "validate_dataset.py",
            ]

            validator = next(
                (p for p in candidates if p.exists()),
                None
            )

        optional_validate(args.output, validator)


if __name__ == "__main__":
    main()
