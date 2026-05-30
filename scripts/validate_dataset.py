#!/usr/bin/env python3
"""
validate_dataset.py

Valida la integridad estructural y lógica de un dataset de cursos para el
planificador de trayectorias.

Comprobaciones:
  - Estructura JSON esperada
  - Metadatos coherentes
  - IDs únicos de cursos
  - Habilidades únicas
  - Cursos con campos obligatorios
  - Prerrequisitos y habilidades referencian habilidades existentes
  - Detecta habilidades huérfanas
  - Construye un grafo de dependencia entre cursos y comprueba que sea DAG
  - Verifica que exista al menos una ruta razonable hacia cada perfil

Uso:
  python validate_dataset.py --input data/dataset.json
  python validate_dataset.py --input data/dataset_synthetic.json

Salida:
  - Imprime diagnóstico en consola
  - Devuelve código 0 si todo está correcto
  - Devuelve código 1 si encuentra errores
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


REQUIRED_TOP_KEYS = {
    "metadata",
    "habilidades",
    "cursos",
    "perfiles_profesionales",
}


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("El archivo JSON raíz debe ser un objeto/diccionario.")
    return data


def is_non_empty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def validate_structure(data: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    missing = REQUIRED_TOP_KEYS - set(data.keys())
    if missing:
        errors.append(f"Faltan claves obligatorias en el JSON: {sorted(missing)}")

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("'metadata' debe ser un objeto/diccionario.")
        return

    habilidades = data.get("habilidades")
    cursos = data.get("cursos")
    perfiles = data.get("perfiles_profesionales")

    if not isinstance(habilidades, list):
        errors.append("'habilidades' debe ser una lista.")
    if not isinstance(cursos, list):
        errors.append("'cursos' debe ser una lista.")
    if not isinstance(perfiles, dict):
        errors.append("'perfiles_profesionales' debe ser un diccionario.")

    if isinstance(metadata, dict):
        if "total_cursos" in metadata and isinstance(cursos, list):
            if metadata["total_cursos"] != len(cursos):
                warnings.append(
                    f"metadata.total_cursos={metadata['total_cursos']} no coincide con len(cursos)={len(cursos)}."
                )
        if "total_habilidades" in metadata and isinstance(habilidades, list):
            if metadata["total_habilidades"] != len(habilidades):
                warnings.append(
                    f"metadata.total_habilidades={metadata['total_habilidades']} no coincide con len(habilidades)={len(habilidades)}."
                )


def validate_habilidades(data: dict[str, Any], errors: list[str], warnings: list[str]) -> set[str]:
    habilidades = data.get("habilidades", [])
    if not isinstance(habilidades, list):
        return set()

    seen = set()
    dup = set()
    for i, h in enumerate(habilidades):
        if not is_non_empty_str(h):
            errors.append(f"Habilidad inválida en índice {i}: debe ser string no vacío.")
            continue
        if h in seen:
            dup.add(h)
        seen.add(h)

    if dup:
        errors.append(f"Habilidades duplicadas detectadas: {sorted(dup)}")

    return seen


def validate_perfiles(
    data: dict[str, Any],
    habilidades_set: set[str],
    errors: list[str],
    warnings: list[str],
) -> dict[str, set[str]]:
    perfiles = data.get("perfiles_profesionales", {})
    if not isinstance(perfiles, dict):
        return {}

    result: dict[str, set[str]] = {}
    for pid, perfil in perfiles.items():
        if not is_non_empty_str(pid):
            errors.append("Se encontró un ID de perfil vacío o inválido.")
            continue
        if not isinstance(perfil, dict):
            errors.append(f"El perfil '{pid}' debe ser un diccionario.")
            continue

        req = perfil.get("habilidades_requeridas")
        if not isinstance(req, list):
            errors.append(f"El perfil '{pid}' debe tener 'habilidades_requeridas' como lista.")
            continue

        req_set: set[str] = set()
        for h in req:
            if not is_non_empty_str(h):
                errors.append(f"Perfil '{pid}' tiene una habilidad requerida inválida: {h!r}")
                continue
            req_set.add(h)
            if h not in habilidades_set:
                errors.append(
                    f"Perfil '{pid}' requiere la habilidad '{h}', pero no existe en el catálogo de habilidades."
                )

        result[pid] = req_set

    if not perfiles:
        warnings.append("No hay perfiles profesionales en el dataset.")

    return result


def validate_courses(
    data: dict[str, Any],
    habilidades_set: set[str],
    errors: list[str],
    warnings: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    cursos = data.get("cursos", [])
    if not isinstance(cursos, list):
        return {}, {}

    by_id: dict[str, dict[str, Any]] = {}
    skill_to_courses: dict[str, list[str]] = defaultdict(list)

    required_fields = {
        "id",
        "nombre",
        "descripcion",
        "prerrequisitos",
        "habilidades",
        "duracion_semanas",
        "nivel",
    }

    allowed_niveles = {"principiante", "intermedio", "avanzado"}

    for idx, curso in enumerate(cursos):
        if not isinstance(curso, dict):
            errors.append(f"Curso en índice {idx} no es un objeto/diccionario.")
            continue

        missing = required_fields - set(curso.keys())
        if missing:
            errors.append(
                f"Curso en índice {idx} ('{curso.get('id', 'sin_id')}') tiene campos faltantes: {sorted(missing)}"
            )
            continue

        cid = curso.get("id")
        nombre = curso.get("nombre")
        descripcion = curso.get("descripcion")
        prereq = curso.get("prerrequisitos")
        teaches = curso.get("habilidades")
        duracion = curso.get("duracion_semanas")
        nivel = curso.get("nivel")

        if not is_non_empty_str(cid):
            errors.append(f"Curso en índice {idx} tiene un 'id' inválido: {cid!r}")
            continue
        if cid in by_id:
            errors.append(f"ID de curso duplicado detectado: '{cid}'")
            continue
        by_id[cid] = curso

        if not is_non_empty_str(nombre):
            errors.append(f"Curso '{cid}' tiene un 'nombre' inválido.")
        if not is_non_empty_str(descripcion):
            errors.append(f"Curso '{cid}' tiene una 'descripcion' inválida.")

        if not isinstance(prereq, list):
            errors.append(f"Curso '{cid}' debe tener 'prerrequisitos' como lista.")
            prereq = []
        if not isinstance(teaches, list):
            errors.append(f"Curso '{cid}' debe tener 'habilidades' como lista.")
            teaches = []

        if not isinstance(duracion, int) or duracion <= 0:
            errors.append(f"Curso '{cid}' tiene 'duracion_semanas' inválida: {duracion!r}")
        elif duracion > 80:
            warnings.append(f"Curso '{cid}' tiene una duración muy alta ({duracion} semanas).")

        if nivel not in allowed_niveles:
            errors.append(
                f"Curso '{cid}' tiene nivel inválido: {nivel!r}. "
                f"Permitidos: {sorted(allowed_niveles)}"
            )

        # Cursos deben enseñar al menos una habilidad
        if not teaches:
            warnings.append(f"Curso '{cid}' no enseña ninguna habilidad.")
        for h in teaches:
            if not is_non_empty_str(h):
                errors.append(f"Curso '{cid}' tiene una habilidad enseñada inválida: {h!r}")
                continue
            if h not in habilidades_set:
                errors.append(
                    f"Curso '{cid}' enseña la habilidad '{h}', pero no existe en el catálogo."
                )
            skill_to_courses[h].append(cid)

        # Prerrequisitos deben referir habilidades existentes
        for h in prereq:
            if not is_non_empty_str(h):
                errors.append(f"Curso '{cid}' tiene un prerrequisito inválido: {h!r}")
                continue
            if h not in habilidades_set:
                errors.append(
                    f"Curso '{cid}' requiere la habilidad '{h}', pero no existe en el catálogo."
                )

        # Evitar que un curso se autodependa por enseñar una habilidad que también requiere
        inter = set(prereq) & set(teaches)
        if inter:
            warnings.append(
                f"Curso '{cid}' comparte prerrequisitos y habilidades impartidas: {sorted(inter)}"
            )

    return by_id, skill_to_courses


def build_course_dependency_graph(
    cursos_by_id: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    """
    Grafo dirigido entre cursos:
      A -> B si A enseña alguna habilidad que B requiere como prerrequisito.
    """
    skill_to_courses: dict[str, list[str]] = defaultdict(list)
    for cid, curso in cursos_by_id.items():
        for h in curso.get("habilidades", []):
            skill_to_courses[h].append(cid)

    graph: dict[str, set[str]] = {cid: set() for cid in cursos_by_id}
    for cid, curso in cursos_by_id.items():
        prereqs = set(curso.get("prerrequisitos", []))
        for skill in prereqs:
            for provider in skill_to_courses.get(skill, []):
                if provider != cid:
                    graph[provider].add(cid)
    return graph


def detect_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    indeg = {u: 0 for u in graph}
    for u, neigh in graph.items():
        for v in neigh:
            indeg[v] = indeg.get(v, 0) + 1

    q = deque([u for u, d in indeg.items() if d == 0])
    visited = 0
    while q:
        u = q.popleft()
        visited += 1
        for v in graph.get(u, set()):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if visited == len(graph):
        return []

    # Recuperar nodos en ciclo aproximado
    cyclic_nodes = [u for u, d in indeg.items() if d > 0]
    return [cyclic_nodes]


def detect_orphan_skills(
    habilidades_set: set[str],
    cursos_by_id: dict[str, dict[str, Any]],
    perfiles_req: dict[str, set[str]],
) -> tuple[set[str], set[str]]:
    taught: set[str] = set()
    required: set[str] = set()

    for curso in cursos_by_id.values():
        taught.update(curso.get("habilidades", []))
        required.update(curso.get("prerrequisitos", []))

    for req in perfiles_req.values():
        required.update(req)

    orphan = habilidades_set - (taught | required)
    teach_only = taught - required
    return orphan, teach_only


def path_to_skill_exists(
    target_skill: str,
    cursos_by_id: dict[str, dict[str, Any]],
) -> bool:
    """
    Comprueba si existe al menos un curso que enseñe la habilidad.
    Es una condición mínima para que la habilidad pueda alcanzarse.
    """
    for curso in cursos_by_id.values():
        if target_skill in curso.get("habilidades", []):
            return True
    return False


def validate_reachability(
    perfiles_req: dict[str, set[str]],
    cursos_by_id: dict[str, dict[str, Any]],
    warnings: list[str],
) -> None:
    for pid, reqs in perfiles_req.items():
        missing = [s for s in reqs if not path_to_skill_exists(s, cursos_by_id)]
        if missing:
            warnings.append(
                f"El perfil '{pid}' requiere habilidades sin curso que las imparta: {missing}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida un dataset de cursos.")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/dataset.json"),
        help="Ruta del archivo JSON del dataset.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Convierte warnings en error de validación.",
    )
    args = parser.parse_args()

    path = args.input
    if not path.exists():
        print(f"✗ No existe el archivo: {path}", file=sys.stderr)
        return 1

    errors: list[str] = []
    warnings: list[str] = []

    try:
        data = load_dataset(path)
    except Exception as e:
        print(f"✗ Error leyendo JSON: {e}", file=sys.stderr)
        return 1

    validate_structure(data, errors, warnings)
    habilidades_set = validate_habilidades(data, errors, warnings)
    perfiles_req = validate_perfiles(data, habilidades_set, errors, warnings)
    cursos_by_id, _ = validate_courses(data, habilidades_set, errors, warnings)

    orphan, teach_only = detect_orphan_skills(habilidades_set, cursos_by_id, perfiles_req)
    if orphan:
        warnings.append(
            f"Habilidades definidas pero no usadas ni requeridas por ningún perfil: {sorted(orphan)}"
        )
    if teach_only:
        warnings.append(
            f"Habilidades enseñadas por cursos pero no exigidas por ningún perfil: {sorted(teach_only)}"
        )

    validate_reachability(perfiles_req, cursos_by_id, warnings)

    graph = build_course_dependency_graph(cursos_by_id)
    cycles = detect_cycles(graph)
    if cycles:
        errors.append(
            "Se detectó un ciclo en el grafo de dependencia entre cursos derivado de prerrequisitos."
        )

    # Resumen
    print("=" * 72)
    print(f"VALIDACIÓN DEL DATASET: {path}")
    print("=" * 72)
    print(f"Cursos: {len(cursos_by_id)}")
    print(f"Habilidades: {len(habilidades_set)}")
    print(f"Perfiles: {len(perfiles_req)}")
    print(f"Dependencias entre cursos: {sum(len(v) for v in graph.values())}")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\nERRORES:")
        for e in errors:
            print(f"  - {e}")
        print("\n✗ Dataset inválido.")
        return 1 if (errors or (args.strict and warnings)) else 0

    if args.strict and warnings:
        print("\n✗ Validación estricta fallida por warnings.")
        return 1

    print("\n✓ Dataset válido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
