"""
validate_dataset.py  (scripts/)

Validación completa y portable de cualquier dataset del proyecto.
Acepta --input para apuntar a cualquier archivo JSON.
Por defecto busca dataset_sintetico.json, luego dataset.json.

Uso:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --input data/dataset.json
    python scripts/validate_dataset.py --input data/dataset_sintetico.json --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

# *** Constantes *******

# Claves mínimas que debe contener el JSON raíz del dataset.
REQUIRED_TOP_KEYS = {"metadata", "habilidades", "cursos", "perfiles_profesionales"}


# *** Carga ********

def load_dataset(path: Path) -> dict:
    """
    Carga un dataset desde disco y verifica que el JSON raíz sea un objeto.

    Parameters
    ----------
    path : Path
        Ruta del archivo JSON a validar.

    Returns
    -------
    dict
        Contenido del dataset cargado.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("El JSON raíz debe ser un objeto/diccionario.")

    return data


def _is_non_empty_str(x) -> bool:
    """Indica si x es una cadena no vacía después de eliminar espacios."""
    return isinstance(x, str) and bool(x.strip())


# *** Validaciones estructurales *******

def validate_structure(data: dict, errors: list[str], warnings: list[str],) -> None:
    """
    Verifica que la estructura general del dataset tenga las claves y tipos esperados.
    """
    missing = REQUIRED_TOP_KEYS - set(data.keys())
    if missing:
        errors.append(f"Faltan claves obligatorias: {sorted(missing)}")

    if not isinstance(data.get("metadata"), dict):
        errors.append("'metadata' debe ser un diccionario.")
    if not isinstance(data.get("habilidades"), list):
        errors.append("'habilidades' debe ser una lista.")
    if not isinstance(data.get("cursos"), list):
        errors.append("'cursos' debe ser una lista.")
    if not isinstance(data.get("perfiles_profesionales"), dict):
        errors.append("'perfiles_profesionales' debe ser un diccionario.")

    meta = data.get("metadata", {})
    cursos = data.get("cursos", [])
    habs = data.get("habilidades", [])

    # Comprobaciones suaves de consistencia entre metadatos y contenido real.
    if isinstance(meta, dict) and isinstance(cursos, list):
        if "total_cursos" in meta and meta["total_cursos"] != len(cursos):
            warnings.append(
                f"metadata.total_cursos={meta['total_cursos']} ≠ len(cursos)={len(cursos)}."
            )

    if isinstance(meta, dict) and isinstance(habs, list):
        if "total_habilidades" in meta and meta["total_habilidades"] != len(habs):
            warnings.append(
                f"metadata.total_habilidades={meta['total_habilidades']} ≠ len(habilidades)={len(habs)}."
            )


def validate_habilidades(data: dict, errors: list[str], warnings: list[str],) -> set[str]:
    """
    Valida la lista global de habilidades del dataset.

    Returns
    -------
    set[str]
        Conjunto de habilidades válidas detectadas.
    """
    habilidades = data.get("habilidades", [])
    if not isinstance(habilidades, list):
        return set()

    seen: set[str] = set()
    dups: set[str] = set()

    for i, h in enumerate(habilidades):
        if not _is_non_empty_str(h):
            errors.append(f"Habilidad inválida en índice {i}: debe ser string no vacío.")
            continue
        if h in seen:
            dups.add(h)
        seen.add(h)

    if dups:
        errors.append(f"Habilidades duplicadas: {sorted(dups)}")

    return seen


def validate_perfiles(data: dict, habilidades_set: set[str], errors: list[str], warnings: list[str],) -> dict[str, set[str]]:
    """
    Valida la sección de perfiles profesionales y sus dependencias sobre habilidades.

    Returns
    -------
    dict[str, set[str]]
        Mapa perfil_id -> conjunto de habilidades requeridas.
    """
    perfiles = data.get("perfiles_profesionales", {})
    if not isinstance(perfiles, dict):
        return {}

    result: dict[str, set[str]] = {}

    for pid, perfil in perfiles.items():
        if not _is_non_empty_str(pid):
            errors.append("Se encontró un ID de perfil vacío o inválido.")
            continue

        if not isinstance(perfil, dict):
            errors.append(f"Perfil '{pid}' debe ser un diccionario.")
            continue

        req = perfil.get("habilidades_requeridas")
        if not isinstance(req, list):
            errors.append(
                f"Perfil '{pid}': 'habilidades_requeridas' debe ser una lista."
            )
            continue

        req_set: set[str] = set()
        for h in req:
            if not _is_non_empty_str(h):
                errors.append(f"Perfil '{pid}': habilidad inválida: {h!r}")
                continue

            req_set.add(h)
            if h not in habilidades_set:
                errors.append(
                    f"Perfil '{pid}': habilidad '{h}' no existe en el catálogo."
                )

        result[pid] = req_set

    if not perfiles:
        warnings.append("No hay perfiles profesionales en el dataset.")

    return result


def validate_courses(data: dict, habilidades_set: set[str], errors: list[str], warnings: list[str],) -> dict[str, dict]:
    """
    Valida la lista de cursos, sus campos obligatorios y su coherencia interna.

    Returns
    -------
    dict[str, dict]
        Mapa curso_id -> curso.
    """
    cursos = data.get("cursos", [])
    if not isinstance(cursos, list):
        return {}

    by_id: dict[str, dict] = {}
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
            errors.append(f"Curso en índice {idx} no es un diccionario.")
            continue

        missing = required_fields - set(curso.keys())
        if missing:
            errors.append(
                f"Curso índice {idx} ('{curso.get('id', 'sin_id')}'): "
                f"campos faltantes: {sorted(missing)}"
            )
            continue

        cid = curso.get("id")
        if not _is_non_empty_str(cid):
            errors.append(f"Curso índice {idx}: 'id' inválido: {cid!r}")
            continue

        if cid in by_id:
            errors.append(f"ID de curso duplicado: '{cid}'")
            continue
        by_id[cid] = curso

        if not _is_non_empty_str(curso.get("nombre")):
            errors.append(f"Curso '{cid}': 'nombre' inválido.")
        if not _is_non_empty_str(curso.get("descripcion")):
            errors.append(f"Curso '{cid}': 'descripcion' inválida.")

        prereq = curso.get("prerrequisitos", [])
        teaches = curso.get("habilidades", [])
        dur = curso.get("duracion_semanas")
        nivel = curso.get("nivel")

        if not isinstance(prereq, list):
            errors.append(f"Curso '{cid}': 'prerrequisitos' debe ser lista.")
            prereq = []
        if not isinstance(teaches, list):
            errors.append(f"Curso '{cid}': 'habilidades' debe ser lista.")
            teaches = []

        if not isinstance(dur, int) or dur <= 0:
            errors.append(f"Curso '{cid}': 'duracion_semanas' inválida: {dur!r}")
        elif dur > 80:
            warnings.append(f"Curso '{cid}': duración muy alta ({dur} semanas).")

        if nivel not in allowed_niveles:
            errors.append(
                f"Curso '{cid}': nivel inválido '{nivel}'. "
                f"Permitidos: {sorted(allowed_niveles)}"
            )

        if not teaches:
            warnings.append(f"Curso '{cid}' no enseña ninguna habilidad.")

        for h in teaches:
            if not _is_non_empty_str(h):
                errors.append(f"Curso '{cid}': habilidad enseñada inválida: {h!r}")
            elif h not in habilidades_set:
                errors.append(
                    f"Curso '{cid}': enseña '{h}' que no existe en el catálogo."
                )

        for h in prereq:
            if not _is_non_empty_str(h):
                errors.append(f"Curso '{cid}': prerrequisito inválido: {h!r}")
            elif h not in habilidades_set:
                errors.append(
                    f"Curso '{cid}': requiere '{h}' que no existe en el catálogo."
                )

        inter = set(prereq) & set(teaches)
        if inter:
            warnings.append(
                f"Curso '{cid}': mismas habilidades en prereqs y enseñadas: "
                f"{sorted(inter)}"
            )

    return by_id


def detect_cycles(graph: dict[str, set[str]]) -> list[str]:
    """
    Detecta ciclos en un grafo dirigido usando el algoritmo de Kahn.

    Returns
    -------
    list[str]
        Lista de nodos involucrados en ciclos. Vacía si el grafo es DAG.
    """
    indeg = {u: 0 for u in graph}
    for u, neigh in graph.items():
        for v in neigh:
            indeg[v] = indeg.get(v, 0) + 1

    q = deque(u for u, d in indeg.items() if d == 0)
    visited = 0

    while q:
        u = q.popleft()
        visited += 1
        for v in graph.get(u, set()):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    return [u for u, d in indeg.items() if d > 0] if visited < len(graph) else []


def build_course_dependency_graph(cursos_by_id: dict[str, dict]) -> dict[str, set[str]]:
    """
    Construye el grafo de dependencia entre cursos.

    Se agrega una arista A → B si el curso A enseña una habilidad que B requiere.
    """
    skill_to_courses: dict[str, list[str]] = defaultdict(list)
    for cid, curso in cursos_by_id.items():
        for h in curso.get("habilidades", []):
            skill_to_courses[h].append(cid)

    graph: dict[str, set[str]] = {cid: set() for cid in cursos_by_id}
    for cid, curso in cursos_by_id.items():
        for skill in curso.get("prerrequisitos", []):
            for provider in skill_to_courses.get(skill, []):
                if provider != cid:
                    graph[provider].add(cid)

    return graph


def validate_reachability(perfiles_req: dict[str, set[str]], cursos_by_id: dict[str, dict], warnings: list[str],) -> None:
    """
    Advierte si algún perfil requiere habilidades que ningún curso enseña.
    """
    taught = set()
    for curso in cursos_by_id.values():
        taught.update(curso.get("habilidades", []))

    for pid, reqs in perfiles_req.items():
        missing = [s for s in reqs if s not in taught]
        if missing:
            warnings.append(
                f"Perfil '{pid}': habilidades sin curso que las imparta: {missing}"
            )


# *** Resumen final *******

def main() -> int:
    """
    Ejecuta la validación completa del dataset y muestra un resumen final.
    """
    base_dir = Path(__file__).resolve().parent.parent
    synth = base_dir / "data" / "dataset.json"
    base = base_dir / "data" / "dataset_old.json"

    default_input = synth if synth.exists() else base

    parser = argparse.ArgumentParser(
        description="Validación completa de un dataset del proyecto."
    )
    parser.add_argument("--input", "-i", type=Path,
        default=default_input,
        help="Ruta del archivo JSON a validar.",
    )
    parser.add_argument("--strict", action="store_true",
        help="Tratar warnings como errores.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"✗ No existe el archivo: {args.input}", file=sys.stderr)
        return 1

    errors: list[str] = []
    warnings: list[str] = []

    try:
        data = load_dataset(args.input)
    except Exception as e:
        print(f"✗ Error leyendo JSON: {e}", file=sys.stderr)
        return 1

    validate_structure(data, errors, warnings)
    habilidades_set = validate_habilidades(data, errors, warnings)
    perfiles_req = validate_perfiles(data, habilidades_set, errors, warnings)
    cursos_by_id = validate_courses(data, habilidades_set, errors, warnings)

    # Habilidades definidas pero no usadas o no requeridas.
    taught = set()
    required = set()

    for curso in cursos_by_id.values():
        taught.update(curso.get("habilidades", []))
        required.update(curso.get("prerrequisitos", []))

    for req in perfiles_req.values():
        required.update(req)

    orphan = habilidades_set - (taught | required)
    teach_only = taught - required

    if orphan:
        warnings.append(
            f"Habilidades definidas pero no usadas ni requeridas: {sorted(orphan)}"
        )
    if teach_only:
        warnings.append(
            f"Habilidades enseñadas pero no requeridas por ningún perfil: "
            f"{sorted(teach_only)}"
        )

    validate_reachability(perfiles_req, cursos_by_id, warnings)

    graph = build_course_dependency_graph(cursos_by_id)
    cyclic = detect_cycles(graph)
    if cyclic:
        errors.append(
            f"Ciclo detectado en el grafo de dependencia entre cursos: {cyclic}"
        )

    # Resumen.
    print("=" * 72)
    print(f"VALIDACIÓN: {args.input.name}")
    print("=" * 72)
    print(f"  Cursos              : {len(cursos_by_id)}")
    print(f"  Habilidades         : {len(habilidades_set)}")
    print(f"  Perfiles            : {len(perfiles_req)}")
    print(f"  Dependencias cursos : {sum(len(v) for v in graph.values())}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠ {w}")

    if errors:
        print(f"\nERRORES ({len(errors)}):")
        for e in errors:
            print(f"  ✗ {e}")

    fallo = bool(errors) or (args.strict and bool(warnings))
    print()
    if fallo:
        print("✗ Dataset inválido.")
    else:
        print("✓ Dataset válido.")
    print("=" * 72)

    return 1 if fallo else 0


if __name__ == "__main__":
    raise SystemExit(main())