"""
fix_dag.py

Limpia el dataset_sintetico.json eliminando los prerrequisitos
que causan ciclos en el grafo de dependencias entre cursos.

Uso:
    python scripts/fix_dag.py
    python scripts/fix_dag.py --input data/dataset_sintetico.json
    python scripts/fix_dag.py --input data/dataset_sintetico.json --dry-run
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
_SYNTH   = BASE_DIR / "data" / "dataset_sintetico.json"
_BASE    = BASE_DIR / "data" / "dataset.json"


def _default_input() -> Path:
    return _SYNTH if _SYNTH.exists() else _BASE


# ── Detección de ciclos ───────────────────────────────────────────────────────

def _construir_grafo_cursos(cursos: list) -> tuple[dict, dict]:
    """
    Construye el grafo de dependencia entre cursos.
    A → B significa: A enseña algo que B requiere (A debe preceder a B).
    Devuelve (grafo, in_degree).
    """
    skill_to_courses: dict[str, list[str]] = defaultdict(list)
    for c in cursos:
        for h in c["habilidades"]:
            skill_to_courses[h].append(c["id"])

    grafo:     dict[str, set[str]] = defaultdict(set)
    in_degree: dict[str, int]      = {c["id"]: 0 for c in cursos}

    for curso in cursos:
        for prereq_hab in curso["prerrequisitos"]:
            for proveedor_id in skill_to_courses.get(prereq_hab, []):
                if proveedor_id != curso["id"]:
                    if curso["id"] not in grafo[proveedor_id]:
                        grafo[proveedor_id].add(curso["id"])
                        in_degree[curso["id"]] += 1

    return grafo, in_degree


def _nodos_en_ciclo(grafo: dict, in_degree: dict) -> set[str]:
    """Devuelve el conjunto de IDs de cursos involucrados en ciclos."""
    grado = dict(in_degree)
    cola  = deque(cid for cid, d in grado.items() if d == 0)
    while cola:
        nodo = cola.popleft()
        for vecino in grafo.get(nodo, set()):
            grado[vecino] -= 1
            if grado[vecino] == 0:
                cola.append(vecino)
    return {cid for cid, d in grado.items() if d > 0}


# ── Rotura de ciclos ──────────────────────────────────────────────────────────

def romper_ciclos(cursos: list, verbose: bool = True) -> tuple[list, list[dict]]:
    """
    Elimina iterativamente los prerrequisitos que causan ciclos.

    Estrategia: en cada iteración identifica los nodos en ciclo,
    toma el primero que tenga un prerrequisito conflictivo y elimina
    ese prerrequisito. Repite hasta que el grafo sea un DAG.

    Devuelve (cursos_limpios, log_cambios).
    """
    log: list[dict] = []
    max_iter = len(cursos) * 2  # límite de seguridad

    for iteracion in range(max_iter):
        skill_to_courses: dict[str, list[str]] = defaultdict(list)
        for c in cursos:
            for h in c["habilidades"]:
                skill_to_courses[h].append(c["id"])

        grafo, in_degree = _construir_grafo_cursos(cursos)
        en_ciclo         = _nodos_en_ciclo(grafo, in_degree)

        if not en_ciclo:
            break

        # Buscar el primer prerrequisito conflictivo en un curso del ciclo
        eliminado = False
        for curso in cursos:
            if curso["id"] not in en_ciclo:
                continue

            for prereq_hab in list(curso["prerrequisitos"]):
                proveedores = skill_to_courses.get(prereq_hab, [])
                # Conflictivo si el proveedor también está en el ciclo
                conflictivo = any(
                    p in en_ciclo and p != curso["id"]
                    for p in proveedores
                )
                if conflictivo:
                    curso["prerrequisitos"].remove(prereq_hab)
                    entrada = {
                        "iteracion":    iteracion + 1,
                        "curso_id":     curso["id"],
                        "curso_nombre": curso.get("nombre", "?"),
                        "prereq_hab":   prereq_hab,
                    }
                    log.append(entrada)
                    if verbose:
                        print(f"    [{iteracion+1}] Eliminado prereq '{prereq_hab}' "
                              f"de curso '{curso['id']}' ({curso.get('nombre', '?')})")
                    eliminado = True
                    break

            if eliminado:
                break

        if not eliminado:
            # No se encontró prerrequisito conflictivo explícito:
            # eliminar el primero disponible del primer curso en ciclo
            for curso in cursos:
                if curso["id"] in en_ciclo and curso["prerrequisitos"]:
                    prereq_hab = curso["prerrequisitos"][0]
                    curso["prerrequisitos"].remove(prereq_hab)
                    entrada = {
                        "iteracion":    iteracion + 1,
                        "curso_id":     curso["id"],
                        "curso_nombre": curso.get("nombre", "?"),
                        "prereq_hab":   prereq_hab,
                        "forzado":      True,
                    }
                    log.append(entrada)
                    if verbose:
                        print(f"    [{iteracion+1}] Eliminado prereq (forzado) "
                              f"'{prereq_hab}' de curso '{curso['id']}'")
                    break

    return cursos, log


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elimina ciclos del DAG en un dataset de cursos."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=_default_input(),
        help="Dataset a limpiar (default: dataset_sintetico.json).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Solo detecta ciclos sin modificar el archivo.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"✗ No existe: {args.input}")
        raise SystemExit(1)

    print("=" * 60)
    print("  FIX DAG — Career Path Planner")
    print("=" * 60)
    print(f"  Archivo: {args.input.name}")

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    cursos = data["cursos"]

    # Diagnóstico inicial
    grafo0, in_degree0 = _construir_grafo_cursos(cursos)
    ciclo0             = _nodos_en_ciclo(grafo0, in_degree0)

    print(f"\n  Cursos totales       : {len(cursos)}")
    print(f"  Cursos en ciclo      : {len(ciclo0)}")

    if not ciclo0:
        print("\n  ✓ El dataset ya es un DAG válido. No hay nada que corregir.")
        return

    if args.dry_run:
        print(f"\n  Cursos afectados: {sorted(ciclo0)}")
        print("\n  (--dry-run: no se modificó el archivo)")
        return

    print(f"\n  Rompiendo ciclos...")
    cursos_limpios, log = romper_ciclos(cursos, verbose=True)

    # Diagnóstico final
    grafo1, in_degree1 = _construir_grafo_cursos(cursos_limpios)
    ciclo1             = _nodos_en_ciclo(grafo1, in_degree1)

    print(f"\n  Prerrequisitos eliminados : {len(log)}")
    print(f"  Cursos en ciclo tras fix  : {len(ciclo1)}")

    if ciclo1:
        print(f"  ⚠ Aún quedan ciclos: {sorted(ciclo1)}")
        print("    Ejecuta el script de nuevo para continuar limpiando.")
    else:
        print("  ✓ DAG válido. No quedan ciclos.")

    # Guardar
    data["cursos"] = cursos_limpios
    with open(args.input, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ Dataset guardado en: {args.input}")

    # Validar con el validador oficial
    validator = BASE_DIR / "scripts" / "validate_dataset.py"
    if validator.exists():
        import subprocess, sys
        print(f"\n  Ejecutando validación final...")
        subprocess.run(
            [sys.executable, str(validator), "--input", str(args.input)],
            capture_output=False,
        )

    print("=" * 60)


if __name__ == "__main__":
    main()