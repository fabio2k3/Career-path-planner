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


# *** Rutas del proyecto ******

# Directorio raíz del proyecto, calculado desde la ubicación de este archivo.
BASE_DIR = Path(__file__).resolve().parent.parent

# Se prioriza el dataset sintético, y si no existe se utiliza el dataset base.
_SYNTH = BASE_DIR / "data" / "dataset_sintetico.json"
_BASE = BASE_DIR / "data" / "dataset.json"


def _default_input() -> Path:
    """
    Devuelve el dataset de entrada por defecto.

    Se prioriza el dataset sintético si existe, ya que suele ser el objetivo
    principal de corrección durante el proceso de limpieza.
    """
    return _SYNTH if _SYNTH.exists() else _BASE


# *** Detección de ciclos *******

def _construir_grafo_cursos(cursos: list) -> tuple[dict, dict]:
    """
    Construye el grafo de dependencia entre cursos.

    La arista A → B significa que el curso A enseña una habilidad que el curso B
    necesita como prerrequisito. También devuelve el grado de entrada de cada nodo.

    Returns
    -------
    tuple[dict, dict]
        - grafo: curso_id -> conjunto de cursos dependientes
        - in_degree: curso_id -> número de entradas
    """
    # Mapa habilidad -> lista de cursos que la enseñan.
    skill_to_courses: dict[str, list[str]] = defaultdict(list)
    for c in cursos:
        for h in c["habilidades"]:
            skill_to_courses[h].append(c["id"])

    grafo: dict[str, set[str]] = defaultdict(set)
    in_degree: dict[str, int] = {c["id"]: 0 for c in cursos}

    # Para cada prerrequisito, conectamos el curso proveedor con el curso que lo requiere.
    for curso in cursos:
        for prereq_hab in curso["prerrequisitos"]:
            for proveedor_id in skill_to_courses.get(prereq_hab, []):
                if proveedor_id != curso["id"]:
                    if curso["id"] not in grafo[proveedor_id]:
                        grafo[proveedor_id].add(curso["id"])
                        in_degree[curso["id"]] += 1

    return grafo, in_degree


def _nodos_en_ciclo(grafo: dict, in_degree: dict) -> set[str]:
    """
    Devuelve el conjunto de IDs de cursos que permanecen en ciclos.

    Se utiliza un enfoque tipo Kahn:
    - se eliminan los nodos con grado de entrada 0;
    - lo que queda al final pertenece a algún ciclo.
    """
    grado = dict(in_degree)
    cola = deque(cid for cid, d in grado.items() if d == 0)

    while cola:
        nodo = cola.popleft()
        for vecino in grafo.get(nodo, set()):
            grado[vecino] -= 1
            if grado[vecino] == 0:
                cola.append(vecino)

    return {cid for cid, d in grado.items() if d > 0}


# *** Rotura de ciclos *******

def romper_ciclos(cursos: list, verbose: bool = True) -> tuple[list, list[dict]]:
    """
    Elimina iterativamente los prerrequisitos que generan ciclos.

    Estrategia:
    - detectar los nodos que aún forman parte de ciclos;
    - localizar un prerrequisito conflictivo en uno de esos cursos;
    - eliminarlo;
    - repetir hasta obtener un DAG.

    Parameters
    ----------
    cursos : list
        Lista de cursos del dataset.
    verbose : bool
        Se conserva por compatibilidad, aunque el proceso ya no imprime
        mensajes internos de depuración.

    Returns
    -------
    tuple[list, list[dict]]
        - cursos_limpios: lista de cursos modificada in-place.
        - log_cambios: historial de prerrequisitos eliminados.
    """
    log: list[dict] = []
    max_iter = len(cursos) * 2  # Límite de seguridad para evitar bucles infinitos.

    for iteracion in range(max_iter):
        # Mapa habilidad -> cursos que la enseñan, necesario para detectar conflictos.
        skill_to_courses: dict[str, list[str]] = defaultdict(list)
        for c in cursos:
            for h in c["habilidades"]:
                skill_to_courses[h].append(c["id"])

        grafo, in_degree = _construir_grafo_cursos(cursos)
        en_ciclo = _nodos_en_ciclo(grafo, in_degree)

        # Si no quedan nodos en ciclo, el grafo ya es válido.
        if not en_ciclo:
            break

        # Intento 1: buscar un prerrequisito conflictivo en un curso que pertenezca al ciclo.
        eliminado = False
        for curso in cursos:
            if curso["id"] not in en_ciclo:
                continue

            for prereq_hab in list(curso["prerrequisitos"]):
                proveedores = skill_to_courses.get(prereq_hab, [])

                # Un prerrequisito es conflictivo si la habilidad proviene de otro curso también en ciclo.
                conflictivo = any(
                    p in en_ciclo and p != curso["id"]
                    for p in proveedores
                )

                if conflictivo:
                    curso["prerrequisitos"].remove(prereq_hab)
                    log.append(
                        {
                            "iteracion": iteracion + 1,
                            "curso_id": curso["id"],
                            "curso_nombre": curso.get("nombre", "?"),
                            "prereq_hab": prereq_hab,
                        }
                    )
                    eliminado = True
                    break

            if eliminado:
                break

        # Intento 2: si no se encontró conflicto explícito, eliminar un prerrequisito cualquiera
        # de un curso que todavía esté en ciclo.
        if not eliminado:
            for curso in cursos:
                if curso["id"] in en_ciclo and curso["prerrequisitos"]:
                    prereq_hab = curso["prerrequisitos"][0]
                    curso["prerrequisitos"].remove(prereq_hab)
                    log.append(
                        {
                            "iteracion": iteracion + 1,
                            "curso_id": curso["id"],
                            "curso_nombre": curso.get("nombre", "?"),
                            "prereq_hab": prereq_hab,
                            "forzado": True,
                        }
                    )
                    break

    return cursos, log


# **** Main ****

def main() -> None:
    """
    Punto de entrada del script.

    - Carga el dataset de entrada.
    - Detecta ciclos.
    - Si se solicita, solo muestra diagnóstico (--dry-run).
    - Si no, elimina prerrequisitos conflictivos y guarda el resultado.
    """
    parser = argparse.ArgumentParser(
        description="Elimina ciclos del DAG en un dataset de cursos."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=_default_input(),
        help="Dataset a limpiar (default: dataset_sintetico.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
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

    # Diagnóstico inicial para saber si el dataset ya es un DAG válido.
    grafo0, in_degree0 = _construir_grafo_cursos(cursos)
    ciclo0 = _nodos_en_ciclo(grafo0, in_degree0)

    print(f"\n  Cursos totales       : {len(cursos)}")
    print(f"  Cursos en ciclo      : {len(ciclo0)}")

    if not ciclo0:
        print("\n  ✓ El dataset ya es un DAG válido. No hay nada que corregir.")
        return

    if args.dry_run:
        print(f"\n  Cursos afectados: {sorted(ciclo0)}")
        print("\n  (--dry-run: no se modificó el archivo)")
        return

    print("\n  Rompiendo ciclos...")
    cursos_limpios, log = romper_ciclos(cursos, verbose=True)

    # Diagnóstico final tras la limpieza.
    grafo1, in_degree1 = _construir_grafo_cursos(cursos_limpios)
    ciclo1 = _nodos_en_ciclo(grafo1, in_degree1)

    print(f"\n  Prerrequisitos eliminados : {len(log)}")
    print(f"  Cursos en ciclo tras fix  : {len(ciclo1)}")

    if ciclo1:
        print(f"  ⚠ Aún quedan ciclos: {sorted(ciclo1)}")
        print("    Ejecuta el script de nuevo para continuar limpiando.")
    else:
        print("  ✓ DAG válido. No quedan ciclos.")

    # Guardar el dataset corregido.
    data["cursos"] = cursos_limpios
    with open(args.input, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ Dataset guardado en: {args.input}")

    # Validación opcional con el script oficial del proyecto.
    validator = BASE_DIR / "scripts" / "validate_dataset.py"
    if validator.exists():
        import subprocess
        import sys

        print("\n  Ejecutando validación final...")
        subprocess.run(
            [sys.executable, str(validator), "--input", str(args.input)],
            capture_output=False,
        )

    print("=" * 60)


if __name__ == "__main__":
    main()