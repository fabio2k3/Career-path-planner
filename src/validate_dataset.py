"""
validate_dataset.py

Valida la integridad del dataset:
1. No existen ciclos en el grafo de prerrequisitos (es un DAG válido).
2. Todos los prerrequisitos referenciados existen en el conjunto de habilidades.
3. Todas las habilidades de los perfiles profesionales existen en el dataset.
4. Estadísticas generales del dataset.
"""

import json
from pathlib import Path
from collections import defaultdict, deque


# ***** Rutas *****
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "dataset.json"
INSTANCES_PATH = BASE_DIR / "data" / "instances" / "instances.json"


# ***** Carga de datos *****
def cargar_dataset():
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)

def cargar_instancias():
    with open(INSTANCES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ***** Validaciones *****

def validar_habilidades_existentes(dataset):
    """Verifica que todo prerrequisito referenciado existe en el pool de habilidades."""
    habilidades_validas = set(dataset["habilidades"])
    errores = []

    for curso in dataset["cursos"]:
        for pre in curso["prerrequisitos"]:
            if pre not in habilidades_validas:
                errores.append(
                    f"Curso '{curso['nombre']}' (id={curso['id']}): "
                    f"prerrequisito '{pre}' no existe en el pool de habilidades."
                )
        for hab in curso["habilidades"]:
            if hab not in habilidades_validas:
                errores.append(
                    f"Curso '{curso['nombre']}' (id={curso['id']}): "
                    f"habilidad enseñada '{hab}' no existe en el pool de habilidades."
                )
    return errores


def construir_grafo_habilidades(dataset):
    """
    Construye el grafo de dependencias entre habilidades.
    h_a → h_b significa: para adquirir h_b se necesita h_a como prerrequisito.
    """
    # Mapeo habilidad → cursos que la enseñan
    hab_a_cursos = defaultdict(list)
    for curso in dataset["cursos"]:
        for hab in curso["habilidades"]:
            hab_a_cursos[hab].append(curso)

    # Grafo: habilidad prereq → habilidades que la necesitan
    grafo = defaultdict(set)
    for curso in dataset["cursos"]:
        for hab_ensenada in curso["habilidades"]:
            for pre in curso["prerrequisitos"]:
                grafo[pre].add(hab_ensenada)

    return grafo


def detectar_ciclos_kahn(dataset):
    """
    Algoritmo de Kahn (topological sort) para detectar ciclos en el grafo de habilidades.
    Si el sort no puede procesar todos los nodos, hay un ciclo.
    """
    habilidades = set(dataset["habilidades"])

    # Construir grafo de habilidades con in-degree
    # Arista: prereq → habilidad_ensenada
    grafo = defaultdict(set)
    in_degree = defaultdict(int)

    for habilidad in habilidades:
        in_degree[habilidad] = 0  # inicializar

    for curso in dataset["cursos"]:
        for hab_ensenada in curso["habilidades"]:
            for pre in curso["prerrequisitos"]:
                if hab_ensenada not in grafo[pre]:
                    grafo[pre].add(hab_ensenada)
                    in_degree[hab_ensenada] += 1

    # Kahn: iniciar con nodos sin dependencias
    cola = deque([h for h in habilidades if in_degree[h] == 0])
    procesados = 0

    while cola:
        nodo = cola.popleft()
        procesados += 1
        for vecino in grafo[nodo]:
            in_degree[vecino] -= 1
            if in_degree[vecino] == 0:
                cola.append(vecino)

    if procesados < len(habilidades):
        # Identificar nodos en el ciclo
        en_ciclo = [h for h in habilidades if in_degree[h] > 0]
        return False, en_ciclo
    return True, []


def validar_perfiles(dataset):
    """Verifica que las habilidades de cada perfil existen en el pool."""
    habilidades_validas = set(dataset["habilidades"])
    errores = []

    for perfil_id, perfil in dataset["perfiles_profesionales"].items():
        for hab in perfil["habilidades_requeridas"]:
            if hab not in habilidades_validas:
                errores.append(
                    f"Perfil '{perfil_id}': habilidad requerida '{hab}' "
                    f"no existe en el pool de habilidades."
                )
    return errores


def validar_instancias(instancias, dataset):
    """Verifica que las habilidades iniciales de cada instancia existen."""
    habilidades_validas = set(dataset["habilidades"])
    perfiles_validos = set(dataset["perfiles_profesionales"].keys())
    errores = []

    for inst in instancias:
        for hab in inst["habilidades_iniciales"]:
            if hab not in habilidades_validas:
                errores.append(
                    f"Instancia '{inst['id']}': habilidad inicial '{hab}' "
                    f"no existe en el pool."
                )
        if inst["perfil_objetivo"] not in perfiles_validos:
            errores.append(
                f"Instancia '{inst['id']}': perfil objetivo "
                f"'{inst['perfil_objetivo']}' no existe."
            )
    return errores


def verificar_alcanzabilidad(dataset):
    """
    Verifica que cada habilidad del dataset puede ser adquirida
    (existe al menos un curso que la enseña).
    """
    habilidades_enseñadas = set()
    for curso in dataset["cursos"]:
        habilidades_enseñadas.update(curso["habilidades"])

    no_enseñadas = set(dataset["habilidades"]) - habilidades_enseñadas
    return list(no_enseñadas)


# ***** Estadísticas ******

def mostrar_estadisticas(dataset, instancias):
    print("\n" + "=" * 55)
    print("  ESTADÍSTICAS DEL DATASET")
    print("=" * 55)

    cursos = dataset["cursos"]
    habilidades = dataset["habilidades"]

    print(f"  Total de cursos          : {len(cursos)}")
    print(f"  Total de habilidades     : {len(habilidades)}")
    print(f"  Total de instancias      : {len(instancias)}")
    print(f"  Perfiles profesionales   : {len(dataset['perfiles_profesionales'])}")

    niveles = defaultdict(int)
    for c in cursos:
        niveles[c["nivel"]] += 1
    print(f"\n  Cursos por nivel:")
    for nivel, cant in sorted(niveles.items()):
        print(f"    {nivel:<15}: {cant}")

    duraciones = [c["duracion_semanas"] for c in cursos]
    print(f"\n  Duración de cursos (semanas):")
    print(f"    Mínima  : {min(duraciones)}")
    print(f"    Máxima  : {max(duraciones)}")
    print(f"    Promedio: {sum(duraciones)/len(duraciones):.1f}")

    sin_prereq = [c for c in cursos if not c["prerrequisitos"]]
    print(f"\n  Cursos sin prerrequisitos (nivel entrada): {len(sin_prereq)}")
    for c in sin_prereq:
        print(f"    - {c['nombre']}")

    print(f"\n  Perfiles y sus habilidades requeridas:")
    for pid, perfil in dataset["perfiles_profesionales"].items():
        print(f"    [{pid}] {perfil['nombre']}: "
              f"{len(perfil['habilidades_requeridas'])} habilidades")


# ***** Main *****

def main():
    print("=" * 55)
    print("  VALIDACIÓN DEL DATASET — Career Path Planner")
    print("=" * 55)

    dataset = cargar_dataset()
    instancias = cargar_instancias()

    errores_totales = []

    # 1. Habilidades existentes
    print("\n[1] Verificando referencias de habilidades...")
    errores = validar_habilidades_existentes(dataset)
    if errores:
        errores_totales.extend(errores)
        for e in errores:
            print(f"    ✗ {e}")
    else:
        print("    ✓ Todas las referencias de habilidades son válidas.")

    # 2. Ciclos (DAG)
    print("\n[2] Verificando que el grafo no tiene ciclos (DAG)...")
    es_dag, nodos_ciclo = detectar_ciclos_kahn(dataset)
    if not es_dag:
        msg = f"¡CICLO DETECTADO! Nodos involucrados: {nodos_ciclo}"
        errores_totales.append(msg)
        print(f"    ✗ {msg}")
    else:
        print("    ✓ El grafo es un DAG válido (sin ciclos).")

    # 3. Perfiles
    print("\n[3] Verificando perfiles profesionales...")
    errores = validar_perfiles(dataset)
    if errores:
        errores_totales.extend(errores)
        for e in errores:
            print(f"    ✗ {e}")
    else:
        print("    ✓ Todos los perfiles son válidos.")

    # 4. Instancias
    print("\n[4] Verificando instancias de prueba...")
    errores = validar_instancias(instancias, dataset)
    if errores:
        errores_totales.extend(errores)
        for e in errores:
            print(f"    ✗ {e}")
    else:
        print(f"    ✓ Las {len(instancias)} instancias son válidas.")

    # 5. Alcanzabilidad
    print("\n[5] Verificando alcanzabilidad de habilidades...")
    no_enseñadas = verificar_alcanzabilidad(dataset)
    if no_enseñadas:
        print(f"    ⚠ Habilidades en el pool sin curso que las enseñe: {no_enseñadas}")
    else:
        print("    ✓ Todas las habilidades pueden ser adquiridas mediante cursos.")

    # Resultado final
    print("\n" + "=" * 55)
    if errores_totales:
        print(f"  RESULTADO: ✗ {len(errores_totales)} error(es) encontrado(s).")
    else:
        print("  RESULTADO: ✓ Dataset válido al 100%.")
    print("=" * 55)

    mostrar_estadisticas(dataset, instancias)


if __name__ == "__main__":
    main()