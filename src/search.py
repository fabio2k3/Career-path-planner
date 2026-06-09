"""
search.py

Algoritmos de búsqueda sobre el DAG de cursos.

Incluye:
- A* con distintos criterios de costo.
- Greedy Best-First Search.
- Generación de k trayectorias alternativas.
- Validación de trayectorias completas.
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import FrozenSet, Optional

from graph import GrafoCursos, Curso


# *** Resultado de búsqueda *****

@dataclass
class SearchResult:
    """
    Contenedor con toda la información relevante de una ejecución de búsqueda.

    Agrupa tanto el resultado funcional de la búsqueda como métricas útiles para
    evaluación y comparación experimental.
    """
    algoritmo: str
    criterio: str
    instancia_id: str
    perfil_objetivo: str
    trayectoria: list
    costo_total_semanas: int
    num_cursos: int
    nodos_expandidos: int
    tiempo_segundos: float
    exito: bool

    def imprimir(self) -> None:
        """Muestra el resultado de búsqueda en un formato legible para consola."""
        sep = "─" * 58
        print(f"\n{sep}")
        print(f"  Algoritmo : {self.algoritmo}  |  Criterio: {self.criterio}")
        print(f"  Instancia : {self.instancia_id}  |  Perfil: {self.perfil_objetivo}")
        print(f"  Resultado : {'✓ Solución encontrada' if self.exito else '✗ Sin solución'}")
        print(sep)

        if self.exito:
            print(f"  Cursos en trayectoria : {self.num_cursos}")
            print(f"  Costo total (semanas) : {self.costo_total_semanas}")
            print(f"  Nodos expandidos      : {self.nodos_expandidos}")
            print(f"  Tiempo de búsqueda    : {self.tiempo_segundos:.4f}s")
            print(f"\n  Trayectoria:")
            for i, c in enumerate(self.trayectoria, 1):
                print(f"    {i:>2}. [{c.nivel[:3].upper()}] {c.nombre} ({c.duracion_semanas}s)")
        else:
            print("  No se encontró trayectoria válida.")

        print(sep)

    def to_dict(self) -> dict:
        """Convierte el resultado en un diccionario serializable."""
        return {
            "algoritmo": self.algoritmo,
            "criterio": self.criterio,
            "instancia_id": self.instancia_id,
            "perfil_objetivo": self.perfil_objetivo,
            "exito": self.exito,
            "num_cursos": self.num_cursos,
            "costo_total_semanas": self.costo_total_semanas,
            "nodos_expandidos": self.nodos_expandidos,
            "tiempo_segundos": round(self.tiempo_segundos, 6),
            "trayectoria_ids": [c.id for c in self.trayectoria],
            "trayectoria_nombres": [c.nombre for c in self.trayectoria],
        }


# *** Nodo interno del heap ******

@dataclass(order=True)
class _Nodo:
    """
    Nodo auxiliar usado por la cola de prioridad.

    El orden se basa en f y, en caso de empate, en tiebreak para mantener una
    expansión estable y determinista.
    """
    f: float
    tiebreak: int
    g: float = field(compare=False)
    habilidades: FrozenSet[str] = field(compare=False)
    num_cursos: int = field(compare=False, default=0)


# *** Reconstrucción de camino ******

def _reconstruir(came_from: dict, estado_final: FrozenSet[str], grafo: GrafoCursos,) -> list[Curso]:
    """
    Reconstruye la trayectoria óptima siguiendo la relación padre → hijo.

    Parameters
    ----------
    came_from : dict
        Mapa estado_actual -> (curso_id, estado_previo).
    estado_final : frozenset[str]
        Estado objetivo alcanzado por la búsqueda.
    grafo : GrafoCursos
        Grafo donde se encuentran los cursos.

    Returns
    -------
    list[Curso]
        Secuencia de cursos desde el inicio hasta el objetivo.
    """
    camino: list[Curso] = []
    estado = estado_final

    while estado in came_from:
        curso_id, estado_prev = came_from[estado]
        camino.append(grafo.cursos[curso_id])
        estado = estado_prev

    camino.reverse()
    return camino


# *** Funciones de costo y heurística ******

def _costo_curso(curso: Curso, criterio: str, alpha: float = 0.5) -> float:
    """
    Calcula el costo de incorporar un curso según el criterio elegido.

    - "semanas": usa la duración real del curso.
    - "cursos" : costo unitario por curso.
    - "balance": combina semanas normalizadas y número de cursos.

    Raises
    ------
    ValueError
        Si el criterio no es reconocido.
    """
    if criterio == "semanas":
        return float(curso.duracion_semanas)
    if criterio == "cursos":
        return 1.0
    if criterio == "balance":
        return alpha * (curso.duracion_semanas / 10.0) + (1.0 - alpha)

    raise ValueError(
        f"Criterio desconocido: '{criterio}'. Usa 'semanas', 'cursos' o 'balance'."
    )


def _heuristica(grafo: GrafoCursos, habilidades: FrozenSet[str], perfil_id: str, criterio: str, alpha: float = 0.5,) -> float:
    """
    Heurística admisible basada en habilidades faltantes.

    La idea es estimar cuántas habilidades del perfil objetivo todavía no han
    sido alcanzadas. Esto mantiene la búsqueda informada sin sobreestimar el
    costo restante.
    """
    faltantes = len(grafo.habilidades_faltantes(habilidades, perfil_id))

    if criterio == "semanas":
        return float(faltantes)
    if criterio == "cursos":
        return float(faltantes)
    if criterio == "balance":
        return alpha * (faltantes / 10.0) + (1.0 - alpha) * float(faltantes)

    return float(faltantes)


# *** Índice inverso: habilidad → cursos que la enseñan ******

def _construir_indice_habilidades(grafo: GrafoCursos) -> dict[str, list[str]]:
    """
    Precalcula qué cursos enseñan cada habilidad.

    Este índice puede servir como base para optimizaciones posteriores cuando
    se necesite recuperar cursos de forma más eficiente.
    """
    indice: dict[str, list[str]] = {}
    for cid, curso in grafo.cursos.items():
        for hab in curso.habilidades:
            indice.setdefault(hab, []).append(cid)
    return indice


def _cursos_tomados_desde_estado(habilidades: FrozenSet[str], grafo: GrafoCursos,) -> FrozenSet[str]:
    """
    Deduce qué cursos ya han sido “absorbidos” por el estado actual.

    Un curso se considera ya tomado si todas sus habilidades están contenidas
    en el conjunto de habilidades acumuladas.
    """
    return frozenset(
        cid for cid, c in grafo.cursos.items()
        if c.habilidades.issubset(habilidades)
    )


# *** A* genérico ******

def astar(grafo: GrafoCursos, habilidades_iniciales: FrozenSet[str], perfil_id: str, instancia_id: str = "?", criterio: str = "semanas",
    alpha: float = 0.5, max_nodos: int = 500_000,
    ) -> SearchResult:
    """
    Ejecuta A* sobre el DAG de cursos.

    El algoritmo optimiza la trayectoria según el criterio elegido y usa una
    heurística admisible para conservar optimalidad en los casos esperados.

    Parameters
    ----------
    criterio : str
        "semanas", "cursos" o "balance".
    alpha : float
        Peso usado en el criterio balanceado.
    max_nodos : int
        Límite de seguridad para evitar expansiones excesivas.

    Returns
    -------
    SearchResult
        Resultado completo de la búsqueda, incluyendo trayectoria y métricas.
    """
    inicio = time.perf_counter()
    counter = 0
    nodos_expandidos = 0

    h0 = _heuristica(grafo, habilidades_iniciales, perfil_id, criterio, alpha)
    frontera: list = []
    heapq.heappush(
        frontera,
        _Nodo(
            f=h0,
            tiebreak=counter,
            g=0.0,
            habilidades=habilidades_iniciales,
            num_cursos=0,
        ),
    )

    g_score: dict[FrozenSet[str], float] = {habilidades_iniciales: 0.0}
    came_from: dict[FrozenSet[str], tuple] = {}

    while frontera:
        if nodos_expandidos >= max_nodos:
            break

        nodo = heapq.heappop(frontera)
        nodos_expandidos += 1
        hab = nodo.habilidades

        # Ignorar nodos desactualizados cuando ya existe un camino mejor.
        if nodo.g > g_score.get(hab, float("inf")):
            continue

        # Si el perfil objetivo ya está satisfecho, reconstruir y devolver.
        if grafo.es_objetivo(hab, perfil_id):
            trayectoria = _reconstruir(came_from, hab, grafo)
            return SearchResult(
                algoritmo="A*",
                criterio=criterio,
                instancia_id=instancia_id,
                perfil_objetivo=perfil_id,
                trayectoria=trayectoria,
                costo_total_semanas=sum(c.duracion_semanas for c in trayectoria),
                num_cursos=len(trayectoria),
                nodos_expandidos=nodos_expandidos,
                tiempo_segundos=time.perf_counter() - inicio,
                exito=True,
            )

        # Determinar qué cursos ya han sido cubiertos por el estado actual.
        cursos_tomados = _cursos_tomados_desde_estado(hab, grafo)

        for curso in grafo.cursos_disponibles(hab, cursos_tomados):
            nuevas_hab = grafo.aplicar_curso(hab, curso)
            nuevo_g = nodo.g + _costo_curso(curso, criterio, alpha)

            # Podar caminos que no mejoran el costo conocido.
            if nuevo_g >= g_score.get(nuevas_hab, float("inf")):
                continue

            g_score[nuevas_hab] = nuevo_g
            came_from[nuevas_hab] = (curso.id, hab)
            h = _heuristica(grafo, nuevas_hab, perfil_id, criterio, alpha)

            counter += 1
            heapq.heappush(
                frontera,
                _Nodo(f=nuevo_g + h, tiebreak=counter, g=nuevo_g, habilidades=nuevas_hab, num_cursos=nodo.num_cursos + 1,),
            )

    return SearchResult(
        algoritmo="A*",
        criterio=criterio,
        instancia_id=instancia_id,
        perfil_objetivo=perfil_id,
        trayectoria=[],
        costo_total_semanas=0,
        num_cursos=0,
        nodos_expandidos=nodos_expandidos,
        tiempo_segundos=time.perf_counter() - inicio,
        exito=False,
    )


# *** Greedy Best-First *******

def greedy(grafo: GrafoCursos, habilidades_iniciales: FrozenSet[str], perfil_id: str, instancia_id: str = "?",) -> SearchResult:
    """
    Ejecuta Greedy Best-First Search.

    Prioriza siempre el estado que aparenta estar más cerca del objetivo según
    la heurística, sin garantizar optimalidad.
    """
    inicio = time.perf_counter()
    counter = 0
    nodos_expandidos = 0

    h0 = float(len(grafo.habilidades_faltantes(habilidades_iniciales, perfil_id)))
    frontera: list = []
    heapq.heappush(
        frontera,
        _Nodo(f=h0, tiebreak=counter, g=0.0, habilidades=habilidades_iniciales,),
    )

    visitados: set[FrozenSet[str]] = set()
    came_from: dict[FrozenSet[str], tuple] = {}
    g_real: dict[FrozenSet[str], float] = {habilidades_iniciales: 0.0}

    while frontera:
        nodo = heapq.heappop(frontera)
        nodos_expandidos += 1
        hab = nodo.habilidades

        if hab in visitados:
            continue
        visitados.add(hab)

        if grafo.es_objetivo(hab, perfil_id):
            trayectoria = _reconstruir(came_from, hab, grafo)
            return SearchResult(
                algoritmo="Greedy",
                criterio="heuristica",
                instancia_id=instancia_id,
                perfil_objetivo=perfil_id,
                trayectoria=trayectoria,
                costo_total_semanas=sum(c.duracion_semanas for c in trayectoria),
                num_cursos=len(trayectoria),
                nodos_expandidos=nodos_expandidos,
                tiempo_segundos=time.perf_counter() - inicio,
                exito=True,
            )

        cursos_tomados = _cursos_tomados_desde_estado(hab, grafo)

        for curso in grafo.cursos_disponibles(hab, cursos_tomados):
            nuevas_hab = grafo.aplicar_curso(hab, curso)

            if nuevas_hab in visitados:
                continue

            h = float(len(grafo.habilidades_faltantes(nuevas_hab, perfil_id)))
            nuevo_g = g_real.get(hab, 0.0) + curso.duracion_semanas

            if nuevas_hab not in came_from or nuevo_g < g_real.get(nuevas_hab, float("inf")):
                came_from[nuevas_hab] = (curso.id, hab)
                g_real[nuevas_hab] = nuevo_g

            counter += 1
            heapq.heappush(
                frontera,
                _Nodo(
                    f=h,
                    tiebreak=counter,
                    g=nuevo_g,
                    habilidades=nuevas_hab,
                ),
            )

    return SearchResult(
        algoritmo="Greedy",
        criterio="heuristica",
        instancia_id=instancia_id,
        perfil_objetivo=perfil_id,
        trayectoria=[],
        costo_total_semanas=0,
        num_cursos=0,
        nodos_expandidos=nodos_expandidos,
        tiempo_segundos=time.perf_counter() - inicio,
        exito=False,
    )


# *** K mejores trayectorias ********

class _GrafoRestringido:
    """
    Adaptador de GrafoCursos que excluye un conjunto de cursos bloqueados.

    Se utiliza para generar trayectorias alternativas sin repetir exactamente
    la misma secuencia de cursos.
    """

    def __init__(self, grafo: GrafoCursos, bloqueados: set[str]):
        self.cursos = {
            cid: c for cid, c in grafo.cursos.items()
            if cid not in bloqueados
        }
        self.perfiles = grafo.perfiles
        self.habilidades = grafo.habilidades

    def cursos_disponibles(self, habilidades_actuales: FrozenSet[str], cursos_tomados: FrozenSet[str],) -> list[Curso]:
        return [
            c for c in self.cursos.values()
            if (
                c.id not in cursos_tomados
                and c.prerrequisitos.issubset(habilidades_actuales)
                and not c.habilidades.issubset(habilidades_actuales)
            )
        ]

    def aplicar_curso(self, habilidades_actuales: FrozenSet[str], curso: Curso,) -> FrozenSet[str]:
        return habilidades_actuales | curso.habilidades

    def es_objetivo(self, habilidades_actuales: FrozenSet[str], perfil_id: str,) -> bool:
        return self.perfiles[perfil_id].habilidades_requeridas.issubset(habilidades_actuales)

    def habilidades_faltantes(self, habilidades_actuales: FrozenSet[str], perfil_id: str,) -> FrozenSet[str]:
        return self.perfiles[perfil_id].habilidades_requeridas - habilidades_actuales


def k_mejores(grafo: GrafoCursos, habilidades_iniciales: FrozenSet[str], perfil_id: str, instancia_id: str = "?",
    k: int = 3, criterio: str = "semanas", max_nodos_alternativa: int = 50_000,
) -> list[SearchResult]:
    """
    Genera hasta k trayectorias distintas, ordenadas por costo ascendente.

    La primera solución se obtiene con A* estándar. Luego, en cada iteración,
    se bloquea uno de los cursos de la mejor trayectoria anterior y se vuelve a
    ejecutar A* sobre un grafo restringido.
    """
    inicio = time.perf_counter()
    nodos_total = 0

    # Primera solución sin restricciones adicionales.
    r0 = astar(grafo, habilidades_iniciales, perfil_id, instancia_id, criterio)
    nodos_total += r0.nodos_expandidos

    if not r0.exito:
        return []

    soluciones: list[SearchResult] = [
        SearchResult(
            algoritmo="K-Mejores (1/k)",
            criterio=criterio,
            instancia_id=instancia_id,
            perfil_objetivo=perfil_id,
            trayectoria=r0.trayectoria,
            costo_total_semanas=r0.costo_total_semanas,
            num_cursos=r0.num_cursos,
            nodos_expandidos=nodos_total,
            tiempo_segundos=time.perf_counter() - inicio,
            exito=True,
        )
    ]

    bloqueados_global: set[str] = set()

    for idx in range(1, k):
        mejor_alternativa: Optional[SearchResult] = None
        mejor_costo = float("inf")
        mejor_bloqueado: Optional[str] = None

        # Se prueban primero los cursos más costosos de la trayectoria anterior.
        candidatos = sorted(
            soluciones[-1].trayectoria,
            key=lambda c: _costo_curso(c, criterio),
            reverse=True,
        )

        for curso_bloquear in candidatos:
            if curso_bloquear.id in bloqueados_global:
                continue

            gr = _GrafoRestringido(grafo, bloqueados_global | {curso_bloquear.id})
            r_alt = astar(
                gr,
                habilidades_iniciales,
                perfil_id,
                instancia_id,
                criterio,
                max_nodos=max_nodos_alternativa,
            )
            nodos_total += r_alt.nodos_expandidos

            if not r_alt.exito:
                continue

            # Evitar duplicar exactamente una solución ya encontrada.
            ids_nueva = frozenset(c.id for c in r_alt.trayectoria)
            if any(frozenset(c.id for c in s.trayectoria) == ids_nueva for s in soluciones):
                continue

            if r_alt.costo_total_semanas < mejor_costo:
                mejor_costo = r_alt.costo_total_semanas
                mejor_alternativa = r_alt
                mejor_bloqueado = curso_bloquear.id

        if mejor_alternativa is None or mejor_bloqueado is None:
            break

        bloqueados_global.add(mejor_bloqueado)
        soluciones.append(
            SearchResult(
                algoritmo=f"K-Mejores ({idx + 1}/k)",
                criterio=criterio,
                instancia_id=instancia_id,
                perfil_objetivo=perfil_id,
                trayectoria=mejor_alternativa.trayectoria,
                costo_total_semanas=mejor_alternativa.costo_total_semanas,
                num_cursos=mejor_alternativa.num_cursos,
                nodos_expandidos=nodos_total,
                tiempo_segundos=time.perf_counter() - inicio,
                exito=True,
            )
        )

    return soluciones


# *** Validador *******

def validar_trayectoria(grafo: GrafoCursos, trayectoria: list[Curso], habilidades_iniciales: FrozenSet[str], perfil_id: str,) -> tuple[bool, str]:
    """
    Verifica que una trayectoria sea válida respecto al grafo y al objetivo.

    Comprueba:
    1. Que cada curso pueda tomarse en el momento correspondiente.
    2. Que al final se hayan alcanzado todas las habilidades del perfil.
    """
    estado = habilidades_iniciales

    for i, curso in enumerate(trayectoria):
        if not curso.prerrequisitos.issubset(estado):
            faltantes = curso.prerrequisitos - estado
            return (
                False,
                f"Paso {i + 1} '{curso.nombre}': prerrequisitos no cumplidos: {sorted(faltantes)}",
            )
        estado = grafo.aplicar_curso(estado, curso)

    if not grafo.es_objetivo(estado, perfil_id):
        faltantes = grafo.habilidades_faltantes(estado, perfil_id)
        return (
            False,
            f"Objetivo no alcanzado. Habilidades faltantes: {sorted(faltantes)}",
        )

    return True, "OK"