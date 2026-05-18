"""
search.py

Implementa A* y Greedy sobre el DAG de cursos.

Optimización de memoria: en lugar de copiar la trayectoria completa en cada
nodo, se usan punteros padre (came_from) para reconstruir el camino al final.

Estado = frozenset de habilidades adquiridas (identifica unívocamente el
         conjunto de cursos útiles tomados en un DAG sin ciclos).
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import FrozenSet, Optional

from graph import GrafoCursos, Curso


# ----- Resultado -----

@dataclass
class SearchResult:
    algoritmo: str
    instancia_id: str
    perfil_objetivo: str
    trayectoria: list         # lista de Curso en orden
    costo_total_semanas: int
    nodos_expandidos: int
    tiempo_segundos: float
    exito: bool

    def imprimir(self):
        sep = "─" * 56
        print(f"\n{sep}")
        print(f"  Algoritmo : {self.algoritmo}")
        print(f"  Instancia : {self.instancia_id}  |  Perfil: {self.perfil_objetivo}")
        print(f"  Resultado : {'✓ Solución encontrada' if self.exito else '✗ Sin solución'}")
        print(sep)
        if self.exito:
            print(f"  Cursos en trayectoria : {len(self.trayectoria)}")
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
        return {
            "algoritmo": self.algoritmo,
            "instancia_id": self.instancia_id,
            "perfil_objetivo": self.perfil_objetivo,
            "exito": self.exito,
            "num_cursos": len(self.trayectoria),
            "costo_total_semanas": self.costo_total_semanas,
            "nodos_expandidos": self.nodos_expandidos,
            "tiempo_segundos": round(self.tiempo_segundos, 6),
            "trayectoria_ids": [c.id for c in self.trayectoria],
            "trayectoria_nombres": [c.nombre for c in self.trayectoria],
        }


# ----- Nodo interno (solo datos de prioridad + estado) -----

@dataclass(order=True)
class _Nodo:
    f: float
    tiebreak: int
    g: int = field(compare=False)
    habilidades: FrozenSet[str] = field(compare=False)
    curso_id: Optional[str] = field(compare=False, default=None)  # acción que llevó aquí


# ----- Reconstrucción de camino desde came_from -----

def _reconstruir(
    came_from: dict,
    estado_final: FrozenSet[str],
    grafo: GrafoCursos,
) -> list[Curso]:
    """Recorre came_from hacia atrás y devuelve la lista de cursos en orden."""
    camino = []
    estado = estado_final
    while estado in came_from:
        curso_id, estado_prev = came_from[estado]
        camino.append(grafo.cursos[curso_id])
        estado = estado_prev
    camino.reverse()
    return camino


# === A* ===

def astar(
    grafo: GrafoCursos,
    habilidades_iniciales: FrozenSet[str],
    perfil_id: str,
    instancia_id: str = "?",
) -> SearchResult:
    """
    A* con heurística admisible h(s) = |H* \\ H_adq(s)|.
    Garantiza la trayectoria de mínimo costo (semanas totales).
    Memoria O(estados): usa came_from en vez de copiar caminos en nodos.
    """
    inicio = time.perf_counter()
    counter = 0
    nodos_expandidos = 0

    # Frontera
    frontera: list[_Nodo] = []
    h0 = grafo.heuristica(habilidades_iniciales, perfil_id)
    heapq.heappush(frontera, _Nodo(f=float(h0), tiebreak=counter, g=0,
                                   habilidades=habilidades_iniciales))

    # Mejor costo conocido por estado
    g_score: dict[FrozenSet[str], int] = {habilidades_iniciales: 0}

    # Punteros padre: estado → (curso_id_que_llevó_aquí, estado_padre)
    came_from: dict[FrozenSet[str], tuple] = {}

    while frontera:
        nodo = heapq.heappop(frontera)
        nodos_expandidos += 1
        hab = nodo.habilidades

        # Nodo obsoleto
        if nodo.g > g_score.get(hab, float("inf")):
            continue

        # Objetivo
        if grafo.es_objetivo(hab, perfil_id):
            trayectoria = _reconstruir(came_from, hab, grafo)
            return SearchResult(
                algoritmo="A*",
                instancia_id=instancia_id,
                perfil_objetivo=perfil_id,
                trayectoria=trayectoria,
                costo_total_semanas=nodo.g,
                nodos_expandidos=nodos_expandidos,
                tiempo_segundos=time.perf_counter() - inicio,
                exito=True,
            )

        # Expandir: cursos disponibles = prerreqs ⊆ hab y no tomados aún
        # "No tomados" se infiere: un curso ya tomado no añade habilidades nuevas
        # (si skills ⊆ hab ya están incluidas, tomarlo de nuevo no mejora)
        cursos_tomados = frozenset(
            cid for cid, c in grafo.cursos.items()
            if c.habilidades.issubset(hab) and c.prerrequisitos.issubset(hab)
        )

        for curso in grafo.cursos_disponibles(hab, cursos_tomados):
            nuevas_hab = grafo.aplicar_curso(hab, curso)
            nuevo_g = nodo.g + curso.duracion_semanas

            if nuevo_g >= g_score.get(nuevas_hab, float("inf")):
                continue

            g_score[nuevas_hab] = nuevo_g
            came_from[nuevas_hab] = (curso.id, hab)
            h = grafo.heuristica(nuevas_hab, perfil_id)
            counter += 1
            heapq.heappush(frontera, _Nodo(
                f=float(nuevo_g + h),
                tiebreak=counter,
                g=nuevo_g,
                habilidades=nuevas_hab,
            ))

    return SearchResult(
        algoritmo="A*", instancia_id=instancia_id, perfil_objetivo=perfil_id,
        trayectoria=[], costo_total_semanas=0,
        nodos_expandidos=nodos_expandidos,
        tiempo_segundos=time.perf_counter() - inicio, exito=False,
    )


# ------ Greedy ------

def greedy(
    grafo: GrafoCursos,
    habilidades_iniciales: FrozenSet[str],
    perfil_id: str,
    instancia_id: str = "?",
) -> SearchResult:
    """
    Greedy Best-First: f(n) = h(n).
    Elige en cada paso el estado con menos habilidades faltantes.
    Rápido pero no garantiza optimalidad.
    """
    inicio = time.perf_counter()
    counter = 0
    nodos_expandidos = 0

    frontera: list[_Nodo] = []
    h0 = grafo.heuristica(habilidades_iniciales, perfil_id)
    heapq.heappush(frontera, _Nodo(f=float(h0), tiebreak=counter, g=0,
                                   habilidades=habilidades_iniciales))

    visitados: set[FrozenSet[str]] = set()
    came_from: dict[FrozenSet[str], tuple] = {}
    g_real: dict[FrozenSet[str], int] = {habilidades_iniciales: 0}

    while frontera:
        nodo = heapq.heappop(frontera)
        nodos_expandidos += 1
        hab = nodo.habilidades

        if hab in visitados:
            continue
        visitados.add(hab)

        if grafo.es_objetivo(hab, perfil_id):
            trayectoria = _reconstruir(came_from, hab, grafo)
            costo = sum(c.duracion_semanas for c in trayectoria)
            return SearchResult(
                algoritmo="Greedy",
                instancia_id=instancia_id,
                perfil_objetivo=perfil_id,
                trayectoria=trayectoria,
                costo_total_semanas=costo,
                nodos_expandidos=nodos_expandidos,
                tiempo_segundos=time.perf_counter() - inicio,
                exito=True,
            )

        cursos_tomados = frozenset(
            cid for cid, c in grafo.cursos.items()
            if c.habilidades.issubset(hab) and c.prerrequisitos.issubset(hab)
        )

        for curso in grafo.cursos_disponibles(hab, cursos_tomados):
            nuevas_hab = grafo.aplicar_curso(hab, curso)
            if nuevas_hab in visitados:
                continue
            h = grafo.heuristica(nuevas_hab, perfil_id)
            nuevo_g = g_real.get(hab, 0) + curso.duracion_semanas
            if nuevas_hab not in came_from or nuevo_g < g_real.get(nuevas_hab, float("inf")):
                came_from[nuevas_hab] = (curso.id, hab)
                g_real[nuevas_hab] = nuevo_g
            counter += 1
            heapq.heappush(frontera, _Nodo(
                f=float(h), tiebreak=counter, g=nuevo_g, habilidades=nuevas_hab,
            ))

    return SearchResult(
        algoritmo="Greedy", instancia_id=instancia_id, perfil_objetivo=perfil_id,
        trayectoria=[], costo_total_semanas=0,
        nodos_expandidos=nodos_expandidos,
        tiempo_segundos=time.perf_counter() - inicio, exito=False,
    )


# ------ Validador ------

def validar_trayectoria(
    grafo: GrafoCursos,
    trayectoria: list,
    habilidades_iniciales: FrozenSet[str],
    perfil_id: str,
) -> tuple[bool, str]:
    """
    Verifica:
      1. Cada curso cumple sus prerrequisitos en el momento de tomarse.
      2. Al final se satisface el perfil objetivo.
    """
    estado = habilidades_iniciales
    for i, curso in enumerate(trayectoria):
        if not curso.prerrequisitos.issubset(estado):
            faltantes = curso.prerrequisitos - estado
            return False, f"Paso {i+1} '{curso.nombre}': prerreqs no cumplidos: {faltantes}"
        estado = grafo.aplicar_curso(estado, curso)

    if not grafo.es_objetivo(estado, perfil_id):
        faltantes = grafo.habilidades_faltantes(estado, perfil_id)
        return False, f"Objetivo no alcanzado. Faltantes: {faltantes}"

    return True, "OK"