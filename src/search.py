"""
search.py

Algoritmos de búsqueda sobre el DAG de cursos:

  - astar(criterio="semanas")  : A* minimizando semanas totales
  - astar(criterio="cursos")   : A* minimizando número de cursos
  - astar(criterio="balance")  : A* con costo combinado α·semanas + β·cursos
  - greedy()                   : Greedy best-first (h = habilidades faltantes)
  - k_mejores()                : genera k trayectorias distintas ordenadas por costo

Optimización de memoria: punteros padre (came_from) en lugar de
copiar la trayectoria completa en cada nodo del heap.

Estado = frozenset de habilidades adquiridas.
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
    criterio: str
    instancia_id: str
    perfil_objetivo: str
    trayectoria: list        # lista de Curso en orden
    costo_total_semanas: int
    num_cursos: int
    nodos_expandidos: int
    tiempo_segundos: float
    exito: bool

    def imprimir(self):
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


# ------ Nodo interno ------

@dataclass(order=True)
class _Nodo:
    f: float
    tiebreak: int
    g: float = field(compare=False)          # costo acumulado real
    habilidades: FrozenSet[str] = field(compare=False)
    num_cursos: int = field(compare=False, default=0)


# ------ Reconstrucción de camino ------

def _reconstruir(
    came_from: dict,
    estado_final: FrozenSet[str],
    grafo: GrafoCursos,
) -> list:
    camino = []
    estado = estado_final
    while estado in came_from:
        curso_id, estado_prev = came_from[estado]
        camino.append(grafo.cursos[curso_id])
        estado = estado_prev
    camino.reverse()
    return camino


# --- Función de costo según criterio -----

def _costo_curso(curso: Curso, criterio: str, alpha: float = 0.5) -> float:
    """
    Devuelve el costo de tomar un curso según el criterio elegido.

    - "semanas" : costo = duración en semanas  (minimiza tiempo total)
    - "cursos"  : costo = 1                    (minimiza número de cursos)
    - "balance" : costo = α·semanas + (1-α)·1  (criterio combinado)
    """
    if criterio == "semanas":
        return float(curso.duracion_semanas)
    elif criterio == "cursos":
        return 1.0
    elif criterio == "balance":
        # Normalización: duracion máxima ~10 semanas, se divide para escalar
        return alpha * (curso.duracion_semanas / 10.0) + (1 - alpha) * 1.0
    else:
        raise ValueError(f"Criterio desconocido: '{criterio}'. "
                         f"Usa 'semanas', 'cursos' o 'balance'.")


def _heuristica(grafo, habilidades, perfil_id, criterio, alpha=0.5) -> float:
    """
    Heurística admisible ajustada al criterio:
    - "semanas" : h = habilidades faltantes (cada una cuesta ≥ 1 semana)
    - "cursos"  : h = habilidades faltantes (cada una requiere ≥ 1 curso)
    - "balance" : h = combinación de ambas
    Siempre admisible: nunca sobreestima el costo real restante.
    """
    faltantes = len(grafo.habilidades_faltantes(habilidades, perfil_id))
    if criterio == "semanas":
        return float(faltantes)          # ≤ costo real (cursos duran ≥ 1s)
    elif criterio == "cursos":
        return float(faltantes)          # ≤ cursos restantes necesarios
    elif criterio == "balance":
        return alpha * (faltantes / 10.0) + (1 - alpha) * float(faltantes)
    return float(faltantes)


# ____ A* genérico _____

def astar(
    grafo,
    habilidades_iniciales: FrozenSet[str],
    perfil_id: str,
    instancia_id: str = "?",
    criterio: str = "semanas",
    alpha: float = 0.5,
    max_nodos: int = 500_000,
) -> SearchResult:
    """
    A* sobre el DAG de cursos.

    Parámetros de criterio:
      criterio="semanas"  → minimiza semanas totales          (costo = duracion)
      criterio="cursos"   → minimiza número de cursos         (costo = 1)
      criterio="balance"  → minimiza α·semanas + (1-α)·cursos (costo combinado)
      alpha               → peso de semanas en criterio balance (default 0.5)

    Garantiza optimalidad con la heurística admisible definida.
    Memoria eficiente: usa came_from en lugar de copiar caminos.
    """
    inicio = time.perf_counter()
    counter = 0
    nodos_expandidos = 0

    h0 = _heuristica(grafo, habilidades_iniciales, perfil_id, criterio, alpha)
    frontera: list = []
    heapq.heappush(frontera, _Nodo(
        f=h0, tiebreak=counter, g=0.0,
        habilidades=habilidades_iniciales, num_cursos=0,
    ))

    g_score: dict[FrozenSet[str], float] = {habilidades_iniciales: 0.0}
    came_from: dict[FrozenSet[str], tuple] = {}

    while frontera:
        if nodos_expandidos >= max_nodos:
            break
        nodo = heapq.heappop(frontera)
        nodos_expandidos += 1
        hab = nodo.habilidades

        if nodo.g > g_score.get(hab, float("inf")):
            continue

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

        cursos_tomados = frozenset(
            cid for cid, c in grafo.cursos.items()
            if c.habilidades.issubset(hab) and c.prerrequisitos.issubset(hab)
        )

        for curso in grafo.cursos_disponibles(hab, cursos_tomados):
            nuevas_hab = grafo.aplicar_curso(hab, curso)
            nuevo_g = nodo.g + _costo_curso(curso, criterio, alpha)

            if nuevo_g >= g_score.get(nuevas_hab, float("inf")):
                continue

            g_score[nuevas_hab] = nuevo_g
            came_from[nuevas_hab] = (curso.id, hab)
            h = _heuristica(grafo, nuevas_hab, perfil_id, criterio, alpha)
            counter += 1
            heapq.heappush(frontera, _Nodo(
                f=nuevo_g + h, tiebreak=counter,
                g=nuevo_g, habilidades=nuevas_hab,
                num_cursos=nodo.num_cursos + 1,
            ))

    return SearchResult(
        algoritmo="A*", criterio=criterio,
        instancia_id=instancia_id, perfil_objetivo=perfil_id,
        trayectoria=[], costo_total_semanas=0, num_cursos=0,
        nodos_expandidos=nodos_expandidos,
        tiempo_segundos=time.perf_counter() - inicio, exito=False,
    )


# ---- Greedy -----

def greedy(
    grafo: GrafoCursos,
    habilidades_iniciales: FrozenSet[str],
    perfil_id: str,
    instancia_id: str = "?",
) -> SearchResult:
    """
    Greedy Best-First: f(n) = h(n) = habilidades faltantes.
    Elige en cada paso el estado con menos habilidades del objetivo pendientes.
    Rápido pero no garantiza optimalidad.
    """
    inicio = time.perf_counter()
    counter = 0
    nodos_expandidos = 0

    h0 = float(len(grafo.habilidades_faltantes(habilidades_iniciales, perfil_id)))
    frontera: list = []
    heapq.heappush(frontera, _Nodo(
        f=h0, tiebreak=counter, g=0.0,
        habilidades=habilidades_iniciales,
    ))

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
                criterio="semanas",
                instancia_id=instancia_id,
                perfil_objetivo=perfil_id,
                trayectoria=trayectoria,
                costo_total_semanas=sum(c.duracion_semanas for c in trayectoria),
                num_cursos=len(trayectoria),
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
            h = float(len(grafo.habilidades_faltantes(nuevas_hab, perfil_id)))
            nuevo_g = g_real.get(hab, 0.0) + curso.duracion_semanas
            if nuevas_hab not in came_from or nuevo_g < g_real.get(nuevas_hab, float("inf")):
                came_from[nuevas_hab] = (curso.id, hab)
                g_real[nuevas_hab] = nuevo_g
            counter += 1
            heapq.heappush(frontera, _Nodo(
                f=h, tiebreak=counter, g=nuevo_g, habilidades=nuevas_hab,
            ))

    return SearchResult(
        algoritmo="Greedy", criterio="semanas",
        instancia_id=instancia_id, perfil_objetivo=perfil_id,
        trayectoria=[], costo_total_semanas=0, num_cursos=0,
        nodos_expandidos=nodos_expandidos,
        tiempo_segundos=time.perf_counter() - inicio, exito=False,
    )


# ---- K mejores trayectorias -----

def k_mejores(
    grafo: GrafoCursos,
    habilidades_iniciales: FrozenSet[str],
    perfil_id: str,
    instancia_id: str = "?",
    k: int = 3,
    criterio: str = "semanas",
) -> list:
    """
    Genera hasta k trayectorias distintas ejecutando A* iterativamente.

    Estrategia: en cada iteracion ejecuta A* y bloquea el curso mas costoso
    de la trayectoria encontrada, forzando al siguiente A* a buscar un
    camino alternativo. Eficiente en memoria porque reutiliza A* estandar.

    Devuelve lista de SearchResult ordenada por costo ascendente.
    """
    inicio = time.perf_counter()
    soluciones = []
    nodos_total = 0

    # Primera solucion: A* estandar sin restricciones
    r0 = astar(grafo, habilidades_iniciales, perfil_id, instancia_id, criterio)
    nodos_total += r0.nodos_expandidos

    if not r0.exito:
        return []

    soluciones.append(SearchResult(
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
    ))

    cursos_bloqueados_global: set = set()

    for idx in range(1, k):
        mejor_alternativa = None
        mejor_costo = float("inf")
        mejor_bloqueado = None

        # Candidatos a bloquear: cursos de la ultima trayectoria,
        # ordenados de mayor a menor duracion
        trayectoria_base = soluciones[-1].trayectoria
        candidatos = sorted(
            trayectoria_base,
            key=lambda c: c.duracion_semanas,
            reverse=True,
        )

        for curso_bloquear in candidatos:
            if curso_bloquear.id in cursos_bloqueados_global:
                continue

            # Crear sub-grafo sin el curso bloqueado
            cursos_filtrados = {
                cid: c for cid, c in grafo.cursos.items()
                if cid != curso_bloquear.id
            }

            # Grafo temporal usando los mismos perfiles y habilidades
            class _GrafoTemp:
                def __init__(self):
                    self.cursos = cursos_filtrados
                    self.perfiles = grafo.perfiles
                    self.habilidades = grafo.habilidades

                def cursos_disponibles(self, hab, tomados):
                    return [
                        c for c in self.cursos.values()
                        if c.id not in tomados
                        and c.prerrequisitos.issubset(hab)
                        and not c.habilidades.issubset(hab)
                    ]

                def aplicar_curso(self, hab, curso):
                    return hab | curso.habilidades

                def es_objetivo(self, hab, perfil_id):
                    return self.perfiles[perfil_id].habilidades_requeridas.issubset(hab)

                def habilidades_faltantes(self, hab, perfil_id):
                    return self.perfiles[perfil_id].habilidades_requeridas - hab

            gt = _GrafoTemp()

            # Correr A* sobre el grafo temporal
            r_alt = astar(gt, habilidades_iniciales, perfil_id,
                          instancia_id, criterio, max_nodos=10_000)
            nodos_total += r_alt.nodos_expandidos

            if not r_alt.exito:
                continue

            # Verificar que no es identica a una solucion ya encontrada
            ids_nueva = frozenset(c.id for c in r_alt.trayectoria)
            ya_vista = any(
                frozenset(c.id for c in s.trayectoria) == ids_nueva
                for s in soluciones
            )
            if ya_vista:
                continue

            if r_alt.costo_total_semanas < mejor_costo:
                mejor_costo = r_alt.costo_total_semanas
                mejor_alternativa = r_alt
                mejor_bloqueado = curso_bloquear.id

        if mejor_alternativa is None:
            break

        cursos_bloqueados_global.add(mejor_bloqueado)
        soluciones.append(SearchResult(
            algoritmo=f"K-Mejores ({idx+1}/k)",
            criterio=criterio,
            instancia_id=instancia_id,
            perfil_objetivo=perfil_id,
            trayectoria=mejor_alternativa.trayectoria,
            costo_total_semanas=mejor_alternativa.costo_total_semanas,
            num_cursos=mejor_alternativa.num_cursos,
            nodos_expandidos=nodos_total,
            tiempo_segundos=time.perf_counter() - inicio,
            exito=True,
        ))

    return soluciones


# ---- Validador -----

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