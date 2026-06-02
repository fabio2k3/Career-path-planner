"""
search.py

Algoritmos de búsqueda sobre el DAG de cursos:

  - astar(criterio="semanas")  : A* minimizando semanas totales
  - astar(criterio="cursos")   : A* minimizando número de cursos
  - astar(criterio="balance")  : A* con costo combinado α·semanas + β·cursos
  - greedy()                   : Greedy best-first (f = h = habilidades faltantes)
  - k_mejores()                : genera k trayectorias distintas ordenadas por costo
  - validar_trayectoria()      : verifica prerrequisitos y objetivo alcanzado

"""

import heapq
import time
from dataclasses import dataclass, field
from typing import FrozenSet, Optional

from graph import GrafoCursos, Curso


# ── Resultado ──────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Resultado completo de una ejecución de búsqueda."""
    algoritmo: str
    criterio: str
    instancia_id: str
    perfil_objetivo: str
    trayectoria: list           # lista de Curso en orden
    costo_total_semanas: int
    num_cursos: int
    nodos_expandidos: int
    tiempo_segundos: float
    exito: bool

    def imprimir(self) -> None:
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
            "algoritmo":            self.algoritmo,
            "criterio":             self.criterio,
            "instancia_id":         self.instancia_id,
            "perfil_objetivo":      self.perfil_objetivo,
            "exito":                self.exito,
            "num_cursos":           self.num_cursos,
            "costo_total_semanas":  self.costo_total_semanas,
            "nodos_expandidos":     self.nodos_expandidos,
            "tiempo_segundos":      round(self.tiempo_segundos, 6),
            "trayectoria_ids":      [c.id for c in self.trayectoria],
            "trayectoria_nombres":  [c.nombre for c in self.trayectoria],
        }


# ── Nodo interno del heap ──────────────────────────────────────────────────────

@dataclass(order=True)
class _Nodo:
    f: float
    tiebreak: int
    g: float           = field(compare=False)
    habilidades: FrozenSet[str] = field(compare=False)
    num_cursos: int    = field(compare=False, default=0)


# ── Reconstrucción de camino ───────────────────────────────────────────────────

def _reconstruir(
    came_from: dict,
    estado_final: FrozenSet[str],
    grafo: GrafoCursos,
) -> list[Curso]:
    """Reconstruye la trayectoria siguiendo los punteros padre."""
    camino: list[Curso] = []
    estado = estado_final
    while estado in came_from:
        curso_id, estado_prev = came_from[estado]
        camino.append(grafo.cursos[curso_id])
        estado = estado_prev
    camino.reverse()
    return camino


# ── Funciones de costo y heurística ───────────────────────────────────────────

def _costo_curso(curso: Curso, criterio: str, alpha: float = 0.5) -> float:
    """
    Costo de tomar un curso según el criterio:

    - "semanas" : duración en semanas  → minimiza tiempo total
    - "cursos"  : 1                    → minimiza número de cursos
    - "balance" : α·(sem/10) + (1-α)  → criterio combinado normalizado
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


def _heuristica(
    grafo: GrafoCursos,
    habilidades: FrozenSet[str],
    perfil_id: str,
    criterio: str,
    alpha: float = 0.5,
) -> float:
    """
    Heurística admisible:  h(s) = |H* \\ s|

    Admisible en todos los criterios porque cada habilidad faltante
    requiere al menos un curso con costo ≥ 1 en cualquier métrica.
    """
    faltantes = len(grafo.habilidades_faltantes(habilidades, perfil_id))
    if criterio == "semanas":
        return float(faltantes)
    if criterio == "cursos":
        return float(faltantes)
    if criterio == "balance":
        return alpha * (faltantes / 10.0) + (1.0 - alpha) * float(faltantes)
    return float(faltantes)


# ── Índice inverso: habilidad → cursos que la enseñan ─────────────────────────

def _construir_indice_habilidades(grafo: GrafoCursos) -> dict[str, list[str]]:
    """
    Precalcula qué cursos enseñan cada habilidad.
    Permite deducir cursos_tomados en O(|habilidades_estado|) en lugar de O(n_cursos).
    """
    indice: dict[str, list[str]] = {}
    for cid, curso in grafo.cursos.items():
        for hab in curso.habilidades:
            indice.setdefault(hab, []).append(cid)
    return indice


def _cursos_tomados_desde_estado(
    habilidades: FrozenSet[str],
    grafo: GrafoCursos,
) -> FrozenSet[str]:
    """
    Deduce el conjunto de cursos cuyos efectos están completamente
    incluidos en el estado actual (habilidades adquiridas).

    Complejidad: O(n_cursos) — se llama solo cuando es necesario.
    En el contexto de A* esto es equivalente al método anterior,
    pero se puede optimizar aún más con el índice si se desea.
    """
    return frozenset(
        cid for cid, c in grafo.cursos.items()
        if c.habilidades.issubset(habilidades)
    )


# ── A* genérico ───────────────────────────────────────────────────────────────

def astar(
    grafo: GrafoCursos,
    habilidades_iniciales: FrozenSet[str],
    perfil_id: str,
    instancia_id: str = "?",
    criterio: str = "semanas",
    alpha: float = 0.5,
    max_nodos: int = 500_000,
) -> SearchResult:
    """
    A* sobre el DAG de cursos.

    Parámetros
    ----------
    criterio : "semanas" | "cursos" | "balance"
        Define la función de costo g(n) y guía la heurística h(n).
    alpha : float in [0, 1]
        Peso de semanas en el criterio "balance" (default 0.5).
    max_nodos : int
        Límite de seguridad de nodos expandidos (evita explosión de memoria).

    Garantías
    ---------
    Óptimo para "semanas" y "cursos" dado que la heurística es admisible.
    Para "balance" también es óptimo con la heurística combinada.
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

        # Nodo obsoleto (g_score actualizado con un camino mejor)
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

        # Determinar cursos ya tomados (sus habilidades están todas en hab)
        cursos_tomados = _cursos_tomados_desde_estado(hab, grafo)

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
        tiempo_segundos=time.perf_counter() - inicio,
        exito=False,
    )


# ── Greedy Best-First ──────────────────────────────────────────────────────────

def greedy(
    grafo: GrafoCursos,
    habilidades_iniciales: FrozenSet[str],
    perfil_id: str,
    instancia_id: str = "?",
) -> SearchResult:
    """
    Greedy Best-First: f(n) = h(n) = |H* \\ s|.

    Elige en cada paso el estado con menos habilidades del objetivo pendientes.
    Rápido y con bajo consumo de memoria, pero no garantiza optimalidad.
    Sirve como variante de comparación experimental frente a A*.
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
            heapq.heappush(frontera, _Nodo(
                f=h, tiebreak=counter, g=nuevo_g, habilidades=nuevas_hab,
            ))

    return SearchResult(
        algoritmo="Greedy", criterio="heuristica",
        instancia_id=instancia_id, perfil_objetivo=perfil_id,
        trayectoria=[], costo_total_semanas=0, num_cursos=0,
        nodos_expandidos=nodos_expandidos,
        tiempo_segundos=time.perf_counter() - inicio,
        exito=False,
    )


# ── K mejores trayectorias ─────────────────────────────────────────────────────

class _GrafoRestringido:
    """
    Adaptador que presenta el mismo contrato que GrafoCursos
    pero excluye un subconjunto de cursos bloqueados.

    Implementa todos los métodos que astar() necesita:
    cursos_disponibles, aplicar_curso, es_objetivo,
    habilidades_faltantes (usada internamente por _heuristica).
    """

    def __init__(self, grafo: GrafoCursos, bloqueados: set[str]):
        self.cursos    = {cid: c for cid, c in grafo.cursos.items()
                          if cid not in bloqueados}
        self.perfiles  = grafo.perfiles
        self.habilidades = grafo.habilidades

    def cursos_disponibles(
        self,
        habilidades_actuales: FrozenSet[str],
        cursos_tomados: FrozenSet[str],
    ) -> list[Curso]:
        return [
            c for c in self.cursos.values()
            if (c.id not in cursos_tomados
                and c.prerrequisitos.issubset(habilidades_actuales)
                and not c.habilidades.issubset(habilidades_actuales))
        ]

    def aplicar_curso(
        self,
        habilidades_actuales: FrozenSet[str],
        curso: Curso,
    ) -> FrozenSet[str]:
        return habilidades_actuales | curso.habilidades

    def es_objetivo(
        self,
        habilidades_actuales: FrozenSet[str],
        perfil_id: str,
    ) -> bool:
        return self.perfiles[perfil_id].habilidades_requeridas.issubset(habilidades_actuales)

    def habilidades_faltantes(
        self,
        habilidades_actuales: FrozenSet[str],
        perfil_id: str,
    ) -> FrozenSet[str]:
        return self.perfiles[perfil_id].habilidades_requeridas - habilidades_actuales


def k_mejores(
    grafo: GrafoCursos,
    habilidades_iniciales: FrozenSet[str],
    perfil_id: str,
    instancia_id: str = "?",
    k: int = 3,
    criterio: str = "semanas",
    max_nodos_alternativa: int = 50_000,
) -> list[SearchResult]:
    """
    Genera hasta k trayectorias distintas.

    Estrategia: ejecuta A* estándar para la primera solución; luego
    bloquea iterativamente el curso de mayor costo de la trayectoria
    previa y re-ejecuta A* sobre el grafo restringido, asegurando
    diversidad de caminos.

    Parámetros
    ----------
    k                      : número máximo de trayectorias a generar.
    criterio               : métrica de optimización (mismo que astar).
    max_nodos_alternativa  : límite de nodos para búsquedas alternativas.

    Devuelve lista de SearchResult ordenada por costo ascendente.
    """
    inicio = time.perf_counter()
    nodos_total = 0

    # Primera solución: A* sin restricciones
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

        # Candidatos: cursos de la última trayectoria, de mayor a menor costo
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
                gr, habilidades_iniciales, perfil_id,
                instancia_id, criterio,
                max_nodos=max_nodos_alternativa,
            )
            nodos_total += r_alt.nodos_expandidos

            if not r_alt.exito:
                continue

            # Descartar si es idéntica a una solución ya encontrada
            ids_nueva = frozenset(c.id for c in r_alt.trayectoria)
            if any(frozenset(c.id for c in s.trayectoria) == ids_nueva
                   for s in soluciones):
                continue

            if r_alt.costo_total_semanas < mejor_costo:
                mejor_costo = r_alt.costo_total_semanas
                mejor_alternativa = r_alt
                mejor_bloqueado = curso_bloquear.id

        if mejor_alternativa is None or mejor_bloqueado is None:
            break

        bloqueados_global.add(mejor_bloqueado)
        soluciones.append(SearchResult(
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
        ))

    return soluciones


# ── Validador ──────────────────────────────────────────────────────────────────

def validar_trayectoria(
    grafo: GrafoCursos,
    trayectoria: list[Curso],
    habilidades_iniciales: FrozenSet[str],
    perfil_id: str,
) -> tuple[bool, str]:
    """
    Verifica dos condiciones:
      1. Cada curso cumple sus prerrequisitos en el momento de tomarse.
      2. Al finalizar la trayectoria, el perfil objetivo está satisfecho.

    Devuelve (True, "OK") o (False, mensaje_de_error).
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