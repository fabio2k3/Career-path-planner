"""
search.py
---------
Algoritmos de búsqueda sobre el DAG de cursos.

  - astar(criterio="semanas")  : A* minimizando semanas totales
  - astar(criterio="cursos")   : A* minimizando número de cursos
  - astar(criterio="balance")  : A* con costo combinado α·semanas + β·cursos
  - greedy()                   : Greedy best-first (h = habilidades faltantes)
  - k_mejores()                : genera k trayectorias distintas ordenadas por costo

Optimización de memoria: punteros padre (came_from) en lugar de copiar
la trayectoria completa en cada nodo del heap.

Estado = frozenset de habilidades adquiridas.

Nota sobre cursos_tomados:
  Se infiere el conjunto de cursos tomados a partir del estado de habilidades,
  asumiendo que cada habilidad es enseñada por exactamente un curso en el dataset.
  Esta asunción es válida para el dataset actual. En datasets donde múltiples
  cursos enseñan las mismas habilidades, el estado debería extenderse a
  (habilidades, cursos_tomados_frozenset) para mayor precisión.
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import FrozenSet, Optional

from graph import GrafoCursos, Curso


# ─── Resultado ────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    algoritmo: str
    criterio: str
    instancia_id: str
    perfil_objetivo: str
    trayectoria: list          # lista de Curso en orden
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


# ─── Nodo interno del heap ────────────────────────────────────────────────────

@dataclass(order=True)
class _Nodo:
    f: float
    tiebreak: int
    g: float = field(compare=False)
    habilidades: FrozenSet[str] = field(compare=False)
    num_cursos: int = field(compare=False, default=0)


# ─── Reconstrucción del camino ────────────────────────────────────────────────

def _reconstruir(came_from: dict, estado_final: FrozenSet[str], grafo: GrafoCursos) -> list:
    """Reconstruye la lista ordenada de cursos desde came_from."""
    camino = []
    estado = estado_final
    while estado in came_from:
        curso_id, estado_prev = came_from[estado]
        camino.append(grafo.cursos[curso_id])
        estado = estado_prev
    camino.reverse()
    return camino


# ─── Función de costo según criterio ─────────────────────────────────────────

def _costo_curso(curso: Curso, criterio: str, alpha: float = 0.5) -> float:
    """
    Retorna el costo de tomar un curso según el criterio elegido.

      "semanas" : costo = duración en semanas  (minimiza tiempo total)
      "cursos"  : costo = 1                    (minimiza número de cursos)
      "balance" : costo = α·(semanas/10) + (1-α)·1  (criterio combinado)
    """
    if criterio == "semanas":
        return float(curso.duracion_semanas)
    elif criterio == "cursos":
        return 1.0
    elif criterio == "balance":
        return alpha * (curso.duracion_semanas / 10.0) + (1.0 - alpha) * 1.0
    raise ValueError(f"Criterio desconocido: '{criterio}'. Usa 'semanas', 'cursos' o 'balance'.")


# ─── Heurística admisible ─────────────────────────────────────────────────────

def _heuristica(grafo, habilidades: FrozenSet[str], perfil_id: str,
                criterio: str, alpha: float = 0.5) -> float:
    """
    Heurística admisible ajustada al criterio.

    h(s) = |H* \\ H_adq(s)| (habilidades del objetivo aún no adquiridas).
    Nunca sobreestima: cada habilidad faltante requiere ≥ 1 curso con costo ≥ 1.
    """
    faltantes = len(grafo.habilidades_faltantes(habilidades, perfil_id))
    if criterio == "semanas":
        return float(faltantes)
    elif criterio == "cursos":
        return float(faltantes)
    elif criterio == "balance":
        return alpha * (faltantes / 10.0) + (1.0 - alpha) * float(faltantes)
    return float(faltantes)


# ─── Inferencia de cursos tomados ─────────────────────────────────────────────

def _inferir_cursos_tomados(grafo: GrafoCursos, habilidades: FrozenSet[str]) -> FrozenSet[str]:
    """
    Infiere el conjunto de cursos ya tomados a partir del estado de habilidades.

    Un curso se considera tomado si: (1) sus prerrequisitos están cubiertos y
    (2) todas sus habilidades ya están adquiridas. Válido cuando cada habilidad
    es enseñada por un único curso en el dataset.
    """
    return frozenset(
        cid for cid, c in grafo.cursos.items()
        if c.prerrequisitos.issubset(habilidades)
        and c.habilidades.issubset(habilidades)
    )


# ─── A* genérico ──────────────────────────────────────────────────────────────

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

    Garantiza optimalidad con la heurística admisible h(s) = |H* \\ H_adq(s)|.
    Memoria eficiente: usa came_from en lugar de copiar caminos completos.

    Parámetros de criterio:
      criterio="semanas"  → minimiza semanas totales         (costo = duración)
      criterio="cursos"   → minimiza número de cursos        (costo = 1)
      criterio="balance"  → criterio combinado               (costo = α·sem + (1-α)·1)
      alpha               → peso de semanas en criterio balance (default 0.5)
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

        # Descartar nodos obsoletos (lazy deletion)
        if nodo.g > g_score.get(hab, float("inf")):
            continue

        # Test de objetivo
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

        # Inferir cursos ya tomados (no repetir cursos en la trayectoria)
        cursos_tomados = _inferir_cursos_tomados(grafo, hab)

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


# ─── Greedy Best-First ────────────────────────────────────────────────────────

def greedy(
    grafo: GrafoCursos,
    habilidades_iniciales: FrozenSet[str],
    perfil_id: str,
    instancia_id: str = "?",
) -> SearchResult:
    """
    Greedy Best-First: f(n) = h(n) = habilidades faltantes.
    Elige en cada paso el estado con menos habilidades del objetivo pendientes.
    Rápido (expande muy pocos nodos) pero no garantiza optimalidad.
    """
    inicio = time.perf_counter()
    counter = 0
    nodos_expandidos = 0

    h0 = float(len(grafo.habilidades_faltantes(habilidades_iniciales, perfil_id)))
    frontera: list = []
    heapq.heappush(frontera, _Nodo(
        f=h0, tiebreak=counter, g=0.0, habilidades=habilidades_iniciales,
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

        cursos_tomados = _inferir_cursos_tomados(grafo, hab)

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


# ─── K mejores trayectorias ───────────────────────────────────────────────────

def _grafo_sin_curso(grafo: GrafoCursos, curso_id_bloqueado: str):
    """
    Retorna un objeto grafo-compatible que excluye el curso indicado.
    Permite generar trayectorias alternativas sin modificar el grafo original.
    """
    cursos_filtrados = {cid: c for cid, c in grafo.cursos.items()
                        if cid != curso_id_bloqueado}

    class _GrafoRestringido:
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

        def es_objetivo(self, hab, pid):
            return self.perfiles[pid].habilidades_requeridas.issubset(hab)

        def habilidades_faltantes(self, hab, pid):
            return self.perfiles[pid].habilidades_requeridas - hab

    return _GrafoRestringido()


def k_mejores(
    grafo: GrafoCursos,
    habilidades_iniciales: FrozenSet[str],
    perfil_id: str,
    instancia_id: str = "?",
    k: int = 3,
    criterio: str = "semanas",
) -> list:
    """
    Genera hasta k trayectorias distintas mediante bloqueo iterativo de cursos.

    Estrategia heurística: en cada iteración se bloquea el curso de mayor
    duración de la última trayectoria y se ejecuta A* sobre el grafo restringido.
    No garantiza las k trayectorias óptimas globales, pero produce alternativas
    variadas y válidas con bajo coste computacional.

    Retorna lista de SearchResult ordenada por costo ascendente.
    """
    inicio = time.perf_counter()
    soluciones = []
    nodos_total = 0

    # Primera solución: A* sin restricciones
    r0 = astar(grafo, habilidades_iniciales, perfil_id, instancia_id, criterio)
    nodos_total += r0.nodos_expandidos
    if not r0.exito:
        return []
    soluciones.append(r0)

    cursos_bloqueados_global: set = set()

    for _ in range(1, k):
        mejor_alternativa: Optional[SearchResult] = None
        mejor_costo = float("inf")
        mejor_bloqueado: Optional[str] = None

        # Intentar bloquear cada curso de la última trayectoria (mayor duración primero)
        trayectoria_base = soluciones[-1].trayectoria
        candidatos = sorted(trayectoria_base, key=lambda c: c.duracion_semanas, reverse=True)

        for curso_bloquear in candidatos:
            if curso_bloquear.id in cursos_bloqueados_global:
                continue

            gt = _grafo_sin_curso(grafo, curso_bloquear.id)
            r_alt = astar(gt, habilidades_iniciales, perfil_id,
                          instancia_id, criterio, max_nodos=10_000)
            nodos_total += r_alt.nodos_expandidos

            if not r_alt.exito:
                continue

            # Verificar que es una trayectoria realmente diferente
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
        soluciones.append(mejor_alternativa)

    return soluciones


# ─── Validador de trayectorias ────────────────────────────────────────────────

def validar_trayectoria(
    grafo: GrafoCursos,
    trayectoria: list,
    habilidades_iniciales: FrozenSet[str],
    perfil_id: str,
) -> tuple:
    """
    Verifica la corrección de una trayectoria:
      1. Cada curso cumple sus prerrequisitos en el momento de tomarse.
      2. Al finalizar la trayectoria se satisface el perfil objetivo.

    Retorna (True, "OK") si es válida, (False, mensaje_error) si no lo es.
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