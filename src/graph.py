"""
graph.py

Representa el grafo dirigido acíclico (DAG) de cursos y habilidades.

Proporciona:
  - Carga del dataset desde JSON.
  - Consulta de cursos disponibles dado un estado (conjunto de habilidades).
  - Cálculo del estado resultante al tomar un curso.
  - Verificación de si un estado satisface un perfil objetivo.

"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import FrozenSet, Optional


# ── Rutas ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
_SYNTH   = BASE_DIR / "data" / "dataset_sintetico.json"
_BASE    = BASE_DIR / "data" / "dataset.json"

def _resolver_dataset() -> Path:
    """Devuelve el dataset disponible (sintético preferido)."""
    if _SYNTH.exists():
        return _SYNTH
    if _BASE.exists():
        return _BASE
    raise FileNotFoundError(
        f"No se encontró ningún dataset en {BASE_DIR / 'data'}. "
        "Genera primero dataset.json o dataset_sintetico.json."
    )

DATASET_PATH   = _resolver_dataset()
INSTANCES_PATH = BASE_DIR / "data" / "instances" / "instances.json"


# ── Estructuras de datos ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Curso:
    """Nodo del DAG: un curso del catálogo."""
    id: str
    nombre: str
    descripcion: str
    prerrequisitos: FrozenSet[str]   # habilidades necesarias antes de tomarlo
    habilidades: FrozenSet[str]      # habilidades que enseña
    duracion_semanas: int
    nivel: str                       # principiante | intermedio | avanzado

    def __repr__(self) -> str:
        return f"Curso({self.id}: {self.nombre})"


@dataclass(frozen=True)
class PerfilProfesional:
    """Perfil objetivo: conjunto de habilidades que debe alcanzarse."""
    id: str
    nombre: str
    descripcion: str
    habilidades_requeridas: FrozenSet[str]

    def __repr__(self) -> str:
        return f"Perfil({self.nombre})"


@dataclass(frozen=True)
class Instancia:
    """Instancia de prueba: punto de partida + perfil objetivo."""
    id: str
    descripcion: str
    habilidades_iniciales: FrozenSet[str]
    perfil_objetivo: str
    objetivo_texto: str

    def __repr__(self) -> str:
        return f"Instancia({self.id}: {self.descripcion})"


# ── Grafo principal ───────────────────────────────────────────────────────────

class GrafoCursos:
    """
    Grafo dirigido acíclico (DAG) de cursos y habilidades.

    Estado de búsqueda : frozenset de habilidades adquiridas.
    Acciones           : cursos cuyas precondiciones están satisfechas.
    Transición         : estado ∪ habilidades_del_curso.
    Función objetivo   : H* ⊆ habilidades_actuales.
    """

    def __init__(self, dataset_path: Optional[Path] = None):
        self.cursos:      dict[str, Curso]             = {}
        self.perfiles:    dict[str, PerfilProfesional] = {}
        self.habilidades: set[str]                     = set()
        self._dataset_path = dataset_path or DATASET_PATH
        self._cargar_dataset()

    # ── Carga ─────────────────────────────────────────────────────────────────

    def _cargar_dataset(self) -> None:
        """Carga el dataset desde JSON y construye el grafo en memoria."""
        with open(self._dataset_path, encoding="utf-8") as f:
            data = json.load(f)

        self.habilidades = set(data["habilidades"])

        for c in data["cursos"]:
            curso = Curso(
                id=c["id"],
                nombre=c["nombre"],
                descripcion=c["descripcion"],
                prerrequisitos=frozenset(c["prerrequisitos"]),
                habilidades=frozenset(c["habilidades"]),
                duracion_semanas=c["duracion_semanas"],
                nivel=c["nivel"],
            )
            self.cursos[curso.id] = curso

        for pid, p in data["perfiles_profesionales"].items():
            perfil = PerfilProfesional(
                id=pid,
                nombre=p["nombre"],
                descripcion=p["descripcion"],
                habilidades_requeridas=frozenset(p["habilidades_requeridas"]),
            )
            self.perfiles[pid] = perfil

    # ── Lógica del espacio de estados ─────────────────────────────────────────

    def cursos_disponibles(
        self,
        habilidades_actuales: FrozenSet[str],
        cursos_tomados: FrozenSet[str],
    ) -> list[Curso]:
        """
        Cursos aplicables en el estado actual.
        Un curso es disponible si:
          1. Sus prerrequisitos están en habilidades_actuales.
          2. No ha sido tomado ya (está en cursos_tomados).
          3. Aporta al menos una habilidad nueva (acción no trivial).
        """
        return [
            curso for curso in self.cursos.values()
            if curso.id not in cursos_tomados
            and curso.prerrequisitos.issubset(habilidades_actuales)
            and not curso.habilidades.issubset(habilidades_actuales)
        ]

    def aplicar_curso(
        self,
        habilidades_actuales: FrozenSet[str],
        curso: Curso,
    ) -> FrozenSet[str]:
        """Transición de estado: s' = habilidades_actuales ∪ habilidades(curso)."""
        return habilidades_actuales | curso.habilidades

    def es_objetivo(
        self,
        habilidades_actuales: FrozenSet[str],
        perfil_id: str,
    ) -> bool:
        """True si el estado satisface el perfil: H* ⊆ habilidades_actuales."""
        perfil = self._get_perfil(perfil_id)
        return perfil.habilidades_requeridas.issubset(habilidades_actuales)

    def habilidades_faltantes(
        self,
        habilidades_actuales: FrozenSet[str],
        perfil_id: str,
    ) -> FrozenSet[str]:
        """Habilidades del objetivo que aún no han sido adquiridas."""
        perfil = self._get_perfil(perfil_id)
        return perfil.habilidades_requeridas - habilidades_actuales

    # ── Heurística ────────────────────────────────────────────────────────────

    def heuristica(
        self,
        habilidades_actuales: FrozenSet[str],
        perfil_id: str,
    ) -> int:
        """
        Heurística admisible para A*:
            h(s) = |H* \\ H_adq(s)|

        Cuenta las habilidades del objetivo que aún faltan.
        Admisible: cada habilidad faltante requiere al menos 1 curso (costo ≥ 1).
        Consistente: al tomar un curso que cubre k habilidades faltantes,
        g aumenta en 1 y h disminuye en ≤ k → f no decrece.
        """
        return len(self.habilidades_faltantes(habilidades_actuales, perfil_id))

    # ── Utilidades ────────────────────────────────────────────────────────────

    def _get_perfil(self, perfil_id: str) -> PerfilProfesional:
        """Devuelve el perfil o lanza ValueError con mensaje claro."""
        perfil = self.perfiles.get(perfil_id)
        if perfil is None:
            disponibles = sorted(self.perfiles.keys())
            raise ValueError(
                f"Perfil '{perfil_id}' no existe en el dataset. "
                f"Perfiles disponibles: {disponibles}"
            )
        return perfil

    def cargar_instancias(self) -> list[Instancia]:
        """Carga las instancias de prueba desde instances.json."""
        with open(INSTANCES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        instancias = []
        for i in data:
            perfil_id = i["perfil_objetivo"]
            if perfil_id not in self.perfiles:
                raise ValueError(
                    f"Instancia '{i['id']}' referencia perfil '{perfil_id}' "
                    f"que no existe en el dataset."
                )
            instancias.append(Instancia(
                id=i["id"],
                descripcion=i["descripcion"],
                habilidades_iniciales=frozenset(i["habilidades_iniciales"]),
                perfil_objetivo=perfil_id,
                objetivo_texto=i["objetivo_texto"],
            ))
        return instancias

    def resumen_estado(
        self,
        habilidades_actuales: FrozenSet[str],
        perfil_id: str,
    ) -> str:
        """Resumen legible del progreso hacia el objetivo."""
        faltantes  = self.habilidades_faltantes(habilidades_actuales, perfil_id)
        perfil     = self._get_perfil(perfil_id)
        adquiridas = habilidades_actuales & perfil.habilidades_requeridas
        return (
            f"Habilidades objetivo adquiridas : {len(adquiridas)}"
            f"/{len(perfil.habilidades_requeridas)}\n"
            f"Habilidades faltantes           : {len(faltantes)}\n"
            f"Detalle faltantes               : {sorted(faltantes)}"
        )