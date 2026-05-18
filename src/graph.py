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
from typing import FrozenSet


# ------ Rutas -----
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH  = BASE_DIR / "data" / "dataset.json"
INSTANCES_PATH = BASE_DIR / "data" / "instances" / "instances.json"


# ------ Estructuras de datos -----

@dataclass(frozen=True)
class Curso:
    """Representa un curso del catálogo."""
    id: str
    nombre: str
    descripcion: str
    prerrequisitos: FrozenSet[str]   # habilidades necesarias antes
    habilidades: FrozenSet[str]      # habilidades que enseña
    duracion_semanas: int
    nivel: str

    def __repr__(self):
        return f"Curso({self.id}: {self.nombre})"


@dataclass(frozen=True)
class PerfilProfesional:
    """Representa un perfil profesional objetivo."""
    id: str
    nombre: str
    descripcion: str
    habilidades_requeridas: FrozenSet[str]

    def __repr__(self):
        return f"Perfil({self.nombre})"


@dataclass(frozen=True)
class Instancia:
    """Representa una instancia del problema de búsqueda."""
    id: str
    descripcion: str
    habilidades_iniciales: FrozenSet[str]
    perfil_objetivo: str
    objetivo_texto: str

    def __repr__(self):
        return f"Instancia({self.id}: {self.descripcion})"


# ------- Grafo principal -------

class GrafoCursos:
    """
    Grafo dirigido acíclico (DAG) de cursos y habilidades.

    Estado de búsqueda : frozenset de habilidades adquiridas.
    Acciones           : cursos cuyas precondiciones están satisfechas.
    Transición         : estado ∪ habilidades_del_curso.
    """

    def __init__(self):
        self.cursos: dict[str, Curso] = {}
        self.perfiles: dict[str, PerfilProfesional] = {}
        self.habilidades: set[str] = set()
        self._cargar_dataset()

    # ------ Carga ------

    def _cargar_dataset(self):
        """Carga el dataset desde el archivo JSON."""
        with open(DATASET_PATH, encoding="utf-8") as f:
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

    # ----- Lógica del espacio de estados ------

    def cursos_disponibles(
        self,
        habilidades_actuales: FrozenSet[str],
        cursos_tomados: FrozenSet[str],
    ) -> list:
        """
        Devuelve los cursos aplicables en el estado actual.
        Un curso es disponible si:
          1. Sus prerrequisitos están todos en habilidades_actuales.
          2. No ha sido tomado anteriormente.
          3. Aporta al menos una habilidad nueva (evita acciones inútiles).
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
        """
        Transición de estado al tomar un curso.
        s' = habilidades_actuales ∪ skills(curso)
        """
        return habilidades_actuales | curso.habilidades

    def es_objetivo(
        self,
        habilidades_actuales: FrozenSet[str],
        perfil_id: str,
    ) -> bool:
        """
        Verifica si el estado actual satisface el perfil objetivo.
        Condición: H* ⊆ habilidades_actuales
        """
        return self.perfiles[perfil_id].habilidades_requeridas.issubset(habilidades_actuales)

    def habilidades_faltantes(
        self,
        habilidades_actuales: FrozenSet[str],
        perfil_id: str,
    ) -> FrozenSet[str]:
        """Habilidades del objetivo que aún no han sido adquiridas."""
        return self.perfiles[perfil_id].habilidades_requeridas - habilidades_actuales

    # ----- Heurística ------

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
        """
        return len(self.habilidades_faltantes(habilidades_actuales, perfil_id))

    # ----- Utilidades -----

    def cargar_instancias(self) -> list:
        """Carga las instancias de prueba desde el archivo JSON."""
        with open(INSTANCES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return [
            Instancia(
                id=i["id"],
                descripcion=i["descripcion"],
                habilidades_iniciales=frozenset(i["habilidades_iniciales"]),
                perfil_objetivo=i["perfil_objetivo"],
                objetivo_texto=i["objetivo_texto"],
            )
            for i in data
        ]

    def resumen_estado(
        self,
        habilidades_actuales: FrozenSet[str],
        perfil_id: str,
    ) -> str:
        """Genera un resumen legible del estado actual vs el objetivo."""
        faltantes  = self.habilidades_faltantes(habilidades_actuales, perfil_id)
        perfil     = self.perfiles[perfil_id]
        adquiridas = habilidades_actuales & perfil.habilidades_requeridas
        return (
            f"Habilidades objetivo adquiridas : {len(adquiridas)}"
            f"/{len(perfil.habilidades_requeridas)}\n"
            f"Habilidades faltantes           : {len(faltantes)}\n"
            f"Detalle faltantes               : {sorted(faltantes)}"
        )