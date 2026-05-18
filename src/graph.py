"""
graph.py

Representación del grafo dirigido acíclico (DAG) de cursos y habilidades.
Provee la clase CourseGraph que carga el dataset y expone la interfaz
necesaria para los algoritmos de búsqueda.
"""

import json
from pathlib import Path
from collections import defaultdict


# ----- Rutas por defecto ------
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = BASE_DIR / "data" / "dataset.json"
DEFAULT_INSTANCES = BASE_DIR / "data" / "instances" / "instances.json"


class Course:
    """Representa un curso del catálogo."""

    def __init__(self, data: dict):
        self.id: str = data["id"]
        self.nombre: str = data["nombre"]
        self.descripcion: str = data["descripcion"]
        self.prerrequisitos: frozenset = frozenset(data["prerrequisitos"])
        self.habilidades: frozenset = frozenset(data["habilidades"])
        self.duracion_semanas: int = data["duracion_semanas"]
        self.nivel: str = data["nivel"]

    def es_disponible(self, habilidades_adquiridas: frozenset) -> bool:
        """
        Retorna True si el curso es aplicable en el estado actual.
        Un curso es disponible si todas sus habilidades prerrequisito
        ya han sido adquiridas por el usuario.
        """
        return self.prerrequisitos.issubset(habilidades_adquiridas)

    def __repr__(self):
        return f"Course({self.id}, '{self.nombre}', dur={self.duracion_semanas}w)"


class CourseGraph:
    """
    Grafo dirigido acíclico (DAG) de cursos y habilidades.

    Nodos     : habilidades (strings)
    Aristas   : prerrequisito → habilidad_enseñada (a través de un curso)
    Estado    : frozenset de habilidades adquiridas
    Acciones  : cursos cuyas precondiciones están satisfechas en el estado
    """

    def __init__(self, dataset_path: Path = DEFAULT_DATASET):
        self._cursos: dict[str, Course] = {}
        self._habilidades: list[str] = []
        self._perfiles: dict[str, dict] = {}
        self._cargar_dataset(dataset_path)

    # ----- Carga ------

    def _cargar_dataset(self, path: Path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        self._habilidades = data["habilidades"]

        for curso_data in data["cursos"]:
            curso = Course(curso_data)
            self._cursos[curso.id] = curso

        self._perfiles = data["perfiles_profesionales"]

    # ----- Acceso a cursos -----

    @property
    def cursos(self) -> list[Course]:
        return list(self._cursos.values())

    def get_curso(self, curso_id: str) -> Course:
        return self._cursos[curso_id]

    # ----- Acceso a perfiles ------

    def get_habilidades_objetivo(self, perfil_id: str) -> frozenset:
        """Retorna el conjunto de habilidades requeridas para un perfil."""
        if perfil_id not in self._perfiles:
            raise ValueError(
                f"Perfil '{perfil_id}' no encontrado. "
                f"Disponibles: {list(self._perfiles.keys())}"
            )
        return frozenset(self._perfiles[perfil_id]["habilidades_requeridas"])

    def get_nombre_perfil(self, perfil_id: str) -> str:
        return self._perfiles[perfil_id]["nombre"]

    def listar_perfiles(self) -> list[str]:
        return list(self._perfiles.keys())

    # ----- Lógica del espacio de estados -----

    def cursos_disponibles(
        self,
        habilidades_adquiridas: frozenset,
        cursos_tomados: frozenset
    ) -> list[Course]:
        """
        Devuelve los cursos que pueden tomarse en el estado actual:
        - Sus prerrequisitos están en habilidades_adquiridas.
        - No han sido tomados aún.
        - Aportan al menos una habilidad nueva (evita acciones inútiles).
        """
        disponibles = []
        for curso in self._cursos.values():
            if curso.id in cursos_tomados:
                continue
            if not curso.es_disponible(habilidades_adquiridas):
                continue
            # Solo incluir si aporta habilidades nuevas
            if not curso.habilidades.issubset(habilidades_adquiridas):
                disponibles.append(curso)
        return disponibles

    def aplicar_curso(
        self,
        habilidades_adquiridas: frozenset,
        curso: Course
    ) -> frozenset:
        """
        Aplica la acción (tomar el curso) y retorna el nuevo estado.
        Efecto: H_adq' = H_adq ∪ skills(curso)
        """
        return habilidades_adquiridas | curso.habilidades

    def es_objetivo(
        self,
        habilidades_adquiridas: frozenset,
        habilidades_objetivo: frozenset
    ) -> bool:
        """Retorna True si el estado actual satisface el objetivo."""
        return habilidades_objetivo.issubset(habilidades_adquiridas)

    def habilidades_faltantes(
        self,
        habilidades_adquiridas: frozenset,
        habilidades_objetivo: frozenset
    ) -> frozenset:
        """Retorna las habilidades del objetivo aún no adquiridas."""
        return habilidades_objetivo - habilidades_adquiridas

    # ----- Verificador de restricciones -----

    def verificar_trayectoria(
        self,
        trayectoria: list[str],
        habilidades_iniciales: frozenset
    ) -> tuple[bool, str]:
        """
        Verifica que una trayectoria (lista de IDs de cursos) es válida:
        - Cada curso existe en el catálogo.
        - En el momento de tomarse, sus prerrequisitos ya están satisfechos.

        Retorna (True, '') si es válida, o (False, motivo) si no.
        """
        habilidades = frozenset(habilidades_iniciales)
        tomados = set()

        for curso_id in trayectoria:
            if curso_id not in self._cursos:
                return False, f"Curso '{curso_id}' no existe en el catálogo."

            curso = self._cursos[curso_id]

            if curso_id in tomados:
                return False, f"Curso '{curso.nombre}' aparece duplicado en la trayectoria."

            if not curso.es_disponible(habilidades):
                faltantes = curso.prerrequisitos - habilidades
                return False, (
                    f"Curso '{curso.nombre}' tomado sin cumplir prerrequisitos. "
                    f"Faltaban: {faltantes}"
                )

            habilidades = self.aplicar_curso(habilidades, curso)
            tomados.add(curso_id)

        return True, ""

    # ------ Utilidades ------

    def resumen_trayectoria(
        self,
        trayectoria: list[str],
        habilidades_iniciales: frozenset,
        habilidades_objetivo: frozenset
    ) -> dict:
        """
        Genera un resumen legible de una trayectoria:
        cursos, duración total, habilidades adquiridas y cobertura del objetivo.
        """
        habilidades = frozenset(habilidades_iniciales)
        cursos_info = []
        duracion_total = 0

        for curso_id in trayectoria:
            curso = self._cursos[curso_id]
            cursos_info.append({
                "id": curso.id,
                "nombre": curso.nombre,
                "nivel": curso.nivel,
                "duracion_semanas": curso.duracion_semanas,
                "habilidades_nuevas": list(curso.habilidades - habilidades),
            })
            habilidades = self.aplicar_curso(habilidades, curso)
            duracion_total += curso.duracion_semanas

        habilidades_alcanzadas = habilidades_objetivo & habilidades
        cobertura = (
            len(habilidades_alcanzadas) / len(habilidades_objetivo) * 100
            if habilidades_objetivo else 0.0
        )

        return {
            "num_cursos": len(trayectoria),
            "duracion_total_semanas": duracion_total,
            "cobertura_objetivo_pct": round(cobertura, 1),
            "habilidades_objetivo_cubiertas": list(habilidades_alcanzadas),
            "habilidades_objetivo_faltantes": list(
                habilidades_objetivo - habilidades
            ),
            "cursos": cursos_info,
        }


# ----- Carga de instancias -----

def cargar_instancias(path: Path = DEFAULT_INSTANCES) -> list[dict]:
    """Carga las instancias de prueba desde el JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)