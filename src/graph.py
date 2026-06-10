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

# *** Rutas del proyecto ******

# Directorio base del proyecto, calculado a partir de la ubicación de este archivo.
BASE_DIR = Path(__file__).resolve().parent.parent

# Se prioriza el dataset sintético si existe; en caso contrario, se usa el original.
_SYNTH = BASE_DIR / "data" / "dataset.json"
_BASE = BASE_DIR / "data" / "dataset_old.json"


def _resolver_dataset() -> Path:
    """
    Devuelve la ruta del dataset disponible.

    Se prioriza el dataset sintético por ser más útil para pruebas y
    experimentación. Si no existe, se utiliza el dataset base.
    """
    if _SYNTH.exists():
        return _SYNTH
    if _BASE.exists():
        return _BASE

    raise FileNotFoundError(
        f"No se encontró ningún dataset en {BASE_DIR / 'data'}. "
        "Genera primero dataset.json o dataset_sintetico.json."
    )


# Ruta final del dataset que será cargado por defecto.
DATASET_PATH = _resolver_dataset()

# Ruta de las instancias de evaluación o prueba.
INSTANCES_PATH = BASE_DIR / "data" / "instances" / "instances.json"


# *** Estructuras de datos ******

@dataclass(frozen=True)
class Curso:
    """
    Representa un nodo del DAG correspondiente a un curso del catálogo.
    """
    id: str
    nombre: str
    descripcion: str
    prerrequisitos: FrozenSet[str]  # Habilidades necesarias antes de tomarlo.
    habilidades: FrozenSet[str]     # Habilidades que el curso enseña.
    duracion_semanas: int
    nivel: str                      # principiante | intermedio | avanzado

    def __repr__(self) -> str:
        return f"Curso({self.id}: {self.nombre})"


@dataclass(frozen=True)
class PerfilProfesional:
    """
    Representa un perfil objetivo definido por un conjunto de habilidades requeridas.
    """
    id: str
    nombre: str
    descripcion: str
    habilidades_requeridas: FrozenSet[str]

    def __repr__(self) -> str:
        return f"Perfil({self.nombre})"


@dataclass(frozen=True)
class Instancia:
    """
    Representa una instancia de prueba con un punto de partida y un perfil objetivo.
    """
    id: str
    descripcion: str
    habilidades_iniciales: FrozenSet[str]
    perfil_objetivo: str
    objetivo_texto: str

    def __repr__(self) -> str:
        return f"Instancia({self.id}: {self.descripcion})"


# *** Grafo principal ******


class GrafoCursos:
    """
    Grafo dirigido acíclico (DAG) de cursos y habilidades.

    Modelo de planificación:
    - Estado de búsqueda: conjunto inmutable de habilidades adquiridas.
    - Acciones: cursos cuyas precondiciones ya se satisfacen.
    - Transición: nuevo estado tras incorporar las habilidades del curso.
    - Objetivo: alcanzar un conjunto de habilidades que contenga el perfil meta.
    """

    def __init__(self, dataset_path: Optional[Path] = None):
        # Diccionario de cursos indexado por su identificador.
        self.cursos: dict[str, Curso] = {}

        # Diccionario de perfiles profesionales indexado por su identificador.
        self.perfiles: dict[str, PerfilProfesional] = {}

        # Conjunto global de habilidades presentes en el dataset.
        self.habilidades: set[str] = set()

        # Permite inyectar un dataset alternativo; si no se indica, se usa el predeterminado.
        self._dataset_path = dataset_path or DATASET_PATH

        # Carga inicial del dataset en memoria.
        self._cargar_dataset()


    # *** Carga ******

    def _cargar_dataset(self) -> None:
        """
        Carga el dataset desde JSON y construye la estructura interna en memoria.
        """
        with open(self._dataset_path, encoding="utf-8") as f:
            data = json.load(f)

        # Registro de todas las habilidades disponibles en el sistema.
        self.habilidades = set(data["habilidades"])

        # Conversión de cada curso del JSON a una instancia inmutable de Curso.
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

        # Conversión de cada perfil profesional del JSON a una instancia inmutable.
        for pid, p in data["perfiles_profesionales"].items():
            perfil = PerfilProfesional(
                id=pid,
                nombre=p["nombre"],
                descripcion=p["descripcion"],
                habilidades_requeridas=frozenset(p["habilidades_requeridas"]),
            )
            self.perfiles[pid] = perfil


    # *** Lógica del espacio de estados ******

    def cursos_disponibles(self, habilidades_actuales: FrozenSet[str], cursos_tomados: FrozenSet[str],) -> list[Curso]:
        """
        Devuelve los cursos aplicables en el estado actual.

        Un curso se considera disponible si:
        1. Sus prerrequisitos ya están satisfechos.
        2. No ha sido tomado previamente.
        3. Aporta al menos una habilidad nueva, evitando acciones triviales.
        """
        return [
            curso for curso in self.cursos.values()
            if curso.id not in cursos_tomados
            and curso.prerrequisitos.issubset(habilidades_actuales)
            and not curso.habilidades.issubset(habilidades_actuales)
        ]

    def aplicar_curso(self, habilidades_actuales: FrozenSet[str], curso: Curso,) -> FrozenSet[str]:
        """
        Calcula el nuevo estado tras tomar un curso.

        La transición se modela como la unión del conjunto actual de habilidades
        con las habilidades aportadas por el curso.
        """
        return habilidades_actuales | curso.habilidades

    def es_objetivo(self, habilidades_actuales: FrozenSet[str], perfil_id: str,) -> bool:
        """
        Indica si el estado actual satisface completamente el perfil objetivo.
        """
        perfil = self._get_perfil(perfil_id)
        return perfil.habilidades_requeridas.issubset(habilidades_actuales)

    def habilidades_faltantes(self, habilidades_actuales: FrozenSet[str], perfil_id: str,) -> FrozenSet[str]:
        """
        Devuelve las habilidades del objetivo que aún no han sido adquiridas.
        """
        perfil = self._get_perfil(perfil_id)
        return perfil.habilidades_requeridas - habilidades_actuales


    # *** Heurística ******

    def heuristica(self, habilidades_actuales: FrozenSet[str], perfil_id: str,) -> int:
        """
        Heurística admisible para A*.

        Definición:
            h(s) = |H* - H_adquiridas(s)|

        La heurística cuenta cuántas habilidades del perfil todavía faltan.
        Es admisible porque cada habilidad faltante requiere al menos una acción
        para poder ser obtenida. También es consistente bajo el modelo de costo
        unitario por curso.
        """
        return len(self.habilidades_faltantes(habilidades_actuales, perfil_id))

    # *** Utilidades internas ******

    def _get_perfil(self, perfil_id: str) -> PerfilProfesional:
        """
        Recupera un perfil profesional por su identificador.

        Lanza un ValueError con un mensaje descriptivo si el perfil no existe.
        """
        perfil = self.perfiles.get(perfil_id)
        if perfil is None:
            disponibles = sorted(self.perfiles.keys())
            raise ValueError(
                f"Perfil '{perfil_id}' no existe en el dataset. "
                f"Perfiles disponibles: {disponibles}"
            )
        return perfil

    def cargar_instancias(self) -> list[Instancia]:
        """
        Carga las instancias de prueba desde instances.json.

        Cada instancia referencia un perfil objetivo que debe existir en el dataset.
        """
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

            instancias.append(
                Instancia(
                    id=i["id"],
                    descripcion=i["descripcion"],
                    habilidades_iniciales=frozenset(i["habilidades_iniciales"]),
                    perfil_objetivo=perfil_id,
                    objetivo_texto=i["objetivo_texto"],
                )
            )

        return instancias

    def resumen_estado(self, habilidades_actuales: FrozenSet[str], perfil_id: str,) -> str:
        """
        Genera un resumen legible del progreso actual hacia el perfil objetivo.
        """
        faltantes = self.habilidades_faltantes(habilidades_actuales, perfil_id)
        perfil = self._get_perfil(perfil_id)
        adquiridas = habilidades_actuales & perfil.habilidades_requeridas

        return (
            f"Habilidades objetivo adquiridas : {len(adquiridas)}/{len(perfil.habilidades_requeridas)}\n"
            f"Habilidades faltantes           : {len(faltantes)}\n"
            f"Detalle faltantes               : {sorted(faltantes)}"
        )