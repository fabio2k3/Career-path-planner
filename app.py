"""
app.py
------
Interfaz visual del sistema Career Path Planner.

Uso:
    streamlit run app.py

Requiere:
    pip install streamlit
"""

import sys
import time
from pathlib import Path

# Añadir src/ al path para importar los módulos del proyecto
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from graph import GrafoCursos
from search import astar, greedy, validar_trayectoria
from llm_integration import pipeline_completo


# ── Configuración de página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Career Path Planner",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS personalizado ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Fuentes */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Fondo y tema oscuro */
    .stApp {
        background-color: #0B0F1A;
        color: #E2E8F0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F1623;
        border-right: 1px solid #1E2D45;
    }

    [data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }

    /* Header principal */
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }

    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        color: #64748B;
        font-weight: 300;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    /* Tarjeta de curso */
    .course-card {
        background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
        border: 1px solid #1E2D45;
        border-left: 3px solid #3B82F6;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        transition: border-left-color 0.2s;
    }

    .course-card:hover {
        border-left-color: #60A5FA;
    }

    .course-num {
        font-family: 'Syne', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: #3B82F6;
        min-width: 28px;
        text-align: center;
        background: rgba(59,130,246,0.1);
        border-radius: 4px;
        padding: 2px 6px;
    }

    .course-name {
        font-size: 0.9rem;
        font-weight: 500;
        color: #E2E8F0;
        flex: 1;
    }

    .course-meta {
        font-size: 0.75rem;
        color: #475569;
        white-space: nowrap;
    }

    /* Badge de nivel */
    .badge-pri {
        background: rgba(52,211,153,0.15);
        color: #34D399;
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-int {
        background: rgba(251,191,36,0.15);
        color: #FBBF24;
        border: 1px solid rgba(251,191,36,0.3);
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .badge-ava {
        background: rgba(239,68,68,0.15);
        color: #F87171;
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Métricas personalizadas */
    .metric-box {
        background: #111827;
        border: 1px solid #1E2D45;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #60A5FA;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.25rem;
    }

    /* Score bar LLM */
    .score-bar-container {
        background: #1E2D45;
        border-radius: 999px;
        height: 8px;
        width: 100%;
        margin: 0.5rem 0;
        overflow: hidden;
    }

    .score-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #3B82F6, #A78BFA);
        transition: width 1s ease;
    }

    /* Sección de evaluación */
    .eval-section {
        background: #111827;
        border: 1px solid #1E2D45;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }

    .eval-section-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.6rem;
    }

    .eval-item {
        font-size: 0.85rem;
        color: #94A3B8;
        padding: 0.3rem 0;
        border-bottom: 1px solid #1E2D45;
        line-height: 1.5;
    }

    .eval-item:last-child {
        border-bottom: none;
    }

    /* Botón principal */
    .stButton > button {
        background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    /* Input y selectbox */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background-color: #111827 !important;
        border: 1px solid #1E2D45 !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
    }

    .stSelectbox > div > div:focus-within {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
    }

    /* Resumen final */
    .summary-card {
        background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(167,139,250,0.08));
        border: 1px solid rgba(59,130,246,0.2);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #CBD5E1;
        line-height: 1.6;
        font-style: italic;
    }

    /* Divider */
    hr {
        border-color: #1E2D45 !important;
        margin: 1.5rem 0 !important;
    }

    /* Ocultar elementos de Streamlit */
    #MainMenu, footer, header { visibility: hidden; }

    /* Radio buttons */
    .stRadio > div {
        gap: 0.5rem;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #3B82F6 !important;
    }

    /* Alert/info boxes */
    .stAlert {
        background-color: #111827 !important;
        border: 1px solid #1E2D45 !important;
        color: #94A3B8 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Carga del grafo (cacheado) ────────────────────────────────────────────────

@st.cache_resource
def cargar_grafo():
    return GrafoCursos()


# ── Helpers de renderizado ────────────────────────────────────────────────────

def _badge_nivel(nivel: str) -> str:
    clases = {
        "principiante": "badge-pri",
        "intermedio":   "badge-int",
        "avanzado":     "badge-ava",
    }
    labels = {
        "principiante": "PRI",
        "intermedio":   "INT",
        "avanzado":     "AVA",
    }
    cls = clases.get(nivel, "badge-pri")
    lbl = labels.get(nivel, nivel[:3].upper())
    return f'<span class="{cls}">{lbl}</span>'


def renderizar_trayectoria(trayectoria: list) -> None:
    for i, curso in enumerate(trayectoria, 1):
        badge = _badge_nivel(curso.nivel)
        st.markdown(f"""
        <div class="course-card">
            <span class="course-num">{i:02d}</span>
            {badge}
            <span class="course-name">{curso.nombre}</span>
            <span class="course-meta">{curso.duracion_semanas} sem</span>
        </div>
        """, unsafe_allow_html=True)


def renderizar_metricas(num_cursos: int, semanas: int, nodos: int, tiempo: float) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{num_cursos}</div>
            <div class="metric-label">Cursos</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{semanas}</div>
            <div class="metric-label">Semanas</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{nodos:,}</div>
            <div class="metric-label">Nodos expandidos</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{tiempo:.3f}s</div>
            <div class="metric-label">Tiempo búsqueda</div>
        </div>""", unsafe_allow_html=True)


def renderizar_evaluacion_llm(evaluacion: dict) -> None:
    puntuacion  = evaluacion.get("puntuacion", 0)
    calidad     = evaluacion.get("nivel_calidad", "?").upper()
    fortalezas  = evaluacion.get("fortalezas", [])
    debilidades = evaluacion.get("debilidades", [])
    sugerencias = evaluacion.get("sugerencias", [])
    resumen     = evaluacion.get("resumen", "")
    modo        = evaluacion.get("modo", "llm_real")

    # Color según calidad
    colores = {
        "EXCELENTE": "#34D399",
        "BUENO":     "#60A5FA",
        "ACEPTABLE": "#FBBF24",
        "DEFICIENTE": "#F87171",
    }
    color = colores.get(calidad, "#60A5FA")
    pct   = int(puntuacion) * 10

    sufijo = " · Evaluación simulada" if modo == "simulado" else ""

    st.markdown(f"""
    <div style="margin-bottom:1rem;">
        <div style="display:flex; align-items:baseline; gap:0.75rem; margin-bottom:0.4rem;">
            <span style="font-family:'Syne',sans-serif; font-size:2.5rem; font-weight:800; color:{color};">{puntuacion}/10</span>
            <span style="font-size:0.8rem; color:{color}; font-weight:600; text-transform:uppercase; letter-spacing:0.08em;">{calidad}{sufijo}</span>
        </div>
        <div class="score-bar-container">
            <div class="score-bar-fill" style="width:{pct}%; background:linear-gradient(90deg, {color}aa, {color});"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        items = "".join(f'<div class="eval-item">✓ {f}</div>' for f in fortalezas)
        st.markdown(f"""
        <div class="eval-section">
            <div class="eval-section-title" style="color:#34D399;">Fortalezas</div>
            {items}
        </div>""", unsafe_allow_html=True)

        items_s = "".join(f'<div class="eval-item">→ {s}</div>' for s in sugerencias)
        st.markdown(f"""
        <div class="eval-section">
            <div class="eval-section-title" style="color:#A78BFA;">Sugerencias</div>
            {items_s}
        </div>""", unsafe_allow_html=True)

    with col_b:
        items_d = "".join(f'<div class="eval-item">✗ {d}</div>' for d in debilidades)
        st.markdown(f"""
        <div class="eval-section">
            <div class="eval-section-title" style="color:#F87171;">Debilidades</div>
            {items_d}
        </div>""", unsafe_allow_html=True)

    if resumen:
        st.markdown(f'<div class="summary-card">"{resumen}"</div>', unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar(grafo: GrafoCursos) -> tuple[str, bool, str | None]:
    """
    Renderiza el sidebar y devuelve:
      (algoritmo, usar_llm, perfil_id_seleccionado_o_None)
    """
    with st.sidebar:
        st.markdown("""
        <div style="margin-bottom:1.5rem;">
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;
                        color:#E2E8F0; margin-bottom:0.25rem;">⚙️ Configuración</div>
            <div style="font-size:0.75rem; color:#475569;">Ajusta el comportamiento del sistema</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Algoritmo de búsqueda**")
        algoritmo = st.radio(
            label="algoritmo",
            options=["A* (óptimo)", "Greedy (rápido)"],
            index=0,
            label_visibility="collapsed",
        )
        alg_key = "astar" if "A*" in algoritmo else "greedy"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Modo**")
        usar_llm = st.toggle("Usar LLM", value=True)

        if usar_llm:
            st.caption("El LLM interpreta tu objetivo y evalúa la trayectoria.")
        else:
            st.caption("Sin LLM: selecciona el perfil manualmente del catálogo.")

        perfil_seleccionado = None
        if not usar_llm:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Perfil objetivo**")
            opciones = {
                f"{p.nombre}": pid
                for pid, p in grafo.perfiles.items()
            }
            nombre_sel = st.selectbox(
                label="perfil",
                options=list(opciones.keys()),
                label_visibility="collapsed",
            )
            perfil_seleccionado = opciones[nombre_sel]

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72rem; color:#334155; line-height:1.6;">
            <div style="font-weight:600; color:#475569; margin-bottom:0.3rem;">DATASET ACTIVO</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"🎓 {len(grafo.cursos)} cursos")
        st.caption(f"🧠 {len(grafo.habilidades)} habilidades")
        st.caption(f"👤 {len(grafo.perfiles)} perfiles")

    return alg_key, usar_llm, perfil_seleccionado


# ── Página principal ──────────────────────────────────────────────────────────

def main() -> None:
    grafo = cargar_grafo()

    # Header
    st.markdown("""
    <div style="padding: 1.5rem 0 1rem 0;">
        <div class="hero-title">Career Path Planner</div>
        <div class="hero-sub">Sistema de Planificación de Trayectoria Profesional · IA & Simulación</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    alg_key, usar_llm, perfil_seleccionado = sidebar(grafo)

    # Input principal
    col_input, col_btn = st.columns([5, 1], gap="small")
    with col_input:
        objetivo = st.text_input(
            label="objetivo",
            placeholder="Ej: Quiero trabajar como ML Engineer desplegando modelos en producción...",
            label_visibility="collapsed",
        )
    with col_btn:
        ejecutar = st.button("Planificar →")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Ejecución ─────────────────────────────────────────────────────────────

    if not ejecutar:
        # Estado vacío — instrucciones
        st.markdown("""
        <div style="text-align:center; padding: 3rem 0; color:#1E2D45;">
            <div style="font-size:3rem; margin-bottom:1rem;">🎯</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:600;
                        color:#1E2D45;">Escribe tu objetivo profesional y pulsa Planificar</div>
            <div style="font-size:0.85rem; color:#1E2D45; margin-top:0.5rem;">
                El sistema encontrará la secuencia óptima de cursos para alcanzarlo
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if not objetivo.strip():
        st.warning("Escribe un objetivo profesional antes de planificar.")
        return

    objetivo = objetivo.strip()

    # ── Modo con LLM ──────────────────────────────────────────────────────────
    if usar_llm:
        algoritmo_fn = astar if alg_key == "astar" else greedy

        col_res, col_eval = st.columns([1, 1], gap="large")

        with col_res:
            st.markdown("""
            <div style="font-family:'Syne',sans-serif; font-size:0.75rem; font-weight:700;
                        text-transform:uppercase; letter-spacing:0.1em; color:#475569;
                        margin-bottom:1rem;">Trayectoria óptima</div>
            """, unsafe_allow_html=True)

            with st.spinner("Analizando objetivo con LLM..."):
                resultado = pipeline_completo(
                    objetivo_texto=objetivo,
                    grafo=grafo,
                    algoritmo_fn=algoritmo_fn,
                    habilidades_iniciales=frozenset(),
                    instancia_id="app_pipeline",
                )

            if not resultado["exito_total"]:
                st.error("No se pudo completar el pipeline. Verifica tu HF_API_KEY en el archivo .env.")
                return

            busqueda   = resultado["paso2_busqueda"]
            evaluacion = resultado["paso3_evaluacion"]
            parseo     = resultado["paso1_parseo"]

            # Perfil detectado
            perfil_nombre = grafo.perfiles[busqueda["perfil_objetivo"]].nombre
            st.markdown(f"""
            <div style="margin-bottom:1rem; padding:0.6rem 1rem;
                        background:rgba(59,130,246,0.08); border-radius:8px;
                        border:1px solid rgba(59,130,246,0.2);">
                <span style="font-size:0.75rem; color:#475569; text-transform:uppercase;
                             letter-spacing:0.08em;">Perfil detectado · </span>
                <span style="font-size:0.9rem; font-weight:600; color:#60A5FA;">{perfil_nombre}</span>
                <span style="font-size:0.75rem; color:#334155;"> · confianza {parseo.get('confianza','?')}</span>
            </div>
            """, unsafe_allow_html=True)

            # Reconstruir lista de Curso desde trayectoria_ids
            trayectoria_obj = [
                grafo.cursos[cid]
                for cid in busqueda["trayectoria_ids"]
            ]

            renderizar_trayectoria(trayectoria_obj)

            st.markdown("<br>", unsafe_allow_html=True)
            renderizar_metricas(
                busqueda["num_cursos"],
                busqueda["costo_total_semanas"],
                busqueda["nodos_expandidos"],
                busqueda["tiempo_segundos"],
            )

        with col_eval:
            st.markdown("""
            <div style="font-family:'Syne',sans-serif; font-size:0.75rem; font-weight:700;
                        text-transform:uppercase; letter-spacing:0.1em; color:#475569;
                        margin-bottom:1rem;">Evaluación LLM</div>
            """, unsafe_allow_html=True)
            renderizar_evaluacion_llm(evaluacion)

    # ── Modo sin LLM ──────────────────────────────────────────────────────────
    else:
        if not perfil_seleccionado:
            st.warning("Selecciona un perfil en el sidebar.")
            return

        algoritmo_fn = astar if alg_key == "astar" else greedy

        with st.spinner("Buscando trayectoria óptima..."):
            r = algoritmo_fn(
                grafo,
                frozenset(),
                perfil_seleccionado,
                "app_sin_llm",
                criterio="cursos",
            )

        if not r.exito:
            st.error("No se encontró trayectoria válida para este perfil.")
            return

        ok, msg = validar_trayectoria(
            grafo, r.trayectoria, frozenset(), perfil_seleccionado
        )

        perfil_nombre = grafo.perfiles[perfil_seleccionado].nombre

        st.markdown(f"""
        <div style="font-family:'Syne',sans-serif; font-size:0.75rem; font-weight:700;
                    text-transform:uppercase; letter-spacing:0.1em; color:#475569;
                    margin-bottom:0.75rem;">
            Trayectoria hacia <span style="color:#60A5FA;">{perfil_nombre}</span>
        </div>
        """, unsafe_allow_html=True)

        col_tray, col_info = st.columns([3, 2], gap="large")

        with col_tray:
            renderizar_trayectoria(r.trayectoria)

        with col_info:
            st.markdown("<br>", unsafe_allow_html=True)
            renderizar_metricas(
                r.num_cursos,
                r.costo_total_semanas,
                r.nodos_expandidos,
                r.tiempo_segundos,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if ok:
                st.success("✓ Trayectoria válida — todos los prerrequisitos respetados")
            else:
                st.error(f"✗ {msg}")

            st.markdown(f"""
            <div style="margin-top:1rem; padding:1rem;
                        background:#111827; border:1px solid #1E2D45;
                        border-radius:10px; font-size:0.8rem; color:#475569;">
                <div style="font-weight:600; color:#334155; margin-bottom:0.5rem;
                            text-transform:uppercase; font-size:0.7rem; letter-spacing:0.08em;">
                    Modo sin LLM
                </div>
                En este modo el sistema busca la trayectoria óptima directamente
                sobre el perfil seleccionado, sin interpretar lenguaje natural
                ni evaluar la calidad semánticamente.
                <br><br>
                Activa el toggle <strong style="color:#60A5FA;">Usar LLM</strong>
                en el sidebar para obtener análisis semántico del objetivo
                y evaluación de la trayectoria.
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
