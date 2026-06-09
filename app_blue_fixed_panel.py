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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
from graph import GrafoCursos
from search import astar, greedy, validar_trayectoria
from llm_integration import pipeline_completo


# ── Configuración de página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Career Path Planner",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Ocultar elementos por defecto de Streamlit y estilos base

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #07111f;
        --bg-2: #0b1728;
        --card: rgba(10, 20, 36, 0.88);
        --card-solid: #0c1728;
        --card-border: rgba(96, 165, 250, 0.16);
        --text: #eaf2ff;
        --muted: #9ab0cf;
        --muted-2: #6f86a8;
        --line: rgba(96, 165, 250, 0.16);
        --line-strong: rgba(96, 165, 250, 0.28);
        --accent: #60a5fa;
        --accent-soft: #38bdf8;
        --success: #2dd4bf;
        --warning: #60a5fa;
        --danger: #fb7185;
        --radius: 22px;
        --radius-sm: 16px;
        --shadow: 0 22px 60px rgba(2, 8, 23, 0.45);
        --shadow-soft: 0 14px 34px rgba(2, 8, 23, 0.28);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(96, 165, 250, 0.18), transparent 30%),
            radial-gradient(circle at top right, rgba(56, 189, 248, 0.10), transparent 26%),
            radial-gradient(circle at bottom left, rgba(37, 99, 235, 0.10), transparent 22%),
            linear-gradient(180deg, #07111f 0%, #0a1324 100%);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .cp-left-panel-shell {
        position: relative;
    }

    .cp-left-panel {
        position: sticky;
        top: 1.1rem;
        align-self: start;
        z-index: 20;
    }

    .cp-left-panel .cp-panel {
        padding: 1rem 1rem 1.1rem;
    }

    .cp-layout {
        align-items: start;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.4rem;
    }

    h1, h2, h3, h4, h5, h6 {
        letter-spacing: -0.04em;
    }

    p, label, span, div {
        color: var(--text);
    }

    hr {
        border-color: var(--line) !important;
        margin: 1rem 0 1.2rem 0 !important;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
        background: rgba(8, 17, 31, 0.88) !important;
        border: 1px solid rgba(96, 165, 250, 0.22) !important;
        border-radius: 16px !important;
        color: var(--text) !important;
        box-shadow: none !important;
        transition: border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
    }

    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stTextArea"] textarea::placeholder {
        color: var(--muted-2) !important;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextArea"] textarea:focus,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
    div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:focus-within {
        border-color: rgba(96, 165, 250, 0.45) !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.16) !important;
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stToggle"] label {
        color: var(--muted) !important;
        font-weight: 600 !important;
    }

    .stButton > button {
        background: linear-gradient(180deg, #3b82f6 0%, #60A5FA 100%) !important;
        color: #f8fbff !important;
        border: 1px solid rgba(147, 197, 253, 0.15) !important;
        border-radius: 16px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        height: 3.05rem !important;
        padding: 0 1.4rem !important;
        box-shadow: 0 14px 30px rgba(37, 99, 235, 0.28);
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        filter: brightness(1.03);
        box-shadow: 0 18px 40px rgba(37, 99, 235, 0.35);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }


    .streamlit-expanderHeader {
        background: rgba(8, 17, 31, 0.88) !important;
        border: 1px solid rgba(96, 165, 250, 0.16) !important;
        border-radius: 16px !important;
        color: var(--text) !important;
        font-size: 0.86rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stExpander"] details {
        border-radius: 16px !important;
        overflow: hidden;
    }

    div[data-testid="stExpander"] div[role="button"] {
        padding: 0.9rem 1rem !important;
    }

    .stAlert {
        border-radius: 18px !important;
        border: 1px solid rgba(96, 165, 250, 0.16) !important;
        box-shadow: var(--shadow-soft);
    }

    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(2, 8, 23, 0.18);
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(96, 165, 250, 0.30);
        border-radius: 999px;
        border: 2px solid transparent;
        background-clip: content-box;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(96, 165, 250, 0.46);
        border: 2px solid transparent;
        background-clip: content-box;
    }

    .cp-panel {
        background: var(--card);
        border: 1px solid var(--card-border);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
    }

    .cp-panel-soft {
        background: rgba(8, 17, 31, 0.72);
        border: 1px solid var(--line);
        border-radius: var(--radius-sm);
        box-shadow: var(--shadow-soft);
    }

    .cp-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.38rem 0.75rem;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(96, 165, 250, 0.18);
        color: #bfdbfe;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .cp-subtle { color: var(--muted); }

    .cp-title {
        font-size: 2.2rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.06em;
        margin: 0.15rem 0 0.55rem 0;
        color: var(--text);
    }

    .cp-desc {
        font-size: 0.96rem;
        line-height: 1.55;
        color: var(--muted);
        margin: 0;
        max-width: 58rem;
    }

    .cp-hero {
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
        border-radius: 24px;
        background:
            linear-gradient(180deg, rgba(8, 17, 31, 0.95) 0%, rgba(12, 24, 44, 0.90) 100%);
        border: 1px solid rgba(96, 165, 250, 0.16);
        box-shadow: var(--shadow-soft);
    }

    .cp-empty {
        text-align: center;
        padding: 4.2rem 1rem 4.6rem;
        border: 1px dashed rgba(96, 165, 250, 0.22);
        border-radius: 28px;
        background: rgba(8, 17, 31, 0.72);
    }

    .cp-empty h4 {
        margin: 0.3rem 0 0.35rem;
        font-size: 1.05rem;
        color: var(--text);
    }

    .cp-empty p {
        margin: 0;
        color: var(--muted);
        font-size: 0.92rem;
    }

    .cp-metric {
        padding: 1rem 1rem 0.95rem;
        background: rgba(8, 17, 31, 0.88);
        border: 1px solid rgba(96, 165, 250, 0.16);
        border-radius: 18px;
        text-align: center;
        box-shadow: var(--shadow-soft);
    }

    .cp-metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        line-height: 1;
        color: var(--text);
    }

    .cp-metric-label {
        margin-top: 0.45rem;
        font-size: 0.7rem;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: var(--muted);
    }

    .cp-table {
        overflow: hidden;
        border-radius: 22px;
        border: 1px solid rgba(96, 165, 250, 0.16);
        background: rgba(8, 17, 31, 0.86);
        box-shadow: var(--shadow);
    }

    .cp-table table {
        width: 100%;
        border-collapse: collapse;
    }

    .cp-table thead th {
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--muted);
        background: rgba(12, 24, 44, 0.95);
        padding: 0.9rem 1rem;
        border-bottom: 1px solid rgba(96, 165, 250, 0.14);
    }

    .cp-table tbody tr {
        border-bottom: 1px solid rgba(96, 165, 250, 0.10);
    }

    .cp-table tbody tr:nth-child(even) {
        background: rgba(255,255,255,0.02);
    }

    .cp-table td {
        padding: 0.95rem 1rem;
        vertical-align: middle;
    }

    .cp-chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 92px;
        padding: 0.28rem 0.7rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .cp-score {
        padding: 1.15rem 1.15rem 1.05rem;
        background: rgba(8, 17, 31, 0.88);
        border: 1px solid rgba(96, 165, 250, 0.16);
        border-radius: 22px;
        box-shadow: var(--shadow-soft);
    }

    .cp-score-top {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.9rem;
    }

    .cp-score-value {
        font-size: 2rem;
        line-height: 1;
        font-weight: 800;
        letter-spacing: -0.06em;
    }

    .cp-score-meta {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        color: var(--muted);
        font-weight: 700;
    }

    .cp-bar {
        background: rgba(96, 165, 250, 0.10);
        border-radius: 999px;
        height: 6px;
        overflow: hidden;
    }

    .cp-bar > div {
        height: 100%;
        border-radius: 999px;
        box-shadow: 0 0 18px rgba(96, 165, 250, 0.45);
    }

    .cp-callout {
        padding: 1rem 1.05rem;
        background: rgba(59, 130, 246, 0.10);
        border: 1px solid rgba(96, 165, 250, 0.18);
        border-radius: 18px;
        color: var(--text);
        font-style: italic;
        box-shadow: var(--shadow-soft);
    }

    .cp-section-label {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        margin-bottom: 0.75rem;
        padding: 0.36rem 0.72rem;
        border-radius: 999px;
        background: rgba(59, 130, 246, 0.10);
        border: 1px solid rgba(96, 165, 250, 0.16);
        color: #bfdbfe;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.11em;
    }

    .cp-sidebar-title {
        margin: 0.7rem 0 0.35rem;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: var(--text);
    }

    .cp-sidebar-subtitle {
        margin: 0;
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .stMarkdown, .stCaption, .stInfo, .stSuccess, .stWarning, .stError {
        color: var(--text);
    }
    
</style>
""", unsafe_allow_html=True)

# ── Cache del grafo ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def cargar_grafo() -> GrafoCursos:
    return GrafoCursos()


# ── Helpers de renderizado ────────────────────────────────────────────────────

_NIVEL_BG = {"principiante": "#0b2a4a", "intermedio": "#0e355d", "avanzado": "#112d4e"}
_NIVEL_TEXT = {"principiante": "#93c5fd", "intermedio": "#60a5fa", "avanzado": "#bfdbfe"}
_NIVEL_LBL = {"principiante": "Principiante", "intermedio": "Intermedio", "avanzado": "Avanzado"}


def _chip(nivel: str) -> str:
    bg = _NIVEL_BG.get(nivel, "rgba(59, 130, 246, 0.12)")
    text = _NIVEL_TEXT.get(nivel, "#bfdbfe")
    lbl = _NIVEL_LBL.get(nivel, nivel)
    return (
        f'<span class="cp-chip" style="background:{bg}; color:{text};">{lbl}</span>'
    )


def render_trayectoria(trayectoria: list) -> None:
    filas = ""
    for i, curso in enumerate(trayectoria, 1):
        filas += f"""
        <tr class="cp-row">
            <td style="width:42px; color:#6F86A8; font-size:0.8rem; font-weight:700;">{i:02d}</td>
            <td style="color:#EAF2FF; font-size:0.94rem; font-weight:500;">{curso.nombre}</td>
            <td>{_chip(curso.nivel)}</td>
            <td style="color:#9AB0CF; font-size:0.84rem; text-align:right; white-space:nowrap;">{curso.duracion_semanas} sem</td>
        </tr>"""

    st.markdown(
        f"""
        <div class="cp-table">
            <table>
                <thead>
                    <tr>
                        <th style="width:42px;">#</th>
                        <th>Curso</th>
                        <th>Nivel</th>
                        <th style="text-align:right;">Duración</th>
                    </tr>
                </thead>
                <tbody>{filas}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metricas(num_cursos: int, semanas: int, nodos: int, tiempo: float) -> None:
    cols = st.columns(4)
    datos = [
        (str(num_cursos), "Cursos"),
        (str(semanas), "Semanas"),
        (f"{nodos:,}", "Nodos expandidos"),
        (f"{tiempo:.3f}s", "Tiempo"),
    ]
    for col, (valor, etiqueta) in zip(cols, datos):
        with col:
            st.markdown(
                f"""
                <div class="cp-metric">
                    <div class="cp-metric-value">{valor}</div>
                    <div class="cp-metric-label">{etiqueta}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_evaluacion(evaluacion: dict) -> None:
    puntuacion = evaluacion.get("puntuacion", 0)
    calidad = evaluacion.get("nivel_calidad", "")
    fortalezas = evaluacion.get("fortalezas", [])
    debilidades = evaluacion.get("debilidades", [])
    sugerencias = evaluacion.get("sugerencias", [])
    resumen = evaluacion.get("resumen", "")
    modo = evaluacion.get("modo", "llm_real")

    color_map = {
        "excelente": "#22D3EE",
        "bueno": "#60A5FA",
        "aceptable": "#38BDF8",
        "deficiente": "#FB7185",
    }
    color = color_map.get(calidad, "#9AB0CF")
    pct = max(0, min(100, int(puntuacion) * 10))
    sufijo = " · simulado" if modo == "simulado" else ""

    st.markdown(
        f"""
        <div class="cp-score">
            <div class="cp-score-top">
                <span class="cp-score-value" style="color:{color};">{puntuacion}/10</span>
                <span class="cp-score-meta">{calidad.capitalize()}{sufijo}</span>
            </div>
            <div class="cp-bar">
                <div style="background:{color}; width:{pct}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if resumen:
        st.markdown(
            f"""<div style="height:10px"></div><div class="cp-callout">“{resumen}”</div>""",
            unsafe_allow_html=True,
        )

    if fortalezas:
        with st.expander("Fortalezas", expanded=True):
            for f in fortalezas:
                st.markdown(
                    f"<p style='color:#B8C7E0; font-size:0.92rem; margin:0.45rem 0;'>✓ &nbsp;{f}</p>",
                    unsafe_allow_html=True,
                )

    if debilidades:
        with st.expander("Debilidades"):
            for d in debilidades:
                st.markdown(
                    f"<p style='color:#B8C7E0; font-size:0.92rem; margin:0.45rem 0;'>✕ &nbsp;{d}</p>",
                    unsafe_allow_html=True,
                )

    if sugerencias:
        with st.expander("Sugerencias"):
            for s in sugerencias:
                st.markdown(
                    f"<p style='color:#B8C7E0; font-size:0.92rem; margin:0.45rem 0;'>→ &nbsp;{s}</p>",
                    unsafe_allow_html=True,
                )



# ── Panel izquierdo ──────────────────────────────────────────────────────────

def render_left_panel(grafo: GrafoCursos) -> tuple[str, bool, str | None]:
    st.markdown(
        """
        <div class="cp-left-panel-shell">
            <div class="cp-panel cp-left-panel" style="padding:1rem 1rem 1.1rem;">
                <div class="cp-kicker">Configuración</div>
                <p class="cp-sidebar-title" style="margin-top:0.7rem;">Ajusta el modo de planificación</p>
                <p class="cp-sidebar-subtitle">Elige el algoritmo y decide si quieres interpretación con LLM.</p>
            </div>
        </div>
        <div style="height:0.9rem"></div>
        """,
        unsafe_allow_html=True,
    )

    alg_label = st.selectbox(
        "Algoritmo",
        options=["A* — Óptimo", "Greedy — Rápido"],
        index=0,
    )
    alg_key = "astar" if "A*" in alg_label else "greedy"

    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
    usar_llm = st.toggle("Usar LLM", value=True)

    perfil_id = None
    if not usar_llm:
        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
        opciones = {p.nombre: pid for pid, p in grafo.perfiles.items()}
        
        # En vez de un dropdown flotante, un radio button contenido
        st.markdown("<p style='color: var(--muted); font-weight: 600; font-size: 0.88rem; margin-bottom: 0.5rem;'>Perfil objetivo</p>", unsafe_allow_html=True)
        
        # Contenedor con altura fija y scroll para que no ocupe toda la pantalla
        with st.container(height=250, border=False):
            seleccion = st.radio(
                "Perfil objetivo", 
                options=list(opciones.keys()),
                label_visibility="collapsed"
            )
        perfil_id = opciones[seleccion]

    return alg_key, usar_llm, perfil_id


# ── App principal ─────────────────────────────────────────────────────────────

def main() -> None:
    grafo = cargar_grafo()

    panel_col, content_col = st.columns([1.08, 4.2], gap="large")

    with panel_col:
        alg_key, usar_llm, perfil_id = render_left_panel(grafo)

    with content_col:
        # Header
        st.markdown(
            """
            <div class="cp-hero">
                <div class="cp-kicker">Career Path Planner</div>
                <h1 class="cp-title">Planificación profesional con IA</h1>

            </div>
            """,
            unsafe_allow_html=True,
        )

        # Input + botón
        col_inp, col_btn = st.columns([6, 1], gap="small")

        with col_inp:
            if usar_llm:
                objetivo = st.text_input(
                    label="objetivo",
                    placeholder="Describe tu objetivo profesional...",
                    label_visibility="collapsed",
                )
            else:
                st.info("Escoge un Perfil Objetivo para continuar.")
                objetivo = ""

        with col_btn:
            ejecutar = st.button("Planificar", use_container_width=True)

        st.divider()

        # ── Estado vacío ──────────────────────────────────────────────────────────
        if not ejecutar:
            st.markdown(
                """
                <div class="cp-empty">
                    <div class="cp-kicker">Listo para empezar</div>
                    <h4>Escribe un objetivo y genera tu trayectoria</h4>
                    <p>Obtendrás una ruta profesional con un diseño más limpio y una lectura más clara.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        if usar_llm and not objetivo.strip():
            st.warning("Escribe un objetivo profesional antes de continuar.")
            return

        objetivo = objetivo.strip()

        # ── Modo con LLM ──────────────────────────────────────────────────────────
        if usar_llm:
            fn = astar if alg_key == "astar" else greedy

            with st.spinner("Generando trayectoria..."):
                resultado = pipeline_completo(
                    objetivo_texto=objetivo,
                    grafo=grafo,
                    algoritmo_fn=fn,
                    habilidades_iniciales=frozenset(),
                    instancia_id="app",
                )

            if not resultado["exito_total"]:
                st.error(
                    "No se pudo completar el pipeline. "
                    "Verifica que HF_API_KEY esté configurada en el archivo .env"
                )
                return

            busqueda = resultado["paso2_busqueda"]
            parseo = resultado["paso1_parseo"]

            perfil_nombre = grafo.perfiles[busqueda["perfil_objetivo"]].nombre
            confianza = parseo.get("confianza", "?")

            st.markdown(
                f"""
                <div style="margin-bottom:1rem;">
                    <div class="cp-kicker">Resultado con LLM</div>
                    <p style="margin:0.55rem 0 0; color:#9AB0CF; font-size:0.92rem;">
                        Perfil detectado:
                        <span style="color:#EAF2FF; font-weight:700;">{perfil_nombre}</span>
                        &nbsp;·&nbsp;confianza {confianza}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div class='cp-section-label'>Trayectoria</div>",
                unsafe_allow_html=True,
            )
            trayectoria = [grafo.cursos[cid] for cid in busqueda["trayectoria_ids"]]
            render_trayectoria(trayectoria)

        # ── Modo sin LLM ──────────────────────────────────────────────────────────
        else:
            if not perfil_id:
                st.warning("Selecciona un perfil en el panel izquierdo.")
                return

            fn = astar if alg_key == "astar" else greedy

            with st.spinner("Buscando trayectoria..."):
                if alg_key == "astar":
                    r = astar(grafo, frozenset(), perfil_id, "app_sin_llm", criterio="cursos")
                else:
                    r = greedy(grafo, frozenset(), perfil_id, "app_sin_llm")

            if not r.exito:
                st.error("No se encontró trayectoria válida para este perfil.")
                return

            ok, msg = validar_trayectoria(grafo, r.trayectoria, frozenset(), perfil_id)
            perfil_nombre = grafo.perfiles[perfil_id].nombre

            st.markdown(
                f"""
                <div style="margin-bottom:1rem;">
                    <div class="cp-kicker">Resultado sin LLM</div>
                    <p style="margin:0.55rem 0 0; color:#9AB0CF; font-size:0.92rem;">
                        Perfil:
                        <span style="color:#EAF2FF; font-weight:700;">{perfil_nombre}</span>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_tray, col_info = st.columns([3, 2], gap="large")

            with col_tray:
                st.markdown(
                    "<div class='cp-section-label'>Trayectoria</div>",
                    unsafe_allow_html=True,
                )
                render_trayectoria(r.trayectoria)

            with col_info:
                st.markdown(
                    "<div class='cp-section-label'>Estado</div>",
                    unsafe_allow_html=True,
                )
                if ok:
                    st.success("Trayectoria válida · prerrequisitos respetados")
                else:
                    st.error(msg)

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.info(
                    "Activa **Usar LLM** en el panel izquierdo para que el sistema "
                    "interprete tu objetivo en lenguaje natural y evalúe "
                    "semánticamente la trayectoria generada.",
                    icon="ℹ️",
                )


if __name__ == "__main__":
    main()