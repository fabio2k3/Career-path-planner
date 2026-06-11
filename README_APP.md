# Career Path Planner — Interfaz Visual

Interfaz web del sistema construida con Streamlit. Permite planificar trayectorias profesionales de forma interactiva, con o sin LLM, directamente desde el navegador.

---

## Instalación adicional

Además de las dependencias base del proyecto, instala Streamlit:

```bash
pip install streamlit
```

---

## Ejecutar la app

Desde la raíz del proyecto:

```bash
streamlit run app.py
```

Se abrirá automáticamente en `http://localhost:8501`.

> Asegúrate de tener el archivo `.env` con tu `HF_API_KEY` configurada antes de lanzar la app. Consulta el `README.md` principal para obtener la clave.

---

## Cómo usar la interfaz

La pantalla se divide en dos zonas: el **panel izquierdo** de configuración y el **área principal** de resultados.

### Panel izquierdo

**Algoritmo** — elige entre dos modos de búsqueda:
- `A* — Óptimo`: encuentra la trayectoria con el mínimo número de cursos garantizado.
- `Greedy — Rápido`: más veloz, sin garantía de optimalidad.

**Usar LLM** — toggle que activa o desactiva el pipeline completo:
- **Activado**: escribes tu objetivo en lenguaje natural y el LLM lo interpreta, detecta el perfil más cercano y evalúa semánticamente la trayectoria resultante.
- **Desactivado**: seleccionas el perfil objetivo directamente de la lista y el algoritmo calcula la trayectoria sin intervención del LLM.

### Área principal

1. Escribe tu objetivo profesional en el campo de texto, por ejemplo:
   - `Quiero trabajar como Data Scientist`
   - `Me interesa el desarrollo backend con Python y microservicios`
   - `Quiero llevar modelos de ML a producción`

2. Pulsa **Planificar**.

3. El sistema muestra:
   - El perfil detectado y el nivel de confianza del LLM (modo con LLM).
   - La trayectoria ordenada con nombre, nivel y duración de cada curso.
   - Las métricas de búsqueda: cursos totales, semanas, nodos expandidos y tiempo.
   - La evaluación LLM con puntuación, fortalezas, debilidades y sugerencias (modo con LLM).

---

## Modos de uso

| Modo | Toggle LLM | Entrada | Qué muestra |
|---|---|---|---|
| Completo con LLM | ✅ Activado | Texto libre | Perfil detectado + trayectoria + evaluación semántica |
| Sin LLM | ❌ Desactivado | Selección de perfil | Trayectoria + validación de prerrequisitos |

---
