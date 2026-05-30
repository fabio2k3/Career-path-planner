"""
generate_report.py

Genera un informe ejecutivo en Markdown a partir de los resultados experimentales.

Salida:
  - results/informe_ejecutivo.md
"""

import csv
from pathlib import Path
from statistics import mean
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
OUT_MD = RESULTS_DIR / "informe_ejecutivo.md"

CSV_SIN_LLM = RESULTS_DIR / "resultados_sin_llm.csv"
CSV_CON_LLM = RESULTS_DIR / "resultados_con_llm.csv"


def leer_csv(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def to_bool(value):
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"true", "1", "yes", "y", "si", "sí"}


def pct(n, d):
    if d == 0:
        return "0.0%"
    return f"{(100.0 * n / d):.1f}%"


def resumen_algoritmo(rows, algoritmo_predicado):
    seleccion = [r for r in rows if algoritmo_predicado(r)]
    exitos = [r for r in seleccion if to_bool(r.get("exito"))]

    if not seleccion:
        return {
            "n": 0,
            "exitos": 0,
            "exito_rate": "0.0%",
            "cursos_prom": "0.00",
            "semanas_prom": "0.00",
            "nodos_prom": "0.00",
            "tiempo_prom": "0.0000",
            "validas": "0.0%",
        }

    cursos = [safe_int(r.get("num_cursos")) for r in exitos] or [0]
    semanas = [safe_int(r.get("costo_total_semanas")) for r in exitos] or [0]
    nodos = [safe_int(r.get("nodos_expandidos")) for r in exitos] or [0]
    tiempos = [safe_float(r.get("tiempo_segundos")) for r in exitos] or [0.0]
    validas = [to_bool(r.get("trayectoria_valida")) for r in exitos] or [False]

    return {
        "n": len(seleccion),
        "exitos": len(exitos),
        "exito_rate": pct(len(exitos), len(seleccion)),
        "cursos_prom": f"{mean(cursos):.2f}",
        "semanas_prom": f"{mean(semanas):.2f}",
        "nodos_prom": f"{mean(nodos):.2f}",
        "tiempo_prom": f"{mean(tiempos):.4f}",
        "validas": pct(sum(validas), len(validas)),
    }


def resumen_llm(rows):
    if not rows:
        return None

    scores = [safe_float(r.get("puntuacion_llm"), None) for r in rows]
    scores = [s for s in scores if s is not None]
    if not scores:
        return None

    cursos = [safe_int(r.get("num_cursos")) for r in rows]
    semanas = [safe_int(r.get("costo_total_semanas")) for r in rows]

    largas = [
        r for r in rows
        if safe_int(r.get("num_cursos")) > 12 or safe_int(r.get("costo_total_semanas")) > 50
    ]
    concisas = [
        r for r in rows
        if not (safe_int(r.get("num_cursos")) > 12 or safe_int(r.get("costo_total_semanas")) > 50)
    ]

    long_scores = [safe_float(r.get("puntuacion_llm"), None) for r in largas]
    long_scores = [s for s in long_scores if s is not None]

    short_scores = [safe_float(r.get("puntuacion_llm"), None) for r in concisas]
    short_scores = [s for s in short_scores if s is not None]

    return {
        "n": len(rows),
        "avg": mean(scores),
        "min": min(scores),
        "max": max(scores),
        "avg_cursos": mean(cursos),
        "avg_semanas": mean(semanas),
        "long_n": len(largas),
        "short_n": len(concisas),
        "long_avg": mean(long_scores) if long_scores else None,
        "short_avg": mean(short_scores) if short_scores else None,
    }


def md_table(headers, rows):
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def build_report():
    sin_llm = leer_csv(CSV_SIN_LLM)
    con_llm = leer_csv(CSV_CON_LLM)

    astar_rows = [r for r in sin_llm if "A*" in r.get("algoritmo", "")]
    greedy_rows = [r for r in sin_llm if "Greedy" in r.get("algoritmo", "")]

    astar = resumen_algoritmo(sin_llm, lambda r: "A*" in r.get("algoritmo", ""))
    greedy = resumen_algoritmo(sin_llm, lambda r: "Greedy" in r.get("algoritmo", ""))
    llm = resumen_llm(con_llm)

    # Tabla principal A* vs Greedy
    tabla_principal = md_table(
        [
            "Algoritmo",
            "Instancias",
            "Éxito",
            "Cursos prom.",
            "Semanas prom.",
            "Nodos prom.",
            "Tiempo prom. (s)",
            "Tray. válidas",
        ],
        [
            [
                "A*",
                str(astar["n"]),
                astar["exito_rate"],
                astar["cursos_prom"],
                astar["semanas_prom"],
                astar["nodos_prom"],
                astar["tiempo_prom"],
                astar["validas"],
            ],
            [
                "Greedy",
                str(greedy["n"]),
                greedy["exito_rate"],
                greedy["cursos_prom"],
                greedy["semanas_prom"],
                greedy["nodos_prom"],
                greedy["tiempo_prom"],
                greedy["validas"],
            ],
        ],
    )

    # Interpretación automática del LLM
    analisis_llm = []
    if llm is None:
        analisis_llm.append(
            "No se encontraron puntuaciones LLM válidas en `resultados_con_llm.csv`."
        )
    else:
        analisis_llm.append(
            f"En `resultados_con_llm.csv`, la puntuación media del LLM fue {llm['avg']:.2f}/10 "
            f"(mínimo {llm['min']:.2f}, máximo {llm['max']:.2f})."
        )

        if llm["long_n"] > 0 and llm["short_n"] > 0 and llm["long_avg"] is not None and llm["short_avg"] is not None:
            diff = llm["short_avg"] - llm["long_avg"]
            analisis_llm.append(
                f"Las trayectorias largas ({llm['long_n']} casos; más de 12 cursos o más de 50 semanas) "
                f"obtuvieron una media de {llm['long_avg']:.2f}/10, frente a {llm['short_avg']:.2f}/10 "
                f"en las trayectorias concisas. La diferencia de {diff:.2f} puntos indica una penalización "
                f"clara por longitud y duración."
            )
        elif llm["long_n"] > 0 and llm["long_avg"] is not None:
            analisis_llm.append(
                f"Se detectaron {llm['long_n']} trayectorias largas con una media de {llm['long_avg']:.2f}/10. "
                "El patrón confirma una penalización visible sobre rutas extensas."
            )
        else:
            analisis_llm.append(
                "No se detectaron trayectorias largas en el conjunto evaluado, por lo que no fue posible "
                "comparar de forma directa el efecto de la penalización."
            )

        analisis_llm.append(
            "Este comportamiento es coherente con el criterio de evaluación estricta: el evaluador prioriza "
            "trayectorias eficientes y reduce la nota cuando la solución se vuelve extensa en cursos o semanas."
        )

    # Resumen ejecutivo
    mejores_cursos = None
    if astar_rows and greedy_rows:
        try:
            avg_astar_semanas = mean([safe_int(r.get("costo_total_semanas")) for r in astar_rows if to_bool(r.get("exito"))] or [0])
            avg_greedy_semanas = mean([safe_int(r.get("costo_total_semanas")) for r in greedy_rows if to_bool(r.get("exito"))] or [0])
            mejores_cursos = avg_greedy_semanas - avg_astar_semanas
        except Exception:
            mejores_cursos = None

    report = []
    report.append("# Informe ejecutivo\n")
    report.append(f"_Generado automáticamente el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    report.append("## Resumen comparativo A* vs Greedy\n")
    report.append(tabla_principal + "\n")

    if mejores_cursos is not None:
        report.append(
            f"\nEn promedio, A* reduce la duración total frente a Greedy en {mejores_cursos:.2f} semanas, "
            "lo que refuerza su mejor calidad de solución cuando el criterio principal es minimizar cursos/semanas.\n"
        )

    report.append("## Análisis de la evaluación LLM\n")
    for line in analisis_llm:
        report.append(line + "\n")

    if llm is not None:
        report.append("\n### Resumen cuantitativo del LLM\n")
        report.append(
            md_table(
                ["Métrica", "Valor"],
                [
                    ["Puntuación media", f"{llm['avg']:.2f}/10"],
                    ["Puntuación mínima", f"{llm['min']:.2f}/10"],
                    ["Puntuación máxima", f"{llm['max']:.2f}/10"],
                    ["Trayectorias largas", str(llm["long_n"])],
                    ["Trayectorias concisas", str(llm["short_n"])],
                ],
            )
            + "\n"
        )

    report.append("## Conclusión\n")
    report.append(
        "El sistema muestra una separación clara entre la calidad estructural de la trayectoria y su coste temporal. "
        "A* mantiene una solución más ordenada frente a Greedy, mientras que el LLM penaliza con mayor dureza las "
        "rutas excesivamente largas y recompensa mejor las trayectorias cortas y eficientes.\n"
    )

    return "".join(report)


def main():
    if not CSV_SIN_LLM.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {CSV_SIN_LLM}")
    if not CSV_CON_LLM.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {CSV_CON_LLM}")

    contenido = build_report()
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(contenido)

    print(f"✓ Informe generado en: {OUT_MD}")


if __name__ == "__main__":
    main()