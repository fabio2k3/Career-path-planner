# Informe ejecutivo — Career Path Planner

_Generado automáticamente el 2026-06-10 10:00:33_

## Resumen comparativo A* vs Greedy

| Algoritmo | Instancias | Éxito | Cursos prom. | Semanas prom. | Nodos prom. | Tiempo prom. (s) | Tray. válidas |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A* | 8 | 100.0% | 9.12 | 40.00 | 21717.88 | 1.2335 | 100.0% |
| Greedy | 8 | 100.0% | 9.25 | 39.50 | 270.00 | 0.0087 | 100.0% |

## Análisis de la evaluación LLM

La puntuación media del LLM fue **8.25/10** (mínimo 7.00, máximo 9.00).

Las trayectorias **largas** (1 casos: >12 cursos o >50 semanas) obtuvieron una media de 7.00/10, frente a 8.43/10 en las **concisas**. La diferencia de 1.43 puntos evidencia la penalización por extensión.

Este comportamiento es coherente con el criterio de evaluación: el LLM prioriza trayectorias eficientes y reduce la nota cuando la solución se vuelve extensa en cursos o semanas.

### Resumen cuantitativo

| Métrica | Valor |
| --- | --- |
| Puntuación media | 8.25/10 |
| Puntuación mínima | 7.00/10 |
| Puntuación máxima | 9.00/10 |
| Trayectorias largas | 1 |
| Trayectorias concisas | 7 |

## Conclusión

El sistema demuestra una separación clara entre la calidad estructural de la trayectoria y su coste temporal. A* produce soluciones óptimas respecto al criterio elegido (número de cursos o semanas), mientras que el componente LLM añade una evaluación semántica que penaliza rutas excesivamente largas y recompensa las trayectorias eficientes y bien orientadas al objetivo.
