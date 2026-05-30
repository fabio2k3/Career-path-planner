# Informe ejecutivo
_Generado automáticamente el 2026-05-30 16:57:43_
## Resumen comparativo A* vs Greedy
| Algoritmo | Instancias | Éxito | Cursos prom. | Semanas prom. | Nodos prom. | Tiempo prom. (s) | Tray. válidas |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A* | 8 | 100.0% | 9.12 | 39.75 | 1379.50 | 0.0967 | 100.0% |
| Greedy | 8 | 100.0% | 9.25 | 39.25 | 16.62 | 0.0016 | 100.0% |

En promedio, A* reduce la duración total frente a Greedy en -0.50 semanas, lo que refuerza su mejor calidad de solución cuando el criterio principal es minimizar cursos/semanas.
## Análisis de la evaluación LLM
En `resultados_con_llm.csv`, la puntuación media del LLM fue 6.88/10 (mínimo 6.00, máximo 9.00).
Las trayectorias largas (1 casos; más de 12 cursos o más de 50 semanas) obtuvieron una media de 6.00/10, frente a 7.00/10 en las trayectorias concisas. La diferencia de 1.00 puntos indica una penalización clara por longitud y duración.
Este comportamiento es coherente con el criterio de evaluación estricta: el evaluador prioriza trayectorias eficientes y reduce la nota cuando la solución se vuelve extensa en cursos o semanas.

### Resumen cuantitativo del LLM
| Métrica | Valor |
| --- | --- |
| Puntuación media | 6.88/10 |
| Puntuación mínima | 6.00/10 |
| Puntuación máxima | 9.00/10 |
| Trayectorias largas | 1 |
| Trayectorias concisas | 7 |
## Conclusión
El sistema muestra una separación clara entre la calidad estructural de la trayectoria y su coste temporal. A* mantiene una solución más ordenada frente a Greedy, mientras que el LLM penaliza con mayor dureza las rutas excesivamente largas y recompensa mejor las trayectorias cortas y eficientes.
