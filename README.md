# Análisis de resultados de la NFL - Redlions

**Autores:**
- Leonardo Iván Anaya González
- Camila Torres Rojas

## Descripción del proyecto

NFL Strategy Lab es una aplicación web interactiva que busca analizar jugadas de fútbol americano utilizando datos reales de seguimiento de jugadores de la NFL.

La idea principal es que el usuario no solamente pueda consultar estadísticas, sino también visualizar cómo se desarrolla una jugada, observar el movimiento de los jugadores y explorar qué otras decisiones se podrían haber tomado durante el juego.

Además, buscamos que la aplicación pueda servir como una herramienta de apoyo para entrenadores, permitiendo comparar rutas, encontrar posibles espacios libres y proponer nuevas jugadas ofensivas.

El proyecto combina análisis de datos, visualizaciones, predicciones y simulaciones para entender mejor las estrategias ofensivas y defensivas.

## Objetivo

Desarrollar una herramienta que permita analizar jugadas de la NFL de manera visual e interactiva, utilizando datos reales para identificar patrones de movimiento, comparar opciones de pase y explorar posibles estrategias.

Nuestra pregunta principal es: **¿Podemos utilizar los datos de posición y movimiento de los jugadores para identificar rutas que generen mayor separación respecto a los defensores y proponer nuevas jugadas ofensivas?**

Nuestra idea es que, al considerar la posición, velocidad y dirección de los jugadores, así como la ubicación de los defensores, podemos obtener más información sobre las opciones disponibles que si solamente observamos el resultado final de una jugada.

También queremos investigar si es posible utilizar esta información para generar recomendaciones que puedan ser útiles durante los entrenamientos.

## ¿Cómo funciona?

La aplicación está organizada en diferentes módulos que permiten analizar los datos desde distintas perspectivas. Algunos ya forman parte de la aplicación y otros se encuentran en desarrollo.

### 1. Visualización de jugadas

Buscamos reconstruir jugadas reales de la NFL utilizando las posiciones registradas de los jugadores.

El usuario puede explorar los movimientos sobre una representación del campo de fútbol americano, reproducir una jugada, pausarla y analizar lo que sucede en distintos momentos.

Esto permite observar cómo se distribuyen los jugadores, cómo se mueven los defensores y qué espacios pueden aparecer durante una jugada.

### 2. Análisis de jugadores

La sección Player Intelligence permite consultar perfiles de jugadores y comparar estadísticas relacionadas con velocidad, aceleración, distancia recorrida y separación.

También incluye tarjetas individuales, gráficas y rankings.

Las puntuaciones deben calcularse utilizando distribuciones de jugadores comparables para evitar resultados engañosos. Estas métricas son indicadores descriptivos y no calificaciones definitivas del rendimiento de un jugador.

### 3. Predicción de movimientos

En Predictive Lab buscamos estimar hacia dónde podría desplazarse un jugador durante los siguientes instantes a partir de su posición y movimiento actual.

Estas predicciones pueden compararse con los movimientos reales registrados para calcular el error del modelo y conocer sus limitaciones.

La intención es estudiar qué tan bien podemos anticipar los movimientos utilizando los datos disponibles.

### 4. Optimización de rutas

Una de las funciones principales que estamos desarrollando es Route Optimizer.

La idea es que un entrenador pueda seleccionar un receptor durante una jugada y analizar qué otras rutas podría seguir para conseguir una mayor separación respecto a los defensores.

Para esto, el programa debe generar distintas trayectorias posibles y compararlas considerando las posiciones y movimientos de los jugadores, el tiempo disponible y las limitaciones del campo.

El objetivo no es simplemente encontrar la ruta que aleje más al receptor, sino identificar una separación que pueda ser útil para recibir un pase.

Las rutas recomendadas se compararán con la trayectoria real del jugador. Esto permitirá explorar posibles alternativas, aunque no significa que una ruta distinta necesariamente hubiera producido un mejor resultado durante el partido.

### 5. Creación de nuevas jugadas

También proponemos desarrollar Playbook Creator, una herramienta para construir y guardar jugadas ofensivas nuevas.

Buscamos que el usuario pueda seleccionar una formación, modificar las posiciones de los jugadores y diseñar diferentes rutas para los receptores.

Además, queremos que el programa genere automáticamente distintas combinaciones de rutas y las evalúe utilizando los modelos de movimiento y separación.

De esta forma, el usuario podría comparar varias propuestas, seleccionar las más interesantes y guardarlas en un catálogo personalizado para consultarlas posteriormente.

La intención es que este catálogo pueda funcionar como una herramienta de apoyo para entrenadores interesados en explorar estrategias durante sus prácticas.

### 6. Estadísticas y evaluación de modelos

La aplicación incluye secciones para consultar estadísticas de los jugadores y analizar los resultados de los modelos utilizados.

Buscamos mostrar gráficas que comparen los movimientos estimados con los reales, calcular errores de predicción y explicar qué variables influyen en los resultados.

También queremos evitar que una simulación se interprete como un hecho. Por eso diferenciamos los datos observados de las predicciones y escenarios hipotéticos.

### 7. Modos de visualización

La aplicación está diseñada para incluir dos modos:

**Modo principiante:** explicaciones sencillas para usuarios que no conocen demasiado sobre fútbol americano, incluyendo conceptos básicos y una interpretación fácil de las estadísticas.

**Modo analista:** información más detallada sobre movimientos, posiciones, separaciones, percentiles, modelos y posibles decisiones tácticas.

Ambos modos utilizan la misma información, pero la presentan con diferentes niveles de detalle.

## Datos y metodología

El proyecto utiliza datos de seguimiento de jugadores de la NFL (tracking data), junto con información de jugadores y jugadas.

Estos datos permiten conocer las posiciones de los jugadores en diferentes momentos y, dependiendo del archivo, variables como velocidad, aceleración y dirección.

Primero revisamos las variables de los conjuntos de datos para identificar qué análisis podemos realizar.

Después utilizamos las posiciones y movimientos registrados para reconstruir jugadas, calcular distancias entre jugadores, identificar espacios disponibles y comparar diferentes situaciones.

Para las predicciones utilizamos modelos de movimiento que estiman las posiciones futuras de los jugadores. La precisión de estas estimaciones debe evaluarse comparándolas con datos reales que no hayan sido utilizados para ajustar el modelo.

En la optimización de rutas buscamos comparar distintas trayectorias físicamente posibles y seleccionar aquellas que proporcionen mayor separación útil bajo las condiciones de la simulación.

Para la generación de jugadas nuevas queremos combinar varias rutas ofensivas y evaluar su comportamiento frente a diferentes escenarios defensivos.

Las simulaciones dependen de los supuestos del modelo y de las variables disponibles. No queremos presentar resultados inventados ni confundir una estimación con lo que realmente ocurrió durante un partido.

## Herramientas

Para desarrollar el proyecto estamos utilizando:

- **Kiro:** asistencia con inteligencia artificial para desarrollar y modificar la aplicación.
- **Python, pandas y NumPy:** procesamiento de datos y cálculos estadísticos.
- **FastAPI:** comunicación entre el backend y la interfaz.
- **React y TypeScript:** desarrollo de la aplicación web.
- **Tailwind CSS:** diseño visual.
- **Recharts:** visualización de estadísticas y gráficas.
- **GitHub:** organización del código y trabajo colaborativo.

Utilizamos las versiones establecidas por los organizadores del hackatón.


## Limitaciones

Los resultados dependen de la información disponible en los conjuntos de datos y de la cantidad de jugadas analizadas.

No todas las decisiones de un partido pueden explicarse únicamente con la posición de los jugadores, ya que también intervienen otros factores que pueden no estar registrados.

Una mayor separación entre un receptor y un defensor no garantiza que un pase sea completado. También influyen la situación de juego, la trayectoria del balón y las decisiones del quarterback.

Además, cuando proponemos una ruta nueva no podemos asumir que los defensores reaccionarían exactamente como lo hicieron en la jugada original.

Por eso, las simulaciones y las jugadas generadas deben entenderse como herramientas para explorar escenarios posibles y no como predicciones exactas de lo que ocurriría durante un partido.

## Resultado esperado

Esperamos desarrollar una aplicación que convierta los datos de tracking de la NFL en información más sencilla de interpretar y que permita al usuario interactuar con las jugadas en lugar de solamente observar estadísticas.

Nuestra intención es que NFL Strategy Lab no se limite a analizar lo que ocurrió, sino que también pueda ayudar a explorar qué otras rutas podrían ser útiles y cómo se podrían combinar para construir nuevas estrategias ofensivas.

El objetivo final es que sea útil tanto para personas que están aprendiendo sobre fútbol americano como para entrenadores y usuarios interesados en analizar, comparar y diseñar jugadas con mayor detalle.
