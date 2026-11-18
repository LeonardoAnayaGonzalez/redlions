# NFL Score Analysis - Redlions
## Autores: - Leonardo Ivan Anaya Gonzalez, - Camila Torres Rojas

NFL Strategy Lab es una aplicación web interactiva que busca analizar jugadas de fútbol americano utilizando datos de la NFL.

La idea principal es que el usuario no solamente pueda consultar estadísticas, sino también visualizar cómo se desarrolla una jugada, observar el movimiento de los jugadores y explorar qué otras decisiones se pudieron haber tomado durante el juego.

El proyecto combina análisis de datos, visualizaciones y simulaciones para entender mejor las estrategias ofensivas y defensivas.

### Objetivo

Desarrollar una herramienta que permita analizar jugadas de la NFL de manera visual e interactiva, utilizando datos reales para identificar patrones de movimiento, comparar opciones de pase y explorar posibles estrategias.

Pregunta principal: ¿Podemos utilizar los datos de posición y movimiento de los jugadores para analizar qué decisiones podrían ser más convenientes durante una jugada?

Nuestra idea es que, al considerar la posición de los jugadores, sus movimientos y la ubicación de los defensores, podemos obtener más información sobre las opciones disponibles que si solamente observamos el resultado final de la jugada.

### ¿Cómo funciona?

La aplicación está pensada para incluir las siguientes funciones:

1. Visualización de jugadas

Reconstruir jugadas de la NFL utilizando los datos de posición de los jugadores.

El usuario podrá observar sus movimientos sobre el campo, reproducir una jugada, pausarla y analizar lo que sucede en diferentes momentos.

2. Análisis de jugadores

Permitir seleccionar jugadores para consultar información sobre su posición, movimiento y distancia respecto a otros jugadores.

Esto nos ayudará a identificar posibles espacios libres, defensores cercanos y situaciones que podrían influir en una decisión ofensiva.

3. Simulación de estrategias

Una de las funciones principales que queremos desarrollar es la posibilidad de detener una jugada y explorar diferentes decisiones.

Por ejemplo, comparar distintas opciones de pase o analizar qué podría ocurrir si un jugador se desplazara hacia otra zona del campo.

La intención es comparar alternativas, no asumir que existe una única decisión correcta.

4. Estadísticas y predicciones

Incluir un dashboard con gráficas y estadísticas relacionadas con las jugadas y los jugadores.

También buscamos incorporar estimaciones que ayuden a comparar distintas situaciones, siempre tomando en cuenta la información disponible en los datos.

5. Modos de visualización

La aplicación contará con dos modos:

Modo principiante: explicaciones sencillas para usuarios que no conocen demasiado sobre fútbol americano.

Modo coach: información más detallada sobre movimientos, posiciones y posibles decisiones tácticas.

### Datos y metodología

El proyecto utiliza datos de la NFL, principalmente información de seguimiento de jugadores (tracking data), información de los jugadores y de las jugadas.

Primero se revisan las variables del dataset para identificar qué análisis podemos realizar.

Después, a partir de las posiciones y movimientos registrados, buscamos reconstruir las jugadas, calcular distancias entre jugadores, identificar espacios disponibles y comparar diferentes opciones.

Las simulaciones y predicciones dependerán de las variables que realmente contengan los datos. No queremos presentar resultados inventados ni confundir una estimación con lo que realmente ocurrió durante un partido.

### Herramientas

Para desarrollar el proyecto estamos utilizando:

Kiro: asistencia con inteligencia artificial para desarrollar la aplicación.

Python y FastAPI: procesamiento de datos y backend.

React y TypeScript: desarrollo de la interfaz.

Tailwind CSS: diseño de la aplicación.

Recharts: visualización de estadísticas y gráficas.

GitHub: organización del código y trabajo colaborativo.

### Limitaciones

Los resultados dependen de la información disponible en el dataset.

No todas las decisiones de un partido pueden explicarse únicamente con la posición de los jugadores, ya que también intervienen otros factores que pueden no estar registrados. Por eso, las simulaciones deben entenderse como herramientas para explorar escenarios posibles y no como predicciones exactas.

### Resultado esperado

Esperamos desarrollar una aplicación que convierta los datos de la NFL en información más sencilla de interpretar y que permita al usuario interactuar con las jugadas en lugar de solamente observar estadísticas.

Nuestro objetivo es que NFL Strategy Lab sea útil tanto para personas que están aprendiendo sobre fútbol americano como para usuarios interesados en analizar estrategias con mayor detalle.
