# Justificación de los parámetros seleccionados y de los resultados obtenidos

El diseño experimental se ha basado en el conjunto de hiperparámetros estándar introducido por DeepMind para el entrenamiento de agentes DQN en Atari 2600 (Mnih et al., 2015), ampliamente replicado en la literatura posterior y considerado el *baseline* de referencia para el entorno **Space Invaders**.

Este conjunto de parámetros define tanto el preprocesado de las observaciones como la arquitectura de la red y la dinámica básica del aprendizaje, proporcionando un punto de partida contrastado y reproducible.

---

## Learning rate y estabilidad del aprendizaje

El valor del *learning rate* se ha fijado inicialmente en **learning_rate = 1e-4**, inferior al valor clásico utilizado en el DQN original (2.5e-4). Esta elección responde a un compromiso explícito entre velocidad de aprendizaje y estabilidad.

Dado que el criterio de evaluación prioriza la **robustez del comportamiento** frente a la optimización de la recompensa media —concretamente, que el mínimo de las recompensas en los últimos 100 episodios supere un umbral—, se ha optado por un *learning rate* más conservador, que reduce la probabilidad de actualizaciones bruscas de los valores Q y minimiza la aparición de episodios catastróficos.

En la fase final del entrenamiento, el *learning rate* se reduce adicionalmente a **5e-5**, favoreciendo una convergencia más estable de la política aprendida.

---

## Actualización de la red objetivo (Polyak averaging)

A diferencia del DQN original, que emplea actualizaciones “duras” de la red objetivo cada cierto número de pasos, se ha utilizado una **actualización suave de la red objetivo** mediante *Polyak averaging*.

Este mecanismo actualiza la red objetivo de forma progresiva como una media móvil exponencial de la red principal, reduciendo la variación temporal del objetivo de entrenamiento y mejorando la estabilidad del aprendizaje. Esta elección resulta especialmente adecuada en un contexto donde se busca minimizar la probabilidad de episodios con bajo rendimiento, incluso a costa de una ligera reducción en la velocidad de aprendizaje inicial.

---

## Entrenamiento por etapas y control de la exploración

Con el objetivo de equilibrar exploración, aprendizaje y estabilidad, el entrenamiento se ha estructurado en **tres etapas**, redefiniendo el *schedule* de la política ε-greedy en cada una de ellas.

Este enfoque permite adaptar dinámicamente el nivel de exploración a la fase de aprendizaje del agente.

---

### Etapa 1: Aprendizaje inicial (hasta ~0.9M pasos)

```python
value_max = 1.0
value_min = 0.10
nb_steps  = 1_000_000
value_test = 0.0
```

En esta etapa, el agente explora de forma intensiva el espacio de estados y acciones, permitiendo la construcción de una representación inicial adecuada del entorno. El descenso progresivo de ε desde 1.0 hasta 0.10 asegura una transición gradual desde la exploración aleatoria hacia el aprovechamiento de la política aprendida.

---

### Etapa 2: Mejora de la consistencia (≈600k pasos)

```python
value_max = 0.10
value_min = 0.02
nb_steps  = 500_000
value_test = 0.0
```

Una vez adquirida una política razonable, se reduce el rango de exploración para reforzar comportamientos consistentes y mejorar la estabilidad del rendimiento. Esta etapa permite incrementar la frecuencia de episodios con recompensas superiores al umbral objetivo, aunque todavía pueden aparecer episodios con bajo rendimiento debido a la exploración residual.

---

### Etapa 3: Estabilización del rendimiento mínimo (≈300k pasos)

```python
value_max = 0.02
value_min = 0.01
nb_steps  = 300_000
value_test = 0.0
```

En la etapa final, la exploración se mantiene en niveles mínimos, limitándose a una pequeña probabilidad de acciones aleatorias con el fin de evitar sobreajuste extremo. El objetivo principal de esta fase es **reducir la variabilidad del rendimiento** y minimizar la aparición de episodios con baja recompensa, maximizando así el valor mínimo observado en la ventana de evaluación.

La reducción adicional del *learning rate* en esta etapa refuerza este objetivo, favoreciendo una convergencia suave y estable de la política final.

---

## Relación con los resultados obtenidos

La combinación de un *learning rate* conservador, la actualización suave de la red objetivo y un esquema de entrenamiento por etapas ha permitido alcanzar el criterio de evaluación planteado, logrando que el mínimo de las recompensas en los últimos 100 episodios supere el umbral establecido, al tiempo que se mantiene una recompensa media elevada.

Este resultado confirma la adecuación de las decisiones de diseño adoptadas, especialmente en un contexto donde la robustez del comportamiento es prioritaria frente a la optimización exclusiva de la recompensa media.

---

# Justificación del uso de wrappers del entorno Atari

Con el fin de reproducir el entorno experimental empleado en los trabajos originales de DeepMind y sus posteriores replicaciones, se han aplicado una serie de wrappers estándar sobre el entorno `SpaceInvaders-v0`. Estos wrappers permiten controlar el preprocesado de observaciones, la dinámica temporal y la definición de episodios, mejorando la estabilidad y eficiencia del aprendizaje sin modificar la lógica del juego subyacente.

## NoopResetEnv

El wrapper `NoopResetEnv` introduce una secuencia aleatoria de acciones nulas (`NOOP`) al inicio de cada episodio, con un número máximo predefinido.

Este procedimiento, introducido por DeepMind, tiene como objetivo:

* aumentar la diversidad de estados iniciales,
* evitar que el agente aprenda políticas excesivamente dependientes de un estado inicial fijo,
* y mejorar la generalización y robustez del comportamiento aprendido.

El uso de `NoopResetEnv` permite que el agente se enfrente a diferentes configuraciones iniciales del entorno, simulando variaciones naturales en el inicio de la partida.

---

## FireResetEnv

En Space Invaders, el juego no comienza hasta que se ejecuta la acción `FIRE`. El wrapper `FireResetEnv` asegura que, tras cada `reset`, se ejecute automáticamente la acción necesaria para iniciar la partida.

Este wrapper:

* garantiza la correcta inicialización del entorno,
* evita episodios triviales en los que el agente no recibe observaciones significativas,
* y permite que el agente aprenda directamente sobre estados relevantes del juego.

---

## MaxAndSkipEnv

El wrapper `MaxAndSkipEnv` aplica dos transformaciones clave:

1. **Frame skipping**, ejecutando la misma acción durante varios pasos consecutivos.
2. **Max-pooling** sobre los dos últimos frames observados.

Este mecanismo:

* reduce la frecuencia efectiva de decisión del agente,
* disminuye el coste computacional del entrenamiento,
* y atenúa el efecto del parpadeo (flickering) característico de los juegos Atari.

El uso de `skip=4` sigue exactamente la configuración empleada por DeepMind y es considerado un estándar en aprendizaje profundo sobre Atari.

---

## EpisodicLifeEnv (solo durante entrenamiento)

El wrapper `EpisodicLifeEnv` redefine el final de un episodio cada vez que el agente pierde una vida, sin reiniciar completamente el entorno.

Este enfoque:

* proporciona una señal de aprendizaje más densa,
* acelera el proceso de aprendizaje,
* y facilita la asociación entre errores locales y sus consecuencias inmediatas.

Durante la fase de entrenamiento, esta redefinición de episodios contribuye a una convergencia más rápida y estable. Sin embargo, para garantizar una evaluación fiel al objetivo del problema, este wrapper se desactiva durante la fase de test, evaluando al agente sobre partidas completas.

---

## Relación con el objetivo de robustez

La combinación de estos wrappers permite:

* entrenar al agente bajo condiciones variadas y realistas,
* reducir la varianza del entrenamiento,
* y mejorar la estabilidad de la política aprendida.

En conjunto, estas decisiones metodológicas contribuyen directamente a alcanzar el criterio de evaluación planteado, basado en la robustez del comportamiento (`min(last100) > 20`), más allá de la mera optimización de la recompensa media.

Perfecto, ese es **el último bloque clave** y además encaja muy bien con tu objetivo de **robustez**.
Te dejo una **redacción académica**, alineada con la literatura y conectada explícitamente con `min(last100) > 20`.

---

# Uso de Double DQN y arquitectura Dueling

Además del DQN estándar, se han incorporado dos mejoras ampliamente aceptadas en la literatura: **Double DQN** y la **arquitectura Dueling**. Ambas extensiones están orientadas a mejorar la estabilidad del aprendizaje y la calidad de la estimación de los valores Q, sin modificar la estructura general del algoritmo.

---

## Double DQN

El algoritmo DQN original tiende a **sobreestimar los valores Q**, debido a que la selección y la evaluación de la acción se realizan utilizando la misma red neuronal. Esta sobreestimación puede derivar en políticas inestables y comportamientos subóptimos, especialmente en entornos estocásticos como los juegos Atari.

Double DQN mitiga este problema desacoplando ambos procesos:

* la **red principal** se utiliza para seleccionar la acción,
* mientras que la **red objetivo** se emplea para evaluar su valor.

Este mecanismo reduce el sesgo optimista en la estimación de los valores Q, produciendo estimaciones más realistas y estables. En el contexto del presente trabajo, la reducción de sobreestimación contribuye a evitar decisiones excesivamente arriesgadas que pueden desembocar en episodios con bajo rendimiento, favoreciendo así un comportamiento más robusto.

---

## Arquitectura Dueling

La arquitectura Dueling separa explícitamente la estimación del **valor del estado** (V(s)) y de la **ventaja de cada acción** (A(s,a)), combinándolos posteriormente para obtener el valor Q:

[
Q(s,a) = V(s) + \left(A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')\right)
]

Esta descomposición resulta especialmente beneficiosa en entornos donde, para muchos estados, la elección concreta de la acción tiene un impacto limitado sobre la recompensa inmediata.

En Space Invaders, numerosos estados comparten un valor intrínseco similar independientemente de la acción ejecutada (por ejemplo, desplazamientos laterales sin disparo). La arquitectura Dueling permite aprender de forma más eficiente el valor de estos estados, mejorando la generalización y la estabilidad del comportamiento aprendido.

---

## Contribución conjunta a la robustez

La combinación de Double DQN y arquitectura Dueling permite:

* reducir el sesgo en la estimación de los valores Q,
* mejorar la eficiencia del aprendizaje de estados relevantes,
* y disminuir la variabilidad del rendimiento entre episodios.

Estas propiedades son especialmente relevantes cuando el criterio de evaluación se basa en el **rendimiento mínimo** observado en una ventana de episodios, ya que ayudan a reducir la probabilidad de decisiones erráticas o excesivamente optimistas que puedan dar lugar a episodios con baja recompensa.

---

## Justificación de la elección metodológica

Tanto Double DQN como la arquitectura Dueling forman parte del conjunto de extensiones estándar del DQN moderno y se han utilizado con éxito en múltiples trabajos posteriores al DQN original, incluyendo variantes como Rainbow DQN.

Su incorporación en este trabajo responde a la necesidad de mejorar la estabilidad y robustez del aprendizaje, manteniendo al mismo tiempo la compatibilidad con el entorno experimental y las restricciones impuestas (keras-rl2, gym clásico y SpaceInvaders-v0).

---

## Relación con los resultados obtenidos

La adopción de estas técnicas, junto con el control progresivo de la exploración y el uso de Polyak averaging, ha contribuido de forma significativa a alcanzar el criterio de evaluación establecido, logrando un comportamiento consistente que evita episodios de bajo rendimiento en la fase de evaluación.

---

