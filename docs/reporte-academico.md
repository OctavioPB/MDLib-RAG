# Biblio-RAG: diseño, implementación y evaluación de un sistema de generación aumentada por recuperación, local y verificable, sobre una biblioteca personal de conocimiento técnico

**Reporte de proyecto · Maestría en Ingeniería de Inteligencia Artificial**

| | |
|---|---|
| **Autor** | Octavio Pérez Bravo |
| **Institución / programa** | [Universidad] · [Maestría en Ingeniería de IA] |
| **Asignatura / profesor** | [Asignatura] · [Profesor] |
| **Fecha** | 2 de noviembre de 2026 |
| **Repositorio** | `Library-RAG` (código, `DECISIONS.md` con 59 decisiones, `PLAN.md`, `eval/`) |

---

## Resumen

Este reporte documenta el diseño, la implementación y la evaluación de **Biblio-RAG**, un sistema de generación aumentada por recuperación (RAG) que opera al 100 % en local sobre una biblioteca personal de documentos Markdown. Durante el proyecto la biblioteca pasó de 2 252 notas (61 753 fragmentos, 18.7 millones de tokens) a unos 25 000 documentos (≈697 000 fragmentos, ≈235 millones de tokens, 5 GB de índice) al incorporarse libros técnicos. El sistema ofrece tres usos —buscador, tutor con citas y asistente de redacción— y un conjunto de herramientas de estudio (práctica con repetición espaciada, mapas conceptuales, detección de lagunas, lector integrado).

El problema de ingeniería se define por cuatro restricciones: ninguna conexión de red en tiempo de ejecución, toda respuesta citada con su fuente y su abstención cuando falta evidencia, medición obligatoria de cada cambio de recuperación, y un presupuesto de memoria para modelos inferior a 8 GB en un equipo de consumo (RTX 5060 de 8 GB, 32 GB de RAM). La metodología combina ciencia del diseño con un registro de decisiones de arquitectura (ADR): 59 decisiones, cada una con contexto, alternativas, decisión y evidencia, y la regla de *medir antes de adoptar*.

Entre los resultados: la recuperación híbrida (BM25 + vectores + fusión RRF) con un *cross-encoder* eleva el *recall@8* de 0.647 a 0.769 sobre 64 preguntas; un LLM de 4B parámetros cuantizado alcanza una fidelidad de 0.87 en respuestas directas, y la alternativa «map-reduce» la reduce a 0.58, por lo que se rechazó como opción por defecto; un modo de razonamiento con un modelo de mezcla de expertos de 26B parámetros ejecutado en RAM eleva de 46 % a 80 % las citas literales del razonamiento y mantiene el 100 % de abstenciones correctas en las 27 preguntas del conjunto de razonamiento. Se discuten además las limitaciones de la evaluación (conjuntos pequeños, parcialmente redactados con asistencia de IA, calificaciones humanas pendientes) y las lecciones sobre el desarrollo asistido por agentes de IA bajo gobernanza explícita.

**Palabras clave:** generación aumentada por recuperación (RAG), recuperación híbrida, reranking, modelos de lenguaje locales, cuantización, presupuesto de memoria, evaluación de RAG, repetición espaciada, ingeniería de sistemas de IA, registro de decisiones de arquitectura.

### Abstract

This report documents the design, implementation and evaluation of **Biblio-RAG**, a fully local retrieval-augmented generation system over a personal Markdown library that grew during the project from 2,252 notes to about 25,000 documents (≈235 M tokens). Under four constraints (no network at runtime, mandatory citations with abstention, measured retrieval changes, and a model memory budget under 8 GB on consumer hardware), the work follows an iterative design-science process recorded in 59 architecture decision records. Hybrid retrieval with a cross-encoder raised section recall@8 from 0.647 to 0.769; a quantized 4B model reached 0.87 answer faithfulness while per-guide map-reduce lowered it to 0.58 and was rejected as default; a reasoning mode on a 26B mixture-of-experts model run from system RAM raised the share of literally quoted reasoning steps from 46 % to 80 % while keeping every expected abstention. The report also discusses evaluation threats and the governance of AI-assisted development.

---

## Tabla de contenido

1. Introducción (incluye qué es y qué no es el sistema)
2. Justificación del proyecto
3. Marco teórico y trabajo relacionado
4. Metodología y proceso de ingeniería
5. Requisitos y diseño de la solución
6. Decisiones de ingeniería
7. Evaluación experimental y resultados
8. Discusión
9. Limitaciones y amenazas a la validez
10. Trabajo futuro
11. Conclusiones
- Referencias
- Apéndice A. Registro de decisiones de arquitectura
- Apéndice B. Pila tecnológica
- Apéndice C. Reproducibilidad
- Apéndice D. Conjuntos de evaluación y su autoría
- Apéndice E. Cronología
- Apéndice F. Glosario
- Apéndice G. Declaración sobre el uso de herramientas de IA

---

## 1. Introducción

### 1.1 Contexto

Un profesional de datos e IA acumula conocimiento técnico en forma de notas propias y de libros: guías de algoritmos, resúmenes de teoría, documentación de herramientas, capítulos de libros de referencia. El conocimiento está escrito, pero su *accesibilidad* se degrada con el volumen. En este proyecto la colección pasó de 2 252 notas a unos 25 000 documentos, de los cuales ≈22 400 son capítulos de libros (en su mayoría en inglés, mientras que las notas propias y las preguntas están en español) y 2 736 son notas propias. A esa escala, la búsqueda por palabras clave falla en las paráfrasis y la lectura lineal es inviable.

Los modelos de lenguaje de gran tamaño ofrecen una interfaz natural para consultar el conocimiento, pero plantean tres problemas cuando el corpus es personal: su conocimiento paramétrico no coincide con el de la biblioteca, pueden afirmar cosas sin respaldo, y su uso mediante servicios externos implica enviar fuera de la máquina contenido propio o con restricciones de redistribución.

### 1.2 Planteamiento del problema

Se requiere un sistema que responda preguntas sobre una biblioteca personal grande y cambiante, que:

1. funcione sin red (ninguna API ni telemetría, independencia operativa);
2. cite la sección exacta de cada afirmación y se abstenga cuando la evidencia no alcance;
3. corra en un equipo de consumo (una GPU de 8 GB de VRAM, de la que solo ≈4.8 GiB están libres con el escritorio abierto);
4. sea medible: cada cambio en la recuperación o en la generación debe justificarse con números.

### 1.3 Preguntas de investigación

- **PI1.** ¿Puede un sistema RAG operar sin red sobre una biblioteca personal grande, con calidad de recuperación útil, en hardware de consumo?
- **PI2.** ¿Qué combinación de recuperación (densa, léxica, fusión, *reranker*) maximiza la recuperación en este corpus y a qué costo de latencia y memoria?
- **PI3.** ¿Hasta dónde llega un LLM de 4B parámetros cuantizado como tutor con citas obligatorias, y qué mecanismos de verificación compensan sus límites?
- **PI4.** ¿Un modelo mayor, ejecutado parcialmente en RAM, mejora el razonamiento y la fidelidad dentro de un modo con capas marcadas y con límites temáticos?
- **PI5.** ¿Qué prácticas de ingeniería (registro de decisiones, medición previa, pruebas, gobernanza del agente de desarrollo) mantienen la coherencia de un proyecto que evoluciona con rapidez?

### 1.4 Objetivos

**General.** Diseñar, implementar y evaluar un sistema RAG local, verificable y medido, sobre una biblioteca personal de documentos Markdown, que sirva como buscador, tutor y asistente de redacción.

**Específicos.**

1. Implementar un pipeline de ingesta estructural, indexación incremental y recuperación híbrida con *reranking*, medido sobre conjuntos de preguntas elaborados por el usuario.
2. Integrar un LLM local detrás de una interfaz sustituible, con validación determinista de citas y abstención explícita.
3. Fijar y verificar un presupuesto de memoria por componente (embeddings, *reranker*, LLM) en el hardware de referencia.
4. Construir herramientas de estudio que reutilicen la recuperación y el LLM sin violar la regla de citas (práctica, mapas, explicación al estilo Feynman, detección de lagunas).
5. Explorar un modo de razonamiento con conocimiento general acotado a los temas que la biblioteca explica.
6. Registrar y justificar cada decisión de arquitectura, con su evidencia.

### 1.5 Alcance y exclusiones

Se trata de un sistema de **un solo usuario**, en una sola máquina de referencia, con un corpus en español e inglés. Quedan fuera: entrenamiento o ajuste fino de modelos (descartado por el ADR-001, salvo una capa de estilo opcional no realizada), despliegue multiusuario, y la comparación contra servicios comerciales de RAG. La interfaz web es parte del alcance porque condiciona el uso real, pero no se evalúa con estudios de usuarios.

### 1.6 Qué es y qué no es Biblio-RAG

Delimitar la identidad del sistema es una decisión de diseño tan importante como sus funciones: cada exclusión evita una promesa que el sistema no puede cumplir y orienta qué se mide. La tabla resume la frontera; el texto posterior la justifica.

| Biblio-RAG **es** | Biblio-RAG **no es** |
|---|---|
| Un sistema RAG: el conocimiento vive en los documentos y se recupera en cada consulta | Un modelo entrenado o ajustado con la biblioteca: no aprende ni «memoriza» sus notas |
| Un buscador, un tutor con citas y un asistente de redacción sobre **una** biblioteca personal | Un chatbot de propósito general ni un sustituto de un modelo de frontera: fuera de los temas que la biblioteca explica se abstiene |
| Un sistema local: tras `models pull` no abre conexiones fuera de `localhost` | Un servicio en la nube, ni una aplicación con cuentas, telemetría o sincronización |
| Un sistema que cita la sección de cada afirmación y declara cuándo no sabe | Un oráculo de verdad: garantiza que la cita existe y está en el fragmento, no que la biblioteca ni el resumen sean correctos |
| Una herramienta de **un** usuario, en **una** máquina de referencia (Windows 11, GPU de 8 GB) | Un producto multiusuario, escalable ni probado en otro hardware, idioma o corpus |
| Un lector del corpus: la biblioteca es de solo lectura y los borradores se escriben fuera de ella | Un editor autónomo: no modifica, reorganiza ni borra notas, ni actúa por su cuenta |
| Un caso de ingeniería con decisiones registradas y mediciones | Una contribución algorítmica: combina técnicas conocidas (BM25, RRF, *cross-encoder*, FSRS) y no propone métodos nuevos |
| Un apoyo al estudio: cuestionarios, repaso espaciado, mapas, lagunas | Un sustituto de leer, comprender y escribir: la técnica de Feynman y la práctica exigen el esfuerzo de la persona |
| Un sistema con un modo de razonamiento que **separa** lo que dicen las notas de lo que aporta el modelo | Un sistema que mezcla conocimiento propio del modelo con las notas sin marcarlo, ni que lo permite en temas ausentes |

**Por qué no es un modelo entrenado.** Un ajuste fino recuerda mal los hechos, no puede citar y obliga a reentrenar con cada cambio (ADR-001). Prometer «un modelo que conoce mi biblioteca» sería prometer algo no verificable; en cambio, un sistema que recupera y cita puede auditarse afirmación por afirmación.

**Por qué no es un asistente general.** La regla 2 del proyecto exige citas o abstención. Un asistente general contestaría casi todo y, con un modelo de 4B parámetros, parte de ello sería inventado (en la prueba de *map-reduce* alucinó en las 3 preguntas ausentes). El valor de este sistema es la **fiabilidad acotada**: una respuesta de «no encuentro esto en tu biblioteca» es un resultado correcto, no un fallo. Por la misma razón no está diseñado ni medido para generar código: es una capacidad que no se evaluó y que queda fuera de los objetivos.

**Por qué no es un servicio en la nube.** La privacidad, la reproducibilidad (modelos fijados por hash) y el costo marginal nulo se pierden en cuanto un dato sale de la máquina (sección 2.3). El precio es un techo de calidad que el sistema compensa con estructura y no con tamaño de modelo.

**Por qué no garantiza la verdad.** Lo que se verifica es estructural: que la cita exista, que la frase citada esté literalmente en el fragmento, que el sistema se abstenga cuando la evidencia es insuficiente. No se verifica que el resumen que acompaña a la cita sea fiel (se observó un caso con cita correcta y resumen invertido), ni que la capa de «conocimiento general» del modo de razonamiento sea correcta; por eso va rotulada como ajena a las notas. Presentarlo como un verificador de hechos ocultaría un riesgo conocido.

**Por qué es de un usuario y una máquina.** Todas las cifras proceden de un corpus, un usuario y un equipo. Generalizarlas sería una afirmación sin evidencia (sección 9); el reporte las presenta como un caso documentado y no como resultados universales.

**Por qué no propone algoritmos nuevos.** El objetivo es ingeniería de sistemas de IA bajo restricciones reales: elegir, combinar, medir y justificar. Su contribución es el método (decisiones con evidencia, verificación determinista sobre modelos probabilísticos, presupuesto de memoria medido) y no una técnica de recuperación inédita.

**Por qué no es un sustituto del estudio.** Las herramientas de estudio existen porque la evidencia indica que la recuperación activa y la práctica distribuida mejoran la retención más que la relectura (sección 3.8). Un sistema que respondiera todo sin exigir esfuerzo empeoraría el aprendizaje que busca apoyar.

### 1.7 Estructura del documento

La sección 1.6 delimita qué es y qué no es el sistema; la 2 justifica el proyecto; la 3 resume el marco teórico; la 4 describe la metodología, el proceso y la gobernanza; la 5 expone los requisitos y la arquitectura; la 6 presenta las decisiones de ingeniería con sus alternativas y evidencia; la 7 reúne los resultados; la 8 los discute; la 9 enumera las limitaciones; las secciones 10 y 11 cierran con trabajo futuro y conclusiones.

---

## 2. Justificación del proyecto

### 2.1 Justificación del problema

**Escala y degradación de la accesibilidad.** La biblioteca creció aproximadamente once veces en número de documentos durante el proyecto (de 2 252 a ≈25 000), y el índice de 419 MB pasó a ≈5 GB. Ningún método de recuperación aislado resuelve el problema: sobre 64 preguntas del usuario, la recuperación densa alcanza un *recall@8* de sección de 0.546, BM25 de 0.648 y la fusión híbrida de 0.647; solo al añadir un *cross-encoder* se llega a 0.769 (sección 7.1). Aun así, cerca de una de cada cuatro preguntas no recupera la sección esperada entre las ocho que lee el tutor, lo que justifica un trabajo de ingeniería sistemático y no una integración trivial de bibliotecas.

**Heterogeneidad del corpus.** Las notas contienen jerarquías de encabezados, bloques de código, fórmulas LaTeX, tablas y diagramas ASCII. El 71 % de los documentos añadidos posteriormente está en inglés, mientras que las consultas se formulan en español. Un sistema genérico de troceado por tamaño fijo y búsqueda puramente léxica o puramente densa no está preparado para ello.

### 2.2 Justificación técnica de RAG frente a otras alternativas

El ADR-001 compara tres estrategias para que un modelo «conozca» la biblioteca:

| Alternativa | Ventaja | Costo / motivo de descarte |
|---|---|---|
| Entrenar un LLM desde cero | Control total | Requiere billones de tokens; la biblioteca aporta millones. Resultado esperado: imitación superficial del estilo con contenido alucinado |
| Ajuste fino (LoRA) sobre las guías | Aprende forma y estilo | Recuerda mal los hechos, no puede citar y obliga a reentrenar con cada cambio |
| **RAG** (elegida) | Citas verificables, actualización inmediata al reindexar, menos alucinación | La calidad depende de la recuperación, que pasa a ser la pieza crítica |

La consecuencia de diseño es que *el conocimiento vive en los documentos y no en los pesos*. El ajuste fino se conserva únicamente como una capa de estilo opcional para la redacción, nunca como fuente de conocimiento.

### 2.3 Justificación de la operación local

El requisito original era la «independencia absoluta de modelos externos». El ADR-002 lo interpreta como **independencia operativa**: en tiempo de ejecución no hay llamadas fuera de `localhost`, ni APIs, ni telemetría, y ningún dato sale de la máquina. Se acepta la dependencia de *origen*: los pesos abiertos fueron entrenados por terceros, pero, una vez descargados y fijados por revisión, no pueden revocarse ni modificarse. Las razones son:

- **Privacidad y derechos:** el corpus contiene notas personales y contenido de libros con restricciones de redistribución.
- **Reproducibilidad:** los modelos se fijan por hash de revisión y se verifican con `biblio doctor`; un servicio externo puede cambiar sin aviso.
- **Costo marginal nulo** y disponibilidad sin conexión.
- **Valor formativo:** operar con restricciones reales de memoria y latencia es el núcleo de la ingeniería de sistemas de IA.

El precio de esta decisión es un techo de calidad asumido: un modelo de 3–4B parámetros cuantizado rinde por debajo de los modelos de frontera. Gran parte del proyecto consiste en compensar ese techo con estructura (recuperación de calidad, validación determinista, abstención) en lugar de con tamaño.

### 2.4 Justificación del componente de estudio

Un sistema que solo responde preguntas desaprovecha el valor de la biblioteca. La evidencia sobre aprendizaje indica que la *práctica de recuperación* y la *práctica distribuida* mejoran la retención más que la relectura (Roediger y Karpicke, 2006; Dunlosky et al., 2013; Cepeda et al., 2006). Por ello se integraron cuestionarios generados desde secciones reales, un planificador de repasos (FSRS), la técnica de explicación con las propias palabras («Explícamelo tú») y mapas conceptuales (Novak y Cañas, 2008). Todos reutilizan la misma recuperación y la misma regla de citas, de modo que la verificación estructural diseñada para el tutor se amortiza en las demás funciones.

### 2.5 Pertinencia para la ingeniería de sistemas de IA

El proyecto es un caso de estudio de ingeniería de sistemas de IA bajo restricciones reales, y aporta:

1. **Evaluación como práctica de primera clase:** líneas base antes de optimizar (ADR-009), métricas de recuperación y de generación, y conjuntos de preguntas elaborados por el usuario.
2. **Gestión explícita de recursos:** un presupuesto de memoria calculado, medido y registrado por componente (secciones 5.6 y 6.5).
3. **Verificación determinista sobre modelos probabilísticos:** comprobación literal de citas, abstención fija, umbrales sobre puntajes de un *reranker*.
4. **Gobernanza del desarrollo asistido por IA:** un contrato (`CLAUDE.md`), procedimientos (*skills*) y decisiones versionadas (ADR) que acotan el trabajo de un agente de programación (sección 4.7).
5. **Honestidad metodológica:** se documentan los defectos de las propias herramientas de evaluación y las decisiones rechazadas, no solo los éxitos.

### 2.6 Beneficiarios, valor y riesgos aceptados

El beneficiario directo es la persona dueña de la biblioteca, que obtiene un buscador, un tutor y un sistema de repaso sobre su propio conocimiento sin exponerlo. El valor transferible es el método y el diseño. Los **riesgos aceptados**, registrados en los ADR, son: techo de calidad de los modelos pequeños (ADR-008), latencia alta del modo de razonamiento (~1 minuto por respuesta, ADR-057), dependencia de la calidad de la recuperación, y un único entorno de referencia (ADR-011).

---

## 3. Marco teórico y trabajo relacionado

Esta sección describe los conceptos, algoritmos y técnicas sobre los que se construye el sistema. Cada subsección indica dónde se usa en el proyecto (ADR o módulo), de modo que el marco teórico pueda leerse como el mapa conceptual de la implementación. El orden sigue el flujo de los datos: representación del texto (3.1–3.3), recuperación (3.4–3.8), generación (3.9–3.12), recursos (3.13), evaluación (3.14), aprendizaje (3.15–3.17) y verificación (3.18).

### 3.1 Generación aumentada por recuperación (RAG)

Lewis et al. (2020) proponen combinar una **memoria paramétrica** (los pesos de un modelo de lenguaje) con una **memoria no paramétrica** (un índice de documentos consultado en cada petición) para tareas intensivas en conocimiento. Gao et al. (2023) distinguen tres generaciones: *naive RAG* (recuperar y concatenar), *advanced RAG* (con pre y posprocesamiento de la consulta y de los resultados) y *modular RAG* (componentes intercambiables). Biblio-RAG pertenece a la tercera: `Embedder`, `Reranker`, `LLMClient` y el recuperador son interfaces (`Protocol`) sustituibles.

La ventaja es doble: el conocimiento se actualiza reindexando, sin reentrenar, y cada respuesta puede atribuirse a una fuente. Los riesgos son tres: que la recuperación no encuentre la evidencia (falso negativo), que encuentre evidencia irrelevante que el generador use igualmente, y que el generador ignore o contradiga la evidencia. Ji et al. (2023) clasifican la *alucinación* en contradicción con la fuente y en contenido no verificable; el proyecto ataca ambas con citas obligatorias, validación determinista y abstención (3.18). Liu et al. (2024) muestran además que los modelos usan peor la información situada en el centro de un contexto largo, un argumento a favor de pocos fragmentos de calidad (6–8) y no de muchos mediocres.

### 3.2 Tokenización y modelos de Transformer

Un **token** es la unidad que consume un modelo de lenguaje; los tokenizadores de subpalabras (Kudo y Richardson, 2018) dividen el texto en piezas frecuentes, de modo que un idioma con palabras largas como el español produce más tokens que palabras. Esto importa porque **todos los presupuestos del sistema se miden en tokens**: el tamaño de los fragmentos, la ventana del embedder, el contexto del LLM y el costo de la caché. Los modelos del proyecto son Transformers (Vaswani et al., 2017), cuyo núcleo es la **autoatención**: cada token pondera a los demás con pesos softmax(QKᵀ/√d)V. Su costo crece con la longitud de la secuencia, de lo que se derivan la ventana máxima del embedder (`max_seq_length`) y el contexto de 8 192 tokens del LLM.

### 3.3 Representaciones densas (*embeddings*)

Un codificador de oraciones (Reimers y Gurevych, 2019) mapea un texto a un vector de dimensión fija, de modo que la cercanía geométrica refleje similitud semántica. Con vectores normalizados, la **similitud coseno** se reduce al producto punto:

> cos(q, d) = (q · d) / (‖q‖ ‖d‖) = q · d  si ‖q‖ = ‖d‖ = 1.

El proyecto usa **BGE-M3** (Chen et al., 2024), un codificador multilingüe de 568 M de parámetros basado en XLM-RoBERTa (Conneau et al., 2020), con 1 024 dimensiones y contexto largo, entrenado para unir recuperación densa, léxica y multivectorial; aquí solo se usa la salida densa. Se elige por tratar español e inglés en un mismo espacio (ADR-004). Un detalle de ingeniería: el texto que se codifica antepone la ruta jerárquica del fragmento («Guía › Sección › Subsección»), de modo que el vector incorpora el contexto del encabezado aunque la cita lo muestre aparte.

### 3.4 Troceado estructural de documentos (*chunking*)

La unidad de recuperación determina el techo de calidad. El troceado por tamaño fijo con solapamiento rompe unidades semánticas; el troceado **estructural** aprovecha la jerarquía de encabezados del Markdown: cada sección es un fragmento, las largas se parten en límites de párrafo, las diminutas se fusionan con su vecina y las **unidades atómicas** (bloques de código, tablas, fórmulas `$$…$$`, diagramas) nunca se parten (ADR-003). Se omite el solapamiento porque la ruta jerárquica ya aporta el contexto que este intenta recuperar. El análisis sintáctico usa un *parser* CommonMark con tokens que conservan el número de línea (`markdown-it-py`), no expresiones regulares, porque `# comentario` dentro de un bloque de código no es un encabezado. Una propiedad verificable es la **conservación**: la concatenación de los textos de los fragmentos cubre el cuerpo del documento sin pérdidas ni duplicados.

### 3.5 Recuperación léxica: BM25 y FTS5

BM25 (Robertson y Zaragoza, 2009) puntúa un documento d frente a una consulta q como

> BM25(q, d) = Σₜ∈q IDF(t) · f(t,d)·(k₁+1) / ( f(t,d) + k₁·(1 − b + b·|d|/avgdl) ),

donde f(t,d) es la frecuencia del término, |d| la longitud del documento, avgdl la longitud media, k₁ la saturación de la frecuencia y b la normalización por longitud. Premia términos raros (IDF alto) y evita que un documento largo gane solo por repetir. Es fuerte con siglas, nombres propios e identificadores; es ciego a las paráfrasis. Se implementa con **FTS5**, la extensión de búsqueda de texto completo de SQLite, que expone `bm25()` con pesos por columna. Dos técnicas relevantes: el tokenizador `unicode61 remove_diacritics 2`, que hace que «regresion» coincida con «regresión» (FTS5 no tiene lematizador para español, y el proyecto no lo simula: la búsqueda densa cubre la morfología); y la **ponderación del encabezado** (0.5, ADR-044), medida empíricamente porque un peso mayor favorecía a las notas grandes.

### 3.6 Recuperación densa y búsqueda del vecino más cercano

La recuperación densa (Karpukhin et al., 2020) codifica consulta y documentos y devuelve los k vectores más cercanos. Hay dos familias de algoritmos: **exactos** (comparar con todos, O(N·d)) y **aproximados** (ANN), como los grafos HNSW (Malkov y Yashunin, 2020), que sacrifican un poco de exhaustividad por velocidad sublineal. El proyecto usa búsqueda **exacta** con `sqlite-vec` (ADR-005): a la escala inicial cuesta milisegundos, elimina el error de aproximación y mantiene un solo archivo. Con ≈697 000 vectores de 1 024 dimensiones el costo lineal crece, y el umbral de reevaluación fijado en 200 000 fragmentos fue superado (sección 10).

### 3.7 Fusión por rangos recíprocos (RRF)

Las listas densa y léxica tienen puntajes en escalas incomparables (distancia vs. BM25). RRF (Cormack et al., 2009) las combina usando solo el **rango** de cada documento:

> RRF(d) = Σᵣ 1 / (k + rangoᵣ(d)),  con k = 60.

No requiere calibrar puntajes, premia a los documentos que aparecen bien situados en varias listas y degrada con suavidad. El valor k = 60 es el propuesto por sus autores y se conserva en la configuración.

### 3.8 *Reranking* con *cross-encoders* y filtrado

Un **bi-encoder** (3.3) codifica consulta y texto por separado; un **cross-encoder** los procesa juntos con atención completa entre ambos (Nogueira y Cho, 2019), por lo que es más preciso pero cuesta una pasada por cada par. De ahí el patrón en dos etapas: una primera etapa barata recupera ≈40 candidatos y el cross-encoder (`bge-reranker-v2-m3`) los reordena. En el proyecto el *reranker* cumple además otras dos funciones: como **puntaje de relevancia absoluto** en la puerta temática del modo de razonamiento (3.12) y en la selección de secciones para los mapas, y como etapa **omitible** cuando la GPU está ocupada (ADR-014). El filtrado de índices de contenidos (ADR-044) es un posprocesado que retira del resultado las secciones que solo enumeran enlaces.

### 3.9 Modelos de lenguaje pequeños, cuantización y caché KV

La memoria de un LLM en inferencia es la suma de tres términos: pesos, caché de claves y valores (KV) y *buffers* del *runtime*. La **cuantización** reduce los pesos a menos bits; el formato **GGUF** de llama.cpp ofrece variantes por bloques como Q4_K_M (alrededor de 4.5 bits por peso efectivos, con una mezcla de precisiones por capa), con una pérdida de calidad pequeña a cambio de unas 3–4 veces menos memoria que fp16 (Dettmers y Zettlemoyer, 2023, estudian el compromiso entre bits y precisión). La caché KV evita recalcular la atención de los tokens ya procesados y crece linealmente con el contexto:

> bytes_KV = 2 · capas · cabezas_KV · dim_cabeza · bytes_por_elemento · n_contexto.

Para Qwen3-4B (36 capas, 8 cabezas KV de dimensión 128) son 144 KiB por token en f16: 1.1 GiB con 8 192 tokens, y ≈0.56 GiB con la caché cuantizada a 8 bits (`q8_0`). La **atención por bloques** (FlashAttention; Dao et al., 2022) reduce el tráfico de memoria y es lo que hace viable cuantizar la caché sin penalización notable. El reparto de capas entre GPU y CPU (*offloading*) permite repartir un modelo entre VRAM y RAM.

### 3.10 Mezcla de expertos (MoE) y ejecución desde RAM

En un modelo MoE (Shazeer et al., 2017) cada capa tiene varios «expertos» y un enrutador activa solo algunos por token. Gemma 4 26B-A4B tiene 25.2 B de parámetros totales y 3.8 B activos, de modo que su costo de cómputo por token es el de un modelo pequeño aunque su conocimiento esté repartido en un modelo grande. La consecuencia de ingeniería es que se pueden **mapear en memoria** (`mmap`) los pesos desde el disco y la RAM y ejecutar con la GPU como apoyo, a una velocidad tolerable (≈51 s por respuesta razonada), a diferencia de un modelo denso del mismo tamaño total.

### 3.11 Ingeniería de prompts, salida estructurada y decodificación

Un LLM se controla con instrucciones en el *prompt* del sistema y del usuario. Técnicas usadas: **instrucciones con formato de salida fijo** (capas marcadas, JSON para planes y juicios), **ejemplos mínimos** (*few-shot*) cuando una regla general no bastó (notas en inglés, ADR-042) y **temperatura baja** (0.2) para respuestas más reproducibles. El razonamiento paso a paso (Wei et al., 2022) mejora las tareas compuestas; en el proyecto se aprovecha de forma controlada: cada paso del «Razonamiento» debe abrir con una frase copiada de un fragmento y citada, de modo que el razonamiento sea auditable. Algunos modelos tienen un modo de «pensamiento» interno; en Gemma se desactiva con `reasoning_effort: "none"` para obtener respuestas directas.

### 3.12 Abstención y detección de cobertura temática

Una respuesta correcta puede ser «no sé». La **abstención** se implementa en el proyecto como una acción del sistema, no del modelo: si la evidencia es insuficiente se devuelve una frase fija sin generar texto. Para decidirlo en preguntas amplias se usa una **puerta temática**: el LLM propone varios nombres del campo (en ambos idiomas, para reducir la sensibilidad a la redacción), cada nombre se convierte en una pregunta definicional, el *reranker* puntúa en CPU las secciones recuperadas y el campo se considera «explicado» si al menos `gate_min` secciones superan `gate_score`. Es un **clasificador por umbrales** sobre una señal de relevancia absoluta, con umbrales fijados con datos etiquetados (21 campos cubiertos con puntaje ≥ 0.76 y 22 ausentes con ≤ 0.15). El LLM solo propone nombres; la decisión la toma el *reranker*, de modo que el modelo no se auto-evalúa.

### 3.13 Gestión de recursos y concurrencia

El equipo de referencia obliga a tratar la memoria como recurso planificado: se calcula con la VRAM *libre* medida, se arbitra el uso de la GPU (el LLM tiene prioridad; la indexación y el *reranker* ceden), se carga un modelo a la vez y se desactiva la caché de prompts del *runtime* (`LLAMA_ARG_CACHE_RAM=0`) tras medir su crecimiento por petición. La **indexación incremental** usa un *hash* SHA-256 del contenido para detectar cambios y una transacción por documento, de modo que una interrupción no deja el índice inconsistente; la reconstrucción completa se hace en un archivo temporal que se intercambia solo al terminar. El servidor serializa el trabajo de los modelos con un candado (`Engine.lock`) porque comparten GPU y memoria.

### 3.14 Evaluación de sistemas RAG

**Recuperación.** Sea Q el conjunto de preguntas y Rᵢ el conjunto de secciones aceptables de la pregunta i:

> recall@k = (1/|Q|) Σᵢ 1[ topₖ(i) ∩ Rᵢ ≠ ∅ ],
> MRR = (1/|Q|) Σᵢ 1/rangoᵢ  (rango de la primera sección aceptable),
> nDCG@k = DCG@k / IDCG@k,  con DCG@k = Σⱼ₌₁ᵏ relⱼ / log₂(j+1)  (Järvelin y Kekäläinen, 2002).

El proyecto mide *recall@k* a nivel de sección, MRR y nDCG@8, desglosados por tipo de pregunta (búsqueda puntual, término exacto, síntesis, ausente). BEIR (Thakur et al., 2021) muestra que el rendimiento depende del dominio, lo que justifica evaluar con preguntas del propio corpus.

**Generación.** Se evalúan la **fidelidad** (afirmaciones respaldadas por la cita / afirmaciones totales), la **cobertura** (puntos clave presentes), la **validez de las citas**, la **abstención correcta** y la latencia. RAGAS (Es et al., 2023) automatiza estas métricas con un modelo juez; aquí se descartó por circularidad (un LLM pequeño juzgando a otro) y por la regla de cero red, y se usa calificación humana sobre una hoja con el texto completo de cada cita. Con 14–70 preguntas, un punto de diferencia equivale a una o dos preguntas, por lo que las comparaciones deben leerse con cautela.

### 3.15 Repetición espaciada y práctica de recuperación

La práctica de recuperación mejora la retención más que la relectura (Roediger y Karpicke, 2006; Dunlosky et al., 2013) y su distribución en el tiempo la amplía (Cepeda et al., 2006). Los planificadores clásicos son las cajas de Leitner y SM-2. **FSRS** (Ye et al., 2022) modela cada elemento con dos variables: la **estabilidad** S (días hasta que la probabilidad de recuerdo cae al 90 %) y la **dificultad** D. En su versión 4.5, la probabilidad de recordar tras t días es

> R(t, S) = (1 + (19/81)·t/S)^(−0.5),

y el próximo repaso se programa cuando R baja al nivel objetivo (`desired_retention`). El proyecto no almacena el calendario: lo **recalcula** repitiendo las respuestas de cada sección en orden (ADR-025), por lo que cambiar los parámetros reprograma todo sin migraciones.

### 3.16 Técnica de Feynman y verificación idea por idea

La técnica de Feynman consiste en explicar un concepto con las propias palabras para exponer lo que no se comprende. «Explícamelo tú» la automatiza en dos pasos: (1) se extraen de las notas las **ideas clave** del tema, que no se muestran antes de la explicación; (2) se compara la explicación del usuario con ellas, y cada observación debe llevar una **cita literal** comprobada por código. La comparación semántica entre ideas usa similitud coseno con un umbral, que es una fuente de error medida (paráfrasis válidas puntúan 0.63–0.72 con un umbral de 0.75).

### 3.17 Mapas conceptuales y extracción de conocimiento

Un mapa conceptual (Novak y Cañas, 2008) representa conceptos como nodos y relaciones como proposiciones etiquetadas («A —permite→ B»). El proyecto lo construye con **extracción de información** por fragmento con el LLM (conceptos y relaciones) y filtra con **verificación textual**: un concepto solo existe si aparece literalmente en su fragmento (ADR-022). Después aplica una selección de secciones por relevancia, una elección de ramas clave por el LLM (que filtra y no añade) y una disposición geométrica calculada en el servidor (ADR-036/037, ADR-051). La **detección de lagunas** (ADR-029/030/047) usa embeddings solo para preseleccionar candidatos de fusión, porque el coseno no separa «el mismo tema reformulado» de «temas vecinos», y delega la decisión al LLM con evidencia citada.

### 3.18 Verificación determinista sobre salidas probabilísticas

El hilo común de las técnicas de generación es **no confiar en el modelo para verificar al modelo**. Los controles son comprobaciones de código sobre su salida:

- **Validez de citas:** cada `[n]` debe referir a un fragmento entregado; una cita inexistente es un error.
- **Cita literal:** la frase entre comillas debe estar en el fragmento, tras normalizar acentos, mayúsculas y espacios (`practice.feynman.contains`), con una longitud mínima de palabras consecutivas para que el acierto no sea trivial.
- **Marcadores huérfanos:** una respuesta que conserva el marcador de ejemplo del prompt (`[n]`) se rechaza, porque los modelos pequeños lo copian.
- **Abstención fija:** la frase de «no encuentro esto» se detecta exactamente.
- **Umbrales sobre un modelo especializado:** el *reranker* decide la cobertura temática en lugar del LLM generativo.

Estas técnicas acotan, pero no eliminan, el riesgo: verifican la presencia de la cita, no que la paráfrasis que la acompaña sea fiel (sección 9).

---

## 4. Metodología y proceso de ingeniería

### 4.1 Enfoque: ciencia del diseño con decisiones registradas

El trabajo sigue el paradigma de la **investigación en ciencia del diseño** (Hevner et al., 2004): se construye un artefacto para un problema real, se evalúa con criterios explícitos y se registran las decisiones y su justificación. Cada ciclo es:

1. **Identificar** el problema o la pregunta (a menudo una falla observada en el uso real).
2. **Medir** el estado actual con un conjunto de preguntas o una prueba.
3. **Proponer** alternativas con su costo.
4. **Decidir** (con el usuario cuando la decisión es no trivial) y escribir el ADR.
5. **Implementar** con pruebas, volver a **medir** y **registrar** ambos números.

### 4.2 Reglas no negociables

El proyecto se rige por seis reglas, fijadas en el contrato del proyecto (`CLAUDE.md`):

1. **Cero red en tiempo de ejecución.** Solo `biblio models pull` abre conexiones externas; los modelos se cargan de disco con `HF_HUB_OFFLINE=1`; las pruebas bloquean sockets (`pytest-socket`).
2. **Toda respuesta generada cita sus fuentes** (`ruta/archivo.md › Sección`) y, si falta evidencia, lo dice.
3. **Ningún cambio de recuperación se acepta sin medirlo**: se corre la evaluación antes y después y se registran ambas cifras.
4. **Presupuesto de memoria inferior a 8 GB para modelos**; todo modelo nuevo o cambio de contexto pasa por un cálculo y una medición.
5. **Las decisiones de arquitectura se registran** y una decisión aceptada no se revierte sin una nueva que la sustituya.
6. **Cambiar el modelo de embeddings o la versión del chunker obliga a reindexar**, y el código se niega a mezclar vectores de versiones distintas.

### 4.3 Registro de decisiones de arquitectura

`DECISIONS.md` contiene 59 ADR numerados (Apéndice A), con un formato fijo: *Contexto*, *Alternativas*, *Decisión*, *Consecuencias*, y *Evidencia* cuando la decisión se apoya en mediciones (con la referencia al archivo de resultados). Un ADR aceptado no se edita salvo su estado; un cambio se registra como un ADR nuevo que lo sustituye (por ejemplo, el ADR-012 sustituye al 008, el 013 al 012 y el 014 al 013 al ir apareciendo evidencia sobre la GPU). Las decisiones no triviales se proponen al usuario con alternativas y recomendación antes de implementarse.

### 4.4 Fases y cronología

El plan inicial define fases con criterios de salida; se añadieron después ampliaciones de estudio y exploración y una fase de razonamiento:

| Fase | Contenido | Hito principal |
|---|---|---|
| 0 | Entorno y garantías | `doctor` verde con red bloqueada en proceso; 0 intentos de red (17 oct.) |
| 1 | Ingesta y troceado | 2 252 archivos → 61.8 k fragmentos, 18.7 M tokens, 0 violaciones de invariantes |
| 2 | Indexación | Índice completo en 2 300 s en GPU; segunda corrida sin cambios: 0.6 s |
| 3 | Evaluación y líneas base | Conjuntos v1 (48 preguntas) y v2 (70); líneas base densa, BM25, híbrida |
| 4 | Recuperación híbrida y *reranking* | `hybrid-rerank` por defecto; ADR-013/014 |
| 5 | Tutor (`ask`) | Qwen3-4B en Ollama; validación de citas; evaluación v3 |
| 6 | Asistente de redacción (`draft`) | Esquema editable y secciones con su propia recuperación |
| 7 | Opcionales | Interfaz web, vigilancia de cambios; grafo de enlaces descartado |
| 9 (registro) | Estudio y exploración | Práctica, mapas, lector, biblioteca, conversaciones (ADR-019 a 055) |
| 8 | Razonamiento | Modo «Razonar», modelo en RAM, resúmenes (ADR-056 a 059, 30 oct.–1 nov.) |

La cronología detallada está en el Apéndice E. El proyecto se inicializó el 5 de octubre de 2026 y el último ADR registrado es del 1 de noviembre de 2026.

### 4.5 Estrategia de evaluación

- **Líneas base antes de optimizar** (ADR-009): primero un buscador sin LLM, medido, y después la generación.
- **Conjuntos de preguntas** redactados por el usuario (regla del proyecto), con secciones esperadas y, cuando es legítimo, secciones alternativas. Las excepciones, en las que el asistente redactó un borrador a pedido, se rotulan en el propio archivo y se tabulan en el Apéndice D.
- **Métricas de recuperación:** *recall@{1, 5, 8, 20}* de sección, *MRR*, *nDCG@8*, por tipo de pregunta (búsqueda puntual, término exacto, síntesis y ausente).
- **Métricas de generación:** fidelidad (afirmaciones respaldadas / afirmaciones), cobertura (puntos clave presentes), abstención, validez de citas, latencia. La fidelidad y la cobertura las califica una persona sobre una hoja que muestra el texto completo de cada fragmento citado.
- **Registro:** cada corrida añade una línea JSONL con fecha y *commit* en `eval/results/`.

### 4.6 Estrategia de calidad del software

- Tipado obligatorio (`mypy --strict` sobre `src/`), formato y reglas (`ruff`), y `pytest` con red bloqueada.
- 528 casos de prueba (374 funciones; ≈9 500 líneas de pruebas frente a ≈75 módulos Python); los tests de troceado usan fixtures reales y cada defecto de parseo se convierte en una fixture nueva.
- Los componentes más propensos al cambio (`Embedder`, `Reranker`, `LLMClient`) se definen como `Protocol` y se inyectan desde la configuración; no hay valores mágicos en el código, todo vive en `config/default.toml`.
- Garantía de red en capas (sección 5.7) y comprobaciones de entorno con `biblio doctor`.

### 4.7 Desarrollo asistido por IA y gobernanza

El sistema se desarrolló con un **agente de programación basado en LLM** (Claude Code) bajo dirección humana. Dado que un agente puede producir mucho código sin criterio de producto, el proyecto lo acota con cuatro mecanismos:

1. **Contrato explícito** (`CLAUDE.md`): reglas no negociables, pila, convenciones y «trampas conocidas» (errores ya cometidos y su causa).
2. **Procedimientos reutilizables** (*skills*) para decisiones de alto riesgo: garantía de operación sin red, presupuesto de memoria, troceado, evaluación de recuperación, diseño de prompts y registro de decisiones.
3. **Memoria externa versionada**: `DECISIONS.md` y `PLAN.md` hacen que cada sesión parta de lo decidido, no de lo recordado.
4. **Humano en las bifurcaciones**: el usuario elige entre alternativas con costos explícitos (modelo, estrategia, interfaz) y redacta o aprueba los conjuntos de evaluación.

Esto introduce una amenaza a la validez que se discute en la sección 9: parte de los conjuntos de evaluación auxiliares fue redactada por el asistente a pedido del usuario y está rotulada como tal.

### 4.8 Entorno de referencia

| Elemento | Valor (ADR-011) |
|---|---|
| Sistema operativo | Windows 11 |
| CPU | AMD Ryzen 7 5700X, 8 núcleos físicos / 16 lógicos |
| RAM | 31.9 GiB (≈9.7 GiB disponibles con la carga habitual del escritorio) |
| GPU | NVIDIA GeForce RTX 5060, 8.0 GiB de VRAM, ≈4.8 GiB libres en reposo |
| Software | Python 3.12 (uv), SQLite 3.50 con `sqlite-vec` y FTS5, PyTorch con CUDA 13.0 |

La memoria *libre* de la GPU, y no la nominal, es el presupuesto real de VRAM: el cálculo parte de lo medido con las aplicaciones habituales abiertas.

---

## 5. Requisitos y diseño de la solución

### 5.1 Requisitos funcionales

- **RF1 Buscar:** devolver secciones ordenadas con su ruta y líneas (`search`).
- **RF2 Explicar:** responder con síntesis de varias guías, con citas (`ask`) y en tres niveles (principiante, intermedio, avanzado; ADR-032).
- **RF3 Redactar:** proponer un esquema editable y escribir un borrador consistente con las guías existentes (`draft`).
- **RF4 Estudiar:** cuestionarios con repaso espaciado, mapas y diagramas, explicación al estilo Feynman, plan de estudio y detección de lagunas.
- **RF5 Leer y explorar:** lector integrado, estantería de libros y organización de notas por carpeta.
- **RF6 Razonar:** respuestas con capas marcadas y conocimiento general acotado.

### 5.2 Requisitos no funcionales

| Requisito | Criterio |
|---|---|
| Privacidad / independencia | Ninguna conexión fuera de `localhost` en ejecución |
| Verificabilidad | Toda afirmación citada; citas inexistentes y marcadores `[n]` detectados |
| Memoria | < 8 GB para modelos; ≥ 10 % de margen en VRAM |
| Latencia | Búsqueda < 3 s; tutor ~3 s por respuesta con modelo cargado |
| Mantenibilidad | Tipado estricto, pruebas, interfaces sustituibles, configuración externa |
| Reproducibilidad | Modelos fijados por hash; versión de chunker y embeddings en el índice |

### 5.3 Arquitectura

```
                          ┌──────────────────────── Interfaces ───────────────────────┐
                          │  CLI (typer)            Web local (stdlib HTTP + JS)      │
                          └──────┬─────────────────────────────┬──────────────────────┘
                                 │                             │
        ┌────────────────────────▼─────────────────────────────▼────────────────────────┐
        │  Generación y estudio                                                          │
        │  tutor · conversaciones · modo Razonar · borradores · práctica (FSRS) · mapas  │
        │  Feynman · lagunas · plan · resúmenes     ── validación de citas ──            │
        └──────────────┬────────────────────────────────────────────┬───────────────────┘
                       │ LLMClient (Protocol)                       │ Retriever (Protocol)
              ┌────────▼─────────┐                       ┌──────────▼───────────────────────┐
              │ Ollama (localhost)│                       │ Primera etapa: densa + BM25 → RRF │
              │ Qwen3-4B / Gemma 4│                       │ Segunda etapa: cross-encoder      │
              └──────────────────┘                       └──────────┬───────────────────────┘
                                                                    │
        ┌───────────────────────────────────────────────────────────▼───────────────────┐
        │  data/index.db (SQLite): documents · chunks · FTS5 · vec0 · index_meta          │
        │  data/practice.db · conversations.db · maps.db · understanding.db (memoria app) │
        └───────────────────────────────────────────────────────────▲───────────────────┘
                                                                    │ indexación incremental
        ┌───────────────────────────────────────────────────────────┴───────────────────┐
        │  Ingesta: descubrimiento · parser (markdown-it-py) · chunker estructural       │
        │  Embeddings BGE-M3 (GPU fp32 al indexar, CPU al consultar)                      │
        └───────────────────────────────────────────────────────────▲───────────────────┘
                                                                    │
                                            Biblioteca de Markdown (solo lectura)
```

### 5.4 Pipeline de indexación

1. **Descubrimiento** de archivos (se excluyen `.obsidian/` y `.trash/`).
2. **Análisis** con `markdown-it-py` (tokens con número de línea) y `pyyaml` para el frontmatter.
3. **Troceado estructural**: una sección por fragmento; las secciones largas se parten por párrafos, las diminutas se fusionan y nunca se parten el código, las tablas, las fórmulas `$$…$$` ni los diagramas.
4. **Embeddings** BGE-M3 sobre `«ruta jerárquica» + texto`, normalizados; en GPU en fp32 al indexar (vectores idénticos a los de CPU), en CPU al consultar.
5. **Almacenamiento** en un único archivo SQLite con una tabla `vec0` (búsqueda exacta) y una tabla FTS5 de contenido externo mantenida por *triggers*.
6. **Incrementalidad** por hash SHA-256 con una transacción por documento; `--full` reconstruye en un archivo temporal y lo intercambia solo si termina bien. `index_meta` registra todo lo que cambia fragmentos o vectores y el índice se niega a mezclar versiones.

### 5.5 Pipeline de recuperación y generación

```
consulta ─► [plan de conversación (LLM, opcional)] ─► densa (CPU) ┐
                                                                   ├─ RRF (k=60) ─► omitir índices
            BM25 (FTS5, peso del encabezado 0.5) ─────────────────┘    de contenidos ─► [reranker GPU]
                                                                                              │
                         ┌────────────────────────────────────────────────────────────────────┘
                         ▼
        ensamblado de contexto (≤ 6–8 fragmentos numerados, presupuesto en tokens del LLM)
                         ▼
        LLM → texto con citas [n] → validación determinista (citas válidas, marcadores, abstención)
```

La validación de citas (regla 2) es posterior y no negociable: una cita a un fragmento inexistente es un error; una respuesta sin citas válidas a una pregunta no ausente se marca como «no fundamentada»; la abstención es una frase fija («No encuentro esto en tu biblioteca.») que el sistema detecta.

### 5.6 Presupuesto de memoria por componente

| Componente | Dispositivo | Memoria | Nota |
|---|---|---|---|
| BGE-M3 (consultas) | CPU | ≈2.3 GB RAM (fp32) | Siempre en CPU |
| BGE-M3 (indexación por lotes) | GPU | pico 2.71 GiB de VRAM (fp32) | Solo sin servidor LLM |
| `bge-reranker-v2-m3` | GPU o CPU | ≈2.3 GB (fp32) | Se omite si la GPU está ocupada |
| Qwen3-4B Q4_K_M, contexto 8k, KV q8_0 | GPU | 3.4 GiB totales, 1.2 GiB de margen | Medido, 100 % en GPU |
| Gemma 4 26B-A4B (QAT, `mmap`) | GPU + RAM | 5.1 GB en VRAM y ≈8 GB de memoria privada | Solo modo Razonar |

La regla de prioridad es que **el LLM manda sobre la GPU**: la indexación por lotes y el *reranker* usan la GPU mientras no hay servidor LLM, y ceden (caen a CPU o se omiten) cuando lo hay (ADR-014).

### 5.7 Garantía de operación sin red

El ADR-002 se implementa en cinco capas: (1) variables de entorno de modo sin conexión fijadas antes de importar cualquier biblioteca de ML; (2) carga de modelos por ruta local y verificación de la revisión fijada; (3) un cliente LLM restringido a direcciones de bucle local, sin resolución de DNS; (4) pruebas con sockets bloqueados que fallan ante cualquier conexión; (5) `biblio doctor`, que verifica el entorno, los modelos y la configuración. La interfaz web aplica además defensas propias (enlace solo a bucle local, verificación de la cabecera `Host`, POST solo en JSON y sin cabeceras CORS) y se prueba que ningún activo estático referencie una URL externa.

### 5.8 Interfaces

Una interfaz de línea de comandos (`typer` + `rich`) y una interfaz web local servida con la biblioteca estándar de Python (ADR-016), con páginas de Conversar, Guías, Practicar, Mapas y Biblioteca, y un lector integrado que renderiza las notas desde los tokens del parser (incluidas fórmulas con KaTeX incluido localmente, ADR-031). Un modo LAN opcional, desactivado por defecto, permite usar el sistema desde teléfonos emparejados de la red doméstica (ADR-035).

---

## 6. Decisiones de ingeniería

Esta sección resume las decisiones más relevantes. Cada subsección indica el ADR de origen, las alternativas consideradas, la decisión y la evidencia. El registro completo está en el Apéndice A.

### 6.1 Fundamentos: conocimiento en documentos, operación sin red (ADR-001, 002)

Se eligió RAG sobre entrenamiento y ajuste fino (sección 2.2) y se definió la independencia como *operativa* (sección 2.3). Sus consecuencias estructurales: un único comando con red (`models pull`), modelos fijados por hash de revisión, licencias revisadas antes de fijar un modelo (BGE-M3 con licencia MIT; el *reranker* y los modelos de lenguaje con Apache-2.0) y pruebas con red bloqueada.

### 6.2 Ingesta e índice (ADR-003, 004, 005, 012)

**Troceado.** Se descartó el troceado por tamaño fijo en favor de secciones con ruta jerárquica. En la biblioteca inicial, 2 252 archivos produjeron 61.8 k fragmentos con una distribución de longitud de 81 / 258 / 646 tokens (percentiles 5 / 50 / 95) y cero violaciones de invariantes (cobertura del texto sin pérdidas ni duplicados, y unidades atómicas intactas).

**Embeddings.** BGE-M3 frente a E5 multilingüe y a modelos solo en inglés (descartados por el idioma). Solo vectores densos en esta etapa.

**Almacenamiento.** Un único archivo SQLite con `sqlite-vec` y FTS5, con búsqueda *exacta* (sin ANN): a esta escala la búsqueda exacta cuesta milisegundos y elimina el error de aproximación; un solo archivo hace trivial el respaldo. El umbral para reconsiderarlo se fijó en ≈200 k fragmentos; el índice actual supera 690 k, de modo que la decisión debe reevaluarse (sección 10).

**Dónde embeber (ADR-012).** La medición con 128 fragmentos reales mostró una diferencia decisiva:

| Dispositivo | tokens/s | Biblioteca inicial completa | Pico de VRAM | Vectores |
|---|---|---|---|---|
| CPU fp32 | 516 | ≈11.5 h | — | referencia |
| **GPU fp32** | 10 299 | **≈35 min** | 2.71 GiB | idénticos (coseno 1.000000) |
| GPU fp16 | 35 195 | ≈10 min | 1.38 GiB | coseno ≥ 0.99976 |

Se eligió **GPU fp32 solo para indexar** (sin servidor LLM) y CPU para las consultas: como los vectores son idénticos, un índice construido en GPU se consulta desde CPU sin mezclar versiones. La indexación completa real tardó 2 300 s.

### 6.3 Recuperación (ADR-006, 009, 013, 014, 044, 045, 052)

**Primero un buscador sin LLM (ADR-009).** Si la recuperación es mala, ningún modelo lo compensa; si es buena, un modelo pequeño rinde por encima de su tamaño.

**Híbrida con RRF y *reranker* (ADR-006).** La recuperación densa falla en términos exactos y BM25 en paráfrasis; la fusión RRF no requiere calibrar puntajes. El *reranker* mejora de forma consistente (sección 7.1) pero domina la latencia en CPU: 33.5 s p95 con 40 candidatos y 9.2 s con 10, frente a 1.9 s en GPU.

**Qué hacer cuando la GPU está ocupada (ADR-013 → 014).** Se midió la alternativa de reordenar solo 10 candidatos en CPU: *recall@8* de 0.634 y MRR 0.565, frente a 0.612 y 0.428 de la recuperación híbrida simple, es decir, un primer resultado mejor pero 20 veces más lento (p95 8.4 s) y sin ganancia de recall dentro del ruido. Se decidió **omitir** el *reranker* cuando un servidor LLM ocupa la GPU y devolver la híbrida simple (p95 0.43 s).

**Notas grandes (ADR-044, 045).** Análisis de los fallos: lo que desplaza a la sección correcta es casi siempre *otra nota* sobre el mismo tema (137 casos), rara vez otra sección de la misma nota (5) o un índice (3). Se actuó sobre dos mecanismos que favorecen a las notas grandes: los 2 172 índices («Tabla de contenidos») y el peso del título en BM25. Se midieron varios pesos del encabezado (0.1, 0.25, 0.5, 1, 2, 4) y se eligió 0.5, el valor que mejora sin empeorar ningún tipo de pregunta. Además, «Solo ubicar» agrupa por nota para mostrar más notas distintas por búsqueda (5.9 → 8.0) sin perder la sección correcta a la vista.

**Búsqueda bilingüe (ADR-052, propuesta).** Cuatro sondeos cuya respuesta solo está en un capítulo en inglés, formulados en español, lo situaron en posiciones 5, 5, 10 y 3 (1, 1, 2 y 1 si se preguntaba en inglés). Traducir la pregunta y conservar los 6 mejores resultados de la consulta original llevó el *recall@8* de 0.722 a 1.000 sobre los 18 casos de libros, a costa de un punto en las 64 preguntas del usuario; por eso permanece como propuesta, apagada, hasta confirmarla con preguntas del usuario sobre libros.

### 6.4 Generación local (ADR-015, 017, 032, 038)

**Elección del LLM (ADR-015).** Candidatos verificados contra sus fichas y no contra la memoria de entrenamiento: Qwen3-4B-Instruct-2507 (Apache-2.0, sin modo de pensamiento), Qwen3.5-4B (con pesos de visión y razonamiento activo por defecto) y Gemma 3 4B (licencia con restricciones de uso). Se eligió Qwen3-4B en cuantización Q4_K_M con caché KV a 8 bits y contexto de 8 192 tokens, servido por Ollama bajo la interfaz sustituible `LLMClient`. Medido: caché 612 MiB, 3.4 GiB totales, 37 de 37 capas en GPU, lectura de prompt a 3 715 tok/s y generación a 90 tok/s.

**Map-reduce: probado y rechazado como predeterminado (ADR-017).** La hipótesis era que resumir cada guía por separado y sintetizar después ayudaría en preguntas que cruzan guías. Con el conjunto v3 (5 búsquedas puntuales, 6 de síntesis, 3 ausentes) y calificación humana sobre el texto completo de las citas:

| | Directa | Map-reduce |
|---|---|---|
| Fidelidad (afirmaciones respaldadas) | **0.87** | 0.58 |
| Respuestas totalmente fieles | **80 %** | 29 % |
| Cobertura en búsquedas puntuales | **0.60** | 0.13 |
| Abstención correcta en preguntas ausentes | **2 de 3** | 0 de 3 (alucinó) |
| Falsas abstenciones en síntesis | 2 de 6 | 0 de 6 |
| Latencia media | **3.1 s** | 13.3 s |

Map-reduce responde síntesis que la ruta directa abandona, pero sus citas a menudo no respaldan lo que afirma, deja de abstenerse y es cuatro veces más lento. Se conservó como opción manual con advertencia, no como predeterminado.

**Conversaciones con tema (ADR-038).** Una conversación de prueba falló de tres maneras (una guía ocupó 7 de los 8 fragmentos por la palabra «ciencia»; un seguimiento corto se buscó tal cual; una corrección se trató como pregunta nueva). La solución es una llamada breve al LLM que lee el tema, los últimos turnos y el mensaje, y devuelve su relación con el tema, la pregunta autónoma y hasta cuatro búsquedas. Una medición previa evitó un error: fusionar todas las búsquedas con igual peso, o limitar fragmentos por guía, redujo el *recall@8* de síntesis de 0.54 a 0.43–0.48; por eso los seis mejores resultados de la propia pregunta conservan su lugar y las demás búsquedas solo completan dos.

### 6.5 Memoria y hardware (ADR-008, 011–015, 057)

La secuencia ADR-008 → 012 → 013 → 014 muestra el método: cada decisión se sustituyó cuando una medición contradijo un supuesto (el tiempo de indexación en CPU, la latencia del *reranker* en CPU, la ganancia del recurso de contingencia). Principios: se calcula con la VRAM *libre* medida; el LLM tiene prioridad en la GPU; todo modelo nuevo pasa por el cálculo de pesos + caché KV + buffers, con un margen mínimo del 10 %, y se verifica empíricamente con un prompt de tamaño máximo.

### 6.6 Funciones de estudio (ADR-020, 022, 025, 027, 029, 030, 047, 051)

**Cuestionarios (ADR-020).** Cada pregunta se genera desde una sección real y la muestra como fuente; las de opción múltiple y verdadero/falso las califica el código; las abiertas las califica el usuario o, de forma experimental, el modelo, con una tasa de acuerdo registrada que avisa si baja de un umbral.

**Repetición espaciada (ADR-025).** Se eligió FSRS frente a SM-2 y cajas de Leitner. El estado de memoria de cada sección *no se almacena*: se recalcula repitiendo sus respuestas en orden, de modo que cambiar los parámetros reprograma todo sin migraciones y ninguna respuesta pasada queda fuera de sincronía.

**Explícamelo tú (ADR-027).** Compara la explicación del usuario con las ideas clave de las notas en dos pasos. Todo punto que el modelo reporta debe llevar una cita *literal* comprobada por código; un punto cuya cita no está en el fragmento se descarta. Esta comprobación existe porque un modelo de 4B parafrasea las citas.

**Mapas conceptuales (ADR-022, 051).** El grafo de enlaces internos es demasiado disperso para sostener un mapa (565 de 61 753 fragmentos tienen enlaces), así que se extraen conceptos por fragmento con el LLM y el código conserva solo los que aparecen literalmente en su fragmento. El ADR-051 aborda una falla de calidad (un mapa de «API» tenía productos como ramas principales) en tres etapas medidas: selección de las secciones que *explican* el tema (precisión 0.44 → 0.70), elección de las ramas clave por el LLM (cobertura de conceptos clave 0.45 → 0.66) y aviso honesto cuando las notas no explican el tema en general. Un hallazgo metodológico: la métrica «el hijo está ligado a su rama por un fragmento» valía 1.00 por construcción, así que las aristas se juzgaron a mano (24–30 % correctas), una advertencia de que una métrica puede mejorar sin que mejore lo que importa.

**Detección de lagunas (ADR-029, 030, 047).** El coseno entre incrustaciones no separa «el mismo tema reformulado» (0.41–0.81) de «temas vecinos» (0.31–0.73), así que los embeddings solo preseleccionan candidatos y decide el LLM. Un subtema solo se acredita como «cubierto» con una cita de al menos `quote_run_words` palabras consecutivas presente en una sección leída.

### 6.7 Razonamiento y modelo mayor en RAM (ADR-056, 057, 058)

**Problema.** El modelo de 4B no relaciona varias guías de una pasada (ADR-017) y sus prompts le prohíben ir más allá de la evidencia, porque cuando se le permitió añadir cosas sin marcarlas inventó citas.

**Diseño (ADR-056).** Un modo «Razonar» con tres capas marcadas: «Según tus notas» (con citas), «Razonamiento» (pasos que abren con una frase copiada del fragmento que cita, comprobada por código) y «Conocimiento general (no está en tus notas)» (sin citas). La capa general solo se permite si una **puerta temática** indica que la biblioteca explica el tema de la pregunta. La puerta es un procedimiento en dos pasos: el LLM nombra el campo de la pregunta de cuatro maneras (en español y en inglés) y, para cada nombre, se reordenan en CPU los candidatos con el *reranker* frente a una pregunta definicional; el campo está «explicado» si al menos 2 secciones superan 0.4. Si no, la respuesta es la frase fija de abstención, sin generar nada. Una medición manual previa fijó los umbrales: las mejores secciones de 21 campos cubiertos puntuaron ≥ 0.76 y las de 22 campos ausentes ≤ 0.15; la sensibilidad a la redacción es alta (≈0.06 para «optimización de carteras» frente a 0.82 para «teoría de portafolios»), por lo que el LLM propone varios nombres.

**Modelo y memoria (ADR-057).** Se investigó el modelo contra sus fichas y tablas públicas (no la memoria del autor del asistente): un 4B con pensamiento no aporta conocimiento nuevo; gpt-oss-20b se descartó por tasas de alucinación publicadas muy altas en preguntas de conocimiento; se eligió **Gemma 4 26B-A4B** (Apache-2.0, 25.2 B de parámetros con 3.8 B activos por *token*, MoE) por su menor tasa de alucinación en resúmenes anclados entre los candidatos (5.2 % en la tabla de Vectara, frente a 5.7 % del Qwen3-4B), su conocimiento y su tamaño (16 GB). La integración reveló tres trampas de memoria del *runtime*, resueltas una a una:

| Problema (Ollama 0.40, Windows) | Medición | Solución |
|---|---|---|
| Copia del modelo completo a memoria privada | 18 GB al cargar, 27 GB con carga, ≈9 GB de intercambio en la evaluación | Modelo derivado con pesos mapeados (`use_mmap`): 8 GB |
| El caché de prompts de llama-server (8 GiB por defecto) crece en RAM | +0.3–0.4 GB por petición (8.0 → 9.7 GB en 5) | `LLAMA_ARG_CACHE_RAM=0`: 8.04 GB constantes, sin pérdida de velocidad |
| Gemma «piensa» por defecto en la API compatible | Respuestas vacías o lentas | `reasoning_effort: "none"` (`think: false` no tiene efecto) |

Además, un modelo a la vez (`OLLAMA_MAX_LOADED_MODELS=1`) y el *reranker* de la puerta se libera antes de la respuesta.

**«Explícamelo tú» en el modo razonado (ADR-058).** Con el mismo modelo y la misma puerta: un tema que las notas no explican se rechaza y cada afirmación «sin respaldo» recibe un veredicto marcado, de conocimiento general, correcta / incorrecta / dudosa; la nota propuesta sigue calculándose solo con lo que dicen las notas.

**Resúmenes a demanda (ADR-056, etapa E).** La biblioteca (≈191 M de tokens al medir) hacía inviable resumir todo por adelantado (≈11 h con el 4B antes de escribir nada, ≈5 días con el modelo mayor), de modo que los resúmenes de capítulos y de libros se generan **a demanda y se guardan** por hash del texto, con ideas clave cuya cita se comprueba literalmente, y se muestran en una pestaña de la biblioteca.

### 6.8 Interfaz y operación

Una interfaz web con servidor de biblioteca estándar frente a FastAPI o a Gradio/Streamlit (ADR-016): se evitaron dependencias que envían estadísticas de uso o cargan recursos externos, incompatibles con la regla 1. Otras decisiones: reindexado por sondeo de metadatos (30 s) en lugar de eventos del sistema de archivos (ADR-018); lanzador de un solo paso que solo descarga lo que falta y tras preguntar (ADR-021); botón «Reset» que mueve la memoria de la aplicación a una copia de respaldo y se niega a tocar la biblioteca (ADR-040); interfaz en inglés con el texto en español como clave y un test que falla si falta una traducción (ADR-046); servidor de imágenes de las notas con cinco comprobaciones de seguridad (ADR-048); estantería de libros y organización de notas por carpeta (ADR-053 a 055).

El ADR-059 documenta un rediseño de Conversar motivado por una medición sobre una captura del usuario: en una ventana de ≈1 480 × 705 px la respuesta ocupaba el 18 % de la altura (118 px) porque el cuadro de escritura ocupaba 235 px y la cabecera 100 px. Se dibujó una propuesta en un lienzo, la persona usuaria la aprobó y se implementó: cuadro de escritura de 110 px, cabecera de una línea con menú, fuentes siempre visibles junto a la respuesta y un historial agrupado por fecha; con ello la respuesta pasó a ver el 57 % (404 px). Se verificó en navegador sin cabeza, en tema claro y oscuro y en tres anchos de ventana.

---

## 7. Evaluación experimental y resultados

Las cifras provienen de `eval/results/*.jsonl` y de los ADR citados. Los conjuntos de evaluación son pequeños: las diferencias de un punto equivalen a una o dos preguntas y deben leerse con esa cautela.

### 7.1 Recuperación

**Líneas base (64 preguntas puntuables, 5 guías; ADR-044 y registro de la fase 3).**

| Modo | *recall@8* de sección | Latencia p95 |
|---|---|---|
| Densa | 0.546 | 354 ms |
| BM25 | 0.648 | 67 ms |
| Híbrida (RRF) | 0.647 | 425 ms |
| Híbrida + *reranker* en GPU (40 candidatos) | **0.769** | 1.85 s |
| Híbrida + *reranker* en CPU (40 candidatos) | = (calidad) | 33.5 s |

El *reranker* en GPU aporta +0.122 de *recall@8*, +0.129 de MRR y +0.138 de nDCG@8 sobre la híbrida, con ganancias por tipo (término exacto +0.238, síntesis +0.183, búsqueda puntual +0.030); neto: nueve preguntas encontradas y tres perdidas.

**Heurísticas para notas grandes (ADR-044).** Sobre la híbrida simple: R@1 0.255 → 0.286, R@8 0.647 → 0.663, MRR 0.458 → 0.481, término exacto R@1 0.048 → 0.095; 12 preguntas mejoran y ninguna empeora. Con el *reranker* el resultado no cambia (R@8 0.769).

**Agrupar por nota (ADR-045).** Notas distintas en pantalla 5.9 → 8.0; sección correcta a la vista 0.688 → 0.719 (con las dos siguientes secciones de cada nota visibles; con solo la mejor, 0.672).

**Búsqueda bilingüe (ADR-052; propuesta).**

| | Preguntas sobre libros (18) | Conjunto del usuario (64) |
|---|---|---|
| Sin traducir | 0.722 | 0.667 |
| Traducida, conservando los 6 primeros | **1.000** | 0.651 |
| Fusión por igual (RRF) | 1.000 | 0.604 |

### 7.2 Generación (conjunto v3; ADR-017)

Véase la tabla de la sección 6.4. La ruta directa alcanza fidelidad 0.87, 80 % de respuestas totalmente fieles y 3.1 s de latencia media; en las búsquedas puntuales obtuvo fidelidad 1.00 (5 de 5 totalmente fieles) y cobertura 0.60. Los puntos débiles medidos son la síntesis entre varias guías (fidelidad 0.53 con la primera calificación) y dos abstenciones falsas, lo que motivó el diseño de conversaciones con plan (ADR-038) y, más tarde, el modo de razonamiento.

### 7.3 Mapas conceptuales

| Etapa (ADR-051) | Medida | Antes → después |
|---|---|---|
| Selección de secciones (precisión de las 10 leídas) | 8 temas, 176 secciones etiquetadas (por el asistente) | 0.44 → 0.70 |
| Selección (contaminación) | ídem | 0.21 → 0.08 |
| Ramas clave elegidas por el LLM (cobertura de conceptos clave) | 89 conceptos clave de 8 temas | 0.45 → 0.66 |
| Precisión de las ramas | ídem | 0.43 → 0.56 |

**Réplica con el modelo mayor (30 oct.).** Con las mismas extracciones del 4B para los 8 temas, la organización por heurística, por el 4B y por Gemma 4 dio:

| Organiza | Cobertura de conceptos clave | Ramas que son clave | Tiempo de organizar |
|---|---|---|---|
| Heurística | 0.40 | 0.38 | 0 s |
| Qwen3-4B | 0.51 | 0.57 | 2 s |
| Gemma 4 26B-A4B | 0.54 | 0.63 | 19 s |

La mejora de Gemma (≈3 puntos de cobertura y 6 de precisión, una rama más por mapa) está dentro del ruido de 8 temas (un concepto clave mueve ≈8 puntos) y cuesta más de lo que muestra la tabla por el cambio de modelo cargado; **no se adoptó**. El límite está en la extracción, no en la organización.

### 7.4 Razonamiento (27 preguntas; ADR-056 a 058)

El conjunto tiene siete tipos (inferencia, relación, aplicación, comprensión, conocimiento general, frontera y ausente); 5 preguntas deben provocar abstención y 2 deben usar la capa de conocimiento general.

| | Tutor 4B | «Razonar» 4B (v2) | «Razonar» Gemma 4 (v3) |
|---|---|---|---|
| Puerta temática correcta (22 abiertas + 5 cerradas) | — | 27 de 27 | 27 de 27 |
| Abstención en frontera y ausentes | 5 de 5 | 5 de 5 | 5 de 5 |
| Abstenciones falsas | 2 de 22 | 0 de 22 | 0 de 22 |
| Respuestas con citas válidas | 20 de 22 | 21 de 22 | 21 de 22 |
| Citas inexistentes | 0 | 2 | 1 |
| Citas del razonamiento copiadas literalmente | — | 46 % (52/114) | **80 %** (71/89) |
| Usa la capa general | — | 12 veces, con errores | 4 veces, correctas |
| Segundos por respuesta (mediana, puerta abierta) | 17 | 35 | 51 |
| Intercambio de memoria añadido durante la corrida | — | — | 0 GB (≥ 5.1 GB libres) |

La primera versión del modo (v1) se abstuvo correctamente solo en 2 de 5 casos porque, pidiéndole que se abstuviera con sus propias palabras, explicó en sus capas que no había nada; la v2 devuelve la frase fija sin generar. Ese defecto es un ejemplo de que la abstención debe ser una acción del sistema y no una instrucción al modelo. Los errores de la capa general del 4B (por ejemplo, describir mal el filtro de información como generalización a distribuciones no gaussianas) desaparecen con el modelo mayor, que lo describe como la forma del filtro de Kalman en términos de la matriz de información (inversa de la covarianza).

**Importante:** la fidelidad y la cobertura calificadas por una persona sobre las hojas de revisión están pendientes; las cifras anteriores son las que se miden automáticamente. Por eso los ADR-056 a 058 siguen como *propuestos*.

### 7.5 «Explícamelo tú» (12 explicaciones, borrador)

| | Qwen3-4B (ADR-027) | Modelo razonado (ADR-058) |
|---|---|---|
| Revisiones que fallaron por completo | 4 de 12 | **0 de 12** |
| Afirmaciones marcadas que no se esperaban | 3 de 13 | **1 de 10** |
| Afirmaciones con respaldo ausente: encontradas | 10 de 12 | 9 de 12 |
| Veredictos de conocimiento general (correcto / incorrecto / dudoso) | — | 6 / 0 / 0 de 6 |
| Tema que las notas no tratan | se comparaba con notas irrelevantes | **rechazado** |
| Ideas faltantes encontradas | 0–2 de 8 | 0–2 de 8 |
| Segundos por explicación (mediana) | 16 | 65 |

La *recuperación de ideas faltantes* es débil con ambos modelos. Parte del problema es de medición: con el umbral de similitud de 0.75, paráfrasis evidentes puntúan 0.63–0.72 con BGE-M3 y no se reconocen, y el borrador lista solo una o dos ideas faltantes por explicación. Por eso esa métrica no es concluyente.

### 7.6 Conversaciones (8 conversaciones, 22 turnos, borrador)

| | Planificador en Qwen3-4B | En Gemma 4 |
|---|---|---|
| Relación leída correctamente (sigue / corrige / nuevo) | 22 de 22 | 22 de 22 |
| Primeros mensajes que alcanzan una sección esperada | 8 de 8 | 8 de 8 |
| Seguimientos que la alcanzan | 9 de 13 | 8 de 13 |

La primera versión del borrador, con una sección esperada por turno, arrojaba 0.43 de *recall* global; al reconocer las notas dedicadas del corpus que el borrador había omitido (38 alternativas verificadas) subió a 0.81. Es un ejemplo del sesgo del conjunto: el error estaba en la referencia y no en el sistema. Reordenar en CPU los candidatos del turno contra su pregunta autónoma ganó, en reproducción sin LLM, un seguimiento de 13 para el 4B (y ninguno para Gemma) a costa de 13 s por mensaje; no se adoptó.

### 7.7 Resumen frente a las preguntas de investigación

| PI | Respuesta apoyada por la evidencia |
|---|---|
| PI1 | Sí: el sistema opera sin red con ≈697 000 fragmentos en el equipo de referencia; la indexación por lotes cabe en 2.71 GiB de VRAM y las consultas corren en CPU en milisegundos. |
| PI2 | La híbrida con *reranker* en GPU da el mejor *recall@8* (0.769); sin GPU libre la mejor relación costo/beneficio es la híbrida simple con heurísticas de encabezado e índices (0.663). |
| PI3 | El 4B cuantizado es fiel en búsquedas puntuales (1.00) y suficiente en síntesis simple (0.87 global); sus límites (síntesis entre guías, citas parafraseadas) se mitigan con validación determinista y abstención, no con técnicas que reducen la fidelidad (map-reduce). |
| PI4 | Parcialmente respondida: el modelo mayor en RAM mejora las citas literales (46 → 80 %) y elimina errores de conocimiento general, a un costo de 1.5–3 veces más de latencia; la mejora de fidelidad y cobertura percibida requiere la calificación humana pendiente. |
| PI5 | El registro de decisiones permitió sustituir 4 decisiones al aparecer evidencia y evitar tres (map-reduce por defecto, *reranker* en CPU, fusión igualitaria en conversaciones) que un criterio *a priori* habría adoptado. |

---

## 8. Discusión

**8.1 Medir antes de adoptar cambia las decisiones.** Cuatro ideas plausibles se rechazaron o relegaron al medirlas: el *map-reduce* como predeterminado (fidelidad 0.58 frente a 0.87), el *reranker* en CPU como contingencia (20 veces más lento sin ganancia de *recall*), la fusión igualitaria de búsquedas en conversaciones (−6 a −11 puntos en síntesis) y el reordenamiento en CPU de los candidatos en seguimientos (+1 caso por +13 s). La regla 3 no frenó el desarrollo: fue lo que evitó acumular complejidad sin beneficio.

**8.2 La verificación debe ser del sistema, no del modelo.** Los modelos pequeños copian el marcador de ejemplo de un prompt (`[n]`; 70 en las notas de map-reduce y 5 respuestas en una corrida), parafrasean las citas (46 % de citas literales en el razonamiento con el 4B), inventan citas cuando se les permite añadir sin marcar y explican con sus palabras que «no hay nada» en lugar de usar la frase de abstención. Las respuestas del proyecto fueron comprobaciones deterministas: citas válidas, citas literales con pliegue de acentos y mayúsculas, abstención fija, umbrales sobre un puntaje de un modelo especializado. Un matiz importante: la comprobación literal garantiza que la *cita* está en el fragmento, no que el *resumen* que la acompaña sea correcto (se observó un caso con una cita correcta y un resumen invertido); es un riesgo residual declarado en el ADR-058.

**8.3 La memoria real no es la teórica.** El cálculo de pesos + caché KV + buffers fue necesario pero insuficiente: el comportamiento del *runtime* (copias a memoria privada, caché de prompts que crece sin límite, pensamiento activado por defecto) añadió hasta 19 GB inesperados. La lección de ingeniería es verificar con una carga realista y *observar el crecimiento por petición*, no solo el pico inicial.

**8.4 La evaluación es parte del sistema y puede fallar.** Se encontraron defectos en las propias herramientas de evaluación: una expresión regular que nunca interpretó las calificaciones por un carácter de control, una hoja de revisión que truncaba el texto citado a 400 caracteres de modo que la fidelidad no era verificable, y un conjunto de referencia sesgado (0.43 → 0.81 al ampliar las alternativas). Cada defecto se convirtió en prueba o en regla (las hojas muestran ahora el texto completo, hay una fixture por error de parseo y un test que impide marcadores sin número).

**8.5 La calidad del recuperador acota todo lo demás.** La puerta temática, los mapas, los cuestionarios, el estudio y el razonamiento dependen de qué secciones entran. Los avances más sólidos (la selección por reranker de los mapas, 0.44 → 0.70; el reranker de la búsqueda, +0.12) fueron de recuperación; los modelos más grandes ayudaron en la *redacción* y el razonamiento, no en encontrar.

**8.6 Sobre el desarrollo con agentes de IA.** La combinación de contrato, procedimientos, registro de decisiones y humano en las bifurcaciones permitió un ritmo alto (el núcleo del sistema, de la fase 0 a la 6, se registra en un día de trabajo) sin perder trazabilidad: cada decisión tiene su ADR y cada cifra su archivo. A la vez, el modo de falla característico del agente fue *producir evidencia propia de baja calidad con convicción* (por ejemplo, conjuntos de evaluación redactados por él con sesgo hacia lo fácil), y la defensa fue rotular su autoría, exigir la aprobación del usuario y medir el sesgo cuando se detectó.

---

## 9. Limitaciones y amenazas a la validez

**Validez interna.**

- Los conjuntos de evaluación son pequeños (48–70 preguntas de recuperación, 14 de generación, 27 de razonamiento, 8 conversaciones) y de un solo usuario; una diferencia de un punto es una o dos preguntas.
- Los modelos corren a temperatura 0.2 y el mismo mapa varía entre ejecuciones; no se repitieron las corridas para estimar varianza.
- Parte de los conjuntos auxiliares fue **redactada por el asistente de IA** a pedido del usuario (Apéndice D): 18 preguntas sobre libros, los 89 conceptos clave de los mapas, y los borradores de razonamiento, Feynman y conversaciones. Está rotulado, y el de razonamiento fue aprobado sin cambios por el usuario, pero el sesgo hacia lo que el sistema resuelve bien es posible.
- La calificación humana de fidelidad y cobertura del modo de razonamiento está pendiente.

**Validez de constructo.** *Recall@k* de sección mide si se encontró *la* sección esperada, no si la respuesta era buena: varias guías pueden responder la misma pregunta (por eso existen las alternativas). La fidelidad por afirmaciones es sensible a cómo se segmentan estas. La métrica «hijo atado a su rama» fue 1.00 por construcción.

**Validez externa.** Un corpus, un usuario, un equipo, un idioma principal (español con terminología en inglés). Los resultados no se generalizan a otros dominios ni a corpus con estructura de encabezados pobre. El umbral de 200 000 fragmentos fijado en el ADR-005 para reconsiderar la búsqueda exacta fue superado (≈697 000), y no se midió el efecto.

**Validez de conclusión.** Las comparaciones entre modelos (4B vs. 26B) confunden tamaño con familia, cuantización y *runtime*; no se aíslan sus efectos. La latencia depende del estado de memoria del equipo (en una corrida con intercambio fue de 105–223 s por respuesta).

**Limitaciones del sistema.** Techo de calidad de los modelos locales; latencia alta del modo de razonamiento (~1 min); un solo modelo cargado a la vez; la capa de conocimiento general puede contener errores que nadie verifica (aunque va marcada); la función de resúmenes no se ha medido con el modelo real; la interfaz no se ha evaluado con estudios de usuarios.

---

## 10. Trabajo futuro

1. **Calificación humana** de fidelidad y cobertura de las tres configuraciones (tutor, razonar con 4B y con Gemma) y aceptación o rechazo de los ADR-056 a 058.
2. **Mejorar la métrica de ideas faltantes** de «Explícamelo tú» (umbral, listas completas) y ampliar los conjuntos de conversaciones con casos reales fallidos del usuario (por ejemplo, términos ausentes en la biblioteca).
3. **Segunda ronda de búsqueda** en conversaciones y resolución de referentes en seguimientos cortos.
4. **Probar el modelo mayor en la extracción de los mapas**, donde está el límite medido, con una estimación de costo (≈5 minutos por mapa).
5. **Reevaluar la búsqueda exacta frente a ANN** y la arquitectura de un solo archivo ahora que el índice supera el umbral del ADR-005.
6. **Medir los resúmenes** con modelos reales: proporción de ideas que sobreviven a la comprobación de citas, tiempo y juicio de calidad.
7. **Búsqueda bilingüe** (ADR-052): decidir con preguntas del usuario sobre libros.
8. **Capa de estilo por ajuste fino** (LoRA) para la redacción, solo si se demuestra un problema de formato y no de contenido.
9. **Replicar** el método en otro corpus y otro equipo para estimar la generalidad.

---

## 11. Conclusiones

Se construyó un sistema RAG local, verificable y medido sobre una biblioteca personal que creció un orden de magnitud durante el proyecto. Dentro de las restricciones (sin red, citas obligatorias, menos de 8 GB para modelos en un equipo de consumo), se logró una recuperación híbrida con *reranker* de *recall@8* 0.769, un tutor local con fidelidad 0.87 en respuestas directas, y un modo de razonamiento con puerta temática y capas marcadas que, con un modelo de mezcla de expertos ejecutado desde RAM, mantiene la abstención correcta y eleva de 46 % a 80 % las citas literales del razonamiento.

Las contribuciones de ingeniería más reutilizables son de método: (1) un registro de 59 decisiones con evidencia, en el que cuatro decisiones se sustituyeron al aparecer mediciones; (2) la separación entre generación probabilística y verificación determinista; (3) el tratamiento de la memoria como presupuesto medido por componente, incluida la lección de que el *runtime* puede añadir decenas de gigabytes que ningún cálculo previo predice; y (4) un esquema de gobernanza (contrato, procedimientos, ADR y humano en las bifurcaciones) que hizo seguro el desarrollo con un agente de IA.

Las limitaciones son claras: conjuntos de evaluación pequeños y parcialmente redactados con asistencia, calificación humana pendiente del modo de razonamiento, y una sola máquina y un solo usuario. El sistema es útil hoy para su dueño y es, sobre todo, un caso documentado de cómo hacer ingeniería de IA con restricciones reales y evidencia.

---

## Referencias

Cepeda, N. J., Pashler, H., Vul, E., Wixted, J. T., y Rohrer, D. (2006). Distributed practice in verbal recall tasks: A review and quantitative synthesis. *Psychological Bulletin, 132*(3), 354–380.

Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., y Liu, Z. (2024). *BGE M3-Embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation* (arXiv:2402.03216).

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., … Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. En *Proceedings of ACL 2020*.

Cormack, G. V., Clarke, C. L. A., y Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. En *Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval* (pp. 758–759).

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., y Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. En *Advances in Neural Information Processing Systems 35*.

Dettmers, T., y Zettlemoyer, L. (2023). The case for 4-bit precision: k-bit inference scaling laws. En *Proceedings of the 40th International Conference on Machine Learning*.

Dunlosky, J., Rawson, K. A., Marsh, E. J., Nathan, M. J., y Willingham, D. T. (2013). Improving students' learning with effective learning techniques. *Psychological Science in the Public Interest, 14*(1), 4–58.

Es, S., James, J., Espinosa-Anke, L., y Schockaert, S. (2023). *RAGAS: Automated evaluation of retrieval augmented generation* (arXiv:2309.15217).

Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., … Wang, H. (2023). *Retrieval-augmented generation for large language models: A survey* (arXiv:2312.10997).

Hevner, A. R., March, S. T., Park, J., y Ram, S. (2004). Design science in information systems research. *MIS Quarterly, 28*(1), 75–105.

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., … Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys, 55*(12), 1–38.

Järvelin, K., y Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems, 20*(4), 422–446.

Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., … Yih, W. (2020). Dense passage retrieval for open-domain question answering. En *Proceedings of EMNLP 2020*.

Kudo, T., y Richardson, J. (2018). SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. En *Proceedings of EMNLP 2018: System Demonstrations*.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., … Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. En *Advances in Neural Information Processing Systems 33*.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., y Liang, P. (2024). Lost in the middle: How language models use long contexts. *Transactions of the Association for Computational Linguistics, 12*, 157–173.

Malkov, Y. A., y Yashunin, D. A. (2020). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 42*(4), 824–836.

Nogueira, R., y Cho, K. (2019). *Passage re-ranking with BERT* (arXiv:1901.04085).

Novak, J. D., y Cañas, A. J. (2008). *The theory underlying concept maps and how to construct and use them* (Technical Report IHMC CmapTools 2006-01 Rev 01-2008). Florida Institute for Human and Machine Cognition.

Reimers, N., y Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. En *Proceedings of EMNLP-IJCNLP 2019*.

Robertson, S., y Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333–389.

Roediger, H. L., y Karpicke, J. D. (2006). Test-enhanced learning: Taking memory tests improves long-term retention. *Psychological Science, 17*(3), 249–255.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., y Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. En *International Conference on Learning Representations*.

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., y Gurevych, I. (2021). BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. En *NeurIPS Datasets and Benchmarks Track*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … Polosukhin, I. (2017). Attention is all you need. En *Advances in Neural Information Processing Systems 30*.

Voorhees, E. M. (1999). The TREC-8 question answering track report. En *Proceedings of the Eighth Text REtrieval Conference (TREC-8)*.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., … Zhou, D. (2022). Chain-of-thought prompting elicits reasoning in large language models. En *Advances in Neural Information Processing Systems 35*.

Ye, J., Su, J., y Cao, Y. (2022). A stochastic shortest path algorithm for optimizing spaced repetition scheduling. En *Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*.

*Fichas de modelo y tablas públicas consultadas (octubre de 2026):* fichas de Qwen3-4B-Instruct-2507, BGE-M3, `bge-reranker-v2-m3` y Gemma 4 26B-A4B en Hugging Face; biblioteca de modelos de Ollama; tabla de alucinación de Vectara (actualización del 22 de octubre de 2026). Los informes técnicos de las familias Qwen3 (Yang et al., 2025, arXiv:2505.09388) y Gemma 3 (Gemma Team, 2025, arXiv:2503.19786) describen los modelos precedentes.

> **Nota para la revisión:** las referencias bibliográficas deben verificarse contra sus fuentes originales (año, páginas, DOI) antes de la entrega. Las cifras del proyecto provienen del repositorio y de las mediciones citadas, no de la literatura.

---

## Apéndice A. Registro de decisiones de arquitectura

Títulos y estados tal como figuran en `DECISIONS.md` (en inglés, el idioma de la documentación del proyecto).

| ADR | Título (en inglés, como en `DECISIONS.md`) | Estado |
|---|---|---|
| 001 | RAG instead of training or fine-tuning as the knowledge substrate | Accepted |
| 002 | Full operational independence; origin dependency accepted | Accepted |
| 003 | Structural chunking by headings with a hierarchical path | Accepted |
| 004 | BGE-M3 as the embedding model, versioned in the index | Accepted |
| 005 | SQLite + sqlite-vec + FTS5 with exact search | Accepted |
| 006 | Hybrid retrieval with RRF and a cross-encoder reranker | Accepted |
| 007 | LLM behind a local OpenAI-compatible server | Superseded by ADR-015 |
| 008 | Memory budget < 8 GB and an assumed quality ceiling | Superseded by ADR-012 |
| 009 | Search engine without an LLM first; evaluation before optimization | Accepted |
| 010 | Python 3.11+, uv, typer | Accepted |
| 011 | Reference hardware: Windows 11, Ryzen 7 5700X, 32 GB RAM, RTX 5060 8 GB (CUDA) | Accepted |
| 012 | Memory budget < 8 GB; GPU for the LLM at query time and for batch embedding while no LLM runs | Superseded by ADR-013 |
| 013 | Memory budget < 8 GB; the GPU serves whichever workload is active, the LLM has priority | Superseded by ADR-014 |
| 014 | As ADR-013, but `search` skips the reranker (plain hybrid) when the GPU is not free | Accepted |
| 015 | Qwen3-4B-Instruct-2507 (Q4_K_M, KV q8_0, 8k context) on Ollama, behind `LLMClient` | Accepted |
| 016 | Local web UI for `search` and `ask`: stdlib HTTP server + own HTML/JS, loopback only | Accepted; loopback-only relaxed by ADR-035 when LAN mode is on |
| 017 | Map-reduce is a manual opt-in for `ask`, off by default | Accepted |
| 018 | Automatic reindexing by polling file metadata (stdlib), GPU if free else CPU | Accepted |
| 019 | Web UI v2: Conversar, Guías and Practicar pages on the stdlib server | Accepted; independent turns superseded by ADR-038 |
| 020 | Practice quizzes generated from library sections, with local history | Accepted |
| 021 | One-step launcher: setup steps that download only when missing, after asking | Accepted; «asked once» superseded by ADR-039 |
| 022 | Concept maps by LLM extraction per fragment, checked against the text | Accepted |
| 023 | Conversation history in a local SQLite file on the server | Accepted; «not memory» superseded by ADR-038 |
| 024 | Concept-map history in a local SQLite file on the server, as ADR-023 | Accepted |
| 025 | Spaced repetition with FSRS over practised sections, 20 reviews a day | Accepted |
| 026 | Mastery map: concepts coloured by how well their sections are remembered today | Accepted |
| 027 | «Explícamelo tú»: explanations checked idea by idea against retrieved notes | Accepted |
| 028 | Study plan per folder: an order proposed by the LLM, states from reading marks and practice | Accepted |
| 029 | Gap detector: abstentions and thin maps become pending topics, offered in Guías | Partly superseded by ADR-030 (merging, thin map) |
| 030 | Gap detector: thin maps include an absent topic; merging confirmed by the LLM | Accepted · Supersedes part of ADR-029 |
| 031 | Built-in reader: notes rendered from parser tokens, KaTeX vendored, passage actions in a side panel | Accepted; «Access» superseded by ADR-033 |
| 032 | Tutor levels (principiante, intermedio, avanzado) as prompt blocks chosen per message | Accepted |
| 033 | A «Biblioteca» page in the menu to browse and open notes in the reader | Accepted · Supersedes part of ADR-031 |
| 034 | Dictionary of a topic or guide: concepts, functions and formulas checked against the notes; flashcards | Accepted |
| 035 | LAN mode: the web UI on the home network for paired phones and tablets, off by default | Accepted · Supersedes part of ADR-016 |
| 036 | Four more diagram types in Mapas, drawn by our own renderer, with Mermaid export | Accepted; «Flujo» combination and Novak layout superseded by ADR-037 |
| 037 | Diagrams: one version per note for flows, the Novak map read left to right | Accepted · Supersedes part of ADR-036 |
| 038 | Conversations that keep their topic, and answers that relate guides | Accepted · Supersedes part of ADR-019, ADR-023 |
| 039 | Changing the library's location from the app or the terminal, reusing the index when it is the same library | Accepted · Supersedes part of ADR-021 |
| 040 | «Reset»: start the app over without its memory, never touching the library | Accepted |
| 041 | OPB visual style with vendored fonts (Fraunces, Plus Jakarta Sans) | Accepted |
| 042 | Notes in English: names and quotes copied as written, a Spanish name beside them | Accepted |
| 043 | The LLM is switched off and on from Ajustes; the app stops only an Ollama it started | Accepted |
| 044 | Tables of contents are not results; BM25 weighs the heading path at 0.5 | Accepted |
| 045 | «Solo ubicar» grouped by note; a search inside a note in the reader | Accepted |
| 046 | An English interface: the Spanish text is the key, a dictionary per language, generated content stays Spanish | Accepted |
| 047 | «Gap Spotter»: missing subtopics from two marked sources, judged against the notes; guides suggested as cards | Accepted |
| 048 | The reader shows the images a note references, served from the library only | Accepted |
| 049 | The reader's layout: a slim reading bar, the note's index beside it, the selection panel waiting | Accepted |
| 050 | The reader's index shows how each section is remembered; practice placed by its heading's title | Accepted |
| 051 | Maps and diagrams built from what explains the topic, not from what mentions it; measured stage by stage | Stages 1–3 implemented and measured for maps; diagrams pending |
| 052 | The chat also searches the question in English; the question's own top 6 keep their places | Proposed (off until measured on book questions) |
| 053 | Books on a shelf, chapters read in their book, and «Seguir leyendo» | Accepted (shelf's cards and area chips superseded by ADR-054) |
| 054 | The shelf gets a toolbar (area, status, sort, grid or list) and cards that say only what tells books apart | Accepted |
| 055 | «Mis notas» by area and topic: topic cards, a list grouped by folder, a page per topic | Accepted |
| 056 | A «Razonar» mode: answers in three marked layers, the model's own knowledge only on topics the library explains | Proposed |
| 057 | Memory budget: the reasoning model may use system RAM; the GPU stays first for the fast model | Proposed |
| 058 | «Explícamelo tú» on the «Razonar» model: topics the notes do not explain are refused, unsupported claims get a marked general verdict | Proposed |
| 059 | Conversar gives the answer the room: a compact composer, a one-line header, the sources beside it, a history by date | Accepted |

---

## Apéndice B. Pila tecnológica

| Capa | Elección | ADR |
|---|---|---|
| Lenguaje y entorno | Python ≥ 3.11 (3.12 con `uv`) | 010 |
| Análisis de Markdown | `markdown-it-py` (tokens con líneas) + `pyyaml` | 003 |
| Embeddings | `BAAI/bge-m3` (1 024 dim.); GPU fp32 al indexar, CPU al consultar | 004, 012 |
| Almacenamiento | SQLite + `sqlite-vec` + FTS5 en un archivo | 005 |
| Recuperación | Densa + BM25 → RRF → `bge-reranker-v2-m3` | 006, 014 |
| LLM del tutor | Qwen3-4B-Instruct-2507 Q4_K_M en Ollama (KV q8_0, contexto 8k) | 015 |
| LLM de razonamiento | Gemma 4 26B-A4B (QAT, `mmap`) en el mismo servidor | 057 |
| Repetición espaciada | FSRS-4.5 propio, sin dependencias | 025 |
| Interfaz | Servidor HTTP de la biblioteca estándar + HTML/JS/CSS propios | 016 |
| CLI | `typer` + `rich` | 010 |
| Calidad | `pytest` (+ `pytest-socket`), `mypy --strict`, `ruff` | 010 |

Dimensión: ≈75 módulos Python y ≈33 000 líneas en `src/` (incluidos prompts y la interfaz web estática), ≈9 500 líneas de pruebas, 83 plantillas de prompts en español, 59 ADR y 528 casos de prueba.

## Apéndice C. Reproducibilidad

```bash
uv sync                                   # dependencias
uv run biblio models pull                 # único comando con red: descarga los pesos fijados por hash
uv run biblio doctor                      # hardware, modelos, extensión sqlite-vec, modo sin conexión
uv run biblio index "<ruta-a-la-biblioteca>" [--full]
uv run biblio search "consulta"           # buscador, sin LLM
powershell scripts/serve-llm.ps1          # servidor LLM con los ajustes medidos (ADR-015, 057)
uv run biblio ask "pregunta" --level intermediate
uv run biblio eval --tag nombre           # recuperación; añade una línea a eval/results/runs.jsonl
uv run biblio ask-eval run --tag nombre   # generación, con hoja de revisión para calificar
uv run biblio ask-eval run --reasoning --reason --tag nombre   # modo Razonar (ADR-056)
uv run biblio eval-feynman --reason       # «Explícamelo tú» (ADR-058)
uv run biblio eval-chains                 # conversaciones (ADR-038)
uv run biblio serve                       # interfaz web local
uv run pytest && uv run ruff check . && uv run mypy src
```

Para reproducir las cifras hacen falta la biblioteca del autor (no publicada) y sus conjuntos de preguntas (`eval/*.jsonl`, excluidos del repositorio público); los archivos de ejemplo (`eval/*.example.jsonl`) muestran el formato.

## Apéndice D. Conjuntos de evaluación y su autoría

| Conjunto | Contenido | Autoría | Estado |
|---|---|---|---|
| `questions.v1` / `questions` (v2) | 48 / 70 preguntas (64 puntuables) de recuperación, 5 guías | Usuario (rutas completadas por el asistente) | Vigente |
| `questionsv3` | 14 preguntas de generación con puntos clave | Usuario | Calificado por el usuario |
| `questions_books` | 18 preguntas cuya respuesta está solo en libros | **Asistente**, a pedido del usuario; rotulado | Vigente; favorece temas solo presentes en libros |
| `maps_centrality` | Secciones etiquetadas (general / específica / incidental) de 8 temas | **Asistente** (campo `labeler` del archivo) | Vigente; no revisado por el usuario en este registro |
| `maps_key_concepts` | 89 conceptos clave de 8 temas, con alias | **Asistente**, a pedido del usuario | Vigente |
| `reasoning` | 27 preguntas de razonamiento (7 tipos) con puntos clave | **Asistente** (borrador), **aprobado sin cambios** por el usuario | Calificación humana pendiente |
| `feynman.draft` | 12 explicaciones con ideas faltantes y afirmaciones sin respaldo | **Asistente** | Borrador sin revisar |
| `chains.draft` | 8 conversaciones, 22 turnos, 38 alternativas | **Asistente** | Borrador sin revisar |

## Apéndice E. Cronología

| Fecha | Hito |
|---|---|
| 2026-10-05 | Inicialización: contrato, plan, registro de decisiones y *skills*; ADR-001 a 011 propuestos |
| 2026-10-17 | Fases 0 a 6: entorno sin red verificado, ingesta, índice completo (2 300 s en GPU), líneas base de recuperación, híbrida y *reranker*, ADR-012 a 015, tutor, redacción, evaluación v3, interfaz web |
| 2026-10-18 a 19 | Práctica, mapas, historial, FSRS, «Explícamelo tú», plan de estudio, detector de lagunas, lector, conversaciones con tema (ADR-017 a 038) |
| 2026-10-20 a 21 | Cambio de ubicación de la biblioteca, «Reset», estilo visual, notas en inglés, heurísticas de notas grandes, interfaz en inglés, mapas con selección medida (ADR-039 a 051) |
| 2026-10-26 a 27 | Búsqueda bilingüe propuesta, estantería de libros, organización de notas (ADR-052 a 055) |
| 2026-10-30 | Modo «Razonar», investigación y elección del modelo, trampas de memoria del *runtime*, «Explícamelo tú» razonado, conversaciones medidas (ADR-056 a 058) |
| 2026-10-31 | Resúmenes de capítulos y libros a demanda y pestaña de resúmenes |
| 2026-11-01 | Rediseño de Conversar (ADR-059) |

## Apéndice F. Glosario

- **ADR** (*Architecture Decision Record*): documento breve que registra una decisión, sus alternativas y su evidencia.
- **BM25:** función de ordenamiento léxico probabilístico.
- **Cross-encoder:** modelo que puntúa conjuntamente consulta y pasaje; más preciso y más lento que un codificador por separado.
- **FSRS:** algoritmo de repetición espaciada basado en estabilidad y dificultad por elemento.
- **KV (caché):** claves y valores de la atención guardados durante la generación; crece con el contexto.
- **MoE (mezcla de expertos):** arquitectura que activa solo algunos «expertos» por *token*.
- **`mmap`:** carga de un archivo en memoria mapeada sin copiarlo a memoria privada.
- **MRR / nDCG / recall@k:** métricas de recuperación.
- **RAG:** generación aumentada por recuperación.
- **RRF:** fusión por rangos recíprocos.
- **Puerta temática:** comprobación de que la biblioteca explica el tema de una pregunta antes de permitir conocimiento general.
- **Abstención:** respuesta fija que indica que la biblioteca no contiene lo preguntado.

## Apéndice G. Declaración sobre el uso de herramientas de IA

[Adaptar a la política de la institución.] El software descrito se desarrolló con un agente de programación basado en un modelo de lenguaje (Claude Code, de Anthropic), bajo dirección y supervisión humanas: la persona responsable definió los requisitos, tomó las decisiones en las bifurcaciones registradas en los ADR y redactó o aprobó los conjuntos de evaluación; el agente implementó el código, las pruebas y la documentación del repositorio, y redactó borradores de varios conjuntos auxiliares de evaluación, identificados en el Apéndice D. Las cifras de este reporte proceden de los archivos de resultados y de los ADR del repositorio. [Indicar aquí, si procede, si este reporte fue redactado, asistido o revisado con una herramienta de IA, y qué partes fueron verificadas por la persona responsable.]
