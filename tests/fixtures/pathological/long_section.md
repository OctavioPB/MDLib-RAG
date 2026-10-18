# Arquitectura

## Flujo de datos

El sistema lee los archivos Markdown de la biblioteca, los divide en fragmentos por encabezados y
guarda cada fragmento junto con su ruta jerárquica en el índice local.

```
+--------+     +---------+
| parser | --> | chunker |
+--------+     +---------+
                    |
                    v
               +--------+
               | índice |
               +--------+
```

El parser usa los tokens de markdown-it para conocer el rango de líneas exacto de cada bloque, lo que
permite citar la sección y las líneas precisas de donde sale cada respuesta.

El chunker nunca parte un bloque de código, una tabla ni una fórmula, porque un fragmento cortado a la
mitad pierde el sentido y confunde tanto al reranker como al modelo de lenguaje.

Las secciones muy cortas se fusionan con su vecina y las muy largas se dividen entre párrafos, siempre
heredando la misma ruta de encabezados para que la cita siga siendo correcta.
