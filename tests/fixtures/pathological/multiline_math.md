# Estadística

## Varianza

La varianza mide la dispersión de una variable aleatoria alrededor de su media y se define como la
esperanza del cuadrado de la desviación respecto a la media.

$$
\operatorname{Var}(X) = E[X^2] - E[X]^2

\text{con } E[X] = \sum_i x_i p_i
$$

Esta identidad es útil para calcular la varianza a mano, pero numéricamente puede sufrir cancelación
catastrófica cuando la media es grande comparada con la desviación típica de los datos.

Por eso los algoritmos en línea, como el de Welford, actualizan la media y la suma de cuadrados de
forma incremental y estable, sin restar dos cantidades grandes y casi iguales al final.
