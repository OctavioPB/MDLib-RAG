# Métricas

## Distancia a la secuencia principal

La inestabilidad de un componente mide qué fracción de sus dependencias apunta hacia fuera y se
usa junto con la abstracción para ubicar cada componente en el plano de diseño.

La distancia perpendicular de un componente a la secuencia principal se calcula como:

$$D(X) = \frac{|A(X) + I(X) - 1|}{\sqrt{2}}$$

Un valor cercano a cero indica que el componente está bien equilibrado entre abstracción y
estabilidad, mientras que valores altos señalan zonas de dolor o de inutilidad.

**La actualización visualizada:**

| Término | Valor |
|---|---|
| A | abstracción |
| I | inestabilidad |
