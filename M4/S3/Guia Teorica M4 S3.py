import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

print("# --- Sesión 3: Análisis Exploratorio de Datos ---")

# --- 1: Correlación de Variables: Análisis y Visualización ---
# ## Correlación de Variables
# La correlación es una medida estadística que describe el grado de relación entre dos variables.
# En ciencia de datos, **comprender la correlación es esencial para identificar patrones, predecir comportamientos y tomar decisiones informadas**.
# Es fundamental recordar que **correlación no implica causalidad**, lo que significa que una relación entre dos variables no necesariamente indica que una causa a la otra.
# Existen diferentes tipos de correlación:
# - **Positiva**: Cuando ambas variables aumentan.
# - **Negativa**: Cuando una variable aumenta y la otra disminuye.
# - **Nula**: Cuando no hay una relación clara entre las variables.

# --- 2: Tablas de Contingencia ---
# ## Tablas de Contingencia
# ### ¿Qué son?
# Una tabla de contingencia es una **tabla de frecuencias utilizada para analizar la relación entre dos variables categóricas**.
# Muestra cuántas veces ocurre cada combinación de categorías y permite evaluar si existe alguna asociación entre ellas.
# ### ¿Cuándo usarlas?
# Son útiles para:
# - Examinar la relación entre dos variables categóricas (por ejemplo, género y preferencia de producto).
# - Calcular probabilidades condicionales y frecuencias relativas.
# - Construir una tabla de chi-cuadrado para evaluar la independencia entre variables.

# --- 3: Interpretación de Tablas de Contingencia y Ejemplo en Python ---
# ## Interpretación de Tablas de Contingencia
# Las filas y columnas de la tabla representan las categorías de cada variable.
# Cada celda muestra la frecuencia de esa combinación específica, permitiendo analizar patrones y asociaciones entre las variables categóricas.

# ### Ejemplo en Python: Creación de una Tabla de Contingencia
# Utilizaremos la función `pd.crosstab()` de Pandas para contar la frecuencia de cada combinación de valores entre dos columnas categóricas.

# Crear un DataFrame con variables categóricas de ejemplo
df = pd.DataFrame({
    'Género': ['Masculino', 'Femenino', 'Femenino', 'Masculino'],
    'Preferencia': ['Deportes', 'Cine', 'Deportes', 'Cine']
})
print("\nDataFrame de ejemplo:")
print(df)

# Crear tabla de contingencia
tabla_contingencia = pd.crosstab(df['Género'], df['Preferencia'])
print("\nTabla de Contingencia Resultante:")
print(tabla_contingencia)

# ### Interpretación de la tabla de contingencia de ejemplo:
# Las filas representan el Género y las columnas la Preferencia.
# - **Filas (Género)**:
#   - **Femenino**: Hay 1 persona de género femenino que prefiere Cine y 1 persona que prefiere Deportes.
#   - **Masculino**: Hay 1 persona de género masculino que prefiere Cine y 1 persona que prefiere Deportes.
# - **Columnas (Preferencia)**:
#   - **Cine**: En total, 2 personas prefieren Cine (1 femenino y 1 masculino).
#   - **Deportes**: En total, 2 personas prefieren Deportes (1 femenino y 1 masculino).


# --- 4: Gráfico Scatterplot ---
# ## Gráfico Scatterplot (Diagrama de Dispersión)
# ### Definición
# Un scatterplot es una **herramienta visual utilizada para examinar la relación entre dos variables numéricas**.
# Cada punto en el gráfico representa una observación en el conjunto de datos.
# ### Casos de uso
# Es ideal para:
# - Visualizar si existe una correlación entre dos variables numéricas.
# - Detectar outliers (valores atípicos) o patrones en los datos.
# - Analizar la relación entre variables en problemas de regresión.

# --- 5: Interpretación de Scatterplot y Ejemplo en Python ---
# ## Interpretación de Gráficos Scatterplot
# - Si los puntos siguen una línea ascendente, hay una **correlación positiva** (cuando una variable aumenta, la otra también).
# - Si los puntos forman una línea descendente, hay una **correlación negativa**.
# - Si los puntos están dispersos sin una forma clara, **no hay correlación**.

# ### Ejemplo en Python: Creación de un Scatterplot
# Utilizaremos `matplotlib.pyplot` para visualizar la relación entre la cantidad vendida de productos y su precio promedio.

# Datos ficticios de ventas (cantidad de productos vendidos y precio promedio)
np.random.seed(42) # Para reproducibilidad
cantidad_vendida = np.random.randint(10, 100, 50) # Número de productos vendidos
precio_promedio = cantidad_vendida * np.random.uniform(0.8, 1.2, 50) # Precio con variación aleatoria

# Crear el scatterplot
plt.figure(figsize=(8, 5))
plt.scatter(cantidad_vendida, precio_promedio, color='blue', alpha=0.5)
plt.xlabel('Cantidad Vendida')
plt.ylabel('Precio Promedio')
plt.title('Relación entre Cantidad Vendida y Precio Promedio (Ejemplo)')
plt.grid(True)
plt.show()
print("El gráfico de dispersión muestra la relación entre 'Cantidad Vendida' y 'Precio Promedio'.")
print("En este ejemplo, dado cómo se generaron los datos (precio_promedio = cantidad_vendida * variación), se espera ver una correlación positiva, donde a mayor cantidad vendida, mayor precio promedio.")


# --- 6: Coeficiente de Correlación de Pearson ---
# ## Coeficiente de Correlación de Pearson (r)
# ### Definición
# El coeficiente de correlación de Pearson (r) es una **medida estadística que cuantifica la relación lineal entre dos variables numéricas**.
# Su rango va de -1 a 1:
# - **1**: Indica una correlación positiva perfecta.
# - **-1**: Indica una correlación negativa perfecta.
# - **0**: Indica ausencia de correlación lineal.
# ### Cálculo
# Se calcula mediante una fórmula que relaciona la covarianza de las variables con el producto de sus desviaciones estándar.
# "La fórmula es: r = Σ((xi - x̄)(yi - ȳ)) / √[Σ(xi - x̄)² Σ(yi - ȳ)²]"

# --- 7 y 8: Ejemplo en Python: Coeficiente de Correlación de Pearson ---
# ## Ejemplo en Python: Cálculo del Coeficiente de Correlación de Pearson
# Podemos calcular el coeficiente de correlación de Pearson en Python usando `scipy.stats.pearsonr()` o `numpy.corrcoef()`.

# CORRECCIÓN: Datos ficticios con valores reales para el análisis
# Datos ficticios: Ventas de un producto y presupuesto de marketing
np.random.seed(42)  # Para reproducibilidad
presupuesto_marketing = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500])
ventas = np.array([10, 15, 18, 25, 30, 32, 38, 42, 45, 50])

print(f"\nPresupuesto de Marketing: {presupuesto_marketing}")
print(f"Ventas: {ventas}")

print("\n### Ejemplo 1: Usando SciPy (`scipy.stats.pearsonr()`)")
coef, p_valor = stats.pearsonr(presupuesto_marketing, ventas)
print(f"Coeficiente de correlación de Pearson (SciPy): {coef:.2f}")
print(f"P-valor (SciPy): {p_valor:.4f}")

print("\n### Ejemplo 2: Usando NumPy (`numpy.corrcoef()`)")
corr_matrix = np.corrcoef(presupuesto_marketing, ventas)
print(f"Matriz de correlación usando NumPy:")
print(corr_matrix)
print(f"Coeficiente de Pearson usando NumPy: {corr_matrix[0,1]:.2f}")

# Crear scatterplot para visualizar la correlación
plt.figure(figsize=(8, 5))
plt.scatter(presupuesto_marketing, ventas, color='red', alpha=0.7)
plt.xlabel('Presupuesto de Marketing')
plt.ylabel('Ventas')
plt.title('Relación entre Presupuesto de Marketing y Ventas')
plt.grid(True)
plt.show()

# ## Interpretación del Coeficiente de Pearson
# Los valores de 'r' se interpretan según la siguiente escala:
# - **0.8 ≤ r ≤ 1.0**: Correlación positiva fuerte
# - **0.5 < r < 0.8**: Correlación positiva moderada
# - **0.2 < r < 0.5**: Correlación positiva débil
# - **-0.2 ≤ r ≤ 0.2**: Correlación despreciable o nula
# - **-0.5 < r < -0.2**: Correlación negativa débil
# - **-0.8 < r < -0.5**: Correlación negativa moderada
# - **-1.0 ≤ r ≤ -0.8**: Correlación negativa fuerte
# Si el p-valor es menor a 0.05, la correlación es estadísticamente significativa.
print(f"\nInterpretación:")
if coef >= 0.8:
    interpretacion = "correlación positiva fuerte"
elif coef >= 0.5:
    interpretacion = "correlación positiva moderada"
elif coef >= 0.2:
    interpretacion = "correlación positiva débil"
elif coef >= -0.2:
    interpretacion = "correlación despreciable o nula"
elif coef >= -0.5:
    interpretacion = "correlación negativa débil"
elif coef >= -0.8:
    interpretacion = "correlación negativa moderada"
else:
    interpretacion = "correlación negativa fuerte"

print(f"Con un coeficiente de {coef:.2f}, tenemos una {interpretacion}.")
print(f"El p-valor de {p_valor:.4f} {'es' if p_valor < 0.05 else 'no es'} estadísticamente significativo (α = 0.05).")

# --- 9: Consideraciones al Usar el Coeficiente de Pearson ---
# ## Consideraciones al Usar el Coeficiente de Pearson
# - **Solo mide relaciones lineales**: Si la relación entre las variables es no lineal (como una curva o parábola), el coeficiente de Pearson puede no detectarla correctamente o subestimar la fuerza de la relación. En estos casos, es mejor utilizar otras medidas como la correlación de Spearman.
# - **Es sensible a outliers**: Los valores extremos o atípicos pueden influir significativamente en el coeficiente y dar resultados engañosos. Es importante identificar y tratar adecuadamente los outliers antes de calcular la correlación.
# - **No implica causalidad**: Una alta correlación entre dos variables no significa que una cause la otra. Pueden estar relacionadas debido a un tercer factor o por pura coincidencia. Siempre se debe complementar con análisis adicionales.
# - **Complementar con visualización**: Un gráfico de dispersión (scatterplot) puede ayudar a interpretar mejor la relación entre las variables y detectar patrones que el coeficiente por sí solo no revela.

# --- 10: Causalidad vs. Correlación ---
# ## Causalidad vs. Correlación
# ### Correlación
# - Mide la relación estadística entre dos variables, indicando si tienden a moverse juntas.
# - No establece dirección del efecto y puede estar influenciada por terceros factores.
# - Se comprueba mediante métodos estadísticos como el coeficiente de Pearson.
# ### Causalidad
# - Ocurre cuando un cambio en una variable provoca directamente un cambio en otra.
# - Requiere pruebas rigurosas y descartar otros factores influyentes.
# - Se demuestra mediante experimentos controlados y estudios longitudinales.

# --- 11: Diferencias Clave entre Correlación y Causalidad ---
# ## Diferencias Clave entre Correlación y Causalidad
# | Característica         | Correlación                                         | Causalidad                                         |
# |------------------------|-----------------------------------------------------|----------------------------------------------------|
# | Relación Directa       | Puede existir sin que una variable cause a la otra  | Existe una relación de causa y efecto              |
# | Dirección del Efecto   | No se puede determinar                              | Se puede establecer mediante pruebas               |
# | Influencia de Factores | Puede estar influida por variables externas         | Se buscan descartar factores externos              |
# | Comprobación           | Métodos estadísticos (Pearson, Spearman, etc.)      | Experimentos controlados y estudios longitudinales |
# | Interpretación         | Indica asociación, no implica causalidad            | Implica que un cambio en una variable afecta a otra|
# *(Tabla basada en la Diapositiva 11 del material fuente)*

# --- 12 y 13: Ejemplos de Confusión entre Correlación y Causalidad ---
# ## Ejemplos de Confusión entre Correlación y Causalidad
# - **Helados y Ahogamientos**: Existe una correlación positiva entre las ventas de helado y los casos de ahogamiento. Esto no significa que comer helado cause ahogamientos, sino que ambos aumentan en verano debido a un tercer factor: el clima cálido que lleva a más personas a comprar helados y a nadar.
# - **Cigüeñas y Nacimientos**: En algunas regiones, se ha observado una correlación entre la población de cigüeñas y la tasa de natalidad. Esto no implica que las cigüeñas traigan bebés, sino que ambas variables pueden estar relacionadas con factores como la ruralidad o el desarrollo económico.
# - **Felicidad y Longevidad**: Aunque existe correlación entre felicidad y longevidad, no podemos afirmar que ser feliz cause directamente una vida más larga. Factores como el nivel socioeconómico, hábitos de salud o genética pueden influir en ambas variables.

# --- 14: Mejores Prácticas ---
# ## Mejores Prácticas
# Para un análisis de datos efectivo y decisiones informadas, se recomienda:
# - **Toma de Decisiones Informadas**: Basadas en un análisis riguroso de los datos.
# - **Combinación de Métodos Estadísticos y Experimentales**: No confiar únicamente en un solo tipo de análisis.
# - **Visualización de Datos**: Complementar los análisis numéricos con gráficos que ayuden a entender mejor las relaciones.
# - **Análisis Crítico**: Cuestionar las relaciones aparentes y no asumir causalidad automáticamente.
# - **Conocimiento del Dominio**: Entender el contexto del problema para interpretar correctamente los resultados.
#
# **La correlación es útil para identificar patrones, pero no implica causalidad.** Para demostrar causalidad, se requieren experimentos, estudios longitudinales y modelos de regresión.
# En ciencia de datos, **es crucial diferenciar entre correlación y causalidad para evitar conclusiones erróneas y tomar decisiones informadas**.
# Muchas veces, encontrar una correlación puede ser el primer paso para investigar una posible causalidad, pero nunca debe ser la única evidencia.
# El análisis riguroso y la combinación de métodos estadísticos con conocimiento del dominio son fundamentales para establecer relaciones causales válidas.