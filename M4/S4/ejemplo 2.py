# ==============================================================================
# ANÁLISIS EXPLORATORIO DE DATOS Y REGRESIONES LINEALES
# ==============================================================================

# ------------------------------------------------------------------------------
# 1: REGRESIONES LINEALES: FUNDAMENTOS Y APLICACIONES
# ------------------------------------------------------------------------------
"""
Las regresiones lineales son una de las técnicas más utilizadas en ciencia de datos y estadística.
Su propósito es modelar la relación entre variables.
Su aplicación es amplia, abarcando campos como economía, finanzas, salud y marketing.
Son fundamentales para la predicción y el análisis de tendencias.
"""

# ------------------------------------------------------------------------------
# 2: ¿QUÉ ES UNA REGRESIÓN LINEAL SIMPLE?
# ------------------------------------------------------------------------------
"""
Definición:
Una regresión lineal simple modela la relación entre una variable dependiente (Y) y una variable independiente (X)
mediante una línea recta.
Su propósito es predecir el valor de Y en función de X, basándose en una relación lineal entre ambas variables.

Ecuación de la Regresión Lineal Simple:
Y = β₀ + β₁X + ε

Donde:
- β₀ (beta cero) es el intercepto (valor de Y cuando X=0).
- β₁ (beta uno) es la pendiente (cambio en Y por unidad de cambio en X).
- ε (épsilon) es el término de error.

Objetivo:
Encontrar los valores óptimos de β₀ y β₁ que mejor ajusten la relación entre las variables,
minimizando la suma de los errores al cuadrado entre los valores observados y los predichos.
"""

# ------------------------------------------------------------------------------
# 3: DETERMINACIÓN DE LOS COEFICIENTES DE REGRESIÓN
# ------------------------------------------------------------------------------
"""
Método de Mínimos Cuadrados:
Para determinar los coeficientes de regresión (β₀ y β₁), se utiliza el método de mínimos cuadrados.
Este método busca minimizar la suma de los errores al cuadrado entre los valores observados y los predichos.

Cálculo de la Pendiente (β₁):
La pendiente se calcula considerando la covarianza entre X e Y dividida por la varianza de X,
utilizando los valores individuales y promedios de las variables.

Cálculo del Intercepto (β₀):
El intercepto representa el valor esperado de Y cuando X es igual a cero.
Se calcula a partir del promedio de Y, la pendiente y el promedio de X.

Ejemplos de aplicación:
- Economía: Predecir el precio de una casa según su tamaño.
- Salud: Relacionar el consumo de calorías con el peso corporal.
- Educación: Analizar la relación entre horas de estudio y calificaciones.
- Negocios: Estimar ventas futuras a partir de la inversión en marketing.
"""


















# ------------------------------------------------------------------------------
# 4 y 5: EJEMPLO DE REGRESIÓN LINEAL SIMPLE EN PYTHON CON SCIKIT-LEARN
# ------------------------------------------------------------------------------
"""
Objetivo del código:
Este código implementa una Regresión Lineal Simple utilizando Python con la librería scikit-learn.
Su objetivo es ajustar una línea recta a un conjunto de datos y hacer predicciones basadas en ella.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Importación de Librerías (explicación de la Diapositiva 5)
# NumPy para manejar datos numéricos.
# Matplotlib para visualizar los datos y la línea de regresión.
# LinearRegression de sklearn.linear_model para crear y entrenar el modelo.

# Datos de ejemplo (del código en la Diapositiva 4)
# Ejemplo: relación lineal simple entre X y Y
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 3, 5, 7, 11])

# 2. Creación y Entrenamiento del modelo (explicación de la Diapositiva 5)
# Se crea un modelo de Regresión Lineal con LinearRegression().
# Se entrena el modelo con modelo.fit(X, Y), ajustando la mejor línea recta a los datos.
modelo = LinearRegression()
modelo.fit(X, Y)

# 3. Obtención de Coeficientes (explicación de la Diapositiva 5)
# Se obtienen β₀ (intercepto) y β₁ (pendiente).
# Estos coeficientes ayudan a interpretar la relación entre X e Y y realizar predicciones.
beta_0 = modelo.intercept_
beta_1 = modelo.coef_

print("--- Ejemplo de Regresión Lineal Simple con Scikit-learn ---")
print(f"Intercepto (β₀): {beta_0}")
print(f"Pendiente (β₁): {beta_1}")

# 4. Visualización y Predicción (explicación de la Diapositiva 5)
# Se utilizan los valores de X para predecir los valores de Y con la ecuación obtenida.
Y_pred = modelo.predict(X)

# Se visualizan los datos originales junto con la línea de regresión.
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color="blue", label="Datos reales")
plt.plot(X, Y_pred, color="red", label="Regresión Lineal")
plt.xlabel("Variable X")
plt.ylabel("Variable Y")
plt.title("Regresión Lineal Simple")
plt.legend()
plt.grid(True)
plt.show()

print("\n") # Salto de línea para separar las secciones




















# ------------------------------------------------------------------------------
# 6: MÉTRICAS DE ERROR Y EVALUACIÓN
# ------------------------------------------------------------------------------
"""
Estas métricas se utilizan para evaluar el rendimiento de un modelo de regresión.

Error Cuadrático Medio (MSE):
- El MSE es el promedio de los errores al cuadrado.
- Penaliza fuertemente los errores grandes, útil cuando se busca minimizar desviaciones significativas.
- Se expresa en unidades cuadradas.
- Fórmula: MSE = (1/n) * Σ(Yi - Ŷi)²

Error Absoluto Medio (MAE):
- El MAE es el promedio de los errores absolutos entre los valores reales y las predicciones.
- Es más robusto ante valores atípicos que el MSE.
- Se mide en las mismas unidades que los datos originales.
- Fórmula: MAE = (1/n) * Σ|Yi - Ŷi|

Coeficiente de Determinación (R²):
- El R² mide qué tan bien el modelo explica la variabilidad de la variable dependiente.
- Un valor de 1 indica que el modelo explica el 100% de la varianza.
- Un valor de 0 indica que el modelo no explica nada.
- Fórmula: R² = 1 - (Σ(Yi - Ŷi)²) / (Σ(Yi - Ȳ)²)
"""

# ------------------------------------------------------------------------------
# 7: EJEMPLOS EN PYTHON (CÁLCULO DE MÉTRICAS)
# ------------------------------------------------------------------------------

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("--- Ejemplos en Python: Cálculo de Métricas de Evaluación ---")
# Valores reales y predicciones de ejemplo (del código en la Diapositiva 7)
Y_real = np.array([2, 3, 5, 7, 11])
Y_pred = np.array([2.2, 3.1, 4.8, 5.9, 7.5])
Y_pred = np.array([2.2, 3.1, 4.8, 5.9, 7.5])

# Cálculo del MSE
mse = mean_squared_error(Y_real, Y_pred)
print(f"Error Cuadrático Medio (MSE): {mse}")

# Cálculo del MAE
mae = mean_absolute_error(Y_real, Y_pred)
print(f"Error Absoluto Medio (MAE): {mae}")

# Cálculo del R²
r2 = r2_score(Y_real, Y_pred)
print(f"Coeficiente de determinación R²: {r2}")

print("\n") # Salto de línea para separar las secciones

































# ------------------------------------------------------------------------------
# 8: IMPLEMENTACIÓN CON STATSMODELS
# ------------------------------------------------------------------------------
"""
Statsmodels permite construir y evaluar modelos de regresión lineal con métricas de error integradas,
ofreciendo un análisis estadístico más completo.
"""

import statsmodels.api as sm

print("--- Implementación con Statsmodels ---")
# Datos de ejemplo (del código en la Diapositiva 8)
X_sm = np.arange(1, 25)
Y_sm = np.random.randint(1, 100, 24)

# 1. Preparación de Datos (explicación de la Diapositiva 8)
# Se añade una columna de 1s para el intercepto, necesario para el ajuste del modelo en statsmodels.
X_sm = sm.add_constant(X_sm)

# 2. Ajuste del Modelo (explicación de la Diapositiva 8)
# Se crea y ajusta el modelo utilizando OLS (Ordinary Least Squares).
# Esto permite obtener los coeficientes y estadísticas de ajuste.
modelo_sm = sm.OLS(Y_sm, X_sm).fit()

# 3. Análisis de Resultados (explicación de la Diapositiva 8)
# Se obtienen valores de coeficientes, R², errores estándar y pruebas de significancia estadística
# (p-values, t-values) para evaluar la calidad del modelo.
print("Resumen del Modelo (Statsmodels):")
print(modelo_sm.summary())

# Nota sobre advertencias de normalidad y curtosis:
print("\nNota: Si ves advertencias como 'omni_normtest is not valid with less than 8 observations' o 'kurtosistest p-value may be inaccurate with fewer than 20 observations', es porque algunas pruebas estadísticas requieren más datos para ser confiables. Si quieres evitar estas advertencias, usa más de 20 observaciones en los datos de ejemplo.")

print("\n") # Salto de línea para separar las secciones






# ------------------------------------------------------------------------------
# 9: REGRESIÓN LINEAL MÚLTIPLE
# ------------------------------------------------------------------------------
"""
Definición:
La Regresión Lineal Múltiple es una extensión de la regresión lineal simple.
Permite modelar la relación entre una variable dependiente (Y) y dos o más variables independientes (X₁, X₂,…, Xₙ).

Ecuación de la Regresión Lineal Múltiple:
Y = β₀ + β₁X₁ + β₂X₂ + … + βₙXₙ + ε

Componentes:
- β₀ (Intercepto): Valor de Y cuando todas las variables independientes son 0.
- β₁, β₂, ..., βₙ (Coeficientes): Miden el efecto de cada variable independiente sobre Y.
- ε (Error residual): Captura variaciones no explicadas por el modelo.

Aplicación:
Un ejemplo práctico es predecir las ventas (Y) de una empresa en función del gasto en publicidad en TV (X₁),
radio (X₂) e internet (X₃).
"""

# ------------------------------------------------------------------------------
# 10: SUPUESTOS DE UNA REGRESIÓN MÚLTIPLE
# ------------------------------------------------------------------------------
"""
Para que los resultados de una regresión lineal sean válidos y fiables, deben cumplirse ciertos supuestos:

1. Linealidad:
   La relación entre las variables independientes y la variable dependiente debe ser lineal.
   Se puede verificar con gráficos de dispersión o transformaciones de variables.

2. Independencia de Errores:
   Los errores (ε) no deben estar correlacionados entre sí.
   Se evalúa mediante el test de Durbin-Watson para detectar autocorrelación en los residuos.

3. Homocedasticidad:
   La varianza de los errores debe ser constante en todos los niveles de las variables independientes.
   Se verifica con gráficos de residuos para detectar patrones de varianza no constante.

4. Normalidad de Errores:
   Los errores deben seguir una distribución normal.
   Este supuesto es importante para que los intervalos de confianza y las pruebas de hipótesis sean válidos.
   Se prueba con tests estadísticos (ej., Shapiro-Wilk) o histogramas de residuos.
"""
# ------------------------------------------------------------------------------
# 11: MÉTODOS DE SELECCIÓN DEL MODELO
# ------------------------------------------------------------------------------
"""
Cuando se trabaja con múltiples variables, existen métodos para seleccionar las variables más adecuadas
e incluso para optimizar el modelo:

1. Forward Selection (Selección Adelante):
   Se inicia con un modelo sin variables predictoras y se añaden variables una por una
   en función de su significancia estadística, hasta que no se encuentran más variables que mejoren el modelo.

2. Backward Elimination (Eliminación Hacia Atrás):
   Se parte de un modelo que incluye todas las variables predictoras disponibles.
   Se eliminan iterativamente las variables menos significativas hasta que todas las variables restantes
   sean estadísticamente significativas.

3. Stepwise Regression (Regresión por Pasos):
   Combina aspectos de Forward Selection y Backward Elimination.
   En cada paso, puede añadir una variable (como en Forward Selection) o eliminar una variable (como en Backward Elimination)
   para optimizar el modelo basándose en criterios estadísticos.
"""