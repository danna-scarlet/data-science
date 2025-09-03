# ==============================================================================
# SESION 5: ANÁLISIS EXPLORATORIO DE DATOS
# VISUALIZACIÓN DE DATOS CON SEABORN
# ==============================================================================

# --- 1: Inicio ---
# Sesión 5: Análisis Exploratorio de Datos
# Esta sesión se enfoca en la visualización de datos como parte esencial del Análisis Exploratorio de Datos (EDA).

# --- 2: Visualización de Datos con Seaborn ---
# La visualización de datos es una parte esencial del análisis exploratorio de datos (EDA).
# Permite comprender patrones, tendencias y relaciones en los datos de manera intuitiva.
# Seaborn es una librería de Python basada en Matplotlib que facilita la creación de gráficos estadísticos atractivos y fáciles de interpretar.

# Características Principales de Seaborn:
# 1. Integración con Pandas: Trabaja directamente con DataFrames de Pandas, facilitando la visualización sin transformaciones adicionales, lo que permite un flujo de trabajo eficiente.
# 2. Estética Mejorada: Proporciona temas de diseño y paletas de colores atractivos por defecto, mejorando la presentación de gráficos con poco esfuerzo.
# 3. Gráficos Estadísticos: Incluye funciones específicas para visualizar distribuciones, relaciones y datos categóricos (ej., histogramas, diagramas de dispersión, boxplots), facilitando el análisis estadístico visual.

# --- 3: Importación e Instalación ---

# Instalación:
# Si aún no tienes Seaborn instalado, puedes hacerlo con el siguiente comando en la terminal:
# pip install seaborn

# Importación Básica:
# Para utilizar Seaborn, primero debemos importarla junto con otras librerías complementarias como Matplotlib y Pandas.
# En algunos ejemplos, también se usa NumPy para simular datos.

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # Necesario para simular datos en ejemplos de distribución

# Configuración:
# Una vez importadas las librerías, podemos comenzar a crear gráficos de manera sencilla.
# Se pueden usar los estilos predeterminados de Seaborn o personalizarlos según las necesidades.

























# --- 4: Gráficos de Distribución - Histplot ---
# La función histplot() es la recomendada en Seaborn para visualizar histogramas, reemplazando al antiguo distplot().
# Permite examinar la distribución de los datos, identificar asimetrías, sesgos o valores atípicos, y evaluar la forma y dispersión de la variable.

# Ejemplo en Python:
print("--- Gráfico de Histograma con KDE (Histplot) ---")
# Datos simulados (Distribución Normal)
data_hist = np.random.randn(1000)

# Gráfico de histograma con KDE activado
plt.figure(figsize=(8, 6))
sns.histplot(data_hist, kde=True, color="blue", bins=30)
plt.title("Distribución de Datos con KDE")
plt.xlabel("Valores")
plt.ylabel("Frecuencia")
plt.grid(True, linestyle='--', alpha=0.7) # Añadir una cuadrícula para mejor visualización
plt.show()

# Salida esperada:
# - Un histograma con 30 bins (barras) que representa la frecuencia de los datos.
# - Una curva KDE (Kernel Density Estimation) opcional que suaviza la distribución.




# --- 5: Gráficos de Distribución - Kdeplot ---
# La función kdeplot() se usa cuando solo queremos la curva de densidad sin el histograma.
# Es ideal para obtener una visualización más suave de la distribución de los datos.

# Ejemplo en Python:
print("\n--- Gráfico de Estimación de Densidad del Kernel (Kdeplot) ---")
# Datos simulados (Distribución Normal)
data_kde = np.random.randn(1000)

# Gráfico de densidad con sombra
plt.figure(figsize=(8, 6))
sns.kdeplot(data_kde, shade=True, color="red")
plt.title("Distribución de Datos con KDE")
plt.xlabel("Valores")
plt.ylabel("Densidad")
plt.grid(True, linestyle='--', alpha=0.7) # Añadir una cuadrícula para mejor visualización
plt.show()

# Salida esperada:
# - Una curva KDE que estima la densidad de probabilidad de la variable.
# - Sombra bajo la curva para visualizar mejor la distribución (shade=True).





















# --- 6: ¿Cuándo usar histplot() y kdeplot()? ---
# Tabla de uso:
# - histplot(): Para analizar frecuencia y distribución de datos discretos o continuos.
# - kdeplot(): Cuando queremos una visualización más suave de la distribución sin histograma.
# - Ambos combinados: Para obtener una visión más detallada y comparar distribuciones.

# --- 7: Gráficos de Dispersión y Correlación ---
# - Jointplot: Combina un gráfico de dispersión con histogramas marginales, permitiendo observar la relación entre dos variables y la distribución individual de cada una. Ideal para analizar correlaciones, detectar patrones y valores atípicos.
# - Pairplot: Genera una matriz de gráficos de dispersión para múltiples variables, permitiendo analizar visualmente todas las correlaciones en un conjunto de datos. Perfecto para comparar relaciones entre múltiples variables en un solo gráfico.
# - Regplot: Ideal para visualizar la relación lineal entre dos variables numéricas mediante una línea de regresión. Permite analizar tendencias, detectar patrones y probar la existencia de relaciones lineales entre variables.

# --- Ejemplos de Gráficos de Dispersión y Correlación ---
# Simular un DataFrame de Pandas con columnas 'Edad', 'Ingresos', 'Ahorro', etc.
data_heatmap = pd.DataFrame({
    'Edad': np.random.randint(18, 65, 50),
    'Ingresos': np.random.normal(30000, 8000, 50),
    'Ahorro': np.random.normal(5000, 2000, 50),
    'Gastos': np.random.normal(15000, 4000, 50)
})
print("\n--- Ejemplo: Jointplot ---")
sns.jointplot(x='Edad', y='Ingresos', data=data_heatmap, kind='scatter', color='purple')
plt.title('Jointplot: Edad vs Ingresos', y=0.90)
plt.suptitle('Jointplot: Edad vs Ingresos', y=0.90)
plt.show()

print("\n--- Ejemplo: Pairplot ---")
sns.pairplot(data_heatmap)
plt.title('Pairplot: Matriz de Gráficos de Dispersión')
plt.suptitle('Pairplot: Matriz de Gráficos de Dispersión', y=1.02)
plt.show()

print("\n--- Ejemplo: Regplot ---")
plt.figure(figsize=(8, 6))
sns.regplot(x='Edad', y='Ahorro', data=data_heatmap, color='green')
plt.title('Regplot: Relación Lineal entre Edad y Ahorro')
plt.xlabel('Edad')
plt.ylabel('Ahorro')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()























# --- 8 & 9: Gráficos de Variables Categóricas ---
# - Barplot: Útil para comparar la media de una variable numérica en diferentes categorías. Muestra la media de una variable numérica agrupada por una variable categórica e incluye barras de error para representar la dispersión de los datos.
# - Countplot: Ideal para visualizar la frecuencia de cada categoría dentro de una variable categórica. Muestra la cantidad de observaciones en cada categoría y permite comparar la distribución de frecuencias en distintos grupos.
# - Boxplot (gráfico de caja y bigotes): Excelente para analizar la distribución, asimetría y outliers en los datos. Visualiza la mediana, rango intercuartil (IQR) y valores atípicos de una variable numérica en diferentes categorías.
# - Violinplot: Combina el boxplot con una estimación de densidad para visualizar mejor la distribución de los datos. Muestra la forma de la distribución en cada categoría y permite comparar la dispersión de los valores en diferentes grupos.

# --- Ejemplos de Gráficos de Variables Categóricas ---
# Simular un DataFrame de Pandas
data_heatmap = pd.DataFrame({
    'Edad': np.random.randint(18, 65, 50),
    'Ingresos': np.random.normal(30000, 8000, 50),
    'Ahorro': np.random.normal(5000, 2000, 50),
    'Gastos': np.random.normal(15000, 4000, 50)
})
np.random.seed(1)
data_cat = data_heatmap.copy()

# Simulamos una variable categórica para los ejemplos
data_cat['Grupo'] = np.random.choice(['A', 'B', 'C'], size=len(data_cat))

print("\n--- Ejemplo: Barplot ---")
plt.figure(figsize=(8, 6))
sns.barplot(x='Grupo', y='Ingresos', data=data_cat, ci='sd', palette='pastel')
plt.title('Barplot: Ingresos Promedio por Grupo')
plt.xlabel('Grupo')
plt.ylabel('Ingresos')
plt.show()

print("\n--- Ejemplo: Countplot ---")
plt.figure(figsize=(8, 6))
sns.countplot(x='Grupo', data=data_cat, palette='Set2')
plt.title('Countplot: Frecuencia de Grupos')
plt.xlabel('Grupo')
plt.ylabel('Frecuencia')
plt.show()

print("\n--- Ejemplo: Boxplot ---")
plt.figure(figsize=(8, 6))
sns.boxplot(x='Grupo', y='Ahorro', data=data_cat, palette='Set3')
plt.title('Boxplot: Distribución de Ahorro por Grupo')
plt.xlabel('Grupo')
plt.ylabel('Ahorro')
plt.show()

print("\n--- Ejemplo: Violinplot ---")
plt.figure(figsize=(8, 6))
sns.violinplot(x='Grupo', y='Gastos', data=data_cat, palette='muted')
plt.title('Violinplot: Distribución de Gastos por Grupo')
plt.xlabel('Grupo')
plt.ylabel('Gastos')
plt.show()




















# --- 10: Heatmap: Visualización de Matrices ---
# Un heatmap es una representación gráfica de datos donde los valores individuales contenidos en una matriz se representan con colores. Se usa comúnmente para visualizar la correlación entre variables en un conjunto de datos.

# Proceso de creación e interpretación de un Heatmap:
# 1. Preparación: Importar librerías y preparar los datos.
# 2. Creación: Generar la matriz de correlación y aplicar sns.heatmap().
# 3. Personalización: Ajustar colores, anotaciones y formato.
# 4. Interpretación: Identificar patrones y correlaciones.

# Ejemplo conceptual de Heatmap (no se proporciona un ejemplo de código en la fuente, pero se describe el proceso):
print("\n--- Concepto de Heatmap para Correlación ---")
# Simular un DataFrame de Pandas
data_heatmap = pd.DataFrame(np.random.rand(10, 5), columns=[f'Var{i}' for i in range(1, 6)])
# Calcular la matriz de correlación (Paso 2: Creación)
correlation_matrix = data_heatmap.corr()

# Generar el Heatmap (Paso 2: Creación, Paso 3: Personalización)
plt.figure(figsize=(8, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Heatmap de Matriz de Correlación")
plt.show()
print("El heatmap permite visualizar las correlaciones entre variables, donde los colores representan la fuerza y dirección de la relación.")
print("Por ejemplo, valores cercanos a 1 (colores cálidos) indican correlación positiva fuerte, cercanos a -1 (colores fríos) indican correlación negativa fuerte, y cercanos a 0 (colores neutros) indican poca correlación.") # Interpretación



































# --- 11: Grillas de Gráficos: PairGrid y FacetGrid ---
# Son herramientas para crear múltiples gráficos que muestran relaciones o distribuciones segmentadas.

# Proceso:
# 1. Crear objeto Grid: Inicializar PairGrid o FacetGrid con los datos y variables a visualizar.
#    - PairGrid: Para relaciones entre múltiples variables (generalmente numéricas).
#    - FacetGrid: Para segmentar gráficos por categorías (variables numéricas o categóricas).
# 2. Mapear funciones: Aplicar diferentes tipos de gráficos a las distintas partes de la grilla.
#    - PairGrid: Usar métodos como map(), map_diag() (diagonal) y map_offdiag() (fuera de la diagonal).
#    - FacetGrid: Usar map() para un solo tipo de gráfico por categoría.
# 3. Personalizar: Ajustar títulos, etiquetas, leyendas y otros elementos visuales para mejorar la interpretación y presentación.
# 4. Visualizar: Mostrar la grilla completa con plt.show() o guardarla como imagen.

# Comparación entre PairGrid y FacetGrid:
# | Característica           | PairGrid                                                                      | FacetGrid                                                   |
# |--------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------|
# | Tipo de datos            | Numéricos                                                                     | Numéricos o categóricos                                     |
# | Relación entre variables | Muestra combinaciones entre múltiples variables                               | Separa los gráficos por categorías                          |
# | Personalización          | Muy flexible (distintos tipos de gráficos en diagonal y fuera de la diagonal) | Se centra en un solo tipo de gráfico por categoría          |
# | Uso recomendado          | Análisis de correlaciones y tendencias entre varias variables numéricas       | Comparaciones dentro de subgrupos categóricos               |

# --- Ejemplo: PairGrid ---
print("\n--- Ejemplo: PairGrid ---")
pairgrid = sns.PairGrid(data_heatmap)
pairgrid.map_upper(sns.scatterplot, color='blue')
pairgrid.map_lower(sns.kdeplot, fill=True, color='orange')
pairgrid.map_diag(sns.histplot, color='green', kde=True)
pairgrid.figure.suptitle('PairGrid: Combinación de Gráficos', fontsize=16, color='darkblue', y=0.90)
plt.show()

# --- Ejemplo: FacetGrid ---
print("\n--- Ejemplo: FacetGrid ---")
facet = sns.FacetGrid(data_cat, col='Grupo', height=4, aspect=1)
facet.map_dataframe(sns.histplot, x='Ahorro', color='purple', kde=True)
facet.set_axis_labels('Ahorro', 'Frecuencia')
facet.figure.suptitle('FacetGrid: Distribución de Ahorro por Grupo',fontsize=16, color='darkblue', y=0.90)
plt.show()

























# Seaborn es una herramienta poderosa para la visualización de datos en Python.
# Cubre una amplia gama de gráficos, desde distribuciones y correlaciones hasta gráficos categóricos y matrices.
# Con estas herramientas, se pueden explorar y comunicar insights de manera efectiva en proyectos de ciencia de datos.

# --- 12 & 13: Actividad Práctica : Generar Un Heatmap para Visualizar la Correlación Entre Variables ---
# Esta sección describe una actividad práctica para generar un heatmap utilizando Seaborn.
# El objetivo es visualizar la correlación entre variables en un conjunto de datos.

# Requerimientos de la actividad:
# 1. Importar Librerías
# 2. Crear un Conjunto de Datos
# 3. Calcular la Matriz de Correlación
# 4. Generar el Heatmap
# 5. Interpretación del Heatmap
# 6. Personalización del Heatmap
# 7. Visualización

# --- Ejemplo práctico: Heatmap de correlación con Seaborn ---
print("\n--- Ejemplo Práctico: Heatmap de Correlación ---")
# 1. Importar librerías (ya importadas arriba)
# 2. Crear un conjunto de datos simulado
np.random.seed(0)
data_heatmap = pd.DataFrame({
    'Edad': np.random.randint(18, 65, 100),
    'Ingresos': np.random.normal(30000, 8000, 100),
    'Gastos': np.random.normal(15000, 4000, 100),
    'Ahorro': np.random.normal(5000, 2000, 100)
})
# 3. Calcular la matriz de correlación
matriz_corr = data_heatmap.corr()
print("Matriz de correlación:\n", matriz_corr)
# 4. Generar el heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_corr, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)
plt.title('Heatmap de Correlación entre Variables')
plt.show()
# 5. Interpretación
print("\nInterpretación: El heatmap muestra la fuerza y dirección de la relación entre variables. Valores cercanos a 1 indican correlación positiva fuerte, cercanos a -1 indican correlación negativa fuerte, y cercanos a 0 indican poca o ninguna correlación.")