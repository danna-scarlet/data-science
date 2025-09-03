"""
1.Análisis Exploratorio de Datos (2puntos)
    •Carga el dataset en un DataFrame de Pandas.
    •Muestra las primeras 5 filas y usa .info() para obtener información sobre los datos.
    •Calcula estadísticas descriptivas con .describe().
    •Genera un histograma del número de entrenamientos semanales.
2.Estadística Descriptiva (2 puntos)
    •Determina el tipo de variable de cada columna.
    •Calcula la media, mediana y moda de la cantidad de medallas obtenidas.
    •Calcula la desviación estándar de la altura de los atletas.
    •Identifica valores atípicos en la columna de peso utilizando un boxplot.
3.Análisis de Correlación (2 puntos)
    •Calcula la correlación de Pearson entre entrenamientos semanales y medallas totales.
    •Crea un gráfico de dispersión (scatterplot) entre peso y medallas totales con Seaborn.
    •Explica si existe correlación entre estas variables.
4.Regresión Lineal (2 puntos)
    •Implementa  un  modelo  de  regresión  lineal  para  predecir  el  número  de  medallas obtenidas en función del número de entrenamientos semanales.
    •Obtén los coeficientes de regresión e interpreta el resultado.
    •Calcula el R² para medir el ajuste del modelo.
    •Usa Seaborn (regplot) para graficar la regresión lineal.
5.Visualización de Datos con Seaborn y Matplotlib (2 puntos)
    •Crea un heatmap de correlación entre todas las variables numéricas.
    •Crea un boxplot de la cantidad de medallas por disciplina deportiva.
    •Personaliza los gráficos con títulos, etiquetas y colores.
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.linear_model import LinearRegression

#1.Análisis Exploratorio de Datos 
# Carga el dataset en un DataFrame de Pandas
df = pd.read_csv("C:\\Users\\danna\\OneDrive\\Documentos\\GitHub\\data-science\\M4\\Evaluación final\\olimpicos.csv")

# Muestra las primeras 5 filas y usa .info() para obtener información sobre los datos.
print('\n-----Análisis exploratorio de datos-----\n')
print(f'\nPrimeras 5 filas del dataset: \n')
print(df.head())
print(f'\nInformación sobre los datos del dataset: \n')
print(df.info())

# Calcula estadísticas descriptivas con .describe().
print(f'\nEstadísticas descriptivas del dataset: \n')
print(df.describe())

# Genera un histograma del número de entrenamientos semanales.
plt.figure(figsize=(10, 6))
sns.histplot(df['Entrenamientos_Semanales'], bins=30)
plt.title('Histograma: Cantidad de atletas v/s entrenamientos Semanales')
plt.xlabel('Entrenamientos Semanales')
plt.ylabel('Cantidad de atletas')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

#2.Estadística Descriptiva
# Determina el tipo de variable de cada columna.
print('\n-------Estadística Descriptiva-------\n')
print(f'\nTipo de variable de cada columna: \n')
print(df.dtypes)

# Calcula la media, mediana y moda de la cantidad de medallas obtenidas.
print("\nMedia de medallas obtenidas:", df['Medallas_Totales'].mean())
print("Mediana de medallas obtenidas:", df['Medallas_Totales'].median())
print("Moda de medallas obtenidas:", df['Medallas_Totales'].mode()[0])

# Calcula la desviación estándar de la altura de los atletas.
print("\nDesviación estándar de la altura de los atletas:", df['Altura_cm'].std())

# Identifica valores atípicos en la columna de peso utilizando un boxplot.
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Peso_kg'])
plt.title('Boxplot: Peso de atletas')
plt.xlabel('Peso (kg)')
Q1 = df['Peso_kg'].quantile(0.25)
Q2 = df['Peso_kg'].quantile(0.50)
Q3 = df['Peso_kg'].quantile(0.75)
plt.text(x=Q1, y=0.45, s=f"Q1: {Q1:.2f}", color='black', ha='center')
plt.text(x=Q2, y=0.45, s=f"Q2: {Q2:.2f}", color='black', ha='center')
plt.text(x=Q3, y=0.45, s=f"Q3: {Q3:.2f}", color='black', ha='center')
plt.show()

#3.Análisis de Correlación (2 puntos)
# Calcula la correlación de Pearson entre entrenamientos semanales y medallas totales.
print('\n-------Análisis de Correlación-------\n')
print("\nCorrelación de Pearson entre entrenamientos semanales y medallas totales:",
      df['Entrenamientos_Semanales'].corr(df['Medallas_Totales']))
      
# Crea un gráfico de dispersión (scatterplot) entre peso y medallas totales con Seaborn.
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Peso_kg', y='Medallas_Totales', data=df)
plt.title('Scatterplot entre Peso y Medallas Totales')
plt.xlabel('Peso (kg)')
plt.ylabel('Medallas Totales')
plt.show()

# Explica si existe correlación entre estas variables.
print("\nExplicación de la correlación:")
if df['Entrenamientos_Semanales'].corr(df['Medallas_Totales']) > 0:
    print("Existe una correlación positiva entre entrenamientos semanales y medallas totales.\n")
elif df['Entrenamientos_Semanales'].corr(df['Medallas_Totales']) < 0:
    print("Existe una correlación negativa entre entrenamientos semanales y medallas totales.\n")
else:
    print("No existe correlación o existe una correlación muy débil entre entrenamientos semanales y medallas totales.\n")

#4.Regresión Lineal (2 puntos)
# Implementa  un  modelo  de  regresión  lineal  para  predecir  el  número  de  medallas obtenidas en función del número de entrenamientos semanales.
print('\n-------Regresión Lineal-------\n')
X = df[['Entrenamientos_Semanales']]
y = df['Medallas_Totales']
model = LinearRegression()
model.fit(X, y)

# Obtén los coeficientes de regresión e interpreta el resultado.
print("\nCoeficientes de regresión:", model.coef_)
print("\nIntercepto de regresión:", model.intercept_)
print(f"\nInterpretación de resultado: Por cada entrenamiento semanal adicional, se espera que el número de medallas totales aumente en {model.coef_[0]}.")

# Calcula el R² para medir el ajuste del modelo.
r_squared = model.score(X, y)
print("\nR²:", r_squared)

# Usa Seaborn (regplot) para graficar la regresión lineal.
plt.figure(figsize=(10, 6))
sns.regplot(x='Entrenamientos_Semanales', y='Medallas_Totales', data=df)
plt.title('Regresión Lineal: Entrenamientos Semanales vs Medallas Totales')
plt.xlabel('Entrenamientos Semanales')
plt.ylabel('Medallas Totales')
plt.show()

#5.Visualización de Datos con Seaborn y Matplotlib (2 puntos)
# Crea un heatmap de correlación entre todas las variables numéricas.
print('\n-------Visualización de Datos-------\n')
plt.figure(figsize=(12, 8))
numericas = df.select_dtypes(include=[np.number])
sns.heatmap(numericas.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap de Correlación entre variables numéricas')
plt.show()

# Crea un boxplot de la cantidad de medallas por disciplina deportiva.
plt.figure(figsize=(12, 6))
sns.boxplot(x='Deporte', y='Medallas_Totales', data=df, palette='Set2')
plt.title('Boxplot: Medallas por Disciplina Deportiva')
plt.xlabel('Disciplina Deportiva')
plt.ylabel('Medallas Totales')
plt.xticks(rotation=45)
plt.show()

