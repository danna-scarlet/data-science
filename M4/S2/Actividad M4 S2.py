"""
Deberás trabajar con un conjunto de datos ficticio para calcular estadísticas descriptivas clave y
visualizar la distribución de los datos. Se utilizarán herramientas como Pandas, NumPy y Matplotlib
para realizar el análisis.
Requerimientos:
1. Definir Variables (2 puntos)
● Identificar el tipo de variable (categórica, cuantitativa discreta o continua) de cada
columna en el conjunto de datos.
2. Construcción de una Tabla de Frecuencia (2 puntos)
● Generar una tabla de frecuencia para una variable categórica y otra para una variable
cuantitativa discreta.
3. Cálculo de Medidas de Tendencia Central (2 puntos)
● Calcular la media, mediana y moda de una variable cuantitativa.
4. Cálculo de Medidas de Dispersión (2 puntos)
● Calcular el rango, varianza y desviación estándar de la misma variable utilizada en el
punto anterior.
5. Visualización de Datos (2 puntos)
Crear un histograma para una variable cuantitativa y un boxplot para otra variable del
dataset
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos ficticio
df = pd.read_csv("C:\\Users\\danna\\OneDrive\\Documentos\\GitHub\\data-science\\M4\\datos_estadistica.csv")
print(df.head(10))
print(df.info())

# 1. Definir Variables
# Identificar el tipo de variable de cada columna

# ID | Cuantitativa discreta
# Nombre | Cualitativa Categórica
# Edad | Cuantitativa discreta
# Ingresos | Cuantitativa continua
# Género | Cualitativa Categórica
# Ciudad | Cualitativa Categórica

# 2. Construcción de una Tabla de Frecuencia
# Tabla de frecuencia para una variable categórica (Género)
tabla_frecuencia_genero = df["Genero"].value_counts().sort_index().to_frame("Frecuencia Absoluta")
tabla_frecuencia_genero["Frecuencia Relativa"] = tabla_frecuencia_genero["Frecuencia Absoluta"] / tabla_frecuencia_genero["Frecuencia Absoluta"].sum()
tabla_frecuencia_genero["Frecuencia Acumulada"] = tabla_frecuencia_genero["Frecuencia Absoluta"].cumsum()
tabla_frecuencia_genero["Frecuencia Relativa Acumulada"] = tabla_frecuencia_genero["Frecuencia Relativa"].cumsum()
print(f'Tabla de frecuencia para variable género: \n {tabla_frecuencia_genero} \n')

# Tabla de frecuencia para una variable cuantitativa discreta (Edad)
tabla_frecuencia_edad = df["Edad"].value_counts().sort_index().to_frame("Frecuencia Absoluta")
tabla_frecuencia_edad["Frecuencia Relativa"] = tabla_frecuencia_edad["Frecuencia Absoluta"] / tabla_frecuencia_edad["Frecuencia Absoluta"].sum()
tabla_frecuencia_edad["Frecuencia Acumulada"] = tabla_frecuencia_edad["Frecuencia Absoluta"].cumsum()
tabla_frecuencia_edad["Frecuencia Relativa Acumulada"] = tabla_frecuencia_edad["Frecuencia Relativa"].cumsum()
print(f'Tabla de frecuencia para variable edad: \n {tabla_frecuencia_edad} \n')

# 3. Cálculo de Medidas de Tendencia Central
# Calcular la media, mediana y moda de la variable Ingresos 
media = df['Ingresos'].mean()
mediana = df['Ingresos'].median()
moda = df['Ingresos'].mode()[0]

print(f'Media de la variable ingresos: {media:.2f}')
print(f'Mediana de la variable ingresos: {mediana:.2f}')
print(f'Moda de la variable ingresos: {moda:.2f}')

# 4. Cálculo de Medidas de Dispersión
# Calcular el rango, varianza y desviación estándar de la variable Ingresos 
categoria = df['Ingresos']
rango = categoria.max() - categoria.min()
varianza = categoria.var()
desviacion_estandar = categoria.std()
print(f'Rango de la variable ingresos: {rango}')
print(f'Varianza de la variable ingresos: {varianza}')
print(f'Desviación estándar de la variable ingresos: {desviacion_estandar}')

#5. Visualización de Datos
# Histograma para la variable Ingresos  
plt.hist(df["Ingresos"], bins=5, edgecolor="black")
plt.xlabel("Ingresos")
plt.ylabel("Frecuencia")
plt.title("Histograma de Ingresos")
plt.show()

# Boxplot para la variable Edad
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Edad'], color='skyblue')
plt.title('Boxplot de Edad', fontsize=16, fontweight='bold')
plt.xlabel('Edad', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.grid(axis='y', color='gray', linestyle='dashed', linewidth=0.5)
plt.tight_layout()
plt.show()