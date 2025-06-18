"""
Un instituto de investigación internacional está analizando datos de migración del siglo XXI para 
identificar tendencias y factores socioeconómicos que afectan los movimientos de población. Se ha 
entregado un dataset en formato CSV con información sobre migraciones entre distintos países, 
incluyendo cantidad de migrantes, razones de migración y variables económicas. 
El instituto necesita que analices, transformes y prepares los datos para extraer información útil que 
ayude en la toma de decisiones. 
Objetivo 
Evaluar la capacidad del estudiante para manejar datos utilizando NumPy y Pandas, aplicando 
técnicas de limpieza, transformación, agrupamiento y combinación de datos en un contexto realista. 
INSTRUCCIONES 
La actividad se divide en 5 secciones, cada una con su propio puntaje. Debes realizar cada tarea en 
Python usando NumPy y Pandas. 
1. Limpieza y Transformación de Datos (3 puntos) 
• Carga el dataset en un DataFrame de Pandas. 
• Identifica y trata valores perdidos en el dataset. 
• Detecta y filtra outliers usando el método del rango intercuartílico (IQR). 
• Reemplaza los valores de la columna "Razon_Migracion" usando mapeo de valores 
(ejemplo: "Económica" → "Trabajo", "Conflicto" → "Guerra"). 
2. Análisis Exploratorio (2 puntos) 
• Muestra las 5 primeras filas del dataset. 
• Obtén información general del dataset con .info() y .describe(). 
• Calcula estadísticas clave: 
o Media y mediana de la cantidad de migrantes. 
o PIB promedio de los países de origen y destino: Usa .value_counts() para contar 
cuántos movimientos de migración ocurrieron por cada razón. 
3. Agrupamiento y Sumarización de Datos (2 puntos) 
• Agrupa los datos por "Razon_Migracion" y calcula la suma total de migrantes para cada 
categoría. 
• Obtén la media del IDH de los países de origen por cada tipo de migración. 
• Ordena el DataFrame de mayor a menor cantidad de migrantes. 
4. Filtros y Selección de Datos (2 puntos) 
• Filtra y muestra solo las migraciones por conflicto. 
• Selecciona y muestra las filas donde el IDH del país de destino sea mayor a 0.90. 
• Crea una nueva columna "Diferencia_IDH" que calcule la diferencia de IDH entre país de 
origen y destino. 
5. Exportación de Datos (1 punto) 
• Guarda el DataFrame final en un nuevo archivo CSV llamado "Migracion_Limpio.csv", sin 
el índice.
"""

import pandas as pd

#1. Limpieza y Transformación de Datos:
#1.1 Carga el dataset en un DataFrame de Pandas.
df = pd.read_csv("C:\\Users\\danna\\OneDrive\\Documentos\\GitHub\\data-science\\M3\\migracion.csv")
print(f"1.1 Dataframe: \n {df.head(10)} \n")

#1.2 Identifica y trata valores perdidos en el dataset.
print(f"1.2 Valores perdidos: \n {df.isnull().sum()} \n")

#1.3 Detecta y filtra outliers usando el método del rango intercuartílico (IQR). 
Q1 = df["Cantidad_Migrantes"].quantile(0.25)
Q3 = df["Cantidad_Migrantes"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

valores_sobre_q3 = df["Cantidad_Migrantes"] >= upper_bound
print(f"1.3 Valores outliers sobre el Q3: {valores_sobre_q3} \n")

valores_bajo_q1 = df["Cantidad_Migrantes"] <= lower_bound
print(f"1.3 Valores outilers bajo el Q1: {valores_bajo_q1} \n")

df_sin_outliers = df[
    (df["Cantidad_Migrantes"] >= lower_bound) &
    (df["Cantidad_Migrantes"] <= upper_bound)
]

print(f"1.3 Data Frame sin outliers: \n {df_sin_outliers} \n")

#1.4 Reemplaza los valores de la columna "Razon_Migracion" usando mapeo de valores (ejemplo: "Económica" → "Trabajo", "Conflicto" → "Guerra"). 

df['Razon_Migracion'] = df['Razon_Migracion'].replace(['Conflicto', 'Económica', 'Educativa'], ['Guerra', 'Trabajo', 'Estudios'])
print(f"1.4 Data Frame con mapeo de valores en razon de migración: \n {df} \n")

#2. Análisis Exploratorio:
#2.1 Muestra las 5 primeras filas del dataset. 
print(f"2.1 5 primeras filas de Dataframe: \n {df.head(5)} \n")

#2.2 Obtén información general del dataset con .info() y .describe(). 
print(f"2.2 Info de data frame: \n")
df.info()
print(f"\n 2.2 Estadistica de data frame: \n {df.describe()} \n")

#2.3 Calcula estadísticas clave: 
# Media y mediana de la cantidad de migrantes. 
# PIB promedio de los países de origen y destino: Usa .value_counts() para contar cuántos movimientos de migración ocurrieron por cada razón. 
media_migrantes = df['Cantidad_Migrantes'].mean()
print(f"2.3 Promedio de migrantes: \n {media_migrantes} \n")

mediana_migrantes = df['Cantidad_Migrantes'].median()
print(f"2.3 Mediana de migrantes: {mediana_migrantes} \n")

media_pib_origen = df['PIB_Origen'].mean()
print(f"2.3 Promedio PIB paises de origen: {media_pib_origen} \n")

media_pib_destino = df['PIB_Destino'].mean()
print(f"2.3 Promedio PIB paises de origen: {media_pib_destino} \n")

print(f"2.3 Movimientos de migración por razón: \n {df['Razon_Migracion'].value_counts()} \n")

#3. Agrupamiento y Sumarización de Datos (2 puntos) 
#3.1 Agrupa los datos por "Razon_Migracion" y calcula la suma total de migrantes para cada categoría.
print(f"3.1 Cantidad de migrantes por razon de migracion: \n {df.groupby("Razon_Migracion")["Cantidad_Migrantes"].sum()} \n")

#3.2 Obtén la media del IDH de los países de origen por cada tipo de migración. 
print(f"3.1 Promedio de IDH de pais de origen por razon de migracion: \n {df.groupby("Razon_Migracion")["IDH_Origen"].mean()} \n")

#3.3 Ordena el DataFrame de mayor a menor cantidad de migrantes. 
df_ascend = df.sort_values(by='Cantidad_Migrantes', ascending=False)
print(f"3.3 Data Frame ordenado por cantidad de migrantes en forma ascendente: \n {df_ascend} \n")

#4. Filtros y Selección de Datos (2 puntos) 
#4.1 Filtra y muestra solo las migraciones por conflicto. 
df_filtrado_guerra = df[df["Razon_Migracion"] == "Guerra"]
print(f"4.1 Data Frame filtrado por Guerra: \n {df_filtrado_guerra} \n")

#4.2 Selecciona y muestra las filas donde el IDH del país de destino sea mayor a 0.90. 
df_filtrado_IDH_dest = df[df['IDH_Destino'] > 0.90]
print(f"4.2 Data Frame filtrado por IDH de pais de destino > 0.90: \n {df_filtrado_IDH_dest} \n")

#4.3 Crea una nueva columna "Diferencia_IDH" que calcule la diferencia de IDH entre país de origen y destino. 
df['Diferencia_IDH'] = df['IDH_Destino'] - df['IDH_Origen']
print(f"4.3 Data Frame con Diferencia IDH: \n {df} \n")

#5. Exportación de Datos (1 punto) 
#• Guarda el DataFrame final en un nuevo archivo CSV llamado "Migracion_Limpio.csv", sin el índice.
df.to_csv("C:\\Users\\danna\\OneDrive\\Documentos\\GitHub\\data-science\\M3\\Migracion_Limpio.csv", index=False)