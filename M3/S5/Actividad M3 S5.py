"""
En esta actividad, deberás resolver los siguientes requerimientos, los cuales te permitirán practicar y
consolidar tus conocimientos sobre la manipulación de datos con Pandas. A través de estos
ejercicios, explorarás diferentes técnicas avanzadas para trabajar con indexación jerárquica,
agrupamiento, pivoteo, despivoteo, y combinación de DataFrames.
REQUERIMIENTOS:
1. Crear un DataFrame con Indexación Jerárquica (2 puntos)
• Crea un DataFrame que contenga las siguientes columnas: "Estudiante", "Materia",
"Calificación".
• Utiliza los siguientes datos de ejemplo para crear el DataFrame:
Estudiante Materia Calificación
Juan Matemáticas 6.5
Juan Historia 5.8
María Matemáticas 4.2
María Historia 6.0
2. Acceder a datos con Indexación Jerárquica (1 punto)
• Consulta la clasificación de María en Historia.
3. Agrupar y Agregar Datos con groupby (2 puntos)
• Agrupa el DataFrame por "Materia" y calcula:
o El promedio de calificaciones por materia.
o La calificación más alta por materia.
4. Pivoteo de DataFrame (2 puntos)
• Convierte el DataFrame para que:
o Las filas representen a los estudiantes.
o Las columnas representen las materias.
o Las celdas contengan sus calificaciones.
5. Despivoteo de DataFrame con melt (1 punto)
• Aplica la función melt para transformar el DataFrame pivoteado a su formato largo.
6. Concatenación y Merge de DataFrames (2 puntos)
• Crea dos DataFrames:
o df1 con las columnas "ID_Estudiante", "Estudiante", "Carrera"
o df2 con las columnas "ID_Estudiante", "Materia", "Calificación"
• Concatena ambos DataFrames a lo largo del eje de filas.
• Luego, realiza un merge de ambos DataFrames basado en la columna
"ID_Estudiante".
"""

import pandas as pd

# 1. Crear un DataFrame con Indexación Jerárquica: 
# 1.1 Crea un DataFrame que contenga las siguientes columnas: "Estudiante", "Materia", "Calificación".
# 1.2 Utiliza los siguientes datos de ejemplo para crear el DataFrame:
#Estudiante Materia Calificación
#Juan Matemáticas 6.5
#Juan Historia 5.8
#María Matemáticas 4.2
#María Historia 6.0

df = pd.read_csv("M3\Actividades\calificaciones_enriquecido.csv")
print(f"Dataframe: \n {df.head(10)} \n")

multi_index_df = df.set_index(["Estudiante", "Materia"])
print(f"multi_index_df: \n {multi_index_df.head(10)} \n")

#2. Acceder a datos con Indexación Jerárquica (1 punto): Consulta la clasificación de María en Historia.
calificacion_maria_df = multi_index_df.loc[("Maria", "Historia"), "Calificacion"]
print(f"Calificación de Maria en Historia: \n {calificacion_maria_df} \n")

#3. Agrupar y Agregar Datos con groupby (2 puntos) Agrupa el DataFrame por "Materia" y calcula:
#3.1 El promedio de calificaciones por materia.
#3.2 La calificación más alta por materia.

df_agrupado_mean = df.groupby("Materia")["Calificacion"].mean()
print(f"Promedio de calificaciones por materia: \n {df_agrupado_mean} \n")

df_agrupado_max = df.groupby("Materia")["Calificacion"].max()
print(f"Calificacion mas alta por materia: \n {df_agrupado_max} \n")

#4. Pivoteo de DataFrame: Convierte el DataFrame para que:
#o Las filas representen a los estudiantes.
#o Las columnas representen las materias.
#o Las celdas contengan sus calificaciones.
df_pivot = df.pivot_table(values="Calificacion", index="Estudiante", columns="Materia", aggfunc="mean")
print(f"Data Frame Pivoteado: \n {df_pivot} \n")

df_reset = df_pivot.reset_index()

#5. Despivoteo de DataFrame con melt 
#• Aplica la función melt para transformar el DataFrame pivoteado a su formato largo.
df_melt = pd.melt(df_reset, id_vars=['Estudiante'], var_name='Materia', value_name='Calificacion')
print(f"Data Frame despivoteado: \n {df_melt} \n")

#6. Concatenación y Merge de DataFrames 
#• Crea dos DataFrames:
#6.1 df1 con las columnas "ID_Estudiante", "Estudiante", "Carrera"
df1 = df[['ID_Estudiante','Estudiante', 'Carrera']]
print(f'6.1 Data Frame df1: \n {df1} \n')

#6.2 df2 con las columnas "ID_Estudiante", "Materia", "Calificación"
df2 = df[['ID_Estudiante','Materia', 'Calificacion']]
print(f'6.2 Data Frame df2: \n {df2} \n')

#6.3 Concatena ambos DataFrames a lo largo del eje de filas.
df_concat = pd.concat([df1,df2], axis=0, ignore_index=True)
print(f'6.3 Data Frame concatenado: \n {df_concat} \n')

#6.4 Luego, realiza un merge de ambos DataFrames basado en la columna "ID_Estudiante".
df_merged = pd.merge(df1, df2, on='ID_Estudiante')
print(f'6.4 Data Frame merge: \n {df_merged} \n')