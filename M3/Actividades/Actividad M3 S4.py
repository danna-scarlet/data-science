"""
Imagina que eres parte del equipo de análisis de datos de una tienda en línea. El gerente te ha enviado 
un archivo CSV llamado ventas.csv, el cual contiene información sobre las ventas realizadas en la 
tienda. Sin embargo, debido a errores en el proceso de recolección de datos, el archivo tiene algunas 
inconsistencias y problemas que deben corregirse antes de que se pueda utilizar para el análisis. 
Tu tarea será aplicar las técnicas que has visto para: 
• Identificar y manejar valores perdidos. 
• Detectar y eliminar valores duplicados. 
• Filtrar y corregir outliers. 
• Reemplazar valores incorrectos. 
• Modificar la estructura del DataFrame. 
REQUERIMIENTOS: 
1. Cargar el archivo CSV y visualizar la información (2 puntos). 
2. Identificar y manejar valores perdidos (2 puntos). 
3. Detectar y eliminar registros duplicados (2 puntos). 
4. Detectar y manejar outliers en la columna "Cantidad" (2 puntos). 
5. Reemplazar valores incorrectos y modificar la estructura del DataFrame (2 puntos).
"""
from pathlib import Path
import pandas as pd 

def read_csv(file_name):
    #Funcion para lectura de csv con pandas
    path = Path(__file__).parent / file_name
    return pd.read_csv(path)

def println(text_init, value, text_end=""):
    print(f"{text_init}: \n {value} {text_end} \n")

#1. Cargar el archivo CSV y visualizar la información
df = read_csv("ventas 1.csv")

print(f"1.1 Data frame de ventas: \n {df} \n")
print(f"1.2 Información del data frame: \n {df.info()} \n")
print(f"1.3 Estadisticas del data frame: \n {df.describe()} \n")
print(f"1.4 5 primeras filas del data frame: \n {df.head()} \n")
print(f"1.5 Tipos de datos del data frame: \n {df.dtypes} \n")

#2. Identificar y manejar valores perdidos
missing_values = df.isnull().sum()
print(f"2.1 Valores perdidos del dataframe: \n {missing_values} \n")

filas_missing_values = df[df.isnull().any(axis=1)]
println("2.2 Filas con valores perdidos", filas_missing_values)

df_copy = df.copy() #crear copia de data frame para modificar

if df_copy["Precio"].isnull().any():
    df_copy["Precio"] = df_copy["Precio"].fillna(df_copy["Precio"].mode()[0])

println("2.3 Registros duplicados en precio sustituidos con moda", df_copy)

#3. Detectar y eliminar registros duplicados

duplicados = df_copy.duplicated()
println("3.1 Registros duplicados", duplicados)
println("3.2 Suma de registros duplicados", duplicados.sum())

if duplicados.any():
    println("3.3 Duplicados en df_copy", df_copy[duplicados])


df_copy_sin_duplicados = df_copy.drop_duplicates()
println("3.4 df_copy sin duplicados", df_copy_sin_duplicados)

#4. Detectar y manejar outliers en la columna "Cantidad" 

Q1 = df_copy["Cantidad"].quantile(0.25)
Q3 = df_copy["Cantidad"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

valores_sobre_q3 = df_copy["Cantidad"] >= upper_bound
println("4.1 Valores outliers sobre el Q3", valores_sobre_q3)

valores_bajo_q1 = df_copy["Cantidad"] <= lower_bound
println("4.2 Valores outilers bajo el Q1", valores_bajo_q1)

df_copy_sin_outliers = df_copy[
    (df_copy["Cantidad"] >= lower_bound) &
    (df_copy["Cantidad"] <= upper_bound)
]

println("4.3 df_copy sin outliers", df_copy_sin_outliers)

#5. Reemplazar valores incorrectos y modificar la estructura del DataFrame 
