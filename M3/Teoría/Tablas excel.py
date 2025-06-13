# SESIÓN 3: TRABAJO CON ARCHIVOS EXCEL EN PANDAS
"""
En esta sección aprenderás a leer, explorar y guardar archivos Excel (.xlsx) usando la librería Pandas en Python.
"""

import pandas as pd

# 1. Leer un archivo Excel
df = pd.read_excel('ruta/al/archivo.xlsx')
# Puedes especificar la hoja con sheet_name
df = pd.read_excel('ruta/al/archivo.xlsx', sheet_name='Hoja1')
#Otros parametros configurables de read excel
df = pd.read_excel("archivo.xlsx", sheet_name="Hoja1", usecols=["Cliente", "Total"], index_col="Cliente")

# 2. Mostrar las primeras filas
def mostrar_primeras_filas(df):
    print(df.head())

# 3. Seleccionar columnas específicas
columnas = ['Columna1', 'Columna2']
df_seleccion = df[columnas]

# 4. Filtrar filas por condición
df_filtrado = df[df['Columna'] > valor]

# 5. Guardar un DataFrame en Excel
df_filtrado = pd.DataFrame({'Ejemplo': [1,2,3]})  # Ejemplo para guardar
df_filtrado.to_excel('ruta/al/nuevo_archivo.xlsx', index=False) #index=False para que no se muestre el indice automatico de python

# 6. Leer todas las hojas de un archivo Excel
hojas = pd.read_excel('ruta/al/archivo.xlsx', sheet_name=None)
for nombre_hoja, df_hoja in hojas.items():
    print(f"Hoja: {nombre_hoja}")
    print(df_hoja.head())

# NOTA: Para trabajar con Excel necesitas instalar openpyxl o xlrd:
# pip install openpyxl