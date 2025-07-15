"""
Una tienda de tecnología ha registrado sus ventas en un archivo llamado ventas.csv. Cada fila del
archivo representa una venta con las siguientes columnas:

REQUERIMIENTOS:
1. Cargar el archivo CSV en un DataFrame (2 puntos).
2. Mostrar las primeras 5 filas del archivo (1 punto).
3. Extraer solo las columnas "Producto" y "Precio" (2 puntos).
4. Filtrar los productos cuyo precio sea mayor a 50 (2 puntos).
5. Guardar el DataFrame filtrado en un nuevo archivo CSV (3 puntos).
"""

import pandas as pd

ruta = "M3/ventas.csv"

#1. Cargar el archivo CSV en un DataFrame 
df = pd.read_csv(ruta)
print(f"1. Dataframe cargado correctamente \n {df} \n")

#2. Mostrar las primeras 5 filas del archivo
print(f"2. Primeras 5 filas: \n {df.head()} \n")

#3. Extraer solo las columnas "Producto" y "Precio" 
producto_precio = df[["Producto", "Precio"]]
print(f"3. Extraer columnas Producto y Precio: \n {producto_precio} \n")

#4. Filtrar los productos cuyo precio sea mayor a 50 
df_filtrado = df[df["Precio"]>50] 
print(f"4. Productos con precio mayor a 50: \n {df_filtrado} \n")

#5. Guardar el DataFrame filtrado en un nuevo archivo CSV
df_filtrado.to_csv("M3/ventas_filtrado.csv", index=False)