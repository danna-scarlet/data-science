'''
1. Instalación e importación de librerías 
• Instalar Seaborn y otras librerías necesarias.
• Importar correctamente Seaborn, Pandas y Matplotlib.
2. Creación del conjunto de datos 
• Crear un conjunto de datos simple con al menos 4 variables numéricas (por ejemplo: Edad, Ingresos, Años de Educación, Horas de Sueño).
3. Cálculo de la matriz de correlación 
• Calcular la matriz de correlación entre las variables numéricas del conjunto de datos usando Pandas.
4. Generación del heatmap 
• Utilizar Seaborn para generar el heatmap que visualiza la matriz de correlación.
• Personalizar el gráfico: usar el parámetro annot=True para mostrar los valores en cada celda, y seleccionar una paleta de colores apropiada (cmap='coolwarm').
5. Interpretación y Personalización del gráfico
• Interpretar brevemente los resultados del heatmap.
• Personalizar el gráfico ajustando el tamaño de la figura y añadiendo líneas de borde entre las celdas.
'''

#1. Instalación e importación de librerías 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#2. Creación conjunto de datos
np.random.seed(42)
n = 200

df = pd.DataFrame({
    'Edad': np.random.randint(18, 65, n),
    'Ingresos': np.random.normal(3000, 900, n),
    'Anios_Educacion': np.random.randint(6, 22, n),
    'Horas_Sueno': np.random.normal(7, 1.2, n),
})

#Lectura del archivo
df = pd.read_csv('C:/Users/danna/OneDrive/Documentos/GitHub/data-science/M4/S5/datos_heatmap.csv')
print(f'\nInformación del DataFrame: \n')
print(f'{df.head(5)} \n')
print(f'{df.info()} \n')

# 3. Cálculo de la matriz de correlación 
matriz_correlacion = df.corr()
print(f'Matriz de correlación: \n {matriz_correlacion} \n')

# 4 y 5 Generación del heatmap y personalizacion
plt.figure(figsize=(8, 7))
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', fmt=".4f", linewidths=.5)
plt.title("Heatmap de Matriz de Correlación")
plt.show()

# valores cercanos a 1, relacion positiva
# valores cercanos negativos, relacion negativa
# valores cercanos a 0, relacion null o debil relacion
'''
                     Edad  Ingresos  Anios_Educacion  Horas_Sueno
Edad             1.000000 -0.007947         0.048179     0.119061
Ingresos        -0.007947  1.000000        -0.047773     0.076570
Anios_Educacion  0.048179 -0.047773         1.000000    -0.129600
Horas_Sueno      0.119061  0.076570        -0.129600     1.00000
'''