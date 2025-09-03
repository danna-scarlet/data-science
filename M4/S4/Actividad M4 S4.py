"""
En esta actividad, deberás aplicar regresión lineal simple para modelar la relación entre dos variables
numéricas.  Implementar  el  modelo  en  Python,  calcular  los  coeficientes  de  regresión  y  
evaluar  el desempeño del modelo usando métricas de error.
REQUERIMIENTOS: 
1. Creación de Datos Simulados (3 puntos)
    ●Generar  dos  listas  de  datos  numéricos  simulados  que  representen  variables  relacionadas 
    (por ejemplo, temperatura ambiente y consumo de energía).
    ●Hay que asegurar que los datos tengan cierta variabilidad y una relación lineal aproximada.
2. Implementación del Modelo de Regresión Lineal (3 puntos)
    ●Utilizar la librería scikit-learn para ajustar un modelo de regresión lineal simple.
    ●Obtener e imprimir los coeficientes de la regresión (intercepto y pendiente).
3. Predicción de Valores con el Modelo (2 puntos)
    ●Usar el modelo entrenado para hacer predicciones sobre los datos.
    ●Guardar los valores predichos en una lista y mostrar los primeros 5 resultados.
4. Evaluación del Modelo (2 puntos)
    ●Calcular métricas de error como el Error Cuadrático Medio (MSE) y el Error Absoluto Medio (MAE).
    ●Interpretar los resultados y comentar el ajuste del modelo.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as statsmodels
from pathlib import Path

path = Path(__file__).parent / 'datos_regresion.csv'
df = pd.read_csv(path)

#Informacion sobre dataframe de datos simulados
print(f'\n Información del dataframe: \n')
print(f'{df.info()} \n')
print(f'{df.head(5)} \n')
print(f'{df.describe()} \n')

#Filtracion de outliers (datos atipicos)
def filtrar_outliers(serie):
    Q1 = serie.quantile(0.25) 
    Q3 = serie.quantile(0.75) 
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return (serie >= limite_inferior) & (serie <= limite_superior)

filter_temperatura = filtrar_outliers(df['Temperatura'])
filter_consumo = filtrar_outliers(df['Consumo'])

#Agregar data que queda del filtrado de temperatura y consumo
df = df[filter_temperatura & filter_consumo]

print(f'Cantidad de filas después de eliminar los outliers: {len(df)} \n')

'''
2. Implementación del Modelo de Regresión Lineal
● Utilizar la librería scikit-learn para ajustar un modelo de regresión lineal simple.
● Obtener e imprimir los coeficientes de la regresión (intercepto y pendiente).
'''
X = df[['Temperatura']].values
Y = df[['Consumo']].values

modelo = LinearRegression() 
modelo.fit(X, Y)

#β₀ (intercepto) y β₁ (pendiente)

beta_0 = modelo.intercept_
beta_1 = modelo.coef_[0]

print(f'β₀ (intercepto): {beta_0}') 
print(f'β₁ (pendiente): {beta_1}') 
print(f'El modelo ajusta una recta: Consumo = β₀ + β₁ * Temperatura')


'''
3. Predicción de Valores con el Modelo
● Usar el modelo entrenado para hacer predicciones sobre los datos.
● Guardar los valores predichos en una lista y mostrar los primeros 5 resultados.
'''
consumo_promedio = df['Consumo'].mean()
print(f'\nEl consumo promedio es: {consumo_promedio:.2f} kwh \n')

consumo_prediccion = modelo.predict(X)
print(f'5 primeros valores en la prediccion de consumo aplicando regresion lineal: \n {consumo_prediccion[:5]} \n')

'''
4. Evaluación del Modelo
● Calcular métricas de error como el Error Cuadrático Medio (MSE) y el Error Absoluto Medio (MAE).
● Interpretar los resultados y comentar el ajuste del modelo.
'''

#Calculo del MSE
mse = mean_squared_error(Y, consumo_prediccion)
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
        
#Calculo del MAE
mae = mean_absolute_error(Y, consumo_prediccion)
print(f"Error Absoluto Medio (MAE): {mae:.2f}") 

#Coeficiente de Determinación (R²)
r_2_score = r2_score(Y, consumo_prediccion)
print(f"Coeficiente de determinación R²: {r_2_score:.2f}")

porcentaje_error = (mae / consumo_promedio) * 100
print(f'Porcentaje aproximado de error de MAE: {porcentaje_error:.2f}% en cuanto a consumo promedio \n')


print("--- Implementación con Statsmodels --- \n")
# Usamos los mismos datos de temperatura y consumo para comparar resultados
X_sm = statsmodels.add_constant(df['Temperatura'])  # Agrega columna de 1s para el intercepto
Y_sm = df['Consumo']

# Ajuste del modelo OLS (Ordinary Least Squares)
modelo_sm = statsmodels.OLS(Y_sm, X_sm).fit()

# Análisis de resultados
print("Resumen del Modelo (Statsmodels):")
print(modelo_sm.summary())


'''
5. Visualizacion de datos
'''
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color="blue", label="Datos reales")
plt.plot(X, consumo_prediccion, color="red", label="Regresión Lineal")
plt.xlabel("Temperatura")
plt.ylabel("Consumo")
plt.title("Regresión Lineal Simple")
plt.legend()
plt.grid(True)
plt.show()