"""
Imagina que estás trabajando como analista de datos en una empresa de tecnología y necesitas
procesar información numérica utilizando NumPy. Para ello, debes desarrollar un conjunto de
operaciones básicas para crear y manipular datos en forma de arreglos.
Instrucciones:
1. Importar la librería NumPy (1 punto).
2. Crear un vector de 10 elementos con valores del 1 al 10 utilizando arange() (1 punto).
• Muestra el vector generado.
3. Generar una matriz de 3x3 con valores aleatorios entre 0 y 1 usando random.rand() (1
punto).
• Muestra la matriz en pantalla.
4. Crear una matriz identidad de tamaño 4x4 utilizando eye() (1 punto).
• Muestra la matriz identidad.
5. Redimensionar el vector creado en el punto 2 en una matriz de 2x5 usando .reshape() (1
punto).
• Muestra la nueva matriz.
6. Seleccionar los elementos mayores a 5 del vector original y mostrarlos (1 punto).
• Utiliza indexación condicional.
7. Realizar una operación matemática entre arreglos (2 puntos).
• Crea dos arreglos de tamaño 5 con arange() y súmalos.
• Muestra el resultado.
8. Aplicar una función matemática a un arreglo (2 puntos).
• Calcula la raíz cuadrada de los elementos del vector original.
• Muestra el resultado.
"""
#1. Importar la librería NumPy
import numpy as np

#2. Crear un vector de 10 elementos con valores del 1 al 10 utilizando arange(). Muestra el vector generado.
vector_original = np.arange(1,11)
print(f"\n2. Vector Original: {vector_original} \n") 

#3. Generar una matriz de 3x3 con valores aleatorios entre 0 y 1 usando random.rand(). Muestra la matriz en pantalla.
matriz_aleatoria = np.random.rand(3,3)
print(f'3. Matriz aleatoria: \n {matriz_aleatoria} \n')

#4. Crear una matriz identidad de tamaño 4x4 utilizando eye() (1 punto). Muestra la matriz identidad.
matriz_identidad = np.eye(4)
print(f'4. Matriz identidad: \n {matriz_identidad} \n')

#5. Redimensionar el vector creado en el punto 2 en una matriz de 2x5 usando .reshape(). Muestra la nueva matriz.
vector_redimensionado = vector_original.reshape(2,5)
print(f'5. Matriz redimensionada: \n {vector_redimensionado} \n')

#6. Seleccionar los elementos mayores a 5 del vector original y mostrarlos. Utiliza indexación condicional.
vector_indexado = vector_original[vector_original > 5]
print(f'6. Vector indexado: \n {vector_indexado} \n')

#7. Realizar una operación matemática entre arreglos. Crea dos arreglos de tamaño 5 con arange() y súmalos. Muestra el resultado.
vector_1 = np.arange(1,6) # [1,2,3,4,5]
vector_2 = np.arange(6,11)# [6,7,8,9,10]
print(f'7. Suma entre arreglos: \n {vector_1 + vector_2} \n')

#8. Aplicar una función matemática a un arreglo. Calcula la raíz cuadrada de los elementos del vector original. Muestra el resultado.
raiz_cuadrada = np.sqrt(vector_original)
print(f'8. Raiz cuadrada del vector original: \n {raiz_cuadrada} \n')