"""
Te han contactado de la empresa DataPro Solutions para desarrollar un conjunto de funciones que
realicen cálculos básicos y operaciones matemáticas específicas. La empresa necesita que estas
funciones sean reutilizables para diversos proyectos y se integren con la librería estándar de Python
para realizar cálculos más avanzados.
Requerimientos:
1. Crea una función calcular_area_rectangulo que reciba dos parámetros (largo y ancho) y
retorne el área del rectángulo (1 punto).
2. Crea una función calcular_circunferencia que reciba el radio de un círculo y retorne su
circunferencia. Usa la constante pi del módulo math (2 puntos).
3. Crea una función calcular_promedio que reciba una lista de números y retorne el promedio
(1 punto).
4. Importa la función mean del módulo statistics y úsala en una nueva función
calcular_promedio_avanzado para calcular el promedio de la misma lista de números del
punto anterior (2 puntos).
5. Crea una función generar_numeros_aleatorios que reciba dos parámetros (cantidad y limite),
y retorne una lista de números aleatorios entre 1 y el límite especificado. Usa el módulo
random (2 puntos).
6. Escribe un programa que utilice cada una de las funciones anteriores e imprime los
resultados de cada cálculo (2 puntos).
"""

import math
import statistics
import random

# calcular area rectangulo
def calcular_area_rectangulo(largo, ancho):
    return largo * ancho

# calcular circunferencia
def calcular_circunferencia(radio):
    return 2 * math.pi * radio

# calcular promedio
def calcular_promedio(numeros):
    return sum(numeros) / len(numeros)

# calcular_promedio_avanzado
def calcular_promedio_avanzado(numeros):
    return statistics.mean(numeros)

# generar_numeros_aleatorios
def generar_numeros_aleatorios(cantidad, limite):
    return [random.randint(1, limite) for _ in range(cantidad)]


# iniciar el ciclo
while True:
    print("1. Calcular area del rectangulo")
    print("2. Calcular circunferencia del circulo")
    print("3. Calcular promedio de numeros")
    print("4. Calcular promedio avanzado de numeros")
    print("5. Generar numeros aleatorios")
    print("6. Salir del menu")
    opcion = int(input("Ingrese una opción: "))

    if opcion == 1:
        print(f"area del rectangulo: {calcular_area_rectangulo(3, 5)}")
    elif opcion == 2:
        print(f"circunferencia del circulo: {calcular_circunferencia(20)}")
    elif opcion == 3:
        print(f"promedio de numeros: {calcular_promedio([3, 4, 5, 6])}")
    elif opcion == 4:
        print(f"promedio de numeros: {calcular_promedio_avanzado([3, 4, 5, 6])}")
    elif opcion == 5:
        print(f" numeros aleatorios: {generar_numeros_aleatorios(5, 20)}")
    elif opcion == 6:
        break
    else:
        print("Ha ingresado una opción invalida, por favor ingrese una opcion")
