"""
Eres contratado/a por una pequeña cadena de librerías llamada "Libros & Bytes"para desarrollar un sistema que gestione su inventario y permita a los usuarios simular una compra en línea. 
Trabajarás solo en la lógica del sistema sin preocuparte de la interfaz visual. El sistema debe cumplir con los siguientes requerimientos y funcionalidades.
Requerimientos:
1.Definir variables básicas y tipos de datos(1 punto):
   o Crea  una  lista  que  contenga  al  menos  cinco  libros,  donde  cada  libro  sea  un diccionario  con  los  atributos título(cadena  de  caracteres), autor(cadena  de caracteres), precio(decimal) y cantidad en stock(entero)
2.Control de flujo (1 punto):
   o Implementa una función llamada mostrar_libros_disponibles()que recorra la lista de libros y muestreen pantalla los libros que tienen más de una unidad en stock usando una sentencia fory una condición if.
3.Condiciones y operadores (1 punto):
   o Solicita al usuario que ingrese un rango de precios (mínimo y máximo) y utiliza una sentencia if elif else para  filtrar los libros en el rango ingresado y mostrarlos  en pantalla.
4.Función personalizada para simular una compra (2 puntos):
   o Crea una función comprar_libros(título, cantidad)que reciba como parámetros el título del libro y la cantidad a comprar. La función debe:
      ▪Verificar  si  el  libro  está  en  el  inventario  y  si  la  cantidad  deseada  está disponible.
      ▪Si la compra es válida, restar la cantidad comprada al stock y mostrar el monto total de la compra.
      ▪Si la cantidad solicitada es mayor al stock disponible, mostrar unmensaje de error.
5.Uso de bucle while para iterar hasta que el usuario decidasalir(1 punto):
   o Implementa un bucle whileque permita al usuario realizar múltiples compras hasta que ingrese una opción de salida.
6.Estructura de datos, gestión de descuentos (2 puntos):
   o Usa un diccionario para almacenar descuentos especiales por autor. Por ejemplo, aplica un 10% de descuento en libro de un autor especifico. 
   o En la función comprar_libro, certifica si el autor tiene descuento y aplícaloal monto total si corresponde.
7.Simulación de una factura (2punto):
   o Al finalizar la compra, muestra un resumen con el total de libros comprados, el monto total pagado y el ahorro por descuentos
"""

#1.Definir variables básicas y tipos de datos(1 punto): 
# Crea  una  lista  que  contenga  al  menos  cinco  libros,  donde  cada  libro  sea  un diccionario  con  los  atributos 
# título(cadena  de  caracteres), autor(cadena  de caracteres), precio(decimal) y cantidaden stock(entero)
libros = [
    {'titulo': 'El Principito', 'autor': 'Antoine de Saint-Exupéry', 'precio': 10.99, 'cantidad': 5},
    {'titulo': '1984', 'autor': 'George Orwell', 'precio': 8.99, 'cantidad': 2},
    {'titulo': 'Cien años de soledad', 'autor': 'Gabriel García Márquez', 'precio': 12.99, 'cantidad': 0},
    {'titulo': 'Don Quijote de la Mancha', 'autor': 'Miguel de Cervantes', 'precio': 14.99, 'cantidad': 3},
    {'titulo': 'La Odisea', 'autor': 'Homero', 'precio': 9.99, 'cantidad': 4}
]

#2.Control de flujo (1 punto): Implementa una función llamada mostrar_libros_disponibles() que recorra la lista de libros y 
# muestre en pantalla los libros que tienen más de una unidad en stock usando una sentencia for y una condición if.
def mostrar_libros_disponibles(libros):
    for libro in libros:
        if libro['cantidad'] > 1:
            print(f"Disponible: {libro['titulo']} por {libro['precio']} USD (Stock: {libro['cantidad']})")

#3.Condiciones y operadores (1 punto): Solicita al usuario que ingrese un rango de precios (mínimo y máximo) y 
# utiliza una sentencia if elif else para filtrar los libros en el rango ingresado y mostrarlos en pantalla.
def filtrar_libros_por_precio(libros):
    precio_min = float(input("Ingrese el precio mínimo para filtrar los libros: "))
    precio_max = float(input("Ingrese el precio máximo para filtrar los libros: "))

    for libro in libros:
        if precio_min <= libro['precio'] <= precio_max:
            print(f"Libro en rango: {libro['titulo']} por {libro['precio']} USD (Stock: {libro['cantidad']})")

#4.Función personalizada para simular una compra (2 puntos): Crea una función comprar_libros(título, cantidad) que reciba como
# parámetros el título del libro y la cantidad a comprar. La función debe:
# Verificar  si  el  libro  está  en  el  inventario  y  si  la  cantidad  deseada  está disponible.
# Si la compra es válida, restar la cantidad comprada al stock y mostrar el monto total de la compra.
# Si la cantidad solicitada es mayor al stock disponible, mostrar un mensaje de error.
def comprar_libros(titulo, cantidad):
    for libro in libros:
        if libro['titulo'] == titulo:
            if libro['cantidad'] >= cantidad:
                libro['cantidad'] = libro['cantidad']-cantidad
                total = libro['precio'] * cantidad
                print(f"Compra realizada: {cantidad} x {titulo} por un total de {total} USD")
            else:
                print(f"Stock insuficiente para {titulo}. Disponible: {libro['cantidad']}")
            return
    print(f"Libro no encontrado: {titulo}")

#5.Uso de bucle while para iterar hasta que el usuario decidasalir(1 punto):
#   o Implementa un bucle while que permita al usuario realizar múltiples compras hasta que ingrese una opción de salida.
def realizar_compras():
    while True:
        titulo = input("Ingrese el título del libro que desea comprar (o 'salir' para finalizar): ")
        if titulo.lower() == 'salir':
            break
        cantidad = int(input("Ingrese la cantidad que desea comprar: "))
        comprar_libros(titulo, cantidad)

#6.Estructura de datos, gestión de descuentos (2 puntos):
#   o Usa un diccionario para almacenar descuentos especiales por autor. Por ejemplo, aplica un 10% de descuento en libro de un autor especifico.
descuentos_autores = {
    'Gabriel García Márquez': 0.2,
    'Miguel de Cervantes': 0.15
}
#   o En la función comprar_libros, certifica si el autor tiene descuento y aplícalo al monto total si corresponde.
def comprar_libros_con_descuento(titulo, cantidad):
    for libro in libros:
        if libro['titulo'] == titulo: 
            if libro['cantidad'] >= cantidad:
                libro['cantidad'] = libro['cantidad'] - cantidad
                total = libro['precio'] * cantidad
                descuento = descuentos_autores.get(libro['autor'], 0)
                total_con_descuento = total * (1 - descuento)
                ahorro = total - total_con_descuento
                print(f"Compra realizada: {cantidad} x {titulo} por un total de {total_con_descuento} USD (Ahorro: {ahorro} USD)")
            else:
                print(f"Stock insuficiente para {titulo}. Disponible: {libro['cantidad']}")
            return
    print(f"Libro no encontrado: {titulo}")

#7. Simulación de una factura (2punto):
#   o Al finalizar la compra, muestra un resumen con el total de libros comprados, el monto total pagado y el ahorro por descuentos
def simular_compra():
    total_libros = 0
    total_pagado = 0
    total_ahorrado = 0

    while True:
        titulo = input("Ingrese el título del libro que desea comprar (o 'salir' para finalizar): ")
        if titulo.lower() == 'salir' or titulo.upper() == 'SALIR':
            break
        cantidad = int(input("Ingrese la cantidad que desea comprar: "))
        for libro in libros:
            if libro['titulo'].lower() == titulo or libro['titulo'].upper() == titulo:
                if libro['cantidad'] >= cantidad:
                    libro['cantidad'] -= cantidad
                    total = libro['precio'] * cantidad
                    descuento = descuentos_autores.get(libro['autor'], 0)
                    total_con_descuento = total * (1 - descuento)
                    ahorro = total - total_con_descuento
                    
                    total_libros += cantidad
                    total_pagado += total_con_descuento
                    total_ahorrado += ahorro

                    print(f"Compra realizada: {cantidad} x {titulo} por un total de {total_con_descuento} USD (Ahorro: {ahorro} USD)")
                else:
                    print(f"Stock insuficiente para {titulo}. Disponible: {libro['cantidad']}")
                break
        else:
            print(f"Libro no encontrado: {titulo}")

    print(f"\nResumen de la compra:")
    print(f"Total de libros comprados: {total_libros}")
    print(f"Monto total pagado: {total_pagado}")
    print(f"Ahorro por descuentos: {total_ahorrado}")

#Menú para generar sistema de compras
print("Bienvenido a Libros & Bytes")
print("1. Mostrar libros disponibles")
print("2. Filtrar libros por precio")
print("3. Comprar libros y mostrar factura final")
print("4. Salir")

while True:
    opcion = input("Seleccione una opción: ")
    if opcion == "1":
        mostrar_libros_disponibles(libros)
    elif opcion == "2":
        filtrar_libros_por_precio(libros)
    elif opcion == "3":
        simular_compra()
    elif opcion == "4":
        print("Gracias por visitar Libros & Bytes")