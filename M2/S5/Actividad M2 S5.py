"""
Eres un trabajador de una tienda en línea que gestiona pedidos y el inventario de productos. Te
solicitan realizar un programa en Python que permita gestionar y organizar los productos en
diferentes estructuras de datos, con el fin de facilitar la búsqueda, organización y agrupación de la
información.
Requerimientos:
1. Crea una lista productos que contenga al menos cinco nombres de productos (1 punto).
2. Agrega dos productos más a la lista productos y luego rescata los primeros tres elementos
en una nueva lista llamada productos_destacados (1 puntos).
3. Crea un diccionario inventario donde cada clave sea el nombre de un producto y el valor sea
la cantidad en stock. Incluye al menos cinco productos con sus cantidades (2 puntos).
4. Agrega un nuevo producto al diccionario inventario y muestra la cantidad en stock de un
producto específico (elige cualquiera de los cinco productos iniciales) (1 punto).
5. Crea una tupla categorías que contenga las categorías de los productos (por ejemplo,
“Electrónica”, “Ropa”, “Alimentos”). Rescata y muestra la segunda categoría (1 punto).
6. Empaqueta las categorías en una tupla y luego desempaquétalas en variables individuales
para que cada categoría quede asignada a una variable (2 puntos).
7. Crea un set productos_unicos a partir de la lista productos y verifica que no existan
elementos duplicados (1 punto).
8. Usa comprensión de listas para crear una lista productos_mayusculas que contenga los
nombres de productos en mayúsculas (1 punto)
"""
# 1. Crea una lista productos que contenga al menos cinco nombres de productos.
lista = ['laptop', 'telefono', 'tablet', 'monitor', 'teclado']
print(f"\n1. Lista de productos: {lista} \n")

# 2. Agrega dos productos más a la lista productos y luego rescata los primeros tres elementos en una nueva lista llamada productos_destacados.
lista.append('mouse')
lista.append('camara')

# slicing [:]
productos_destacados = lista[:3]
print(f"2. Lista de productos actualizada: {lista} \n Lista de productos destacados: {productos_destacados} \n")

# 3. Crea un diccionario inventario donde cada clave sea el nombre de un producto y el valor sea la cantidad en stock. Incluye al menos cinco productos con sus cantidades.
inventario = {
    'laptop': 10,
    'telefono': 20,
    'tablet' : 30,
    'monitor' : 50,
    'teclado' : 70
}

inventario_detallado = {
    'laptop':{
            'inventario': 10,
            'precio': 1500000
        },
    'telefono': {
            'inventario': 10,
            'precio': 700000
        },
    'tablet' : {
            'inventario': 10,
            'precio': 300000
        },
    'monitor' : {
            'inventario': 10,
            'precio': 400000
        },
    'teclado' : {
            'inventario': 10,
            'precio': 100000
        }
}

print (f"3. Diccionario de inventario: \n {inventario} \n Diccionario de inventario detallado: \n {inventario_detallado} \n")

# 4. Agrega un nuevo producto al diccionario inventario y muestra la cantidad en stock de un producto específico (elige cualquiera de los cinco productos iniciales).

# ingresar nuevo producto
inventario['mouse'] = 10

# ingresar nuevo producto
inventario_detallado['mouse'] = {
    'inventario': 10,
    'precio' : 50000
}

#actualizar producto
inventario['laptop'] = 20

#actualizar producto
inventario_detallado['laptop']['inventario'] = 20

print(f'4. Diccionario de inventario actualizado: \n {inventario} \n Diccionario de inventario detallado actualizado: \n {inventario_detallado} \n')

# 5. Crea una tupla categorías que contenga las categorías de los productos (por ejemplo, “Electrónica”, “Ropa”, “Alimentos”). Rescata y muestra la segunda categoría.
categorias = ('Electronica', 'Ropa', 'Alimentos')
print(f'5. Segunda categoria de productos: {categorias[1]} \n') 

# 6. Empaqueta las categorías en una tupla y luego desempaquétalas en variables individuales para que cada categoría quede asignada a una variable.
categoria1, categoria2, categoria3 = categorias
print(f'6. Categoria 1: {categoria1} \n Categoria 2: {categoria2} \n Categoria 3: {categoria3} \n')

# 7. Crea un set productos_unicos a partir de la lista productos y verifica que no existan elementos duplicados.
lista.append('laptop')
productos_unicos = set(lista)
print(f'7. Productos unicos: {productos_unicos} \n')

# 8. Usa comprensión de listas para crear una lista productos_mayusculas que contenga los nombres de productos en mayúsculas.

# [temp for temp in lista]
productos_mayusculas = [_.upper() for _ in lista]
print(f'8. Productos en mayusculas: {productos_mayusculas} \n')