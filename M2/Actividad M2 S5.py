# 1. Crea una lista productos que contenga al menos cinco nombres de productos.
lista = ['laptop', 'telefono', 'tablet', 'monitor', 'teclado']

# 2. Agrega dos productos más a la lista productos y luego rescata los primeros tres elementos en una nueva lista llamada productos_destacados.
lista.append('mouse')
lista.append('camara')

# slicing [:]
productos_destacados = lista[:3]
print(f'productos_destacados: {productos_destacados}')

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

# 4. Agrega un nuevo producto al diccionario inventario y muestra la cantidad en stock de un producto específico (elige cualquiera de los cinco productos iniciales).
#actualizar producto
inventario['laptop'] = 20
print(f'inventario: {inventario['laptop']}')

# ingresar nuevo producto
inventario['mouse'] = 10
print(f'diccionario inventario: {inventario}')

#actualizar producto
inventario_detallado['laptop']['inventario'] = 20
print(f'inventario detallado laptop: {inventario_detallado['laptop']['inventario']}')

# ingresar nuevo producto
inventario_detallado['mouse'] = {
    'inventario': 10,
    'precio' : 50000
}

print(f'diccionario inventario detallado: {inventario_detallado.get('laptop')}')
print(f'diccionario inventario detallado: {inventario_detallado}')

# 5. Crea una tupla categorías que contenga las categorías de los productos (por ejemplo, “Electrónica”, “Ropa”, “Alimentos”). Rescata y muestra la segunda categoría.
categorias = ('Electronica', 'Ropa', 'Alimentos')
print(f'Segunda categoria: {categorias[1]}') 

# 6. Empaqueta las categorías en una tupla y luego desempaquétalas en variables individuales para que cada categoría quede asignada a una variable.
categoria1, categoria2, categoria3 = categorias
print(f'Categoria 1 {categoria1}')
print(f'Cateogiria 2: {categoria2}')

# 7. Crea un set productos_unicos a partir de la lista productos y verifica que no existan elementos duplicados.
lista.append('laptop')
productos_unicos = set(lista)
print(f'productos_unicos: {productos_unicos}')

# 8. Usa comprensión de listas para crear una lista productos_mayusculas que contenga los nombres de productos en mayúsculas.

# [temp for temp in lista]
productos_mayusculas = [_.upper() for _ in lista]
print(f'productos_mayusculas: {productos_mayusculas}')