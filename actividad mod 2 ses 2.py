# Crear 3 variables en Python
precio_producto = 1200
cantidad = int(input("Ingresa la cantidad de productos que deseas:"))
descuento = 20  # Almacena un descuento en porcentaje

# Calcular el precio total sin descuento
total_sin_descuento = precio_producto * cantidad

# Calcular el monto de descuento
monto_descuento = (total_sin_descuento * descuento) / 100

# Calcular el precio total con descuento
total_con_descuento = total_sin_descuento - monto_descuento

# Imprime los resultados de cada cálculo con mensajes claros
print(f"Total sin descuento: {total_sin_descuento}")
print(f"Monto de descuento: {int(monto_descuento)}")
print(f"Total con descuento: {int(total_con_descuento)}")
