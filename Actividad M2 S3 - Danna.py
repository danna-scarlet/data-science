"""
1. Solicita al usuario que ingrese el nombre y la calificación de un estudiante. Evalúa si la calificación es aprobatoria (nota mayor o igual a 60) 
usando una condición if else. Imprime si el estudiante ha aprobado o no.
2.  Usa un bucle while para permitir la entrada de datos de varios estudiantes, hasta que el
usuario decida salir (2 puntos).
3. Dentro del bucle while, solicita las calificaciones de tres materias diferentes para cada
estudiante (por ejemplo, Matemáticas, Ciencias e Inglés). Calcula el promedio de las tres
notas (2 puntos).
4. Usa una estructura if elif else para evaluar el promedio obtenido y asignar un comentario:
“Excelente” si el promedio es 90 o más, “Bueno” si el promedio está entre 75 y 89, y “Necesita
mejorar” si es menos de 75 (2 puntos).
5. Implementa un bucle for para mostrar el nombre y los comentarios de todos los estudiantes
ingresados (2 puntos).
6. Usa una expresión ternaria para agregar una nota adicional en los comentarios si el
estudiante tiene un promedio de 100 “¡Puntuación perfecta!” (1 punto).
"""
# Inicializar lista vacia para almacenar estudiantes y comentarios
estudiantes = []

while True:
    # Ingresar nombre de estudiante
    nombre = input("Ingresa el nombre del estudiante o 'Salir' para terminar y ver un resumen de los datos ingresados: ")

    # Evaluar opción para salir
    if nombre.upper() == "SALIR" or nombre.lower() == "salir":
        break

    # Solicitar las calificaciones de tres materias diferentes para cada estudiante
    matematica = float(input("Ingrese la calificación para matematicas: "))
    ciencia = float(input("Ingrese la calificación para ciencias: "))
    ingles = float(input("Ingrese la calificación para ingles: "))

    # Calcular promedio de las calificaciones
    promedio = (matematica + ciencia + ingles) / 3
    print(f"El promedio de notas es {promedio}")

    # Evaluar promedio obtenido y asignar un comentario
    if promedio >= 6:
        comentario = "Excelente"
    elif 4 >= promedio < 6:
        comentario = "Bueno"
    else:
        comentario = "Necesita mejorar"

    # Expresión ternaria
    comentario = comentario + " ¡Puntuación perfecta!" if promedio == 7 else comentario

    # Almacenar estudiantes en la lista vacia al comienzo
    estudiantes.append((nombre, promedio, comentario))

# Bucle o ciclo for para mostrar nombres y comentarios de todos los estudiantes ingresados
for estudiante in estudiantes:
    print(f"Estudiante {estudiante[0]} tiene un promedio de nota de {estudiante[1]}, {estudiante[2]}.")
