"""
Eres un analista de datos en un club de fútbol que busca mejorar el rendimiento de los jugadores. Te
han proporcionado un archivo con datos sobre los futbolistas del equipo, incluyendo su nombre,
posición, edad, goles y asistencias en la última temporada. Tu tarea es analizar estos datos usando
Pandas para responder preguntas clave.

INSTRUCCIONES:
1. Importa la librería Pandas y crea un DataFrame con los siguientes datos: (1 punto).
2. Selecciona una columna y muestra los nombres de todos los jugadores (1 punto).
3. Filtra jugadores con más de 10 goles y muestra solo su nombre y cantidad de goles (1 punto).
4. Agrega una nueva columna al DataFrame llamada Puntos, donde cada jugador obtiene
Puntos = (Goles * 4) + (Asistencias * 2) (1 punto).
5. Calcula el promedio de goles de todos los jugadores (1 punto).
6. Obtén el máximo y mínimo de asistencias en el equipo (1 punto).
7. Cuenta cuántos jugadores hay por posición (Delantero, Mediocampista) (1 punto).
8. Ordena el DataFrame en función de los goles en orden descendente (1 punto).
9. Aplica describe() para obtener estadísticas generales del DataFrame (1 punto).
10. Usa value_counts() para contar cuántos jugadores hay en cada posición (1 punto).
"""

import pandas as pd

# 1. Importa la librería Pandas y crea un DataFrame
data = {
    'jugador': ['Lionel Messi', 'Cristiano Ronaldo', 'Kevin De Bruyne', 'Kylian Mbappé', 'Luka Modric'],
    'posicion': ['Delantero', 'Delantero', 'Mediocampista', 'Delantero', 'Mediocampista'],
    'edad': [35, 38, 31, 24, 37],
    'goles': [20,18,8,25,3],
    'asistencias': [10,5,15,12,8]
}

df = pd.DataFrame(data)
print(f"1. Dataframe creado: \n {df} \n")

# 2. Selecciona una columna y muestra los nombres de todos los jugadores
print(f"2. Nombre de los jugadores: \n {df['jugador']} \n")

# 3. Filtra jugadores con más de 10 goles y muestra solo su nombre y cantidad de goles.
jugadores_10_goles = df[df['goles'] > 10][['jugador','goles']]
print(f"3. Jugadores que han hecho mas de 10 goles: \n {jugadores_10_goles} \n")

# 4. Agrega una nueva columna al DataFrame llamada Puntos, donde cada jugador obtiene Puntos = (Goles * 4) + (Asistencias * 2).
df['puntos'] = (df['goles'] * 4) + (df['asistencias'] * 2 )
print(f"4. Dataframe con columna Puntos:\n {df} \n")

# 5. Calcula el promedio de goles de todos los jugadores.
promedio_goles = df['goles'].mean()
print(f"5. Promedio de goles de todos los jugadores: {promedio_goles} \n")

# 6. Obtén el máximo y mínimo de asistencias en el equipo.
asist_max = df['asistencias'].max()
asist_min = df['asistencias'].min()
print(f'6. Cantidad máxima de asistencias: {asist_max} \n Cantidad mínima de asistencias: {asist_min} \n')

# 7. Cuenta cuántos jugadores hay por posición (Delantero, Mediocampista).
jugadores_delantero = df["posicion"].value_counts().get("Delantero")
jugadores_mediocampista = df["posicion"].value_counts().get("Mediocampista")
print(f"7. Cantidad de jugadores delanteros: {jugadores_delantero} \n Cantidad de jugadores mediocampistas: {jugadores_mediocampista} \n")

# 8. Ordena el DataFrame en función de los goles en orden descendente.
df_goles_orden_descendent = df.sort_values('goles', ascending = False)
print(f"8. DataFrame ordenado por la cantidad descendente de goles: \n {df_goles_orden_descendent} \n")

# 9. Aplica describe() para obtener estadísticas generales del DataFrame.
print(f"9. Estadisticas generales del DataFrame: \n {df.describe(include='all')} \n")

# 10. Usa value_counts() para contar cuántos jugadores hay en cada posición.
jugadores_por_posicion = df['posicion'].value_counts()
print(f"10. Jugadores por posición: \n {jugadores_por_posicion} \n")