# SESIÓN 3: EXTRAYENDO TABLAS DE LA WEB CON PANDAS
"""
En esta sección aprenderás a extraer tablas directamente desde páginas web usando la función pd.read_html() de Pandas.
"""

import pandas as pd

# 1. Leer todas las tablas de una página web
url = 'https://es.wikipedia.org/wiki/Anexo:Pa%C3%ADses_por_poblaci%C3%B3n'
tablas = pd.read_html(url)
print(f"Se encontraron {len(tablas)} tablas en la página.")

# 2. Seleccionar una tabla específica (por ejemplo, la primera)
tabla_paises = tablas[0]
print(tabla_paises.head())

# 3. Guardar la tabla extraída en un archivo CSV
tabla_paises.to_csv('paises_poblacion.csv', index=False)

# NOTA: pd.read_html requiere que la página tenga tablas HTML bien formateadas y que tengas instalado lxml:
# pip install lxml

# Puedes explorar otras páginas web con tablas públicas para practicar.
