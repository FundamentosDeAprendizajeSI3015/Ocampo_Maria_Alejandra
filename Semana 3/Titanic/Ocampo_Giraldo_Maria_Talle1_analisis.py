import pandas as pd
import numpy as np

# Importo pandas para manejar datos en tablas y numpy para trabajar con valores numéricos y NaN

# Creo un diccionario con información básica de algunos países
data = {
    "pais": ["Brasil", "Rusia", "India", "China", "Sudafrica", "Brasil"],
    "capital": ["Brasilia", "Moscu", "Nueva Delhi", "Beijing", "Pretoria", "Brasilia"],
    "area": [8515767, 17098242, np.nan, 9596961, 1221037, 8515767],
    "population": [213000000, 146000000, 1380000000, 1440000000, np.nan, 213000000]
}

# Convierto el diccionario en un DataFrame para poder analizarlo mejor
df = pd.DataFrame(data)

# Muestro el dataset original para ver cómo están los datos inicialmente
print("Dataset inicial:")
print(df)

# Muestro las primeras filas para tener una vista rápida del contenido
print("\nPrimeras filas:")
print(df.head())

# Reviso cuántas filas y columnas tiene el DataFrame
print("\nDimensiones:")
print(df.shape)

# Obtengo información general sobre tipos de datos y valores nulos
print("\nInformación del DataFrame:")
print(df.info())

# Calculo estadísticas básicas como promedio, mínimo y máximo
print("\nEstadísticas descriptivas:")
print(df.describe())

# Reviso cuántos valores nulos hay en cada columna
print("\nValores nulos por columna:")
print(df.isnull().sum())

# Relleno los valores faltantes del área usando el promedio
df["area"].fillna(df["area"].mean(), inplace=True)

# Relleno los valores faltantes de población usando la mediana
df["population"].fillna(df["population"].median(), inplace=True)

# Muestro el dataset después de completar los valores nulos
print("\nDataset tras imputar NaN:")
print(df)

# Cambio el nombre de la columna population a poblacion para usar español
df.rename(columns={"population": "poblacion"}, inplace=True)

# Creo una nueva columna que calcula la densidad poblacional
df["densidad"] = df["poblacion"] / df["area"]

# Muestro el DataFrame con la nueva columna agregada
print("\nDataset con nueva columna:")
print(df)

# Normalizo el texto de la columna país para evitar problemas de mayúsculas o espacios
df["pais"] = df["pais"].str.lower().str.strip()

# Identifico qué filas están duplicadas
print("\nFilas duplicadas:")
print(df.duplicated())

# Elimino las filas duplicadas del DataFrame
df.drop_duplicates(inplace=True)

# Muestro el dataset limpio sin duplicados
print("\nDataset sin duplicados:")
print(df)

# Filtro los países que tienen un área mayor a un millón de km²
df = df[df["area"] > 1_000_000]

# Muestro el DataFrame después del filtrado
print("\nDataset filtrado por área:")
print(df)

# Convierto las columnas numéricas a tipo entero para mayor claridad
df["area"] = df["area"].astype(int)
df["poblacion"] = df["poblacion"].astype(int)

# Verifico los tipos de datos finales
print("\nTipos de datos finales:")
print(df.dtypes)

# Agrupo los datos por país y calculo el promedio de población
promedio_poblacion = df.groupby("pais")["poblacion"].mean()

# Muestro el resultado de la agrupación
print("\nPromedio de población por país:")
print(promedio_poblacion)

# Aplico One Hot Encoding a la columna país para dejarla lista para modelos de ML
df_encoded = pd.get_dummies(df, columns=["pais"])

# Muestro el DataFrame final codificado
print("\nDataset con One Hot Encoding:")
print(df_encoded)
