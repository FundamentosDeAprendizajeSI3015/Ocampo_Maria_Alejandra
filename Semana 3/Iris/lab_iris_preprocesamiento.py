import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from scipy import stats

# Importo librerías para cargar el dataset, manejar datos en tabla, hacer partición train/test y escalar variables

# Cargo el dataset Iris directamente desde scikit-learn
print("Cargando dataset Iris...")
iris = load_iris()

# Paso los datos a un DataFrame para poder explorarlos y trabajarlos con pandas
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Agrego la columna target, que representa la clase de cada flor
df["target"] = iris.target

# Muestro las primeras filas para confirmar que todo quedó bien cargado
print("Primeras filas:")
print(df.head())

# Reviso la estructura general del dataset: tipos de datos y cantidad de valores no nulos
print("Información del dataset:")
print(df.info())

# Miro estadísticas básicas para tener una idea rápida de rangos y distribución
print("Descripción estadística:")
print(df.describe())

# Verifico cuántos valores diferentes hay en cada columna
print("Valores únicos por columna:")
print(df.nunique())

# Reviso si hay valores nulos por columna
print("Verificando valores nulos:")
print(df.isna().sum())

# En Iris normalmente no hay nulos, pero dejo el ejemplo de imputación por si uso otro dataset después
if df.isna().sum().sum() > 0:
    df = df.fillna(df.mean(numeric_only=True))

# Hago una detección simple de outliers usando z-score sobre columnas numéricas
# Esto me sirve para identificar registros raros, aunque no los elimino automáticamente
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outlier_mask = (z_scores > 3).any(axis=1)
print(f"Registros detectados como posibles outliers: {outlier_mask.sum()}")

# Si yo quisiera eliminar los outliers, podría aplicar este filtro
# df = df[~outlier_mask]

# Separo variables de entrada (X) y la salida (y)
X = df.drop(columns=["target"])
y = df["target"]

# Escalo las variables para que todas queden en una escala comparable
# Esto es útil para modelos que son sensibles a la magnitud de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convierto el resultado escalado de nuevo a DataFrame para mantener nombres de columnas
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Muestro un ejemplo del resultado del escalado
print("Ejemplo de datos escalados:")
print(X_scaled.head())

# Parto los datos en entrenamiento y prueba
# Uso stratify=y para mantener la proporción de clases en ambos conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Imprimo tamaños para confirmar que la partición quedó bien
print("Tamaños de los conjuntos:")
print("Train:", X_train.shape, " Test:", X_test.shape)

# Creo una carpeta de salida para guardar los archivos procesados
output_dir = Path("./data_output")
output_dir.mkdir(exist_ok=True)

# Defino las rutas de exportación para train y test
train_path = output_dir / "iris_train.parquet"
test_path = output_dir / "iris_test.parquet"

# Guardo train y test en formato parquet incluyendo la columna target
X_train.assign(target=y_train).to_parquet(train_path, index=False)
X_test.assign(target=y_test).to_parquet(test_path, index=False)

# Confirmo los archivos exportados
print("Archivos exportados:")
print(train_path)
print(test_path)

print("Laboratorio finalizado correctamente.")
