import json
from pathlib import Path
import warnings

# Importo librerías básicas para leer JSON, manejar rutas y evitar mensajes de advertencia innecesarios
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importo numpy y pandas para manipular datos, y herramientas de sklearn para escalado y preparación de conjuntos

# Defino los nombres de los archivos esperados y la carpeta de salida
# La idea es que este script funcione sin pasar argumentos, solo ejecutándolo
DATA_CSV = "fintech_top_sintetico_2025.csv"
DATA_DICT = "fintech_top_sintetico_dictionary.json"
OUTDIR = Path("./data_output_finanzas_sintetico")

# Defino una fecha de corte para hacer una partición por tiempo
# Todo lo anterior va a train y lo posterior va a test
SPLIT_DATE = "2025-09-01"

# Estas son las columnas que espero encontrar porque así fue diseñado el dataset sintético
DATE_COL = "Month"
ID_COLS = ["Company"]

# Columnas categóricas (se transforman con one-hot encoding)
CAT_COLS = ["Country", "Region", "Segment", "Subsegment", "IsPublic", "Ticker"]

# Columnas numéricas principales que se usan para análisis y escalado
NUM_COLS = [
    "Users_M", "NewUsers_K", "TPV_USD_B", "TakeRate_pct", "Revenue_USD_M",
    "ARPU_USD", "Churn_pct", "Marketing_Spend_USD_M", "CAC_USD", "CAC_Total_USD_M",
    "Close_USD", "Private_Valuation_USD_B"
]

# Columna(s) de precio que se usan para crear retornos y log-retornos
PRICE_COLS = ["Close_USD"]

# Empiezo cargando el diccionario de datos para validar que el archivo existe y para imprimir contexto del dataset
print("\n=== 0) Cargando diccionario de datos ===")
dict_path = Path(DATA_DICT)

# Si el archivo JSON no existe, detengo el programa para evitar errores más adelante
if not dict_path.exists():
    raise FileNotFoundError(
        f"No se encontró {DATA_DICT}. Asegúrate de tener el archivo en la misma carpeta."
    )

# Leo el JSON del diccionario de datos para conocer descripción y periodo
with open(dict_path, "r", encoding="utf-8") as f:
    data_dict = json.load(f)

print("Descripción:", data_dict.get("description", "(sin descripción)"))
print("Periodo:", data_dict.get("period", "(desconocido)"))

# Cargo el CSV del dataset sintético desde el mismo directorio
print("\n=== 1) Cargando CSV sintético ===")
csv_path = Path(DATA_CSV)

# Valido que el CSV exista para evitar fallos de lectura
if not csv_path.exists():
    raise FileNotFoundError(
        f"No se encontró {DATA_CSV}. Asegúrate de tener el archivo en la misma carpeta."
    )

df = pd.read_csv(csv_path)
print("Shape:", df.shape)

# Verifico que la columna de fecha exista porque la necesito para ordenar y para el split temporal
if DATE_COL not in df.columns:
    raise KeyError(f"La columna de fecha '{DATE_COL}' no existe en el CSV.")

# Convierto la columna de fecha a datetime y ordeno por fecha y empresa para mantener consistencia temporal
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.sort_values([DATE_COL] + ID_COLS).reset_index(drop=True)

print("Primeras filas:")
print(df.head(3))

# Hago una exploración rápida para entender estructura del dataset y revisar si hay valores faltantes
print("\n=== 2) EDA rápido ===")
print("Info:")
print(df.info())

print("\nNulos por columna (top 15):")
print(df.isna().sum().sort_values(ascending=False).head(15))

# Aplico limpieza mínima: imputación simple para evitar problemas en escalado y modelos
print("\n=== 3) Limpieza ===")

# Para columnas numéricas uso la mediana porque es más robusta ante valores extremos
for c in NUM_COLS:
    if c in df.columns and df[c].isna().any():
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

# Para columnas categóricas uso un marcador fijo para no perder información por NaN
for c in CAT_COLS:
    if c in df.columns and df[c].isna().any():
        df[c] = df[c].fillna("__MISSING__")

# Creo variables nuevas basadas en precio: retornos y log-retornos
# Esto es típico en datos financieros porque captura variación relativa y estabilidad con log
print("\n=== 4) Ingeniería de rasgos (retornos) ===")

if all([pc in df.columns for pc in PRICE_COLS]):
    for pc in PRICE_COLS:
        # Calculo el retorno porcentual por empresa siguiendo el orden temporal
        df[pc + "_ret"] = (
            df.sort_values([ID_COLS[0], DATE_COL])
              .groupby(ID_COLS)[pc]
              .pct_change()
        )

        # Calculo el log-retorno con log(1 + retorno) para evitar problemas cuando retorno es pequeño
        df[pc + "_logret"] = np.log1p(df[pc + "_ret"])

        # Los primeros valores de pct_change quedan en NaN, así que los relleno con 0.0
        df[pc + "_ret"] = df[pc + "_ret"].fillna(0.0)
        df[pc + "_logret"] = df[pc + "_logret"].fillna(0.0)
else:
    print("[INFO] Columnas de precio no disponibles; se omite cálculo de retornos.")

# Actualizo la lista de numéricos usados incluyendo las columnas nuevas de retornos
extra_num = [
    c for c in
    [pc + "_ret" for pc in PRICE_COLS] + [pc + "_logret" for pc in PRICE_COLS]
    if c in df.columns
]
NUM_USED = [c for c in NUM_COLS if c in df.columns] + extra_num

# Preparo X para modelado: elimino fecha e identificadores porque no son variables predictoras directas
print("\n=== 5) Preparación de X: codificación one-hot y escalado ===")
X = df.drop(columns=[DATE_COL] + ID_COLS, errors="ignore").copy()

# Aplico one-hot encoding a variables categóricas para convertirlas en columnas binarias
# drop_first=True deja una categoría como referencia y evita columnas redundantes
cat_in_X = [c for c in CAT_COLS if c in X.columns]
X = pd.get_dummies(X, columns=cat_in_X, drop_first=True)

# Hago partición temporal usando la fecha de corte definida en SPLIT_DATE
cutoff = pd.to_datetime(SPLIT_DATE)
idx_train = df[DATE_COL] < cutoff
idx_test = df[DATE_COL] >= cutoff

X_train, X_test = X.loc[idx_train].copy(), X.loc[idx_test].copy()

# Escalo solo las columnas numéricas para que queden en la misma escala
# El scaler se ajusta solo en train para evitar fuga de datos hacia test
num_in_X = [c for c in NUM_USED if c in X_train.columns]
scaler = StandardScaler()

if num_in_X:
    X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X])
    X_test[num_in_X] = scaler.transform(X_test[num_in_X])
else:
    print("[INFO] No se encontraron columnas numéricas para escalar.")

print("Shapes -> X_train:", X_train.shape, " X_test:", X_test.shape)

# Exporto los resultados en parquet y guardo documentación de lo que hice
print("\n=== 6) Exportación ===")

OUTDIR.mkdir(parents=True, exist_ok=True)

train_path = OUTDIR / "fintech_train.parquet"
test_path = OUTDIR / "fintech_test.parquet"

# Guardo solo X porque este dataset no tiene objetivo por defecto
X_train.to_parquet(train_path, index=False)
X_test.to_parquet(test_path, index=False)

# Guardo un esquema procesado para dejar evidencia del preprocesamiento y facilitar reproducibilidad
processed_schema = {
    "source_csv": str(csv_path.resolve()),
    "source_dict": str(dict_path.resolve()),
    "date_col": DATE_COL,
    "id_cols": ID_COLS,
    "categorical_cols_used": cat_in_X,
    "numeric_cols_used": num_in_X,
    "engineered_cols": extra_num,
    "split": {
        "type": "time_split",
        "cutoff": SPLIT_DATE,
        "train_rows": int(idx_train.sum()),
        "test_rows": int(idx_test.sum()),
    },
    "X_train_shape": list(X_train.shape),
    "X_test_shape": list(X_test.shape),
    "notes": [
        "Dataset 100% sintético con fines académicos; no representa métricas reales.",
        "Evité fuga de datos: el escalador se ajusta en TRAIN y se aplica a TEST."
    ]
}

with open(OUTDIR / "processed_schema.json", "w", encoding="utf-8") as f:
    json.dump(processed_schema, f, ensure_ascii=False, indent=2)

# Guardo el listado de columnas finales para tener claro qué features usar al entrenar modelos
with open(OUTDIR / "features_columns.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(X_train.columns))

print("\nArchivos exportados:")
print(" -", train_path)
print(" -", test_path)
print(" -", OUTDIR / "processed_schema.json")
print(" -", OUTDIR / "features_columns.txt")

print("\n✔ Listo. Recuerda: este dataset es sintético para práctica académica.")
