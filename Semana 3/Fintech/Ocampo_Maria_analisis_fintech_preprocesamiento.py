import argparse
from pathlib import Path
import json
import warnings

# Importo librerías estándar para manejo de argumentos, rutas, archivos JSON y control de advertencias
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importo librerías principales para manipulación de datos y preprocesamiento

# Función que intenta leer un CSV usando el encoding indicado
# Si falla por encoding, reintenta automáticamente con latin-1
def try_read_csv(path, sep, encoding):
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except UnicodeDecodeError:
        print("[WARN] Problema de encoding. Reintentando con 'latin-1'.")
        return pd.read_csv(path, sep=sep, encoding="latin-1")

# Función para convertir columnas a tipo numérico cuando sea posible
# Se usa errors="ignore" para no dañar columnas realmente categóricas
def coerce_numeric(df, cols=None):
    if cols is None:
        cols = df.select_dtypes(include=["object", "string"]).columns
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

# Función para aplicar winsorización a columnas numéricas
# Limita los valores extremos usando cuantiles definidos
def winsorize_df(df, numeric_cols, lower_q=0.01, upper_q=0.99):
    for c in numeric_cols:
        lo = df[c].quantile(lower_q)
        hi = df[c].quantile(upper_q)
        df[c] = df[c].clip(lo, hi)
    return df

# Función que detecta outliers usando el criterio IQR
# Retorna una máscara booleana para cada valor fuera del rango esperado
def iqr_outlier_mask(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)

# Función auxiliar solo para imprimir títulos de sección de forma clara
def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# Función para convertir argumentos tipo texto separados por coma en listas limpias
def parse_list_arg(arg):
    if arg is None or len(arg.strip()) == 0:
        return []
    return [a.strip() for a in arg.split(",") if a.strip()]

# Configuro los argumentos que se pasan por línea de comandos
# Esto hace el script reutilizable con distintos datasets financieros
parser = argparse.ArgumentParser(
    description="Laboratorio de preprocesamiento para dataset financiero personalizado"
)

parser.add_argument("--input", required=True, help="Ruta al CSV de entrada")
parser.add_argument("--sep", default=",", help="Separador del CSV")
parser.add_argument("--encoding", default="utf-8", help="Encoding del archivo")

parser.add_argument("--date-col", default=None, help="Columna de fecha")
parser.add_argument("--id-cols", default="", help="Columnas identificadoras")
parser.add_argument("--categorical-cols", default="", help="Columnas categóricas")
parser.add_argument("--numeric-cols", default="", help="Columnas numéricas")
parser.add_argument("--price-cols", default="", help="Columnas de precio")
parser.add_argument("--target-col", default=None, help="Columna objetivo")

parser.add_argument("--missing-tokens", default="NA,N/A,na,NaN,?,-999,", help="Tokens de valores faltantes")
parser.add_argument("--time-split", action="store_true", help="Activar partición temporal")
parser.add_argument("--split-date", default=None, help="Fecha de corte para time split")
parser.add_argument("--test-size", type=float, default=0.2, help="Tamaño del conjunto de prueba")

parser.add_argument("--winsorize", nargs=2, type=float, default=None, help="Cuantiles para winsorización")
parser.add_argument("--outdir", default="./data_output_finanzas", help="Directorio de salida")

args = parser.parse_args()

# Inicio del proceso de carga del archivo CSV
print_section("1) CARGA DEL CSV")

input_path = Path(args.input)
if not input_path.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {input_path}")

# Leo el archivo CSV manejando posibles problemas de encoding
missing_values = parse_list_arg(args.missing_tokens)
df = try_read_csv(input_path, sep=args.sep, encoding=args.encoding)

# Limpio espacios en columnas de texto y reemplazo tokens de missing por NaN
obj_cols = df.select_dtypes(include=["object", "string"]).columns
for c in obj_cols:
    df[c] = df[c].astype(str).str.strip()
    df[c] = df[c].replace({tok: np.nan for tok in missing_values})

print("Shape:", df.shape)
print("Columnas:", list(df.columns))

# Tipificación de fechas si el usuario la define
print_section("2) TIPIFICACIÓN Y FECHAS")

if args.date_col and args.date_col in df.columns:
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")
    print(f"Nulos en fecha:", df[args.date_col].isna().sum())
else:
    print("No se especificó columna de fecha.")

# Defino listas de columnas según argumentos o inferencia automática
id_cols = parse_list_arg(args.id_cols)
cat_cols_user = parse_list_arg(args.categorical_cols)
num_cols_user = parse_list_arg(args.numeric_cols)
price_cols = parse_list_arg(args.price_cols)

if not num_cols_user:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
else:
    num_cols = [c for c in num_cols_user if c in df.columns]

if not cat_cols_user:
    exclude = set(id_cols + ([args.date_col] if args.date_col else []) + ([args.target_col] if args.target_col else []))
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude]
else:
    cat_cols = [c for c in cat_cols_user if c in df.columns]

print("Numéricas:", num_cols)
print("Categóricas:", cat_cols)

# Exploración inicial del dataset
print_section("3) EXPLORACIÓN INICIAL")

print(df.info())
print("\nDescribe (numéricos):")
print(df[num_cols].describe().T)

print("\nValores nulos por columna:")
print(df.isna().sum().sort_values(ascending=False).head(20))

print("\nValores únicos:")
print(df.nunique().sort_values(ascending=False).head(10))

# Limpieza e imputación de valores faltantes
print_section("4) LIMPIEZA Y OUTLIERS")

for c in df.columns:
    if c != args.date_col:
        df[c] = pd.to_numeric(df[c], errors="ignore")

for c in num_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

for c in cat_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna("__MISSING__")

# Detección de outliers usando IQR solo para reporte
outlier_report = {}
for c in num_cols:
    mask = iqr_outlier_mask(df[c].astype(float))
    outlier_report[c] = int(mask.sum())

print("Outliers detectados por columna:")
print(outlier_report)

# Aplico winsorización solo si el usuario lo solicita
if args.winsorize:
    lq, uq = args.winsorize
    df[num_cols] = winsorize_df(df[num_cols], num_cols, lq, uq)

# Ingeniería básica de variables financieras
print_section("5) INGENIERÍA DE RASGOS")

return_cols = []
logret_cols = []

if price_cols:
    if args.date_col:
        sort_cols = id_cols + [args.date_col] if id_cols else [args.date_col]
        df = df.sort_values(sort_cols)

    for pc in price_cols:
        if pc in df.columns:
            if id_cols:
                df[pc + "_ret"] = df.groupby(id_cols)[pc].pct_change()
                df[pc + "_logret"] = np.log1p(df[pc + "_ret"])
            else:
                df[pc + "_ret"] = df[pc].pct_change()
                df[pc + "_logret"] = np.log1p(df[pc + "_ret"])

            return_cols.append(pc + "_ret")
            logret_cols.append(pc + "_logret")

for c in return_cols + logret_cols:
    df[c] = df[c].fillna(0.0)

num_cols = sorted(set(num_cols + return_cols + logret_cols))

# Separación de variables y escalado sin fuga de información
print_section("6) SEPARACIÓN Y ESCALADO")

if args.target_col and args.target_col in df.columns:
    y = df[args.target_col]
    X = df.drop(columns=[args.target_col])
else:
    y = None
    X = df.copy()

for c in id_cols + ([args.date_col] if args.date_col else []):
    if c in X.columns:
        X = X.drop(columns=[c])

X = pd.get_dummies(X, columns=[c for c in cat_cols if c in X.columns], drop_first=True)

scaler = StandardScaler()
num_in_X = [c for c in num_cols if c in X.columns]

# Partición temporal o aleatoria según configuración
if args.time_split and args.date_col:
    cutoff = pd.to_datetime(args.split_date)
    train_idx = df[args.date_col] < cutoff
    X_train, X_test = X[train_idx], X[~train_idx]
    y_train, y_test = y[train_idx], y[~train_idx] if y is not None else (None, None)
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y if y is not None else None
    )

X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X])
X_test[num_in_X] = scaler.transform(X_test[num_in_X])

print("Shapes finales:", X_train.shape, X_test.shape)

# Exportación de resultados
print_section("7) EXPORTACIÓN")

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

def attach_target(Xdf, yser):
    return Xdf if yser is None else Xdf.assign(**{args.target_col: yser})

train_df = attach_target(X_train, y_train)
test_df = attach_target(X_test, y_test)

train_df.to_parquet(outdir / "finance_train.parquet", index=False)
test_df.to_parquet(outdir / "finance_test.parquet", index=False)

# Guardo un diccionario de datos para documentar el preprocesamiento
schema = {
    "source_file": str(input_path),
    "date_col": args.date_col,
    "id_cols": id_cols,
    "categorical_cols": cat_cols,
    "numeric_cols": num_cols,
    "price_cols": price_cols,
    "target_col": args.target_col,
    "time_split": args.time_split,
    "test_size": args.test_size,
    "train_shape": X_train.shape,
    "test_shape": X_test.shape,
}

with open(outdir / "finance_data_dictionary.json", "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2, ensure_ascii=False)

print("Laboratorio de finanzas finalizado correctamente.")
