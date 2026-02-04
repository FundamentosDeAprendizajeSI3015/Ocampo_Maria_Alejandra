import pandas as pd
import numpy as np

# =========================================================
# TITANIC - LIMPIEZA Y PREPARACIÓN DE DATOS (PANDAS)
# =========================================================
# IMPORTANTE:
# - La limpieza es crucial porque el modelo aprende de los datos.
# - Si hay NaN, errores, duplicados o tipos incorrectos, el modelo aprende mal (GIGO).
# - Objetivo: dejar un dataset consistente para análisis y/o modelado.
# =========================================================

# 1) CARGAR EL CONJUNTO DE DATOS
RUTA_CSV = "train.csv"   # <-- Cambia esto por tu ruta real
df = pd.read_csv(RUTA_CSV)

print("\n======================")
print("1) Dataset cargado ✅")
print("======================")

# 2) COMANDOS DE INSPECCIÓN RÁPIDA
print("\n--- Inspección rápida ---")
print("Head:\n", df.head())
print("\nTail:\n", df.tail())
print("\nShape (filas, columnas):", df.shape)

print("\nInfo:")
print(df.info())

print("\nTipos de datos (dtypes):")
print(df.dtypes)

print("\nDescribe (numéricas):")
print(df.describe())

print("\nValores únicos por columna:")
print(df.nunique())

print("\n======================")
print("2) Inspección hecha ✅")
print("======================")


# 3) MANEJO DE VALORES FALTANTES (NaNs) + VERIFICACIÓN
print("\n--- NaNs por columna ---")
nulos_por_col = df.isnull().sum()
print(nulos_por_col)

print("\n% de NaN por columna:")
print((df.isnull().sum() / len(df)) * 100)

# Comentario importante:
# En Titanic típicamente hay NaN en Age, Cabin y Embarked.
# - Cabin suele tener muchos NaN -> normalmente se elimina o se convierte en "Unknown".
# - Age se puede imputar (mediana por clase/sexo, o mediana general).
# - Embarked tiene pocos NaN -> se rellena con la moda.

# Rellenar Embarked con moda (si existe)
if "Embarked" in df.columns:
    if df["Embarked"].isnull().sum() > 0:
        moda_embarked = df["Embarked"].mode()[0]
        df["Embarked"] = df["Embarked"].fillna(moda_embarked)

# Imputar Age con mediana (más robusto que media si hay outliers)
if "Age" in df.columns:
    if df["Age"].isnull().sum() > 0:
        mediana_age = df["Age"].median()
        df["Age"] = df["Age"].fillna(mediana_age)

# Cabin: convertir NaN a "Unknown" (no lo borro para que puedas usarlo si quieres)
if "Cabin" in df.columns:
    df["Cabin"] = df["Cabin"].fillna("Unknown")

# Verificación después del manejo
print("\n--- Verificación NaNs después de imputar ---")
print(df.isnull().sum())

print("\n======================")
print("3) NaNs manejados ✅")
print("======================")


# 4) MANIPULACIÓN DE FILAS Y COLUMNAS
# Ejemplo: crear columnas útiles y eliminar columnas poco útiles para ML
# (No es obligatorio eliminar, pero es típico en Titanic.)

# Crear columna "FamilySize" si existen SibSp y Parch
if {"SibSp", "Parch"}.issubset(df.columns):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Crear columna "IsAlone"
if "FamilySize" in df.columns:
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Renombrar columnas (opcional, ejemplo)
# df.rename(columns={"PassengerId": "ID"}, inplace=True)

print("\n--- Columnas actuales ---")
print(df.columns)

print("\n======================")
print("4) Manipulación hecha ✅")
print("======================")


# 5) LIMPIEZA DE TEXTO Y MANEJO DE DUPLICADOS
# Limpieza de texto: espacios extra, mayúsculas inconsistentes
# (En Titanic, 'Name', 'Ticket', 'Cabin', 'Embarked' pueden beneficiarse)

def limpiar_texto(col):
    return col.astype(str).str.strip()

for col in ["Name", "Ticket", "Cabin", "Embarked", "Sex"]:
    if col in df.columns:
        df[col] = limpiar_texto(df[col])

# Duplicados:
# Titanic normalmente no debería tener duplicados por PassengerId,
# pero verificamos por seguridad.
print("\n--- Duplicados totales ---")
print(df.duplicated().sum())

# Si existe PassengerId, verificamos duplicados por esa llave
if "PassengerId" in df.columns:
    dup_id = df.duplicated(subset=["PassengerId"]).sum()
    print("Duplicados por PassengerId:", dup_id)
    if dup_id > 0:
        df.drop_duplicates(subset=["PassengerId"], inplace=True)

print("\n======================")
print("5) Texto y duplicados listos ✅")
print("======================")


# 6) CONSISTENCIA Y VALIDACIÓN LÓGICA
# Reglas típicas de consistencia en Titanic:
# - Age no debería ser negativa ni absurdamente alta
# - Fare no debería ser negativo
# - Pclass debería estar en {1,2,3}

if "Age" in df.columns:
    df = df[(df["Age"] >= 0) & (df["Age"] <= 100)]  # regla razonable

if "Fare" in df.columns:
    df = df[df["Fare"] >= 0]

if "Pclass" in df.columns:
    df = df[df["Pclass"].isin([1, 2, 3])]

print("\n--- Verificación rápida de rangos ---")
if "Age" in df.columns:
    print("Age min/max:", df["Age"].min(), df["Age"].max())
if "Fare" in df.columns:
    print("Fare min/max:", df["Fare"].min(), df["Fare"].max())
if "Pclass" in df.columns:
    print("Clases presentes:", df["Pclass"].unique())

print("\n======================")
print("6) Validación lógica ✅")
print("======================")


# 7) TRANSFORMACIÓN DE TIPOS Y FILTRADO
# Convertir tipos si es necesario:
# Ej: Survived a int, Pclass a category, Sex/Embarked a category

for col in ["Survived", "Pclass", "SibSp", "Parch"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)

# Ejemplo de filtrado: solo mayores de edad (si el profe lo pide)
# df_adultos = df[df["Age"] >= 18]

print("\n--- Tipos después de transformación ---")
print(df.dtypes)

print("\n======================")
print("7) Tipos y filtrado ✅")
print("======================")


# 8) AGREGACIÓN Y AGRUPAMIENTO (groupby / pivot)
# Ejemplos útiles para Titanic: tasa de supervivencia por sexo y clase

if {"Survived", "Sex"}.issubset(df.columns):
    surv_by_sex = df.groupby("Sex")["Survived"].mean()
    print("\n--- Supervivencia promedio por Sex ---")
    print(surv_by_sex)

if {"Survived", "Pclass"}.issubset(df.columns):
    surv_by_class = df.groupby("Pclass")["Survived"].mean()
    print("\n--- Supervivencia promedio por Pclass ---")
    print(surv_by_class)

# Tabla pivote (más “tipo Excel”)
if {"Survived", "Sex", "Pclass"}.issubset(df.columns):
    pivot = df.pivot_table(index="Sex", columns="Pclass", values="Survived", aggfunc="mean")
    print("\n--- Pivot: supervivencia promedio (Sex x Pclass) ---")
    print(pivot)

print("\n======================")
print("8) Agregación hecha ✅")
print("======================")


# 9) ONE HOT ENCODING (al menos una)
# Se aplica a variables categóricas: Sex, Embarked (y otras si quieres)

cols_categoricas = []
for col in ["Sex", "Embarked"]:
    if col in df.columns:
        cols_categoricas.append(col)

df_encoded = pd.get_dummies(df, columns=cols_categoricas, drop_first=True)

print("\n--- Columnas después de One Hot Encoding ---")
print(df_encoded.columns)

print("\n======================")
print("9) One Hot Encoding ✅")
print("======================")


# FINAL: guardar versión limpia (opcional)
df_encoded.to_csv("titanic_limpio_encoded.csv", index=False)
print("\nArchivo guardado: titanic_limpio_encoded.csv ✅")
