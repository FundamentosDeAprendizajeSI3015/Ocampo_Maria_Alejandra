# -*- coding: utf-8 -*-
"""
Script generado a partir del notebook `chocolate_ml_ciclo_vida.ipynb`.
Objetivo: ejecutar el flujo de análisis/ML como un archivo .py.
"""

# 3. Procesamiento de Datos

#Cargar Datos
import pandas as pd

df = pd.read_csv("flavors_of_cacao.csv")
df.head()
df.columns = (
    df.columns
      .str.replace("\n", " ", regex=False)
      .str.strip()
)

#Inspección inicial
df.info()
df.describe()

# Limpiar los nombres de las columnas
# los nombres originales de las columnas contenían saltos de línea (\n) y espacios irregulares, lo cual:
# Dificulta el acceso a las columnas en el código
# Pandas exige que el nombre sea exactamente igual.

# Selección de variables relevantes

# Variables definidas en la formulación del problema
columns_used = [
    "Cocoa Percent",
    "Bean Type",
    "Broad Bean Origin",
    "Company Location",
    "Rating"
]

# Nos quedamos solo con las variables relevantes
df_proc = df[columns_used].copy()

df_proc.head()

# Limpiar porcentaje de cacao (ej. "70%" → 70)

# Limpieza de datos y valores faltantes

# Convertir Cocoa Percent de texto ("70%") a valor numérico (70.0)
df_proc["Cocoa Percent"] = (
    df_proc["Cocoa Percent"]
      .astype(str)
      .str.replace("%", "", regex=False)
      .astype(float)
)

# Verificar valores faltantes
df_proc.isnull().sum()

# Imputación de valores faltantes
# - Numérico: media
df_proc["Cocoa Percent"] = df_proc["Cocoa Percent"].fillna(df_proc["Cocoa Percent"].mean())

# - Categóricos: etiqueta "Unknown"
for col in ["Bean Type", "Broad Bean Origin", "Company Location"]:
    df_proc[col] = df_proc[col].fillna("Unknown")

df_proc.isnull().sum()

# Creación de la variable objetivo (Y)

# Función para transformar la columna 'quality' en una etiqueta binaria (buena/mala).
# Creación de la variable objetivo (Quality)

def quality_label(rating):
    if rating <= 2.5:
        return 1   # Baja calidad
    elif rating <= 3.75:
        return 2   # Calidad media
    else:
        return 3   # Alta calidad

df_proc["Quality"] = df_proc["Rating"].apply(quality_label)

# Revisar distribución de clases
df_proc["Quality"].value_counts()

# Codificación de variables categóricas

# Codificación de variables categóricas

# X preliminar (sin Rating ni Quality)
X_raw = df_proc.drop(columns=["Rating", "Quality"])

# One-Hot Encoding
X_encoded = pd.get_dummies(
    X_raw,
    columns=["Bean Type", "Broad Bean Origin", "Company Location"],
    drop_first=True
)

X_encoded.head()

# Normalización de variables numéricas

# Normalización de variables numéricas

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_encoded["Cocoa Percent"] = scaler.fit_transform(
    X_encoded[["Cocoa Percent"]]
)

# Detección simple de outliers (análisis, no eliminación)

# Detección de outliers (exploratoria)

try:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.boxplot(df_proc["Cocoa Percent"])
    plt.title("Boxplot de Cocoa Percent")
    plt.ylabel("Porcentaje de cacao")
    plt.savefig("boxplot_cocoa_percent.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: boxplot_cocoa_percent.png")
except Exception as e:
    print(f"⚠ Advertencia al crear gráfico: {e}")

# Definición final de X y Y (listos para ML)

# Definición final de X y Y

X = X_encoded
y = df_proc["Quality"]

# Dimensiones del problema
n, m = X.shape
print(f"X ∈ R^({n} × {m})")
print("y ∈ {1, 2, 3}")

# Visualizaciones exploratorias (EDA ligero)

# Visualización exploratoria

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Distribución de clases
    plt.figure()
    sns.countplot(x=y)
    plt.title("Distribución de niveles de calidad (Quality)")
    plt.savefig("quality_distribution.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: quality_distribution.png")
    
    # Distribución de Cocoa Percent (sin normalizar)
    plt.figure()
    sns.histplot(df_proc["Cocoa Percent"], bins=20)
    plt.title("Distribución del porcentaje de cacao")
    plt.savefig("cocoa_percent_distribution.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: cocoa_percent_distribution.png")
except Exception as e:
    print(f"⚠ Advertencia al crear gráficos: {e}")

# En esta etapa se realizó el procesamiento completo de los datos, incluyendo selección de variables relevantes, limpieza, imputación de valores faltantes, creación de la variable objetivo, codificación de variables categóricas, normalización de variables numéricas y análisis exploratorio. El conjunto de datos resultante queda preparado para el entrenamiento de modelos de clasificación.

# ENTRENAMIENTO (Etapa 4)

# 4. Entrenamiento del modelo

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# División del dataset (train, validation, test)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# Modelo base: regresión logística multiclase
model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluación inicial
val_pred = model.predict(X_val)
accuracy_score(y_val, val_pred)

# Ajuste de hiperparámetros con Grid Search
param_grid = {
    "C": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    ),
    param_grid,
    scoring="accuracy",
    cv=5
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
grid.best_params_

# 5.3 Evaluación con métricas clásicas (código)

from sklearn.metrics import classification_report

# Predicciones del modelo final
y_test_pred = best_model.predict(X_test)

# Reporte completo
print(classification_report(y_test, y_test_pred))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_test_pred)

try:
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.title("Matriz de confusión")
    plt.savefig("confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: confusion_matrix.png")
except Exception as e:
    print(f"⚠ Advertencia: {e}")

train_acc = best_model.score(X_train, y_train)
test_acc = best_model.score(X_test, y_test)

train_acc, test_acc

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

# Binarizar etiquetas
y_test_bin = label_binarize(y_test, classes=[1,2,3])
y_score = best_model.predict_proba(X_test)

try:
    plt.figure()
    
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Clase {i+1} (AUC = {roc_auc:.2f})")
    
    plt.plot([0,1], [0,1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC multiclase")
    plt.legend()
    plt.savefig("roc_curves.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: roc_curves.png")
except Exception as e:
    print(f"⚠ Advertencia: {e}")

from sklearn.metrics import precision_recall_curve

try:
    plt.figure()
    
    for i in range(3):
        precision, recall, _ = precision_recall_curve(
            y_test_bin[:, i], y_score[:, i]
        )
        plt.plot(recall, precision, label=f"Clase {i+1}")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precision–Recall")
    plt.legend()
    plt.savefig("precision_recall_curves.png", dpi=100, bbox_inches='tight')
    plt.close()
    print("✓ Gráfico guardado: precision_recall_curves.png")
except Exception as e:
    print(f"⚠ Advertencia: {e}")
