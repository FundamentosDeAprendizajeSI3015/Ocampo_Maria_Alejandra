"""
Análisis de Machine Learning - Dataset Titanic
Regresión Lineal (Ridge & Lasso) + Regresión Logística
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import (mean_absolute_error, r2_score,
                             accuracy_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from scipy.stats import uniform, loguniform

# ─────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2
np.random.seed(RANDOM_STATE)

COLORS = {
    "train"  : "#4C72B0",
    "test"   : "#DD8452",
    "ridge"  : "#55A868",
    "lasso"  : "#C44E52",
    "neutral": "#8172B2",
}

plt.rcParams.update({
    "figure.dpi"     : 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size"      : 11,
})

# ═══════════════════════════════════════════════════════════════════
# 1. CARGA Y LIMPIEZA DEL DATASET
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  CARGA Y PREPARACIÓN DEL DATASET")
print("=" * 60)

# Resolver rutas relativas: buscar el CSV en la misma carpeta del script
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "Titanic-Dataset.csv"
if not data_path.exists():
    # fallback: intentar ruta relativa al cwd
    data_path = Path("Titanic-Dataset.csv")
if not data_path.exists():
    raise FileNotFoundError(f"No se encontró 'Titanic-Dataset.csv' en {BASE_DIR} ni en el directorio actual.")

df = pd.read_csv(data_path)
print(f"\nDimensiones originales: {df.shape}")
print(f"Valores nulos:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ── Imputación de valores faltantes ──────────────────────────────
# Age: mediana por Pclass y Sex (más preciso que mediana global)
df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Embarked: moda (solo 2 nulos)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin: demasiados nulos → extraer si tiene cabina (1/0)
df["Has_Cabin"] = df["Cabin"].notna().astype(int)
df.drop(columns=["Cabin"], inplace=True)

# ── Ingeniería de características ────────────────────────────────
# Tamaño familiar
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Viajero solo
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Título extraído del nombre (proxy de estatus social)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
rare_titles = df["Title"].value_counts()
rare_titles = rare_titles[rare_titles < 10].index
df["Title"] = df["Title"].replace(rare_titles, "Rare")

# ── Codificación de variables categóricas ────────────────────────
df["Sex_enc"]      = LabelEncoder().fit_transform(df["Sex"])
df["Embarked_enc"] = LabelEncoder().fit_transform(df["Embarked"])
df["Title_enc"]    = LabelEncoder().fit_transform(df["Title"])

# ── Eliminar columnas irrelevantes ───────────────────────────────
df.drop(columns=["PassengerId", "Name", "Ticket", "Sex", "Embarked", "Title"],
        inplace=True)

print(f"\nDimensiones finales: {df.shape}")
print(f"Columnas: {list(df.columns)}")
print(f"Nulos restantes: {df.isnull().sum().sum()}")

# ═══════════════════════════════════════════════════════════════════
# 2. REGRESIÓN LINEAL  →  Variable objetivo: Fare
#    Justificación: Fare es continua y depende de Pclass,
#    FamilySize, Embarked, etc.
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  REGRESIÓN LINEAL  —  Predicción de FARE")
print("=" * 60)

TARGET_REG = "Fare"
FEATURES_REG = ["Pclass", "Age", "SibSp", "Parch", "FamilySize",
                "IsAlone", "Has_Cabin", "Sex_enc",
                "Embarked_enc", "Title_enc", "Survived"]

X_reg = df[FEATURES_REG]
y_reg = df[TARGET_REG]

# ── 2.1 División entrenamiento / prueba ──────────────────────────
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"\nEntrenamiento: {X_train_r.shape[0]} muestras")
print(f"Prueba       : {X_test_r.shape[0]} muestras")

# ── 2.2 Gráfica conjunto de entrenamiento y prueba ───────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(range(len(y_train_r)), y_train_r.values,
           color=COLORS["train"], alpha=0.5, s=15, label="Entrenamiento")
ax.scatter(range(len(y_train_r), len(y_train_r) + len(y_test_r)),
           y_test_r.values,
           color=COLORS["test"], alpha=0.7, s=15, label="Prueba")
ax.set(title="Distribución de Fare — Entrenamiento vs Prueba",
       xlabel="Índice de muestra", ylabel="Fare (£)")
ax.legend()
plt.tight_layout()
# Guardar en carpeta local outputs
outputs_dir = BASE_DIR / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(outputs_dir / "01_train_test_split.png")
plt.close()
print("  → Guardada: 01_train_test_split.png")

# ── 2.3 Pipelines Ridge y Lasso ──────────────────────────────────
pipeline_ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  Ridge())
])

pipeline_lasso = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  Lasso(max_iter=10_000))
])

# ── 2.4 Distribuciones de parámetros para búsqueda aleatoria ─────
param_dist_ridge = {
    "model__alpha": loguniform(1e-3, 1e3),   # escala log para alpha
}

param_dist_lasso = {
    "model__alpha": loguniform(1e-4, 1e2),
}

# ── 2.5 Búsqueda aleatoria + cross-validation (cv=5) ─────────────
search_ridge = RandomizedSearchCV(
    pipeline_ridge, param_dist_ridge,
    n_iter=50, cv=5, scoring="r2",
    random_state=RANDOM_STATE, n_jobs=-1
)

search_lasso = RandomizedSearchCV(
    pipeline_lasso, param_dist_lasso,
    n_iter=50, cv=5, scoring="r2",
    random_state=RANDOM_STATE, n_jobs=-1
)

# ── 2.6 Entrenamiento ────────────────────────────────────────────
search_ridge.fit(X_train_r, y_train_r)
search_lasso.fit(X_train_r, y_train_r)

best_ridge = search_ridge.best_estimator_
best_lasso = search_lasso.best_estimator_

# ── 2.7 Mejores parámetros ───────────────────────────────────────
print(f"\nMejor alpha Ridge : {search_ridge.best_params_['model__alpha']:.6f}")
print(f"Mejor alpha Lasso : {search_lasso.best_params_['model__alpha']:.6f}")

# ── 2.8 Métricas: R² y MAE ───────────────────────────────────────
for name, model in [("Ridge", best_ridge), ("Lasso", best_lasso)]:
    y_pred = model.predict(X_test_r)
    r2  = r2_score(y_test_r, y_pred)
    mae = mean_absolute_error(y_test_r, y_pred)
    print(f"\n  {name}")
    print(f"    R²  = {r2:.4f}")
    print(f"    MAE = {mae:.4f}")

# ── 2.9 Gráfica predicciones Ridge y Lasso ───────────────────────
y_pred_ridge = best_ridge.predict(X_test_r)
y_pred_lasso = best_lasso.predict(X_test_r)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, name, y_pred, color in zip(
    axes,
    ["Ridge", "Lasso"],
    [y_pred_ridge, y_pred_lasso],
    [COLORS["ridge"], COLORS["lasso"]]
):
    # Línea perfecta de referencia
    lims = [min(y_test_r.min(), y_pred.min()),
            max(y_test_r.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", alpha=0.4, lw=1.5, label="Predicción perfecta")

    ax.scatter(y_test_r, y_pred, color=color, alpha=0.5, s=20, label=name)
    ax.set(title=f"{name} — Real vs Predicho\n"
                 f"R²={r2_score(y_test_r, y_pred):.4f}  "
                 f"MAE={mean_absolute_error(y_test_r, y_pred):.2f}",
           xlabel="Fare real (£)", ylabel="Fare predicho (£)")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(outputs_dir / "02_linear_regression_predictions.png")
plt.close()
print("  → Guardada: 02_linear_regression_predictions.png")

# ═══════════════════════════════════════════════════════════════════
# 3. REGRESIÓN LOGÍSTICA  →  Variable objetivo: Survived (0 / 1)
#    Justificación: binaria por naturaleza (murió / sobrevivió)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  REGRESIÓN LOGÍSTICA  —  Clasificación de SURVIVED")
print("=" * 60)

TARGET_CLS = "Survived"
FEATURES_CLS = ["Pclass", "Age", "SibSp", "Parch", "FamilySize",
                "IsAlone", "Fare", "Has_Cabin",
                "Sex_enc", "Embarked_enc", "Title_enc"]

X_cls = df[FEATURES_CLS]
y_cls = df[TARGET_CLS]

print(f"\nClases  →  No sobrevivió: {(y_cls==0).sum()}  |  "
      f"Sobrevivió: {(y_cls==1).sum()}")

# ── 3.1 División entrenamiento / prueba ──────────────────────────
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y_cls   # preservar proporción de clases
)
print(f"\nEntrenamiento: {X_train_c.shape[0]} muestras")
print(f"Prueba       : {X_test_c.shape[0]} muestras")

# ── 3.2 Pipeline ─────────────────────────────────────────────────
pipeline_log = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
])

# ── 3.3 Distribuciones de parámetros ─────────────────────────────
param_dist_log = {
    "model__C"      : loguniform(1e-3, 1e3),          # inverso de regularización
    "model__penalty": ["l1", "l2"],                    # tipo de regularización
    "model__solver" : ["liblinear", "saga"],           # soportan l1 y l2
}

# ── 3.4 Búsqueda aleatoria + cross-validation ────────────────────
search_log = RandomizedSearchCV(
    pipeline_log, param_dist_log,
    n_iter=50, cv=5, scoring="f1",
    random_state=RANDOM_STATE, n_jobs=-1
)

# ── 3.5 Entrenamiento ────────────────────────────────────────────
search_log.fit(X_train_c, y_train_c)
best_log = search_log.best_estimator_

# ── 3.6 Mejores parámetros ───────────────────────────────────────
bp = search_log.best_params_
print(f"\nMejores parámetros Logística:")
print(f"  C       = {bp['model__C']:.6f}")
print(f"  penalty = {bp['model__penalty']}")
print(f"  solver  = {bp['model__solver']}")

# ── 3.7 Métricas en prueba ───────────────────────────────────────
y_pred_log  = best_log.predict(X_test_c)
y_proba_log = best_log.predict_proba(X_test_c)[:, 1]

acc = accuracy_score(y_test_c, y_pred_log)
f1  = f1_score(y_test_c, y_pred_log)
print(f"\n  Accuracy = {acc:.4f}")
print(f"  F1-Score = {f1:.4f}")

# ── 3.8 Gráfica: probabilidades predichas por clase ──────────────
fig, ax = plt.subplots(figsize=(9, 4))
for label, color, name in zip([0, 1],
                               [COLORS["train"], COLORS["test"]],
                               ["No sobrevivió (0)", "Sobrevivió (1)"]):
    mask = y_test_c == label
    ax.hist(y_proba_log[mask], bins=25, alpha=0.6,
            color=color, label=name, edgecolor="white", linewidth=0.5)

ax.axvline(0.5, color="black", lw=1.5, ls="--", label="Umbral = 0.5")
ax.set(title=f"Distribución de probabilidades predichas\n"
             f"Accuracy={acc:.4f}  F1={f1:.4f}",
       xlabel="Probabilidad de Sobrevivir", ylabel="Frecuencia")
ax.legend()
plt.tight_layout()
plt.savefig(outputs_dir / "03_logistic_proba.png")
plt.close()
print("  → Guardada: 03_logistic_proba.png")

# ── 3.9 Matriz de confusión ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test_c, y_pred_log)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Sobrevivió", "Sobrevivió"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Matriz de Confusión — Regresión Logística\n"
             f"Accuracy={acc:.4f}  F1={f1:.4f}")
plt.tight_layout()
plt.savefig(outputs_dir / "04_confusion_matrix.png")
plt.close()
print("  → Guardada: 04_confusion_matrix.png")

# ═══════════════════════════════════════════════════════════════════
# 4. RESUMEN FINAL
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  RESUMEN DE RESULTADOS")
print("=" * 60)
print(f"\n  Regresión Lineal (Fare)")
print(f"    Ridge  →  R²={r2_score(y_test_r, y_pred_ridge):.4f}  "
      f"MAE={mean_absolute_error(y_test_r, y_pred_ridge):.2f}")
print(f"    Lasso  →  R²={r2_score(y_test_r, y_pred_lasso):.4f}  "
      f"MAE={mean_absolute_error(y_test_r, y_pred_lasso):.2f}")
print(f"\n  Regresión Logística (Survived)")
print(f"    Accuracy = {acc:.4f}")
print(f"    F1-Score = {f1:.4f}")
print("\nArchivos generados:")
print("  01_train_test_split.png")
print("  02_linear_regression_predictions.png")
print("  03_logistic_proba.png")
print("  04_confusion_matrix.png")
print("  titanic_ml.py")