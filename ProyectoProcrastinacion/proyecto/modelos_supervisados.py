# =============================================================
# modelos_supervisados.py
# Modelos supervisados para clasificar nivel de procrastinacion
#
# Modelos:
#   1. Arbol de Decision  (principal)
#   2. Regresion Logistica
#   3. Regresion Lineal   (complemento ordinal)
#
# Experimento 1: data_etiquetada.csv  (etiquetas originales)
# Experimento 2: data_corregida.csv   (etiquetas corregidas)
# =============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)

# ─────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
GRAF_DIR = os.path.join(BASE_DIR, "graficos_modelos")
os.makedirs(GRAF_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────
df_orig = pd.read_csv(os.path.join(DATA_DIR, "data_etiquetada.csv"))
df_corr = pd.read_csv(os.path.join(DATA_DIR, "data_corregida.csv"))

print(f"Dataset original : {df_orig.shape}")
print(f"Dataset corregido: {df_corr.shape}")

FEATURES  = [c for c in df_orig.columns if c not in ("procrastinacion_score", "etiqueta")]
LABEL_COL = "etiqueta"
CLASES    = ["bajo", "medio", "alto"]

# Codificacion: bajo=0, medio=1, alto=2
le = LabelEncoder()
le.fit(CLASES)
LABEL_NAMES = le.classes_.tolist()   # sklearn ordena alfabeticamente

scaler = StandardScaler()

def preparar(df):
    X = df[FEATURES].values.astype(float)
    y = le.transform(df[LABEL_COL].values)
    return X, y

X_orig, y_orig = preparar(df_orig)
X_corr, y_corr = preparar(df_corr)

X_orig_s = scaler.fit_transform(X_orig)
X_corr_s = scaler.transform(X_corr)

print(f"\nClases: {list(zip(CLASES, le.transform(CLASES)))}")
print(f"Features: {FEATURES}\n")


# ─────────────────────────────────────────────────────────────
# 2. DIVISION TRAIN / TEST  (70 % entreno, 30 % prueba)
#    stratify=y garantiza proporciones iguales en ambos splits
# ─────────────────────────────────────────────────────────────
X_tr_o, X_te_o, y_tr_o, y_te_o = train_test_split(
    X_orig, y_orig, test_size=0.30, random_state=42, stratify=y_orig
)
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_corr, y_corr, test_size=0.30, random_state=42, stratify=y_corr
)

# Para regresion logistica necesitamos la version escalada
X_tr_o_s, X_te_o_s, _, _ = train_test_split(
    X_orig_s, y_orig, test_size=0.30, random_state=42, stratify=y_orig
)
X_tr_c_s, X_te_c_s, _, _ = train_test_split(
    X_corr_s, y_corr, test_size=0.30, random_state=42, stratify=y_corr
)

print(f"Tamano train: {len(X_tr_o)} | Tamano test: {len(X_te_o)}")


# ─────────────────────────────────────────────────────────────
# FUNCION AUXILIAR: metricas rapidas
# ─────────────────────────────────────────────────────────────
def metricas(y_real, y_pred, nombre, exp):
    acc = accuracy_score(y_real, y_pred)
    f1  = f1_score(y_real, y_pred, average='weighted', zero_division=0)
    cm  = confusion_matrix(y_real, y_pred, labels=range(len(LABEL_NAMES)))
    fp  = cm.sum(axis=0) - np.diag(cm)
    fn  = cm.sum(axis=1) - np.diag(cm)
    print(f"[{exp}] {nombre}:")
    print(f"  Accuracy: {acc:.3f}  |  F1-weighted: {f1:.3f}")
    print(f"  FP: {dict(zip(LABEL_NAMES, fp))}  |  FN: {dict(zip(LABEL_NAMES, fn))}")
    print(classification_report(y_real, y_pred,
                                 target_names=LABEL_NAMES, zero_division=0))
    return acc, f1, cm, fp, fn


# ═════════════════════════════════════════════════════════════
# MODELO 1: ARBOL DE DECISION
# ═════════════════════════════════════════════════════════════
print("=" * 55)
print("MODELO 1: ARBOL DE DECISION")
print("=" * 55)

dt_orig = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=42)
dt_corr = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=42)

dt_orig.fit(X_tr_o, y_tr_o)
dt_corr.fit(X_tr_c, y_tr_c)

y_pred_dt_o = dt_orig.predict(X_te_o)
y_pred_dt_c = dt_corr.predict(X_te_c)

acc_dt_o, f1_dt_o, cm_dt_o, fp_dt_o, fn_dt_o = metricas(y_te_o, y_pred_dt_o, "Decision Tree", "Original")
acc_dt_c, f1_dt_c, cm_dt_c, fp_dt_c, fn_dt_c = metricas(y_te_c, y_pred_dt_c, "Decision Tree", "Corregido")


# ── GRAFICO 1: Visualizacion del Arbol de Decision
# Se grafica el arbol entrenado con datos CORREGIDOS (mejor CV)
fig, ax = plt.subplots(figsize=(22, 10))
plot_tree(
    dt_corr,
    feature_names=FEATURES,
    class_names=LABEL_NAMES,
    filled=True,
    rounded=True,
    impurity=True,
    proportion=False,
    ax=ax,
    fontsize=9
)
ax.set_title(
    "Arbol de Decision — Dataset Corregido\n"
    f"Acc={acc_dt_c:.2f}  |  F1={f1_dt_c:.2f}  |  max_depth=4",
    fontsize=14, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "01_arbol_decision.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 01_arbol_decision.png")


# ── GRAFICO 2: Importancia de variables (del arbol corregido)
importancias = pd.Series(dt_corr.feature_importances_, index=FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
colores = ['tomato' if v > importancias.mean() else 'steelblue' for v in importancias.values]
importancias.plot(kind='barh', ax=ax, color=colores, edgecolor='black')
ax.axvline(importancias.mean(), color='gray', linestyle='--',
           alpha=0.8, label=f'Media = {importancias.mean():.3f}')
ax.set_xlabel("Importancia (reduccion de impureza Gini)", fontsize=11)
ax.set_title(
    "Importancia de Variables — Arbol de Decision\n"
    "Rojo = por encima de la media | Azul = por debajo",
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "02_importancia_variables.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 02_importancia_variables.png")


# ── GRAFICO 3: Matriz de Confusion (Original vs Corregido, lado a lado)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, cm_data, y_real, y_pred, titulo, acc_val, f1_val, fp, fn in [
    (axes[0], cm_dt_o, y_te_o, y_pred_dt_o,
     "Dataset Original", acc_dt_o, f1_dt_o, fp_dt_o, fn_dt_o),
    (axes[1], cm_dt_c, y_te_c, y_pred_dt_c,
     "Dataset Corregido", acc_dt_c, f1_dt_c, fp_dt_c, fn_dt_c),
]:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_data,
                                   display_labels=LABEL_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
    ax.set_title(
        f"Arbol de Decision — {titulo}\n"
        f"Acc={acc_val:.2f}  F1={f1_val:.2f}\n"
        f"FP={fp.sum()}  FN={fn.sum()}",
        fontsize=11, fontweight='bold'
    )

plt.suptitle(
    "Matriz de Confusion — Arbol de Decision\n"
    "Diagonal = aciertos | Fuera de diagonal = errores (FP/FN)",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "03_confusion_arbol.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 03_confusion_arbol.png")


# ═════════════════════════════════════════════════════════════
# MODELO 2: REGRESION LOGISTICA
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("MODELO 2: REGRESION LOGISTICA")
print("=" * 55)

rl_orig = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
rl_corr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

rl_orig.fit(X_tr_o_s, y_tr_o)
rl_corr.fit(X_tr_c_s, y_tr_c)

y_pred_rl_o = rl_orig.predict(X_te_o_s)
y_pred_rl_c = rl_corr.predict(X_te_c_s)

acc_rl_o, f1_rl_o, cm_rl_o, fp_rl_o, fn_rl_o = metricas(y_te_o, y_pred_rl_o, "Reg Logistica", "Original")
acc_rl_c, f1_rl_c, cm_rl_c, fp_rl_c, fn_rl_c = metricas(y_te_c, y_pred_rl_c, "Reg Logistica", "Corregido")


# ── GRAFICO 4: Curvas ROC — Arbol + Logistica (ambos datasets)
# One-vs-Rest: una curva por clase
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
COLORES_ROC = ['#e74c3c', '#3498db', '#2ecc71']

for ax, modelos_info, titulo in [
    (axes[0],
     [(dt_orig, X_te_o,   "Arbol Orig"),
      (rl_orig, X_te_o_s, "Logistica Orig")],
     "Dataset ORIGINAL"),
    (axes[1],
     [(dt_corr, X_te_c,   "Arbol Corr"),
      (rl_corr, X_te_c_s, "Logistica Corr")],
     "Dataset CORREGIDO"),
]:
    y_te_use = y_te_o if "ORIGINAL" in titulo else y_te_c
    y_bin = label_binarize(y_te_use, classes=range(len(LABEL_NAMES)))

    estilos = ['-', '--']
    for (modelo, X_te_use, nombre_mod), estilo in zip(modelos_info, estilos):
        if not hasattr(modelo, "predict_proba"):
            continue
        y_prob = modelo.predict_proba(X_te_use)
        for i, (cls_name, col) in enumerate(zip(LABEL_NAMES, COLORES_ROC)):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, color=col, linestyle=estilo, linewidth=2,
                    label=f"{nombre_mod} / {cls_name} (AUC={auc:.2f})")

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label="Azar")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"Curvas ROC — {titulo}", fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

plt.suptitle("Curvas ROC (One-vs-Rest): Arbol de Decision vs Regresion Logistica",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "04_curvas_roc.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 04_curvas_roc.png")


# ═════════════════════════════════════════════════════════════
# MODELO 3: REGRESION LINEAL  (complemento ordinal)
# Las etiquetas se tratan como numeros: bajo=0, medio=1, alto=2
# Sirve para ver la DIRECCION del efecto de cada variable,
# no como clasificador principal.
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("MODELO 3: REGRESION LINEAL (complemento)")
print("=" * 55)

def ajustar_regresion_lineal(X_tr, X_te, y_tr, y_te, nombre_exp):
    reglin = LinearRegression()
    reglin.fit(X_tr, y_tr)
    y_pred_cont  = reglin.predict(X_te)
    y_pred_class = np.clip(np.round(y_pred_cont), 0, len(LABEL_NAMES)-1).astype(int)
    r2  = reglin.score(X_te, y_te)
    acc = accuracy_score(y_te, y_pred_class)
    f1  = f1_score(y_te, y_pred_class, average='weighted', zero_division=0)
    print(f"[{nombre_exp}] Regresion Lineal:")
    print(f"  R2={r2:.3f}  |  Acc (redondeado): {acc:.3f}  |  F1: {f1:.3f}")
    return reglin, y_pred_cont, r2, acc

reg_orig, pred_cont_o, r2_o, _ = ajustar_regresion_lineal(
    X_tr_o_s, X_te_o_s, y_tr_o, y_te_o, "Original"
)
reg_corr, pred_cont_c, r2_c, _ = ajustar_regresion_lineal(
    X_tr_c_s, X_te_c_s, y_tr_c, y_te_c, "Corregido"
)

# ── GRAFICO 5: Regresion Lineal — valores reales vs predichos
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
JITTER = 0.08   # pequeno ruido para ver puntos superpuestos

for ax, y_te, y_pred_cont, nombre_exp, r2 in [
    (axes[0], y_te_o, pred_cont_o, "Dataset Original",  r2_o),
    (axes[1], y_te_c, pred_cont_c, "Dataset Corregido", r2_c),
]:
    jitter_y = np.random.default_rng(0).uniform(-JITTER, JITTER, size=len(y_te))
    ax.scatter(y_pred_cont, y_te + jitter_y,
               alpha=0.7, s=70, color='steelblue', edgecolors='black', linewidths=0.5)

    # Linea de prediccion perfecta (diagonal)
    lim_min = min(y_pred_cont.min(), 0) - 0.2
    lim_max = max(y_pred_cont.max(), len(LABEL_NAMES)-1) + 0.2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2,
            label="Prediccion perfecta")

    ax.set_xlabel("Valor predicho (continuo)", fontsize=11)
    ax.set_ylabel("Valor real (con jitter para visibilidad)", fontsize=11)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_title(f"Regresion Lineal — {nombre_exp}\nR²={r2:.3f}",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("Regresion Lineal: Valores Reales vs Predichos\n"
             "(etiqueta ordinal: bajo=0, medio=1, alto=2)",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "05_regresion_lineal.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 05_regresion_lineal.png")


# ═════════════════════════════════════════════════════════════
# TABLA COMPARATIVA FINAL
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("TABLA COMPARATIVA — ORIGINAL vs CORREGIDO")
print("=" * 55)

resultados = [
    {"Dataset": "Original",  "Modelo": "Arbol Decision",  "Accuracy": acc_dt_o, "F1-weighted": f1_dt_o},
    {"Dataset": "Corregido", "Modelo": "Arbol Decision",  "Accuracy": acc_dt_c, "F1-weighted": f1_dt_c},
    {"Dataset": "Original",  "Modelo": "Reg Logistica",   "Accuracy": acc_rl_o, "F1-weighted": f1_rl_o},
    {"Dataset": "Corregido", "Modelo": "Reg Logistica",   "Accuracy": acc_rl_c, "F1-weighted": f1_rl_c},
]

df_res = pd.DataFrame(resultados)
print(df_res.to_string(index=False))
df_res.to_csv(os.path.join(DATA_DIR, "metricas_modelos.csv"), index=False)
print("\nMetricas guardadas en data/metricas_modelos.csv")


# ═════════════════════════════════════════════════════════════
# CONCLUSIONES
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("CONCLUSIONES")
print("=" * 55)

for modelo_nombre in ["Arbol Decision", "Reg Logistica"]:
    orig = df_res[(df_res["Dataset"] == "Original") & (df_res["Modelo"] == modelo_nombre)].iloc[0]
    corr = df_res[(df_res["Dataset"] == "Corregido") & (df_res["Modelo"] == modelo_nombre)].iloc[0]
    delta = corr["F1-weighted"] - orig["F1-weighted"]
    direccion = "MEJORA" if delta > 0 else "BAJA"
    print(f"\n  {modelo_nombre}:")
    print(f"    Original  -> Acc={orig['Accuracy']:.3f}  F1={orig['F1-weighted']:.3f}")
    print(f"    Corregido -> Acc={corr['Accuracy']:.3f}  F1={corr['F1-weighted']:.3f}")
    print(f"    Diferencia F1: {delta:+.3f}  ({direccion} con dataset corregido)")

mejor = df_res.sort_values("F1-weighted", ascending=False).iloc[0]
print(f"\n  Mejor resultado global: {mejor['Modelo']} con dataset {mejor['Dataset']}")
print(f"  F1-weighted = {mejor['F1-weighted']:.3f}")

print("\n  Variables mas importantes segun el arbol de decision:")
top3 = pd.Series(dt_corr.feature_importances_, index=FEATURES).sort_values(ascending=False).head(3)
for feat, val in top3.items():
    print(f"    - {feat}: {val:.3f}")

print("\n-" * 28)
print("modelos_supervisados.py completado.")
