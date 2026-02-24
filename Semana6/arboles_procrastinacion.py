import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

random_state = 42
plt.rc('font', family='serif', size=12)

# Carpeta de gráficos
script_dir = Path(__file__).resolve().parent
graficos_dir = script_dir / 'graficos'
graficos_dir.mkdir(exist_ok=True)

def guardar(fig, nombre):
    fig.savefig(graficos_dir / nombre, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  Guardado: graficos/{nombre}')

# =============================================================================
# 1. CARGA DEL DATASET (robusta: prueba varios nombres/encodings)
# =============================================================================
csv_names = [
    'procastinacion_encuesta.csv',
    'procrastinacion_encuesta.csv',
    'Procrastinacion.csv',
    'Procrastination.csv',
    'procastinacion_encuesta.csv'
]

data = None
last_exc = None
attempts = [{}, {'encoding': 'latin-1'}, {'sep': ';'}, {'encoding': 'latin-1', 'sep': ';'}]
for name in csv_names:
    p = script_dir / name
    if not p.exists():
        continue
    for kwargs in attempts:
        try:
            data = pd.read_csv(p, **kwargs)
            print(f"Cargado: {p.name} -> opciones: {kwargs}")
            break
        except Exception as e:
            last_exc = e
    if data is not None:
        break

if data is None:
    found = list(script_dir.glob('*.csv'))
    print('No pude leer los CSV esperados. CSVs encontrados en la carpeta:', [f.name for f in found])
    if last_exc is not None:
        print('\nÚltimo error al intentar leer un CSV:')
        print(repr(last_exc))
    raise FileNotFoundError('CSV no encontrado. Añade el nombre correcto a csv_names o coloca el CSV en la carpeta.')

print('Shape:', data.shape)
print(data.head())

# =============================================================================
# 2. LIMPIEZA Y TRANSFORMACIÓN
# =============================================================================
df = data.copy()

# Eliminar columnas inútiles (metadatos y 100% nulas)
cols_drop = ['id', 'hora_inicio', 'hora_fin', 'correo', 'nombre', 'hora_ultima_modificacion']
df.drop(columns=[c for c in cols_drop if c in df.columns], inplace=True)

# Limpiar strings
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip().str.lower()

# Eliminar duplicados
df.drop_duplicates(inplace=True)

print(f'\nShape limpio: {df.shape}')
print(df.head())

# =============================================================================
# 3. EXPLORACIÓN GRÁFICA
# =============================================================================

# Distribución de cada variable categórica
for idx, col in enumerate(df.columns):
    fig, ax = plt.subplots(figsize=(8, 4))
    df[col].value_counts().plot(kind='bar', ax=ax, color='coral', edgecolor='white')
    ax.set_title(f'Distribución de {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frecuencia')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    guardar(fig, f'01_dist_{idx+1:02d}_{col[:35]}.png')

# =============================================================================
# 4. COLUMNA OBJETIVO Y FEATURES
# =============================================================================
# Target: frecuencia_dejar_ultimo_momento
# Features: las otras 7 frecuencias

target_col = 'frecuencia_dejar_ultimo_momento'
feature_cols = [c for c in df.columns if c != target_col]

print(f'\nTarget: {target_col}')
print(df[target_col].value_counts())
print(f'\nFeatures: {feature_cols}')

X = df[feature_cols]
# Eliminar clases con <2 muestras (evita problemas en stratify y folds)
class_counts = df[target_col].value_counts()
rare_classes = class_counts[class_counts < 2].index.tolist()
if rare_classes:
    print(f"Clases con <2 muestras (se eliminarán): {rare_classes}")
    mask = ~df[target_col].isin(rare_classes)
    X = X[mask].reset_index(drop=True)
    y = df.loc[mask, target_col].reset_index(drop=True)
else:
    y = df[target_col].reset_index(drop=True)

# Codificar target
le = LabelEncoder()
y = le.fit_transform(y)
print(f'\nClases codificadas: {dict(enumerate(le.classes_))}')

# =============================================================================
# 5. DIVISIÓN: TRAIN / VALIDACIÓN / TEST
# =============================================================================
# 70% train | 15% validación | 15% test
# Si alguna clase tiene menos de 2 muestras, no se puede usar `stratify=y`.
class_counts = pd.Series(y).value_counts()
if (class_counts < 2).any():
    print('Advertencia: algunas clases tienen <2 muestras, se usará stratify=None.')
    stratify_arg = None
else:
    stratify_arg = y

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=random_state, stratify=stratify_arg
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
)

print(f'\nTrain : {X_train.shape}')
print(f'Val   : {X_val.shape}')
print(f'Test  : {X_test.shape}')

# Gráfico distribución de clases
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, ys, titulo, color in zip(
    axes,
    [y_train, y_val, y_test],
    ['Train', 'Validación', 'Test'],
    ['steelblue', 'mediumseagreen', 'coral']
):
    pd.Series(ys).value_counts().plot(kind='bar', ax=ax, color=color, edgecolor='white')
    ax.set_title(f'Clases - {titulo}')
    ax.set_xlabel('Clase')
    ax.set_ylabel('Frecuencia')
plt.tight_layout()
guardar(fig, '02_distribucion_splits.png')

# =============================================================================
# 6. PIPELINE
# =============================================================================
# Todas las features son categóricas -> OrdinalEncoder
preprocessor = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), feature_cols)
])

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=random_state))
])

pipeline_gb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=random_state))
])

# =============================================================================
# 7. ENTRENAMIENTO CON GRIDSEARCHCV
# =============================================================================
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5, 7],
    'classifier__min_samples_leaf': [1, 3, 5]
}

rf = GridSearchCV(pipeline_rf, cv=3, param_grid=param_grid, scoring='accuracy', n_jobs=-1, verbose=1)
gb = GridSearchCV(pipeline_gb, cv=3, param_grid=param_grid, scoring='accuracy', n_jobs=-1, verbose=1)

print('\nEntrenando Random Forest...')
rf.fit(X_train, y_train)
print(f'Mejores params RF: {rf.best_params_}')

print('\nEntrenando Gradient Boosting...')
gb.fit(X_train, y_train)
print(f'Mejores params GB: {gb.best_params_}')

# =============================================================================
# 8. MÉTRICAS: TRAIN / VALIDACIÓN / TEST
# =============================================================================
def evaluar(model, nombre):
    resultados = {}
    print(f'\n{"="*50}')
    print(f'MODELO: {nombre}')
    print(f'{"="*50}')
    for split_name, X_eval, y_eval in [
        ('Train',      X_train, y_train),
        ('Validacion', X_val,   y_val),
        ('Test',       X_test,  y_test)
    ]:
        y_pred = model.predict(X_eval)
        acc  = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_eval, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_eval, y_pred, average='weighted', zero_division=0)
        print(f'  [{split_name}] Accuracy={acc:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | F1={f1:.4f}')
        resultados[split_name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    return resultados

metricas_rf = evaluar(rf, 'Random Forest')
metricas_gb = evaluar(gb, 'Gradient Boosting')

# Reportes detallados
print('\n=== Reporte RF (Test) ===')
labels_present = np.unique(y_test)
print(classification_report(y_test, rf.predict(X_test), labels=labels_present,
                                target_names=[le.classes_[i] for i in labels_present]))
print('=== Reporte GB (Test) ===')
labels_present = np.unique(y_test)
print(classification_report(y_test, gb.predict(X_test), labels=labels_present,
                                target_names=[le.classes_[i] for i in labels_present]))

# =============================================================================
# 9. MATRICES DE CONFUSIÓN (Train / Validación / Test)
# =============================================================================
for model, nombre, fname in [
    (rf, 'Random Forest',     'rf'),
    (gb, 'Gradient Boosting', 'gb')
]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (X_eval, y_eval, split) in zip(axes, [
        (X_train, y_train, 'Train'),
        (X_val,   y_val,   'Validacion'),
        (X_test,  y_test,  'Test')
    ]):
        cm = confusion_matrix(y_eval, model.predict(X_eval))
        labels_eval = np.unique(y_eval)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=[le.classes_[i] for i in labels_eval])
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'{nombre} — {split}')
        ax.tick_params(axis='x', labelrotation=30)
    plt.tight_layout()
    guardar(fig, f'03_matriz_confusion_{fname}.png')

# =============================================================================
# 10. COMPARACIÓN VISUAL DE MÉTRICAS (Test)
# =============================================================================
metricas_nombres = ['accuracy', 'precision', 'recall', 'f1']
labels = ['Accuracy', 'Precision', 'Recall', 'F1']
rf_vals = [metricas_rf['Test'][m] for m in metricas_nombres]
gb_vals = [metricas_gb['Test'][m] for m in metricas_nombres]

x = np.arange(len(metricas_nombres))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width/2, rf_vals, width, label='Random Forest',     color='steelblue', edgecolor='white')
bars2 = ax.bar(x + width/2, gb_vals, width, label='Gradient Boosting', color='coral',     edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Valor')
ax.set_title('Comparación de métricas — Test set')
ax.legend()
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
guardar(fig, '04_comparacion_metricas.png')

# =============================================================================
# 11. IMPORTANCIA DE FEATURES — RANDOM FOREST
# =============================================================================
importancias = rf.best_estimator_.named_steps['classifier'].feature_importances_
indices = np.argsort(importancias)[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(len(feature_cols)), importancias[indices], color='steelblue', edgecolor='white')
ax.set_xticks(range(len(feature_cols)))
ax.set_xticklabels([feature_cols[i] for i in indices], rotation=40, ha='right')
ax.set_title('Importancia de features — Random Forest')
ax.set_ylabel('Importancia')
plt.tight_layout()
guardar(fig, '05_importancia_features.png')


# =============================================================================
# 12. VISUALIZACIÓN DEL ÁRBOL — RANDOM FOREST (un árbol del bosque)
# =============================================================================
from sklearn.tree import export_graphviz, plot_tree

# Extraer un árbol individual del mejor Random Forest
mejor_rf = rf.best_estimator_.named_steps['classifier']
arbol = mejor_rf.estimators_[0]

# Obtener nombres de features tras el preprocesador
try:
    feature_names_out = rf.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
    feature_names_out = [f.replace('cat__', '') for f in feature_names_out]
except Exception:
    feature_names_out = feature_cols

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    arbol,
    feature_names=feature_names_out,
    class_names=le.classes_,
    filled=True,
    rounded=True,
    max_depth=3,       # limita profundidad para que sea legible
    fontsize=9,
    ax=ax
)
ax.set_title('Árbol de decisión (árbol #1 del Random Forest, profundidad máx. 3)')
plt.tight_layout()
guardar(fig, '06_arbol_random_forest.png')

# También graficar un árbol del Gradient Boosting
mejor_gb = gb.best_estimator_.named_steps['classifier']
arbol_gb = mejor_gb.estimators_[0, 0]  # primer árbol, primera clase

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    arbol_gb,
    feature_names=feature_names_out,
    class_names=None,   # GB usa residuos, no clases directas
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9,
    ax=ax
)
ax.set_title('Árbol de decisión (árbol #1 del Gradient Boosting, profundidad máx. 3)')
plt.tight_layout()
guardar(fig, '06_arbol_gradient_boosting.png')

print(f'\nTodos los graficos guardados en: {graficos_dir}')
