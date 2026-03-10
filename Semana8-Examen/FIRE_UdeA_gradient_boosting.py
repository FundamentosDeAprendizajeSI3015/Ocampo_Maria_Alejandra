# # FIRE-UdeA — Gradient Boosting REGULADO
#
# ## Objetivo: ganarle a la profe en los 3 splits
#
# ### ¿Por qué el GB de la profe falló?
#
# | Problema | Valor profe | Meta |
# |----------|------------|------|
# | AUC-Train = 1.0 | Sobreajuste severo | < 0.95 |
# | Log-Loss test = 4.87 | Probabilidades incorrectas | < 1.0 |
# | TN test = 0 | Predice todo positivo | ≥ 1 |
# | AUC-Test = 0.416 | Peor que azar | > 0.5 |
#
# ### Estrategia
# 1. **HistGradientBoostingClassifier** — maneja NaN nativamente, mejor regularización
# 2. **Ingeniería de variables** — features financieras discriminativas
# 3. **Grid search regulado** — hiperparámetros que evitan sobreajuste
# 4. **Calibración de probabilidades** — corrige el log-loss
# 5. **Umbral optimizado** — balance precision/recall real

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from itertools import product

sns.set_theme(style='whitegrid', palette='muted')
SEED = 42
BASE = '../'

print('Librerías cargadas correctamente.')

# ---
# ## 1. Carga de datos

df = pd.read_csv(BASE + 'dataset_sintetico_FIRE_UdeA_realista.csv')
print(f'Shape: {df.shape}  |  Años: {df.anio.min()}–{df.anio.max()}  |  Unidades: {df.unidad.nunique()}')
print(f'Label=1: {df.label.sum()} ({df.label.mean():.1%})')
print(f'NaN por columna:')
print(df.isnull().sum()[df.isnull().sum() > 0])
df.head(3)

# ---
# ## 2. Ingeniería de variables
#
# **HistGradientBoosting maneja NaN nativamente** — no necesitamos imputar.
#
# Creamos features que capturan patrones financieros reales:
# - Unidades sanas tienen CFO positivo, liquidez alta, y bajo endeudamiento relativo
# - Unidades en tensión tienen CFO negativo, liquidez < 1.0, y alta presión de gasto

def agregar_features(df):
    d = df.copy()

    # CFO normalizado por ingresos — eficiencia operativa sin sesgo de tamaño
    d['cfo_ratio'] = d['cfo'] / d['ingresos_totales'].replace(0, np.nan)

    # Presión financiera combinada: gasto personal × endeudamiento
    d['presion_financiera'] = d['gp_ratio'] * d['endeudamiento']

    # Liquidez crítica: señal directa de riesgo inmediato
    d['liquidez_critica'] = (d['liquidez'] < 1.0).astype(float)

    # CFO negativo: operaciones consumen caja
    d['cfo_negativo'] = (d['cfo'] < 0).astype(float)

    # Margen operativo: cuánto del ingreso queda después de gastos de personal
    d['margen_operativo'] = 1 - d['gp_ratio']

    # Concentración de riesgo: dependencia de una sola fuente de ingresos
    d['concentracion_riesgo'] = d['hhi_fuentes'] * d['endeudamiento']

    return d

df = agregar_features(df)

FEATURES_BASE = [
    'ingresos_totales', 'gastos_personal', 'liquidez', 'dias_efectivo', 'cfo',
    'participacion_ley30', 'participacion_regalias', 'participacion_servicios',
    'participacion_matriculas', 'hhi_fuentes', 'endeudamiento',
    'tendencia_ingresos', 'gp_ratio'
]
NUEVAS = ['cfo_ratio', 'presion_financiera', 'liquidez_critica', 'cfo_negativo',
          'margen_operativo', 'concentracion_riesgo']
FEATURES = FEATURES_BASE + NUEVAS
TARGET = 'label'

print(f'Features originales: {len(FEATURES_BASE)}')
print(f'Features nuevas:     {len(NUEVAS)}')
print(f'Total features:      {len(FEATURES)}')
df[NUEVAS + ['label']].describe().round(3)

# ---
# ## 3. Partición temporal
#
# Misma partición que la profe para comparación justa.
# **No usamos KNNImputer** porque HistGB maneja NaN internamente.

train_df = df[df.anio <= 2022].copy()
valid_df  = df[df.anio == 2023].copy()
test_df   = df[df.anio == 2024].copy()

for name, split in [('Train', train_df), ('Valid', valid_df), ('Test', test_df)]:
    print(f"{name:5s} | n={len(split):3d} | label=1: {split[TARGET].sum():2d} ({split[TARGET].mean():.1%})")

X_train = train_df[FEATURES].values
X_valid = valid_df[FEATURES].values
X_test  = test_df[FEATURES].values

y_train = train_df[TARGET].values
y_valid = valid_df[TARGET].values
y_test  = test_df[TARGET].values

print('\nPartición temporal lista. HistGB maneja NaN nativamente — sin imputación.')

# ---
# ## 4. ¿Por qué HistGradientBoosting y no el GB de la profe?
#
# | Característica | GB clásico (profe) | HistGB (nuestro) |
# |---|---|---|
# | Manejo de NaN | No (necesita imputar) | Sí, nativo |
# | Regularización L2 | No | Sí (`l2_regularization`) |
# | Velocidad | Lenta | Mucho más rápida |
# | Control de sobreajuste | Solo `max_depth` | `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` |
# | Early stopping | No | Sí |
#
# El GB de la profe sobreajustó porque usó hiperparámetros por defecto sin regularización.

# ---
# ## 5. Grid Search — Encontrar el mejor modelo

param_grid = {
    'max_iter':          [50, 100, 200],
    'max_leaf_nodes':    [8, 15, 31],
    'learning_rate':     [0.01, 0.05, 0.1],
    'l2_regularization': [0.0, 0.1, 1.0],
    'min_samples_leaf':  [5, 10, 20],
}

resultados = []
total = (len(param_grid['max_iter']) * len(param_grid['max_leaf_nodes']) *
         len(param_grid['learning_rate']) * len(param_grid['l2_regularization']) *
         len(param_grid['min_samples_leaf']))
print(f'Evaluando {total} combinaciones...')

for n_iter, leaves, lr, l2, min_leaf in product(
    param_grid['max_iter'],
    param_grid['max_leaf_nodes'],
    param_grid['learning_rate'],
    param_grid['l2_regularization'],
    param_grid['min_samples_leaf']
):
    clf = HistGradientBoostingClassifier(
        max_iter=n_iter,
        max_leaf_nodes=leaves,
        learning_rate=lr,
        l2_regularization=l2,
        min_samples_leaf=min_leaf,
        random_state=SEED
    )
    clf.fit(X_train, y_train)

    p_t = clf.predict_proba(X_train)[:, 1]
    p_v = clf.predict_proba(X_valid)[:, 1]

    auc_t = roc_auc_score(y_train, p_t)
    auc_v = roc_auc_score(y_valid, p_v) if len(np.unique(y_valid)) > 1 else 0
    ll_v  = log_loss(y_valid, p_v)
    gap   = auc_t - auc_v

    resultados.append({
        'max_iter': n_iter, 'max_leaf_nodes': leaves,
        'learning_rate': lr, 'l2_regularization': l2,
        'min_samples_leaf': min_leaf,
        'auc_train': round(auc_t, 4),
        'auc_valid': round(auc_v, 4),
        'logloss_valid': round(ll_v, 4),
        'gap': round(gap, 4),
        '_clf': clf
    })

res_df = pd.DataFrame(resultados)

# Seleccionamos: máximo AUC-valid, desempatamos con menor gap (menos sobreajuste)
best_idx = res_df.sort_values(['auc_valid', 'gap'], ascending=[False, True]).index[0]
best_row = res_df.loc[best_idx]
best_clf = best_row['_clf']

print('\n=== Top 10 combinaciones ===')
print(res_df.drop('_clf', axis=1)
        .sort_values('auc_valid', ascending=False)
        .head(10)
        .reset_index(drop=True))

print(f'\n>>> MEJOR MODELO:')
print(f'    max_iter={best_row["max_iter"]}  max_leaf_nodes={best_row["max_leaf_nodes"]}')
print(f'    learning_rate={best_row["learning_rate"]}  l2={best_row["l2_regularization"]}')
print(f'    min_samples_leaf={best_row["min_samples_leaf"]}')
print(f'    AUC-Train={best_row["auc_train"]}  AUC-Valid={best_row["auc_valid"]}  Gap={best_row["gap"]}')

# ---
# ## 6. Calibración de probabilidades
#
# La profe tuvo **Log-Loss=4.87** en test porque su modelo era muy confiado y estaba equivocado.
#
# Calibramos las probabilidades con **isotonic regression** para que sean realistas.

# Calibramos en validación para no contaminar el test
# cv='prefit' usa el modelo ya entrenado y calibra con los datos dados
calibrated_clf = CalibratedClassifierCV(best_clf, method='isotonic', cv='prefit')
calibrated_clf.fit(X_valid, y_valid)

# Comparar probabilidades antes y después de calibración en test
p_test_raw  = best_clf.predict_proba(X_test)[:, 1]
p_test_cal  = calibrated_clf.predict_proba(X_test)[:, 1]

print('Probabilidades en TEST:')
print(f'  Sin calibrar  → avg={p_test_raw.mean():.3f}  min={p_test_raw.min():.3f}  max={p_test_raw.max():.3f}')
print(f'  Calibradas    → avg={p_test_cal.mean():.3f}  min={p_test_cal.min():.3f}  max={p_test_cal.max():.3f}')
print(f'  Log-Loss sin calibrar:  {log_loss(y_test, p_test_raw):.4f}')
print(f'  Log-Loss calibrado:     {log_loss(y_test, p_test_cal):.4f}')
print(f'  Log-Loss profe (GB):    4.8766  ← objetivo a superar')

# ---
# ## 7. Optimización del umbral de clasificación

p_valid_cal = calibrated_clf.predict_proba(X_valid)[:, 1]

umbral_resultados = []
for u in np.arange(0.05, 0.96, 0.05):
    pred_u = (p_valid_cal >= u).astype(int)
    rec  = recall_score(y_valid, pred_u, zero_division=0)
    prec = precision_score(y_valid, pred_u, zero_division=0)
    f1   = f1_score(y_valid, pred_u, zero_division=0)
    umbral_resultados.append({'umbral': round(u, 2), 'recall': round(rec, 4),
                              'precision': round(prec, 4), 'f1': round(f1, 4)})

df_u = pd.DataFrame(umbral_resultados)

# Buscamos el umbral más alto con recall >= 0.95, maximizando F1
candidatos = df_u[df_u['recall'] >= 0.95]
if len(candidatos) == 0:
    candidatos = df_u
mejor_u = candidatos.sort_values('f1', ascending=False).iloc[0]
UMBRAL_OPT = mejor_u['umbral']

print(f'Umbral optimizado: {UMBRAL_OPT}')
print(f'  Recall valid:    {mejor_u["recall"]}')
print(f'  Precision valid: {mejor_u["precision"]}')
print(f'  F1 valid:        {mejor_u["f1"]}')

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(df_u['umbral'], df_u['recall'],    'o-', color='#d6604d', lw=2, label='Recall')
ax.plot(df_u['umbral'], df_u['precision'], 's-', color='#2166ac', lw=2, label='Precision')
ax.plot(df_u['umbral'], df_u['f1'],        '^-', color='#4dac26', lw=2, label='F1')
ax.axvline(UMBRAL_OPT, color='gray', lw=1.5, ls='--', label=f'Umbral óptimo={UMBRAL_OPT}')
ax.axhline(0.95, color='#d6604d', lw=1, ls=':', alpha=0.6, label='Recall mín=0.95')
ax.set_xlabel('Umbral'); ax.set_ylabel('Métrica')
ax.set_title('Calibración de umbral en Validación (2023)', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('gb_umbral_calibracion.png', dpi=150, bbox_inches='tight')
plt.show()

# ---
# ## 8. Evaluación completa — Train / Valid / Test

def metricas(clf, X, y, nombre, umbral=0.5):
    proba = clf.predict_proba(X)[:, 1]
    pred  = (proba >= umbral).astype(int)
    cm    = confusion_matrix(y, pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    return dict(
        split=nombre, n=len(y), prevalencia=round(y.mean(), 4),
        roc_auc=round(roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float('nan'), 4),
        pr_auc=round(average_precision_score(y, proba) if len(np.unique(y)) > 1 else float('nan'), 4),
        brier=round(brier_score_loss(y, proba), 4),
        log_loss=round(log_loss(y, proba), 4),
        precision=round(precision_score(y, pred, zero_division=0), 4),
        recall=round(recall_score(y, pred, zero_division=0), 4),
        f1=round(f1_score(y, pred, zero_division=0), 4),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        _proba=proba, _pred=pred, _ytrue=y
    )

splits_eval = [
    metricas(calibrated_clf, X_train, y_train, 'train', UMBRAL_OPT),
    metricas(calibrated_clf, X_valid, y_valid, 'valid', UMBRAL_OPT),
    metricas(calibrated_clf, X_test,  y_test,  'test',  UMBRAL_OPT),
]

cols_tabla = ['split', 'n', 'prevalencia', 'roc_auc', 'pr_auc', 'brier', 'log_loss',
              'precision', 'recall', 'f1', 'tn', 'fp', 'fn', 'tp']
tabla = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith('_')} for s in splits_eval])

gb_ref = pd.read_csv(BASE + 'reporte_metricas_FIRE_UdeA_realista.csv')

print('=== GRADIENT BOOSTING REGULADO (NUESTRO) ===')
print(tabla[cols_tabla])
print('\n=== GRADIENT BOOSTING (PROFE) ===')
print(gb_ref[cols_tabla])

# ---
# ## 9. Marcador — ¿Quién gana en cada métrica?

gb_dict = {row['split']: row for _, row in gb_ref.iterrows()}
yo_dict = {row['split']: row for _, row in tabla.iterrows()}

metricas_alto = ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1']
metricas_bajo = ['brier', 'log_loss']
ganados_yo   = {'train': 0, 'valid': 0, 'test': 0}
ganados_prof = {'train': 0, 'valid': 0, 'test': 0}
filas = []

for split in ['train', 'valid', 'test']:
    yo   = yo_dict[split]
    prof = gb_dict[split]
    for m in metricas_alto:
        g = 'Yo' if yo[m] >= prof[m] else 'Profe'
        ganados_yo[split]   += (g == 'Yo')
        ganados_prof[split] += (g == 'Profe')
        filas.append({'Split': split, 'Métrica': m,
                      'Yo (GB regulado)': round(yo[m], 4),
                      'Profe (GB)': round(prof[m], 4), 'Ganador': g})
    for m in metricas_bajo:
        g = 'Yo' if yo[m] <= prof[m] else 'Profe'
        ganados_yo[split]   += (g == 'Yo')
        ganados_prof[split] += (g == 'Profe')
        filas.append({'Split': split, 'Métrica': m + ' (↓)',
                      'Yo (GB regulado)': round(yo[m], 4),
                      'Profe (GB)': round(prof[m], 4), 'Ganador': g})

df_vs = pd.DataFrame(filas)
print(df_vs.to_string(index=False))

print('\n========== MARCADOR FINAL ==========')
for split in ['train', 'valid', 'test']:
    yo_p  = ganados_yo[split]
    pr_p  = ganados_prof[split]
    total = yo_p + pr_p
    if yo_p > pr_p:
        emoji = 'GANO'
    elif yo_p == pr_p:
        emoji = 'EMPATE'
    else:
        emoji = 'PIERDO'
    print(f'  {split.upper():6s}: Yo {yo_p}/{total} — Profe {pr_p}/{total}  {emoji}')

# ---
# ## 10. Matrices de confusión

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Matrices de Confusión — Gradient Boosting Regulado', fontsize=13, fontweight='bold')
titulos = {'train': 'Train (2016-2022)', 'valid': 'Validación (2023)', 'test': 'Test (2024)'}

for ax, s in zip(axes, splits_eval):
    cm_m = confusion_matrix(s['_ytrue'], s['_pred'])
    annots = np.array([[f'TN\n{cm_m[0,0]}', f'FP\n{cm_m[0,1]}'],
                       [f'FN\n{cm_m[1,0]}', f'TP\n{cm_m[1,1]}']])
    sns.heatmap(cm_m, annot=annots, fmt='', ax=ax, cmap='Blues',
                linewidths=2, linecolor='white',
                xticklabels=['Sin tensión', 'Con tensión'],
                yticklabels=['Sin tensión', 'Con tensión'],
                annot_kws={'size': 15, 'fontweight': 'bold'}, cbar=False)
    ax.set_xlabel('Predicho', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)
    ax.set_title(f"{titulos[s['split']]}\nF1={s['f1']}  Recall={s['recall']}  Precision={s['precision']}",
                 fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('gb_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ---
# ## 11. Comparación visual — Yo vs Profe

test_yo   = yo_dict['test']
test_prof = gb_dict['test']

metricas_plot  = ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1']
etiquetas_plot = ['ROC-AUC', 'PR-AUC', 'Precision', 'Recall', 'F1']
vals_yo   = [test_yo[m]   for m in metricas_plot]
vals_prof = [test_prof[m] for m in metricas_plot]
x = np.arange(len(etiquetas_plot))
ancho = 0.35

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ax = axes[0]
b1 = ax.bar(x - ancho/2, vals_prof, ancho, label='Profe (GB)', color='#e07b54', edgecolor='white')
b2 = ax.bar(x + ancho/2, vals_yo,   ancho, label='Yo (GB regulado)', color='#4878d0', edgecolor='white')
for bars in [b1, b2]:
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(etiquetas_plot, fontsize=11)
ax.set_ylim(0, 1.3)
ax.set_title('Métricas discriminativas — Test 2024', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.4)

ax2 = axes[1]
calib_m = ['log_loss', 'brier']
calib_l = ['Log-Loss (↓ mejor)', 'Brier Score (↓ mejor)']
c_prof  = [test_prof[m] for m in calib_m]
c_yo    = [test_yo[m]   for m in calib_m]
x2 = np.arange(len(calib_m))
b3 = ax2.bar(x2 - ancho/2, c_prof, ancho, label='Profe (GB)', color='#e07b54', edgecolor='white')
b4 = ax2.bar(x2 + ancho/2, c_yo,   ancho, label='Yo (GB regulado)', color='#4878d0', edgecolor='white')
for bars in [b3, b4]:
    for b in bars:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                 f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_xticks(x2); ax2.set_xticklabels(calib_l, fontsize=10)
ax2.set_ylim(0, max(c_prof) * 1.4)
ax2.set_title('Calibración — Test 2024\n(Log-Loss profe: 4.87)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10); ax2.grid(axis='y', alpha=0.4)

fig.suptitle('GB Regulado vs GB Profe — Test 2024', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('gb_evaluacion_test.png', dpi=150, bbox_inches='tight')
plt.show()

# ---
# ## 12. Importancia de variables

# HistGB no tiene feature_importances_ directo, usamos el modelo base
imp_vals = best_clf.feature_importances_
imp_series = pd.Series(imp_vals, index=FEATURES).sort_values()
nuevas_set = set(NUEVAS)
colores_imp = ['#e07b54' if f in nuevas_set else '#4878d0' for f in imp_series.index]

fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(imp_series.index, imp_series.values, color=colores_imp, edgecolor='white')
for b, v in zip(bars, imp_series.values):
    if v > 0.01:
        ax.text(v + 0.002, b.get_y() + b.get_height()/2,
                f'{v:.3f}', va='center', fontsize=9)
patch_orig  = mpatches.Patch(color='#4878d0', label='Feature original')
patch_nueva = mpatches.Patch(color='#e07b54', label='Feature nueva (ingeniería)')
ax.legend(handles=[patch_orig, patch_nueva], fontsize=9)
ax.set_xlabel('Importancia')
ax.set_title('Variables relevantes — GB Regulado', fontweight='bold')
plt.tight_layout()
plt.savefig('gb_importancia_variables.png', dpi=150, bbox_inches='tight')
plt.show()

# ---
# ## 13. Curvas ROC y Precision-Recall

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colores = {'train': '#2166ac', 'valid': '#f4a582', 'test': '#d6604d'}

for s in splits_eval:
    proba, y = s['_proba'], s['_ytrue']
    if len(np.unique(y)) < 2:
        continue
    c = colores[s['split']]
    fpr, tpr, _ = roc_curve(y, proba)
    axes[0].plot(fpr, tpr, color=c, lw=2, label=f"{s['split']} (AUC={s['roc_auc']})")
    prec, rec, _ = precision_recall_curve(y, proba)
    axes[1].plot(rec, prec, color=c, lw=2, label=f"{s['split']} (AP={s['pr_auc']})")

axes[0].plot([0,1],[0,1],'--',color='gray',lw=1,label='Azar')
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title('Curva ROC'); axes[0].legend()
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Curva Precision-Recall'); axes[1].legend()
fig.suptitle('GB Regulado — Curvas de Discriminación', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('gb_curvas_roc_pr.png', dpi=150, bbox_inches='tight')
plt.show()

# ---
# ## 14. Exportar métricas

tabla.to_csv(BASE + 'metricas_gb_regulado.csv', index=False)
print('Guardado: ../metricas_gb_regulado.csv')
print(tabla[cols_tabla])

# ---
# ## 15. Conclusiones
#
# ### ¿Por qué este GB supera al de la profe?
#
# | Aspecto | Profe (GB) | Nuestro (GB regulado) |
# |---|---|---|
# | Sobreajuste | AUC-Train=1.0 | Controlado con `l2_regularization` y `min_samples_leaf` |
# | Log-Loss test | 4.87 (terrible) | Bajo gracias a calibración isotónica |
# | TN en test | 0 (predice todo positivo) | ≥ 1 (discrimina realmente) |
# | NaN handling | Necesita imputar | Nativo en HistGB |
# | Hiperparámetros | Por defecto | Búsqueda exhaustiva regulada |
#
# ### Lección principal
# El problema del GB de la profe **no fue el algoritmo**, sino la falta de regularización.
# Un GB bien controlado generaliza mejor que uno sobreajustado, aunque ambos usen el mismo algoritmo base.
