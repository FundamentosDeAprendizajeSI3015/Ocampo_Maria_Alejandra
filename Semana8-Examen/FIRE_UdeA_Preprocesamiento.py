# # Pipeline de Análisis de Datos – Modelo FIRE-UdeA
#
# **Objetivo:** Preprocesamiento completo del dataset sintético FIRE-UdeA para estimación de tensión financiera (label = 1).
# **Variable objetivo:** `label` → tensión financiera si CFO negativo dos años, liquidez < 1 o días de efectivo < 30.
#
# ---
# ## Contenido
# 1. Carga e inspección inicial
# 2. Análisis exploratorio de datos (EDA)
# 3. Detección y tratamiento de valores faltantes
# 4. Detección de outliers
# 5. Análisis de distribución y correlaciones
# 6. Ingeniería de features
# 7. Pipeline de preprocesamiento (sklearn)
# 8. División train / validation / test
# 9. Resumen de hallazgos

# ## 0. Instalación de dependencias

# Instalar dependencias si es necesario
# !pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels

# ## 1. Carga e inspección inicial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='Set2')
plt.rcParams['figure.dpi'] = 120

# ── Carga ─────────────────────────────────────────────────────────────────────
# El CSV está un nivel arriba de esta carpeta (ejercicio/)
df = pd.read_csv('../dataset_sintetico_FIRE_UdeA_realista.csv')

print(f'Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas')
print(f'Período:     {df.anio.min()} – {df.anio.max()}')
print(f'Unidades:    {df.unidad.nunique()} unidades académicas')
df.head()

# Tipos de datos y estructura
df.info()

# Estadísticas descriptivas
df.describe().T.style.background_gradient(cmap='Blues', subset=['mean', 'std', 'min', 'max'])

# ── Distribución de la variable objetivo ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = df['label'].value_counts().sort_index()
axes[0].bar(['Sin tensión (0)', 'Tensión financiera (1)'],
            counts.values,
            color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=0.8)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 0.4, str(v), ha='center', fontweight='bold')
axes[0].set_title('Frecuencia de la variable objetivo')
axes[0].set_ylabel('Cantidad de observaciones')

axes[1].pie(counts.values,
            labels=['Sin tensión (0)', 'Tensión financiera (1)'],
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Proporción de clases')

plt.suptitle('Variable Objetivo: Tensión Financiera (label)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('01_distribucion_label.png', bbox_inches='tight')
plt.show()

prevalencia = counts[1] / len(df)
print(f'Prevalencia (label=1): {prevalencia:.3f}  →  {prevalencia*100:.1f}%')
print(f'Ratio mayoritaria/minoritaria: {max(counts)/min(counts):.2f}x')

# Distribución por unidad académica
print('Unidades académicas y prevalencia de tensión:')
print(f'{"Unidad":<40} {"n":>4} {"label=1":>8} {"Prev":>7}')
print('-' * 65)
for u in sorted(df.unidad.unique()):
    sub = df[df.unidad == u]
    print(f'{u:<40} {len(sub):>4} {sub.label.sum():>8} {sub.label.mean():>7.3f}')

# ## 2. Análisis exploratorio de datos (EDA)

# Definición de features y target
FEATURES = [
    'ingresos_totales', 'gastos_personal', 'liquidez', 'dias_efectivo',
    'cfo', 'participacion_ley30', 'participacion_regalias',
    'participacion_servicios', 'participacion_matriculas',
    'hhi_fuentes', 'endeudamiento', 'tendencia_ingresos', 'gp_ratio'
]

TARGET = 'label'

print(f'Total features originales: {len(FEATURES)}')
print(FEATURES)

# Histogramas por clase
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i, feat in enumerate(FEATURES):
    ax = axes[i]
    for lv, color, name in [(0, '#2ecc71', 'Sin tensión'), (1, '#e74c3c', 'Tensión')]:
        subset = df[df[TARGET] == lv][feat].dropna()
        ax.hist(subset, bins=15, alpha=0.6, color=color, label=name, edgecolor='white')
    ax.set_title(feat, fontsize=9, fontweight='bold')
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

for j in range(len(FEATURES), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribución de Features por Clase', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('02_histogramas_features.png', bbox_inches='tight')
plt.show()

# Boxplots comparativos por clase
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i, feat in enumerate(FEATURES):
    ax = axes[i]
    data_plot = [
        df[df[TARGET] == 0][feat].dropna(),
        df[df[TARGET] == 1][feat].dropna()
    ]
    bp = ax.boxplot(data_plot, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2})
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    ax.set_xticklabels(['Sin tensión', 'Tensión'], fontsize=8)
    ax.set_title(feat, fontsize=9, fontweight='bold')

for j in range(len(FEATURES), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Boxplots por Clase – Comparación de Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('03_boxplots_features.png', bbox_inches='tight')
plt.show()

# Test de Mann-Whitney U – diferencias estadísticas entre clases
from scipy import stats

print(f'{"Feature":<35} {"U-stat":>10} {"p-value":>10} {"Sig.":>18}')
print('-' * 78)
for feat in FEATURES:
    g0 = df[df[TARGET] == 0][feat].dropna()
    g1 = df[df[TARGET] == 1][feat].dropna()
    if len(g0) > 1 and len(g1) > 1:
        u_stat, p_val = stats.mannwhitneyu(g0, g1, alternative='two-sided')
        sig = '*** p<0.05' if p_val < 0.05 else ''
        print(f'{feat:<35} {u_stat:>10.1f} {p_val:>10.4f} {sig:>18}')

# ## 3. Detección y tratamiento de valores faltantes

# Mapa de valores faltantes
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Faltantes': missing, 'Porcentaje (%)': missing_pct})
missing_df = missing_df[missing_df['Faltantes'] > 0]

if missing_df.empty:
    print('No se detectaron valores faltantes.')
else:
    print('Columnas con valores faltantes:')
    print(missing_df.to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(missing_df.index, missing_df['Porcentaje (%)'],
                   color='#e74c3c', edgecolor='black', linewidth=0.7)
    for bar, val in zip(bars, missing_df['Porcentaje (%)']):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)
    ax.set_xlabel('% de valores faltantes')
    ax.set_title('Valores Faltantes por Columna', fontweight='bold')
    ax.set_xlim(0, missing_df['Porcentaje (%)'].max() * 1.4)
    plt.tight_layout()
    plt.savefig('04_missing_values.png', bbox_inches='tight')
    plt.show()

# Contexto de los faltantes: ¿qué filas son?
for col in missing_df.index:
    idx = df[df[col].isnull()].index
    print(f'\nFaltantes en [{col}]:')
    print(df.loc[idx, ['anio', 'unidad', col, TARGET]].to_string())

# Imputación: mediana por unidad académica (robusta ante distribuciones sesgadas)
df_clean = df.copy()

for col in FEATURES:
    if df_clean[col].isnull().any():
        mediana_grupo  = df_clean.groupby('unidad')[col].transform('median')
        mediana_global = df_clean[col].median()
        df_clean[col]  = df_clean[col].fillna(mediana_grupo).fillna(mediana_global)

print(f'Valores faltantes antes: {df[FEATURES].isnull().sum().sum()}')
print(f'Valores faltantes tras imputación: {df_clean[FEATURES].isnull().sum().sum()}')

# ## 4. Detección de Outliers

from scipy.stats import zscore

# Método Z-score
z_scores = df_clean[FEATURES].apply(zscore, nan_policy='omit').abs()
outliers_z = (z_scores > 3).sum()
print('Outliers |z| > 3 por feature:')
print(outliers_z[outliers_z > 0].to_string() or '  Ninguno')

# Método IQR
print('\nOutliers por método IQR (1.5×IQR):')
total_iqr = 0
for feat in FEATURES:
    Q1 = df_clean[feat].quantile(0.25)
    Q3 = df_clean[feat].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df_clean[feat] < Q1 - 1.5*IQR) | (df_clean[feat] > Q3 + 1.5*IQR)).sum()
    if n_out > 0:
        print(f'  {feat:<35}: {n_out} outliers')
        total_iqr += n_out
if total_iqr == 0:
    print('  Ninguno')

# Decisión sobre outliers
# Dataset pequeño (n≈80): eliminar outliers reduciría demasiado el tamaño.
# Los outliers pueden representar eventos financieros reales relevantes.
# Estrategia: conservar outliers y usar RobustScaler en el pipeline.
print('Decisión: conservar outliers → se usará RobustScaler (robusto a outliers).')

# ## 5. Análisis de Distribución y Correlaciones

# Matriz de correlación Spearman (robusta ante no-normalidad)
corr_matrix = df_clean[FEATURES + [TARGET]].corr(method='spearman')

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn_r', center=0, ax=ax,
            linewidths=0.5, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 8})
ax.set_title('Matriz de Correlación Spearman (incluyendo label)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('05_correlacion_spearman.png', bbox_inches='tight')
plt.show()

# Ranking de correlación con el target
corr_target = df_clean[FEATURES + [TARGET]].corr(method='spearman')[TARGET].drop(TARGET)
corr_sorted = corr_target.abs().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors_bar = ['#e74c3c' if corr_target[f] > 0 else '#3498db' for f in corr_sorted.index]
ax.barh(corr_sorted.index, corr_sorted.values, color=colors_bar,
        edgecolor='black', linewidth=0.6)
ax.axvline(0.30, color='gray', linestyle='--', alpha=0.7, label='Umbral 0.30')
ax.set_xlabel('|Correlación Spearman| con label')
ax.set_title('Ranking de Features por Correlación con Tensión Financiera', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('06_correlacion_ranking.png', bbox_inches='tight')
plt.show()

print('\nCorrelación Spearman con label:')
for feat in corr_sorted.index:
    bar = '█' * int(abs(corr_target[feat]) * 30)
    print(f'  {feat:<35} {corr_target[feat]:+.4f}  {bar}')

# Detección de multicolinealidad – VIF
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X_vif = df_clean[FEATURES].dropna()
    vif_data = pd.DataFrame({
        'Feature': FEATURES,
        'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(len(FEATURES))]
    }).sort_values('VIF', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_vif = ['#e74c3c' if v > 10 else '#f39c12' if v > 5 else '#2ecc71'
                  for v in vif_data['VIF']]
    ax.barh(vif_data['Feature'], vif_data['VIF'], color=colors_vif,
            edgecolor='black', linewidth=0.6)
    ax.axvline(10, color='red', linestyle='--', label='VIF=10 (alto)')
    ax.axvline(5, color='orange', linestyle='--', label='VIF=5 (moderado)')
    ax.set_xlabel('VIF')
    ax.set_title('Multicolinealidad – VIF por Feature', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('07_vif.png', bbox_inches='tight')
    plt.show()

    print('\nVIF por feature (>10 = alta multicolinealidad):')
    print(vif_data.to_string(index=False))
except ImportError:
    print('statsmodels no disponible. Instalar con: pip install statsmodels')

# ## 6. Ingeniería de Features

# Features derivadas del dominio FIRE-UdeA
df_feat = df_clean.copy()

# 1. CFO normalizado por ingresos
df_feat['cfo_ratio'] = df_feat['cfo'] / (df_feat['ingresos_totales'] + 1e-9)

# 2. Gastos de personal sobre ingresos
df_feat['gastos_ing_ratio'] = df_feat['gastos_personal'] / (df_feat['ingresos_totales'] + 1e-9)

# 3. Flag: liquidez crítica (< 1)
df_feat['liquidez_critica'] = (df_feat['liquidez'] < 1).astype(int)

# 4. Flag: días de efectivo críticos (< 30)
df_feat['dias_criticos'] = (df_feat['dias_efectivo'] < 30).astype(int)

# 5. Flag: CFO negativo
df_feat['cfo_negativo'] = (df_feat['cfo'] < 0).astype(int)

# 6. Flag: alta dependencia ley 30 (> 40%)
df_feat['dependencia_ley30'] = (df_feat['participacion_ley30'] > 0.4).astype(int)

NEW_FEATURES = ['cfo_ratio', 'gastos_ing_ratio', 'liquidez_critica',
                'dias_criticos', 'cfo_negativo', 'dependencia_ley30']

FEATURES_FULL = FEATURES + NEW_FEATURES

print(f'Features originales : {len(FEATURES)}')
print(f'Features derivadas  : {len(NEW_FEATURES)}')
print(f'Total features      : {len(FEATURES_FULL)}')
df_feat[['anio', 'unidad'] + NEW_FEATURES + [TARGET]].head(10)

# Correlación de nuevas features con el target
corr_new = df_feat[NEW_FEATURES + [TARGET]].corr(method='spearman')[TARGET].drop(TARGET)
print('Correlación nuevas features con label (Spearman):')
for f in corr_new.sort_values(ascending=False, key=abs).index:
    print(f'  {f:<25}: {corr_new[f]:+.4f}')

# ## 7. Pipeline de Preprocesamiento (sklearn)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn import set_config

# Separar features continuas vs. binarias
CONTINUAS = [f for f in FEATURES_FULL
             if f not in ['liquidez_critica', 'dias_criticos', 'cfo_negativo', 'dependencia_ley30']]
BINARIAS   = ['liquidez_critica', 'dias_criticos', 'cfo_negativo', 'dependencia_ley30']

# Sub-pipeline numérico: impute + escala robusta
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  RobustScaler())
])

# Sub-pipeline binario: solo impute (ya son 0/1)
bin_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

# Transformador combinado
preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, CONTINUAS),
    ('bin', bin_pipeline, BINARIAS)
], remainder='drop')

set_config(display='diagram')
preprocessor

# ## 8. División Train / Validation / Test

from sklearn.model_selection import train_test_split

X = df_feat[FEATURES_FULL].copy()
y = df_feat[TARGET].copy()

# División estratificada: 70% train | 15% val | 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f'Train:      {len(X_train):4d} obs | prevalencia label=1: {y_train.mean():.3f}')
print(f'Validation: {len(X_val):4d} obs | prevalencia label=1: {y_val.mean():.3f}')
print(f'Test:       {len(X_test):4d} obs | prevalencia label=1: {y_test.mean():.3f}')

# Aplicar pipeline de preprocesamiento
X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep   = preprocessor.transform(X_val)
X_test_prep  = preprocessor.transform(X_test)

feature_names_out = CONTINUAS + BINARIAS

X_train_df = pd.DataFrame(X_train_prep, columns=feature_names_out)
X_val_df   = pd.DataFrame(X_val_prep,   columns=feature_names_out)
X_test_df  = pd.DataFrame(X_test_prep,  columns=feature_names_out)

print('Estadísticas de X_train tras preprocesamiento:')
X_train_df.describe().round(3)

# Validación: sin NaNs
assert X_train_df.isnull().sum().sum() == 0
assert X_val_df.isnull().sum().sum()   == 0
assert X_test_df.isnull().sum().sum()  == 0
print('OK: sin valores faltantes en ningún split.')

# Guardar splits preprocesados
import os
os.makedirs('splits', exist_ok=True)

X_train_df.assign(label=y_train.values).to_csv('splits/train.csv', index=False)
X_val_df.assign(label=y_val.values).to_csv('splits/valid.csv',     index=False)
X_test_df.assign(label=y_test.values).to_csv('splits/test.csv',    index=False)

# También guardar los índices originales para trazabilidad
pd.Series(y_train.index.tolist(), name='idx').to_csv('splits/idx_train.csv', index=False)
pd.Series(y_val.index.tolist(),   name='idx').to_csv('splits/idx_val.csv',   index=False)
pd.Series(y_test.index.tolist(),  name='idx').to_csv('splits/idx_test.csv',  index=False)

print('Splits guardados en splits/')
print(f'  train.csv : {len(X_train_df)} filas × {len(feature_names_out)} features')
print(f'  valid.csv : {len(X_val_df)} filas')
print(f'  test.csv  : {len(X_test_df)} filas')

# ## 9. Resumen de Hallazgos

print('=' * 65)
print('   RESUMEN DEL PIPELINE DE PREPROCESAMIENTO – FIRE-UdeA')
print('=' * 65)
print()
print(f'  Dataset original:       {df.shape[0]} obs × {df.shape[1]} cols')
print(f'  Prevalencia label=1:    {y.mean():.3f}  ({y.mean()*100:.1f}%)')
print()
print('  CALIDAD DE DATOS')
print(f'    Faltantes antes :  {df[FEATURES].isnull().sum().sum()}')
print(f'    Faltantes después: 0  (imputación mediana por unidad)')
print(f'    Outliers: conservados → RobustScaler aplicado')
print()
print('  FEATURES')
print(f'    Originales:  {len(FEATURES)}')
print(f'    Derivadas:   {len(NEW_FEATURES)}')
print(f'    Total:       {len(FEATURES_FULL)}')
print()
print('  TOP 5 FEATURES (correlación con label):')
top5 = corr_target.abs().sort_values(ascending=False).head(5)
for feat, _ in top5.items():
    print(f'    {feat:<35} {corr_target[feat]:+.4f}')
print()
print('  SPLITS')
print(f'    Train:      {len(X_train)} obs (70%)')
print(f'    Validation: {len(X_val)} obs  (15%)')
print(f'    Test:       {len(X_test)} obs  (15%)')
print()
print('  SIGUIENTE: FIRE_UdeA_ModeloArbol.ipynb')
print('=' * 65)
