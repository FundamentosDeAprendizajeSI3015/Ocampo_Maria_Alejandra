"""
Pipeline de Análisis - Encuesta de Procrastinación (Escala Likert)
Maria Alejandra Ocampo Giraldo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys

# Config
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Rutas
CARPETA_GRAFICOS = "graficos"
ARCHIVO_OUTPUT_CSV = "data_procesada.csv"
ARCHIVO_OUTPUT_TXT = "resultados_analisis.txt"

os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

# Capturar outputs
class OutputCapture:
    def __init__(self):
        self.outputs = []
    def write(self, text):
        self.outputs.append(text)
        sys.__stdout__.write(text)
    def flush(self):
        pass

capture = OutputCapture()
sys.stdout = capture

print("="*70)
print("PIPELINE - ENCUESTA DE PROCRASTINACIÓN")
print("="*70)
print()

# ============================================================================
# 1. CARGA Y EXPLORACIÓN
# ============================================================================
print("="*70)
print("1. CARGA Y EXPLORACIÓN")
print("="*70)

df_original = pd.read_csv("procastinacion_encuesta.csv")
print(f"Dimensiones: {df_original.shape[0]} filas x {df_original.shape[1]} columnas")
print()

print("Columnas del dataset:")
for i, col in enumerate(df_original.columns, 1):
    print(f"  {i}. {col}")
print()

print("Vista previa:")
print(df_original.head(3))
print()

# ============================================================================
# 2. SELECCIÓN DE VARIABLES RELEVANTES
# ============================================================================
print("="*70)
print("2. SELECCIÓN DE VARIABLES RELEVANTES")
print("="*70)

# Variables de la escala Likert (las que comienzan con frecuencia_)
variables_likert = [
    'frecuencia_planificacion_tareas',
    'frecuencia_dividir_tareas',
    'frecuencia_autonomia_sin_presion',
    'frecuencia_concentracion_estudio',
    'frecuencia_evitar_tareas_dificiles',
    'frecuencia_revisar_celular_estudio',
    'frecuencia_interferencia_redes',
    'frecuencia_dejar_ultimo_momento'
]

df = df_original[variables_likert].copy()
print(f"Variables seleccionadas: {len(variables_likert)}")
for var in variables_likert:
    print(f"  - {var}")
print(f"\nNueva dimensión: {df.shape}")
print()

# ============================================================================
# 3. LIMPIEZA INICIAL
# ============================================================================
print("="*70)
print("3. LIMPIEZA INICIAL - NANS Y DUPLICADOS")
print("="*70)

# Detección NaNs
nans_inicial = df.isnull().sum()
print("NaNs por columna:")
print(nans_inicial)
print(f"Total NaNs: {nans_inicial.sum()}")
print()

# Tratamiento NaNs
filas_antes = len(df)
df = df.dropna()
print(f"Filas eliminadas por NaNs: {filas_antes - len(df)}")

# Duplicados
duplicados = df.duplicated().sum()
if duplicados > 0:
    df = df.drop_duplicates()
    print(f"Duplicados eliminados: {duplicados}")
else:
    print("No hay duplicados")

# Reset index
df = df.reset_index(drop=True)

# Limpieza de espacios
for col in df.columns:
    df[col] = df[col].str.strip()

print(f"\nDimensión después de limpieza: {df.shape}")
print()

# Verificar NaNs final
print("Verificación NaNs después de limpieza:")
print(f"Total NaNs restantes: {df.isnull().sum().sum()}")
print()

# ============================================================================
# 4. VALORES ÚNICOS Y VALIDACIÓN
# ============================================================================
print("="*70)
print("4. VALORES ÚNICOS Y VALIDACIÓN")
print("="*70)

print("Valores únicos por variable:")
for col in df.columns:
    valores = df[col].unique()
    print(f"\n{col}:")
    print(f"  Valores: {list(valores)}")
    print(f"  Total únicos: {len(valores)}")
print()

# ============================================================================
# 5. CONVERSIÓN ESCALA LIKERT A NUMÉRICO (1-5)
# ============================================================================
print("="*70)
print("5. CONVERSIÓN ESCALA LIKERT A NUMÉRICO")
print("="*70)

# Mapeo de escala Likert a números 1-5
mapeo_likert = {
    'Nunca': 1,
    'Rara vez': 2,
    'A veces': 3,
    'Frecuentemente': 4,
    'Siempre': 5
}

print("Mapeo aplicado:")
for k, v in mapeo_likert.items():
    print(f"  {k} → {v}")
print()

# Aplicar mapeo
df_numerico = df.copy()
for col in df.columns:
    df_numerico[col] = df[col].map(mapeo_likert)

print("Verificación de conversión:")
print(df_numerico.head())
print()

# Verificar si quedaron NaNs después del mapeo
nans_mapeo = df_numerico.isnull().sum()
if nans_mapeo.sum() > 0:
    print("ADVERTENCIA: NaNs encontrados después del mapeo")
    print(nans_mapeo[nans_mapeo > 0])
    df_numerico = df_numerico.dropna()
    print(f"Filas eliminadas: {len(df) - len(df_numerico)}")
else:
    print("Conversión exitosa sin NaNs")
print()

# ============================================================================
# 6. ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================
print("="*70)
print("6. ESTADÍSTICAS DESCRIPTIVAS")
print("="*70)

print("Tendencia Central:")
tendencia = df_numerico.agg(['mean', 'median'])
print(tendencia.round(2))
print()

print("Moda por variable:")
for col in df_numerico.columns:
    moda = df_numerico[col].mode()[0] if len(df_numerico[col].mode()) > 0 else 'N/A'
    print(f"  {col}: {moda}")
print()

print("Dispersión:")
dispersion = df_numerico.agg(['std', 'var', 'min', 'max'])
print(dispersion.round(2))
print()

print("Resumen completo:")
print(df_numerico.describe().round(2))
print()

# ============================================================================
# 7. DETECCIÓN DE OUTLIERS (IQR)
# ============================================================================
print("="*70)
print("7. DETECCIÓN DE OUTLIERS (IQR)")
print("="*70)

outliers_totales = 0
for col in df_numerico.columns:
    Q1 = df_numerico[col].quantile(0.25)
    Q2 = df_numerico[col].quantile(0.50)
    Q3 = df_numerico[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    
    outliers = df_numerico[(df_numerico[col] < limite_inf) | (df_numerico[col] > limite_sup)]
    outliers_totales += len(outliers)
    
    print(f"\n{col}:")
    print(f"  Q1: {Q1:.2f}, Q2: {Q2:.2f}, Q3: {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Límites: [{limite_inf:.2f}, {limite_sup:.2f}]")
    print(f"  Outliers: {len(outliers)}")

print(f"\nTotal outliers detectados: {outliers_totales}")
print()

# Boxplots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, col in enumerate(df_numerico.columns):
    axes[idx].boxplot(df_numerico[col].dropna(), vert=True)
    axes[idx].set_title(col.replace('frecuencia_', ''), fontsize=9)
    axes[idx].set_ylabel('Escala Likert (1-5)')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/01_boxplots_outliers.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 01_boxplots_outliers.png")
print()

# ============================================================================
# 8. HISTOGRAMAS
# ============================================================================
print("="*70)
print("8. HISTOGRAMAS")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, col in enumerate(df_numerico.columns):
    axes[idx].hist(df_numerico[col].dropna(), bins=5, range=(0.5, 5.5), 
                   edgecolor='black', alpha=0.7, color='steelblue')
    axes[idx].set_title(col.replace('frecuencia_', ''), fontsize=9)
    axes[idx].set_xlabel('Escala Likert (1-5)')
    axes[idx].set_ylabel('Frecuencia')
    axes[idx].set_xticks([1, 2, 3, 4, 5])
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/02_histogramas.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 02_histogramas.png")
print()

# ============================================================================
# 9. SCATTER PLOTS
# ============================================================================
print("="*70)
print("9. SCATTER PLOTS")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# Relaciones importantes
pares = [
    ('frecuencia_planificacion_tareas', 'frecuencia_dejar_ultimo_momento'),
    ('frecuencia_concentracion_estudio', 'frecuencia_revisar_celular_estudio'),
    ('frecuencia_dividir_tareas', 'frecuencia_evitar_tareas_dificiles'),
    ('frecuencia_autonomia_sin_presion', 'frecuencia_dejar_ultimo_momento'),
    ('frecuencia_interferencia_redes', 'frecuencia_revisar_celular_estudio'),
    ('frecuencia_planificacion_tareas', 'frecuencia_dividir_tareas')
]

for idx, (var1, var2) in enumerate(pares):
    axes[idx].scatter(df_numerico[var1], df_numerico[var2], alpha=0.6, color='steelblue')
    axes[idx].set_xlabel(var1.replace('frecuencia_', ''), fontsize=8)
    axes[idx].set_ylabel(var2.replace('frecuencia_', ''), fontsize=8)
    axes[idx].set_title(f'{var2.replace("frecuencia_", "")} vs {var1.replace("frecuencia_", "")}', fontsize=9)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xticks([1, 2, 3, 4, 5])
    axes[idx].set_yticks([1, 2, 3, 4, 5])

plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/03_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 03_scatter_plots.png")
print()

# ============================================================================
# 10. PROPORCIONES Y FRECUENCIAS
# ============================================================================
print("="*70)
print("10. PROPORCIONES Y FRECUENCIAS")
print("="*70)

print("Proporciones por variable (valores 1-5):")
for col in df_numerico.columns[:3]:  # Primeras 3 para no saturar
    print(f"\n{col}:")
    props = df_numerico[col].value_counts(normalize=True).sort_index()
    for valor, prop in props.items():
        print(f"  {int(valor)}: {prop:.2%}")
print()

# Gráfico de barras apiladas
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, col in enumerate(df_numerico.columns):
    counts = df_numerico[col].value_counts().sort_index()
    counts.plot(kind='bar', ax=axes[idx], color='steelblue', edgecolor='black')
    axes[idx].set_title(col.replace('frecuencia_', ''), fontsize=9)
    axes[idx].set_xlabel('Escala Likert')
    axes[idx].set_ylabel('Frecuencia')
    axes[idx].set_xticklabels(['1', '2', '3', '4', '5'], rotation=0)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/04_frecuencias_barras.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 04_frecuencias_barras.png")
print()

# ============================================================================
# 11. ENCODINGS
# ============================================================================
print("="*70)
print("11. ENCODINGS")
print("="*70)

print("CONVERSIÓN LIKERT (ya aplicada):")
print("  Escala textual → Numérica (1-5)")
print()

print("LABEL ENCODING (ejemplo):")
le = LabelEncoder()
df_numerico['planificacion_label'] = le.fit_transform(df_numerico['frecuencia_planificacion_tareas'])
print(f"  Variable: frecuencia_planificacion_tareas")
print(f"  Valores únicos: {sorted(df_numerico['frecuencia_planificacion_tareas'].unique())}")
print()

print("BINARY ENCODING (ejemplo):")
# Convertir a binario: bajo (1-2) vs alto (4-5)
df_numerico['concentracion_alta'] = (df_numerico['frecuencia_concentracion_estudio'] >= 4).astype(int)
print(f"  Variable creada: concentracion_alta")
print(f"  Distribución: {df_numerico['concentracion_alta'].value_counts().to_dict()}")
print()

print("ONE HOT ENCODING (ejemplo):")
# One hot para una variable categórica
df_onehot = pd.get_dummies(df_numerico, columns=['frecuencia_dejar_ultimo_momento'], prefix='ultimo_momento')
print(f"  Variable: frecuencia_dejar_ultimo_momento")
print(f"  Nuevas columnas: {len([c for c in df_onehot.columns if c.startswith('ultimo_momento_')])}")
print()

# ============================================================================
# 12. CORRELACIÓN PEARSON
# ============================================================================
print("="*70)
print("12. CORRELACIÓN PEARSON")
print("="*70)

corr_pearson = df_numerico[df_numerico.columns[:8]].corr(method='pearson')
print("Matriz de correlación:")
print(corr_pearson.round(2))
print()

print("Correlaciones significativas (|r| > 0.5):")
encontradas = False
for i in range(len(corr_pearson.columns)):
    for j in range(i+1, len(corr_pearson.columns)):
        if abs(corr_pearson.iloc[i, j]) > 0.5:
            print(f"  {corr_pearson.columns[i]} <-> {corr_pearson.columns[j]}: {corr_pearson.iloc[i, j]:.3f}")
            encontradas = True
if not encontradas:
    print("  No se encontraron correlaciones mayores a 0.5")
print()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_pearson, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación de Pearson', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/05_correlacion_pearson.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 05_correlacion_pearson.png")
print()

# ============================================================================
# 13. CORRELACIÓN SPEARMAN
# ============================================================================
print("="*70)
print("13. CORRELACIÓN SPEARMAN")
print("="*70)

corr_spearman = df_numerico[df_numerico.columns[:8]].corr(method='spearman')
print("Matriz de correlación:")
print(corr_spearman.round(2))
print()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='viridis', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación de Spearman', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/06_correlacion_spearman.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 06_correlacion_spearman.png")
print()

# ============================================================================
# 14. DECISIÓN ELIMINAR VARIABLES
# ============================================================================
print("="*70)
print("14. DECISIÓN ELIMINAR VARIABLES")
print("="*70)

alta_corr = False
for i in range(len(corr_pearson.columns)):
    for j in range(i+1, len(corr_pearson.columns)):
        if abs(corr_pearson.iloc[i, j]) > 0.85:
            print(f"  Alta correlación: {corr_pearson.columns[i]} <-> {corr_pearson.columns[j]}: {corr_pearson.iloc[i, j]:.3f}")
            alta_corr = True

if not alta_corr:
    print("DECISIÓN: No se eliminan variables")
    print("RAZÓN: No hay correlaciones superiores a 0.85")
else:
    print("Se recomienda revisar variables con correlación > 0.85")
print()

# ============================================================================
# 15. ESCALADO STANDARD
# ============================================================================
print("="*70)
print("15. ESCALADO STANDARD")
print("="*70)

scaler = StandardScaler()
df_escalado = df_numerico[df_numerico.columns[:8]].copy()
df_escalado_std = pd.DataFrame(
    scaler.fit_transform(df_escalado),
    columns=df_escalado.columns
)

print("StandardScaler aplicado")
print("\nEstadísticas después del escalado:")
print(df_escalado_std.describe().loc[['mean', 'std']].round(4))
print()

# Comparación visual
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

var_ejemplo = df_numerico.columns[0]
axes[0].hist(df_numerico[var_ejemplo], bins=5, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_title('Original (Escala 1-5)')
axes[0].set_xlabel(var_ejemplo.replace('frecuencia_', ''))
axes[0].set_ylabel('Frecuencia')
axes[0].grid(True, alpha=0.3)

axes[1].hist(df_escalado_std[var_ejemplo], bins=15, edgecolor='black', alpha=0.7, color='coral')
axes[1].set_title('Standard Scaled')
axes[1].set_xlabel(var_ejemplo.replace('frecuencia_', ''))
axes[1].set_ylabel('Frecuencia')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/07_comparacion_escalado.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 07_comparacion_escalado.png")
print()

# ============================================================================
# 16. LOG TRANSFORM (si aplica)
# ============================================================================
print("="*70)
print("16. LOG TRANSFORM")
print("="*70)

# Para escala Likert no es común, pero lo demostramos
var_log = df_numerico.columns[4]
df_numerico['log_transform_demo'] = np.log1p(df_numerico[var_log])

print(f"Log transform aplicado a: {var_log}")
print("NOTA: Para escala Likert no es necesario, pero se demuestra")
print()

# ============================================================================
# 17. AGRUPACIÓN Y AGREGACIÓN
# ============================================================================
print("="*70)
print("17. AGRUPACIÓN Y AGREGACIÓN")
print("="*70)

# Agrupar por nivel de procrastinación (dejar para último momento)
agrupacion = df_numerico.groupby('frecuencia_dejar_ultimo_momento').agg({
    'frecuencia_planificacion_tareas': 'mean',
    'frecuencia_concentracion_estudio': 'mean',
    'frecuencia_revisar_celular_estudio': 'mean'
}).round(2)

print("Promedios agrupados por 'dejar_ultimo_momento':")
print(agrupacion)
print()

# ============================================================================
# 18. GUARDAR DATASET PROCESADO
# ============================================================================
print("="*70)
print("18. GUARDAR DATASET PROCESADO")
print("="*70)

# Dataset final con todas las variables procesadas
df_final = df_escalado_std.copy()
df_final['planificacion_label'] = df_numerico['planificacion_label']
df_final['concentracion_alta'] = df_numerico['concentracion_alta']

df_final.to_csv(ARCHIVO_OUTPUT_CSV, index=False)

print(f"Dataset guardado: {ARCHIVO_OUTPUT_CSV}")
print(f"Dimensiones: {df_final.shape[0]} filas x {df_final.shape[1]} columnas")
print()

print("Variables en dataset final:")
for col in df_final.columns:
    print(f"  - {col}")
print()

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("="*70)
print("PIPELINE COMPLETADO")
print("="*70)
print()
print("ARCHIVOS GENERADOS:")
print(f"  1. CSV procesado: {ARCHIVO_OUTPUT_CSV}")
print(f"  2. Carpeta gráficos: {CARPETA_GRAFICOS}/")
print(f"     - 01_boxplots_outliers.png")
print(f"     - 02_histogramas.png")
print(f"     - 03_scatter_plots.png")
print(f"     - 04_frecuencias_barras.png")
print(f"     - 05_correlacion_pearson.png")
print(f"     - 06_correlacion_spearman.png")
print(f"     - 07_comparacion_escalado.png")
print()
print(f"DATASET ORIGINAL: {df_original.shape[0]} respuestas")
print(f"DATASET PROCESADO: {df_final.shape[0]} respuestas")
print(f"VARIABLES ANALIZADAS: {len(variables_likert)}")
print()
print("ESCALA LIKERT CONVERTIDA:")
print("  Nunca → 1")
print("  Rara vez → 2")
print("  A veces → 3")
print("  Frecuentemente → 4")
print("  Siempre → 5")
print()
print("="*70)

# Guardar outputs en TXT
sys.stdout = sys.__stdout__
with open(ARCHIVO_OUTPUT_TXT, 'w', encoding='utf-8') as f:
    f.writelines(capture.outputs)

print(f"\nResultados guardados en: {ARCHIVO_OUTPUT_TXT}")
print("\n¡PIPELINE EJECUTADO EXITOSAMENTE!")