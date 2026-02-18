"""
Pipeline de Análisis - Procrastinación Estudiantil
Maria Alejandra Ocampo Giraldo
"""

# Librerías para análisis, limpieza, visualización y preprocesamiento del dataset.
# También se usan para crear carpetas y guardar los resultados del pipeline.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import os
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
try:
    import umap.umap_ as umap
except Exception:
    umap = None

# Config visual. Le dice a seaborn que todas las gráficas tengan fondo blanco con cuadrícula.
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_GRAFICOS = os.path.join(SCRIPT_DIR, "graficos")
ARCHIVO_OUTPUT_CSV = os.path.join(SCRIPT_DIR, "data_procesada.csv")
ARCHIVO_OUTPUT_TXT = os.path.join(SCRIPT_DIR, "resultados_analisis.txt")

# Crear carpeta graficos
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

# Capturar outputs
#clase para guardar todo lo que se imprime en pantalla.
class OutputCapture:
    def __init__(self):
        self.outputs = []

#Cada vez que hago print() se  Guarda el texto en la lista outputs    
    def write(self, text):
        self.outputs.append(text)
        sys.__stdout__.write(text)
    
    def flush(self):
        pass

capture = OutputCapture()
sys.stdout = capture

print("="*70)
print("PIPELINE DE ANÁLISIS - PROCRASTINACIÓN ESTUDIANTIL")
print("="*70)
print()

# -----------------
# 1. CARGA DATASET
# -----------------
print("="*70)
print("1. CARGA DATASET")
print("="*70)

df_original = pd.read_csv(os.path.join(SCRIPT_DIR, "Procrastination.csv"))
print(f"Filas: {df_original.shape[0]}")
print(f"Columnas: {df_original.shape[1]}")
print()

# ------------------------
# 2. INSPECCIÓN RÁPIDA
# ------------------------
# nombres de columnas, tipos de datos y primeras filas.
# Esto nos ayuda a entender la estructura antes de limpiarlo.

print("="*70)
print("2. INSPECCIÓN RÁPIDA")
print("="*70)

print("Columnas disponibles:")
for i, col in enumerate(df_original.columns, 1):
    print(f"  {i}. {col}")
print()

print("Info del dataset:")
print(df_original.info())
print()

print("Primeras 3 filas:")
print(df_original.head(3))
print()

# -------------------------------------
# 3. SELECCIÓN DE VARIABLES RELEVANTES
# -------------------------------------
print("="*70)
print("3. SELECCIÓN DE VARIABLES RELEVANTES")
print("="*70)

variables_seleccionadas = [
    "study_hours_per_week",
    "cgpa",
    "assignment_delay_frequency",
    "stress_due_to_procrastination",
    "use_of_time_management",
    "procrastination_reasons",
    "hours_spent_on_mobile_non_academic",
    "procrastination_and_grade_outcome"
]

df = df_original[variables_seleccionadas].copy()
print(f"Variables seleccionadas: {len(variables_seleccionadas)}")
for var in variables_seleccionadas:
    print(f"  - {var}")
print(f"\nNueva dimensión: {df.shape[0]} filas x {df.shape[1]} columnas")
print()

# ---------------------------
# 4. NANS - DETECCIÓN
# ---------------------------
#Cuántos valores faltantes hay por columna.Cuántos hay en total.Qué porcentaje representan.

print("="*70)
print("4. NANS - DETECCIÓN")
print("="*70)

nans_inicial = df.isnull().sum()
print("NaNs por columna:")
print(nans_inicial)
print(f"\nTotal NaNs: {nans_inicial.sum()}")
print(f"Porcentaje de NaNs: {(nans_inicial.sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")
print()

# ---------------------------
# 5. NANS - TRATAMIENTO
# ----------------------------

print("="*70)
print("5. NANS - TRATAMIENTO")
print("="*70)

# Elimina filas con valores faltantes.
filas_antes = len(df)
df = df.dropna()
filas_despues = len(df)
print(f"Estrategia: Eliminación de filas con NaNs")
print(f"Filas antes: {filas_antes}")
print(f"Filas después: {filas_despues}")
print(f"Filas eliminadas: {filas_antes - filas_despues}")
print()

# ---------------------------
# 6. NANS - VERIFICACIÓN
# -----------------------------
print("="*70)
print("6. NANS - VERIFICACIÓN")
print("="*70)

nans_final = df.isnull().sum()
print("NaNs después del tratamiento:")
print(nans_final)
print(f"\nTotal NaNs restantes: {nans_final.sum()}")
print("Estado: Dataset limpio de NaNs" if nans_final.sum() == 0 else "Aún hay NaNs")
print()

# -------------------------------------
# 7. MANIPULACIÓN FILAS/COLUMNAS
# --------------------------------------
print("="*70)
print("7. MANIPULACIÓN FILAS/COLUMNAS")
print("="*70)

print(f"Dimensión actual: {df.shape}")
print(f"Índice: {df.index.min()} a {df.index.max()}")

# Reset index después de eliminar filas
df = df.reset_index(drop=True)
print(f"Índice reseteado: 0 a {len(df)-1}")
print()

# -----------------------
# 8. LIMPIEZA TEXTO
# -----------------------
# Se limpian espacios en columnas de texto para evitar errores en mapeos.
# También  se verifica si quedaron valores vacíos después de la limpieza.

print("="*70)
print("8. LIMPIEZA TEXTO")
print("="*70)

columnas_texto = df.select_dtypes(include=['object']).columns
print(f"Columnas de texto: {len(columnas_texto)}")

for col in columnas_texto:
    df[col] = df[col].str.strip()
    vacios = (df[col] == '').sum()
    if vacios > 0:
        print(f"  {col}: {vacios} valores vacíos encontrados")

print("Limpieza de espacios completada")
print()

# ---------------
# 9. DUPLICADOS
# ----------------
print("="*70)
print("9. DUPLICADOS")
print("="*70)

duplicados = df.duplicated().sum()
print(f"Filas duplicadas: {duplicados}")

if duplicados > 0:
    print(f"Filas restantes: {len(df)}")
else:
    print("No hay duplicados en el dataset")
print()

# ----------------------------------
# 10. VALIDACIÓN LÓGICA/CONSISTENCIA
# -----------------------------------
#Esta parte sirve para revisar si los valores tienen sentido y están bien estructurados.
print("="*70)
print("10. VALIDACIÓN LÓGICA/CONSISTENCIA")
print("="*70)

print("Valores únicos por variable:")
for col in df.columns:
    unicos = df[col].nunique()
    print(f"  {col}: {unicos} valores únicos")
    if df[col].dtype == 'object':
        print(f"    Ejemplos: {list(df[col].unique()[:3])}")
print()

# -----------------------------
# 11. TIPOS + CONVERSIÓN
# -----------------------------
# Convertimos variables categóricas a valores numéricos
# para que el dataset quede listo para análisis estadístico y modelos de ML.
# Se usan mapeos ordinales, rangos promedio y codificación binaria.
print("="*70)
print("11. TIPOS + CONVERSIÓN")
print("="*70)

print("Tipos de datos antes de conversión:")
print(df.dtypes)
print()

# Mapeos ordinales a numéricos
# hora promedio dentro de cada rango
mapeo_horas_estudio = {
    '0-5 hours': 2.5,
    '6-10 hours': 8,
    '11-15 hours': 13,
    '16-20 hours': 18,
    'More than 20 hours': 22
}
#el punto medio del rango.
mapeo_cgpa = {
    'Below 2.50': 2.25,
    '2.50 - 2.99': 2.75,
    '3.00 - 3.49': 3.25,
    '3.50 - 3.74': 3.62,
    '3.75 - 4.00': 3.87
}
# se crea una escala ordinal.
mapeo_frecuencia = {
    'Never': 0,
    'Occasionally': 1,
    'Sometimes': 2,
    'Often': 3,
    'Always': 4
}
# Escala ordinal creciente
mapeo_stress = {
    'Not at all': 0,
    'Slightly': 1,
    'Moderately': 2,
    'Significantly': 3
}
# Otra variable de rango convertida a promedio.
mapeo_mobile = {
    '1-2 hours': 1.5,
    '3-4 hours': 3.5,
    'More than 4 hours': 5
}

mapeo_binario = {
    'Yes': 1,
    'No': 0
}

# Aplicar transformaciones
df['study_hours_per_week'] = df['study_hours_per_week'].map(mapeo_horas_estudio)
df['cgpa'] = df['cgpa'].map(mapeo_cgpa)
df['assignment_delay_frequency'] = df['assignment_delay_frequency'].map(mapeo_frecuencia)
df['stress_due_to_procrastination'] = df['stress_due_to_procrastination'].map(mapeo_stress)
df['use_of_time_management'] = df['use_of_time_management'].map(mapeo_frecuencia)
df['hours_spent_on_mobile_non_academic'] = df['hours_spent_on_mobile_non_academic'].map(mapeo_mobile)
df['procrastination_and_grade_outcome'] = df['procrastination_and_grade_outcome'].map(mapeo_binario)

print("Transformaciones aplicadas:")
print("  - study_hours_per_week: rangos a valores numéricos")
print("  - cgpa: rangos a valores numéricos")
print("  - assignment_delay_frequency: ordinal a numérico (0-4)")
print("  - stress_due_to_procrastination: ordinal a numérico (0-3)")
print("  - use_of_time_management: ordinal a numérico (0-4)")
print("  - hours_spent_on_mobile_non_academic: rangos a valores numéricos")
print("  - procrastination_and_grade_outcome: binario (0/1)")
print()

print("Tipos después de conversión:")
print(df.dtypes)
print()

# -------------
# 12. FILTRADO
# -------------
print("="*70)
print("12. FILTRADO")
print("="*70)

# Verificar nulos después del mapeo
nulos_post_mapeo = df.isnull().sum()
if nulos_post_mapeo.sum() > 0:
    print("NaNs encontrados después del mapeo:")
    print(nulos_post_mapeo[nulos_post_mapeo > 0])
    filas_pre_filtro = len(df)
    df = df.dropna()
    print(f"Filas eliminadas: {filas_pre_filtro - len(df)}")
else:
    print("No hay NaNs después del mapeo")

print(f"Dimensión final: {df.shape}")
print()

# --------------------------
# 13. AGRUPACIÓN/AGREGACIÓN
# --------------------------
# Separa los estudiantes según qué tanto procrastinan y  cómo cambian sus notas, horas de estudio y estrés”
print("="*70)
print("13. AGRUPACIÓN/AGREGACIÓN")
print("="*70)

agrupacion = df.groupby('assignment_delay_frequency').agg({
    'cgpa': ['mean', 'std', 'count'],
    'study_hours_per_week': 'mean',
    'stress_due_to_procrastination': 'mean',
    'procrastination_and_grade_outcome': 'mean'
}).round(2)

print("Estadísticas por frecuencia de retraso en tareas:")
print(agrupacion)
print()

# ----------------------
# 14. ONE HOT ENCODING
# ----------------------
print("="*70)
print("14. ONE HOT ENCODING")
print("="*70)

# Guardar copia antes de One Hot
df_pre_onehot = df.copy()

# One Hot a procrastination_reasons
df_onehot = pd.get_dummies(df, columns=['procrastination_reasons'], prefix='reason')

print(f"Variable codificada: procrastination_reasons")
print(f"Columnas antes: {df.shape[1]}")
print(f"Columnas después: {df_onehot.shape[1]}")
print(f"Nuevas columnas creadas: {df_onehot.shape[1] - df.shape[1]}")
print(f"\nNuevas variables binarias:")
reason_cols = [col for col in df_onehot.columns if col.startswith('reason_')]
for i, col in enumerate(reason_cols[:5], 1):
    print(f"  {i}. {col}")
if len(reason_cols) > 5:
    print(f"  ... y {len(reason_cols) - 5} más")
print()

# ---------------------------
# 15. EDA - TENDENCIA CENTRAL
# --------------------------
# ¿Cuántas horas estudian en promedio?, ¿Cuál es el CGPA promedio?, ¿Qué tan frecuente procrastinan en promedio?
print("="*70)
print("15. EDA - TENDENCIA CENTRAL")
print("="*70)

vars_numericas = ['study_hours_per_week', 'cgpa', 'assignment_delay_frequency',
                  'stress_due_to_procrastination', 'use_of_time_management',
                  'hours_spent_on_mobile_non_academic', 'procrastination_and_grade_outcome']

tendencia = df[vars_numericas].agg(['mean', 'median'])
print("Medidas de tendencia central:")
print(tendencia.round(2))
print()

# --------------------------
# 16. EDA - DISPERSIÓN
# -----------
# ¿Hay mucha diferencia entre estudiantes? ¿Hay valores muy extremos? ¿Qué tan variable es el estrés o el CGPA?
print("="*70)
print("16. EDA - DISPERSIÓN")
print("="*70)

dispersion = df[vars_numericas].agg(['std', 'var', 'min', 'max'])
print("Medidas de dispersión:")
print(dispersion.round(2))
print()

print("Rangos:")
for var in vars_numericas:
    rango = df[var].max() - df[var].min()
    print(f"  {var}: {rango:.2f}")
print()

# ----------------------------------------
# 17. EDA - POSICIÓN + OUTLIERS (IQR)
# ------------------------------------------
# Calculamos cuartiles e identificamos valores atípicos
# usando el método del IQR.

print("="*70)
print("17. EDA - POSICIÓN + OUTLIERS (IQR)")
print("="*70)

outliers_totales = 0
for var in vars_numericas:
    Q1 = df[var].quantile(0.25)
    Q2 = df[var].quantile(0.50)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    
    outliers = df[(df[var] < limite_inf) | (df[var] > limite_sup)]
    outliers_totales += len(outliers)
    
    print(f"\n{var}:")
    print(f"  Q1 (25%): {Q1:.2f}")
    print(f"  Q2 (50%): {Q2:.2f}")
    print(f"  Q3 (75%): {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Límites: [{limite_inf:.2f}, {limite_sup:.2f}]")
    print(f"  Outliers: {len(outliers)}")

print(f"\nTotal de outliers detectados: {outliers_totales}")
print()

# Gráfico Boxplots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, var in enumerate(vars_numericas):
    axes[idx].boxplot(df[var].dropna(), vert=True)
    axes[idx].set_title(f'{var}', fontsize=10)
    axes[idx].set_ylabel('Valor')
    axes[idx].grid(True, alpha=0.3)

axes[7].axis('off')
plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/01_boxplots_outliers.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 01_boxplots_outliers.png")
print()

# -----------------------
# 18. HISTOGRAMAS
# -------------------------
# Visualizamos la distribución de cada variable
# para ver su forma y concentración de valores.

print("="*70)
print("18. HISTOGRAMAS")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, var in enumerate(vars_numericas):
    axes[idx].hist(df[var].dropna(), bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    axes[idx].set_title(f'{var}', fontsize=10)
    axes[idx].set_xlabel('Valor')
    axes[idx].set_ylabel('Frecuencia')
    axes[idx].grid(True, alpha=0.3)

axes[7].axis('off')
plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/02_histogramas.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 02_histogramas.png")
print()

# ----------------------
# 19. SCATTER PLOTS
# ----------------------
# Analizamos relaciones entre variables
# para identificar posibles patrones o tendencias.

print("="*70)
print("19. SCATTER PLOTS")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

axes[0].scatter(df['study_hours_per_week'], df['cgpa'], alpha=0.5, color='steelblue')
axes[0].set_xlabel('Horas de estudio/semana')
axes[0].set_ylabel('CGPA')
axes[0].set_title('CGPA vs Horas de Estudio')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(df['assignment_delay_frequency'], df['cgpa'], alpha=0.5, color='coral')
axes[1].set_xlabel('Frecuencia de retraso')
axes[1].set_ylabel('CGPA')
axes[1].set_title('CGPA vs Frecuencia de Retraso')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(df['hours_spent_on_mobile_non_academic'], df['study_hours_per_week'], 
                alpha=0.5, color='green')
axes[2].set_xlabel('Horas en móvil')
axes[2].set_ylabel('Horas de estudio')
axes[2].set_title('Horas Estudio vs Móvil')
axes[2].grid(True, alpha=0.3)

axes[3].scatter(df['assignment_delay_frequency'], df['stress_due_to_procrastination'], 
                alpha=0.5, color='purple')
axes[3].set_xlabel('Frecuencia de retraso')
axes[3].set_ylabel('Nivel de estrés')
axes[3].set_title('Estrés vs Retraso')
axes[3].grid(True, alpha=0.3)

axes[4].scatter(df['use_of_time_management'], df['cgpa'], alpha=0.5, color='orange')
axes[4].set_xlabel('Uso de gestión del tiempo')
axes[4].set_ylabel('CGPA')
axes[4].set_title('CGPA vs Gestión del Tiempo')
axes[4].grid(True, alpha=0.3)

axes[5].scatter(df['hours_spent_on_mobile_non_academic'], df['stress_due_to_procrastination'], 
                alpha=0.5, color='red')
axes[5].set_xlabel('Horas en móvil')
axes[5].set_ylabel('Nivel de estrés')
axes[5].set_title('Estrés vs Horas Móvil')
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/03_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 03_scatter_plots.png")
print()

# ---------------------------------------------
# 20. PROBABILIDADES/PROPORCIONES CATEGÓRICAS
# ---------------------------------------------
# Calculamos proporciones de variables categóricas
# para entender su distribución en el dataset.

print("="*70)
print("20. PROBABILIDADES/PROPORCIONES CATEGÓRICAS")
print("="*70)

print("Proporciones de procrastination_and_grade_outcome:")
props_grade = df['procrastination_and_grade_outcome'].value_counts(normalize=True).sort_index()
print(props_grade)
print()

print("Proporciones de assignment_delay_frequency:")
props_delay = df['assignment_delay_frequency'].value_counts(normalize=True).sort_index()
print(props_delay)
print()

print("Proporciones de stress_due_to_procrastination:")
props_stress = df['stress_due_to_procrastination'].value_counts(normalize=True).sort_index()
print(props_stress)
print()

# Gráficos de barras
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

df['procrastination_and_grade_outcome'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Impacto en Calificaciones')
axes[0].set_xlabel('0=No, 1=Sí')
axes[0].set_ylabel('Frecuencia')
axes[0].tick_params(axis='x', rotation=0)
axes[0].grid(True, alpha=0.3)

df['assignment_delay_frequency'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Frecuencia de Retraso')
axes[1].set_xlabel('0=Never ... 4=Always')
axes[1].set_ylabel('Frecuencia')
axes[1].tick_params(axis='x', rotation=0)
axes[1].grid(True, alpha=0.3)

df['stress_due_to_procrastination'].value_counts().sort_index().plot(kind='bar', ax=axes[2], color='green')
axes[2].set_title('Nivel de Estrés')
axes[2].set_xlabel('0=Nada ... 3=Significativo')
axes[2].set_ylabel('Frecuencia')
axes[2].tick_params(axis='x', rotation=0)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/04_proporciones_categoricas.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 04_proporciones_categoricas.png")
print()

# ------------------------------------------
# 21. ENCODINGS: ONEHOT + LABEL + BINARY
# -------------------------------------------
# Mostramos los distintos tipos de codificación aplicados:
# One Hot para variables nominales, Label Encoding para categóricas
# y codificación binaria para variables Sí/No.

print("="*70)
print("21. ENCODINGS: ONEHOT + LABEL + BINARY")
print("="*70)

print("ONE HOT ENCODING:")
print(f"  Ya aplicado a: procrastination_reasons")
print(f"  Columnas generadas: {len(reason_cols)}")
print()

print("LABEL ENCODING:")
le = LabelEncoder()
df['assignment_delay_label'] = le.fit_transform(df['assignment_delay_frequency'])
print(f"  Variable: assignment_delay_frequency")
print(f"  Clases: {list(le.classes_)}")
print(f"  Codificación: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print()

print("BINARY ENCODING:")
print(f"  Variable: procrastination_and_grade_outcome")
print(f"  Valores: 0 (No) y 1 (Yes)")
print(f"  Distribución: {df['procrastination_and_grade_outcome'].value_counts().to_dict()}")
print()

# ---------------------------
# 22. CORRELACIÓN PEARSON
# --------------------------
# 22. Calculamos correlación Pearson (relación lineal) y guardamos el heatmap.

print("="*70)
print("22. CORRELACIÓN PEARSON")
print("="*70)

corr_pearson = df[vars_numericas].corr(method='pearson')
print("Matriz de correlación de Pearson:")
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

plt.figure(figsize=(10, 8))
sns.heatmap(corr_pearson, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación de Pearson', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/05_correlacion_pearson.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 05_correlacion_pearson.png")
print()

# -----------------------------
# 23. CORRELACIÓN SPEARMAN
# -----------------------------
# 23. Calculamos correlación Spearman (por rangos/monótona) y guardamos el heatmap.

print("="*70)
print("23. CORRELACIÓN SPEARMAN")
print("="*70)

corr_spearman = df[vars_numericas].corr(method='spearman')
print("Matriz de correlación de Spearman:")
print(corr_spearman.round(2))
print()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_spearman, annot=True, fmt='.2f', cmap='viridis', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación de Spearman', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/06_correlacion_spearman.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 06_correlacion_spearman.png")
print()

# ------------------------------------
# 24. DECISIÓN DE ELIMINAR VARIABLES
# --------------------------------------
# 24. Revisamos si hay variables muy correlacionadas (|r|>0.85) para decidir si eliminar alguna.

print("="*70)
print("24. DECISIÓN DE ELIMINAR VARIABLES")
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
    print("Todas las variables aportan información única")
print()

# ---------------------
# 25. ESCALADO MINMAX
# ---------------------
#  Escalamos con MinMax para llevar variables numéricas al rango [0, 1].


print("="*70)
print("25. ESCALADO MINMAX")
print("="*70)

scaler_minmax = MinMaxScaler()
df_minmax = df.copy()
df_minmax[vars_numericas] = scaler_minmax.fit_transform(df[vars_numericas])

print("MinMaxScaler aplicado a variables numéricas")
print("\nEstadísticas después del escalado:")
print(df_minmax[vars_numericas].describe().loc[['min', 'max']].round(4))
print()

# -------------------
# 26. ESCALADO STANDARD
# -----------------------
# Escalamos con StandardScaler para centrar datos (media 0) y normalizar (std 1).

print("="*70)
print("26. ESCALADO STANDARD")
print("="*70)

scaler_standard = StandardScaler()
df_standard = df.copy()
df_standard[vars_numericas] = scaler_standard.fit_transform(df[vars_numericas])

print("StandardScaler aplicado a variables numéricas")
print("\nEstadísticas después del escalado:")
print(df_standard[vars_numericas].describe().loc[['mean', 'std']].round(4))
print()

# Gráfico comparativo
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

var_ejemplo = 'cgpa'
axes[0].hist(df[var_ejemplo], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_title('Original')
axes[0].set_xlabel(var_ejemplo)
axes[0].set_ylabel('Frecuencia')
axes[0].grid(True, alpha=0.3)

axes[1].hist(df_minmax[var_ejemplo], bins=15, edgecolor='black', alpha=0.7, color='coral')
axes[1].set_title('MinMax Scaled')
axes[1].set_xlabel(var_ejemplo)
axes[1].set_ylabel('Frecuencia')
axes[1].grid(True, alpha=0.3)

axes[2].hist(df_standard[var_ejemplo], bins=15, edgecolor='black', alpha=0.7, color='green')
axes[2].set_title('Standard Scaled')
axes[2].set_xlabel(var_ejemplo)
axes[2].set_ylabel('Frecuencia')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/07_comparacion_escalado.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 07_comparacion_escalado.png")
print()

# ---------------------
# 27. LOG TRANSFORM
# ----------------------
# Aplicamos log(1+x) a horas de móvil para reducir sesgo y suavizar valores extremos.

print("="*70)
print("27. LOG TRANSFORM")
print("="*70)

var_log = 'hours_spent_on_mobile_non_academic'
df['log_mobile_hours'] = np.log1p(df[var_log])

print(f"Log transform aplicado a: {var_log}")
print("\nEstadísticas ANTES del log:")
print(df[var_log].describe().round(2))
print("\nEstadísticas DESPUÉS del log:")
print(df['log_mobile_hours'].describe().round(2))
print()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df[var_log], bins=15, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_title('Original')
axes[0].set_xlabel(var_log)
axes[0].set_ylabel('Frecuencia')
axes[0].grid(True, alpha=0.3)

axes[1].hist(df['log_mobile_hours'], bins=15, edgecolor='black', alpha=0.7, color='coral')
axes[1].set_title('Log Transform')
axes[1].set_xlabel('log(mobile_hours + 1)')
axes[1].set_ylabel('Frecuencia')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CARPETA_GRAFICOS}/08_log_transform.png', dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 08_log_transform.png")
print()

# -----------------------------
# 28. GUARDAR DATASET PROCESADO
# -------------------------------
# 28. Creamos el dataset final (escalado + features extra) y lo guardamos en CSV.

print("="*70)
print("28. GUARDAR DATASET PROCESADO")
print("="*70)

df_final = df_standard.copy()
df_final['log_mobile_hours'] = df['log_mobile_hours']
df_final['assignment_delay_label'] = df['assignment_delay_label']

df_final.to_csv(ARCHIVO_OUTPUT_CSV, index=False)

# Reducción de dimensionalidad y visualizaciones (PCA, t-SNE, UMAP)
# Se generan y guardan embeddings (.csv), modelos (.joblib) y gráficos en la carpeta de gráficos
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Usamos las variables numéricas escaladas para calcular embeddings
features_for_emb = df_standard[vars_numericas]

# PCA (2 componentes)
pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(features_for_emb)
pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
pca_df['assignment_delay_label'] = df_final['assignment_delay_label'].values
pca_df.to_csv(os.path.join(CARPETA_GRAFICOS, 'pca_embeddings.csv'), index=False)
joblib.dump(pca, os.path.join(MODELS_DIR, 'pca_model.joblib'))

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='assignment_delay_label', palette='viridis', alpha=0.8)
plt.title('PCA (2 componentes)')
plt.legend(title='assignment_delay_label')
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, '09_pca.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 09_pca.png")

# t-SNE
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
tsne_components = tsne.fit_transform(features_for_emb)
tsne_df = pd.DataFrame(tsne_components, columns=['Dim1', 'Dim2'])
tsne_df['assignment_delay_label'] = df_final['assignment_delay_label'].values
tsne_df.to_csv(os.path.join(CARPETA_GRAFICOS, 'tsne_embeddings.csv'), index=False)
joblib.dump(tsne, os.path.join(MODELS_DIR, 'tsne_model.joblib'))

plt.figure(figsize=(8, 6))
sns.scatterplot(data=tsne_df, x='Dim1', y='Dim2', hue='assignment_delay_label', palette='viridis', alpha=0.8)
plt.title('t-SNE (2 dimensiones)')
plt.legend(title='assignment_delay_label')
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, '10_tsne.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico guardado: 10_tsne.png")

# UMAP (si está disponible)
if umap is not None:
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_components = umap_model.fit_transform(features_for_emb)
    umap_df = pd.DataFrame(umap_components, columns=['UMAP1', 'UMAP2'])
    umap_df['assignment_delay_label'] = df_final['assignment_delay_label'].values
    umap_df.to_csv(os.path.join(CARPETA_GRAFICOS, 'umap_embeddings.csv'), index=False)
    joblib.dump(umap_model, os.path.join(MODELS_DIR, 'umap_model.joblib'))

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='assignment_delay_label', palette='viridis', alpha=0.8)
    plt.title('UMAP (2 dimensiones)')
    plt.legend(title='assignment_delay_label')
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_GRAFICOS, '11_umap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico guardado: 11_umap.png")
else:
    print("UMAP no está disponible (biblioteca no instalada). Para activar UMAP, instale 'umap-learn'.")

print(f"Dataset guardado: {ARCHIVO_OUTPUT_CSV}")
print(f"Dimensiones: {df_final.shape[0]} filas x {df_final.shape[1]} columnas")
print(f"\nVariables incluidas:")
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
print(f"     - 04_proporciones_categoricas.png")
print(f"     - 05_correlacion_pearson.png")
print(f"     - 06_correlacion_spearman.png")
print(f"     - 07_comparacion_escalado.png")
print(f"     - 08_log_transform.png")
print()
print(f"DATASET ORIGINAL: {df_original.shape[0]} filas")
print(f"DATASET PROCESADO: {df_final.shape[0]} filas")
print(f"FILAS ELIMINADAS: {df_original.shape[0] - df_final.shape[0]}")
print()
print("="*70)

# Guardar outputs
sys.stdout = sys.__stdout__
with open(ARCHIVO_OUTPUT_TXT, 'w', encoding='utf-8') as f:
    f.writelines(capture.outputs)

print(f"\nArchivo de resultados guardado: {ARCHIVO_OUTPUT_TXT}")
print("\n¡PIPELINE EJECUTADO EXITOSAMENTE!")