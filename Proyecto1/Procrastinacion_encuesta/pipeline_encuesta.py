"""
Pipeline Análisis Encuesta Procrastinación
Maria Alejandra Ocampo Giraldo

"""
# Importación de librerías para manipulación de datos, visualización de gráficos,
# preprocesamiento para Machine Learning y manejo de rutas del sistema.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys

# Configuración visual de gráficos y definición de rutas para outputs del pipeline.
# Config
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Rutas 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_GRAFICOS = os.path.join(SCRIPT_DIR, "graficos")
ARCHIVO_OUTPUT_CSV = os.path.join(SCRIPT_DIR, "data_procesada.csv")
ARCHIVO_OUTPUT_TXT = os.path.join(SCRIPT_DIR, "resultados_analisis.txt")

os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

# Capturar outputs de consola y almacenarla para generar un archivo de resultados.
class OutputCapture:
    def __init__(self):
        self.outputs = []
        #e guarda el texto en la lista outputs.
    def write(self, text):
        self.outputs.append(text)
        sys.__stdout__.write(text)
    def flush(self):
        pass

capture = OutputCapture()
sys.stdout = capture

print("="*70)
print("ANÁLISIS ENCUESTA PROCRASTINACIÓN")
print("="*70)
print()

# Localizar el archivo CSV en rutas probables
csv_name = "procastinacion_encuesta.csv"
## Definición del nombre del dataset y posibles rutas para hacerlo portable y evitar errores de ubicación.
candidate_paths = [
    os.path.join(SCRIPT_DIR, csv_name),
    os.path.join(os.path.dirname(SCRIPT_DIR), "Procrastinacion_encuesta", csv_name),
    os.path.join(os.path.dirname(SCRIPT_DIR), "Procrastinacion_Encuesta", csv_name),
    os.path.join(os.getcwd(), csv_name)
]

# Bsqueda de la ruta válida del CSV y manejo de error si no se encuentra.
csv_path = None
for p in candidate_paths:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    sys.stdout = sys.__stdout__
    raise FileNotFoundError(f"No se encontró {csv_name}. Busqué en: {candidate_paths}")

# Carga conjunto de datos y verificar tamaño.
df_original = pd.read_csv(csv_path) #Carga el archivo CSV en un DataFrame de pandas.
print(f"1. CARGA: {df_original.shape[0]} filas x {df_original.shape[1]} columnas\n")

# Inspección rápida
#  columnas, tipos de datos y primeras filas.
print("="*70)
print("2. INSPECCIÓN RÁPIDA")
print("="*70)
print("Archivo leído:", os.path.relpath(csv_path, SCRIPT_DIR))
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

# Selección variables
# Selección explícita de variables relevantes para el análisis de procrastinación,
# basadas en la pregunta de investigación y su pertinencia estadística.
variables = [
    'frecuencia_planificacion_tareas',
    'frecuencia_dividir_tareas',
    'frecuencia_autonomia_sin_presion',
    'frecuencia_concentracion_estudio',
    'frecuencia_evitar_tareas_dificiles',
    'frecuencia_revisar_celular_estudio',
    'frecuencia_interferencia_redes',
    'frecuencia_dejar_ultimo_momento'
]

df = df_original[variables].copy()
print(f"3. VARIABLES SELECCIONADAS: {len(variables)}\n")

# Verificación NaNs y duplicados
print("4. VERIFICACIÓN:")
print(f"   NaNs: {df.isnull().sum().sum()}")
print(f"   Duplicados: {df.duplicated().sum()}\n")


# Transformación de variables categóricas tipo Likert a escala numérica ordinal (1–5).
mapeo = {'Nunca': 1, 'Rara vez': 2, 'A veces': 3, 'Frecuentemente': 4, 'Siempre': 5}

df_num = df.copy()
for col in df.columns:
    df_num[col] = df[col].astype(str).str.strip().map(mapeo)

print("5. CONVERSIÓN LIKERT → NUMÉRICO (1-5)")
print("   Nunca→1, Rara vez→2, A veces→3, Frecuentemente→4, Siempre→5\n")

# Estadísticas descriptivas generales
# 6. MEDIDAS DE TENDENCIA CENTRAL
print("6. TENDENCIA CENTRAL:")
print(df_num.agg(['mean', 'median']).round(2))
print()

# MEDIDAS DE DISPERSIÓN
print(" DISPERSIÓN:")
print(df_num.agg(['std', 'var']).round(2))
print()

# MEDIDAS DE POSICIÓN
print("POSICIÓN (percentiles):")
print(df_num.quantile([0.25, 0.5, 0.75]).round(2))
print()

# ESTADÍSTICAS DESCRIPTIVAS GENERALES
print(" ESTADÍSTICAS DESCRIPTIVAS:")
print(df_num.describe().round(2))
print()



# Outliers IQR
# Detección de valores atípicos mediante el método IQR (Q1–Q3 ± 1.5*IQR).
# No se eliminan outliers debido a que los datos provienen de una escala Likert (1–5),
# donde los valores extremos representan respuestas válidas y no errores de medición.

print("8. OUTLIERS (IQR):")
for col in df_num.columns:
    Q1, Q3 = df_num[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = df_num[(df_num[col] < Q1-1.5*IQR) | (df_num[col] > Q3+1.5*IQR)]
    print(f"   {col.replace('frecuencia_', '')}: {len(outliers)} outliers")
print()

# Gráfico 1: Boxplots
# Generación de boxplots para visualizar distribución, dispersión y posibles outliers de cada variable.
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()
for idx, col in enumerate(df_num.columns):
    axes[idx].boxplot(df_num[col].dropna())
    axes[idx].set_title(col.replace('frecuencia_', ''), fontsize=9)
    axes[idx].set_ylabel('Escala 1-5')
    axes[idx].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, '01_boxplots.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico 1: boxplots guardado\n")

# Gráfico 2
# Generación de histogramas para visualizar la distribución de frecuencias
# de cada variable en la escala Likert (1–5).

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()
for idx, col in enumerate(df_num.columns):
    axes[idx].hist(df_num[col].dropna(), bins=5, range=(0.5, 5.5), edgecolor='black', alpha=0.7)
    axes[idx].set_title(col.replace('frecuencia_', ''), fontsize=9)
    axes[idx].set_xticks([1,2,3,4,5])
    axes[idx].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, '02_histogramas.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico 2: histogramas guardado\n")

# Gráfico 3: Scatter plots
# Visualización de posibles relaciones y patrones de correlación entre variables seleccionadas.
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
pares = [
    ('frecuencia_planificacion_tareas', 'frecuencia_dejar_ultimo_momento'),
    ('frecuencia_concentracion_estudio', 'frecuencia_revisar_celular_estudio'),
    ('frecuencia_dividir_tareas', 'frecuencia_evitar_tareas_dificiles'),
    ('frecuencia_autonomia_sin_presion', 'frecuencia_dejar_ultimo_momento'),
    ('frecuencia_interferencia_redes', 'frecuencia_revisar_celular_estudio'),
    ('frecuencia_planificacion_tareas', 'frecuencia_dividir_tareas')
]
for idx, (v1, v2) in enumerate(pares):
    axes[idx].scatter(df_num[v1], df_num[v2], alpha=0.6)
    axes[idx].set_xlabel(v1.replace('frecuencia_', ''), fontsize=8)
    axes[idx].set_ylabel(v2.replace('frecuencia_', ''), fontsize=8)
    axes[idx].set_xticks([1,2,3,4,5])
    axes[idx].set_yticks([1,2,3,4,5])
    axes[idx].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, '03_scatter_plots.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico 3: scatter plots guardado\n")

# Encodings
# Aplicación de diferentes técnicas de codificación (label, binaria y one-hot)
# para preparar variables para modelado en Machine Learning.
print("9. ENCODINGS:")
le = LabelEncoder()
df_num['label_planificacion'] = le.fit_transform(df_num['frecuencia_planificacion_tareas'].astype(str))
df_num['binary_concentracion'] = (df_num['frecuencia_concentracion_estudio'] >= 4).astype(int)
df_onehot = pd.get_dummies(df_num, columns=['frecuencia_dejar_ultimo_momento'], prefix='ultimo')
print(f"   Label Encoding: label_planificacion")
print(f"   Binary Encoding: binary_concentracion (alta≥4)")
print(f"   One Hot: {len([c for c in df_onehot.columns if c.startswith('ultimo')])} columnas\n")

# Correlación Pearson
# Valores positivos (rojo) significan relación directa: cuando una aumenta, la otra también.
# Valores negativos (azul) indican relación inversa: cuando una aumenta, la otra disminuye.
# Valores cercanos a 0 representan poca o ninguna relación lineal.
corr = df_num[variables].corr(method='pearson')
print("10. CORRELACIÓN PEARSON:")
print(corr.round(2))
print()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Correlación Pearson')
plt.tight_layout()
plt.savefig(os.path.join(CARPETA_GRAFICOS, '04_correlacion.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Gráfico 4: correlación guardado\n")

# Decisión eliminar variables
# Evaluación de multicolinealidad: se identifican pares de variables con
# correlación alta (>0.85) para decidir si es necesario eliminar alguna.
print("11. DECISIÓN ELIMINAR VARIABLES:")
alta_corr = False
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.85:
            print(f"    Alta: {corr.columns[i]} <-> {corr.columns[j]}: {corr.iloc[i, j]:.2f}")
            alta_corr = True
if not alta_corr:
    print("    No se eliminan variables (correlación < 0.85)\n")

# Escalado
# Estandarización de variables usando StandardScaler para centrar los datos
# en media 0 y desviación estándar 1, preparándolos para modelado.
scaler = StandardScaler()
df_escalado = pd.DataFrame(
    scaler.fit_transform(df_num[variables].fillna(df_num[variables].mean())),
    columns=variables
)
print("12. ESCALADO STANDARDSCALER:")
print(df_escalado.describe().loc[['mean', 'std']].round(4))
print()




# Guardar CSV procesado
# Creación del dataset final listo para modelado, incluyendo variables escaladas
# y variables adicionales generadas, y guardado en un nuevo archivo CSV.
df_final = df_escalado.copy()
df_final['label_planificacion'] = df_num['label_planificacion'].values
df_final['binary_concentracion'] = df_num['binary_concentracion'].values
df_final.to_csv(ARCHIVO_OUTPUT_CSV, index=False)

print("="*70)
print("PIPELINE COMPLETADO")
print("="*70)
print(f"\nArchivos generados:")
print(f"  - {os.path.basename(ARCHIVO_OUTPUT_CSV)} ({df_final.shape[0]} filas x {df_final.shape[1]} columnas)")
print(f"  - {os.path.join('graficos','01_boxplots.png')}")
print(f"  - {os.path.join('graficos','02_histogramas.png')}")
print(f"  - {os.path.join('graficos','03_scatter_plots.png')}")
print(f"  - {os.path.join('graficos','04_correlacion.png')}")
print()

# Conclusiones
print("CONCLUSIONES:")
print(f"1. Promedio procrastinación: {df_num['frecuencia_dejar_ultimo_momento'].mean():.2f}/5")
print(f"2. Variable más alta: {df_num.mean().idxmax().replace('frecuencia_', '')}")
print(f"3. Variable más baja: {df_num.mean().idxmin().replace('frecuencia_', '')}")
corr_vals = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
print(f"4. Correlación más fuerte: {corr_vals.iloc[1]:.2f}")
print()

# Guardar TXT
sys.stdout = sys.__stdout__
with open(ARCHIVO_OUTPUT_TXT, 'w', encoding='utf-8') as f:
    f.writelines(capture.outputs)

print(f"Resultados guardados en: {os.path.basename(ARCHIVO_OUTPUT_TXT)}")
print("¡ÉXITO!")
