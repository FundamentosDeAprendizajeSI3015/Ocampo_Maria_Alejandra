# =============================================================
# preprocesamiento.py
# Carga, limpieza, conversión Likert y EDA del dataset.
# NO se crean etiquetas aquí — eso ocurre en clustering.py
# =============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')           # backend sin ventanas (para guardar PNGs)
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────────────
# RUTAS  (relativas al script; funciona desde cualquier directorio)
# ─────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_ORIG  = os.path.join(
    BASE_DIR, "..",
    "Factores que influyen en la procrastinación académica en estudiantes universitarios(1-48).csv"
)
DATA_DIR  = os.path.join(BASE_DIR, "data")
GRAF_DIR  = os.path.join(BASE_DIR, "graficos_preprocesamiento")

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(GRAF_DIR,  exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. CARGA DEL DATASET
# ─────────────────────────────────────────────────────────────
df_raw = pd.read_csv(CSV_ORIG)
print(f"Registros cargados : {len(df_raw)}")
print(f"Columnas originales: {list(df_raw.columns)}\n")

# ─────────────────────────────────────────────────────────────
# 2. ELIMINAR COLUMNAS ADMINISTRATIVAS
#    Se eliminan las primeras 6 columnas:
#    ID | Hora inicio | Hora fin | Correo | Nombre | Última modificación
#    Solo se conservan las 8 preguntas Likert.
# ─────────────────────────────────────────────────────────────
df = df_raw.iloc[:, 6:].copy()   # columnas índice 6 en adelante

# ─────────────────────────────────────────────────────────────
# 3. RENOMBRAR COLUMNAS  (preguntas → nombres cortos descriptivos)
# ─────────────────────────────────────────────────────────────
df.columns = [
    "planificacion",   # ¿Con qué frecuencia planificas tus tareas académicas?
    "organizacion",    # ¿Divides tareas grandes en partes más pequeñas?
    "autonomia",       # ¿Cumples sin presión externa (recordatorios/supervisión)?
    "concentracion",   # ¿Mantienes concentración durante el estudio?
    "evitacion",       # ¿Evitas iniciar tareas difíciles o poco interesantes?
    "uso_celular",     # ¿Revisas el celular mientras estudias?
    "redes_sociales",  # ¿Las redes sociales interfieren con tu tiempo de estudio?
    "procrastinacion", # ¿Dejas tareas para el último momento?
]
FEATURES = df.columns.tolist()
print(f"Columnas renombradas: {FEATURES}\n")

# ─────────────────────────────────────────────────────────────
# 4. CONVERTIR ESCALA LIKERT → NUMÉRICO  (1–5)
#    1=Nunca  2=Rara vez  3=A veces  4=Frecuentemente  5=Siempre
# ─────────────────────────────────────────────────────────────
LIKERT_MAP = {
    "Nunca":          1,
    "Rara vez":       2,
    "A veces":        3,
    "Frecuentemente": 4,
    "Siempre":        5,
}

for col in FEATURES:
    df[col] = df[col].astype(str).str.strip()   # eliminar espacios residuales
    df[col] = df[col].map(LIKERT_MAP)

# ─────────────────────────────────────────────────────────────
# 5. MANEJO DE VALORES NULOS
#    Usamos la mediana por columna: es robusta para escala ordinal
# ─────────────────────────────────────────────────────────────
nulos = df.isnull().sum()
print(f"Valores nulos por columna:\n{nulos}\n")

for col in FEATURES:
    if df[col].isnull().any():
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)
        print(f"  → '{col}' imputada con mediana = {mediana}")

df = df.astype(int)

print(f"\nShape final: {df.shape}")
print(df.describe().round(2))
print()

# ─────────────────────────────────────────────────────────────
# 6. EDA — ANÁLISIS EXPLORATORIO DE DATOS
# ─────────────────────────────────────────────────────────────

# ── 6.1  Distribución de respuestas por variable (barras de conteo)
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
palette = sns.color_palette("Set2", 5)
etiquetas_x = ['N(1)', 'RV(2)', 'AV(3)', 'F(4)', 'S(5)']

for i, col in enumerate(FEATURES):
    counts = df[col].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    bars = axes[i].bar(counts.index, counts.values, color=palette,
                       edgecolor='black', width=0.7)
    axes[i].set_title(col.replace('_', ' ').capitalize(), fontsize=11, fontweight='bold')
    axes[i].set_xlabel("Likert", fontsize=9)
    axes[i].set_ylabel("Frecuencia", fontsize=9)
    axes[i].set_xticks([1, 2, 3, 4, 5])
    axes[i].set_xticklabels(etiquetas_x, fontsize=8)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            axes[i].text(bar.get_x() + bar.get_width()/2, h + 0.15,
                         str(int(h)), ha='center', va='bottom', fontsize=8)

plt.suptitle(
    "Distribución de respuestas por variable\n"
    "N=Nunca | RV=Rara vez | AV=A veces | F=Frecuente | S=Siempre",
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "01_distribuciones.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 01_distribuciones.png")

# ── 6.2  Boxplots comparativos
fig, ax = plt.subplots(figsize=(12, 6))
bp = df.boxplot(ax=ax, grid=False, patch_artist=True,
                medianprops=dict(color='red', linewidth=2.5),
                boxprops=dict(facecolor='lightblue', alpha=0.7))
ax.set_title("Boxplot comparativo por variable", fontsize=13, fontweight='bold')
ax.set_ylabel("Escala Likert (1-5)", fontsize=11)
ax.tick_params(axis='x', rotation=30, labelsize=10)
ax.set_ylim(0.5, 5.5)
ax.axhline(3, color='gray', linestyle='--', alpha=0.6, label='Punto medio (3)')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "02_boxplots.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 02_boxplots.png")

# ── 6.3  Mapa de calor de correlaciones (Pearson)
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(
    corr, annot=True, fmt=".2f", cmap='RdYlGn_r', center=0,
    ax=ax, square=True, linewidths=0.8, vmin=-1, vmax=1,
    cbar_kws={"shrink": 0.8, "label": "Pearson r"}
)
ax.set_title("Mapa de correlación entre variables (Pearson)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "03_correlaciones.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 03_correlaciones.png")

# ── 6.4  Histogramas con curva KDE
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, col in enumerate(FEATURES):
    sns.histplot(df[col], kde=True, bins=5, ax=axes[i],
                 color='darkorange', edgecolor='black', alpha=0.7)
    axes[i].set_title(col.replace('_', ' ').capitalize(), fontsize=11, fontweight='bold')
    axes[i].set_xlabel("Valor Likert")
    axes[i].set_xlim(0.5, 5.5)

plt.suptitle("Histogramas con densidad estimada (KDE)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "04_histogramas_kde.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 04_histogramas_kde.png")

# ── 6.5  Pairplot de las 4 variables más relacionadas con procrastinación
vars_clave = ["planificacion", "uso_celular", "redes_sociales", "procrastinacion"]
pp = sns.pairplot(df[vars_clave], diag_kind='kde',
                  plot_kws={'alpha': 0.6, 'color': 'steelblue'},
                  height=2.5)
pp.fig.suptitle("Pairplot — Variables clave (planificación, celular, redes, procrastinación)",
                y=1.02, fontsize=12, fontweight='bold')
pp.savefig(os.path.join(GRAF_DIR, "05_pairplot.png"), dpi=130, bbox_inches='tight')
plt.close()
print("Guardado: 05_pairplot.png")

# ── 6.6  Radar chart — perfil promedio del estudiante encuestado
medias = df.mean()
N = len(FEATURES)
valores = medias.values.tolist()
valores += valores[:1]     # cerrar el polígono
angulos = [n / float(N) * 2 * np.pi for n in range(N)]
angulos += angulos[:1]
etiq_radar = [c.replace('_', '\n') for c in FEATURES]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.fill(angulos, valores, color='steelblue', alpha=0.25)
ax.plot(angulos, valores, 'o-', color='steelblue', linewidth=2)
ax.set_xticks(angulos[:-1])
ax.set_xticklabels(etiq_radar, fontsize=10)
ax.set_ylim(0, 5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.yaxis.set_tick_params(labelsize=8)
ax.set_title("Perfil promedio del estudiante\n(Media por variable)",
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "06_radar_perfil.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 06_radar_perfil.png")

# ── 6.7  Tabla de estadísticas descriptivas como imagen
desc = df.describe().round(2)
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
tabla = ax.table(
    cellText=desc.values,
    colLabels=desc.columns,
    rowLabels=desc.index,
    cellLoc='center',
    loc='center'
)
tabla.auto_set_font_size(False)
tabla.set_fontsize(9)
tabla.scale(1.2, 1.6)
ax.set_title("Estadísticas descriptivas del dataset limpio",
             fontsize=13, fontweight='bold', pad=15, loc='center')
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "07_estadisticas.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 07_estadisticas.png")

# ─────────────────────────────────────────────────────────────
# 7. GUARDAR DATA LIMPIA
# ─────────────────────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, "data_limpia.csv")
df.to_csv(out_path, index=False)
print(f"\nOK data_limpia.csv guardado en: {out_path}")
print("-" * 55)
print("preprocesamiento.py completado exitosamente.")
