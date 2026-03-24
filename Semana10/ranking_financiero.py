"""
Ranking financiero por facultad — UdeA
Responde: ¿cuáles son los peores financieros y se deben echar?

Correr con:  python ranking_financiero.py
         o:  py -3.11 ranking_financiero.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rc('font', family='serif', size=11)

# ══════════════════════════════════════════════════════════════════
#  CONFIGURACION — cambia el dataset aqui
# ══════════════════════════════════════════════════════════════════
DATASET_PATH = r'C:\Users\maria\OneDrive\Imágenes\Escritorio\Automatico\Ocampo_Maria_Alejandra\Semana10\dataset_sintetico_FIRE_UdeA_realista.csv'

# Indicadores donde MAYOR = MEJOR (se normaliza directo)
INDICADORES_POSITIVOS = ['liquidez', 'dias_efectivo', 'cfo',
                          'tendencia_ingresos']

# Indicadores donde MENOR = MEJOR (se invierte en el score)
INDICADORES_NEGATIVOS = ['endeudamiento', 'gp_ratio', 'hhi_fuentes']

# Peso de cada indicador en el score final (deben sumar 1)
PESOS = {
    'liquidez':           0.20,  # capacidad de pago
    'dias_efectivo':      0.10,  # dias con efectivo
    'cfo':                0.15,  # flujo de caja operacional
    'tendencia_ingresos': 0.15,  # crecimiento
    'endeudamiento':      0.15,  # deuda (menor es mejor)
    'gp_ratio':           0.15,  # gastos personal / ingresos (menor es mejor)
    'hhi_fuentes':        0.10,  # concentracion de fuentes (menor = mas diverso)
}

# ══════════════════════════════════════════════════════════════════

base_dir = os.path.dirname(os.path.abspath(__file__))
carpeta  = os.path.join(base_dir, 'graficas_ranking')
os.makedirs(carpeta, exist_ok=True)

def guardar(fig, nombre):
    fig.savefig(os.path.join(carpeta, nombre), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Guardada: {nombre}")

# ── 1. Cargar dataset ──────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)
print(f"Dataset: {df.shape[0]} filas x {df.shape[1]} columnas")

# ── 2. One-Hot Encoding de 'unidad' (facultad) ────────────────────
# Esto convierte cada facultad en una columna binaria
ohe = pd.get_dummies(df['unidad'], prefix='fac').astype(int)
facultades = df['unidad'].unique()
print(f"\nFacultades encontradas ({len(facultades)}):")
for f in sorted(facultades):
    print(f"  · {f}")

# ── 3. Imputar NaN con mediana por facultad ───────────────────────
indicadores = list(PESOS.keys())
for col in indicadores:
    df[col] = df.groupby('unidad')[col].transform(lambda x: x.fillna(x.median()))
    df[col] = df[col].fillna(df[col].median())  # fallback global

# ── 4. Calcular score financiero por año-facultad ─────────────────
# Normalizar cada indicador a [0,1] global
df_score = df[['anio', 'unidad', 'label']].copy()

for col in indicadores:
    mn = df[col].min(); mx = df[col].max()
    rng = mx - mn if mx != mn else 1.0
    if col in INDICADORES_NEGATIVOS:
        df_score[col + '_norm'] = 1.0 - (df[col] - mn) / rng   # invertir
    else:
        df_score[col + '_norm'] = (df[col] - mn) / rng

# Score ponderado por fila
df_score['score'] = sum(
    PESOS[col] * df_score[col + '_norm'] for col in indicadores
)

# ── 5. Promedio del score por facultad (todos los años) ───────────
resumen = df_score.groupby('unidad').agg(
    score_medio   = ('score',  'mean'),
    score_std     = ('score',  'std'),
    n_anios       = ('anio',   'count'),
    label_0_pct   = ('label',  lambda x: (x == 0).mean() * 100),   # % inestable
    label_1_pct   = ('label',  lambda x: (x == 1).mean() * 100),   # % estable
).reset_index().sort_values('score_medio')

# Añadir columna con la tendencia temporal del score
tendencias = []
for fac in resumen['unidad']:
    sub = df_score[df_score['unidad'] == fac].sort_values('anio')
    if len(sub) >= 2:
        z = np.polyfit(sub['anio'], sub['score'], 1)
        tendencias.append(z[0])
    else:
        tendencias.append(0.0)
resumen['tendencia_score'] = tendencias

# Ranking: 1 = PEOR
resumen['ranking'] = range(1, len(resumen) + 1)

print("\n" + "="*70)
print("  RANKING FINANCIERO — de PEOR a MEJOR")
print("="*70)
print(f"{'#':<4} {'Facultad':<38} {'Score':<8} {'%Inestable':<12} {'Tendencia'}")
print("-"*70)
for _, row in resumen.iterrows():
    tend_str = "↑ mejora" if row['tendencia_score'] > 0.005 else \
               "↓ empeora" if row['tendencia_score'] < -0.005 else "→ estable"
    print(f"{int(row['ranking']):<4} {row['unidad']:<38} {row['score_medio']:.3f}   "
          f"{row['label_0_pct']:.1f}%        {tend_str}")

# Umbral: los que tienen score < 0.4 se consideran en riesgo
EN_RIESGO = resumen[resumen['score_medio'] < 0.4]
print(f"\n  ► {len(EN_RIESGO)} facultades con score < 0.40 (zona de riesgo):")
for _, r in EN_RIESGO.iterrows():
    print(f"    · {r['unidad']}  (score={r['score_medio']:.3f})")

print("="*70)

# ── 6. Detalle por indicador de los PEORES ────────────────────────
peores_3 = resumen.head(3)['unidad'].tolist()
print(f"\n  Detalle indicadores — 3 PEORES facultades:")
for fac in peores_3:
    sub = df[df['unidad'] == fac][indicadores].mean()
    print(f"\n  {fac}")
    for ind in indicadores:
        val = sub[ind]
        marca = "⚠" if (ind in INDICADORES_NEGATIVOS and val > df[ind].mean()) or \
                       (ind in INDICADORES_POSITIVOS  and val < df[ind].mean()) else " "
        print(f"    {marca} {ind:<22}: {val:.4f}  (media global: {df[ind].mean():.4f})")

# ══════════════════════════════════════════════════════════════════
#  GRAFICAS
# ══════════════════════════════════════════════════════════════════
COLORES_RANK = ['#d62728' if s < 0.35 else '#ff7f0e' if s < 0.45 else '#2ca02c'
                for s in resumen['score_medio']]

# ── Fig 1: Ranking general (barras horizontales) ──────────────────
fig, ax = plt.subplots(figsize=(10, max(5, len(resumen)*0.55)))
bars = ax.barh(resumen['unidad'], resumen['score_medio'],
               color=COLORES_RANK, alpha=0.88, edgecolor='white')
ax.axvline(0.4, color='red', lw=1.5, ls='--', label='Umbral riesgo (0.40)')
ax.axvline(resumen['score_medio'].mean(), color='navy', lw=1.2, ls=':',
           label=f"Media = {resumen['score_medio'].mean():.2f}")
for bar, v in zip(bars, resumen['score_medio']):
    ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
            f'{v:.3f}', va='center', fontsize=9)
ax.set_xlabel('Score financiero (0=peor, 1=mejor)')
ax.set_title('Ranking Financiero por Facultad — UdeA\n'
             'Rojo=zona riesgo | Naranja=alerta | Verde=saludable')
ax.legend(fontsize=9)
ax.set_xlim(0, 1.05)
plt.tight_layout()
guardar(fig, 'fig1_ranking_general.png')

# ── Fig 2: % inestable vs score (dispersión por facultad) ─────────
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(resumen['score_medio'], resumen['label_0_pct'],
                c=resumen['score_medio'], cmap='RdYlGn',
                s=120, zorder=5, vmin=0, vmax=1)
for _, row in resumen.iterrows():
    ax.annotate(row['unidad'].replace('Facultad de ', '').replace('Facultad ', ''),
                (row['score_medio'], row['label_0_pct']),
                xytext=(6, 3), textcoords='offset points', fontsize=8)
plt.colorbar(sc, ax=ax, label='Score financiero')
ax.axvline(0.4, color='red', lw=1.2, ls='--', alpha=0.7, label='Umbral riesgo')
ax.axhline(50, color='gray', lw=1.0, ls=':', alpha=0.7, label='50% inestable')
ax.set_xlabel('Score financiero medio')
ax.set_ylabel('% de años clasificados como inestable (label=0)')
ax.set_title('Score financiero vs Inestabilidad por facultad\n'
             'Abajo-derecha = financieramente fuertes  |  Arriba-izquierda = preocupantes')
ax.legend(fontsize=9)
plt.tight_layout()
guardar(fig, 'fig2_score_vs_inestabilidad.png')

# ── Fig 3: Evolucion temporal del score por facultad ──────────────
df_score_anio = df_score.groupby(['anio', 'unidad'])['score'].mean().reset_index()
fig, ax = plt.subplots(figsize=(10, 6))
COLORES10 = plt.cm.tab10.colors
for i, fac in enumerate(sorted(df_score_anio['unidad'].unique())):
    sub = df_score_anio[df_score_anio['unidad'] == fac].sort_values('anio')
    lw  = 2.8 if fac in peores_3 else 1.2
    ls  = '-'  if fac in peores_3 else '--'
    ax.plot(sub['anio'], sub['score'], lw=lw, ls=ls,
            color=COLORES10[i % 10], marker='o', ms=4,
            label=fac.replace('Facultad de ', '').replace('Facultad ', ''))
ax.axhline(0.4, color='red', lw=1.2, ls=':', alpha=0.7, label='Umbral riesgo')
ax.set_xlabel('Año')
ax.set_ylabel('Score financiero')
ax.set_title('Evolución del score financiero por facultad\n'
             'Línea continua gruesa = peores 3 facultades')
ax.legend(fontsize=7, ncol=2, loc='upper left')
plt.tight_layout()
guardar(fig, 'fig3_evolucion_temporal.png')

# ── Fig 4: Heatmap de indicadores por facultad ────────────────────
medias_fac = df.groupby('unidad')[indicadores].mean()
# Normalizar para el heatmap
medias_norm = medias_fac.copy()
for col in indicadores:
    mn = medias_fac[col].min(); mx = medias_fac[col].max()
    rng = mx - mn if mx != mn else 1.0
    if col in INDICADORES_NEGATIVOS:
        medias_norm[col] = 1.0 - (medias_fac[col] - mn) / rng
    else:
        medias_norm[col] = (medias_fac[col] - mn) / rng

# Ordenar por score medio
orden = resumen.sort_values('score_medio')['unidad'].tolist()
medias_norm = medias_norm.loc[orden]
nombres_cortos = [n.replace('Facultad de ', '').replace('Facultad ', '')
                  for n in medias_norm.index]

fig, ax = plt.subplots(figsize=(12, max(5, len(orden)*0.55)))
im = ax.imshow(medias_norm.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(range(len(indicadores)))
ax.set_xticklabels(indicadores, rotation=35, ha='right')
ax.set_yticks(range(len(nombres_cortos)))
ax.set_yticklabels(nombres_cortos)
# Anotar valores
for i in range(len(orden)):
    for j in range(len(indicadores)):
        v = medias_norm.values[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                fontsize=8, color='black' if 0.3 < v < 0.8 else 'white')
plt.colorbar(im, ax=ax, label='Score normalizado (verde=mejor, rojo=peor)')
ax.set_title('Heatmap financiero por facultad\n'
             'Filas ordenadas de PEOR (arriba) a MEJOR (abajo)')
plt.tight_layout()
guardar(fig, 'fig4_heatmap_indicadores.png')

# ── Fig 5: Veredicto — radar del PEOR vs el MEJOR ─────────────────
peor  = orden[0]
mejor = orden[-1]
cats  = indicadores
N     = len(cats)
angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

vals_peor  = list(medias_norm.loc[peor].values)  + [medias_norm.loc[peor].values[0]]
vals_mejor = list(medias_norm.loc[mejor].values) + [medias_norm.loc[mejor].values[0]]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.plot(angles, vals_peor,  'o-', lw=2, color='#d62728',
        label=peor.replace('Facultad de ', ''))
ax.fill(angles, vals_peor,  alpha=0.15, color='#d62728')
ax.plot(angles, vals_mejor, 'o-', lw=2, color='#2ca02c',
        label=mejor.replace('Facultad de ', ''))
ax.fill(angles, vals_mejor, alpha=0.15, color='#2ca02c')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(cats, size=9)
ax.set_ylim(0, 1)
ax.set_title('Peor vs Mejor facultad\n(mayor área = mejor desempeño)', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
plt.tight_layout()
guardar(fig, 'fig5_radar_peor_vs_mejor.png')

# ── Veredicto final ────────────────────────────────────────────────
print("\n" + "="*70)
print("  VEREDICTO: ¿SE ECHAN LOS FINANCIEROS?")
print("="*70)
n_riesgo = len(EN_RIESGO)
n_total  = len(resumen)
print(f"\n  De {n_total} facultades:")
print(f"    · {n_riesgo} en ZONA DE RIESGO (score < 0.40) → se recomienda intervención")
print(f"    · {n_total - n_riesgo} con desempeño aceptable\n")
for _, r in resumen.iterrows():
    tend = r['tendencia_score']
    if r['score_medio'] < 0.35:
        veredicto = "ECHAR — desempeño critico, sin mejora"
    elif r['score_medio'] < 0.40 and tend < 0:
        veredicto = "ECHAR — en riesgo y empeorando"
    elif r['score_medio'] < 0.40 and tend >= 0:
        veredicto = "ADVERTENCIA — en riesgo pero mejorando, dar seguimiento"
    elif r['score_medio'] < 0.55:
        veredicto = "SEGUIMIENTO — desempeño moderado"
    else:
        veredicto = "OK — desempeño saludable"
    print(f"  {r['unidad']:<38} → {veredicto}")
print("="*70)
print(f"\nGraficas en: {carpeta}")
