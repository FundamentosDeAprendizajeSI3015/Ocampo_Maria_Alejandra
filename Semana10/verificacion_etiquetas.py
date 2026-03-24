import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACION — cambia estos valores
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH  = r'C:\Users\maria\OneDrive\Imágenes\Escritorio\Automatico\Ocampo_Maria_Alejandra\Semana10\dataset_sintetico_FIRE_UdeA_realista.csv'
COLUMNA_LABEL = 'label'

EPS         = 0.8   # <── radio del vecindario DBSCAN (ajusta este valor)
MIN_SAMPLES = 5     # <── minimo de vecinos para formar un cluster

# ══════════════════════════════════════════════════════════════════════════════

plt.rc('font', family='serif', size=12)
base_dir = os.path.dirname(os.path.abspath(__file__))
carpeta  = os.path.join(base_dir, 'graficas_verificacion')
os.makedirs(carpeta, exist_ok=True)

# ── 1. Cargar dataset ──────────────────────────────────────────────────────────
df = pd.read_csv(DATASET_PATH)
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

labels_originales = df[COLUMNA_LABEL].values

# ── 2. Preparar features (sin label ni identificadores) ───────────────────────
cols_excluir  = [COLUMNA_LABEL, 'anio', 'unidad', 'ingresos_totales', 'gastos_personal']
cols_features = [c for c in df.columns if c not in cols_excluir]

X = df[cols_features].copy()
for col in X.columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.columns:
    media = X[col].mean()
    std   = X[col].std()
    if std > 0:
        X[col] = (X[col] - media) / std

data = X.values
print(f"Features usadas: {list(X.columns)}")

# ── 3. PCA 2D para graficar ───────────────────────────────────────────────────
def pca_2d(M):
    M_c = M - M.mean(axis=0)
    _, _, Vt = np.linalg.svd(M_c, full_matrices=False)
    return M_c @ Vt[:2].T

data_2d = pca_2d(data)

# ── 4. DBSCAN manual ──────────────────────────────────────────────────────────
def dbscan(M, eps, min_samples):
    n = len(M)
    labels = -np.ones(n, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def get_neighbors(i):
        return np.where(np.linalg.norm(M - M[i], axis=1) <= eps)[0]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nb = get_neighbors(i)
        if len(nb) < min_samples:
            labels[i] = -1
        else:
            labels[i] = cluster_id
            queue = list(nb)
            while queue:
                j = queue.pop()
                if not visited[j]:
                    visited[j] = True
                    nb2 = get_neighbors(j)
                    if len(nb2) >= min_samples:
                        queue.extend(nb2)
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1
    return labels

print(f"\nEjecutando DBSCAN con eps={EPS}, min_samples={MIN_SAMPLES} ...")
labels_dbscan = dbscan(data_2d, EPS, MIN_SAMPLES)
ids_clusters  = sorted(set(labels_dbscan) - {-1})
n_clusters    = len(ids_clusters)
n_ruido       = (labels_dbscan == -1).sum()
print(f"Clusters encontrados: {n_clusters}  |  Puntos ruido: {n_ruido}")

# ── 5. VERIFICACION por vecindad local ────────────────────────────────────────
# Para cada punto: mira sus vecinos dentro de EPS.
# Si la MAYORIA de sus vecinos tiene el mismo label → punto CONSISTENTE
# Si la mayoria tiene label distinto → punto SOSPECHOSO (posible error manual)

n = len(data_2d)
estado = np.full(n, 'sin_vecinos', dtype=object)   # por defecto

for i in range(n):
    dists = np.linalg.norm(data_2d - data_2d[i], axis=1)
    vecinos_idx = np.where((dists <= EPS) & (dists > 0))[0]

    if len(vecinos_idx) == 0:
        estado[i] = 'sin_vecinos'
        continue

    labels_vecinos = labels_originales[vecinos_idx]
    voto_mayoria   = int(pd.Series(labels_vecinos).mode()[0])

    if voto_mayoria == labels_originales[i]:
        estado[i] = 'consistente'
    else:
        estado[i] = 'sospechoso'

# ── 6. Calcular porcentajes reales ────────────────────────────────────────────
mask_0 = labels_originales == 0
mask_1 = labels_originales == 1

total_0 = mask_0.sum()
total_1 = mask_1.sum()

consist_0    = ((estado == 'consistente') & mask_0).sum()
consist_1    = ((estado == 'consistente') & mask_1).sum()
sospechoso_0 = ((estado == 'sospechoso')  & mask_0).sum()
sospechoso_1 = ((estado == 'sospechoso')  & mask_1).sum()
sinvec_0     = ((estado == 'sin_vecinos') & mask_0).sum()
sinvec_1     = ((estado == 'sin_vecinos') & mask_1).sum()

pct_correcto_0  = consist_0    / total_0 * 100
pct_correcto_1  = consist_1    / total_1 * 100
pct_sospechoso_0 = sospechoso_0 / total_0 * 100
pct_sospechoso_1 = sospechoso_1 / total_1 * 100

print("\n" + "="*55)
print("  VERIFICACION DE ETIQUETAS — vecindad local (eps={})".format(EPS))
print("="*55)
print(f"  Total label=0 : {total_0}  |  Total label=1 : {total_1}")
print()
print(f"  Label 0 — consistentes  : {consist_0:4d}  ({pct_correcto_0:.1f}%)")
print(f"  Label 0 — sospechosos   : {sospechoso_0:4d}  ({pct_sospechoso_0:.1f}%)")
print(f"  Label 0 — sin vecinos   : {sinvec_0:4d}")
print()
print(f"  Label 1 — consistentes  : {consist_1:4d}  ({pct_correcto_1:.1f}%)")
print(f"  Label 1 — sospechosos   : {sospechoso_1:4d}  ({pct_sospechoso_1:.1f}%)")
print(f"  Label 1 — sin vecinos   : {sinvec_1:4d}")
print()
print(f"  >>> % realmente 0 (consistente): {pct_correcto_0:.1f}%")
print(f"  >>> % realmente 1 (consistente): {pct_correcto_1:.1f}%")
print("="*55)

# ── 7. Graficas ───────────────────────────────────────────────────────────────
mask_ruido = labels_dbscan == -1

# Fig 1 — Etiquetas originales
fig, ax = plt.subplots()
for lv, nombre, color in [(0, 'Label 0 (inestable)', 'steelblue'),
                           (1, 'Label 1 (estable)',  'tomato')]:
    m = labels_originales == lv
    ax.scatter(data_2d[m, 0], data_2d[m, 1], label=nombre, color=color, alpha=0.6, s=20)
ax.set_title('Etiquetas originales (manuales)')
ax.set_xlabel('Componente 1'); ax.set_ylabel('Componente 2')
ax.legend(); fig.set_size_inches(5*1.6, 5); plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'fig1_labels_originales.png'), dpi=120); plt.close()
print("Guardada: fig1_labels_originales.png")

# Fig 2 — Clusters DBSCAN
cmap_db = plt.cm.get_cmap('tab10', max(n_clusters, 1))
fig, ax = plt.subplots()
ax.scatter(data_2d[mask_ruido, 0], data_2d[mask_ruido, 1],
           color='lightgray', alpha=0.4, s=15, label='Ruido')
for cid in ids_clusters:
    m = labels_dbscan == cid
    n0 = (labels_originales[m] == 0).sum()
    n1 = (labels_originales[m] == 1).sum()
    ax.scatter(data_2d[m, 0], data_2d[m, 1], color=cmap_db(cid),
               alpha=0.7, s=20, label=f'C{cid}  (0:{n0} / 1:{n1})')
ax.set_title(f'DBSCAN  eps={EPS}  |  clusters={n_clusters}  ruido={n_ruido}')
ax.set_xlabel('Componente 1'); ax.set_ylabel('Componente 2')
ax.legend(fontsize=8); fig.set_size_inches(5*1.6, 5); plt.tight_layout()
plt.savefig(os.path.join(carpeta, f'fig2_dbscan_eps{EPS}.png'), dpi=120); plt.close()
print(f"Guardada: fig2_dbscan_eps{EPS}.png")

# Fig 3 — Puntos sospechosos resaltados
fig, ax = plt.subplots()
m_c0 = (estado == 'consistente') & mask_0
m_c1 = (estado == 'consistente') & mask_1
m_s0 = (estado == 'sospechoso')  & mask_0
m_s1 = (estado == 'sospechoso')  & mask_1
m_sv = estado == 'sin_vecinos'

ax.scatter(data_2d[m_c0, 0], data_2d[m_c0, 1], color='steelblue', alpha=0.5, s=15, label=f'Label 0 OK ({consist_0})')
ax.scatter(data_2d[m_c1, 0], data_2d[m_c1, 1], color='tomato',    alpha=0.5, s=15, label=f'Label 1 OK ({consist_1})')
ax.scatter(data_2d[m_s0, 0], data_2d[m_s0, 1], color='darkorange', alpha=0.9, s=60,
           marker='X', label=f'Label 0 SOSPECHOSO ({sospechoso_0})')
ax.scatter(data_2d[m_s1, 0], data_2d[m_s1, 1], color='purple',    alpha=0.9, s=60,
           marker='X', label=f'Label 1 SOSPECHOSO ({sospechoso_1})')
ax.scatter(data_2d[m_sv, 0], data_2d[m_sv, 1], color='lightgray', alpha=0.4, s=10, label='Sin vecinos')
ax.set_title(f'Puntos sospechosos (eps={EPS})\n0→{pct_sospechoso_0:.1f}% duda  |  1→{pct_sospechoso_1:.1f}% duda')
ax.set_xlabel('Componente 1'); ax.set_ylabel('Componente 2')
ax.legend(fontsize=8); fig.set_size_inches(5*1.6, 5); plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'fig3_sospechosos.png'), dpi=120); plt.close()
print("Guardada: fig3_sospechosos.png")

# Fig 4 — Barras: consistente vs sospechoso por label
fig, ax = plt.subplots()
x = np.array([0, 1])
w = 0.35
bars_c = ax.bar(x - w/2, [pct_correcto_0,   pct_correcto_1],   w, label='Consistente', color=['steelblue','tomato'],   alpha=0.85)
bars_s = ax.bar(x + w/2, [pct_sospechoso_0, pct_sospechoso_1], w, label='Sospechoso',  color=['darkorange','purple'],  alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(['Label 0 (inestable)', 'Label 1 (estable)'])
ax.set_ylim(0, 115)
ax.set_ylabel('%')
ax.set_title('Verificacion de etiquetas manuales')
ax.legend()
for bar in list(bars_c) + list(bars_s):
    h = bar.get_height()
    if h > 0:
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
fig.set_size_inches(5*1.6, 5); plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'fig4_barras_verificacion.png'), dpi=120); plt.close()
print("Guardada: fig4_barras_verificacion.png")

print(f"\nListo! Graficas en:\n{carpeta}")
