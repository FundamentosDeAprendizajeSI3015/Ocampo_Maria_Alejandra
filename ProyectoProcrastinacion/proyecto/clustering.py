# =============================================================
# clustering.py
# Análisis NO supervisado: K-Means, DBSCAN, Fuzzy C-Means,
# Subtractive Clustering.
# Luego: etiquetado base, detección de inconsistencias,
# corrección de hasta el 30 % de etiquetas.
# =============================================================

import os
import warnings
warnings.filterwarnings("ignore")
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, confusion_matrix, ConfusionMatrixDisplay
)

# -- UMAP (pip install umap-learn). Fallback: t-SNE
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
    print("OK UMAP disponible.")
except ImportError:
    from sklearn.manifold import TSNE
    UMAP_AVAILABLE = False
    print("AVISO: UMAP no disponible — usando t-SNE. Instala con: pip install umap-learn")

# -- Fuzzy C-Means (pip install scikit-fuzzy)
try:
    import skfuzzy as fuzz
    FCM_AVAILABLE = True
    print("OK scikit-fuzzy disponible (Fuzzy C-Means).")
except ImportError:
    FCM_AVAILABLE = False
    print("AVISO: scikit-fuzzy no disponible. Instala con: pip install scikit-fuzzy")

# -------------------------------------------------------------
# RUTAS
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
GRAF_DIR = os.path.join(BASE_DIR, "graficos_clustering")
os.makedirs(GRAF_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. CARGA DE DATOS
# -------------------------------------------------------------
df = pd.read_csv(os.path.join(DATA_DIR, "data_limpia.csv"))
print(f"\nDataset cargado: {df.shape}\n")

FEATURES = df.columns.tolist()
X = df.values.astype(float)

# -------------------------------------------------------------
# 2. NORMALIZACIÓN
#    StandardScaler -> media 0, std 1  (K-Means, DBSCAN)
#    MinMaxScaler   -> [0, 1]          (FCM, Subtractive)
# -------------------------------------------------------------
scaler_std = StandardScaler()
scaler_mm  = MinMaxScaler()
X_std = scaler_std.fit_transform(X)
X_mm  = scaler_mm.fit_transform(X)

# -------------------------------------------------------------
# 3. REDUCCIÓN DE DIMENSIONALIDAD PARA VISUALIZACIÓN
#    UMAP captura estructura no lineal mejor que PCA.
#    PCA se usa como complemento siempre disponible.
# -------------------------------------------------------------
print("Calculando reducciones 2D...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_std)
print(f"  PCA varianza explicada: {pca.explained_variance_ratio_.sum():.2%}")

if UMAP_AVAILABLE:
    reducer    = UMAP(n_components=2, random_state=42,
                      n_neighbors=min(10, len(X)-1), min_dist=0.1)
    X_2d       = reducer.fit_transform(X_std)
    RED_NOMBRE = "UMAP"
else:
    perp = min(15, max(5, len(X) // 3))
    reducer    = TSNE(n_components=2, random_state=42, perplexity=perp)
    X_2d       = reducer.fit_transform(X_std)
    RED_NOMBRE = "t-SNE"

print(f"  {RED_NOMBRE} completado: {X_2d.shape}\n")


# -------------------------------------------------------------
# 4. K-MEANS
# -------------------------------------------------------------
print("=" * 55)
print("ALGORITMO 1: K-MEANS")
print("=" * 55)

# -- Método del codo + Silhouette para elegir k óptimo
inertias, sil_scores = [], []
K_RANGE = range(2, 9)

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_std)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_std, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_RANGE, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=3, color='red', linestyle='--', label='k=3 seleccionado')
axes[0].set_xlabel("Número de clusters (k)", fontsize=12)
axes[0].set_ylabel("Inercia (WCSS)", fontsize=12)
axes[0].set_title("Método del Codo — K-Means", fontsize=12, fontweight='bold')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(K_RANGE, sil_scores, 'gs-', linewidth=2, markersize=8)
axes[1].axvline(x=3, color='red', linestyle='--', label='k=3 seleccionado')
axes[1].set_xlabel("Número de clusters (k)", fontsize=12)
axes[1].set_ylabel("Silhouette Score", fontsize=12)
axes[1].set_title("Silhouette Score vs k", fontsize=12, fontweight='bold')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle("K-Means — Selección del k óptimo", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "01_kmeans_codo.png"), dpi=150, bbox_inches='tight')
plt.close()

# k = 3 corresponde a los 3 niveles de procrastinación (bajo/medio/alto)
K_OPTIMO = 3
km_final = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=20)
km_labels = km_final.fit_predict(X_std)

km_sil = silhouette_score(X_std, km_labels)
km_db  = davies_bouldin_score(X_std, km_labels)
km_ch  = calinski_harabasz_score(X_std, km_labels)
print(f"K-Means (k=3): Silhouette={km_sil:.3f}  DB={km_db:.3f}  CH={km_ch:.1f}")

# -- Visualizar K-Means
COLORES_3 = ['#e74c3c', '#3498db', '#2ecc71']

def scatter_clusters(ax, X_plot, labels, n_clusters, titulo, colors=None):
    """Dibuja scatter plot de clusters con colores distintos."""
    if colors is None:
        colors = sns.color_palette("tab10", n_clusters)
    unique = sorted(set(labels))
    for cl, col in zip(unique, colors):
        mask = labels == cl
        nombre = "Ruido" if cl == -1 else f"Cluster {cl}"
        marker = 'x' if cl == -1 else 'o'
        ax.scatter(X_plot[mask, 0], X_plot[mask, 1], c=[col],
                   label=f"{nombre} (n={mask.sum()})",
                   alpha=0.8, s=80, marker=marker,
                   edgecolors='k' if cl != -1 else 'none', linewidths=0.5)
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
scatter_clusters(axes[0], X_2d,  km_labels, K_OPTIMO, f"K-Means ({RED_NOMBRE})", COLORES_3)
scatter_clusters(axes[1], X_pca, km_labels, K_OPTIMO, "K-Means (PCA)", COLORES_3)
axes[0].set_xlabel(f"{RED_NOMBRE} 1"); axes[0].set_ylabel(f"{RED_NOMBRE} 2")
axes[1].set_xlabel("PC 1");             axes[1].set_ylabel("PC 2")
plt.suptitle(f"K-Means (k=3) | Silhouette={km_sil:.3f}", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "02_kmeans_clusters.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 02_kmeans_clusters.png")


# -------------------------------------------------------------
# 5. DBSCAN
# -------------------------------------------------------------
print("\n" + "=" * 55)
print("ALGORITMO 2: DBSCAN")
print("=" * 55)

# -- k-distance graph para estimar eps óptimo
nbrs = NearestNeighbors(n_neighbors=4).fit(X_std)
distances, _ = nbrs.kneighbors(X_std)
k_dist = np.sort(distances[:, -1])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_dist, 'b-', linewidth=2)
ax.axhline(y=1.5, color='red', linestyle='--', label='eps ≈ 1.5')
ax.set_xlabel("Puntos (ordenados por distancia)", fontsize=11)
ax.set_ylabel("Distancia al 4.° vecino más cercano", fontsize=11)
ax.set_title("k-distance graph — Estimación de eps para DBSCAN", fontsize=12, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "03_dbscan_kdist.png"), dpi=150, bbox_inches='tight')
plt.close()

# -- Aplicar DBSCAN con eps inicial y ajuste automático si hace falta
for eps_val in [1.5, 2.0, 2.5]:
    dbscan    = DBSCAN(eps=eps_val, min_samples=3)
    db_labels = dbscan.fit_predict(X_std)
    n_cl_db   = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_ruido   = (db_labels == -1).sum()
    print(f"  eps={eps_val}: clusters={n_cl_db}, outliers={n_ruido}")
    if n_cl_db >= 2 and n_ruido < len(X) * 0.4:
        print(f"  -> Usando eps={eps_val}")
        break

# -- Visualizar DBSCAN
colores_db = sns.color_palette("tab10", len(set(db_labels)))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
scatter_clusters(axes[0], X_2d,  db_labels, n_cl_db, f"DBSCAN ({RED_NOMBRE})", colores_db)
scatter_clusters(axes[1], X_pca, db_labels, n_cl_db, "DBSCAN (PCA)", colores_db)
axes[0].set_xlabel(f"{RED_NOMBRE} 1"); axes[0].set_ylabel(f"{RED_NOMBRE} 2")
axes[1].set_xlabel("PC 1");             axes[1].set_ylabel("PC 2")
plt.suptitle(f"DBSCAN | Clusters={n_cl_db}, Outliers={n_ruido}", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "04_dbscan_clusters.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 04_dbscan_clusters.png")


# -------------------------------------------------------------
# 6. FUZZY C-MEANS (skfuzzy)
#    A diferencia de K-Means, cada punto pertenece parcialmente
#    a todos los clusters (grado de membresía ∈ [0,1]).
#    FPC (Fuzzy Partition Coefficient) guía la elección de c.
# -------------------------------------------------------------
print("\n" + "=" * 55)
print("ALGORITMO 3: FUZZY C-MEANS")
print("=" * 55)

if FCM_AVAILABLE:
    # skfuzzy espera la transpuesta: (n_features × n_samples)
    X_fcm      = X_mm.T
    fpc_scores = []
    C_RANGE    = range(2, 7)
    best_fpc, best_c, best_u, best_cntr = -1, 3, None, None

    for c in C_RANGE:
        cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
            X_fcm, c=c, m=2.0, error=0.005, maxiter=1000, init=None
        )
        fpc_scores.append(fpc)
        if fpc > best_fpc:
            best_fpc, best_c, best_u, best_cntr = fpc, c, u, cntr

    # Gráfico FPC
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(C_RANGE), fpc_scores, 'mo-', linewidth=2, markersize=8)
    ax.axvline(x=best_c, color='red', linestyle='--', label=f'c={best_c} seleccionado')
    ax.set_xlabel("Número de clusters (c)", fontsize=12)
    ax.set_ylabel("FPC (mayor = mejor separación)", fontsize=12)
    ax.set_title("Fuzzy C-Means — Selección de c por FPC", fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, "05_fcm_fpc.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Etiqueta crisp: cluster con mayor grado de membresía
    fcm_labels = np.argmax(best_u, axis=0)
    print(f"FCM: c={best_c}, FPC={best_fpc:.4f}")

    # Scatter FCM
    pal_fcm = sns.color_palette("Set1", best_c)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    scatter_clusters(axes[0], X_2d,  fcm_labels, best_c, f"Fuzzy C-Means ({RED_NOMBRE})", pal_fcm)
    scatter_clusters(axes[1], X_pca, fcm_labels, best_c, "Fuzzy C-Means (PCA)", pal_fcm)
    axes[0].set_xlabel(f"{RED_NOMBRE} 1"); axes[0].set_ylabel(f"{RED_NOMBRE} 2")
    axes[1].set_xlabel("PC 1");             axes[1].set_ylabel("PC 2")
    plt.suptitle(f"Fuzzy C-Means (c={best_c}) | FPC={best_fpc:.3f}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, "06_fcm_clusters.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Guardado: 06_fcm_clusters.png")

    # Heatmap de membresías
    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(best_u, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_yticks(range(best_c))
    ax.set_yticklabels([f"Cluster {i}" for i in range(best_c)])
    ax.set_xlabel("Estudiante (índice)", fontsize=11)
    ax.set_title("Grados de membresía FCM — cada columna es un estudiante",
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label="Grado de membresía")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAF_DIR, "07_fcm_membresias.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("Guardado: 07_fcm_membresias.png")

else:
    print("FCM omitido (scikit-fuzzy no instalado). Se usará K-Means como sustituto.")
    fcm_labels = km_labels.copy()
    best_c     = K_OPTIMO


# -------------------------------------------------------------
# 7. SUBTRACTIVE CLUSTERING  (implementación propia)
#    Algoritmo de Chiu (1994): estima automáticamente centros
#    basándose en la densidad local de cada punto.
#    Los centros se usan para inicializar K-Means.
# -------------------------------------------------------------
print("\n" + "=" * 55)
print("ALGORITMO 4: SUBTRACTIVE CLUSTERING")
print("=" * 55)


def subtractive_clustering(X_norm, ra=0.5, rb_factor=1.5,
                            accept_ratio=0.5, reject_ratio=0.15,
                            max_centers=10):
    """
    Subtractive Clustering (Chiu, 1994).

    Parámetros
    ----------
    X_norm      : array normalizado [0,1]  (n_samples × n_features)
    ra          : radio de vecindad (influye en granularidad de clusters)
    rb_factor   : rb = ra * rb_factor  (zona de supresión, tipicamente 1.5)
    accept_ratio: umbral alto para aceptar un centro directamente (0.5)
    reject_ratio: umbral bajo para detener la búsqueda (0.15)
    max_centers : límite de centros para evitar bucles largos

    Retorna
    -------
    centers  : coordenadas de los centros encontrados
    idx_list : índices en X_norm de cada centro
    """
    rb = ra * rb_factor
    n  = len(X_norm)

    # Potencial de cada punto: Σ_j exp(‑||xi-xj||² / (ra/2)²)
    potential = np.zeros(n)
    for i in range(n):
        diffs   = X_norm - X_norm[i]
        dists_sq = np.sum(diffs ** 2, axis=1)
        potential[i] = np.sum(np.exp(-dists_sq / (ra / 2) ** 2))

    idx_list       = []
    P_max_inicial  = potential.max()

    for _ in range(max_centers):
        idx_max  = int(np.argmax(potential))
        P_actual = potential[idx_max]

        if P_actual < reject_ratio * P_max_inicial:
            break   # potencial demasiado bajo -> no más centros

        if P_actual > accept_ratio * P_max_inicial:
            idx_list.append(idx_max)          # aceptar directamente
        else:
            # Condición de montaña: balance entre distancia y potencial
            if len(idx_list) > 0:
                dist_min = min(
                    np.linalg.norm(X_norm[idx_max] - X_norm[c])
                    for c in idx_list
                )
                if (dist_min / ra) + (P_actual / P_max_inicial) >= 1.0:
                    idx_list.append(idx_max)
                else:
                    potential[idx_max] = 0.0
                    continue
            else:
                idx_list.append(idx_max)

        # Suprimir potencial en vecindad del nuevo centro
        diffs    = X_norm - X_norm[idx_max]
        dists_sq = np.sum(diffs ** 2, axis=1)
        potential -= P_actual * np.exp(-dists_sq / (rb / 2) ** 2)
        potential  = np.maximum(potential, 0.0)

    centers = X_norm[idx_list] if idx_list else X_norm[:K_OPTIMO]
    return centers, idx_list


# Aplicar Subtractive Clustering
sc_centers_norm, sc_idx = subtractive_clustering(X_mm, ra=0.5)
n_sc = len(sc_centers_norm)
print(f"Subtractive Clustering detectó {n_sc} centro(s)")

# Usar los centros SC para inicializar K-Means (en espacio estándar)
if n_sc >= 2:
    sc_centers_orig = scaler_mm.inverse_transform(sc_centers_norm)
    sc_centers_std  = scaler_std.transform(sc_centers_orig)
    k_sc = min(n_sc, len(X) - 1)
    km_sc = KMeans(n_clusters=k_sc, init=sc_centers_std[:k_sc],
                   n_init=1, max_iter=500, random_state=42)
    sc_labels = km_sc.fit_predict(X_std)
    print(f"  K-Means inicializado con centros SC (k={k_sc})")
else:
    print("  SC encontró < 2 centros; se usa k=3 con init='k-means++'")
    sc_labels = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=10).fit_predict(X_std)
    n_sc = K_OPTIMO

# -- Visualizar Subtractive Clustering
pal_sc = sns.color_palette("Dark2", len(set(sc_labels)))
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
scatter_clusters(axes[0], X_2d,  sc_labels, n_sc, f"Subtractive Clustering ({RED_NOMBRE})", pal_sc)
scatter_clusters(axes[1], X_pca, sc_labels, n_sc, "Subtractive Clustering (PCA)", pal_sc)

# Marcar centros SC en PCA
if n_sc >= 2:
    c_orig = scaler_mm.inverse_transform(sc_centers_norm)
    c_std  = scaler_std.transform(c_orig)
    c_pca  = pca.transform(c_std)
    axes[1].scatter(c_pca[:, 0], c_pca[:, 1], marker='*', s=350,
                    c='gold', edgecolors='black', linewidths=1,
                    zorder=6, label='Centros SC')
    axes[1].legend(fontsize=9)

axes[0].set_xlabel(f"{RED_NOMBRE} 1"); axes[0].set_ylabel(f"{RED_NOMBRE} 2")
axes[1].set_xlabel("PC 1");             axes[1].set_ylabel("PC 2")
plt.suptitle(f"Subtractive Clustering | n_centros={n_sc}", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "08_subtractive_clusters.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 08_subtractive_clusters.png")


# -------------------------------------------------------------
# 8. COMPARACIÓN DE MÉTRICAS DE TODOS LOS ALGORITMOS
# -------------------------------------------------------------
print("\n" + "=" * 55)
print("RESUMEN COMPARATIVO DE CLUSTERING")
print("=" * 55)

metricas = {}

def calc_metricas(nombre, X_scaled, labels):
    lbl = labels[labels != -1] if -1 in labels else labels
    Xs  = X_scaled[labels != -1] if -1 in labels else X_scaled
    if len(set(lbl)) < 2:
        return
    metricas[nombre] = {
        "n_clusters":         len(set(lbl)),
        "silhouette":         round(silhouette_score(Xs, lbl), 4),
        "davies_bouldin":     round(davies_bouldin_score(Xs, lbl), 4),
        "calinski_harabasz":  round(calinski_harabasz_score(Xs, lbl), 2),
    }

calc_metricas("K-Means",          X_std, km_labels)
calc_metricas("DBSCAN",           X_std, db_labels)
if FCM_AVAILABLE:
    calc_metricas("Fuzzy C-Means",X_std, fcm_labels)
calc_metricas("Subtractive+KM",   X_std, sc_labels)

df_met = pd.DataFrame(metricas).T
print(df_met.to_string())
df_met.to_csv(os.path.join(DATA_DIR, "metricas_clustering.csv"))

# Gráfico comparativo
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colores_bar = sns.color_palette("muted", len(df_met))

for ax, met, tit, mejor in [
    (axes[0], "silhouette",        "Silhouette\n(↑ mejor)",        True),
    (axes[1], "davies_bouldin",    "Davies-Bouldin\n(↓ mejor)",    False),
    (axes[2], "calinski_harabasz", "Calinski-Harabasz\n(↑ mejor)", True),
]:
    vals = df_met[met]
    bars = ax.bar(vals.index, vals.values, color=colores_bar, edgecolor='black')
    ax.set_title(tit, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=20, labelsize=9)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"{bar.get_height():.3f}", ha='center', fontsize=8)

plt.suptitle("Comparación de métricas de clustering", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "09_comparacion_metricas.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 09_comparacion_metricas.png")


# -------------------------------------------------------------
# 9. CREACIÓN DE ETIQUETAS ORIGINALES
#
#    Variables PROTECTORAS (alta puntuación -> menos procrastinación):
#      planificacion, organizacion, autonomia, concentracion
#    Variables de RIESGO (alta puntuación -> más procrastinación):
#      evitacion, uso_celular, redes_sociales, procrastinacion
#
#    Score = media(vars_riesgo) + media(6 - vars_protectoras)  / 2
#    Rango 1–5:  bajo ≤ 2.33  |  2.33 < medio ≤ 3.67  |  alto > 3.67
# -------------------------------------------------------------
print("\n" + "=" * 55)
print("ETIQUETADO BASE (bajo / medio / alto)")
print("=" * 55)

VARS_PROT  = ["planificacion", "organizacion", "autonomia", "concentracion"]
VARS_RIESG = ["evitacion", "uso_celular", "redes_sociales", "procrastinacion"]
UMBRAL_L, UMBRAL_H = 2.33, 3.67
LABEL_ORDER = ["bajo", "medio", "alto"]

score_riesgo = df[VARS_RIESG].mean(axis=1)
score_prot   = (6 - df[VARS_PROT]).mean(axis=1)
proc_score   = (score_riesgo + score_prot) / 2     # ∈ [1, 5]

def score_a_etiqueta(s):
    if s <= UMBRAL_L: return "bajo"
    if s <= UMBRAL_H: return "medio"
    return "alto"

df["procrastinacion_score"] = proc_score.round(3)
df["etiqueta"]              = proc_score.apply(score_a_etiqueta)

print("Distribución de etiquetas originales:")
print(df["etiqueta"].value_counts())


# -------------------------------------------------------------
# 10. MAPEAR CLUSTERS K-MEANS -> ETIQUETAS
#     El cluster con menor score promedio = "bajo", etc.
# -------------------------------------------------------------

def mapear_clusters(labels_array, score_series, n_labels=3):
    """Asigna bajo/medio/alto a cada cluster según su score promedio."""
    tmp = pd.DataFrame({'c': labels_array, 's': score_series.values})
    tmp = tmp[tmp['c'] != -1]   # ignorar ruido DBSCAN si existe
    orden = tmp.groupby('c')['s'].mean().sort_values().index.tolist()
    if len(orden) >= n_labels:
        mapa = {cl: LABEL_ORDER[i] for i, cl in enumerate(orden[:n_labels])}
    else:
        # Más clusters: clasificar por rangos de score
        scores_cl = tmp.groupby('c')['s'].mean()
        mapa = {cl: score_a_etiqueta(scores_cl[cl]) for cl in orden}
    return pd.Series(labels_array).map(mapa).fillna("medio").values

df["km_cluster"] = km_labels
km_score_by_cl   = df.groupby("km_cluster")["procrastinacion_score"].mean()
print(f"\nScore promedio K-Means por cluster:\n{km_score_by_cl.round(3)}")

df["km_etiqueta"]  = mapear_clusters(km_labels,  proc_score)
df["fcm_etiqueta"] = mapear_clusters(fcm_labels, proc_score)
df["sc_etiqueta"]  = mapear_clusters(sc_labels,  proc_score)

# Voto mayoritario entre los tres métodos
def voto(row):
    votos = [row["km_etiqueta"], row["fcm_etiqueta"], row["sc_etiqueta"]]
    return Counter(votos).most_common(1)[0][0]

df["voto_clustering"] = df.apply(voto, axis=1)


# -------------------------------------------------------------
# 11. DETECCIÓN DE INCONSISTENCIAS
# -------------------------------------------------------------
df["inconsistencia"] = df["etiqueta"] != df["km_etiqueta"]
n_incons   = df["inconsistencia"].sum()
pct_incons = n_incons / len(df) * 100
print(f"\nInconsistencias detectadas: {n_incons} ({pct_incons:.1f}%)")

COLORES_ETQ = {"bajo": "#2ecc71", "medio": "#f39c12", "alto": "#e74c3c"}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, X_plot, tit in [(axes[0], X_2d, RED_NOMBRE), (axes[1], X_pca, "PCA")]:
    for etq, col in COLORES_ETQ.items():
        mask = (df["etiqueta"] == etq).values
        ax.scatter(X_plot[mask, 0], X_plot[mask, 1], c=col,
                   label=f"{etq.capitalize()} ({mask.sum()})",
                   alpha=0.8, s=80, edgecolors='k', linewidths=0.5)
    incons_mask = df["inconsistencia"].values
    ax.scatter(X_plot[incons_mask, 0], X_plot[incons_mask, 1],
               marker='X', s=220, c='black', zorder=5,
               label=f"Inconsistente ({incons_mask.sum()})")
    ax.set_title(f"Etiquetas originales ({tit})\n✗ = posible error de etiqueta",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlabel(f"{tit} 1" if tit != "PCA" else "PC 1")
    ax.set_ylabel(f"{tit} 2" if tit != "PCA" else "PC 2")

plt.suptitle("Etiquetas originales con inconsistencias detectadas por K-Means",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "10_etiquetas_originales.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 10_etiquetas_originales.png")


# -------------------------------------------------------------
# 12. CORRECCIÓN DE ETIQUETAS (máximo 30 %)
#     Se corrigen primero los casos donde la discrepancia
#     entre score y etiqueta es mayor (más probable que sean errores).
# -------------------------------------------------------------
MAX_CAMBIOS = int(len(df) * 0.30)
print(f"\nCorrección de etiquetas: máx. {MAX_CAMBIOS} cambios ({MAX_CAMBIOS/len(df)*100:.0f}%)")

df["necesita_correccion"] = df["etiqueta"] != df["voto_clustering"]

# Priorizar por magnitud de discrepancia del score respecto al umbral de clase
CENTRO_CLASE = {"bajo": 1.5, "medio": 3.0, "alto": 4.5}
df["discrepancia"] = df.apply(
    lambda r: abs(r["procrastinacion_score"] - CENTRO_CLASE.get(r["etiqueta"], 3.0)),
    axis=1
)

df["etiqueta_corregida"] = df["etiqueta"].copy()
n_corregidos = 0
candidatos   = df[df["necesita_correccion"]].sort_values("discrepancia", ascending=False)

for idx in candidatos.index:
    if n_corregidos >= MAX_CAMBIOS:
        break
    df.at[idx, "etiqueta_corregida"] = df.at[idx, "voto_clustering"]
    n_corregidos += 1

pct_corr = n_corregidos / len(df) * 100
print(f"Correcciones aplicadas: {n_corregidos} ({pct_corr:.1f}%)")
print("\nDistribución original -> corregida:")
comp = pd.concat([
    df["etiqueta"].value_counts().rename("original"),
    df["etiqueta_corregida"].value_counts().rename("corregida")
], axis=1).fillna(0).astype(int)
print(comp)


# -------------------------------------------------------------
# 13. VISUALIZACIÓN: ANTES VS DESPUÉS DE CORRECCIÓN + UMAP
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
cambios_mask = (df["etiqueta"] != df["etiqueta_corregida"]).values

for ax, col_etq, titulo in [
    (axes[0], "etiqueta",          "Etiquetas ORIGINALES"),
    (axes[1], "etiqueta_corregida","Etiquetas CORREGIDAS"),
]:
    for etq, col in COLORES_ETQ.items():
        mask = (df[col_etq] == etq).values
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=col,
                   label=f"{etq.capitalize()} ({mask.sum()})",
                   alpha=0.8, s=100, edgecolors='k', linewidths=0.5)
    if col_etq == "etiqueta" and cambios_mask.sum() > 0:
        ax.scatter(X_2d[cambios_mask, 0], X_2d[cambios_mask, 1],
                   marker='D', s=220, facecolors='none', edgecolors='purple',
                   linewidths=2, zorder=5, label=f"Corregido ({cambios_mask.sum()})")
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlabel(f"{RED_NOMBRE} 1"); ax.set_ylabel(f"{RED_NOMBRE} 2")

plt.suptitle(
    f"Corrección de etiquetas | {n_corregidos} cambios ({pct_corr:.0f}%)\n"
    "Justificación: voto mayoritario K-Means + FCM + Subtractive",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "11_correccion_etiquetas.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 11_correccion_etiquetas.png")

# -------------------------------------------------------------
# 14. MATRIZ DE CONFUSIÓN: original vs corregida
#     Muestra qué etiquetas se cambiaron y a cuál (falsos +/-)
# -------------------------------------------------------------
cm = confusion_matrix(df["etiqueta"], df["etiqueta_corregida"], labels=LABEL_ORDER)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_ORDER)
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title(
    "Etiqueta ORIGINAL (eje Y) vs CORREGIDA (eje X)\n"
    "Diagonal = sin cambio | Fuera diagonal = etiqueta corregida",
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(GRAF_DIR, "12_confusion_etiquetas.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Guardado: 12_confusion_etiquetas.png")

# -------------------------------------------------------------
# 15. GUARDAR DATASETS
# -------------------------------------------------------------
# data_etiquetada.csv: etiquetas originales (base en score promedio)
cols_out   = FEATURES + ["procrastinacion_score", "etiqueta"]
df[cols_out].to_csv(os.path.join(DATA_DIR, "data_etiquetada.csv"), index=False)

# data_corregida.csv: etiquetas ajustadas por clustering (hasta 30%)
df_corr = df[FEATURES + ["procrastinacion_score", "etiqueta_corregida"]].copy()
df_corr = df_corr.rename(columns={"etiqueta_corregida": "etiqueta"})
df_corr.to_csv(os.path.join(DATA_DIR, "data_corregida.csv"), index=False)

print(f"\nOK data_etiquetada.csv guardado ({len(df)} registros)")
print(f"OK data_corregida.csv  guardado ({len(df_corr)} registros)")
print("-" * 55)
print("clustering.py completado exitosamente.")
