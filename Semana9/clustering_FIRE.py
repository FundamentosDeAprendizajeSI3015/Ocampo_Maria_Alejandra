import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random_state = 42
np.random.seed(random_state)
plt.rc('font', family='serif', size=12)

# ── Cargar dataset ─────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(base_dir, 'dataset_limpio_para_clustering.csv')).values
print(f"Dataset cargado: {data.shape[0]} filas, {data.shape[1]} columnas")

# ── Crear carpeta de graficas ──────────────────────────────────────────────────
carpeta = os.path.join(base_dir, 'graficas_clustering')
os.makedirs(carpeta, exist_ok=True)

# ── PCA manual (para visualizar en 2D) ────────────────────────────────────────
def pca_2d(X):
    X_c = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    return X_c @ Vt[:2].T

data_2d = pca_2d(data)
print("PCA completado")

# ── KMeans manual ─────────────────────────────────────────────────────────────
def kmeans(X, k, max_iter=300):
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            if (labels == j).any():
                centroids[j] = X[labels == j].mean(axis=0)
    inertia = sum(np.linalg.norm(X[labels == j] - centroids[j])**2 for j in range(k))
    return labels, inertia

# ── DBSCAN manual ─────────────────────────────────────────────────────────────
def dbscan(X, eps=0.5, min_samples=5):
    n = len(X)
    labels = -np.ones(n, dtype=int)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)

    def neighbors(i):
        dists = np.linalg.norm(X - X[i], axis=1)
        return np.where(dists <= eps)[0]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nb = neighbors(i)
        if len(nb) < min_samples:
            labels[i] = -1  # ruido
        else:
            labels[i] = cluster_id
            queue = list(nb)
            while queue:
                j = queue.pop()
                if not visited[j]:
                    visited[j] = True
                    nb2 = neighbors(j)
                    if len(nb2) >= min_samples:
                        queue.extend(nb2)
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1
    return labels

# ── Metodo del codo ────────────────────────────────────────────────────────────
print("Calculando metodo del codo...")
inert = []
k_range = list(range(1, 11))
for k in k_range:
    _, iner = kmeans(data_2d, k)
    inert.append(iner)
    print(f"  K={k} inercia={iner:.1f}")

# ── KMeans K=2 ────────────────────────────────────────────────────────────────
labels_k2, iner_k2 = kmeans(data_2d, 2)
print(f"KMeans K=2 — inercia: {iner_k2:.2f}")

# ── KMeans K optimo ───────────────────────────────────────────────────────────
K_OPTIMO = 3   # <── cambia segun el codo
labels_kopt, iner_kopt = kmeans(data_2d, K_OPTIMO)
print(f"KMeans K={K_OPTIMO} — inercia: {iner_kopt:.2f}")

# ── DBSCAN ────────────────────────────────────────────────────────────────────
print("Calculando DBSCAN...")
labels_db = dbscan(data_2d, eps=0.8, min_samples=5)
n_clusters_db = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_ruido = (labels_db == -1).sum()
print(f"DBSCAN — clusters: {n_clusters_db}  |  ruido: {n_ruido}")

# ── Guardar graficas ───────────────────────────────────────────────────────────

# Fig 1 — datos sin etiqueta
fig, ax = plt.subplots()
ax.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6)
ax.set_title('Dataset FIRE UdeA (PCA 2D)')
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
fig.set_size_inches(5*1.6, 5)
plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'fig1_datos_originales.png'), dpi=120)
plt.close()
print("Guardada: fig1_datos_originales.png")

# Fig 2 — KMeans K=2
fig, ax = plt.subplots()
ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_k2, cmap='tab10', alpha=0.7)
ax.set_title(f'KMeans K=2  |  inercia={iner_k2:.0f}')
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
fig.set_size_inches(5*1.6, 5)
plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'fig2_kmeans_k2.png'), dpi=120)
plt.close()
print("Guardada: fig2_kmeans_k2.png")

# Fig 3 — Metodo del codo
fig, ax = plt.subplots()
ax.plot(k_range, inert, marker='o')
ax.set_title('Metodo del Codo')
ax.set_xlabel('K')
ax.set_ylabel('Inercia')
ax.set_xticks(k_range)
fig.set_size_inches(5*1.6, 5)
plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'fig3_metodo_codo.png'), dpi=120)
plt.close()
print("Guardada: fig3_metodo_codo.png")

# Fig 4 — KMeans K optimo
fig, ax = plt.subplots()
ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_kopt, cmap='tab10', alpha=0.7)
ax.set_title(f'KMeans K={K_OPTIMO}  |  inercia={iner_kopt:.0f}')
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
fig.set_size_inches(5*1.6, 5)
plt.tight_layout()
plt.savefig(os.path.join(carpeta, f'fig4_kmeans_k{K_OPTIMO}.png'), dpi=120)
plt.close()
print(f"Guardada: fig4_kmeans_k{K_OPTIMO}.png")

# Fig 5 — DBSCAN
fig, ax = plt.subplots()
ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels_db, cmap='tab10', alpha=0.7)
ax.set_title(f'DBSCAN  |  clusters={n_clusters_db}  |  ruido={n_ruido}')
ax.set_xlabel('Componente 1')
ax.set_ylabel('Componente 2')
fig.set_size_inches(5*1.6, 5)
plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'fig5_dbscan.png'), dpi=120)
plt.close()
print("Guardada: fig5_dbscan.png")

print(f"\nListo! Graficas en:\n{carpeta}")
