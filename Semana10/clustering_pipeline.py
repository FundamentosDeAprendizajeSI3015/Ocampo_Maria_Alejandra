"""
Bonus Track — Clustering Pipeline
  • Subtractive Clustering  (Chiu 1994)
  • Fuzzy C-Means           (Bezdek 1981)

Correr con:  py -3.11 clustering_pipeline.py
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from umap import UMAP

plt.rc('font', family='serif', size=11)
COLORES = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
           '#42d4f4','#f032e6','#bfef45','#fabebe','#469990']

# ══════════════════════════════════════════════════════════════════
#  CONFIGURACION
# ══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--csv',
        default=r'C:\Users\maria\OneDrive\Imágenes\Escritorio\Automatico\Ocampo_Maria_Alejandra\Semana10\dataset_sintetico_FIRE_UdeA_realista.csv')
    p.add_argument('--label-col',  default='label')
    p.add_argument('--n-clusters', type=int,   default=None,  help='K para FCM (None = usa Subtractive)')
    p.add_argument('--ra',         type=float, default=0.5)
    p.add_argument('--rb',         type=float, default=None)
    p.add_argument('--eps-upper',  type=float, default=0.5)
    p.add_argument('--eps-lower',  type=float, default=0.15)
    p.add_argument('--fuzz',       type=float, default=2.0,   help='Exponente m del FCM (m>1)')
    p.add_argument('--transpose',  action='store_true')
    return p.parse_args()

# ══════════════════════════════════════════════════════════════════
#  CARGA Y LIMPIEZA
# ══════════════════════════════════════════════════════════════════
def cargar(csv_path, label_col, transpose):
    df = pd.read_csv(csv_path)
    if transpose:
        df = df.T
    labels_ext = None
    if label_col and label_col != 'None' and label_col in df.columns:
        labels_ext = df[label_col].values
        df = df.drop(columns=[label_col])
    drop = ['anio', 'unidad', 'ingresos_totales', 'gastos_personal']
    df   = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')
    df   = df.select_dtypes(include=[np.number])
    for col in df.columns:
        df[col] = df[col].fillna(df[col].median())
    print(f"Muestras: {df.shape[0]}  Features: {df.shape[1]}")
    return df.values, df.columns.tolist(), labels_ext

# ══════════════════════════════════════════════════════════════════
#  SUBTRACTIVE CLUSTERING  — Chiu (1994)
# ══════════════════════════════════════════════════════════════════
def subtractive(X, ra=0.5, rb=None, eps_up=0.5, eps_lo=0.15):
    if rb is None: rb = 1.5 * ra
    alpha = 4.0 / ra**2
    beta  = 4.0 / rb**2
    D2    = ((X[:,None,:] - X[None,:,:])**2).sum(2)
    pot   = np.exp(-alpha * D2).sum(1)
    pot0  = pot.copy()
    centros, pot_vals, D1 = [], [], None

    while True:
        i = int(np.argmax(pot))
        P = pot[i]
        if D1 is None: D1 = P
        r = P / D1
        if r >= eps_up:
            centros.append(X[i].copy()); pot_vals.append(P)
        elif r <= eps_lo:
            break
        else:
            if centros:
                dmin = min(np.linalg.norm(X[i]-c) for c in centros)
                if dmin/ra + r >= 1.0:
                    centros.append(X[i].copy()); pot_vals.append(P)
                else:
                    pot[i] = 0.; continue
            else: break
        d2 = ((X - X[i])**2).sum(1)
        pot -= P * np.exp(-beta * d2)
        pot  = np.maximum(pot, 0.)
        pot[i] = 0.
        if len(centros) >= len(X): break

    centros = np.array(centros) if centros else X[[0]]
    labels  = np.argmin(((X[:,None,:]-centros[None,:,:])**2).sum(2), axis=1)
    return centros, labels, pot0, pot_vals

# ══════════════════════════════════════════════════════════════════
#  FUZZY C-MEANS  — Bezdek (1981)
# ══════════════════════════════════════════════════════════════════
def fcm(X, c, m=2.0, max_iter=300, tol=1e-6, init_centers=None):
    np.random.seed(42)
    n  = len(X)
    ex = 2.0 / (m - 1.0)
    U  = np.random.dirichlet(np.ones(c), size=n)

    if init_centers is not None and len(init_centers) == c:
        D2    = ((X[:,None,:]-init_centers[None,:,:])**2).sum(2)
        dists = np.sqrt(np.maximum(D2, 1e-12))
        ratio = dists[:,:,None] / dists[:,None,:]
        U     = 1.0 / (ratio**ex).sum(2)

    hist = []
    for it in range(max_iter):
        U0      = U.copy()
        Um      = U**m
        centers = (Um.T @ X) / Um.sum(0)[:,None]
        D2      = ((X[:,None,:]-centers[None,:,:])**2).sum(2)
        dists   = np.sqrt(np.maximum(D2, 1e-12))
        ratio   = dists[:,:,None] / dists[:,None,:]
        U       = 1.0 / (ratio**ex).sum(2)
        hist.append(float(((U**m)*D2).sum()))
        if np.linalg.norm(U-U0,'fro') < tol:
            print(f"  FCM convergio iter {it+1}"); break
    else:
        print(f"  FCM: {max_iter} iteraciones")

    return centers, U, np.argmax(U,1), hist

# ══════════════════════════════════════════════════════════════════
#  GRAFICAS
# ══════════════════════════════════════════════════════════════════
def save(fig, path):
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  OK  {os.path.basename(path)}")

# ─────────────────────────────────────────────────────────────────
# Fig 1  Potencial inicial de Subtractive
# ─────────────────────────────────────────────────────────────────
def g1_potencial(Z, pot0, carpeta):
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(Z[:,0], Z[:,1], c=pot0, cmap='YlOrRd', s=22, alpha=0.85)
    plt.colorbar(sc, ax=ax, label='Potencial P(xᵢ)')
    ax.set_title('Subtractive — Potencial inicial\n'
                 'Rojo = alta densidad = candidato a centro de cluster', pad=10)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    save(fig, os.path.join(carpeta, 'fig1_sub_potencial.png'))

# ─────────────────────────────────────────────────────────────────
# Fig 2  Clusters Subtractive + potencial decreciente
# ─────────────────────────────────────────────────────────────────
def g2_sub(Z, labels, cx, pot_vals, k, sil, carpeta):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for cid in range(k):
        m = labels == cid
        axes[0].scatter(Z[m,0], Z[m,1], color=COLORES[cid%len(COLORES)],
                        alpha=0.65, s=22, label=f'C{cid+1}  (n={m.sum()})')
    axes[0].scatter(cx[:,0], cx[:,1], c='black', s=220, marker='*',
                    zorder=10, label='Centros')
    axes[0].set_title(f'Subtractive Clustering\nK={k} encontrado automaticamente'
                      f'  |  Silhouette={sil:.3f}')
    axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')
    axes[0].legend(fontsize=9)

    bars = axes[1].bar(range(1, k+1), pot_vals,
                       color=[COLORES[i%len(COLORES)] for i in range(k)], alpha=0.85)
    for bar, v in zip(bars, pot_vals):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(pot_vals)*0.01,
                     f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    axes[1].set_title('Potencial de cada centro al ser elegido\n'
                      '(teóricamente decreciente — Chiu 1994)')
    axes[1].set_xlabel('Orden de selección'); axes[1].set_ylabel('P*k')
    axes[1].set_xticks(range(1, k+1))

    plt.suptitle('Subtractive Clustering (Chiu 1994)', fontsize=13, y=1.01)
    plt.tight_layout()
    save(fig, os.path.join(carpeta, 'fig2_sub_clusters.png'))

# ─────────────────────────────────────────────────────────────────
# Fig 3  Convergencia J de FCM
# ─────────────────────────────────────────────────────────────────
def g3_convergencia(hist, carpeta):
    fig, ax = plt.subplots(figsize=(8, 4))
    iters = range(1, len(hist)+1)
    ax.plot(iters, hist, color='steelblue', lw=2.2)
    ax.fill_between(iters, hist, alpha=0.15, color='steelblue')
    ax.scatter([len(hist)], [hist[-1]], color='steelblue', s=60, zorder=5)
    ax.set_title('Fuzzy C-Means — Convergencia de la función objetivo J\n'
                 'J decrece hasta estabilizarse (criterio de parada)')
    ax.set_xlabel('Iteración'); ax.set_ylabel('J = Σᵢ Σₖ uᵢₖᵐ · ‖xᵢ − vₖ‖²')
    ax.annotate(f'Valor final\nJ={hist[-1]:.2f}',
                xy=(len(hist), hist[-1]),
                xytext=(max(1, len(hist)-len(hist)//3), hist[0]*0.6),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')
    plt.tight_layout()
    save(fig, os.path.join(carpeta, 'fig3_fcm_convergencia.png'))

# ─────────────────────────────────────────────────────────────────
# Fig 4  FCM: asignacion dura + certeza (lo distintivo del FCM)
# ─────────────────────────────────────────────────────────────────
def g4_fcm_blanda(Z, U, labels, cx, k, sil, carpeta):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Izquierda: colores por cluster
    for cid in range(k):
        m = labels == cid
        axes[0].scatter(Z[m,0], Z[m,1], color=COLORES[cid%len(COLORES)],
                        alpha=0.65, s=22, label=f'C{cid+1}  (n={m.sum()})')
    axes[0].scatter(cx[:,0], cx[:,1], c='black', s=220, marker='*',
                    zorder=10, label='Centros')
    axes[0].set_title(f'FCM — Asignación dura (argmax U)\nSilhouette={sil:.3f}')
    axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')
    axes[0].legend(fontsize=9)

    # Derecha: certeza de asignación (la parte "fuzzy")
    certeza = U.max(axis=1)
    sc = axes[1].scatter(Z[:,0], Z[:,1], c=certeza,
                         cmap='RdYlGn', vmin=1/k, vmax=1, s=22, alpha=0.85)
    axes[1].scatter(cx[:,0], cx[:,1], c='black', s=220, marker='*', zorder=10)
    cb = plt.colorbar(sc, ax=axes[1])
    cb.set_label('max(uᵢₖ)  — certeza de pertenencia')
    axes[1].set_title('FCM — Grado de certeza por punto\n'
                      f'Verde = certero (uᵢₖ≈1)  |  Rojo = frontera difusa (uᵢₖ≈1/K={round(1/k,2)})')
    axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')

    plt.suptitle('Fuzzy C-Means (Bezdek 1981) — asignación BLANDA', fontsize=13, y=1.01)
    plt.tight_layout()
    save(fig, os.path.join(carpeta, 'fig4_fcm_clusters.png'))

# ─────────────────────────────────────────────────────────────────
# Fig 5  Heatmap de la matriz U
# ─────────────────────────────────────────────────────────────────
def g5_heatmap(U, k, carpeta, max_pts=120):
    n_show = min(max_pts, len(U))
    idx    = np.linspace(0, len(U)-1, n_show, dtype=int)
    U_show = U[idx]

    fig, ax = plt.subplots(figsize=(min(16, n_show//5+4), k+2))
    im = ax.imshow(U_show.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Grado de pertenencia uᵢₖ')
    ax.set_yticks(range(k))
    ax.set_yticklabels([f'Cluster {k+1}' for k in range(k)])
    ax.set_xlabel(f'Punto (muestra de {n_show}/{len(U)})')
    ax.set_title('FCM — Matriz de membresías U\n'
                 'Cada columna suma 1  |  Columna uniforme = punto en zona de frontera')
    plt.tight_layout()
    save(fig, os.path.join(carpeta, 'fig5_fcm_membresias.png'))

# ─────────────────────────────────────────────────────────────────
# Fig 6  Comparacion Subtractive vs FCM
# ─────────────────────────────────────────────────────────────────
def g6_comparacion(Z, ls, lf, k_s, k_f, cs, cf, sil_s, sil_f, carpeta):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for cid in range(k_s):
        m = ls == cid
        axes[0].scatter(Z[m,0], Z[m,1], color=COLORES[cid%len(COLORES)],
                        alpha=0.65, s=22, label=f'C{cid+1} (n={m.sum()})')
    axes[0].scatter(cs[:,0], cs[:,1], c='black', s=220, marker='*', zorder=10)
    axes[0].set_title(f'Subtractive  K={k_s} (automático)\nSilhouette={sil_s:.3f}')
    axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')
    axes[0].legend(fontsize=9)

    for cid in range(k_f):
        m = lf == cid
        axes[1].scatter(Z[m,0], Z[m,1], color=COLORES[cid%len(COLORES)],
                        alpha=0.65, s=22, label=f'C{cid+1} (n={m.sum()})')
    axes[1].scatter(cf[:,0], cf[:,1], c='black', s=220, marker='*', zorder=10)
    axes[1].set_title(f'Fuzzy C-Means  K={k_f}\nSilhouette={sil_f:.3f}')
    axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')
    axes[1].legend(fontsize=9)

    plt.suptitle('Comparación: Subtractive (K automático) vs Fuzzy C-Means (asignación blanda)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    save(fig, os.path.join(carpeta, 'fig6_comparacion.png'))

# ─────────────────────────────────────────────────────────────────
# Fig 7 & 8  Perfil medio por cluster
# ─────────────────────────────────────────────────────────────────
def g7_perfiles(X_std, labels, k, feat_names, titulo, fname, carpeta):
    fig, ax = plt.subplots(figsize=(max(10, len(feat_names)*0.8), 5))
    x = np.arange(len(feat_names))
    for cid in range(k):
        m = labels == cid
        if not m.any(): continue
        mu  = X_std[m].mean(0)
        std = X_std[m].std(0)
        ax.plot(x, mu, color=COLORES[cid%len(COLORES)], lw=2.2,
                marker='o', ms=5, label=f'Cluster {cid+1}  (n={m.sum()})')
        ax.fill_between(x, mu-std, mu+std, color=COLORES[cid%len(COLORES)], alpha=0.1)
    ax.axhline(0, color='gray', lw=0.8, ls='--', label='Media global')
    ax.set_xticks(x); ax.set_xticklabels(feat_names, rotation=40, ha='right')
    ax.set_ylabel('Valor estandarizado (z-score)')
    ax.set_title(f'{titulo} — Perfil medio por cluster (±1σ)\n'
                 'Muestra qué variables caracterizan a cada grupo')
    ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, os.path.join(carpeta, fname))

# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    if not os.path.isfile(args.csv):
        print(f"ERROR: no se encuentra '{args.csv}'"); sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    carpeta  = os.path.join(base_dir, 'graficas_bonus')
    os.makedirs(carpeta, exist_ok=True)

    # ── Cargar ────────────────────────────────────────────────────
    X_raw, feat_names, _ = cargar(args.csv, args.label_col, args.transpose)
    X_std = StandardScaler().fit_transform(X_raw)
    X_n01, mn, rng = (lambda X: ((X-X.min(0))/np.where(X.max(0)-X.min(0)==0,1,X.max(0)-X.min(0)),
                                  X.min(0), np.where(X.max(0)-X.min(0)==0,1,X.max(0)-X.min(0))))(X_raw)

    # ── UMAP ──────────────────────────────────────────────────────
    print("Calculando UMAP...")
    Z = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1).fit_transform(X_std)

    # centroide UMAP de cada cluster
    def cx_umap(labels, k):
        return np.array([Z[labels==i].mean(0) for i in range(k)])

    # ── Subtractive ───────────────────────────────────────────────
    rb = args.rb or 1.5*args.ra
    print(f"\nSubtractive  ra={args.ra}  rb={rb}  ε↑={args.eps_upper}  ε↓={args.eps_lower}")
    sc, ls, pot0, pv = subtractive(X_n01, args.ra, rb, args.eps_upper, args.eps_lower)
    k_s = len(sc)
    print(f"  K encontrado: {k_s}")
    sil_s = silhouette_score(X_std, ls) if k_s > 1 else 0.0
    print(f"  Silhouette: {sil_s:.4f}")

    # centros Subtractive en std para init FCM
    sc_std = StandardScaler().fit(X_raw).transform(sc * rng + mn)

    # ── FCM ───────────────────────────────────────────────────────
    k_f   = args.n_clusters or k_s
    initc = sc_std if k_f == k_s else None
    print(f"\nFCM  K={k_f}  m={args.fuzz}  init={'Subtractive' if initc is not None else 'aleatoria'}")
    vc, U, lf, hist = fcm(X_std, k_f, args.fuzz, init_centers=initc)
    sil_f = silhouette_score(X_std, lf) if k_f > 1 else 0.0
    print(f"  Silhouette: {sil_f:.4f}")

    # ── Graficas ──────────────────────────────────────────────────
    print(f"\nGraficas → {carpeta}")
    g1_potencial(Z, pot0, carpeta)
    g2_sub(Z, ls, cx_umap(ls, k_s), pv, k_s, sil_s, carpeta)
    g3_convergencia(hist, carpeta)
    g4_fcm_blanda(Z, U, lf, cx_umap(lf, k_f), k_f, sil_f, carpeta)
    g5_heatmap(U, k_f, carpeta)
    g6_comparacion(Z, ls, lf, k_s, k_f,
                   cx_umap(ls, k_s), cx_umap(lf, k_f),
                   sil_s, sil_f, carpeta)
    g7_perfiles(X_std, ls, k_s, feat_names, 'Subtractive', 'fig7_sub_perfiles.png', carpeta)
    g7_perfiles(X_std, lf, k_f, feat_names, 'Fuzzy C-Means', 'fig8_fcm_perfiles.png', carpeta)

    # ── Resumen ───────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"  Dataset   : {os.path.basename(args.csv)}")
    print(f"  Muestras  : {len(X_raw)}  |  Features: {X_raw.shape[1]}")
    print(f"  Subtractive  K={k_s}  Silhouette={sil_s:.3f}")
    for i in range(k_s): print(f"    C{i+1}: {int((ls==i).sum())} puntos")
    print(f"  FCM  K={k_f}  m={args.fuzz}  Silhouette={sil_f:.3f}")
    for i in range(k_f):
        n_h = int((lf==i).sum())
        u_m = float(U[lf==i, i].mean()) if n_h else 0
        print(f"    C{i+1}: {n_h} puntos  membresía media={u_m:.3f}")
    print("="*50)

if __name__ == '__main__':
    main()
