import numpy as np
import pandas as pd
from pathlib import Path

import plotly.express as px
import plotly.io as pio

from sklearn.preprocessing import StandardScaler

# UMAP requiere: pip install umap-learn
import umap


# Nombre fijo del archivo CSV (debe estar en la misma carpeta)
CSV_NAME = "fintech_top_sintetico_2025.csv"
OUT_HTML = "reporte_fintech.html"

# Columna para colorear (si existe). Puedes cambiarla a "Region" o "Country"
COLOR_COL = "Segment"


def main():
    # Leo el CSV desde la carpeta actual
    csv_path = Path(CSV_NAME)
    if not csv_path.exists():
        raise FileNotFoundError(f"No encontré {CSV_NAME} en esta carpeta: {Path().resolve()}")

    df = pd.read_csv(csv_path)

    # Selecciono solo columnas numéricas y me quedo con 4 para que sea rápido
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        raise ValueError("Necesito al menos 2 columnas numéricas para graficar.")

    features = num_cols[:4]

    # Si la columna de color no existe, no coloreo
    color = COLOR_COL if COLOR_COL in df.columns else None

    # 1) Pairplot simple (scatter matrix)
    fig_pair = px.scatter_matrix(
        df,
        dimensions=features,
        color=color,
        title="Fintech: Pairplot (Scatter Matrix)"
    )
    fig_pair.update_traces(diagonal_visible=False)
    fig_pair.update_layout(height=850)

    # 2) UMAP 2D
    X = df[features].copy()

    # Si hay nulos, los relleno con mediana
    for c in features:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # Escalo antes de UMAP para que no se sesgue por magnitudes
    X_scaled = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(n_components=2, random_state=42)
    emb = reducer.fit_transform(X_scaled)

    umap_df = pd.DataFrame(emb, columns=["UMAP_1", "UMAP_2"])
    if color is not None:
        umap_df[color] = df[color].astype(str)

    fig_umap = px.scatter(
        umap_df,
        x="UMAP_1",
        y="UMAP_2",
        color=color if color in umap_df.columns else None,
        title="Fintech: UMAP 2D"
    )
    fig_umap.update_layout(height=600)

    # 3) Genero un HTML con ambos gráficos
    pair_html = pio.to_html(fig_pair, include_plotlyjs="cdn", full_html=False)
    umap_html = pio.to_html(fig_umap, include_plotlyjs=False, full_html=False)

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Reporte Fintech</title>
</head>
<body style="font-family: Arial; margin: 20px;">
  <h2>Reporte Fintech (CSV)</h2>
  <p>Archivo: {CSV_NAME}</p>
  {pair_html}
  <hr/>
  {umap_html}
</body>
</html>
"""

    Path(OUT_HTML).write_text(html, encoding="utf-8")
    print(f"✅ Listo. Se generó: {Path(OUT_HTML).resolve()}")


if __name__ == "__main__":
    main()
