# Semana 9 — Clustering sobre Dataset FIRE UdeA

## Descripción

Se aplicaron técnicas de agrupamiento no supervisado (clustering) sobre un dataset sintético de indicadores financieros de facultades de la Universidad de Antioquia (FIRE UdeA), siguiendo el mismo flujo del notebook `ejAgrupamiento_kmeans_dbscan.ipynb`.

---

## Archivos

| Archivo | Descripción |
|---|---|
| `dataset_sintetico_FIRE_UdeA_realista.csv` | Dataset original |
| `limpieza_dataset.py` | Script de limpieza y preprocesamiento |
| `dataset_limpio_para_clustering.csv` | Dataset limpio listo para clustering |
| `clustering_FIRE.py` | Script principal de clustering |
| `graficas_clustering/` | Carpeta con las gráficas generadas |

---

## Paso 1 — Limpieza del dataset (`limpieza_dataset.py`)

El dataset original tenía los siguientes problemas:

- **Valores faltantes (NaN)** en columnas como `endeudamiento`, `liquidez`, `cfo`, `participacion_ley30`
- **Columnas no útiles para clustering**: `anio`, `unidad` (identificadores), `ingresos_totales`, `gastos_personal` (valores absolutos en escala ~10¹¹), `label` (etiqueta supervisada)

### Qué se hizo:

1. Se seleccionaron solo los **indicadores financieros relativos** (ratios y proporciones):
   - `liquidez`, `dias_efectivo`, `cfo`
   - `participacion_ley30`, `participacion_regalias`, `participacion_servicios`, `participacion_matriculas`
   - `hhi_fuentes`, `endeudamiento`, `tendencia_ingresos`, `gp_ratio`

2. Se imputaron los NaN con la **mediana** de cada columna (robusta a outliers)

3. Se escalaron todas las columnas con **StandardScaler** (media=0, desviación=1)

4. Se exportó el resultado a `dataset_limpio_para_clustering.csv`

---

## Paso 2 — Clustering (`clustering_FIRE.py`)

Implementado desde cero usando solo `numpy`, `pandas` y `matplotlib` por incompatibilidad de `sklearn` y `umap` con Python 3.13.

### Técnicas aplicadas:

#### PCA (visualización)
- Reducción a 2 dimensiones mediante descomposición SVD
- Permite graficar los clusters en un plano 2D

#### KMeans
- Se probó con **K=2** como punto de partida
- Se aplicó el **método del codo** (K=1 hasta K=10) para encontrar el K óptimo
- Se volvió a entrenar con el **K óptimo = 3**
- Se reporta la **inercia** en cada caso

#### DBSCAN
- Parámetros: `eps=0.8`, `min_samples=5`
- Identifica clusters de forma automática y marca puntos como **ruido** (label = -1)

---

## Gráficas generadas (`graficas_clustering/`)

| Archivo | Contenido |
|---|---|
| `fig1_datos_originales.png` | Dataset proyectado en 2D con PCA, sin etiquetas |
| `fig2_kmeans_k2.png` | Clusters KMeans con K=2 |
| `fig3_metodo_codo.png` | Curva del codo para elegir K óptimo |
| `fig4_kmeans_k3.png` | Clusters KMeans con K óptimo |
| `fig5_dbscan.png` | Clusters DBSCAN con puntos de ruido |

---

## Cómo ejecutar

```bash
# 1. Limpiar el dataset
python limpieza_dataset.py

# 2. Correr el clustering (las graficas quedan en graficas_clustering/)
python clustering_FIRE.py
```
