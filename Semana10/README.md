# Semana 10 — Verificación de Etiquetas, Clustering Avanzado y Ranking Financiero

## Descripción general

Esta semana se trabajaron tres tareas sobre el dataset sintético FIRE UdeA:

1. **Verificación de etiquetas manuales** usando DBSCAN como validador independiente
2. **Clustering avanzado** con Subtractive Clustering (Chiu 1994) y Fuzzy C-Means (Bezdek 1981)
3. **Ranking financiero por facultad** para identificar los peores desempeños y tomar decisiones

---

## Archivos

| Archivo | Descripción |
|---|---|
| `dataset_sintetico_FIRE_UdeA_realista.csv` | Dataset original con etiquetas manuales |
| `verificacion_etiquetas.py` | Verificación de etiquetas con DBSCAN por vecindad |
| `clustering_pipeline.py` | Pipeline Subtractive + Fuzzy C-Means con UMAP |
| `ranking_financiero.py` | Ranking y veredicto por facultad |
| `graficas_verificacion/` | Gráficas de verificación de etiquetas |
| `graficas_bonus/` | Gráficas del pipeline de clustering |
| `graficas_ranking/` | Gráficas del ranking financiero |

---

## Parte 1 — Verificación de etiquetas (`verificacion_etiquetas.py`)

### Problema
Las etiquetas `label` (0=inestable, 1=estable) fueron asignadas manualmente y pueden contener errores.

### Método
Se usó **DBSCAN por vecindad local**: para cada punto se observan sus vecinos dentro del radio `EPS`. Si la mayoría de sus vecinos tienen un label diferente al suyo → el punto es **sospechoso** (posible error de etiquetado).

> Este enfoque es más honesto que el mapeo por mayoría de clusters, que siempre daría 100%.

### Resultados

| Label | Consistentes | Sospechosos |
|---|---|---|
| **0 — Inestable** | 69.0% | 26.2% |
| **1 — Estable** | 55.3% | 34.2% |

### Interpretación

- El **Label 0** tiene un 69% de consistencia aceptable, aunque el 26.2% sospechoso indica que algunos puntos etiquetados como inestables tienen vecinos estables.
- El **Label 1** es el más preocupante: el **34.2% sospechoso** significa que aproximadamente **1 de cada 3 unidades etiquetadas como estables podría estar mal clasificada**. Estas unidades aparentan estabilidad financiera pero sus indicadores reales las ubican en zona de riesgo.
- Las etiquetas manuales **no son completamente confiables** y se recomienda revisión antes de tomar decisiones.

### Parámetros ajustables
```python
EPS         = 0.8   # radio de vecindad (subirlo = menos sospechosos)
MIN_SAMPLES = 5
DATASET_PATH = r'ruta\al\dataset.csv'
```

---

## Parte 2 — Clustering Avanzado (`clustering_pipeline.py`)

### Algoritmos implementados desde cero

#### Subtractive Clustering (Chiu, 1994)
- Cada punto es candidato a centro de cluster
- Se calcula el **potencial de densidad** de cada punto: `P(xi) = Σ exp(-α·‖xi−xj‖²)`
- El punto con mayor potencial se convierte en el primer centro
- Se **resta su influencia** sobre los demás puntos y se repite
- El proceso para cuando ningún candidato supera el umbral `ε_lower`
- **Ventaja:** determina K automáticamente, sin especificarlo

#### Fuzzy C-Means (Bezdek, 1981)
- Cada punto pertenece a **todos los clusters** con un grado de membresía `u_ik ∈ [0,1]`
- Las filas de la matriz U suman 1: `Σk u_ik = 1`
- Minimiza: `J = Σi Σk u_ik^m · ‖xi − vk‖²`
- Los centros se actualizan como promedios ponderados por `u_ik^m`
- **Ventaja frente a KMeans:** los puntos en zonas de frontera no se asignan forzosamente a un solo cluster

### Visualización con UMAP
Se usa **UMAP** (Uniform Manifold Approximation and Projection) para proyectar los datos a 2D. UMAP preserva mejor la estructura local y global que PCA, haciendo los clusters más distinguibles visualmente.

### Gráficas generadas (`graficas_bonus/`)

| Gráfica | Contenido |
|---|---|
| `fig1` | Potencial inicial — zonas rojas son candidatos a centro |
| `fig2` | Clusters Subtractive + potencial decreciente de cada centro |
| `fig3` | Convergencia de la función objetivo J del FCM |
| `fig4` | Clusters FCM coloreados por certeza (verde=seguro, rojo=frontera) |
| `fig5` | Heatmap de la matriz de membresías U |
| `fig6` | Comparación lado a lado Subtractive vs FCM con Silhouette |
| `fig7` | Perfil medio por variable — Subtractive |
| `fig8` | Perfil medio por variable — FCM |

### Cómo correr
```bash
py -3.11 clustering_pipeline.py
py -3.11 clustering_pipeline.py --n-clusters 4
py -3.11 clustering_pipeline.py --ra 0.4 --rb 0.6
py -3.11 clustering_pipeline.py --csv ruta\otro_dataset.csv
```

---

## Parte 3 — Ranking Financiero (`ranking_financiero.py`)

### Objetivo
Identificar qué facultades tienen el peor desempeño financiero y responder: **¿a quién se debe echar?**

### Método
Se calculó un **score financiero ponderado** por facultad (0 = peor, 1 = mejor) usando los indicadores:

| Indicador | Peso | Dirección |
|---|---|---|
| `liquidez` | 20% | Mayor = mejor |
| `cfo` | 15% | Mayor = mejor |
| `tendencia_ingresos` | 15% | Mayor = mejor |
| `endeudamiento` | 15% | Menor = mejor |
| `gp_ratio` | 15% | Menor = mejor |
| `dias_efectivo` | 10% | Mayor = mejor |
| `hhi_fuentes` | 10% | Menor = mejor |

### Resultados

| # | Facultad | Score | % Inestable | Veredicto |
|---|---|---|---|---|
| 1 | Sedes y Seccionales | 0.491 | 40% | SEGUIMIENTO |
| 2 | Facultad de Ingeniería | 0.501 | 30% | SEGUIMIENTO |
| 3 | Institutos | 0.540 | 50% | SEGUIMIENTO |
| 4 | Medicina | 0.541 | 50% | SEGUIMIENTO |
| 5 | Nivel Central | 0.543 | 60% | SEGUIMIENTO |
| 6 | Ciencias Económicas | 0.554 | 60% | OK |
| 7 | Educación | 0.559 | 70% | OK |
| 8 | Derecho | 0.571 | 60% | OK |

### Conclusión

> Ninguna facultad cayó en zona crítica de **ECHAR** (score < 0.35), pero **Sedes y Seccionales** y **Facultad de Ingeniería** son las más débiles financieramente y requieren intervención urgente. Si en el próximo período sus indicadores no mejoran, procedería la decisión de reemplazar al responsable financiero.

Las facultades con `label=1` sospechoso (etiquetadas como estables pero con indicadores dudosos) son las más preocupantes porque reportaron estar bien cuando los datos dicen lo contrario.

---

## Cómo ejecutar todo

```bash
# 1. Verificacion de etiquetas
python verificacion_etiquetas.py

# 2. Clustering avanzado (requiere Python 3.11)
py -3.11 clustering_pipeline.py

# 3. Ranking financiero
python ranking_financiero.py
```
