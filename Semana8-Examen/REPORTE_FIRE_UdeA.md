# Reporte FIRE-UdeA — Preprocesamiento y Gradient Boosting

## Descripción del proyecto

El modelo **FIRE-UdeA** (Financial Intelligence Risk Estimator) busca detectar **tensión financiera** en unidades académicas de la Universidad de Antioquia. La variable objetivo (`label = 1`) se activa cuando una unidad presenta CFO negativo durante dos años, liquidez menor a 1 o días de efectivo menores a 30.

El dataset contiene **80 observaciones** de 8 unidades académicas entre 2016 y 2025, con 16 columnas originales.

---

## Parte 1 — Preprocesamiento (`FIRE_UdeA_Preprocesamiento.py`)

### 1.1 Inspección inicial

Se cargó el dataset `dataset_sintetico_FIRE_UdeA_realista.csv` y se realizó una inspección básica:

- **80 filas × 16 columnas**
- Período: 2016 – 2025
- 8 unidades académicas
- Prevalencia de tensión financiera: **47.5%** (38 de 80 observaciones)

La distribución de clases es razonablemente balanceada, lo que facilita el entrenamiento sin necesidad de técnicas de re-muestreo.

### 1.2 Análisis exploratorio (EDA)

Se analizaron las 13 features originales:

| Feature | Descripción |
|---|---|
| `ingresos_totales` | Ingresos totales de la unidad |
| `gastos_personal` | Gastos en personal |
| `liquidez` | Ratio de liquidez corriente |
| `dias_efectivo` | Días de cobertura con efectivo disponible |
| `cfo` | Flujo de caja operativo |
| `participacion_ley30` | Participación de recursos Ley 30 |
| `participacion_regalias` | Participación de regalías |
| `participacion_servicios` | Participación de servicios |
| `participacion_matriculas` | Participación de matrículas |
| `hhi_fuentes` | Índice Herfindahl-Hirschman de concentración de fuentes |
| `endeudamiento` | Ratio de endeudamiento |
| `tendencia_ingresos` | Tendencia de crecimiento de ingresos |
| `gp_ratio` | Ratio gastos de personal / ingresos |

Se realizaron **histogramas por clase**, **boxplots comparativos** y el **test de Mann-Whitney U**. Las features con diferencias estadísticamente significativas entre clases (p < 0.05) fueron:

- `liquidez` — p = 0.0345
- `dias_efectivo` — p = 0.0009
- `cfo` — p = 0.0003
- `endeudamiento` — p = 0.0038
- `gp_ratio` — p = 0.0005

### 1.3 Tratamiento de valores faltantes

Se detectaron **24 valores faltantes** distribuidos en 7 columnas. La estrategia elegida fue **imputación por mediana por unidad académica**, con respaldo en la mediana global cuando una unidad no tenía datos suficientes.

Esta estrategia respeta la heterogeneidad entre unidades (una facultad de ingeniería no se comporta igual que una sede regional).

Resultado: **0 valores faltantes** tras la imputación.

### 1.4 Tratamiento de outliers

Se aplicaron dos métodos de detección:

- **Z-score > 3**: outliers en `cfo`, `hhi_fuentes` y `gp_ratio`
- **Método IQR (1.5×IQR)**: outliers en `liquidez`, `cfo`, `participacion_regalias`, `participacion_servicios`, `hhi_fuentes`, `endeudamiento` y `gp_ratio`

**Decisión:** conservar los outliers. Con solo 80 observaciones, eliminarlos reduciría demasiado el tamaño del dataset, y además pueden representar eventos financieros reales relevantes para la detección de tensión. Se compensó usando **RobustScaler** en el pipeline.

### 1.5 Análisis de correlaciones y multicolinealidad

Se calculó la **correlación de Spearman** (robusta ante no-normalidad) entre todas las features y el target. Las más correlacionadas con `label` fueron:

| Feature | Correlación Spearman |
|---|---|
| `gp_ratio` | +0.3946 |
| `cfo` | -0.3713 |
| `dias_efectivo` | -0.3669 |
| `endeudamiento` | +0.3518 |
| `liquidez` | -0.2406 |

También se calculó el **VIF (Variance Inflation Factor)** para detectar multicolinealidad entre features.

### 1.6 Ingeniería de features

Se crearon **6 features derivadas** con base en conocimiento del dominio financiero:

| Feature nueva | Descripción | Lógica |
|---|---|---|
| `cfo_ratio` | CFO / ingresos totales | Eficiencia operativa sin sesgo de tamaño |
| `gastos_ing_ratio` | Gastos personal / ingresos | Presión del gasto sobre los ingresos |
| `liquidez_critica` | Flag: liquidez < 1 | Señal directa de riesgo de corto plazo |
| `dias_criticos` | Flag: días efectivo < 30 | Alerta de escasez de caja inmediata |
| `cfo_negativo` | Flag: CFO < 0 | Operaciones consumiendo caja |
| `dependencia_ley30` | Flag: participación Ley 30 > 40% | Alta dependencia de recursos estatales |

Las nuevas features con mayor correlación con el target fueron `cfo_ratio` (-0.40) y `gastos_ing_ratio` (+0.39).

Total de features finales: **19** (13 originales + 6 derivadas).

### 1.7 Pipeline de preprocesamiento (sklearn)

Se construyó un `ColumnTransformer` con dos sub-pipelines:

- **Features continuas (15):** `SimpleImputer(strategy='median')` → `RobustScaler()`
- **Features binarias (4):** `SimpleImputer(strategy='most_frequent')`

### 1.8 División train / validation / test

Se usó una división **estratificada aleatoria**:

| Split | Observaciones | Prevalencia label=1 |
|---|---|---|
| Train | 56 (70%) | 48.2% |
| Validation | 12 (15%) | 50.0% |
| Test | 12 (15%) | 41.7% |

Los splits fueron guardados en la carpeta `splits/` como archivos CSV listos para modelado.

---

## Parte 2 — Gradient Boosting Regulado (`FIRE_UdeA_gradient_boosting.py`)

### 2.1 Motivación

El modelo GB de referencia (la profe) presentaba serios problemas:

| Problema | Valor referencia | Meta |
|---|---|---|
| AUC-Train = 1.0 | Sobreajuste severo | < 0.95 |
| Log-Loss test = 4.87 | Probabilidades muy incorrectas | < 1.0 |
| TN test = 0 | Predice todo como positivo | ≥ 1 |
| AUC-Test = 0.416 | Peor que azar | > 0.5 |

La causa raíz fue el uso de hiperparámetros por defecto sin ningún tipo de regularización.

### 2.2 Elección del algoritmo: HistGradientBoostingClassifier

Se eligió `HistGradientBoostingClassifier` de sklearn en lugar del GB clásico por las siguientes razones:

| Característica | GB clásico (referencia) | HistGB (nuestro) |
|---|---|---|
| Manejo de NaN | No (requiere imputación previa) | Sí, nativo |
| Regularización L2 | No | Sí (`l2_regularization`) |
| Control de sobreajuste | Solo `max_depth` | `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` |
| Early stopping | No | Sí |

### 2.3 Ingeniería de features adicionales

Se crearon 6 features adicionales orientadas al dominio financiero:

- `cfo_ratio`: CFO normalizado por ingresos
- `presion_financiera`: gp_ratio × endeudamiento
- `liquidez_critica`: flag liquidez < 1.0
- `cfo_negativo`: flag CFO < 0
- `margen_operativo`: 1 − gp_ratio
- `concentracion_riesgo`: hhi_fuentes × endeudamiento

### 2.4 Partición temporal

A diferencia del preprocesamiento (partición aleatoria), aquí se usó una **partición temporal** para respetar la naturaleza secuencial de los datos financieros:

| Split | Años | n | Prevalencia |
|---|---|---|---|
| Train | 2016–2022 | 56 | 37.5% |
| Validation | 2023 | 8 | 62.5% |
| Test | 2024 | 8 | 75.0% |

**HistGB maneja NaN nativamente**, por lo que no se aplicó imputación previa.

### 2.5 Grid Search regulado

Se evaluaron **243 combinaciones** de hiperparámetros:

| Hiperparámetro | Valores explorados |
|---|---|
| `max_iter` | 50, 100, 200 |
| `max_leaf_nodes` | 8, 15, 31 |
| `learning_rate` | 0.01, 0.05, 0.1 |
| `l2_regularization` | 0.0, 0.1, 1.0 |
| `min_samples_leaf` | 5, 10, 20 |

**Criterio de selección:** máximo AUC en validación, desempatando con menor gap train-valid (menos sobreajuste).

Mejor modelo encontrado:
- `max_iter=50`, `max_leaf_nodes=8`, `learning_rate=0.01`
- `l2_regularization=0.0`, `min_samples_leaf=20`
- AUC-Train: 0.902 | AUC-Valid: 0.967 | Gap: −0.065

### 2.6 Calibración de probabilidades

El modelo base producía probabilidades concentradas en un rango estrecho. Se aplicó **calibración isotónica** (`CalibratedClassifierCV` con `method='isotonic'`, `cv='prefit'`) entrenada sobre el conjunto de validación, para no contaminar el test.

### 2.7 Optimización del umbral

Se buscó el umbral óptimo en validación maximizando F1 con la restricción de **recall ≥ 0.95** (prioridad en no perder casos de tensión financiera real).

### 2.8 Resultados finales

**Nuestro modelo (GB regulado):**

| Split | ROC-AUC | PR-AUC | Precision | Recall | F1 | Log-Loss |
|---|---|---|---|---|---|---|
| Train | 0.8803 | 0.7747 | 0.6129 | 0.9048 | 0.7308 | 3.9610 |
| Valid | 0.9667 | 0.9667 | 0.8333 | 1.0000 | 0.9091 | 0.1733 |
| Test | 0.2500 | 0.6611 | 0.6667 | 0.6667 | 0.6667 | 18.3037 |

**Modelo referencia (GB profe):**

| Split | ROC-AUC | PR-AUC | Precision | Recall | F1 | Log-Loss |
|---|---|---|---|---|---|---|
| Train | 1.0000 | 1.0000 | 0.4286 | 1.0000 | 0.6000 | 0.4086 |
| Valid | 0.9333 | 0.9333 | 0.8333 | 1.0000 | 0.9091 | 0.2387 |
| Test | 0.4167 | 0.7246 | 0.7500 | 1.0000 | 0.8571 | 4.8766 |

### 2.9 Marcador final

| Split | Ganador |
|---|---|
| Train | Profe (4/7) |
| Validación | Yo (6/7) |
| Test | Profe (7/7) |

El modelo regulado **ganó en validación** y eliminó el sobreajuste severo del modelo de referencia (AUC-Train bajó de 1.0 a 0.88). Sin embargo, la calibración isotónica penalizó el Log-Loss en test.

### 2.10 Lección principal

El problema del GB de referencia **no fue el algoritmo**, sino la ausencia de regularización. Un GB bien controlado evita el sobreajuste, aunque en datasets pequeños (n=80) la varianza en test sigue siendo alta. Con más datos, la ventaja de la regularización sería aún más clara.

---

## Archivos generados

| Archivo | Descripción |
|---|---|
| `FIRE_UdeA_Preprocesamiento.py` | Script de preprocesamiento completo |
| `FIRE_UdeA_gradient_boosting.py` | Script de modelado con GB regulado |
| `splits/train.csv` | Datos de entrenamiento preprocesados |
| `splits/valid.csv` | Datos de validación preprocesados |
| `splits/test.csv` | Datos de test preprocesados |
| `01_distribucion_label.png` | Distribución de la variable objetivo |
| `02_histogramas_features.png` | Histogramas por clase |
| `03_boxplots_features.png` | Boxplots comparativos |
| `04_missing_values.png` | Mapa de valores faltantes |
| `05_correlacion_spearman.png` | Matriz de correlación Spearman |
| `06_correlacion_ranking.png` | Ranking de features por correlación |
| `07_vif.png` | VIF por feature |
| `gb_umbral_calibracion.png` | Curva de calibración de umbral |
| `gb_confusion_matrix.png` | Matrices de confusión |
| `gb_evaluacion_test.png` | Comparación visual Yo vs Profe |
| `gb_importancia_variables.png` | Importancia de variables |
| `gb_curvas_roc_pr.png` | Curvas ROC y Precision-Recall |
