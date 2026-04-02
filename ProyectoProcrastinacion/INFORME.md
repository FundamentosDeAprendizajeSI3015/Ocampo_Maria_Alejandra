# Análisis de Procrastinación Académica en Estudiantes Universitarios

**Curso:** Fundamentos de Aprendizaje Automatico  
**Dataset:** 48 respuestas — Encuesta Likert (1–5)  
**Objetivo:** Identificar patrones, clasificar nivel de procrastinación (bajo/medio/alto) y determinar qué variables tienen mayor influencia.

---

## Estructura del Proyecto

```
proyecto/
├── preprocesamiento.py          # Limpieza + EDA
├── clustering.py                # Análisis NO supervisado
├── modelos_supervisados.py      # Clasificación supervisada
├── data/
│   ├── data_limpia.csv          # Dataset limpio (sin etiquetas)
│   ├── data_etiquetada.csv      # Con etiquetas originales
│   ├── data_corregida.csv       # Con etiquetas corregidas por clustering
│   └── metricas_modelos.csv     # Resultados de los modelos
├── graficos_preprocesamiento/   # 7 gráficos del EDA
├── graficos_clustering/         # 9 gráficos del clustering
└── graficos_modelos/            # 5 gráficos de modelos supervisados
```

---

## Variables del Dataset

| Nombre corto | Pregunta original | Tipo |
|---|---|---|
| `planificacion` | ¿Con qué frecuencia planificas tus tareas? | Protectora |
| `organizacion` | ¿Divides tareas grandes en partes más pequeñas? | Protectora |
| `autonomia` | ¿Cumples sin presión externa? | Protectora |
| `concentracion` | ¿Mantienes concentración al estudiar? | Protectora |
| `evitacion` | ¿Evitas tareas difíciles o poco interesantes? | Riesgo |
| `uso_celular` | ¿Revisas el celular mientras estudias? | Riesgo |
| `redes_sociales` | ¿Las redes sociales interfieren con tu estudio? | Riesgo |
| `procrastinacion` | ¿Dejas tareas para el último momento? | Riesgo |

**Variables protectoras:** mayor puntuación → menos procrastinación.  
**Variables de riesgo:** mayor puntuación → más procrastinación.

---

## Paso 1 — Preprocesamiento (`preprocesamiento.py`)

### ¿Qué se hizo?

1. **Carga del CSV original** (48 filas, 14 columnas)
2. **Eliminación de columnas administrativas:** ID, hora inicio, hora fin, correo, nombre, última modificación
3. **Renombrado de columnas:** las preguntas largas se reemplazaron con nombres cortos descriptivos
4. **Conversión Likert → numérico:**

| Respuesta | Valor |
|---|---|
| Nunca | 1 |
| Rara vez | 2 |
| A veces | 3 |
| Frecuentemente | 4 |
| Siempre | 5 |

5. **Imputación de nulos:** se usó la **mediana** por columna (más robusta que la media para datos ordinales)
6. **EDA — Análisis exploratorio:** distribuciones, boxplots, correlaciones, pairplot y radar chart

### Gráficos generados

| Archivo | Contenido |
|---|---|
| `01_distribuciones.png` | Frecuencia de cada valor Likert por variable |
| `02_boxplots.png` | Distribución y dispersión por variable |
| `03_correlaciones.png` | Mapa de calor de correlaciones Pearson |
| `04_histogramas_kde.png` | Histogramas con curva de densidad |
| `05_pairplot.png` | Relaciones entre variables clave |
| `06_radar_perfil.png` | Perfil promedio del estudiante |
| `07_estadisticas.png` | Tabla de estadísticas descriptivas |

### Hallazgos del EDA

- `redes_sociales` tiene la media más alta (3.79) — es el factor de riesgo más frecuente
- `organizacion` tiene la media más baja (2.81) — los estudiantes poco dividen sus tareas
- Correlación positiva entre `evitacion` y `procrastinacion` — quien evita también deja todo al final

---

## Paso 2 — Clustering NO Supervisado (`clustering.py`)

### ¿Por qué clustering primero?

El clustering se usa **antes** de los modelos supervisados para:
- Descubrir grupos naturales en los datos **sin imponer etiquetas**
- Validar si las etiquetas asignadas manualmente son coherentes con la estructura real
- Corregir posibles errores de etiquetado (la profesora indicó que hasta el 30% pueden estar mal)

---

### Algoritmos aplicados

#### K-Means (k=3)

El método del codo y el Silhouette Score coincidieron en `k=3`, lo que tiene sentido conceptualmente: **bajo, medio y alto** nivel de procrastinación.

- **Silhouette Score = 0.156** (datos Likert son naturalmente difusos, no hay clusters perfectamente separados)
- **Davies-Bouldin = 1.715** (menor es mejor)

> Con datos de escala Likert el Silhouette bajo es esperado: los valores son discretos (1–5) y no forman nubes claramente separadas en el espacio.

#### DBSCAN

DBSCAN es un algoritmo basado en densidad que encuentra clusters de forma arbitraria y detecta outliers (ruido).

- Resultado: no encontró clusters densos bien definidos
- Conclusión: **los datos no tienen regiones de alta densidad clara**, confirmando que la escala Likert produce una distribución uniforme sin separaciones abruptas

#### Fuzzy C-Means (FCM)

A diferencia de K-Means, FCM asigna a cada punto un **grado de membresía parcial** a cada cluster (valor entre 0 y 1). Esto es más realista: un estudiante puede ser "60% medio y 40% alto".

> Requiere instalar `scikit-fuzzy`: `pip install scikit-fuzzy`  
> Sin la librería, el script usa K-Means como sustituto automático.

#### Subtractive Clustering

Algoritmo de Chiu (1994) que **estima automáticamente el número de centros** basándose en la densidad local de cada punto. No requiere definir k a priori.

- Detectó **10 centros** con los parámetros por defecto (radio `ra=0.5`)
- Estos centros se usaron para inicializar K-Means, dando una partición alternativa

---

### Cómo se crearon las etiquetas

Se calculó un **score de procrastinación** para cada estudiante:

```
score = ( media(vars_riesgo) + media(6 - vars_protectoras) ) / 2
```

Las variables protectoras se invierten (6 − x) para que una puntuación alta siempre signifique **más procrastinación**.

| Rango del score | Etiqueta |
|---|---|
| ≤ 2.33 | bajo |
| 2.33 – 3.67 | medio |
| > 3.67 | alto |

Distribución resultante: **5 bajo / 32 medio / 11 alto**

---

### Corrección de etiquetas (hasta 30%)

Se comparó la etiqueta original con el **voto mayoritario** de K-Means, FCM y Subtractive Clustering. Los casos donde los tres métodos concordaban en una etiqueta diferente a la original fueron corregidos.

- Inconsistencias detectadas: **15 (31.2%)**
- Correcciones aplicadas: **14 (29.2%)** — dentro del límite del 30%

Distribución tras corrección: **16 bajo / 24 medio / 8 alto** (mejor balance de clases)

---

### Gráficos generados

| Archivo | Contenido |
|---|---|
| `01_kmeans_codo.png` | Método del codo + Silhouette para elegir k |
| `02_kmeans_clusters.png` | Clusters K-Means en UMAP y PCA |
| `03_dbscan_kdist.png` | k-distance graph para estimar eps |
| `04_dbscan_clusters.png` | Resultado de DBSCAN |
| `08_subtractive_clusters.png` | Clusters Subtractive + centros marcados |
| `09_comparacion_metricas.png` | Silhouette, DB y CH para cada algoritmo |
| `10_etiquetas_originales.png` | Etiquetas con inconsistencias marcadas (✗) |
| `11_correccion_etiquetas.png` | Antes vs después de la corrección |
| `12_confusion_etiquetas.png` | Qué etiquetas cambiaron y a cuál |

---

## Paso 3 — Modelos Supervisados (`modelos_supervisados.py`)

### División train/test

```
70% entrenamiento (33 muestras)  |  30% prueba (15 muestras)
División estratificada → proporciones iguales de clases en train y test
```

Se entrenó cada modelo **dos veces**: una con el dataset original y otra con el corregido.

---

### Modelo 1: Árbol de Decisión

El árbol de decisión aprende reglas tipo *"si redes_sociales > 3.5 Y organizacion ≤ 2.5 → alto"*. Es el modelo más **interpretable** y fácil de explicar.

**Parámetros usados:**
- `max_depth=4` — árbol de profundidad máxima 4 (evita sobreajuste)
- `min_samples_leaf=3` — cada hoja necesita al menos 3 muestras

| Dataset | Accuracy | F1-weighted |
|---|---|---|
| Original | 0.800 | 0.798 |
| Corregido | 0.733 | 0.740 |

**Variables más importantes según el árbol:**
1. `redes_sociales` — 57.7%
2. `organizacion` — 30.2%
3. `planificacion` — 8.7%

---

### Modelo 2: Regresión Logística

La regresión logística estima la **probabilidad** de pertenecer a cada clase. Requiere datos normalizados (StandardScaler). Es más potente que el árbol cuando las relaciones son aproximadamente lineales.

| Dataset | Accuracy | F1-weighted |
|---|---|---|
| Original | **0.933** | **0.924** |
| Corregido | 0.867 | 0.867 |

---

### Modelo 3: Regresión Lineal (complemento)

Se usa como análisis de apoyo, **no como clasificador principal**. Las etiquetas se tratan como ordinales (bajo=0, medio=1, alto=2) y se predice un valor continuo que luego se redondea.

Sirve para interpretar la **dirección e intensidad** del efecto de cada variable sobre la procrastinación: coeficiente positivo → aumenta la procrastinación; negativo → la reduce.

| Dataset | R² |
|---|---|
| Original | 0.238 |
| Corregido | −0.668 |

> El R² negativo en el dataset corregido indica que la regresión lineal no es el modelo adecuado aquí. Se incluye solo como herramienta interpretativa, no como clasificador.

---

### Gráficos generados

| Archivo | Contenido |
|---|---|
| `01_arbol_decision.png` | Árbol completo con nodos, ramas y clases |
| `02_importancia_variables.png` | Qué variables usa más el árbol |
| `03_confusion_arbol.png` | Matrices de confusión (Original vs Corregido) con FP/FN |
| `04_curvas_roc.png` | ROC del Árbol y Logística (ambos datasets) |
| `05_regresion_lineal.png` | Valores reales vs predichos |

---

## Comparación: Dataset Original vs Corregido

| Modelo | Dataset Original F1 | Dataset Corregido F1 | Diferencia |
|---|---|---|---|
| Árbol de Decisión | 0.798 | 0.740 | −0.058 |
| Regresión Logística | **0.924** | 0.867 | −0.057 |

### ¿Por qué el dataset original da mejores métricas en este caso?

Con solo **N=48 muestras**, las diferencias entre datasets son muy sensibles. Al corregir 14 etiquetas (29%), se redistribuyen las clases y el test set cambia de composición, lo que puede hacer que los modelos tengan más dificultad en el split particular utilizado.

Sin embargo, el dataset corregido tiene **mejor balance de clases** (16/24/8 vs 5/32/11), lo que lo hace más representativo del fenómeno real. En un dataset más grande, las correcciones guiadas por clustering habrían producido mejoras más claras y estables.

---

## Conclusiones Generales

1. **Las redes sociales y la organización son los factores más influyentes** en la procrastinación académica, según el árbol de decisión.

2. **La Regresión Logística fue el mejor modelo** con F1 = 0.924 en el dataset original — supera al árbol gracias a su capacidad de separar clases con fronteras lineales en el espacio de las 8 variables.

3. **DBSCAN confirmó que no hay clusters perfectamente densos**, lo cual es esperable con datos Likert: los estudiantes no caen en grupos completamente separados sino en un espectro continuo.

4. **El etiquetado manual basado en score promedio tiene limitaciones**: 31% de inconsistencias detectadas respecto al clustering. El clustering actúa como segunda opinión objetiva sobre los niveles de procrastinación.

5. **Con N=48, todos los resultados deben interpretarse con cautela.** Un dataset más grande daría métricas más estables y conclusiones más sólidas.

---

## Cómo Ejecutar el Proyecto

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar en orden
python preprocesamiento.py
python clustering.py
python modelos_supervisados.py
```

> Para activar Fuzzy C-Means: `pip install scikit-fuzzy`  
> Para activar UMAP: `pip install umap-learn` (ya incluido en requirements.txt)
