
Copy

# Fundamentos de Aprendizaje Automático — SI3015
 
**Estudiante:** Maria Alejandra Ocampo Giraldo  
**Curso:** SI3015 · Fundamentos de Aprendizaje Automático
 
---
 
Repositorio con las actividades del curso organizadas por semanas y proyectos prácticos, con énfasis en análisis exploratorio, preprocesamiento de datos y construcción de pipelines de Machine Learning.
 
---
 
## 📁 Contenido por semana
 
### 🔹 Semana 2 — Introducción al flujo de trabajo en ML
- Análisis inicial de un dataset de chocolates
- Exploración de datos y estudio de las etapas del ciclo de vida del dato
---
 
### 🔹 Semana 3 — Preprocesamiento con datasets reales
- Talleres prácticos con datasets **Fintech**, **Iris** y **Titanic**
- Limpieza, exploración y análisis del proceso de preparación de datos
---
 
### 🔹 Semana 4 — EDA y transformación sobre Titanic
- Cálculo de medidas estadísticas: tendencia central, dispersión y posición
- Detección de outliers mediante IQR
- Visualizaciones: histogramas y gráficos de dispersión
- Codificación de variables categóricas, análisis de correlación y escalado de features
- Generación de dataset procesado listo para modelado
---
 
### 🔹 Semana 5 — Regresión lineal y logística
- Ingeniería de características sobre Titanic: imputación de `Age`, extracción de `Title`, creación de `FamilySize` e `IsAlone`
- Modelos aplicados: **Ridge**, **Lasso** y **Regresión Logística**
- Búsqueda de hiperparámetros con `RandomizedSearchCV` y validación cruzada 5-fold
- Regresión para predecir `Fare`: R² = 0.46 · MAE = 18.50
- Clasificación para predecir `Survived`: Accuracy = 79.3% · F1 = 72.6%
---
 
### 🔹 Semana 6 — Clasificación con modelos de ensamble
Dataset académico con variable objetivo `preparacion_laboral`
 
- Exploración: histogramas, matriz de correlación
- Limpieza: eliminación de duplicados, imputación con mediana
- División estratificada: 60% train / 20% validación / 20% test
- Pipeline con `ColumnTransformer`: `StandardScaler` + `OneHotEncoder`
- Modelos entrenados:
  - 🌳 **Random Forest** (200 estimadores)
  - 🚀 **Gradient Boosting** (200 estimadores, corrección secuencial de errores)
- Visualización de árboles individuales para interpretabilidad
- Evaluación con Accuracy, Precision, Recall, F1, AUC, matrices de confusión y curvas ROC
---
 
### 🔹 Semana 8 — Mini Proyecto: Clasificación Financiera FIRE-UdeA
Dataset: `dataset_sintetico_FIRE_UdeA_realista.csv` · Variable objetivo: `label` (estable / crítica)
 
- EDA completo con visualizaciones interactivas: Scatter Matrix, Coordenadas Paralelas, UMAP 2D y 3D
- Limpieza y pipeline de preprocesamiento con `ColumnTransformer`
- Modelos optimizados con `GridSearchCV` (cv=5):
  - 🌳 **Random Forest**
  - 🚀 **Gradient Boosting**
- Evaluación en train, validación y test: Accuracy, Precision, Recall, F1, AUC
- 🌐 Dashboard financiero interactivo desarrollado en Next.js
---
 
### 🔹 Semana 9 — Clustering: KMeans y DBSCAN
Dataset: FIRE-UdeA · Enfoque no supervisado
 
- Preprocesamiento: imputación + `StandardScaler` + `OneHotEncoder`
- Reducción de dimensión con **PCA** para visualización 2D
- Selección de k óptimo: método del codo + Silhouette Score
- **KMeans** con k óptimo y visualización de clusters en espacio PCA
- **DBSCAN** por densidad con identificación de puntos de ruido
- Evaluación contra etiquetas reales mediante **Adjusted Rand Index (ARI)**
---
 
### 🔹 Semana 10 — Clustering Avanzado: Subtractive + Fuzzy C-Means
Dataset: FIRE-UdeA · Comparación de cuatro algoritmos
 
- Implementaciones propias desde cero:
  - 🔵 **Subtractive Clustering**: detección automática de centros por potencial de densidad
  - 🟡 **Fuzzy C-Means (FCM)**: asignación borrosa con grados de pertenencia μ
- Visualización 2D y 3D con PCA para los cuatro métodos (KMeans, DBSCAN, Subtractive, FCM)
- Evaluación con ARI contra etiquetas reales
- Análisis de errores por unidad académica con exportación a CSV
---
 
## 📂 Proyecto 1
 
Dos subcarpetas independientes (`Encuesta` y `Kaggle`), cada una con pipeline completo:
 
| Etapa | Descripción |
|---|---|
| Carga y limpieza | Manejo de nulos, tipos y duplicados |
| Transformación | Escala Likert a numérico, encoding de variables |
| Estadísticas descriptivas | Media, mediana, moda, IQR |
| Detección de outliers | Método IQR |
| Visualización | Boxplots, histogramas, scatter plots, heatmaps |
| Análisis de correlación | Matriz de correlación entre variables |
| Escalado | `StandardScaler` |
| Salida | Dataset procesado listo para modelos de ML |
 
---
 
## 🏁 Proyecto Final — Reemplazabilidad de la IA en los oficios
 
**Problema:** ¿Qué tan vulnerables son los distintos trabajos y profesiones ante el avance de la Inteligencia Artificial?
 
**Enfoque:** Análisis basado en datos de características laborales para predecir y clasificar el nivel de automatización al que están expuestos distintos oficios.
 
**Pipeline desarrollado:**
- Recopilación y limpieza de datos sobre características de ocupaciones
- Ingeniería de características relevantes al riesgo de automatización
- Modelos de clasificación y análisis de factores determinantes
- Visualización de resultados por sector, tipo de tarea y nivel de cualificación
**Conclusión clave:** Las tareas rutinarias y repetitivas presentan mayor vulnerabilidad frente a la IA, mientras que las habilidades interpersonales, creativas y de pensamiento crítico siguen siendo difícilmente reemplazables.
 
---
 
## 🛠️ Tecnologías utilizadas
 
`Python` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn` · `Scikit-learn` · `Plotly` · `UMAP` · `Next.js`
 
---
 
## 💡 Conclusiones generales
 
- La calidad del preprocesamiento determina el rendimiento de cualquier modelo
- La visualización es clave para entender la estructura real de los datos
- El escalado de features es un paso no negociable antes del modelado
- Los modelos de ensamble ofrecen una línea base sólida en clasificación
- El clustering permite validar si las etiquetas reflejan la estructura natural del dato
- Alta precisión en entrenamiento no garantiza buen desempeño en producción
