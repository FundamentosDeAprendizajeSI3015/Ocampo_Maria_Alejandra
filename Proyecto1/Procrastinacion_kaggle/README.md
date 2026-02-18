
# Pipeline de Análisis – Procrastinación Estudiantil (Kaggle)

**Autora:** Maria Alejandra Ocampo Giraldo  
**Curso:** Fundamentos de Aprendizaje Automático  

Este proyecto implementa un **pipeline completo de procesamiento y análisis de datos** usando un **dataset de Kaggle** sobre procrastinación estudiantil.

La pregunta de investigación que guía el análisis es:

> **¿Qué factores influyen en la procrastinación académica?**

El objetivo del pipeline es dejar un dataset **limpio, transformado y listo** para análisis estadístico y futuros modelos de Machine Learning, además de generar visualizaciones y métricas que permiten interpretar patrones relacionados con la procrastinación.

---

##  ¿Qué hace este código?

El script realiza un flujo completo de procesamiento:

### 1) Carga y exploración del dataset
- Lee el archivo `Procrastination.csv`.
- Muestra tamaño del dataset (filas/columnas).
- Imprime columnas disponibles, tipos de datos y primeras filas.

### 2) Selección de variables relevantes
Trabaja solo con las columnas que aportan información para la pregunta de investigación, por ejemplo:
- horas de estudio
- CGPA (promedio académico)
- frecuencia de retraso de tareas
- estrés asociado a procrastinación
- uso de gestión del tiempo
- razones de procrastinación
- uso del celular no académico
- impacto en calificaciones

### 3) Limpieza y calidad de datos
- Detección de valores faltantes (NaNs).
- Eliminación de filas con NaNs (tratamiento directo).
- Limpieza de texto (elimina espacios en columnas categóricas).
- Verificación de duplicados.
- Validación lógica (cantidad de valores únicos por variable).

### 4) Conversión de datos a numéricos
Convierte variables categóricas a valores numéricos para poder analizarlas y modelarlas:
- Rangos → promedios (ej. “0-5 hours” → 2.5)
- Escalas ordinales (ej. Never…Always → 0–4)
- Variables binarias (Yes/No → 1/0)

### 5) Agrupación y agregación (estadística)
Agrupa estudiantes por **frecuencia de retraso en tareas** y calcula:
- media, desviación y conteo del CGPA
- media de estrés
- media de horas de estudio
- proporción de impacto en calificaciones

### 6) One Hot Encoding
Aplica **One Hot Encoding** a `procrastination_reasons` para convertir razones en columnas binarias.

### 7) EDA (análisis exploratorio)
Calcula:
- tendencia central (media/mediana)
- dispersión (std/var/min/max)
- posición (cuartiles) + detección de outliers con IQR

Y genera gráficos:
- boxplots
- histogramas
- scatter plots
- proporciones categóricas (barras)
- correlaciones (Pearson y Spearman)

### 8) Escalado y transformación
Aplica:
- **MinMaxScaler** (rango 0–1)
- **StandardScaler** (media 0, std 1)
- **Log Transform** para suavizar sesgo en horas de móvil

### 9) Dataset final y reducción de dimensionalidad
Guarda un dataset final (`data_procesada.csv`) que incluye:
- variables numéricas escaladas (StandardScaler)
- variable adicional `log_mobile_hours`
- variable `assignment_delay_label`

Además genera embeddings y gráficas con:
- **PCA**
- **t-SNE**
- **UMAP** (si está instalado)

Y guarda:
- archivos `.csv` con embeddings
- modelos `.joblib` para reutilizar

### 10) Reporte de salida
Todo lo que se imprime en consola se guarda en:
- `resultados_analisis.txt`

---

##  Estructura de salida (outputs)

Al ejecutar, el script crea:

###  Archivos principales
- `data_procesada.csv` → dataset final listo para modelos
- `resultados_analisis.txt` → reporte completo del pipeline

###  Carpeta `graficos/`
Contiene imágenes generadas automáticamente (nombres aproximados):
- `01_boxplots_outliers.png`
- `02_histogramas.png`
- `03_scatter_plots.png`
- `04_proporciones_categoricas.png`
- `05_correlacion_pearson.png`
- `06_correlacion_spearman.png`
- `07_comparacion_escalado.png`
- `08_log_transform.png`
- `09_pca.png`
- `10_tsne.png`
- `11_umap.png` *(si UMAP está instalado)*

###  Carpeta `models/`
- `pca_model.joblib`
- `tsne_model.joblib`
- `umap_model.joblib` *(si UMAP está disponible)*

---

##  ¿Cómo ejecutar el script?

### 1) Requisitos
Asegúrate de tener Python instalado (recomendado: **Python 3.10+**).

### 2) Instalar dependencias
En la terminal, dentro de la carpeta del proyecto:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
````

Si quieres activar UMAP:

```bash
pip install umap-learn
```

### 3) Verificar archivo de entrada

El script espera que el dataset esté en la misma carpeta del script con este nombre:

* `Procrastination.csv`

### 4) Ejecutar

Desde la carpeta donde está el script:

```bash
python pipeline_procrastinacion.py
```

*(Cambia el nombre si tu archivo `.py` se llama diferente.)*


---

##  Notas

* Si UMAP no está instalado, el script sigue funcionando y solo omite esa parte.
* Los outputs se guardan automáticamente, no necesitas crear carpetas manualmente.

---


