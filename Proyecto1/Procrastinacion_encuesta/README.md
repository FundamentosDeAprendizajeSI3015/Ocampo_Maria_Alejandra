
# Pipeline Análisis Encuesta de Procrastinación

**Autor:** Maria Alejandra Ocampo Giraldo  
Pipeline en Python para análisis exploratorio (EDA) y preprocesamiento de una encuesta de procrastinación académica. El script carga el dataset, transforma respuestas tipo Likert a valores numéricos, genera estadísticas y visualizaciones, analiza correlaciones y guarda salidas listas para modelado.

---

##  Objetivo
Preparar y analizar un dataset de encuesta (escala Likert) para:
- Entender el comportamiento de las variables (distribución, dispersión, outliers).
- Identificar relaciones entre variables (correlación).
- Generar un dataset procesado y escalado listo para modelos de Machine Learning.

---

##  Entradas y salidas

###  Entrada
- `procastinacion_encuesta.csv` (debe estar en una de las rutas buscadas por el script).

###  Salidas generadas
- `data_procesada.csv` → dataset escalado + variables derivadas.
- `resultados_analisis.txt` → registro completo de la salida en consola.
- Carpeta `graficos/` con:
  - `01_boxplots.png`
  - `02_histogramas.png`
  - `03_scatter_plots.png`
  - `04_correlacion.png`

---

##  Flujo del pipeline

1. **Carga e inspección del dataset**
   - Lectura del CSV y revisión de columnas, tipos de datos y primeras filas.

2. **Selección de variables relevantes**
   - Se trabaja con 8 variables de frecuencia relacionadas con planificación, concentración, distracción digital y procrastinación.

3. **Verificación de calidad**
   - Conteo de valores nulos (NaNs) y duplicados.

4. **Transformación Likert → Numérico**
   - Conversión de respuestas textuales a escala ordinal 1–5.

5. **Estadísticas descriptivas**
   - Medidas de tendencia central (media y mediana).
   - Medidas de dispersión (desviación estándar y varianza).
   - Medidas de posición (percentiles).
   - Resumen general (`describe()`).

6. **Detección de outliers (IQR)**
   - Identificación de valores atípicos con el método IQR.
   - No se eliminan por tratarse de respuestas válidas en escala Likert (1–5).

7. **Visualización**
   - Boxplots, histogramas y scatter plots para analizar distribución y relaciones.
   - Heatmap de correlación de Pearson.

8. **Codificación (Encoding)**
   - `label_planificacion`: Label Encoding sobre planificación.
   - `binary_concentracion`: variable binaria (1 si concentración ≥ 4).
   - One-Hot Encoding de `frecuencia_dejar_ultimo_momento` (generado como `df_onehot` para análisis/soporte).

9. **Análisis de correlación y multicolinealidad**
   - Revisión de correlaciones altas (> 0.85) para decidir eliminación de variables (si aplica).

10. **Escalado (StandardScaler)**
   - Estandarización de variables para modelos sensibles a escala.

11. **Exportación**
   - Construcción de `df_final` y guardado del CSV procesado.
   - Guardado del log completo en TXT.

---

##  Cómo ejecutar

1. Instalar dependencias:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
`

2. Ejecutar el script:

   ```bash
   python pipeline_encuesta.py
   ```


