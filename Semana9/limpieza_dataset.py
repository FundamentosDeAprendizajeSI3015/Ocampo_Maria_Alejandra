import pandas as pd
import os

# --- 1. Cargar dataset con ruta absoluta ---
base_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_dir, 'dataset_sintetico_FIRE_UdeA_realista.csv')
df = pd.read_csv(input_path)

print("=== Dataset original ===")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
print("\nValores faltantes por columna:")
print(df.isnull().sum())

# --- 2. Separar columnas ---
# 'label' es supervisado, no se usa en clustering
# 'anio' y 'unidad' son categóricas
col_categoricas = ['anio', 'unidad']
col_numericas = [c for c in df.columns if c not in col_categoricas + ['label']]

# --- 3. Imputar valores faltantes con la mediana (robusta a outliers) ---
for col in col_numericas:
    mediana = df[col].median()
    df[col] = df[col].fillna(mediana)

print("\nValores faltantes después de imputación:")
print(df[col_numericas].isnull().sum())

# --- 4. Escalar variables numéricas manualmente (StandardScaler) ---
df_escalado = df[col_numericas].copy()
for col in col_numericas:
    media = df_escalado[col].mean()
    std = df_escalado[col].std()
    df_escalado[col] = (df_escalado[col] - media) / std
df_escalado.columns = [c + '_scaled' for c in col_numericas]

# --- 5. Escalar 'anio' ---
media_anio = df['anio'].mean()
std_anio = df['anio'].std()
df_anio = pd.DataFrame({'anio_scaled': (df['anio'] - media_anio) / std_anio})

# --- 6. Codificar 'unidad' con OneHotEncoding ---
df_unidad = pd.get_dummies(df['unidad'], prefix='unidad').astype(int)

# --- 7. Armar dataset final para clustering (sin 'label') ---
df_limpio = pd.concat([df_anio, df_unidad, df_escalado], axis=1)

print("\n=== Dataset limpio ===")
print(f"Filas: {df_limpio.shape[0]}, Columnas: {df_limpio.shape[1]}")
print(f"Valores faltantes totales: {df_limpio.isnull().sum().sum()}")

# --- 8. Guardar CSV ---
output_path = os.path.join(base_dir, 'dataset_limpio_para_clustering.csv')
df_limpio.to_csv(output_path, index=False)
print(f"\nCSV guardado en: {output_path}")
