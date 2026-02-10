"""
EDA_TITANIC.py
==============
Archivo en Python para hacer el EDA del dataset Titanic, punto por punto:

1) Cargar el conjunto de datos y explorarlo (incluye exploración gráfica básica).
2) Medidas de tendencia central (media, mediana, moda).
3) Medidas de dispersión (varianza, desviación estándar, rango, IQR).
4) Medidas de posición (cuartiles/percentiles) y eliminación de outliers si es necesario (método IQR).
5) Histogramas para analizar la distribución.
6) Gráficos de dispersión entre dos columnas para analizar relación.

IMPORTANTE (ruta):
- Este script asume que "Titanic-Dataset.csv" está en el MISMO folder que este .py

Cómo ejecutar:
- Abre terminal en VSCode en la carpeta Semana4 y corre:
  python EDA_TITANIC.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica
import matplotlib.pyplot as plt
import os


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Obtener la ruta del archivo actual y usarla como directorio base
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "Titanic-Dataset.csv")  # <-- ruta absoluta
OUTLIER_FACTOR = 1.5              # <-- estándar IQR (1.5)


# ============================================================
# FUNCIONES DE APOYO
# ============================================================

def print_title(title: str) -> None:
    """Imprime títulos de secciones de forma clara."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Devuelve una lista de columnas numéricas (int/float)."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def medidas_tendencia_central(df: pd.DataFrame, cols_num: list) -> None:
    """
    Calcula e imprime:
    - Media
    - Mediana
    - Moda (primera)
    """
    print_title("2) MEDIDAS DE TENDENCIA CENTRAL")

    media = df[cols_num].mean(numeric_only=True)
    mediana = df[cols_num].median(numeric_only=True)
    moda = df[cols_num].mode(numeric_only=True).iloc[0]

    print("\nMedia:")
    print(media)

    print("\nMediana:")
    print(mediana)

    print("\nModa (primera por columna):")
    print(moda)


def medidas_dispersion(df: pd.DataFrame, cols_num: list) -> None:
    """
    Calcula e imprime:
    - Varianza
    - Desviación estándar
    - Rango
    - IQR
    """
    print_title("3) MEDIDAS DE DISPERSIÓN")

    varianza = df[cols_num].var(numeric_only=True)
    desv_std = df[cols_num].std(numeric_only=True)
    rango = df[cols_num].max(numeric_only=True) - df[cols_num].min(numeric_only=True)

    q1 = df[cols_num].quantile(0.25, numeric_only=True)
    q3 = df[cols_num].quantile(0.75, numeric_only=True)
    iqr = q3 - q1

    print("\nVarianza:")
    print(varianza)

    print("\nDesviación estándar:")
    print(desv_std)

    print("\nRango (max - min):")
    print(rango)

    print("\nIQR (Q3 - Q1):")
    print(iqr)


def medidas_posicion(df: pd.DataFrame, cols_num: list) -> None:
    """
    Calcula e imprime:
    - Cuartiles (Q1, Q2, Q3)
    - Percentiles (10%, 90%, 95%, 99%)
    """
    print_title("4) MEDIDAS DE POSICIÓN")

    cuartiles = df[cols_num].quantile([0.25, 0.50, 0.75], numeric_only=True)
    percentiles = df[cols_num].quantile([0.10, 0.90, 0.95, 0.99], numeric_only=True)

    print("\nCuartiles (Q1, Q2, Q3):")
    print(cuartiles)

    print("\nPercentiles (10%, 90%, 95%, 99%):")
    print(percentiles)


def resumen_y_filtrado_outliers_iqr(
    df: pd.DataFrame,
    columnas: list,
    factor: float = 1.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detecta outliers usando el método IQR:
      lim_inf = Q1 - factor*IQR
      lim_sup = Q3 + factor*IQR

    - Genera un resumen con conteo de outliers por columna.
    - Filtra el dataframe usando una máscara conjunta (se eliminan filas outlier
      en cualquiera de las columnas seleccionadas).

    Retorna:
    - df_filtrado: DataFrame sin outliers (según la regla conjunta)
    - resumen_outliers: DataFrame con límites y cantidad de outliers
    """
    df_temp = df.copy()
    resumen = []

    # Máscara global: True = se conserva
    mask_global = pd.Series(True, index=df_temp.index)

    for col in columnas:
        if col not in df_temp.columns:
            continue

        q1 = df_temp[col].quantile(0.25)
        q3 = df_temp[col].quantile(0.75)
        iqr = q3 - q1

        # Si IQR es 0 o NaN, evitamos límites raros
        if pd.isna(iqr) or iqr == 0:
            resumen.append({"columna": col, "outliers": 0, "lim_inf": np.nan, "lim_sup": np.nan})
            continue

        lim_inf = q1 - factor * iqr
        lim_sup = q3 + factor * iqr

        # Outlier si está fuera del rango
        out_mask = (df_temp[col] < lim_inf) | (df_temp[col] > lim_sup)
        out_count = int(out_mask.sum())

        resumen.append({"columna": col, "outliers": out_count, "lim_inf": lim_inf, "lim_sup": lim_sup})

        # No eliminamos NaN por comparación: NaN se conserva
        mask_col = (~out_mask) | (df_temp[col].isna())
        mask_global &= mask_col

    resumen_outliers = pd.DataFrame(resumen).sort_values("outliers", ascending=False)
    df_filtrado = df_temp[mask_global].copy()

    return df_filtrado, resumen_outliers


def histogramas(df: pd.DataFrame, cols_num: list, bins: int = 20) -> None:
    """Grafica histogramas para columnas numéricas."""
    print_title("5) HISTOGRAMAS (DISTRIBUCIÓN DE COLUMNAS NUMÉRICAS)")

    df[cols_num].hist(figsize=(12, 8), bins=bins)
    plt.suptitle("Histogramas de columnas numéricas")
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "histogramas.png"), dpi=100, bbox_inches='tight')
    print("[OK] Grafico guardado: histogramas.png")
    plt.close()


def scatter_plot(df: pd.DataFrame, x: str, y: str) -> None:
    """
    Grafica dispersión entre dos columnas para analizar relación.
    Se eliminan nulos SOLO en esas columnas para que el gráfico quede limpio.
    """
    data = df[[x, y]].dropna()

    plt.figure(figsize=(7, 5))
    plt.scatter(data[x], data[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Dispersión: {x} vs {y}")
    plt.tight_layout()
    filename = os.path.join(SCRIPT_DIR, f"scatter_{x}_vs_{y}.png")
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"[OK] Grafico guardado: scatter_{x}_vs_{y}.png")
    plt.close()


# ============================================================
# MAIN (EJECUCIÓN PRINCIPAL)
# ============================================================

def main() -> None:
    # --------------------------------------------------------
    # 1) CARGAR EL DATASET + EXPLORACIÓN (incluye gráfica básica)
    # --------------------------------------------------------
    print_title("1) CARGA DEL DATASET + EXPLORACIÓN INICIAL")

    df = pd.read_csv(CSV_PATH)

    print("[OK] Dataset cargado:", CSV_PATH)
    print("Dimensiones (filas, columnas):", df.shape)

    print("\nPrimeras 5 filas (head):")
    print(df.head())

    print("\nInformación general (info):")
    df.info()

    print("\nConteo de nulos por columna:")
    print(df.isna().sum())

    print("\nDescripción estadística (describe) - numéricas:")
    print(df.describe())

    # Columnas numéricas detectadas
    cols_num = get_numeric_columns(df)
    print("\nColumnas numéricas detectadas:", cols_num)

    # Exploración gráfica básica (rápida):
    # - Barras: sobrevivientes
    # - Barras: clase
    print("\nExploración gráfica básica (barras):")

    if "Survived" in df.columns:
        plt.figure(figsize=(5, 4))
        df["Survived"].value_counts().plot(kind="bar")
        plt.title("Conteo de Survived (0 = No, 1 = Sí)")
        plt.xlabel("Survived")
        plt.ylabel("Cantidad")
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, "survived_count.png"), dpi=100, bbox_inches='tight')
        print("[OK] Grafico guardado: survived_count.png")
        plt.close()

    if "Pclass" in df.columns:
        plt.figure(figsize=(5, 4))
        df["Pclass"].value_counts().sort_index().plot(kind="bar")
        plt.title("Conteo por Pclass")
        plt.xlabel("Pclass")
        plt.ylabel("Cantidad")
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, "pclass_count.png"), dpi=100, bbox_inches='tight')
        print("[OK] Grafico guardado: pclass_count.png")
        plt.close()

    # --------------------------------------------------------
    # 2) MEDIDAS DE TENDENCIA CENTRAL
    # --------------------------------------------------------
    medidas_tendencia_central(df, cols_num)

    # --------------------------------------------------------
    # 3) MEDIDAS DE DISPERSIÓN
    # --------------------------------------------------------
    medidas_dispersion(df, cols_num)

    # --------------------------------------------------------
    # 4) MEDIDAS DE POSICIÓN + OUTLIERS
    # --------------------------------------------------------
    medidas_posicion(df, cols_num)

    print_title("4.1) DETECCIÓN Y (SI ES NECESARIO) ELIMINACIÓN DE OUTLIERS (IQR)")

    # Elegimos columnas típicas donde Titanic suele tener outliers claros.
    columnas_outliers = [c for c in ["Fare", "Age", "SibSp", "Parch"] if c in df.columns]

    df_sin_outliers, resumen_out = resumen_y_filtrado_outliers_iqr(
        df=df,
        columnas=columnas_outliers,
        factor=OUTLIER_FACTOR
    )

    print("Columnas evaluadas para outliers:", columnas_outliers)
    print("\nResumen de outliers por columna (IQR):")
    print(resumen_out)

    print("\nTamaño original:", df.shape)
    print("Tamaño sin outliers (filtro conjunto):", df_sin_outliers.shape)

    # Decisión práctica:
    # - Para gráficos (histogramas y dispersión) usar df_sin_outliers para que se vean mejor.
    df_graficos = df_sin_outliers
    cols_num_graficos = get_numeric_columns(df_graficos)

    # --------------------------------------------------------
    # 5) HISTOGRAMAS
    # --------------------------------------------------------
    histogramas(df_graficos, cols_num_graficos, bins=20)

    # --------------------------------------------------------
    # 6) DISPERSIÓN ENTRE DOS COLUMNAS
    # --------------------------------------------------------
    print_title("6) GRÁFICOS DE DISPERSIÓN (RELACIÓN ENTRE 2 COLUMNAS)")

    # Escogemos pares comunes y útiles en Titanic
    pares = []
    if "Age" in df_graficos.columns and "Fare" in df_graficos.columns:
        pares.append(("Age", "Fare"))
    if "Pclass" in df_graficos.columns and "Fare" in df_graficos.columns:
        pares.append(("Pclass", "Fare"))

    if len(pares) == 0:
        print("⚠️ No se encontraron pares adecuados para dispersión (revisa columnas).")
    else:
        for x, y in pares:
            print(f"Mostrando dispersión: {x} vs {y}")
            scatter_plot(df_graficos, x, y)

    print_title("[OK] EDA COMPLETADO")
    print("Listo. Este archivo cumple los puntos del EDA solicitados (sin transformaciones).")


if __name__ == "__main__":
    main()
