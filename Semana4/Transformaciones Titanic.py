"""
TRANSFORMACIONES_TITANIC_FINAL.py
================================
Este archivo genera UN SOLO CSV final (dataset Titanic transformado) cumpliendo:

• One Hot Encoding, Label Encoding, Binary Encoding.
• Correlación de columnas (para justificar posible eliminación).
• Escalado (StandardScaler por defecto; puedes cambiar a MinMaxScaler).
• Transformación logarítmica si es necesaria (Fare suele ser sesgada).
"""

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Se usa BinaryEncoder.
try:
    from category_encoders.binary import BinaryEncoder
    HAY_BINARY_ENCODER = True
except Exception:
    HAY_BINARY_ENCODER = False


# ============================================================
# 0) RUTAS (para que NO falle el FileNotFoundError)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Titanic-Dataset.csv")

# Un solo archivo final (lo que me pediste)
SALIDA_FINAL = os.path.join(BASE_DIR, "titanic_final_transformado.csv")


# ============================================================
# 1) CARGA + LIMPIEZA MÍNIMA (antes de transformar)
# ============================================================

def cargar_y_limpiar() -> pd.DataFrame:
    """
    Carga el dataset y realiza una limpieza mínima para poder transformar sin errores.
    - Age: relleno con mediana (porque hay nulos y la mediana es robusta)
    - Embarked: relleno con moda (categoría más frecuente)
    - Fare: por seguridad, relleno con mediana si hubiera nulos
    - Cabin: creo una variable binaria HasCabin (tener cabina o no)
    """
    print("=" * 90)
    print("1) CARGA + LIMPIEZA MÍNIMA")
    print("=" * 90)

    print("Ruta CSV:", CSV_PATH)
    print("¿Existe el archivo?:", os.path.exists(CSV_PATH))

    df = pd.read_csv(CSV_PATH)

    print("\nDimensiones iniciales:", df.shape)
    print("\nNulos por columna (antes):")
    print(df.isna().sum())

    # Age: imputación con mediana
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    # Embarked: imputación con moda
    if "Embarked" in df.columns and df["Embarked"].isna().sum() > 0:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])

    # Fare: imputación con mediana si aplica
    if "Fare" in df.columns and df["Fare"].isna().sum() > 0:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Cabin: crear indicador (esto me ayuda a aprovechar la info sin depender del texto de Cabin)
    if "Cabin" in df.columns:
        df["HasCabin"] = df["Cabin"].notna().astype(int)

    print("\nNulos por columna (después):")
    print(df.isna().sum())

    return df


# ============================================================
# 2) TRANSFORMACIÓN LOGARÍTMICA (si es necesaria)
# ============================================================

def aplicar_log_si_necesario(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fare suele ser muy sesgada a la derecha (hay pocos boletos muy caros).
    Entonces aplico log1p(Fare) si el sesgo es fuerte (|skew| > 1).
    """
    print("\n" + "=" * 90)
    print("2) TRANSFORMACIÓN LOGARÍTMICA (SI ES NECESARIA)")
    print("=" * 90)

    df2 = df.copy()

    if "Fare" not in df2.columns:
        print("No existe Fare, no aplico log.")
        return df2

    skew_fare = float(df2["Fare"].dropna().skew())
    print(f"Skewness(Fare) = {skew_fare:.3f}")

    # Regla práctica: si el sesgo es grande, transformo
    if abs(skew_fare) > 1:
        df2["Fare_log1p"] = np.log1p(df2["Fare"])
        print(" Se creó Fare_log1p = log(1 + Fare) para reducir sesgo.")
        print("Skewness(Fare_log1p) =", float(df2["Fare_log1p"].skew()))
    else:
        print("Fare no tiene sesgo fuerte según el umbral. No se crea Fare_log1p.")

    return df2


# ============================================================
# 3) ONE HOT ENCODING
# ============================================================

def one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    One Hot Encoding para columnas categóricas con pocas categorías:
    - Sex
    - Embarked
    Uso drop_first=True para evitar colinealidad (dummy trap).
    """
    print("\n" + "=" * 90)
    print("3) ONE HOT ENCODING")
    print("=" * 90)

    df_ohe = df.copy()
    cols_ohe = [c for c in ["Sex", "Embarked"] if c in df_ohe.columns]

    df_ohe = pd.get_dummies(df_ohe, columns=cols_ohe, drop_first=True)

    print("Columnas One-Hot:", cols_ohe)
    print("Dimensiones luego de One-Hot:", df_ohe.shape)

    return df_ohe


# ============================================================
# 4) LABEL ENCODING (como demostración clara)
# ============================================================

def label_encoding_demo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label Encoding: lo muestro claramente creando una columna nueva.
    En Titanic, puedo aplicarlo a Sex (si existe) para obtener 0/1.
    OJO: Label Encoding puede meter un orden artificial si la variable tiene >2 categorías.
    """
    print("\n" + "=" * 90)
    print("4) LABEL ENCODING")
    print("=" * 90)

    df_le = df.copy()

    if "Sex" in df_le.columns:
        le = LabelEncoder()
        df_le["Sex_Label"] = le.fit_transform(df_le["Sex"].astype(str))
        print(" Se creó Sex_Label con LabelEncoder.")
        print("Clases:", list(le.classes_))
        print(df_le[["Sex", "Sex_Label"]].head())
    else:
        print("No existe Sex, no aplico LabelEncoder.")

    return df_le


# ============================================================
# 5) BINARY ENCODING (Ticket)
# ============================================================

def binary_encoding_ticket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary Encoding para una columna de alta cardinalidad (Ticket).
    - Si category_encoders está instalado => uso BinaryEncoder
    - Si no => hago una versión manual (factorize + bits)

    Nota: El objetivo es cumplir el punto y dejar el dataset numérico.
    """
    print("\n" + "=" * 90)
    print("5) BINARY ENCODING (Ticket)")
    print("=" * 90)

    df_bin = df.copy()

    if "Ticket" not in df_bin.columns:
        print("No existe Ticket, no aplico Binary Encoding.")
        return df_bin

    if HAY_BINARY_ENCODER:
        enc = BinaryEncoder(cols=["Ticket"])
        df_bin = enc.fit_transform(df_bin)
        print(" BinaryEncoder aplicado con category_encoders.")
        print("Dimensiones luego de Binary Encoding:", df_bin.shape)
    else:
        print(" category_encoders no está instalado, aplico Binary Encoding manual.")
        print("   (Si quieres el encoder oficial: pip install category_encoders)")

        codes, _ = pd.factorize(df_bin["Ticket"].astype(str))
        df_bin["Ticket_code"] = codes

        max_code = int(df_bin["Ticket_code"].max())
        n_bits = int(np.ceil(np.log2(max_code + 1))) if max_code > 0 else 1

        for bit in range(n_bits):
            df_bin[f"Ticket_bin_{bit}"] = ((df_bin["Ticket_code"].values >> bit) & 1)

        # Elimino Ticket (texto) para dejar dataset más “model-ready”
        df_bin = df_bin.drop(columns=["Ticket"])

        print(f" Binary Encoding manual aplicado. Bits generados: {n_bits}")
        print("Dimensiones luego de Binary Encoding manual:", df_bin.shape)

    return df_bin


# ============================================================
# 6) CORRELACIÓN (para justificar posible eliminación)
# ============================================================

def correlacion_y_redundancia(df: pd.DataFrame, umbral: float = 0.85) -> pd.DataFrame:
    """
    Calcula correlación Pearson entre columnas numéricas.
    Luego lista pares con correlación alta (|r| >= umbral) para justificar eliminación.
    """
    print("\n" + "=" * 90)
    print("6) CORRELACIÓN (POSIBLE ELIMINACIÓN DE COLUMNAS)")
    print("=" * 90)

    df_num = df.select_dtypes(include=[np.number]).copy()
    corr = df_num.corr()

    cols = corr.columns.tolist()
    pares_altos = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iloc[i, j])
            if abs(r) >= umbral:
                pares_altos.append((cols[i], cols[j], r))

    print(f"Umbral usado: |r| >= {umbral}")
    if not pares_altos:
        print("No se encontraron pares altamente correlacionados con ese umbral.")
    else:
        print("Pares altamente correlacionados (posible redundancia):")
        for a, b, r in sorted(pares_altos, key=lambda x: abs(x[2]), reverse=True):
            print(f"- {a} vs {b} => r = {r:.3f}")

        print("\nNota: No elimino automáticamente columnas porque depende del criterio del modelo.")
        print("      Esto sirve para justificar en el informe si decides eliminar alguna.")

    return corr


# ============================================================
# 7) ESCALADO (StandardScaler o MinMaxScaler)
# ============================================================

def escalar(df: pd.DataFrame, metodo: str = "standard") -> pd.DataFrame:
    """
    Escala columnas numéricas.
    - metodo = "standard" => StandardScaler (recomendado si luego uso modelos lineales, SVM, etc.)
    - metodo = "minmax"   => MinMaxScaler (0 a 1)

    No escalo:
    - Survived (target) si existe
    - PassengerId (ID)
    """
    print("\n" + "=" * 90)
    print("7) ESCALADO DE COLUMNAS")
    print("=" * 90)

    df_scaled = df.copy()

    # Selecciono numéricas
    cols_num = df_scaled.select_dtypes(include=[np.number]).columns.tolist()

    # Excluir target y id
    for col_excluir in ["Survived", "PassengerId"]:
        if col_excluir in cols_num:
            cols_num.remove(col_excluir)

    print("Columnas numéricas a escalar:", cols_num)
    print("Método:", metodo)

    if metodo.lower() == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    df_scaled[cols_num] = scaler.fit_transform(df_scaled[cols_num])

    print(" Escalado aplicado.")
    return df_scaled


# ============================================================
# 8) LIMPIEZA FINAL (eliminar columnas no útiles para ML)
# ============================================================

def limpieza_final(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimino columnas de texto que no suelen usarse directamente en ML:
    - Name, Cabin (porque ya creé HasCabin), etc.
    Esto deja el dataset más limpio y totalmente numérico.
    """
    print("\n" + "=" * 90)
    print("8) LIMPIEZA FINAL (QUITAR TEXTO/REDUNDANTE)")
    print("=" * 90)

    df_final = df.copy()

    # Columnas típicas que quito para dataset final
    cols_drop = []
    for c in ["Name", "Cabin"]:
        if c in df_final.columns:
            cols_drop.append(c)

    # Si quedó "Sex" texto y ya tengo Sex_Label o One-Hot, la puedo quitar
    if "Sex" in df_final.columns:
        cols_drop.append("Sex")

    # Si quedó Embarked texto, la quito (si hice One-Hot)
    if "Embarked" in df_final.columns:
        cols_drop.append("Embarked")

    if cols_drop:
        df_final = df_final.drop(columns=cols_drop)
        print("Columnas eliminadas:", cols_drop)
    else:
        print("No se eliminaron columnas adicionales.")

    # Aseguro que no queden categóricas (si quedaran, es señal de que falta codificar algo)
    no_num = df_final.select_dtypes(exclude=[np.number]).columns.tolist()
    if no_num:
        print(" Advertencia: quedaron columnas NO numéricas:", no_num)
        print("   (Puedes decidir codificarlas o eliminarlas)")
    else:
        print(" Dataset final completamente numérico (listo para ML).")

    print("Dimensiones finales:", df_final.shape)
    return df_final


# ============================================================
# 9) CONCLUSIONES COMO COMENTARIOS (listas para tu informe)
# ============================================================

def imprimir_conclusiones(df_original: pd.DataFrame, df_final: pd.DataFrame, corr: pd.DataFrame) -> None:
    """
    Imprime conclusiones, pero también las dejo como comentario aquí abajo
    para que las puedas copiar/pegar en el informe.
    """
    print("\n" + "=" * 90)
    print("CONCLUSIONES (también están comentadas dentro del código)")
    print("=" * 90)

    # Métricas rápidas para conclusiones
    conclusiones = []

    if "Survived" in df_original.columns:
        tasa = float(df_original["Survived"].mean()) * 100
        conclusiones.append(f"- La tasa de supervivencia es aproximadamente {tasa:.1f}%, lo que sugiere un desbalance moderado en la clase objetivo.")

    # Nulos importantes (antes de limpiar)
    nulos = df_original.isna().sum().sort_values(ascending=False)
    top_nulos = nulos[nulos > 0].head(3)
    if len(top_nulos) > 0:
        conclusiones.append(
            "- Se detectaron valores faltantes en columnas clave (por ejemplo Age). "
            "Por eso se aplicó imputación con mediana/moda para evitar perder muchas filas."
        )

    if "Fare" in df_original.columns:
        skew_fare = float(df_original["Fare"].dropna().skew())
        conclusiones.append(
            f"- 'Fare' presenta sesgo (skewness={skew_fare:.2f}), normalmente por pocos boletos muy caros. "
            "Se recomienda/usa transformación logarítmica (log1p) para estabilizar la distribución."
        )

    # Correlaciones con Survived
    if "Survived" in corr.columns:
        corr_surv = corr["Survived"].drop("Survived").sort_values(key=lambda s: s.abs(), ascending=False).head(5)
        conclusiones.append(
            "- Según correlación Pearson, algunas variables numéricas muestran relación con la supervivencia (top 5 por |r|): "
            + ", ".join([f"{k} (r={v:.2f})" for k, v in corr_surv.items()])
            + ". Esto sugiere que pueden ser predictoras útiles (sin implicar causalidad)."
        )

    conclusiones.append(
        "- Al aplicar One-Hot, Label y Binary Encoding, el dataset queda en formato numérico, "
        "lo cual es esencial para entrenar modelos de aprendizaje automático."
    )

    conclusiones.append(
        "- El escalado (StandardScaler/MinMax) es importante porque muchos modelos se ven afectados por diferencias de escala "
        "(por ejemplo, regresión logística, SVM, KNN)."
    )

    conclusiones.append(
        f"- El dataset final transformado quedó con {df_final.shape[0]} filas y {df_final.shape[1]} columnas, "
        "listo para modelado."
    )

    for c in conclusiones:
        print(c)

    # ------------------------------------------------------------
    # CONCLUSIONES (LISTAS PARA PEGAR EN INFORME) - COMO COMENTARIOS
    # ------------------------------------------------------------
    # 1) La tasa de supervivencia es aproximadamente X%, lo que sugiere desbalance moderado en la variable objetivo.
    # 2) Se detectaron valores faltantes (Age, etc.). Se aplicó imputación con mediana/moda para no perder muchas filas.
    # 3) Fare presenta sesgo a la derecha por valores altos. La transformación log1p ayuda a estabilizar la distribución.
    # 4) Con la correlación Pearson se identifican variables con relación a Survived, útiles como predictoras.
    # 5) Con One-Hot/Label/Binary Encoding las variables categóricas quedan numéricas, necesarias para ML.
    # 6) El escalado mejora el rendimiento/estabilidad en modelos sensibles a la magnitud de las variables.
    # 7) El dataset final queda limpio, transformado y listo para entrenar modelos con mayor probabilidad de buen desempeño.


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # 1) Cargar + limpiar
    df_original = pd.read_csv(CSV_PATH)  # para conclusiones sobre el dataset original (sin imputación)
    df = cargar_y_limpiar()

    # 2) Log si hace falta
    df = aplicar_log_si_necesario(df)

    # 3) One-Hot (Sex, Embarked)
    df = one_hot_encoding(df)

    # 4) Label Encoding (demo con Sex, pero ojo: si ya one-hot, Sex puede no existir)
    # Por eso lo aplico ANTES de quitar columnas, y solo si existe.
    # En este flujo, Sex ya pudo haberse convertido a dummies y no existir.
    # Para cumplir el punto, lo aplico sobre una copia base y luego integro Sex_Label si existe.
    df_le_base = label_encoding_demo(pd.read_csv(CSV_PATH))
    if "Sex_Label" in df_le_base.columns:
        # Integro Sex_Label al dataframe principal por PassengerId
        if "PassengerId" in df.columns and "PassengerId" in df_le_base.columns:
            df = df.merge(df_le_base[["PassengerId", "Sex_Label"]], on="PassengerId", how="left")
            print(" Sex_Label integrado al dataset final por PassengerId.")
        else:
            # si no hay PassengerId por algún motivo, la agrego por índice
            df["Sex_Label"] = df_le_base["Sex_Label"].values[:len(df)]
            print("Sex_Label integrado por índice (fallback).")

    # 5) Binary Encoding (Ticket) -> lo aplico sobre una copia base y luego integro columnas resultantes
    df_bin_base = binary_encoding_ticket(pd.read_csv(CSV_PATH))

    # Integro columnas binarias a df usando PassengerId (mejor que por índice)
    if "PassengerId" in df.columns and "PassengerId" in df_bin_base.columns:
        # obtengo columnas nuevas binarias (todas excepto PassengerId y Survived y otras obvias)
        cols_bin = [c for c in df_bin_base.columns if c.startswith("Ticket_")]
        if not cols_bin:
            # si fue manual, las columnas serán Ticket_bin_*
            cols_bin = [c for c in df_bin_base.columns if c.startswith("Ticket_bin_")] + (
                ["Ticket_code"] if "Ticket_code" in df_bin_base.columns else []
            )

        if cols_bin:
            df = df.merge(df_bin_base[["PassengerId"] + cols_bin], on="PassengerId", how="left")
            print(" Columnas de Binary Encoding integradas al dataset final.")
        else:
            print(" No se detectaron columnas binarias nuevas para integrar.")
    else:
        print(" No pude integrar Binary Encoding por PassengerId (revisa columnas).")

    # 6) Correlación (sobre df actual, ya numérico en gran parte)
    corr = correlacion_y_redundancia(df, umbral=0.85)

    # 7) Escalado (elige: "standard" o "minmax")
    df = escalar(df, metodo="standard")

    # 8) Limpieza final (quitar texto)
    df_final = limpieza_final(df)

    # 9) Guardar UN SOLO CSV final
    df_final.to_csv(SALIDA_FINAL, index=False)
    print("\n CSV FINAL GUARDADO EN:")
    print(SALIDA_FINAL)

    # 10) Conclusiones (impresas + comentadas)
    imprimir_conclusiones(df_original=df_original, df_final=df_final, corr=corr)


if __name__ == "__main__":
    main()
