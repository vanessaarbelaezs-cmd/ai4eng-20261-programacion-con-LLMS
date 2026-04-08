import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def generar_caso_de_uso_detectar_anomalias_financieras():
    """
    Genera un caso de uso aleatorio (input/output) para la función
    detectar_anomalias_financieras.
    """
    # 1. Parámetros aleatorios
    n_filas = np.random.randint(50, 200)
    n_nulos = np.random.randint(0, 10)
    contaminacion = np.random.uniform(0.01, 0.2) # Proporción de anomalías

    # 2. Generar datos base (Transacciones normales)
    # Simulamos: Monto (media 500), Hora (0-23), ID_Terminal (1-100)
    data = {
        'monto': np.random.normal(500, 100, n_filas),
        'hora': np.random.randint(0, 24, n_filas),
        'id_terminal': np.random.randint(1, 101, n_filas)
    }
    df = pd.DataFrame(data)

    # 3. Insertar algunas anomalías manuales para asegurar que el modelo tenga qué detectar
    n_outliers = int(n_filas * contaminacion)
    indices_outliers = np.random.choice(df.index, n_outliers, replace=False)
    df.loc[indices_outliers, 'monto'] = df.loc[indices_outliers, 'monto'] * 10 # Montos exagerados

    # 4. Insertar valores nulos (NaN) aleatoriamente para probar la limpieza
    for _ in range(n_nulos):
        fila = np.random.randint(0, n_filas)
        col = np.random.choice(df.columns)
        df.at[fila, col] = np.nan

    # --- Lógica para generar el OUTPUT esperado ---
    # Paso A: Limpieza de nulos (Requerimiento del problema)
    df_limpio = df.dropna().copy()

    # Paso B: Entrenamiento y predicción
    model = IsolationForest(contamination=contaminacion, random_state=42)
    # El output esperado según el problema es un numpy.ndarray de 1 y -1
    y_pred = model.fit_predict(df_limpio)

    # 5. Estructurar el Input y el Output
    input_val = {
        "df": df,
        "proporcion_anomalias": contaminacion
    }

    output_val = y_pred

    return input_val, output_val

# Ejemplo de uso:
input_data, expected_output = generar_caso_de_uso_preparar_datos()
print(f"Filas generadas: {len(input_data['df'])}")
print(f"Proporción: {input_data['proporcion_anomalias']:.2f}")
print(f"Predicciones (primeras 5): {expected_output[:5]}")

def detectar_anomalias_financieras(df, proporcion_anomalias):
    """
    Detecta anomalías en transacciones bancarias.

    1. Limpia filas con valores nulos (NaN).
    2. Entrena un modelo IsolationForest.
    3. Devuelve un array con 1 (normal) y -1 (anomalía).
    """
    # Requerimiento 1: Limpieza de valores nulos
    # Es vital usar .copy() para evitar warnings si se modifica el DF original
    df_limpio = df.dropna().copy()

    # Requerimiento 2: Configurar el modelo
    # Usamos random_state=42 para que el resultado sea reproducible y coincida
    # con el generador de casos de uso si este también lo usa.
    modelo = IsolationForest(
        contamination=proporcion_anomalias,
        random_state=42
    )

    # Requerimiento 3: Entrenar y predecir
    # fit_predict devuelve 1 para 'inliers' y -1 para 'outliers'
    predicciones = modelo.fit_predict(df_limpio)

    return predicciones

# --- PRUEBA DE COMPROBACIÓN ---

# 1. Generamos el caso de uso con la función que te di anteriormente
input_data, output_esperado = generar_caso_de_uso_preparar_datos()

# 2. Ejecutamos tu función de solución con el input generado
resultado_obtenido = detectar_anomalias_financieras(
    input_data['df'],
    input_data['proporcion_anomalias']
)

# 3. Comprobación de integridad
son_identicos = np.array_equal(resultado_obtenido, output_esperado)

print(f"¿El resultado coincide con el esperado?: {son_identicos}")
print(f"Registros procesados (sin nulos): {len(resultado_obtenido)}")
