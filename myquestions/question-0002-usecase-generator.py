import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def generar_caso_de_uso_segmentar_clientes_ecommerce():
    """
    Genera un caso de uso aleatorio para la función de segmentación de clientes.
    Retorna un diccionario 'input' con un DataFrame y un array 'output' con las etiquetas.
    """
    # 1. Configuración aleatoria del tamaño del dataset (entre 10 y 50 filas)
    n_filas = np.random.randint(10, 51)

    # 2. Generación de datos aleatorios para 'Ingreso_Anual' y 'Puntuacion_Gasto'
    data = {
        'Ingreso_Anual': np.random.uniform(15000, 150000, n_filas),
        'Puntuacion_Gasto': np.random.uniform(1, 100, n_filas)
    }

    df = pd.DataFrame(data)

    # 3. Introducción de valores nulos (NaN) de forma aleatoria (aprox. 10% de probabilidad)
    for col in df.columns:
        mask = np.random.random(n_filas) < 0.1
        df.loc[mask, col] = np.nan

    # --- CÁLCULO DEL OUTPUT ESPERADO ---
    # Paso A: Limpieza (Eliminar NaNs)
    df_limpio = df.dropna().copy()

    # Paso B: Escalamiento
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_limpio)

    # Paso C: Modelado (K-Means con 3 clusters)
    # Usamos random_state fijo aquí para que el output sea determinista para este input específico
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(datos_escalados)

    # --- RESULTADOS ---
    entrada = {"df_clientes": df}
    salida = clusters

    return entrada, salida

# Ejemplo de uso:
input_data, expected_output = generar_caso_de_uso_segmentar_clientes_ecommerce()
print("Input (Primeras filas):\n", input_data["df_clientes"].head())
print("\nOutput (Clusters):\n", expected_output)

#verificacion del caso de uso y la solucion
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def segmentar_clientes_ecommerce(df_clientes):
    """
    Solución al Problema 2:
    1. Limpia valores nulos.
    2. Escala los datos.
    3. Predice 3 clusters usando K-Means.
    """
    # 1. Limpieza: Elimina filas con cualquier valor nulo (NaN)
    df_limpio = df_clientes.dropna().copy()

    # 2. Escalamiento: Normalizar las columnas numéricas
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_limpio)

    # 3. Modelado: K-Means con n_clusters=3
    # Nota: Usamos random_state=42 para que coincida con el generador y sea comparable
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')

    # 4. Resultado: Predicción de clusters
    etiquetas = kmeans.fit_predict(datos_escalados)

    return etiquetas

# --- BLOQUE DE COMPROBACIÓN ---

# 1. Generamos un caso de uso con la función que te di antes
input_data, output_esperado = generar_caso_de_uso_segmentar_clientes_ecommerce()

# 2. Ejecutamos TU nueva función con ese input
resultado_funcion = segmentar_clientes_ecommerce(input_data["df_clientes"])

# 3. Comparamos si son idénticos
son_iguales = (resultado_funcion == output_esperado).all()

print(f"¿La función resolvió el caso correctamente?: {son_iguales}")
if son_iguales:
    print("¡Perfecto! Los clusters generados coinciden con el oráculo.")
else:
    print("Hay una discrepancia en los resultados.")
