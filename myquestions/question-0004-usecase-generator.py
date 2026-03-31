import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def generar_caso_de_uso_preparar_datos():
    """
    Genera un caso de uso aleatorio (input/output) para la función
    calcular_importancia_caracteristicas.
    """

    # 1. Componente Aleatorio: Definir dimensiones y nombres de columnas
    n_filas = np.random.randint(50, 200)
    n_features = np.random.randint(3, 7)
    nombres_columnas = [f"Factor_{i}" for i in range(n_features)] + ["Renuncia"]

    # 2. Creación de datos aleatorios
    # Creamos un DataFrame con valores flotantes y algunos NaN aleatorios
    data = np.random.rand(n_filas, n_features + 1)
    df = pd.DataFrame(data, columns=nombres_columnas)

    # Introducir algunos NaNs aleatorios para probar la limpieza
    for _ in range(np.random.randint(5, 15)):
        row = np.random.randint(0, n_filas)
        col = np.random.randint(0, n_features + 1)
        df.iat[row, col] = np.nan

    # El output de esta función de generación (el input de la función a testear)
    # debe ser el diccionario con el DataFrame original
    input_dict = {"df_empleados": df.copy()}

    # --- Lógica para calcular el OUTPUT esperado ---

    # A. Limpieza: Eliminar NaNs
    df_clean = df.dropna()

    # B. Preparación: Separar X (independientes) e y (target)
    X = df_clean.iloc[:, :-1]
    y = df_clean.iloc[:, -1]

    # Aseguramos que 'y' sea discreto para clasificación (0 o 1)
    y = (y > 0.5).astype(int)

    # C. Modelado: RandomForest con 100 estimadores y random_state 42
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    # D. Extracción: Feature importances
    output_expected = modelo.feature_importances_

    return input_dict, output_expected

# --- Ejemplo de ejecución ---
input_data, output_data = generar_caso_de_uso_preparar_datos()

print("### INPUT (Primeras 5 filas del DataFrame generado) ###")
print(input_data["df_empleados"].head())
print("\n### OUTPUT ESPERADO (Importancia de características) ###")
print(output_data)

#verificacion caso de uso y solucion
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def generar_caso_de_uso():
    # Generar datos sintéticos: 20 filas, 5 variables + 1 target
    df = pd.DataFrame(np.random.rand(20, 6), columns=[f'v{i}' for i in range(5)] + ['target'])
    df['target'] = (df['target'] > 0.5).astype(int)
    df.iloc[0, 0] = np.nan # Insertar un nulo para probar limpieza
    
    # Calcular el output esperado (referencia)
    ref = calcular_importancia_caracteristicas(df.copy())
    return {"df_empleados": df}, ref

def calcular_importancia_caracteristicas(df_empleados):
    df = df_empleados.dropna()
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    modelo = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    return modelo.feature_importances_

# --- Verificación ---
entrada, esperado = generar_caso_de_uso()
real = calcular_importancia_caracteristicas(entrada["df_empleados"])

print(f"¿Coinciden?: {np.allclose(real, esperado)}")
print(f"Importancias:\n{real}")
