import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generar_caso_de_uso_preparar_datos():
    """
    Genera un caso de uso aleatorio para la función de reducción de dimensiones.
    Devuelve un diccionario (input) y un array de numpy (output).
    """
    # 1. Configuración aleatoria del tamaño de la muestra
    n_filas = np.random.randint(10, 50)
    n_columnas = 15

    # 2. Generar datos aleatorios
    data = np.random.randn(n_filas, n_columnas) * 100
    columnas = [f"sensor_{i+1}" for i in range(n_columnas)]
    df = pd.DataFrame(data, columns=columnas)

    # 3. Introducir valores nulos (NaN) aleatoriamente en algunas filas
    # Decidimos cuántas filas tendrán nulos (entre 0 y el 20% de las filas)
    n_nulos = np.random.randint(0, max(1, int(n_filas * 0.2)))
    if n_nulos > 0:
        indices_nulos = np.random.choice(df.index, n_nulos, replace=False)
        for idx in indices_nulos:
            col_random = np.random.choice(columnas)
            df.loc[idx, col_random] = np.nan

    # --- Cálculo del OUTPUT esperado (Siguiendo la lógica del problema) ---

    # Paso 1: Limpieza (Eliminar filas con NaN)
    df_limpio = df.dropna()

    # Paso 2: Escalamiento (StandardScaler)
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_limpio)

    # Paso 3 y 4: Reducción (PCA a 2 componentes)
    pca = PCA(n_components=2)
    output_esperado = pca.fit_transform(datos_escalados)

    # --- Formatear Retorno ---

    # El input es un diccionario con la clave del argumento esperado
    input_dict = {
        "df_lecturas": df
    }

    return input_dict, output_esperado

# Ejemplo de uso:
input_data, output_data = generar_caso_de_uso_preparar_datos()
print(f"Input DataFrame shape: {input_data['df_lecturas'].shape}")
print(f"Output array shape: {output_data.shape}")


#validacion del caso de uso y la solucion
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def reducir_dimensiones_sensores(df_lecturas):
    """
    Resuelve el problema de reducción de dimensiones:
    1. Limpia nulos.
    2. Escala datos (Media 0, Varianza 1).
    3. Aplica PCA para reducir a 2 componentes.
    """
    # 1. Limpieza: Elimina filas con cualquier valor nulo
    df_limpio = df_lecturas.dropna()

    # 2. Escalamiento: StandardScaler requiere que los datos tengan media 0 y var 1
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_limpio)

    # 3. Reducción: Configurar PCA para 2 componentes
    pca = PCA(n_components=2)

    # 4. Transformación: Entrenar y aplicar la reducción
    resultado_pca = pca.fit_transform(datos_escalados)

    return resultado_pca

    # 1. Generamos un caso de uso aleatorio con la función anterior
input_data, output_esperado = generar_caso_de_uso_preparar_datos()

# 2. Ejecutamos la función de resolución con el input generado
resultado_real = reducir_dimensiones_sensores(input_data["df_lecturas"])

# 3. Comprobamos si son iguales
son_iguales = np.allclose(resultado_real, output_esperado)

print(f"¿La función resolvió el caso correctamente?: {son_iguales}")

if son_iguales:
    print("¡Perfecto! El generador y la función están sincronizados.")
else:
    print("Hay una discrepancia en los resultados.")
