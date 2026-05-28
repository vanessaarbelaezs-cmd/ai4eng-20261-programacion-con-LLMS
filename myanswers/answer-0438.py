import pandas as pd

def detectar_solapamientos(df):
    """
    Detecta solapamientos temporales por entidad en un DataFrame.
    Ordena por entidad e inicio, evalúa si la fila actual inicia antes de 
    que termine la anterior de la misma entidad sin usar bucles explícitos.
    """
    # Ordenar por entidad_id e inicio de forma ascendente
    df_ordenado = df.sort_values(by=["entidad_id", "inicio"]).copy()
    
    # Asegurar el tipo de dato datetime para evitar errores de comparación
    df_ordenado["inicio"] = pd.to_datetime(df_ordenado["inicio"])
    df_ordenado["fin"] = pd.to_datetime(df_ordenado["fin"])
    
    # Obtener el tiempo de fin de la fila anterior por cada entidad (groupby + shift)
    fin_anterior = df_ordenado.groupby("entidad_id")["fin"].shift(1)
    
    # Es True si el inicio actual ocurre antes de que termine el intervalo anterior
    df_ordenado["solapado"] = (df_ordenado["inicio"] < fin_anterior)
    
    # Reemplazar los valores NaN por False (aplica para el primer registro de cada entidad)
    df_ordenado["solapado"] = df_ordenado["solapado"].fillna(False)
    
    # Devolver el DataFrame con el índice completamente reiniciado
    return df_ordenado.reset_index(drop=True)
