import pandas as pd

def detectar_solapamientos(df):
    """
    Detecta solapamientos temporales por entidad en un DataFrame.
    Ordena por entidad e inicio, evalúa si la fila actual inicia antes de 
    que termine la anterior de la misma entidad sin usar bucles explícitos.
    """
    # Ordenar por entidad_id e inicio ascendente
    df_ordenado = df.sort_values(by=["entidad_id", "inicio"]).copy()
    
    # Convertir a datetime por seguridad
    df_ordenado["inicio"] = pd.to_datetime(df_ordenado["inicio"])
    df_ordenado["fin"] = pd.to_datetime(df_ordenado["fin"])
    
    # Obtener el fin del intervalo anterior por cada entidad usando groupby + shift
    fin_anterior = df_ordenado.groupby("entidad_id")["fin"].shift(1)
    
    # Es True cuando el inicio actual es menor que el fin anterior
    df_ordenado["solapado"] = (df_ordenado["inicio"] < fin_anterior).fillna(False)
    
    # Devolver el DataFrame con el índice reiniciado
    return df_ordenado.reset_index(drop=True)

