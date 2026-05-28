
from sklearn.feature_selection import VarianceThreshold

def diagnosticar_dataset(df, target_col, umbral_varianza=0.01, umbral_correlacion=0.9):
    """
    Genera un reporte automatizado de calidad del dataset para Machine Learning.
    Evalúa nulos, remueve variables de baja varianza tras imputar la mediana
    y extrae pares redundantes del triángulo superior con correlación alta.
    """
    # Identificar características numéricas excluyendo el target
    features_num = [c for c in df.columns if c != target_col and df[c].dtype in ["float64", "int64"]]
    n_features_original = len(features_num)
    X = df[features_num]
    
    # Análisis de faltantes (% de nulos por columna numérica redondeado a 2 decimales)
    porcentaje_nulos = {}
    for col in features_num:
        pct = round(float(X[col].isna().mean() * 100), 2)
        porcentaje_nulos[col] = pct
        
    # Imputación temporal con la mediana para evitar que VarianceThreshold falle
    X_filled = X.copy()
    for col in X_filled.columns:
        mediana = X_filled[col].median()
        X_filled[col] = X_filled[col].fillna(mediana)
        
    # Detección de features cuasi-constantes (Sklearn)
    selector = VarianceThreshold(threshold=umbral_varianza)
    selector.fit(X_filled)
    mascara_varianza = selector.get_support()
    
    # Capturar nombres de columnas que tienen varianza estrictamente menor al umbral
    features_cuasi_constantes = [
        features_num[i] for i in range(n_features_original) if not mascara_varianza[i]
    ]
    
    # Pares redundantes (Numpy/Pandas - triángulo superior estricto)
    corr_matrix = X_filled.corr().abs()
    pares_redundantes = []
    
    for i in range(n_features_original):
        for j in range(i + 1, n_features_original):
            val = corr_matrix.iloc[i, j]
            if val > umbral_correlacion:
                pares_redundantes.append((
                    features_num[i],
                    features_num[j],
                    round(float(val), 4)
                ))
                
    # Cálculo de features útiles finales
    n_features_utiles = n_features_original - len(features_cuasi_constantes)
    
    # Construcción del reporte final esperado
    reporte = {
        "porcentaje_nulos": porcentaje_nulos,
        "features_cuasi_constantes": features_cuasi_constantes,
        "pares_redundantes": pares_redundantes,
        "n_features_original": n_features_original,
        "n_features_utiles": n_features_utiles
    }
    
    return reporte
