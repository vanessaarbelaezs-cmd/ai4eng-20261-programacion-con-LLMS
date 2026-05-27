from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_eficiencia_energetica(X, y):
    """
    Evalúa la eficiencia energética prediciendo la carga térmica.
    Divide los datos, escala sin cometer data leakage y devuelve el MAE del modelo KNN.
    """
    # Dividir los datos en entrenamiento (80%) y prueba (20%) con random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Escalar las características utilizando StandardScaler (fit solo con X_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar un modelo KNeighborsRegressor con n_neighbors=5
    modelo_knn = KNeighborsRegressor(n_neighbors=5)
    modelo_knn.fit(X_train_scaled, y_train)
    
    # Realizar predicciones sobre el conjunto de prueba escalado
    predicciones = modelo_knn.predict(X_test_scaled)
    
    # Calcular y devolver el Error Absoluto Medio (MAE)
    mae_resultado = mean_absolute_error(y_test, predicciones)
    
    return float(mae_resultado)
