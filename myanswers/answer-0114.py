import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


def clasificacion_riesgo_credito(X, y, test_size, random_state):
    """
    Evalúa un modelo SVC para predicción de riesgo de crédito.
    Ajusta el StandardScaler solo con el set de entrenamiento para evitar data leakage.
    """
    # - Dividir los arrays X e y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # - Instanciar StandardScaler y ajustar EXCLUSIVAMENTE con X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # - Instanciar y entrenar modelo SVC con kernel 'rbf'
    model = SVC(kernel='rbf', random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # - Realizar predicciones sobre el conjunto de prueba escalado
    y_pred = model.predict(X_test_scaled)
    
    # - Calcular y devolver métricas en un diccionario
    metricas = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='macro')
    }
    
    return metricas
