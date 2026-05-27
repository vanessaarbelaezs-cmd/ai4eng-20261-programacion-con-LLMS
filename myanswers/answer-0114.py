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


# GENERADOR DE CASOS DE USO 

def generar_caso_de_uso_clasificacion_riesgo_credito():
    n_samples = random.randint(150, 250)
    n_features = random.randint(4, 8)
    
    X = np.random.randn(n_samples, n_features) * random.randint(1, 10)
    y = np.random.randint(0, 2, size=n_samples)
    
    test_size = round(random.uniform(0.2, 0.4), 2)
    rs = random.randint(1, 100)
    
    input_data = {
        'X': X,
        'y': y,
        'test_size': test_size,
        'random_state': rs
    }
    
    # Ground Truth interno
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rs)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = SVC(kernel='rbf', random_state=rs)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    output_data = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='macro')
    }
    
    return input_data, output_data


# BLOQUE FINAL PARA IMPRIMIR LOS RESULTADOS
if __name__ == "__main__":
    # Ejecutamos el generador para obtener un caso aleatorio
    entrada, salida_esperada = generar_caso_de_uso_clasificacion_riesgo_credito()
    
    print("==================================================")
    print("             DATOS DE ENTRADA GENERADOS           ")
    print("==================================================")
    print(f"• Tamaño de la matriz X (Filas, Columnas): {entrada['X'].shape}")
    print(f"• Longitud del vector y (Etiquetas):      {entrada['y'].shape}")
    print(f"• Proporción de test (test_size):          {entrada['test_size']}")
    print(f"• Semilla aleatoria (random_state):        {entrada['random_state']}")
    print("\n" + "="*50)
    
    # Ejecutamos TU función solución pasando el diccionario de entrada
    resultado_de_tu_funcion = clasificacion_riesgo_credito(**entrada)
    
    print("               RESULTADOS OBTENIDOS               ")
    print("==================================================")
    print(f"• Métricas calculadas por tu función:\n  {resultado_de_tu_funcion}")
    print(f"• Métricas esperadas por el generador:\n  {salida_esperada}")
    print("-"*50)
    
    # Verificación de éxito
    if resultado_de_tu_funcion == salida_esperada:
        print("✅ ¡ÉXITO TOTAL! Las respuestas coinciden de forma exacta.")
    else:
        print("❌ ERROR: Hubo una discrepancia en los resultados.")
    print("==================================================")
