import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Importar la función para generar datos
from generate_synthetic_data import generate_synthetic_salary_data

def cargar_y_preprocesar_datos():
    """
    Carga los datos sintéticos y realiza preprocesamiento básico
    """
    print("Cargando y preprocesando datos...")
    
    # Generar datos sintéticos
    data = generate_synthetic_salary_data(1000)
    
    # Separar features y target
    X = data.drop('salario', axis=1)
    y = data['salario']
    
    # Verificar valores nulos
    print(f"Valores nulos en features: {X.isnull().sum().sum()}")
    print(f"Valores nulos en target: {y.isnull().sum()}")
    
    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalización de features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Forma de datos de entrenamiento: {X_train_scaled.shape}")
    print(f"Forma de datos de prueba: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def entrenar_modelo_regresion_lineal(X_train, y_train):
    """
    Entrena un modelo de regresión lineal
    """
    print("Entrenando modelo de regresión lineal...")
    
    # Configurar hiperparámetros
    hiperparametros = {
        'fit_intercept': True,
        'copy_X': True,
        'n_jobs': -1,
        'positive': False
    }
    
    # Crear y entrenar modelo
    modelo = LinearRegression(**hiperparametros)
    modelo.fit(X_train, y_train)
    
    print("Modelo entrenado exitosamente")
    return modelo, hiperparametros

def evaluar_modelo(modelo, X_test, y_test):
    """
    Evalúa el modelo y calcula métricas
    """
    print("Evaluando modelo...")
    
    # Predicciones
    y_pred = modelo.predict(X_test)
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metricas = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
    
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    return metricas, y_pred

def experimento_mlflow():
    """
    Ejecuta el experimento completo con MLflow
    """
    # Configurar MLflow
    mlflow.set_experiment("Prediccion_Salarios_Regresion_Lineal")
    
    with mlflow.start_run(run_name="regresion_lineal_salarios"):
        print("=== INICIANDO EXPERIMENTO MLFLOW ===")
        
        # 1. Cargar y preprocesar datos
        X_train, X_test, y_train, y_test, scaler = cargar_y_preprocesar_datos()
        
        # 2. Entrenar modelo
        modelo, hiperparametros = entrenar_modelo_regresion_lineal(X_train, y_train)
        
        # 3. Evaluar modelo
        metricas, y_pred = evaluar_modelo(modelo, X_test, y_test)
        
        # 4. Registrar hiperparámetros en MLflow
        print("Registrando hiperparámetros...")
        mlflow.log_params(hiperparametros)
        
        # 5. Registrar métricas en MLflow
        print("Registrando métricas...")
        mlflow.log_metrics(metricas)
        
        # 6. Guardar el modelo como artefacto
        print("Guardando modelo...")
        mlflow.sklearn.log_model(
            modelo, 
            "modelo_regresion_lineal",
            registered_model_name="prediccion_salarios"
        )
        
        # 7. Guardar el scaler como artefacto
        mlflow.sklearn.log_model(
            scaler,
            "scaler",
            registered_model_name="scaler_salarios"
        )
        
        # 8. Crear y guardar gráfico de resultados
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Salario Real')
        plt.ylabel('Salario Predicho')
        plt.title('Predicciones vs Valores Reales')
        plt.tight_layout()
        
        # Guardar gráfico
        plt.savefig('predicciones_vs_reales.png')
        mlflow.log_artifact('predicciones_vs_reales.png')
        
        # 9. Guardar información adicional
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_samples_test", X_test.shape[0])
        
        print("=== EXPERIMENTO COMPLETADO ===")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Experimento: {mlflow.active_run().info.experiment_id}")
        
        return modelo, scaler, metricas

def cargar_modelo_desde_mlflow(run_id):
    """
    Función para cargar un modelo guardado desde MLflow
    """
    modelo_cargado = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo_regresion_lineal")
    scaler_cargado = mlflow.sklearn.load_model(f"runs:/{run_id}/scaler")
    return modelo_cargado, scaler_cargado

def hacer_prediccion_ejemplo(modelo, scaler):
    """
    Ejemplo de cómo hacer predicciones con el modelo entrenado
    """
    print("\n=== EJEMPLO DE PREDICCIÓN ===")
    
    # Datos de ejemplo (un empleado)
    empleado_ejemplo = np.array([[
        30,    # edad
        5,     # experiencia_anos
        16,    # educacion_anos
        40,    # horas_trabajo
        10,    # proyectos_completados
        2      # certificaciones
    ]])
    
    # Normalizar datos
    empleado_scaled = scaler.transform(empleado_ejemplo)
    
    # Hacer predicción
    salario_predicho = modelo.predict(empleado_scaled)[0]
    
    print(f"Empleado ejemplo:")
    print(f"- Edad: {empleado_ejemplo[0][0]} años")
    print(f"- Experiencia: {empleado_ejemplo[0][1]} años")
    print(f"- Educación: {empleado_ejemplo[0][2]} años")
    print(f"- Horas trabajo: {empleado_ejemplo[0][3]} horas/semana")
    print(f"- Proyectos completados: {empleado_ejemplo[0][4]}")
    print(f"- Certificaciones: {empleado_ejemplo[0][5]}")
    print(f"\nSalario predicho: ${salario_predicho:,.2f}")

if __name__ == "__main__":
    # Ejecutar experimento completo
    modelo, scaler, metricas = experimento_mlflow()
    
    # Ejemplo de predicción
    hacer_prediccion_ejemplo(modelo, scaler)
    
    print("\n=== RESUMEN ===")
    print("El experimento se ha completado exitosamente.")
    print("Puedes ver los resultados en la interfaz de MLflow ejecutando:")
    print("mlflow ui") 