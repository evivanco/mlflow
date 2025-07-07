import mlflow
import numpy as np
import pandas as pd
from generate_synthetic_data import generate_synthetic_salary_data

def cargar_modelo_entrenado(run_id):
    """
    Carga el modelo y scaler desde MLflow usando el run_id
    """
    try:
        modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo_regresion_lineal")
        scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/scaler")
        print(f"Modelo cargado exitosamente desde run_id: {run_id}")
        return modelo, scaler
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None, None

def predecir_salario_empleado(modelo, scaler, datos_empleado):
    """
    Predice el salario de un empleado usando el modelo entrenado
    
    Args:
        modelo: Modelo de regresión lineal entrenado
        scaler: Scaler usado para normalizar los datos
        datos_empleado: Lista con [edad, experiencia_anos, educacion_anos, 
                                  horas_trabajo, proyectos_completados, certificaciones]
    """
    # Convertir a array numpy
    datos_array = np.array([datos_empleado])
    
    # Normalizar datos
    datos_scaled = scaler.transform(datos_array)
    
    # Hacer predicción
    salario_predicho = modelo.predict(datos_scaled)[0]
    
    return salario_predicho

def ejemplo_predicciones_multiples(modelo, scaler):
    """
    Ejemplo de predicciones para múltiples empleados
    """
    print("\n=== PREDICCIONES PARA MÚLTIPLES EMPLEADOS ===")
    
    # Lista de empleados de ejemplo
    empleados = [
        [25, 2, 14, 35, 5, 1],    # Empleado junior
        [30, 5, 16, 40, 10, 2],   # Empleado intermedio
        [35, 8, 18, 45, 15, 3],   # Empleado senior
        [40, 12, 20, 50, 25, 5],  # Empleado experto
        [50, 20, 22, 55, 40, 8]   # Empleado muy experimentado
    ]
    
    categorias = ["Junior", "Intermedio", "Senior", "Experto", "Muy Experto"]
    
    for i, (empleado, categoria) in enumerate(zip(empleados, categorias)):
        salario = predecir_salario_empleado(modelo, scaler, empleado)
        
        print(f"\n{categoria}:")
        print(f"  Edad: {empleado[0]} años")
        print(f"  Experiencia: {empleado[1]} años")
        print(f"  Educación: {empleado[2]} años")
        print(f"  Horas trabajo: {empleado[3]} horas/semana")
        print(f"  Proyectos: {empleado[4]}")
        print(f"  Certificaciones: {empleado[5]}")
        print(f"  Salario predicho: ${salario:,.2f}")

def comparar_con_datos_reales(modelo, scaler):
    """
    Compara predicciones con datos reales del dataset
    """
    print("\n=== COMPARACIÓN CON DATOS REALES ===")
    
    # Generar datos de prueba
    data_test = generate_synthetic_salary_data(10)
    
    # Tomar los primeros 5 registros
    muestra = data_test.head(5)
    
    for idx, row in muestra.iterrows():
        # Datos del empleado (sin salario)
        datos_empleado = row.drop('salario').values
        
        # Salario real
        salario_real = row['salario']
        
        # Salario predicho
        salario_predicho = predecir_salario_empleado(modelo, scaler, datos_empleado)
        
        # Calcular diferencia
        diferencia = abs(salario_real - salario_predicho)
        porcentaje_error = (diferencia / salario_real) * 100
        
        print(f"\nEmpleado {idx + 1}:")
        print(f"  Salario real: ${salario_real:,.2f}")
        print(f"  Salario predicho: ${salario_predicho:,.2f}")
        print(f"  Diferencia: ${diferencia:,.2f} ({porcentaje_error:.1f}%)")

def main():
    """
    Función principal que demuestra el uso del modelo
    """
    print("=== EJEMPLO DE USO DEL MODELO DE PREDICCIÓN DE SALARIOS ===")
    
    # NOTA: Necesitas reemplazar este run_id con el que obtuviste al ejecutar el experimento
    # Puedes encontrar el run_id en la salida del script principal o en la interfaz de MLflow
    run_id = "fd1c19a5dee949ad950d2b98b19ea57b"  # Reemplazar con el run_id real
    
    # Cargar modelo
    modelo, scaler = cargar_modelo_entrenado(run_id)
    
    if modelo is None or scaler is None:
        print("No se pudo cargar el modelo. Asegúrate de:")
        print("1. Haber ejecutado primero mlflow_regression_example.py")
        print("2. Usar el run_id correcto")
        print("3. Tener MLflow configurado correctamente")
        return
    
    # Ejemplo de predicción individual
    print("\n=== PREDICCIÓN INDIVIDUAL ===")
    empleado_ejemplo = [28, 4, 16, 42, 8, 2]
    salario = predecir_salario_empleado(modelo, scaler, empleado_ejemplo)
    
    print(f"Empleado ejemplo:")
    print(f"- Edad: {empleado_ejemplo[0]} años")
    print(f"- Experiencia: {empleado_ejemplo[1]} años")
    print(f"- Educación: {empleado_ejemplo[2]} años")
    print(f"- Horas trabajo: {empleado_ejemplo[3]} horas/semana")
    print(f"- Proyectos completados: {empleado_ejemplo[4]}")
    print(f"- Certificaciones: {empleado_ejemplo[5]}")
    print(f"\nSalario predicho: ${salario:,.2f}")
    
    # Ejemplos de predicciones múltiples
    ejemplo_predicciones_multiples(modelo, scaler)
    
    # Comparación con datos reales
    comparar_con_datos_reales(modelo, scaler)

if __name__ == "__main__":
    main() 