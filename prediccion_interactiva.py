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
        print(f"✅ Modelo cargado exitosamente desde run_id: {run_id}")
        return modelo, scaler
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return None, None

def predecir_salario_empleado(modelo, scaler, datos_empleado):
    """
    Predice el salario de un empleado usando el modelo entrenado
    """
    # Convertir a array numpy
    datos_array = np.array([datos_empleado])
    
    # Normalizar datos
    datos_scaled = scaler.transform(datos_array)
    
    # Hacer predicción
    salario_predicho = modelo.predict(datos_scaled)[0]
    
    return salario_predicho

def obtener_datos_empleado():
    """
    Solicita al usuario los datos del empleado de forma interactiva
    """
    print("\n" + "="*50)
    print("📊 PREDICCIÓN DE SALARIO - DATOS DEL EMPLEADO")
    print("="*50)
    
    try:
        # Solicitar datos con validación
        print("\nPor favor, ingresa los datos del empleado:")
        
        edad = float(input("• Edad (22-65 años): "))
        if not (22 <= edad <= 65):
            print("⚠️  Advertencia: La edad debe estar entre 22 y 65 años")
        
        experiencia = float(input("• Años de experiencia laboral (0-35): "))
        if not (0 <= experiencia <= 35):
            print("⚠️  Advertencia: La experiencia debe estar entre 0 y 35 años")
        
        educacion = float(input("• Años de educación formal (12-22): "))
        if not (12 <= educacion <= 22):
            print("⚠️  Advertencia: La educación debe estar entre 12 y 22 años")
        
        horas = float(input("• Horas trabajadas por semana (30-60): "))
        if not (30 <= horas <= 60):
            print("⚠️  Advertencia: Las horas deben estar entre 30 y 60")
        
        proyectos = int(input("• Número de proyectos completados: "))
        if proyectos < 0:
            print("⚠️  Advertencia: El número de proyectos debe ser positivo")
        
        certificaciones = int(input("• Número de certificaciones obtenidas: "))
        if certificaciones < 0:
            print("⚠️  Advertencia: El número de certificaciones debe ser positivo")
        
        return [edad, experiencia, educacion, horas, proyectos, certificaciones]
        
    except ValueError:
        print("❌ Error: Por favor ingresa valores numéricos válidos")
        return None

def mostrar_resultado_prediccion(datos_empleado, salario_predicho):
    """
    Muestra el resultado de la predicción de forma atractiva
    """
    print("\n" + "="*60)
    print("🎯 RESULTADO DE LA PREDICCIÓN")
    print("="*60)
    
    print(f"\n📋 Datos del empleado:")
    print(f"   • Edad: {datos_empleado[0]} años")
    print(f"   • Experiencia: {datos_empleado[1]} años")
    print(f"   • Educación: {datos_empleado[2]} años")
    print(f"   • Horas trabajo: {datos_empleado[3]} horas/semana")
    print(f"   • Proyectos completados: {datos_empleado[4]}")
    print(f"   • Certificaciones: {datos_empleado[5]}")
    
    print(f"\n💰 SALARIO PREDICHO:")
    print(f"   ${salario_predicho:,.2f} USD anuales")
    print(f"   ${salario_predicho/12:,.2f} USD mensuales")
    
    # Categorizar el salario
    if salario_predicho < 40000:
        categoria = "Junior"
    elif salario_predicho < 60000:
        categoria = "Intermedio"
    elif salario_predicho < 80000:
        categoria = "Senior"
    elif salario_predicho < 100000:
        categoria = "Experto"
    else:
        categoria = "Muy Experto"
    
    print(f"\n🏷️  Categoría estimada: {categoria}")
    
    print("\n" + "="*60)

def menu_principal():
    """
    Menú principal de la aplicación
    """
    print("\n" + "="*60)
    print("🤖 PREDICTOR DE SALARIOS - MLflow + Regresión Lineal")
    print("="*60)
    print("\nOpciones disponibles:")
    print("1. 🔮 Hacer predicción de salario")
    print("2. 📊 Ver ejemplos predefinidos")
    print("3. 🔄 Entrenar nuevo modelo")
    print("4. ❌ Salir")
    
    return input("\nSelecciona una opción (1-4): ")

def mostrar_ejemplos_predefinidos(modelo, scaler):
    """
    Muestra ejemplos de predicciones para diferentes perfiles
    """
    print("\n" + "="*60)
    print("📊 EJEMPLOS PREDEFINIDOS")
    print("="*60)
    
    ejemplos = [
        {
            "nombre": "Desarrollador Junior",
            "datos": [24, 1, 14, 35, 3, 1],
            "descripcion": "Recién graduado, poca experiencia"
        },
        {
            "nombre": "Desarrollador Intermedio",
            "datos": [28, 4, 16, 40, 8, 2],
            "descripcion": "Algunos años de experiencia"
        },
        {
            "nombre": "Desarrollador Senior",
            "datos": [32, 7, 18, 45, 15, 3],
            "descripcion": "Experiencia sólida"
        },
        {
            "nombre": "Arquitecto de Software",
            "datos": [38, 12, 20, 50, 25, 5],
            "descripcion": "Mucha experiencia y responsabilidades"
        },
        {
            "nombre": "Tech Lead",
            "datos": [42, 15, 22, 55, 35, 7],
            "descripcion": "Liderazgo técnico"
        }
    ]
    
    for i, ejemplo in enumerate(ejemplos, 1):
        salario = predecir_salario_empleado(modelo, scaler, ejemplo["datos"])
        
        print(f"\n{i}. {ejemplo['nombre']}")
        print(f"   Descripción: {ejemplo['descripcion']}")
        print(f"   Datos: Edad={ejemplo['datos'][0]}, Exp={ejemplo['datos'][1]}, Edu={ejemplo['datos'][2]}")
        print(f"   Salario predicho: ${salario:,.2f} USD")

def entrenar_nuevo_modelo():
    """
    Opción para entrenar un nuevo modelo
    """
    print("\n" + "="*60)
    print("🔄 ENTRENAR NUEVO MODELO")
    print("="*60)
    
    print("\nPara entrenar un nuevo modelo, ejecuta:")
    print("python mlflow_regression_example.py")
    print("\nLuego copia el Run ID que aparece al final y úsalo en esta aplicación.")

def main():
    """
    Función principal de la aplicación interactiva
    """
    print("🚀 Iniciando Predictor de Salarios...")
    
    # Solicitar Run ID
    print("\n" + "="*60)
    print("📋 CONFIGURACIÓN INICIAL")
    print("="*60)
    
    print("\nPara usar esta aplicación, necesitas el Run ID de un modelo entrenado.")
    print("Si aún no tienes uno, ejecuta primero: python mlflow_regression_example.py")
    
    run_id = input("\nIngresa el Run ID del modelo (o presiona Enter para usar ejemplo): ").strip()
    
    if not run_id:
        print("⚠️  No se proporcionó Run ID. Por favor ejecuta primero el entrenamiento.")
        print("Ejecuta: python mlflow_regression_example.py")
        return
    
    # Cargar modelo
    modelo, scaler = cargar_modelo_entrenado(run_id)
    
    if modelo is None or scaler is None:
        print("\n❌ No se pudo cargar el modelo. Verifica:")
        print("1. Que el Run ID sea correcto")
        print("2. Que hayas ejecutado el entrenamiento primero")
        print("3. Que MLflow esté configurado correctamente")
        return
    
    # Bucle principal
    while True:
        opcion = menu_principal()
        
        if opcion == "1":
            # Hacer predicción
            datos = obtener_datos_empleado()
            if datos:
                salario = predecir_salario_empleado(modelo, scaler, datos)
                mostrar_resultado_prediccion(datos, salario)
                
                # Preguntar si quiere hacer otra predicción
                continuar = input("\n¿Hacer otra predicción? (s/n): ").lower()
                if continuar != 's':
                    break
                    
        elif opcion == "2":
            # Mostrar ejemplos
            mostrar_ejemplos_predefinidos(modelo, scaler)
            input("\nPresiona Enter para continuar...")
            
        elif opcion == "3":
            # Entrenar nuevo modelo
            entrenar_nuevo_modelo()
            input("\nPresiona Enter para continuar...")
            
        elif opcion == "4":
            # Salir
            print("\n👋 ¡Gracias por usar el Predictor de Salarios!")
            break
            
        else:
            print("❌ Opción no válida. Por favor selecciona 1-4.")

if __name__ == "__main__":
    main() 