import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from generate_synthetic_data import generate_synthetic_salary_data

class PredictorSalarios:
    def __init__(self):
        self.modelo = None
        self.scaler = None
        self.entrenado = False
    
    def entrenar_modelo(self):
        """
        Entrena el modelo de regresión lineal con datos sintéticos
        """
        print("🔄 Entrenando modelo de predicción de salarios...")
        
        # Generar datos sintéticos
        data = generate_synthetic_salary_data(1000)
        
        # Separar features y target
        X = data.drop('salario', axis=1)
        y = data['salario']
        
        # Dividir en train y test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Normalización
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar modelo
        self.modelo = LinearRegression()
        self.modelo.fit(X_train_scaled, y_train)
        
        # Evaluar modelo
        y_pred = self.modelo.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.entrenado = True
        
        print("✅ Modelo entrenado exitosamente!")
        print(f"📊 Métricas del modelo:")
        print(f"   • R² Score: {r2:.4f}")
        print(f"   • RMSE: ${np.sqrt(mse):,.2f}")
        
        return True
    
    def predecir_salario(self, datos_empleado):
        """
        Predice el salario basado en los datos del empleado
        
        Args:
            datos_empleado: Lista con [edad, experiencia, educacion, horas, proyectos, certificaciones]
        """
        if not self.entrenado:
            print("❌ El modelo no está entrenado. Ejecutando entrenamiento...")
            self.entrenar_modelo()
        
        # Convertir a array y normalizar
        datos_array = np.array([datos_empleado])
        datos_scaled = self.scaler.transform(datos_array)
        
        # Hacer predicción
        salario_predicho = self.modelo.predict(datos_scaled)[0]
        
        return salario_predicho

def obtener_datos_empleado():
    """
    Solicita al usuario los datos del empleado
    """
    print("\n" + "="*50)
    print("📊 INGRESA LOS DATOS DEL EMPLEADO")
    print("="*50)
    
    try:
        print("\nPor favor, ingresa los datos:")
        
        edad = float(input("• Edad (22-65 años): "))
        experiencia = float(input("• Años de experiencia laboral (0-35): "))
        educacion = float(input("• Años de educación formal (12-22): "))
        horas = float(input("• Horas trabajadas por semana (30-60): "))
        proyectos = int(input("• Número de proyectos completados: "))
        certificaciones = int(input("• Número de certificaciones obtenidas: "))
        
        return [edad, experiencia, educacion, horas, proyectos, certificaciones]
        
    except ValueError:
        print("❌ Error: Por favor ingresa valores numéricos válidos")
        return None

def mostrar_resultado(datos, salario_predicho):
    """
    Muestra el resultado de la predicción
    """
    print("\n" + "="*60)
    print("🎯 RESULTADO DE LA PREDICCIÓN")
    print("="*60)
    
    print(f"\n📋 Datos del empleado:")
    print(f"   • Edad: {datos[0]} años")
    print(f"   • Experiencia: {datos[1]} años")
    print(f"   • Educación: {datos[2]} años")
    print(f"   • Horas trabajo: {datos[3]} horas/semana")
    print(f"   • Proyectos completados: {datos[4]}")
    print(f"   • Certificaciones: {datos[5]}")
    
    print(f"\n💰 SALARIO PREDICHO:")
    print(f"   ${salario_predicho:,.2f} USD anuales")
    print(f"   ${salario_predicho/12:,.2f} USD mensuales")
    
    # Categorizar
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
    print("="*60)

def mostrar_ejemplos(predictor):
    """
    Muestra ejemplos de predicciones
    """
    print("\n" + "="*60)
    print("📊 EJEMPLOS DE PREDICCIONES")
    print("="*60)
    
    ejemplos = [
        {"nombre": "Desarrollador Junior", "datos": [24, 1, 14, 35, 3, 1]},
        {"nombre": "Desarrollador Intermedio", "datos": [28, 4, 16, 40, 8, 2]},
        {"nombre": "Desarrollador Senior", "datos": [32, 7, 18, 45, 15, 3]},
        {"nombre": "Arquitecto de Software", "datos": [38, 12, 20, 50, 25, 5]},
        {"nombre": "Tech Lead", "datos": [42, 15, 22, 55, 35, 7]}
    ]
    
    for ejemplo in ejemplos:
        salario = predictor.predecir_salario(ejemplo["datos"])
        print(f"\n{ejemplo['nombre']}: ${salario:,.2f} USD")

def main():
    """
    Función principal
    """
    print("🚀 PREDICTOR DE SALARIOS - VERSIÓN SIMPLE")
    print("="*60)
    
    # Crear predictor
    predictor = PredictorSalarios()
    
    # Entrenar modelo automáticamente
    print("\n🔄 Inicializando modelo...")
    predictor.entrenar_modelo()
    
    while True:
        print("\n" + "="*60)
        print("🤖 MENÚ PRINCIPAL")
        print("="*60)
        print("1. 🔮 Hacer predicción de salario")
        print("2. 📊 Ver ejemplos")
        print("3. 🔄 Reentrenar modelo")
        print("4. ❌ Salir")
        
        opcion = input("\nSelecciona una opción (1-4): ")
        
        if opcion == "1":
            datos = obtener_datos_empleado()
            if datos:
                salario = predictor.predecir_salario(datos)
                mostrar_resultado(datos, salario)
                
                continuar = input("\n¿Hacer otra predicción? (s/n): ").lower()
                if continuar != 's':
                    break
                    
        elif opcion == "2":
            mostrar_ejemplos(predictor)
            input("\nPresiona Enter para continuar...")
            
        elif opcion == "3":
            print("\n🔄 Reentrenando modelo...")
            predictor.entrenar_modelo()
            input("Presiona Enter para continuar...")
            
        elif opcion == "4":
            print("\n👋 ¡Gracias por usar el Predictor de Salarios!")
            break
            
        else:
            print("❌ Opción no válida. Por favor selecciona 1-4.")

if __name__ == "__main__":
    main() 