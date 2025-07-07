import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_synthetic_salary_data(n_samples=1000):
    """
    Genera datos sintéticos realistas para predecir salario basado en características del empleado.
    """
    np.random.seed(42)
    
    # Variables independientes
    edad = np.random.normal(35, 10, n_samples).clip(22, 65)
    experiencia_anos = np.random.normal(8, 6, n_samples).clip(0, 35)
    educacion_anos = np.random.normal(16, 2, n_samples).clip(12, 22)
    horas_trabajo = np.random.normal(40, 5, n_samples).clip(30, 60)
    proyectos_completados = np.random.poisson(15, n_samples)
    certificaciones = np.random.poisson(3, n_samples)
    
    # Crear correlaciones realistas
    # El salario base depende de la experiencia y educación
    salario_base = 30000 + (experiencia_anos * 2500) + (educacion_anos * 1500)
    
    # Bonificaciones por proyectos y certificaciones
    bonus_proyectos = proyectos_completados * 500
    bonus_certificaciones = certificaciones * 1000
    
    # Penalización por edad muy joven o muy mayor
    factor_edad = np.where(edad < 25, 0.8, np.where(edad > 55, 0.9, 1.0))
    
    # Salario final con ruido
    salario = (salario_base + bonus_proyectos + bonus_certificaciones) * factor_edad
    ruido = np.random.normal(0, 5000, n_samples)
    salario = salario + ruido
    salario = salario.clip(25000, 150000)
    
    # Crear DataFrame
    data = pd.DataFrame({
        'edad': edad.round(1),
        'experiencia_anos': experiencia_anos.round(1),
        'educacion_anos': educacion_anos.round(1),
        'horas_trabajo': horas_trabajo.round(1),
        'proyectos_completados': proyectos_completados,
        'certificaciones': certificaciones,
        'salario': salario.round(2)
    })
    
    return data

def save_synthetic_data():
    """
    Genera y guarda los datos sintéticos en CSV
    """
    print("Generando datos sintéticos para predicción de salarios...")
    data = generate_synthetic_salary_data(1000)
    
    # Guardar en CSV
    data.to_csv('datos_salarios_sinteticos.csv', index=False)
    
    print(f"Datos generados exitosamente:")
    print(f"- Filas: {len(data)}")
    print(f"- Columnas: {len(data.columns)}")
    print(f"- Variables independientes: {list(data.columns[:-1])}")
    print(f"- Variable dependiente: {data.columns[-1]}")
    print("\nEstadísticas básicas:")
    print(data.describe())
    
    print(f"\nDatos guardados en 'datos_salarios_sinteticos.csv'")
    
    return data

if __name__ == "__main__":
    data = save_synthetic_data() 