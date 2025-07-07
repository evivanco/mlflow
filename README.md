# Ejemplo MLflow con Regresión Lineal - Predicción de Salarios

Este proyecto demuestra un flujo de trabajo completo de MLOps usando MLflow para entrenar un modelo de regresión lineal que predice salarios basado en características del empleado.

## 📁 Estructura del Proyecto

```
├── generate_synthetic_data.py    # Generador de datos sintéticos
├── mlflow_regression_example.py  # Script principal con MLflow
├── prediccion_interactiva.py     # Interfaz interactiva con MLflow
├── prediccion_simple.py          # Versión simple sin MLflow
├── ejemplo_uso_modelo.py         # Ejemplo de uso del modelo
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Este archivo
```

## 🚀 Instalación y Configuración

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Verificar instalación de MLflow
```bash
mlflow --version
```

## 📊 Datos Sintéticos

El proyecto incluye un generador de datos sintéticos que crea un dataset realista para predicción de salarios con las siguientes variables:

**Variables independientes:**
- `edad`: Edad del empleado (22-65 años)
- `experiencia_anos`: Años de experiencia laboral (0-35 años)
- `educacion_anos`: Años de educación formal (12-22 años)
- `horas_trabajo`: Horas trabajadas por semana (30-60 horas)
- `proyectos_completados`: Número de proyectos completados
- `certificaciones`: Número de certificaciones obtenidas

**Variable dependiente:**
- `salario`: Salario anual en dólares ($25,000 - $150,000)

## 🔧 Uso del Proyecto

### 🎯 Opción 1: Predicción Simple (Recomendada para principiantes)

La forma más fácil de usar el predictor sin configurar MLflow:

```bash
python prediccion_simple.py
```

**Características:**
- ✅ No requiere configuración previa
- ✅ Entrena el modelo automáticamente
- ✅ Interfaz interactiva amigable
- ✅ Validación de datos en tiempo real
- ✅ Ejemplos predefinidos incluidos

### 🎯 Opción 2: Predicción con MLflow (Para MLOps)

Flujo completo con tracking de experimentos:

#### Paso 1: Generar datos sintéticos (opcional)
```bash
python generate_synthetic_data.py
```

#### Paso 2: Ejecutar experimento MLflow
```bash
python mlflow_regression_example.py
```

Este script:
- Genera datos sintéticos
- Preprocesa los datos (normalización)
- Entrena un modelo de regresión lineal
- Registra hiperparámetros y métricas en MLflow
- Guarda el modelo y scaler como artefactos
- Crea visualizaciones de resultados

#### Paso 3: Ver resultados en MLflow UI
```bash
mlflow ui
```

Abre tu navegador en `http://localhost:5000` para ver:
- Experimentos y runs
- Métricas registradas
- Artefactos guardados
- Gráficos de resultados

#### Paso 4: Usar el modelo entrenado
```bash
python prediccion_interactiva.py
```

**Nota:** Necesitas el Run ID que aparece al final del experimento.

### 🎯 Opción 3: Ejemplo programático
```bash
python ejemplo_uso_modelo.py
```

## 🎮 Cómo Hacer Predicciones

### Usando la Versión Simple (Más Fácil)

1. **Ejecuta el script:**
   ```bash
   python prediccion_simple.py
   ```

2. **El modelo se entrena automáticamente**

3. **Selecciona "Hacer predicción de salario"**

4. **Ingresa los datos del empleado:**
   - Edad (22-65 años)
   - Años de experiencia (0-35)
   - Años de educación (12-22)
   - Horas de trabajo por semana (30-60)
   - Número de proyectos completados
   - Número de certificaciones

5. **¡Obtén la predicción!**

### Usando la Versión MLflow

1. **Ejecuta el entrenamiento:**
   ```bash
   python mlflow_regression_example.py
   ```

2. **Copia el Run ID** que aparece al final

3. **Ejecuta la predicción:**
   ```bash
   python prediccion_interactiva.py
   ```

4. **Pega el Run ID** cuando te lo solicite

5. **¡Haz predicciones!**

## 📈 Métricas Registradas

El experimento registra las siguientes métricas:
- **MSE**: Error cuadrático medio
- **RMSE**: Raíz del error cuadrático medio
- **MAE**: Error absoluto medio
- **R²**: Coeficiente de determinación

## 🎯 Características del Flujo de Trabajo

### ✅ Funcionalidades Implementadas

1. **Generación de datos sintéticos realistas**
   - 1000 registros con correlaciones realistas
   - Variables numéricas con distribución normal
   - Ruido controlado para simular datos reales

2. **Preprocesamiento completo**
   - Verificación de valores nulos
   - Normalización con StandardScaler
   - División train/test (80/20)

3. **Entrenamiento de modelo**
   - Regresión lineal con scikit-learn
   - Hiperparámetros configurados
   - Validación cruzada implícita

4. **Tracking con MLflow**
   - Registro de hiperparámetros
   - Logging de métricas de evaluación
   - Guardado de modelo y scaler
   - Artefactos adicionales (gráficos)

5. **Visualización de resultados**
   - Gráfico de predicciones vs valores reales
   - Estadísticas descriptivas
   - Ejemplos de predicción

6. **Interfaces de usuario**
   - Versión simple sin MLflow
   - Versión interactiva con MLflow
   - Validación de datos en tiempo real
   - Categorización automática de salarios

### 🔄 Flujo de Trabajo MLOps

```
Datos Sintéticos → Preprocesamiento → Entrenamiento → Evaluación → Registro MLflow → Despliegue
```

## 📋 Ejemplo de Salida

### Versión Simple
```
🚀 PREDICTOR DE SALARIOS - VERSIÓN SIMPLE
============================================================

🔄 Inicializando modelo...
🔄 Entrenando modelo de predicción de salarios...
✅ Modelo entrenado exitosamente!
📊 Métricas del modelo:
   • R² Score: 0.8234
   • RMSE: $3,513.64

============================================================
🤖 MENÚ PRINCIPAL
============================================================
1. 🔮 Hacer predicción de salario
2. 📊 Ver ejemplos
3. 🔄 Reentrenar modelo
4. ❌ Salir

Selecciona una opción (1-4): 1

==================================================
📊 INGRESA LOS DATOS DEL EMPLEADO
==================================================

Por favor, ingresa los datos:

• Edad (22-65 años): 28
• Años de experiencia laboral (0-35): 5
• Años de educación formal (12-22): 16
• Horas trabajadas por semana (30-60): 40
• Número de proyectos completados: 10
• Número de certificaciones obtenidas: 2

============================================================
🎯 RESULTADO DE LA PREDICCIÓN
============================================================

📋 Datos del empleado:
   • Edad: 28.0 años
   • Experiencia: 5.0 años
   • Educación: 16.0 años
   • Horas trabajo: 40.0 horas/semana
   • Proyectos completados: 10
   • Certificaciones: 2

💰 SALARIO PREDICHO:
   $65,432.10 USD anuales
   $5,452.68 USD mensuales

🏷️  Categoría estimada: Intermedio
============================================================
```

### Versión MLflow
```
=== INICIANDO EXPERIMENTO MLFLOW ===
Cargando y preprocesando datos...
Valores nulos en features: 0
Valores nulos en target: 0
Forma de datos de entrenamiento: (800, 6)
Forma de datos de prueba: (200, 6)
Entrenando modelo de regresión lineal...
Modelo entrenado exitosamente
Evaluando modelo...
MSE: 12345678.90
RMSE: 3513.64
MAE: 2847.23
R²: 0.8234
Registrando hiperparámetros...
Registrando métricas...
Guardando modelo...
=== EXPERIMENTO COMPLETADO ===
Run ID: abc123def456
Experimento: 1
```

## 🛠️ Personalización

### Modificar hiperparámetros
Edita la función `entrenar_modelo_regresion_lineal()` en `mlflow_regression_example.py`:

```python
hiperparametros = {
    'fit_intercept': True,
    'copy_X': True,
    'n_jobs': -1,
    'positive': False
}
```

### Agregar nuevas métricas
En la función `evaluar_modelo()`, puedes agregar más métricas:

```python
from sklearn.metrics import mean_absolute_percentage_error
mape = mean_absolute_percentage_error(y_test, y_pred)
metricas['mape'] = mape
```

### Cambiar el dataset
Modifica `generate_synthetic_data.py` para crear datos con diferentes características o usa tu propio dataset.

## 🔍 Troubleshooting

### Error: "No module named 'mlflow'"
```bash
pip install mlflow
```

### Error: "No se pudo cargar el modelo"
- Verifica que el run_id sea correcto
- Asegúrate de haber ejecutado el experimento primero
- Confirma que MLflow esté configurado correctamente

### Error: "Permission denied" al guardar archivos
- Verifica permisos de escritura en el directorio
- Usa un directorio temporal si es necesario

### Error: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

## 🎯 Recomendaciones de Uso

### Para Principiantes
- Usa `prediccion_simple.py` - No requiere configuración
- Perfecto para aprender y experimentar
- Interfaz amigable y validación automática

### Para MLOps y Producción
- Usa `mlflow_regression_example.py` + `prediccion_interactiva.py`
- Tracking completo de experimentos
- Versionado de modelos
- Reproducibilidad garantizada

### Para Desarrollo
- Modifica `generate_synthetic_data.py` para tus datos
- Ajusta hiperparámetros según tus necesidades
- Agrega nuevas métricas de evaluación

## 📚 Recursos Adicionales

- [Documentación oficial de MLflow](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [MLOps Best Practices](https://mlflow.org/docs/latest/tracking.html)

## 🤝 Contribuciones

Este es un ejemplo. Siéntete libre de:
- Reportar issues o sugerencias 
