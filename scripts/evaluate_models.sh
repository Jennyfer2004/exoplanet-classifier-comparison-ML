#!/bin/bash

echo "Iniciando evaluación de modelos..."

# Asegurarse de que el entorno virtual esté activado

# Crear directorios si no existen
mkdir -p outputs/figures
mkdir -p outputs/reports

# Evaluar todos los modelos
python -c "
from src.models.evaluate import evaluate_all_models
from src.data.preprocessing import preprocess_data

# Cargar y preprocesar datos
X_train, X_val, X_test, y_train, y_val, y_test, imputer, scaler = preprocess_data('data/raw/PS_2025.11.28_09.38.12.csv')

# Nombres de los modelos a evaluar
model_names = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'LightGBM']

# Evaluar modelos
metrics_df = evaluate_all_models(X_val, y_val, model_names)

print('Evaluación de todos los modelos completada')
"

echo "Evaluación completada. Resultados guardados en outputs/reports/ y outputs/figures/"