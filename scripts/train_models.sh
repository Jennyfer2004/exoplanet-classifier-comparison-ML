#!/bin/bash

echo "Iniciando entrenamiento de modelos..."

# Asegurarse de que el entorno virtual esté activado

# Crear directorios si no existen
mkdir -p outputs/models
mkdir -p outputs/reports

# Entrenar todos los modelos
python -c "
from src.models.train import train_decision_tree, train_random_forest, train_logistic_regression, train_lightgbm
from src.data.preprocessing import preprocess_data
from src.utils import set_seed

# Fijar semilla para reproducibilidad
set_seed(42)

# Cargar y preprocesar datos
X_train, X_val, X_test, y_train, y_val, y_test, imputer, scaler = preprocess_data('data/raw/exoplanets.csv')

# Entrenar modelos
dt_model = train_decision_tree(X_train, y_train, X_test, y_test)
rf_model = train_random_forest(X_train, y_train, X_test, y_test)
lr_model = train_logistic_regression(X_train, y_train, X_test, y_test)
lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)

print('Entrenamiento de todos los modelos completado')
"

echo "Entrenamiento completado. Modelos guardados en outputs/models/"