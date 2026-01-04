import pandas as pd
import numpy as np
import os
import joblib
import os
import json

from datetime import datetime

from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

import lightgbm as lgb


def train_decision_tree(X_train,y_train,X_test,y_test): 
    dt = DecisionTreeClassifier(random_state=42)

    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_leaf': [5, 10, 20, 50],
        'min_samples_split': [2, 10, 20],
        'criterion': ['gini', 'entropy'],
        'max_features': [None, 'sqrt', 'log2']
    }

    #  probará 100 combinaciones aleatorias
    random_search = RandomizedSearchCV(estimator=dt, param_distributions=param_grid, 
                                    n_iter=100, cv=5, n_jobs=-1, verbose=1, 
                                    scoring='accuracy', random_state=42)

    print("Iniciando la búsqueda aleatoria de hiperparámetros...")
    random_search.fit(X_train, y_train)

    print("\n--- Resultados de la Búsqueda Aleatoria ---")
    print(f"Mejores hiperparámetros encontrados: {random_search.best_params_}")
    print(f"Mejor puntuación de validación cruzada (accuracy): {random_search.best_score_:.4f}")

    best_dt_model_random = random_search.best_estimator_
    y_pred_best_dt_random = best_dt_model_random.predict(X_test)

    print("\n--- DecisionTree (Mejor Modelo Aleatorio) ---")
    print(f"Accuracy en el conjunto de prueba: {accuracy_score(y_test, y_pred_best_dt_random):.4f}")
    print(classification_report(y_test, y_pred_best_dt_random))
    
    model_path = os.path.join('outputs', 'models', 'decision_tree.pkl')
    joblib.dump(best_dt_model_random, model_path)
    
    params_path = os.path.join('outputs', 'models', 'decision_tree_params.json')
    with open(params_path, 'w') as f:
        json.dump({
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'test_accuracy': accuracy_score(y_test, y_pred_best_dt_random),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=4)
        
    return best_dt_model_random

def train_random_forest(X_train,y_train,X_test,y_test):

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_grid = {
        'n_estimators': [100, 200, 300],          # Número de árboles en el bosque
        'max_depth': [10, 15, 20, None],          # Máxima profundidad de los árboles
        'min_samples_leaf': [1, 3, 5],            # Mínimo de muestras en un nodo hoja
        'min_samples_split': [2, 5, 10],          # Mínimo de muestras para dividir un nodo
        'max_features': ['sqrt', 'log2', 0.7]     # Número de características a considerar en cada división
    }

    # cv=5 validación cruzada de 5 pliegues.
    # scoring='accuracy' la métrica a optimizar es la precisión.
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                            cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

    print("Iniciando la búsqueda de los mejores hiperparámetros para RandomForest... (Esto puede tardar varios minutos)")
    grid_search.fit(X_train, y_train)

    print("\n--- Resultados de la Búsqueda de Hiperparámetros ---")
    print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")
    print(f"Mejor puntuación de validación cruzada (accuracy): {grid_search.best_score_:.4f}")

    best_rf_model = grid_search.best_estimator_

    y_pred_best_rf = best_rf_model.predict(X_test)

    print("\n--- RandomForest (Mejor Modelo Encontrado) ---")
    print(f"Accuracy en el conjunto de prueba: {accuracy_score(y_test, y_pred_best_rf):.4f}")
    print(classification_report(y_test, y_pred_best_rf))
    
    model_path = os.path.join('outputs', 'models', 'random_forest.pkl')
    
    joblib.dump(best_rf_model, model_path)
    
    params_path = os.path.join('outputs', 'models', 'random_forest_params.json')
    with open(params_path, 'w') as f:
        json.dump({
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'test_accuracy': accuracy_score(y_test, y_pred_best_rf),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=4)
        
    return best_rf_model

def train_logistic_regression(X_train,y_train,X_test,y_test):
    
    lr = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverso de la fuerza de regularización. Valores pequeños = más regularización.
        'penalty': ['l1', 'l2']                # Tipo de regularización. L1 (Lasso) puede llevar a coeficientes cero. L2 (Ridge) los hace pequeños.
    }

    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, 
                            cv=5, n_jobs=-1, verbose=1, scoring='accuracy')

    print("Iniciando la búsqueda de los mejores hiperparámetros para LogisticRegression...")
    grid_search.fit(X_train, y_train)

    print("\n--- Resultados de la Búsqueda de Hiperparámetros ---")
    print(f"Mejores hiperparámetros encontrados: {grid_search.best_params_}")
    print(f"Mejor puntuación de validación cruzada (accuracy): {grid_search.best_score_:.4f}")

    best_lr_model = grid_search.best_estimator_

    y_pred_best_lr = best_lr_model.predict(X_test)

    print("\n--- LogisticRegression (Mejor Modelo Encontrado) ---")
    print(f"Accuracy en el conjunto de prueba: {accuracy_score(y_test, y_pred_best_lr):.4f}")
    print(classification_report(y_test, y_pred_best_lr))
    
    model_path = os.path.join('outputs', 'models', 'logistic_regression.pkl')
    
    joblib.dump(best_lr_model, model_path)
    
    params_path = os.path.join('outputs', 'models', 'logistic_regression_params.json')
    with open(params_path, 'w') as f:
        json.dump({
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'test_accuracy': accuracy_score(y_test, y_pred_best_lr),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=4)
        
    return best_lr_model

def train_lightgbm(X_train,y_train,X_test,y_test):

    lgb_base = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    lgb_base.fit(X_train, y_train)

    y_pred_lgb_base = lgb_base.predict(X_test)
    print("\n--- LightGBM (Base) ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lgb_base):.4f}")
    print(classification_report(y_test, y_pred_lgb_base))

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1')

    grid_search.fit(X_train, y_train)

    print("\n--- Mejores Hiperparámetros Encontrados por GridSearchCV ---")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score F1: {grid_search.best_score_:.4f}")

    best_rf = grid_search.best_estimator_
    y_pred_best_rf = best_rf.predict(X_test)

    print("\n--- RandomForest (Mejor Modelo de GridSearch) ---")
    print(f"Accuracy en test: {accuracy_score(y_test, y_pred_best_rf):.4f}")
    print(classification_report(y_test, y_pred_best_rf))
    
    model_path = os.path.join('outputs', 'models', 'lightgbm.pkl')
    
    joblib.dump(best_rf, model_path)
    
    params_path = os.path.join('outputs', 'models', 'lightgbm_params.json')
    with open(params_path, 'w') as f:
        json.dump({
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'test_accuracy': accuracy_score(y_test, y_pred_best_rf),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=4)
        
    return lgb_base