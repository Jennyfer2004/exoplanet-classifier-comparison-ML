from preprocesamiento import *

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,classification_report

import lightgbm as lgb


def baseline_decisiontreeclassifier(X_train,y_train,X_test,y_test): 
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
    
    return best_dt_model_random

def randomforestclassifier(X_train,y_train,X_test,y_test):

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_grid = {
        'n_estimators': [100, 200, 300],          # Número de árboles en el bosque
        'max_depth': [10, 15, 20, None],          # Máxima profundidad de los árboles
        'min_samples_leaf': [1, 3, 5],            # Mínimo de muestras en un nodo hoja
        'min_samples_split': [2, 5, 10],          # Mínimo de muestras para dividir un nodo
        'max_features': ['sqrt', 'log2', 0.7]     # Número de características a considerar en cada división
    }

    # cv=5 validación cruzada de 5 pliegues.
    # scoring='accuracy' le dice que la métrica a optimizar es la precisión.
    # verbose=2 muestra más detalle del progreso.
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
    
    return best_rf_model

def logisticregression(X_train,y_train,X_test,y_test):
    
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
    
    return best_lr_model

def XGBoost_LightGBM(X_train,y_train,X_test,y_test):

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
    
    return lgb_base

def evaluacion(best_dt_model_random,best_rf_model,lr_regularized,lgb_base,X_val,y_val):

    if isinstance(X_val, np.ndarray):

        X_val_df = pd.DataFrame(X_val)
    else:
        X_val_df = X_val


    all_results = []

    models_to_evaluate = [
        ("Decision Tree", best_dt_model_random),
        ("Random Forest", best_rf_model),
        ("Logistic Regression", lr_regularized),
        ("LightGBM", lgb_base)
]

    for model_name, model in models_to_evaluate:

        y_pred = model.predict(X_val_df)
        
        try:
            y_pred_proba = model.predict_proba(X_val_df)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
        except AttributeError:
            auc = np.nan # Asignar Not a Number si no se puede calcular

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='binary') # 'binary' para clasificación binaria
        recall = recall_score(y_val, y_pred, average='binary')
        f1 = f1_score(y_val, y_pred, average='binary')

        results = {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC": auc
        }
        
        all_results.append(results)

    df_metrics = pd.DataFrame(all_results)

    print((df_metrics.round(4).to_string(index=False)))

    return models_to_evaluate

def main_models():
    _,X,y=main()

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # 2. Escalado: Estandarizamos los datos (media 0, desviación 1). ¡Esencial para LogisticRegression y bueno para otros!
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 3. División de Datos: Separamos en entrenamiento y prueba.
    # Stratify=y es MUY IMPORTANTE para mantener la proporción de clases en ambos sets.
    X_train, X_test1, y_train, y_test1 = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test1, y_test1, test_size=0.5, random_state=42, stratify=y_test1
    )
    print(f"Datos de entrenamiento: {X_train.shape}")
    print(f"Datos de prueba: {X_test.shape}")
    
    best_dt_model_random = baseline_decisiontreeclassifier(X_train,y_train,X_test,y_test)
    best_rf_model = randomforestclassifier(X_train,y_train,X_test,y_test)
    lr_regularized = logisticregression(X_train,y_train,X_test,y_test)
    lgb_base = XGBoost_LightGBM(X_train,y_train,X_test,y_test)
    return  X_val,y_val,X_train, y_train,evaluacion(best_dt_model_random,best_rf_model,lr_regularized,lgb_base,X_val,y_val)
    