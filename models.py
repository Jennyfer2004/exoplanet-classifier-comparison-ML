from preprocesamiento import *

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def Baseline_DecisionTreeClassifier(): 
    pass

def main():
    _,X,y=main(preprocesing())

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