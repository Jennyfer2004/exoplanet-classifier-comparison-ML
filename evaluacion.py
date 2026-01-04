from models import *

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score,confusion_matrix
from sklearn.model_selection import learning_curve

def main():
    X_val,y_val ,X_train, y_train,models_to_evaluate=main_models()
    
    for i in range(len(models_to_evaluate)):
        best_model = models_to_evaluate[i][1] 
        y_pred = best_model.predict(X_val)

        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Candidato', 'Confirmado'], yticklabels=['Candidato', 'Confirmado'])
        plt.ylabel('Verdadero')
        plt.xlabel('Predicho')
        plt.title(f'Matriz de Confusión para {best_model.__class__.__name__}')
        plt.show()
        
        y_proba = best_model.predict_proba(X_val)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_val, y_proba)
        auc_score = roc_auc_score(y_val, y_proba)

        plt.plot(fpr, tpr, label=f'{best_model.__class__.__name__} (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio (AUC = 0.50)')
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos (Recall)')
        plt.title('Curva ROC')
        plt.legend()
        plt.show()
        
        try:
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(12, 6))
            plt.title("Importancia de Características - Random Forest")
            plt.bar(range(len(indices)), importances[indices], align="center")
            plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=90)
            plt.xlim([-1, len(indices)])
            plt.tight_layout()
            plt.show()
        except:
            pass

        train_sizes, train_scores, val_scores = learning_curve(
            best_model, X_train, y_train, cv=5, scoring='f1', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Puntuación de Entrenamiento")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.plot(train_sizes, val_mean, 'o-', color="g", label="Puntuación de Validación Cruzada")
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="g")

        plt.title("Curva de Aprendizaje")
        plt.xlabel("Tamaño del conjunto de entrenamiento")
        plt.ylabel("Puntuación F1")
        plt.legend(loc="best")
        plt.show()