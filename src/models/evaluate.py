import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import os

def evaluate_all_models(X_val, y_val, model_names, output_dir='outputs/reports'):

    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    for model_name in model_names:
        # Cargar modelo
        model_path = os.path.join('outputs', 'models', f'{model_name.lower().replace(" ", "_")}.pkl')
        model = joblib.load(model_path)
        
        y_pred = model.predict(X_val)
        
        try:
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
        except AttributeError:
            auc = np.nan
        
        # Calcular métricas
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='binary')
        recall = recall_score(y_val, y_pred, average='binary')
        f1 = f1_score(y_val, y_pred, average='binary')
        
        # Guardar resultados
        results = {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "AUC": auc
        }
        
        all_results.append(results)
        
        # Guardar reporte de clasificación
        report = classification_report(y_val, y_pred)
        report_path = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Classification Report for {model_name}\n\n")
            f.write(report)
        
        # Generar y guardar matriz de confusión
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        fig_path = os.path.join('outputs', 'figures', f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Crear DataFrame con todas las métricas
    df_metrics = pd.DataFrame(all_results)
    
    # Guardar tabla de métricas
    metrics_path = os.path.join(output_dir, 'all_models_metrics.csv')
    df_metrics.to_csv(metrics_path, index=False)
    
    # Generar gráfico comparativo
    plt.figure(figsize=(12, 8))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    for metric in metrics:
        plt.subplot(2, 3, metrics.index(metric) + 1)
        sns.barplot(x='Model', y=metric, data=df_metrics)
        plt.title(f'{metric} Comparison')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    comparison_path = os.path.join('outputs', 'figures', 'models_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n--- Tabla Comparativa de Métricas ---")
    print(df_metrics.round(4).to_string(index=False))
    
    return df_metrics