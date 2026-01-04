import os
import argparse
import time
from datetime import datetime

from src.data.preprocessing import preprocess_data
from src.models.train import train_decision_tree, train_random_forest, train_logistic_regression, train_lightgbm
from src.models.evaluate import evaluate_all_models

def main(args):
    """Función principal que ejecuta todo el pipeline"""
    start_time = time.time()
    
    print("EXO-ML-ARENA: Comparación de Modelos para Clasificación de Exoplanetas")

    print(f"Fecha y hora de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n Preprocesando datos...")

    X_train, X_val, X_test, y_train, y_val, y_test, imputer, scaler = preprocess_data(
    args.data_path
    )
    args.seed=42
    print(1111)
    print("\n Entrenando modelos...")
    # if args.train:
    dt_model = train_decision_tree(X_train, y_train, X_test, y_test)
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)#, random_state=args.seed)
    lr_model = train_logistic_regression(X_train, y_train, X_test, y_test)#, random_state=args.seed)
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)#, random_state=args.seed)
    print("Entrenamiento de todos los modelos completado")
    # else:
    #     print("Entrenamiento omitido (--no-train)")
    
    print("\n Evaluando modelos...")
    model_names = ['Decision Tree', 'Random Forest', 'Logistic Regression', 'LightGBM']
    metrics_df = evaluate_all_models(X_val, y_val, model_names)
    
    print("\n Generando reporte final...")
    report_path = os.path.join('outputs', 'reports', 'final_report.txt')
    with open(report_path, 'w') as f:
        f.write("EXO-ML-ARENA: Reporte Final\n")
        f.write("\n")
        f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Semilla utilizada: {args.seed}\n")
        f.write(f"Ruta de datos: {args.data_path}\n")
        f.write("\n\n")
        f.write("Métricas de todos los modelos:\n\n")
        f.write(metrics_df.round(4).to_string(index=False))
    
    # Calcular tiempo de ejecución
    end_time = time.time()
    execution_time = end_time - start_time
    
    print("PROCESO COMPLETADO")
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos ({execution_time/60:.2f} minutos)")
    print(f"Reporte guardado en: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline completo para comparación de modelos de clasificación de exoplanetas')
    print(1)
    parser.add_argument('--data_path', type=str, default='data/raw/PS_2025.11.28_09.38.12.csv',
                        help='Ruta al archivo de datos')
    print(2)
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proporción de datos para prueba')
    parser.add_argument('--val_size', type=float, default=0.5,
                        help='Proporción de datos no entrenamiento para validación')
    parser.add_argument('--no-train', action='store_true',
                        help='Omitir entrenamiento de modelos (usar modelos existentes)')
    
    args = parser.parse_args()
    
    main(args)