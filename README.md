# exoplanet-classifier-comparison-ML
Objetivo 

Este proyecto implementa y compara varios algoritmos de Machine Learning (Decision Tree, Random Forest, Logistic Regression y LightGBM) para la clasificación de candidatos a exoplanetas. El objetivo es identificar el modelo más preciso para distinguir entre exoplanetas confirmados y falsos positivos, utilizando datos observacionales del archivo de exoplanetas de la NASA. 
Datos 
Fuente 

Los datos utilizados en este proyecto fueron extraídos del NASA Exoplanet Archive el 18 de diciembre de 2025. El archivo de datos completo y la documentación asociada están disponibles públicamente en:
https://exoplanetarchive.ipac.caltech.edu/docs/exonews_archive.html  
Características (Features) 

Se seleccionaron un conjunto de 14 características astronómicas y estelares para entrenar los modelos: 
Característica
 	
Descripción
 
 pl_orbper	Período Orbital (días) 
pl_orbsmax	Semieje Mayor de la Órbita (UA) 
pl_rade	Radio del Planeta (en radios terrestres) 
pl_orbeccen	Excentricidad Orbital 
pl_insol	Flujo de Insolación (relativo a la Tierra) 
pl_eqt	Temperatura de Equilibrio del Planeta (K) 
st_teff	Temperatura Efectiva Estelar (K) 
st_rad	Radio Estelar (en radios solares) 
st_mass	Masa Estelar (en masas solares) 
st_met	Metalicidad Estelar ([dex]) 
st_logg	Gravedad Superficial Estelar (log(g) en cgs) 
sy_snum	Número de Estrellas en el Sistema 
sy_pnum	Número de Planetas en el Sistema 
sy_gaiamag	Magnitud Gaia del Sistema 
 
  
Variable Objetivo (Target) 

La variable objetivo es la clasificación del objeto, que puede ser: 

     Confirmado: El objeto ha sido verificado como un exoplaneta.
     Falso Positivo: El objeto fue inicialmente identificado como candidato, pero análisis posteriores descartaron que fuera un planeta.
     

División de Datos 

El conjunto de datos se divide de la siguiente manera para garantizar una evaluación robusta: 

     Conjunto de Entrenamiento (80%): Utilizado para entrenar los modelos.
     Conjunto de Validación (10%): Utilizado para ajustar hiperparámetros y comparar modelos.
     Conjunto de Prueba (10%): Utilizado para la evaluación final del rendimiento del modelo seleccionado.
     

La división se realiza de forma estratificada (stratify=y) para mantener la proporción de clases en todos los conjuntos. 
Requisitos de Hardware 

     CPU: 4+ núcleos recomendados.
     RAM: 8GB+ recomendados.
     Almacenamiento: ~500 MB de espacio libre para datos y modelos.
     Tiempo de ejecución: ~15-30 minutos (dependiendo del hardware).
     

Configuración del Entorno 
Método 1: Usando pip 
bash
 
  
# Crear un entorno virtual
  python -m venv exo-ml-env
  source exo-ml-env/bin/activate  
  En Windows: exo-ml-env\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
 
 
 
Método 2: Usando conda 
bash
 
  
# Crear un entorno conda
conda create -n exo-ml-env python=3.8
conda activate exo-ml-env

# Instalar dependencias
pip install -r requirements.txt
 
 
 
Ejecución del Proyecto 
Para ejecutar todo el pipeline de principio a fin: 
bash
 
  
python main.py
 
 
 
Para ejecutar pasos individuales: 
1. Descargar y preparar datos: 
bash
 
  
bash scripts/download_data.sh
 
 
 
2. Entrenar modelos: 
bash
 
  
bash scripts/train_models.sh
 
 
 
3. Evaluar modelos: 
bash
 
  
bash scripts/evaluate_models.sh
 
 
 
Nota sobre Reproducibilidad 

Este proyecto fija semillas aleatorias en todas las librerías (numpy, random, scikit-learn) para garantizar la reproducibilidad. Sin embargo, algunos modelos (como Random Forest) pueden mostrar ligeras variaciones debido a la paralelización (n_jobs=-1) y diferencias en la arquitectura del hardware. Se ha configurado SKLEARN_SITE_JOBLIB para mitigar este efecto en la medida de lo posible. 
Esquema de Validación 

Se utiliza una validación cruzada de 5 pliegues (cv=5) para la optimización de hiperparámetros con GridSearchCV y RandomizedSearchCV. La métrica principal para la optimización es el accuracy. Posteriormente, todos los modelos se evalúan en un conjunto de validación separado para una comparación justa, utilizando métricas como Accuracy, Precision, Recall, F1-Score y AUC. 


Todos los resultados de los experimentos, incluyendo los mejores hiperparámetros encontrados y las métricas de rendimiento en el conjunto de validación, se registran automáticamente en la carpeta outputs/reports/. Los modelos entrenados se guardan en outputs/models/ y las figuras generadas (matrices de confusión, gráficos comparativos) en outputs/figures/. 
