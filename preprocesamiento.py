import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from dotenv import load_dotenv
import os 

from sklearn.impute import KNNImputer

load_dotenv()
cols_error_y_flags = os.getenv('COLS_ERROR_FLAGS')
cols_muchos_nulos = os.getenv('COLS_MUCHOS_NULOS')
cols_referencia = os.getenv('COLS_REFERENCIAS')
df_conservados = os.getenv('DF')
name = os.getenv("NAME")
features=os.getenv("FEATURES")

# Preprocesamiento para el archivo PS_2025.11.28_09.38.12.csv

def nulos(df):
    nulos = df.isnull().sum()
    df_nulos = nulos.reset_index().rename(columns={'index': 'Columna', 0: 'Nulos'})
    resumen_nulos = pd.DataFrame({'Cantidad_Nulos': df.isnull().sum(),
        'Porcentaje_Nulos': (df.isnull().sum() / len(df)) * 100})

    resumen_nulos = resumen_nulos.sort_values(by='Cantidad_Nulos', ascending=False)
    nombre_archivo_salida = 'reporte_nulos.csv'
    resumen_nulos.to_csv(nombre_archivo_salida)

    nulos_ordenados = df.isnull().sum().sort_values(ascending=False)

    plt.figure(figsize=(20, 12)) # Hacemos la figura más grande para que quepan las etiquetas
    nulos_ordenados.plot(kind='bar', color='skyblue')

    plt.title('Cantidad de Valores Nulos por Columna', fontsize=16)
    plt.ylabel('Número de Nulos', fontsize=12)
    plt.xlabel('Columna', fontsize=12)
    plt.xticks(rotation=90, ha='right') 
    plt.tight_layout() # Ajusta todo para que no se solape
    plt.show()
def columnas_nulas(df):

    columnas_a_eliminar = cols_muchos_nulos + cols_error_y_flags + cols_referencia

    # Eliminar las columnas
    df_limpio = df.drop(columns=columnas_a_eliminar)

    print(f"Dimensiones originales: {df.shape}")
    print(f"Dimensiones después de limpiar: {df_limpio.shape}")
    print("\nColumnas restantes:")
    print(len(df_limpio.columns.tolist()))
    return df_limpio

def nulos_filas(df_limpio):
    # Calcula el número de nulos en cada fila
    nulos_por_fila = df_limpio.isnull().sum(axis=1) 

    print("--- Estadísticas de nulos por fila ---")
    print(nulos_por_fila.describe(),len(df_limpio.columns))
    
    nulos_por_fila = df_limpio.isnull().sum(axis=1)

    # Crear el histograma
    plt.figure(figsize=(12, 7))
    nulos_por_fila.hist(bins=29, edgecolor='black', align='left') # 28+1 bins
    plt.title('Distribución de Nulos por Fila')
    plt.xlabel('Número de Nulos en la Fila')
    plt.ylabel('Cantidad de Planetas (Filas)')
    plt.xticks(range(0, 29, 2)) # Marcar cada 2 en el eje X
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    
    umbral_conservador = 16
    df_conservadas1 = df_limpio[nulos_por_fila <= umbral_conservador]
    print(f"Umbral >{umbral_conservador} nulos: Se conservan {len(df_conservadas1)} de {len(df_limpio)} filas.")

    # Eliminar filas con más de 10 nulos (el percentil 75)
    umbral_equilibrado = 10
    df_conservadas2 = df_limpio[nulos_por_fila <= umbral_equilibrado]
    print(f"Umbral >{umbral_equilibrado} nulos: Se conservan {len(df_conservadas2)} de {len(df_limpio)} filas.")
     
def rellenar(df_limpio):
    
    error_cols = [
        'pl_orbpererr1', 'pl_orbpererr2', 'st_tefferr1', 'st_tefferr2',
        'sy_vmagerr1', 'sy_vmagerr2', 'sy_kmagerr1', 'sy_kmagerr2',
        'sy_gaiamagerr1', 'sy_gaiamagerr2'
    ]

    for col in error_cols:
        # Rellenar con la mediana de los errores de esa columna
        df_limpio[col].fillna(df_limpio[col].median(), inplace=True)

    df_filled_knn = df_limpio.copy()

    # KNNImputer funciona solo con valores numéricos.
    numeric_cols_knn = df_filled_knn.select_dtypes(include=np.number).columns

    # Creamos el imputador. 'n_neighbors=5', buscará las 5 filas más similares.
    imputer = KNNImputer(n_neighbors=5)
    df_filled_knn[numeric_cols_knn] = imputer.fit_transform(df_filled_knn[numeric_cols_knn])

    print("\nDataFrame después de rellenar con KNNImputer:")
    print(df_filled_knn.head())

    print("\nVerificación de nulos después del relleno con KNN:")
    print(df_filled_knn.isnull().sum())
    columnas_a_eliminar=["st_metratio","Unnamed: 0"]
    df_limpio = df_filled_knn.drop(columns=columnas_a_eliminar)
    return df_limpio


def preprocesing():
    df = pd.read_csv(name, skiprows=96, header=0)
    nulos = nulos(df)
    df_limpio = columnas_nulas(df)
    nulos_filas(df_limpio)
    
    columnas_a_eliminar = ["disc_year","disc_facility","pl_controv_flag","pl_insolerr1", "pl_insolerr2" , "default_flag","pl_name"]
    df_limpio = df.drop(columns=columnas_a_eliminar)
    df_limpio = rellenar(df_limpio)
    # df_limpio.to_csv("non-null data.csv")

    return df_limpio

def main(df):

    df_transit = df[df['discoverymethod'] == 'Transit'].copy()

    # Convertimos 'soltype' a 1 (Confirmado) y 0 (Candidato)
    df_transit['is_confirmed'] = df_transit['soltype'].apply(
        lambda x: 1 if 'Confirmed' in x else 0
    )

    X = df_transit[features]
    y = df_transit['is_confirmed']

    X = X.dropna()
    y = y[X.index]

    print(f"Datos listos para el modelo: {X.shape[0]} filas")
    return df_transit,X,y