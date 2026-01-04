import pandas as pd
import numpy as np
# Cambia el backend de matplotlib antes de importarlo
import matplotlib
matplotlib.use('Agg')  # Usa backend no interactivo
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os 

from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

load_dotenv()
# cols_error_y_flags = os.getenv('COLS_ERROR_FLAGS')
# cols_muchos_nulos = os.getenv('COLS_MUCHOS_NULOS')
# cols_referencia = os.getenv('COLS_REFERENCIAS')
# df_conservados = os.getenv('DF')
# # name_ = os.getenv("NAME")
# features=os.getenv("FEATURES")

cols_muchos_nulos = ['pl_eqterr1', 'pl_eqterr2', 'st_spectype', 'pl_orbeccenerr1', 'pl_orbeccenerr2','pl_bmasse', 'pl_bmassj', 'pl_bmasseerr1', 'pl_bmasseerr2', 'pl_bmassjerr1', 'pl_bmassjerr2', 'pl_bmasselim', 'pl_bmassjlim', 'pl_bmassprov']
cols_error_y_flags= ['pl_orbsmaxerr1', 'pl_orbsmaxerr2', 'pl_radeerr1', 'pl_radeerr2', 'pl_radjerr1', 'pl_radjerr2', 'st_raderr1', 'st_raderr2',  'st_masserr1', 'st_masserr2', 'st_meterr1', 'st_meterr2', 'st_loggerr1', 'st_loggerr2', 'pl_radelim', 'pl_radjlim', 'pl_orbperlim','pl_orbsmaxlim', 'st_tefflim', 'st_radlim', 'st_masslim', 'st_metlim', 'st_logglim','pl_eqtlim', 'pl_orbeccenlim', 'pl_insollim']
cols_referencia = ['pl_refname', 'st_refname', 'sy_refname', 'rowupdate', 'pl_pubdate', 'releasedate', 'ra', 'rastr', 'dec', 'decstr', 'sy_dist', 'sy_disterr1', 'sy_disterr2']
features= ['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_orbeccen', 'pl_insol', 'pl_eqt','st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg','sy_snum', 'sy_pnum', 'sy_gaiamag']

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
    print(23)
    # Crear el histograma

    plt.figure(figsize=(12, 7))
    nulos_por_fila.hist(bins=29, edgecolor='black', align='left')  # 28+1 bins
    plt.title('Distribución de Nulos por Fila')
    plt.xlabel('Número de Nulos en la Fila')
    plt.ylabel('Cantidad de Planetas (Filas)')
    plt.xticks(range(0, 29, 2))  # Marcar cada 2 en el eje X
    plt.grid(axis='y', alpha=0.75)
    
    # Guardar la figura en lugar de mostrarla
    fig_path = os.path.join('outputs', 'figures', 'nulos_por_fila.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(2)
    umbral_conservador = 16
    df_conservadas1 = df_limpio[nulos_por_fila <= umbral_conservador]
    print(f"Umbral >{umbral_conservador} nulos: Se conservan {len(df_conservadas1)} de {len(df_limpio)} filas.")

    # Eliminar filas con más de 10 nulos (el percentil 75)
    umbral_equilibrado = 10
    df_conservadas2 = df_limpio[nulos_por_fila <= umbral_equilibrado]
    print(f"Umbral >{umbral_equilibrado} nulos: Se conservan {len(df_conservadas2)} de {len(df_limpio)} filas.")
    return df_conservadas2
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
    columnas_a_eliminar=["st_metratio"]
    df_limpio = df_filled_knn.drop(columns=columnas_a_eliminar)
    return df_limpio


def load_data(name="../../data/raw/PS_2025.11.28_09.38.12.csv"):

    df = pd.read_csv(name, skiprows=96, header=0)
    
    # nulos(df)
    print("pase ni")
    df_limpio = columnas_nulas(df)
    a=nulos_filas(df_limpio)
    print(111)
    columnas_a_eliminar = ["disc_year","disc_facility","pl_controv_flag","pl_insolerr1", "pl_insolerr2" , "default_flag","pl_name"]
    df_limpio = df.drop(columns=columnas_a_eliminar)
    df_limpio = rellenar(df_limpio)
    # df_limpio.to_csv("non-null data.csv")

    return df_limpio

def preprocess_data(name):
    print(name)
    df=load_data(name)
    print(2)
    df_transit = df[df['discoverymethod'] == 'Transit'].copy()
    print(666)

    # Convertimos 'soltype' a 1 (Confirmado) y 0 (Candidato)
    df_transit['is_confirmed'] = df_transit['soltype'].apply(
        lambda x: 1 if 'Confirmed' in x else 0
    )
    print(6666)
    X = df_transit[features]
    y = df_transit['is_confirmed']

    X = X.dropna()
    y = y[X.index]
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
 
    return X_train, X_val, X_test, y_train, y_val, y_test, imputer, scaler    
    
