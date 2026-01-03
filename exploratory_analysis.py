from preprocesamiento import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=preprocesing()


df_final,_,_=main(df)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6) 

columna_a_graficar = 'st_teff'

plt.figure(figsize=(10, 5))
sns.histplot(data=df, x=columna_a_graficar, kde=True, bins=30) # kde=True añade una curva de densidad
plt.title(f'Distribución de {columna_a_graficar}')
plt.xlabel('Temperatura Efectiva de la Estrella (K)')
plt.ylabel('Frecuencia')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pl_orbsmax', y='pl_orbper', alpha=0.5) # alpha para ver puntos superpuestos
plt.title('Relación entre Semi-eje Mayor y Período Orbital')
plt.xlabel('Semi-eje Mayor (AU)')
plt.ylabel('Período Orbital (días)')
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='st_teff', y='st_mass', scatter_kws={'alpha':0.3})
plt.title('Relación entre Temperatura y Masa Estelar')
plt.xlabel('Temperatura Efectiva de la Estrella (K)')
plt.ylabel('Masa de la Estrella (Masas Solares)')
plt.show()

print("\nDistribución de la Variable Objetivo (is_confirmed):")
print(df_final['is_confirmed'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='is_confirmed', data=df_final)
plt.title('Distribución de Planetas Confirmados vs. Candidatos')
plt.xlabel('Estado (0: Candidato, 1: Confirmado)')
plt.ylabel('Cantidad')
plt.xticks([0, 1], ['Candidato', 'Confirmado'])
plt.show()

features_numericas = df_final.select_dtypes(include=np.number)

features_numericas.hist(figsize=(15, 12), bins=50, xlabelsize=8, ylabelsize=8)
plt.suptitle("Distribución de las Características Numéricas", y=1.02)
plt.tight_layout()
plt.show()

features_clave = ['pl_rade', 'pl_orbper', 'st_teff', 'st_mass']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparación de Características Clave por Estado', fontsize=16)

for i, feature in enumerate(features_clave):
    row, col = i // 2, i % 2
    sns.boxplot(x='is_confirmed', y=feature, data=df_final, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} vs. Estado')
    axes[row, col].set_xlabel('Estado (0: Candidato, 1: Confirmado)')
    axes[row, col].set_xticklabels(['Candidato', 'Confirmado'])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

correlation_matrix = df_final.corr(numeric_only=True)

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Mapa de Calor de Correlaciones')
plt.show()


