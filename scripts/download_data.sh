#!/bin/bash

DATA_FILE="data/raw/PS_2025.11.28_09.38.12.csv"
# URL para que el usuario lo descargue manualmente
DATA_URL="https://exoplanetarchive.ipac.caltech.edu/docs/exonews_archive.html"

echo "Verificando los datos necesarios para el proyecto..."

# Crear directorios si no existen
mkdir -p data/raw

# Verificar si el archivo de datos ya existe
if [ -f "$DATA_FILE" ]; then
    echo "✅ Datos encontrados en: $DATA_FILE"
    echo "Puedes continuar con el siguiente paso."
else
    echo "❌ ERROR: El archivo de datos no fue encontrado en $DATA_FILE"
    echo ""
    echo "Por favor, sigue estos pasos para obtener los datos manualmente:"
    echo "1. Visita la página del NASA Exoplanet Archive:"
    echo "   $DATA_URL"
    echo "2. Navega hasta la sección de descarga de la tabla (generalmente un botón 'Download Table')."
    echo "3. Descarga la tabla completa en formato CSV."
    echo "4. Guarda el archivo descargado con el nombre 'exoplanets.csv' dentro de la carpeta 'data/raw/' de este proyecto."
    echo ""
    echo "Una vez hayas colocado el archivo, vuelve a ejecutar este script para verificar."
    exit 1 # Termina el script con un código de error
fi
