import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew

# Directorios de las imágenes esteganográficas y originales
directorio_imagenes_esteganografia = 'Q:\\imagenes\\UERD'
directorio_imagenes_originales = "Q:\\imagenes\\Cover"
archivos_imagenes_esteganograficas = [f for f in os.listdir(directorio_imagenes_esteganografia) if f.endswith(('.bmp', '.png', '.jpg'))]
archivos_imagenes_originales = [f for f in os.listdir(directorio_imagenes_originales) if f.endswith(('.bmp', '.png', '.jpg'))]


# Limitar a las primeras 5000 imágenes
archivos_imagenes_esteganograficas = archivos_imagenes_esteganograficas[:5000]
archivos_imagenes_originales = archivos_imagenes_originales[:5000]


def escala_grises(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def calcular_glcm(image):
    distances = [1]
    angles = [0]
    glcm = graycomatrix(image, distances, angles, 256, symmetric=True, normed=True)
    return glcm

def extraer_caracteristicas(glcm):
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    asimetria = skew(glcm.flatten()) 
    return contrast, homogeneity, asimetria

def calculate_mse_psnr(original, stego):
    mse = np.mean((original - original) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 10 * np.log10(255**2 / mse)
    return mse, psnr

def count_local_minima_maxima(image):
    # Convertir la imagen a float32
    image = image.astype(np.float32)

    # Crear un kernel de vecinos para encontrar los máximos locales
    kernel = np.ones((3, 3), np.uint8)
    local_max = cv2.dilate(image, kernel) == image
    local_min = cv2.erode(image, kernel) == image

    # Contar los máximos y mínimos locales
    num_local_maxima = np.sum(local_max).astype(int)
    num_local_minima = np.sum(local_min).astype(int)
    
    return num_local_minima, num_local_maxima

datos = []

for archivo_estego, archivo_original in zip(archivos_imagenes_esteganograficas, archivos_imagenes_originales):
    ruta_completa_estego = os.path.join(directorio_imagenes_esteganografia, archivo_estego)
    ruta_completa_original = os.path.join(directorio_imagenes_originales, archivo_original)
    
    imagen_estego = cv2.imread(ruta_completa_estego)
    imagen_original = cv2.imread(ruta_completa_original)

    if imagen_estego is None or imagen_original is None:
        print(f"Error: No se pudo leer la imagen en {ruta_completa_estego} o {ruta_completa_original}")
        continue

    # Convertir ambas imágenes a escala de grises
    escala_gris_estego = escala_grises(imagen_estego)
    escala_gris_original = escala_grises(imagen_original)
    
    # Calcular GLCM y extraer características
    glcm = calcular_glcm(escala_gris_estego)
    contrast, homogeneity, asimetria = extraer_caracteristicas(glcm)
    
    # Calcular MSE y PSNR comparando la imagen original con la esteganográfica
    mse, psnr = calculate_mse_psnr(escala_gris_original, escala_gris_estego)
    
    # Contar mínimos y máximos locales en la imagen esteganográfica
    minima, maxima = count_local_minima_maxima(escala_gris_estego)
    
    # Crear el vector de características
    vector_esteganografica = [contrast, homogeneity, asimetria, mse, psnr, minima, maxima]

    # Agregar todos los datos al dataset
    datos.append({
        "archivo_estego": archivo_estego,
        "archivo_original": archivo_original,
        "contraste": contrast,
        "Asimetria":asimetria,
        "homogeneidad": homogeneity,
        "mse": mse,
        "psnr": psnr,
        "minimos_locales": minima,
        "maximos_locales": maxima,
        "Algoritmo": "ORIGINAL"
    })

# Guardar los datos en un CSV
df = pd.DataFrame(datos)
df.to_csv("esteganografia_caracteristicas_ORIGINAL.csv", index=False)
print("Proceso completado y datos guardados en esteganografia_caracteristicas.csv")