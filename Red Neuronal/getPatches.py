import cv2
import os
import csv
import numpy as np

# Función para obtener parches de los fotogramas del dataset. 
# 
# directorio: directorio donde se encuentra el dataset original.
def getPatches(directorio):

    # Creamos dos listas. Una para almacenar cada parche del dataset
    # y otra que almacena las matrices asociadas a cada uno de ellos.
    imagenes = []
    matrices = []

    # Recorremos cada fichero del directiorio actual.
    for root, dirs, files in os.walk(directorio):
        for file in files:

            # Obtenemos una imagen.
            if file.endswith('.png'):
                nombre_imagen = os.path.splitext(file)[0]

                # Comprobamos si dicha imagen tiene un fichero .csv asociado.
                ruta_csv = os.path.join(root, nombre_imagen + '.csv')

                # Si es así...
                if os.path.exists(ruta_csv):

                    # Leemos la imagen y la almacenamos en la lista.
                    img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                    imagenes.append(img)

                    # Abrimos el fichero .csv y leemos su contenido como
                    # una matriz de 75x75. Después, la almacenamos
                    # en su hueco correspondiente en la lista de matrices.
                    with open(ruta_csv, 'r') as f:
                        reader = csv.reader(f)
                        matriz = list(reader)
                        matriz = np.array(matriz).astype(int)
                        matrices.append(matriz)
    
    imagenes = np.array(imagenes)
    matrices = np.array(matrices)
    return imagenes, matrices

