import os
from PIL import Image
import csv
from skimage import io
from NucleusSegmentation import NucleusSegmentation
import numpy as np

directory = 'C:\\Users\\fjgn7\\OneDrive\\Escritorio\\CreateDataset\\Dataset\\Training\\H'
cellsInfo = {'MinSize': 150, 'MinMean': 10, 'MaxMean': 120, 'MinSolidity': 0.88}

patch_size = 75
output_dir = os.path.join(directory, os.path.basename(directory) + "_patches")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = Image.open(os.path.join(directory, filename))
        width, height = image.size
        for i in range(0, width, patch_size):
            for j in range(0, height, patch_size):
                box = (i, j, i + patch_size, j + patch_size)
                if box[2] > width or box[3] > height:
                    box = (width - patch_size, height - patch_size, width, height)
                patch = image.crop(box)
                name, ext = os.path.splitext(filename)
                patch_name = f"{name}_{i}_{j}"

                patch.save(os.path.join(output_dir, f"{patch_name}.png"))
                
                imagen = io.imread(os.path.join(directory, 'H_patches', patch_name + '.png'))
                matrix = NucleusSegmentation(imagen, cellsInfo)
                
                int_matrix = matrix[0].astype(int)

                # Almacenamos solo aquellos parches que contengan píxeles
                # tipo 1 o tipo 2.
                if (np.sum(int_matrix) > 0):

                    reshaped_matrix = np.reshape(int_matrix, (75, 75))

                    # En aquellas posiciones donde la matriz sea 1, ponemos
                    # un 0 si el conjunto es N o un 2 si es tipo 2.
                    reshaped_matrix[np.where(reshaped_matrix == 1)] = 2


                    with open(os.path.join(output_dir, f"{patch_name}.csv"), 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(int_matrix)
                
                else:
                    os.remove(os.path.join(output_dir, f"{patch_name}.png"))
                    
                







