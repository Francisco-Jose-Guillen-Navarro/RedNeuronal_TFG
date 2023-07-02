import os
import pandas as pd
from skimage import io
from NucleusSegmentation import NucleusSegmentation
from EvaluateDetection import EvaluateDetection

# Obtenemos el directiorio del dataset
datasetDirectory = '..\dataset'

# Leemos las imágenes EDF
allEDF = [f for f in os.listdir(os.path.join(datasetDirectory, 'EDF')) if f.endswith('.png')]

# Obtenemos la información correspondiente a cada frame del dataset
datasetInfo = pd.read_csv(os.path.join(datasetDirectory, 'labels.csv'))

# Obtenemos la información correspondiente al conjunto de frames 
# de testing
testInd = datasetInfo['set'].to_numpy().nonzero()[0]
testing = datasetInfo[datasetInfo['set'] == True]

# Inicializamos las variables que almacenarán las coordenadas de 
# los núcleos detectados a mano, el resultado de la operación de 
# segmentación y las imágenes que componen el conjunto de testing
testGroundTruth = [None] * len(testing)
testSegmentationResult = [None] * len(testing)
allEDFImages = [None] * len(testing)

# Por cada frame en el conjunto de testing...
for s in range(len(testing)):

    # Obtenemos las imágenes que lo componen
    allEDFImages[s] = io.imread(os.path.join(datasetDirectory, 'EDF', testing['frame'].iloc[s] + '.png'))
    
    # Obtenemos las coordenadas reales de los núcleos en dichas imágenes
    testGroundTruth[s] = pd.read_csv(os.path.join(datasetDirectory, 'EDF', testing['frame'].iloc[s] + '.csv'), header=None).values

# Definimos los parámetros del método
cellsInfo = {'MinSize': 150, 'MinMean': 10, 'MaxMean': 120, 'MinSolidity': 0.88}

# Realizamos la segmentación a cada imagen y obtenemos el resultado
for s in range(len(testing)):
    _, testSegmentationResult[s] = NucleusSegmentation(allEDFImages[s], cellsInfo)

# Evaluamos el resultado de la segmentación
P, R, stdP, stdR = EvaluateDetection(testGroundTruth, testSegmentationResult)

# Mostramos los resultados obtenidos por pantalla
print('Min Size: {}\tMin Mean: {}\tMax Mean: {}\tMin Solidity: {:.2f}\n\n'.format(cellsInfo['MinSize'], cellsInfo['MinMean'], cellsInfo['MaxMean'], cellsInfo['MinSolidity']))
print(('\t{:.3f} ({:.3f})' * 2 + '\n\t{:.3f}\n').format(P, stdP, R, stdR, 2 * P * R / (P + R)))








