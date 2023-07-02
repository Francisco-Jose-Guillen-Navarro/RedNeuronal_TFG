from getPatches import getPatches
import numpy as np
from keras.models import load_model
import pandas as pd
import model_evaluation_utils as meu
from weighted_categorical_crossentropy import WeightedCategoricalCrossentropy


datasetDirectory = './Dataset/Test'

# Obtenemos el conjunto de test y 
# su conjunto de etiquetas correspondiente.
testPatches, testLabels = getPatches(datasetDirectory)

# Escribimos en un fichero los resultados reales para 
# poder compararlos con las predicciones de la red.
with open('testPatches.txt', 'w') as f:
    for matrix in testLabels:
        np.savetxt(f, matrix, fmt='%d')
        f.write('\n\n')

print("Empieza la evaluación >:|")

# Cargamos el modelo entrenado.
cnn = load_model('tfg_net.h5', custom_objects={'loss': WeightedCategoricalCrossentropy})

# Hacemos las predicciones sobre el conjunto
# de datos de test.
testPredictions = cnn.predict(testPatches)

testPredictions = np.argmax(testPredictions, axis=-1)

# Almacenamos en un fichero las predicciones devueltas
# por la red.
with open('testPredictions.txt', 'w') as f:
    for matrix in testPredictions:
        np.savetxt(f, matrix, fmt='%d')
        f.write('\n\n')

# Aplanamos las matrices de etiquetas verdaderas y predichas.
testLabels = testLabels.reshape(-1)
testPredictions = testPredictions.reshape(-1)

# Calculamos y mostramos las métricas por pantalla.
meu.get_metrics(true_labels=testLabels,
                predicted_labels=testPredictions)



