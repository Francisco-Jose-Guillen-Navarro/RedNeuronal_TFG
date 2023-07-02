from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from getPatches import getPatches
from keras import layers
from weighted_categorical_crossentropy import WeightedCategoricalCrossentropy
import keras.backend as K
import numpy as np

# Definimos la localización del Dataset.
datasetDirectory = './Dataset/Training/'

# Definimos los pesos de clase para compensar el 
# desbalanceo entre clases.
class_weight = K.constant([1.0, 10.0, 20.0])

# Obtenemos el conjunto de datos de entrenamiento.
print("Obteniendo el conjunto de entrenamiento...")
trainingPatches, trainingLabels = getPatches(datasetDirectory) 

# Convertimos las etiquetas a un formato compatible con la función de 
# pérdida.
trainingLabels = np.array(trainingLabels).astype('int32')

print(len(trainingPatches))
print(len(trainingLabels))

print("Empieza el entrenamiento >:|")
# Creamos un modelo secuencial.
model = Sequential()

# Agregamos la primera capa de convolución para obtener
# características y patrones relenvantes.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(75, 75, 1)))
model.add(BatchNormalization())

# Agregamos la primera capa de max pooling reduciendo así
# el tamaño de los datos.
model.add(MaxPooling2D((2, 2)))

# Desactivamos aleatoriamente algunas unidades del modelo
# para evitar el overfitting.
model.add(Dropout(0.25))

# Agregamos la segunda capa de convolución.
model.add(Conv2D(64, (3, 3), activation='relu'))

# Normalizamos los datos para acelerar el entrenamiento y
# mejorar el rendimiento del modelo.
model.add(BatchNormalization())

# Agregar la segunda capa de max pooling.
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Agregamos la tercera capa de convolución.
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())

# Agregar la tercera capa de max pooling.
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Aplanamos la salida para hacerla compatible con
# las capas fully connected.
model.add(Flatten())

# Agregamos la primera capa fully connected.
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Agregamos una capa fully connected con 75x75x3 neuronas.
model.add(Dense(75*75*3))

# Agregamos la capa de salida con 75x75 neuronas para adaptar 
# la salida a un matriz de 75x75.
model.add(layers.Reshape((75, 75, 3)))

# Aplicamos una función softmax para obtener las probabilidades, 
# por cada píxel, de que este pertenezca a una clase u otra.
model.add(layers.Softmax())

# Compilar el modelo con una tasa de aprendizaje constante de 0.001 y
# haciendo uso de la función de pérdida personalizada.
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=WeightedCategoricalCrossentropy(class_weight),
              metrics=['accuracy'])

# Entrenamos el modelo con los datos de entrenamiento durante 100 épocas.
model.fit(trainingPatches, trainingLabels, epochs=100)

# Guardamos la CNN entrenada.
model.save('tfg_net.h5')
