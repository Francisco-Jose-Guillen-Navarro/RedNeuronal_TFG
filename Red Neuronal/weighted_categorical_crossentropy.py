import keras.backend as K

def WeightedCategoricalCrossentropy(class_weight):
    def loss(y_true, y_pred):

        # Convertimos los objetivos a una forma compatible con la 
        # función de pérdida.
        # y_true = K.cast(y_true, 'int32')
        # y_true = K.one_hot(y_true, num_classes=3)
        
        # Calculamos la pérdida comparando los valores reales con
        # los predichos.
        loss = K.categorical_crossentropy(y_true, y_pred)
        
        # Aplicamos los pesos de clase.
        weights = K.gather(class_weight, K.argmax(y_true, axis=-1))
        loss = loss * weights
        
        return loss
    
    return loss