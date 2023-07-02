import numpy as np
from scipy.ndimage import label
from scipy.spatial.distance import cdist

def EvaluateDetection(groundTruth, detectionResult, allowedDistance=10):

    # Definimos tres arrays para representar, para cada
    # imagen, el número de falsos negativos, verdaderos
    # positivos y los falsos positivos.
    imagesFN = np.zeros(len(groundTruth))
    imagesTP = np.zeros(len(groundTruth))
    imagesFP = np.zeros(len(groundTruth))

    # Recorremos cada conjunto de coordenadas marcadas
    # del conjunto groudTruth.
    for i in range(len(groundTruth)):

        # Si no hay un resultado para esa región, se marcan todos esos 
        # píxeles como falsos negativos.
        if not detectionResult[i]:
            imagesFN[i] += groundTruth[i].shape[0]
            continue

        # Si el resultado de la detección no es una lista, hacemos que lo sea.
        if not isinstance(detectionResult[i], list):
            if detectionResult[i].shape[1] == 2:
                detectedPoints = detectionResult[i]
                detectionResult[i] = [np.full((np.amax(detectedPoints[:, 0]) + allowedDistance + 1,
                                               np.amax(detectedPoints[:, 1]) + allowedDistance + 1), False)
                                      for _ in range(detectedPoints.shape[0])]
                for l in range(len(detectionResult[i])):
                    detectionResult[i][l][detectedPoints[l, 0], detectedPoints[l, 1]] = True
                    detectionResult[i][l] = cdist(np.argwhere(detectionResult[i][l]), detectedPoints[l].reshape(1, -1)).min(axis=1).reshape(detectionResult[i][l].shape) <= allowedDistance
            else:
                L, num = label(detectionResult[i])
                if not num:
                    imagesFN[i] += groundTruth[i].shape[0]
                    continue
                detectionResult[i] = [(L == l) for l in range(1, num + 1)]

        # Vemos si en la región analizada actual hay algún núcleo.
        imageSize = detectionResult[i][0].shape
        foreground = np.any(np.stack(detectionResult[i], axis=-1), axis=-1)

        # Creamos una matriz intersección que tenga tantas líneas como coordenadas
        # marcadas y tantas columnas como regiones analizadas.
        intersectionMat = np.full((groundTruth[i].shape[0], len(detectionResult[i])), False)

        # Por cada coordenada... 
        for m in range(groundTruth[i].shape[0] - 1, -1, -1):

            # Si no hay ningún núcleo detectado por la evaluación,
            # eliminamos esa fila de la matriz intersección y 
            # añadimos un falso negativo.
            if not foreground[tuple(groundTruth[i][m])]:
                intersectionMat = np.delete(intersectionMat, m, axis=0)
                imagesFN[i] += 1
                continue

            # Si en la coordenada actual se ha detectado un núcleo, 
            # marcamos ese píxel como verdadero en la matriz intersección.
            for d in range(len(detectionResult[i])):
                if detectionResult[i][d][tuple(groundTruth[i][m])]:
                    intersectionMat[m, d] = True

        changed = True
        while changed:
            changed = False

            # Recorremos las filas de la matriz intersección.
            for m in reversed(range(intersectionMat.shape[0])):

                # Si no hay ninguna región detectada, eliminamos
                # la fila y añadimos un falso negativo.
                if not np.any(intersectionMat[m]):
                    intersectionMat = np.delete(intersectionMat, m, axis=0)
                    imagesFN[i] += 1
                    changed = True
                    continue

                # Si hay un 1 (es decir, tenemos una región detectada), 
                # eliminamos la fila y la columna correspondiente a ese 
                # píxel.
                elif np.count_nonzero(intersectionMat[m] == 1):
                    intersectionMat = np.delete(intersectionMat, np.where(intersectionMat[m]), axis=1)
                    intersectionMat = np.delete(intersectionMat, m, axis=0)
                    imagesTP[i] += 1
                    changed = True

            # Rcorremos las columnas de la matriz intersección.
            for d in reversed(range(intersectionMat.shape[1])):

                # Si no hay ninguna región detectada, eliminamos
                # la fila y añadimos un falso negativo.
                if not np.any(intersectionMat[:, d]):
                    intersectionMat = np.delete(intersectionMat, d, axis=1)
                    imagesFP[i] += 1
                    changed = True
                    continue

                # Si hay un 1, eliminamos la fila y la columna correspondiente
                # a ese píxel.
                elif np.count_nonzero(intersectionMat[:, d] == 1):
                    intersectionMat = np.delete(intersectionMat, np.where(intersectionMat[:, d]), axis=0)
                    intersectionMat = np.delete(intersectionMat, d, axis=1)
                    imagesTP[i] += 1
                    changed = True

        # Añadimos un verdadero positivo por cada región 
        # maracada como verdadera en la matriz intersección.
        imagesTP[i] += min(*intersectionMat.shape)

        # Si hay más filas que columnas, se añaden como falsos
        # negativos. Si esto sucede, indica que hemos detectado
        # menos núcleos que los que debería.
        if intersectionMat.shape[0] > intersectionMat.shape[1]:
            imagesFN[i] += intersectionMat.shape[0] - intersectionMat.shape[1]
        
        # Hacemos lo contrario si hay más columnas que filas. 
        # En este caso, se marca la diferencia como falsos positivos.
        elif intersectionMat.shape[0] < intersectionMat.shape[1]:
            imagesFP[i] += intersectionMat.shape[1] - intersectionMat.shape[0]

    # Calculamos los resultados.    
    imagesP = imagesTP / (imagesTP + imagesFP)
    imagesR = imagesTP / (imagesTP + imagesFN)
    imagesP[np.isnan(imagesP) & (np.array(list(map(len, groundTruth))) > 0)] = 0
    imagesP[np.isnan(imagesP) & (np.array(list(map(len, groundTruth))) == 0)] = 1
    imagesR[np.isnan(imagesR) & (np.array(list(map(len, detectionResult))) > 0)] = 0
    imagesR[np.isnan(imagesR) & (np.array(list(map(len, detectionResult))) == 0)] = 1

    P = np.sum(imagesTP) / (np.sum(imagesTP) + np.sum(imagesFP))
    R = np.sum(imagesTP) / (np.sum(imagesTP) + np.sum(imagesFN))
    stdP = np.std(imagesP)
    stdR = np.std(imagesR)

    return P, R, stdP, stdR