import numpy as np
from scipy import ndimage
from skimage import morphology, measure
from skimage.morphology import reconstruction

def NucleusSegmentation(I, cellsInfo=None):

    # Comprobamos si la imagen que recibimos como parámetro
    # es RGB. Si es así, la pasamos a blanco y negro.
    if I.ndim == 3:
        I = np.mean(I, axis=2)

    # Si no hemos recibido una estructura cellsInfo, la creamos.
    if cellsInfo is None:
        cellsInfo = {'MinSize': 150, 'MinMean': 30,
                     'MaxMean': 150, 'MinSolidity': 0.9}

    # Definimos la intensidad mínima y máxima.
    lowN = cellsInfo['MinMean'] // 10 * 10
    highN = -(-cellsInfo['MaxMean'] // 10) * 10

    # Aplicamos un filtro para eliminar el ruido de la imagen.
    I = ndimage.filters.median_filter(I, size=(5, 5))

    # Creamos la matriz nuclei.
    nuclei = np.zeros_like(I)

    # Obtenemos todos los píxeles de la imagen.
    allPixels = I.size

    # De menor a mayor intensidad... 
    for thresh in range(lowN, highN + 1, 10):

        # Creamos una imagen binaria a partir de aquellos
        # píxeles de I que sean menores o iguales a la intensidad 
        # actual.
        binaryImage = I <= thresh

        # Si más del 20% de los píxeles están presentes en la imagen binaria,
        # salimos del bucle 
        if binaryImage.sum() > allPixels / 5:
            break

        # Etiquetamos como "regiones" aquellas zonas de la imagen binaria
        # donde hay varios 1's juntos.
        blobs = measure.label(binaryImage)

        # Obtenemos las propiedades de esas regiones.
        regProp = measure.regionprops(blobs)

        # Creamos una lista en la que cada hueco representa a una región.
        # Las regiones que estén maracadas como verdaderas al final de la 
        # ejecución, serán identificadas como núcleos.
        addTheseRegions = np.ones(len(regProp), dtype=bool)

        # Eliminamos las regiones que sean o muy pequeñas o que no tengan la 
        # solidez suficiente.
        removeTooConcaveTooSmallBlobs = (
            np.array([r.area for r in regProp]) < cellsInfo['MinSize']) | (
            np.array([r.solidity for r in regProp]) < cellsInfo['MinSolidity'])

        addTheseRegions[removeTooConcaveTooSmallBlobs] = False

        # Vemos si ya hay regiones marcadas como núcleos.
        pixelsAlreadyInNuclei = (blobs != 0) & (nuclei != 0)
        blobsAlreadyInNuclei = np.unique(blobs[pixelsAlreadyInNuclei])

        # Marcamos como núcleos aquellas regiones de 1's que están en
        # la matriz nuclei.
        nuclei = measure.label(nuclei)

        # Obtenemos las propiedades de las regiones maracadas como núcleos.
        nucRegProp = measure.regionprops(nuclei)
        
        # Si ya hay regiones marcadas como núcleos, comprobamos que estas
        # tienen la solidez suficiente. Si no es así, las descartamos.
        if blobsAlreadyInNuclei.size > 0:
            for j in np.transpose(blobsAlreadyInNuclei):
                intersectWithThese = np.unique(nuclei[blobs == j])
                if regProp[j - 1].solidity < max(
                        [nucRegProp[k - 1].solidity for k in intersectWithThese if k > 0]):
                    addTheseRegions[j - 1] = False

        # Obtenemos las coordenadas de las regiones a añadir como núcleos 
        # y las marcamos dentro de la matriz nuclei.
        coords = [regProp[i].coords for i in range(len(regProp)) if addTheseRegions[i]]
        
        for i in range(len(coords)):
            for j in range(len(coords[i])):
                nuclei[coords[i][j, 0]][coords[i][j, 1]] = 1

        nuclei = nuclei.astype(bool)

    # Eliminamos los núcleos que no tengan el tamaño mínimo.
    nuclei = morphology.remove_small_objects(ndimage.morphology.binary_fill_holes(nuclei), min_size=cellsInfo['MinSize'], connectivity=2)

    # Dilatamos la matriz nuclei para reasaltar los núcleos.
    dilatedSeg = ndimage.morphology.binary_dilation(nuclei, structure=morphology.disk(1))
    regionsLabel, _ = measure.label(nuclei, return_num=True)
    dilatedRegionsLabel, numOfDilatedRegions = measure.label(dilatedSeg, return_num=True)
    
    for l in range(numOfDilatedRegions):
        if len(np.unique(regionsLabel[dilatedRegionsLabel == l])) >= 3:
            dilatedSeg[dilatedRegionsLabel == l] = nuclei[dilatedRegionsLabel == l]
    
    nuclei = dilatedSeg

    # Eliminamos los núcleos que estén en los bordes de la matriz.
    imageBoundary = np.zeros_like(nuclei).astype(bool)
    imageBoundary[[0, -1], :] = True
    imageBoundary[:, [0, -1]] = True

    imageBoundary = np.minimum(imageBoundary, nuclei)
    nuclei[reconstruction(imageBoundary, nuclei).astype(bool)] = False

    # Obtenemos una lista de máscaras que representan a cada núcleo
    # que hemos encontrado.
    L, num = measure.label(nuclei, return_num=True)
    masks = [L == i for i in range(num)]

    print("Imagen procesada")

    return nuclei, masks