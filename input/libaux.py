"""

######################################################################
# libaux.py
# libreria de funciones auxiliares de componentes de APP-IA-RETOUCHER
# en python no hay function prototypes
#
# funciones que retornan imagenes:
# -------------------------------
#   imgResize(src, percent)
#   transform(img1 , func)
#   painthist(img, title, numbins, filepath_histo) -->genera un file
#
# funciones relacionadas con histogramas y funciones de transformacion
# --------------------------------------------------------------------
#   cumulatedDistance(img1 , img2)
#   getFuncEQ(img)    --> NO SE USA
#   getFuncEQinv(img) --> NO SE USA
#   compose(f1 , f2)
#   denoise(f) --> NO SE USA
#   calculateHistogram(img)
#   calculateCurves(hist_orig, hist_trans)
#   CalculateRGBYUVCurves(orig_img, trans_img)
#   getZoom(img1 , img2)
#
# funciones orientadas a la RNA
# -----------------------------
#   getImgDetails(filepath)
#   getInputFromImg(filepath)
#   loadModel(profiledir)
#   saveModel(model, profiledir)
#
######################################################################

"""
import cv2
# para instalar la libreria openCV simplemente:
# pip3 install opencv-python
# para aprender opencv https://www.geeksforgeeks.org/opencv-python-tutorial
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
import tensorflow as tf
import colorsys
import keras
print(keras.__version__)


# ==========================================================================
def loadModel(profiledir):
    """
    carga el mdelo en formato keras HDF5 (.h5) de un directorio
    Parameters:
    - profiledir: directorio del profile

    Returns:
      model : modelo commpilado
      None: si el modelo no existe
    """
    # cargamos en formato Keras HDF5
    # idea: usar la ultima parte del profile dir (.../pepito/ para
    # asignar el nombre pepito_model.h5
    cad = profiledir + "/" + "model.h5"
    if os.path.exists(cad) is False:
        # print("file no existe")
        return None
    model = tf.keras.models.load_model(cad)
    model.compile(optimizer='adam', loss='mse')
    return model


# ==========================================================================
def saveModel(model, profiledir):
    """

    guarda un modelo entrenado en un directorio de perfil
    en formato keras HDF5 (.h5)

    Parameters:
    - model: el modelo
    - profiledir: directorio del profile

    Returns
    -1 si el directorio no existe
    """
    # salvamos en formato Keras HDF5
    model.save(profiledir + '\\model.h5')
    return None


# =======================================================================
def compose(f1, f2):
    """
    compone dos transformaciones en una
    Parameters:
    - f1: una funcion de transformacion
    - f2: otra funcion de transformacion

    Returns:
      numpy array : funcion de transformacion compuesta
                    f=f2(f1(x))

    """
    g = np.take_along_axis(f2, f1, 1)
    return g


# ===============================================================================
def getImgDetails(filepath):
    """
    consulta metadatos de un fichero de imagen
    y retorna una tupla con los 4 parametros de iluminacion

    Parameters:
    - filepath: path del fichero de imagen

    Returns:
      una tupla de 6 campos
      exposure,iso, flash, aperture, BrightnessValue, ShutterSpeedValue
    """
    # open the image
    image = Image.open(filepath)
    # extracting the exif metadata
    exifdata = image._getexif()

    # Initialises variables with default values
    exposure = 0
    iso = 0
    flash = 0
    aperture = 0
    bright = 0
    speed = 0

    # looping through all the tags present in exifdata
    if exifdata is not None:
        for tagid in exifdata:
            # getting the tag name instead of tag id
            tagname = TAGS.get(tagid, tagid)

            # passing the tagid to get its respective value
            value = exifdata.get(tagid)

            if tagname == "ExposureTime":
                exposure = value
            elif tagname == "ISOSpeedRatings":
                iso = value
            elif tagname == "Flash":
                flash = value
            elif tagname == "ApertureValue":
                aperture = value
            elif tagname == "BrightnessValue":
                bright = value
            elif tagname == "ShutterSpeedValue":
                speed = value

    return exposure, iso, flash, aperture, bright, speed


# ===============================================================================
def transformRGBYUV(orig_img, curvesRGB, curvesYUV):
    """
    dadas una imagen y 6 LUT (lookup tables) de transformacion
    RGB y YUV, las aplica secuencialmente y genera una imagen
    transformada
    Parameters:
      orig_img : imagen bruto
      curvesRGB : 3 LUT RGB
      curvesYUV : 3 LUT YUV

    Returns:
      imagen en formato RGB transformada
    """
    channels = np.array(cv2.split(orig_img), dtype=np.uint8)  # importante dejarlo en 8bit
    # aplicamos la LUT RGB
    for i in range(len(channels)):
        channels[i] = cv2.LUT(channels[i], curvesRGB[i])
    # generamos imagen intermedia sin alterar orig
    orig2_img = cv2.merge(channels)
    # pasamos a YUV sin alterar imagenes
    orig2_img = cv2.cvtColor(orig2_img, cv2.COLOR_BGR2YCrCb, 3)

    # aplicamos la LUT YUV
    channels = np.array(cv2.split(orig2_img), dtype=np.uint8)  # importante dejarlo en 8bit
    for i in range(len(channels)):
        channels[i] = cv2.LUT(channels[i], curvesYUV[i])

    orig3_img = cv2.merge(channels)

    # ahora pasamos a BGR
    final_img = cv2.cvtColor(orig3_img, cv2.COLOR_YCrCb2BGR, 3)

    # retornamos la imagen orig con las transformaciones RGB y YUV
    return final_img


# ===========================================================
def imgResize(src, percent):
    """
    Redimensiona una imagen (ancho y alto)

    Parameters:
    - src: la imagen
    - percent : el porcentaje de reduccion por ejemplo 20 es 20%

    Returns:
      retorna una imagen en el formato manejado por OpenCV (cv2)
    """
    # percent by which the image is resized
    scale_percent = percent

    # calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)
    src2 = np.array(src, dtype=np.ubyte)

    # resize image
    output = cv2.resize(src2, dsize)
    return output


# ===============================================================================
def calculateHistogram(img):
    """
    retorna los 3 histogramas RGB de una imagen
    Si no es en color, retorna un solo histograma

    Parameters:
    - img: la imagen

    Returns:
      histograma en formato lista de python (no numpy)
      con dimension (3 , 256)
      se puede transformar a array numpy, no problem
    """
    channels = cv2.split(img)
    hist = []
    for channel in channels:
        hist.append(cv2.calcHist([channel], [0], None, [256], [0, 256]))

    return hist


# ========================================================================
def lut2poly(y, grado, init=0, fin=255):
    x = list(range(len(y)))  # lista de numeros de 0 a 255
    y = y.astype(int)  # decimales no nos interesan

    # calculo de init: busco el comienzo de la funcion que siempre es abrupto
    if (init != -1):  # -1 desactiva este mecanismo
        init = 0
        for k in range(0, 256):
            if (y[k] > 0):
                init = k
                break
    else:
        # print("init es -1")
        init = 0

    # mejora de fin
    if (fin != -1):
        fin = 255
        for k in range(254, 0, -1):
            if (y[k] != y[k + 1]):
                fin = k
                break
    else:
        fin = 255

    # le daremos mas peso a la primera y ultima muestra, para que
    # el polinomio pase por los puntos inicial y final
    pesos = [1] * 256  # construccion del array de pesos
    # los pesos anteriores a init se ignoran
    for i in range(0, init):
        pesos[i] = 0

    for i in range(255, fin, -1):
        pesos[i] = 0
    pesos[init] = 4
    pesos[fin] = 4  # el ultimo peso
    pesos = np.array(pesos)
    coef = np.polyfit(x, y, grado, w=pesos)  # pesos son criticos en fotos oscuras
    lut = np.polyval(coef, x)  # 256 valores "retocados" con el polinomio

    # ajuste de los primeros valores hasta init, que siempre son cero en componentes UV
    # en RGB init es cero, al igual que en luminancia
    for k in range(0, init):
        lut[k] = y[k]
    for k in range(255, fin, -1):
        lut[k] = y[k]

    # esto no hace falta pero no queremos decimales
    lut = lut.astype(int)

    # ajuste de positivos en el intervalo de 8 bit
    lut[lut < 0] = 0
    lut[lut > 255] = 255
    return lut


# =============================================================================
def calculateCurvesRGB(hist_orig, hist_trans):
    """
    calcula 3 lookup table (LUT) de 256 valores que nos proporciona
    la transformacion de la imagen original para llegar al mismo histograma
    que la imagen transformada. Son 3 por que es R , G , B

    Parameters:
    - hist_orig: histograma de imagen original (3 dimensiones RGB)
    - hist_trans: histogrma de imagen final (3 dimensiones RGB)

    Returns:
      array de dimension (3 , 256) valores con la transformacion que debe
      sufrir la imagen "orig" para tener el histograma de la imagen "trans"

    """
    lut = []
    for i in range(len(hist_orig)):  # esto es un 3 (R, G, B)
        orig_cdf = np.cumsum(hist_orig[i]) / sum(hist_orig[i])
        trans_cdf = np.cumsum(hist_trans[i]) / sum(hist_trans[i])
        j = 0
        lookup_table = np.zeros((256,))
        for k in range(256):
            while trans_cdf[j] < orig_cdf[k] and j < 255:
                j += 1
            lookup_table[k] = j

        grado = 4  # grado RGB
        # tanto init como fin deberian permitirse para ajustar lo mejor posible
        # la curva pero si no pongo polinomial el tramo final, se cuantiza la foto 538
        lookup_table = lut2poly(lookup_table, grado, fin=-1)
        lut.append(lookup_table)
    return lut


# =============================================================================
def calculateCurvesYUV(hist_orig, hist_trans):
    """
    calcula 3 lookup table (LUT) de 256 valores que nos proporciona
    la transformacion de la imagen original para llegar al mismo histograma
    que la imagen transformada. Son 3 por que es R , G , B

    Parameters:
    - hist_orig: histograma de imagen original (3 dimensiones RGB)
    - hist_trans: histogrma de imagen final (3 dimensiones RGB)

    Returns:
      array de dimension (3 , 256) valores con la transformacion que debe
      sufrir la imagen "orig" para tener el histograma de la imagen "trans"
    """
    lut = []
    for i in range(len(hist_orig)):  # esto es un 3 (Y, U, V)
        orig_cdf = np.cumsum(hist_orig[i]) / sum(hist_orig[i])
        trans_cdf = np.cumsum(hist_trans[i]) / sum(hist_trans[i])
        j = 0
        lookup_table = np.zeros((256,))
        for k in range(256):
            while trans_cdf[j] < orig_cdf[k] and j < 255:
                j += 1
            lookup_table[k] = j

        if (i == 0):

            grado = 3  # poco grado para la luz, de lo contrario fotos oscuras salen mal
            lookup_table = lut2poly(lookup_table, grado, init=-1, fin=-1)

        else:
            grado = 10  # grado para color
            # en principio si hay 8 grupos, necesitamos el doble para unir dos maximos (16)
            # pero con grado 8 queda bien, no parece que haga falta mas
            lookup_table = lut2poly(lookup_table, grado)

        lut.append(lookup_table)

    return lut


# ===============================================================================
def CalculateRGBYUVCurves(orig_img, trans_img):
    """

    dadas dos imagenes en formato RGB, calcula
    las lookup tables de RGB y YUV (realmente YCrCb) para llegar desde la
    imagen original a la transformada

    Parameters:
      orig_img : imagen bruto
      tras_img : imagen transformada (por fotografo)

    Returns:
      una lista de dimension (2, (3 , 256)) con las 6 LUT
      LUT significa LookUp Table
      (curvasRGB, curvasYUV)

    """
    # procesamiento RGB
    orig_histRGB = calculateHistogram(orig_img)
    trans_histRGB = calculateHistogram(trans_img)
    # calculamos la LUT ("look-up table") RGB
    curvesRGB = calculateCurvesRGB(orig_histRGB, trans_histRGB)

    channelsRGB = np.array(cv2.split(orig_img), dtype=np.uint8)  # importante dejarlo en 8bit
    # aplicamos la LUT RGB
    for i in range(len(channelsRGB)):
        channelsRGB[i] = cv2.LUT(channelsRGB[i], curvesRGB[i])
    # generamos imagen intermedia sin alterar orig
    orig2_img = cv2.merge(channelsRGB)
    # pasamos a YUV sin alterar imagenes
    orig2YUV_img = cv2.cvtColor(orig2_img, cv2.COLOR_BGR2YCrCb, 3)
    transYUV_img = cv2.cvtColor(trans_img, cv2.COLOR_BGR2YCrCb, 3)
    orig2YUV_hist = calculateHistogram(orig2YUV_img)
    transYUV_hist = calculateHistogram(transYUV_img)
    # calculamos la LUT ("look-up table") YUV
    curvesYUV = calculateCurvesYUV(orig2YUV_hist, transYUV_hist)

    # retornamos las 6 LUT (3 de RGB y 3 de YUV)
    return curvesRGB, curvesYUV


# ===============================================================================
def CumulatedDistance(img1, img2):
    """
    calcula la distancia entre dos imagenes
    usando el algoritmo de distancia acumulada.
    el resultado es la suma de las distancias acumuladas R, G , B
    la distancia es independiente de la resolucion
    de las imagenes. Pueden incluso ser de distinto tamaño

    Parameters:
    - img1 : una imagen ya cargada en formato OpenCV
    - img2 : otra imagen ya cargada en formato OpenCV

    Returns:
      el resultado es un numero entero resultante de
      la suma de las 3 distancias acumuladas R, G, B
    """
    color = ('b', 'g', 'r')
    dist = 0
    im1 = cv2.split(img1)
    im2 = cv2.split(img2)
    numbins = 100  # con 20, 100 o 256 bins da resultados similares (diagrama de dispersion similar)
    for i, col in enumerate(color):
        hist1, bins1 = np.histogram(im1[i].ravel(), bins=numbins, range=[0, 256], weights=None, density=None)
        hist2, bins2 = np.histogram(im2[i].ravel(), bins=numbins, range=[0, 256], weights=None, density=None)
        hist1 = np.array(hist1 / sum(hist1))
        hist2 = np.array(hist2 / sum(hist2))
        a = np.cumsum(hist1)
        b = np.cumsum(hist2)
        da = np.array(abs(a - b))  # esto es variacion
        suma = np.sum(da)  # da es un array numpy que contiene diferencias de probabilidad en cada bin
        dist = dist + suma

    return dist


# ===============================================================================
def painthist(img, title, numbins, filepath_histo):
    fig, ax = plt.subplots()
    im1 = cv2.split(img)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist1, bins1 = np.histogram(im1[i].ravel(), bins=numbins, range=[0, 256], weights=None, density=True)  # OK
        plt.plot(hist1, color=col)
        plt.xlim([0, numbins])
    plt.title(title)
    plt.ylim(0, 0.05)  # ojo con los ejes. esto es para limitar el eje Y y asi poder comparar visualmente histogramas
    plt.savefig(filepath_histo)
    plt.close()


# ===============================================================================
# dada una funcion de transformacion la suaviza para quitar ruido y acercarnos al histograma deseado
# se puede suavizar haciendo la mediana de los valores centrados en cada valor. Si hacemos una mediana de 10 valores,
# los 10 primeros y los 10 ultimos seran iguales.
def denoise(f):
    """
    dada una funcion de transformacion la suaviza para
    quitar ruido y acercarnos al histograma deseado
    se puede suavizar haciendo la mediana de los valores
    centrados en cada valor. Si hacemos una mediana de 10 valores,
    los 10 primeros y los 10 ultimos seran iguales.

    Parameters:
      f : funcion de transformacion (un array de 768 elementos)

    Returns:
      g : funcion f transformada
    """
    g = np.zeros(256 * 3, dtype=np.uint8)
    g.shape = (3, 256)
    count = 2
    a = np.zeros(count * 2, dtype=np.uint8)
    for color in (0, 1, 2):
        for i in range(0, 256):
            sum = 0
            for j in range(i - count, i + count):
                k = max(0, j)
                k = min(255, k)
                sum = sum + f[color][k]
                a[j - i + count] = f[color][k]
            g[color][i] = sum / (count * 2)
    return g


# ===============================================================================
# calcula la funcion (=array de transformacion)que ecualiza un histograma
def getFuncEQ(img):

    # la funcion de ecualizacion tiene 256 entradas porque ese es el rango de la señal
    feq = np.zeros(256 * 3, dtype=np.uint8)
    feq.shape = (3, 256)  # es una funcion con 3 arrays de transformacion
    denom = img.shape[0] * img.shape[1]  # ancho x alto
    im1 = cv2.split(img)  # imprescindible!!!

    for i in (0, 1, 2):  # R , G , B
        # bins con num pixeles, no probabilidad. deben ser 256 bins en este caso. no pueden ser menos
        hist, bins = np.histogram(im1[i].ravel(), bins=256, range=[0, 256], weights=None, density=None)

        # optimizacion
        a = np.cumsum(hist)
        feq[i] = np.array((a * 255) / denom)

    return feq


# ===============================================================================
# esta funcion calcula la inversa de ecualizacion de una imagen
# se parece la la ecualizacion directa, y tiene tres componentes [i]
# en una ecualizacion f(valor)= newvalor, es decir es un array  f[i][valor] = valor'
# en la inversa f(newvalor)= valor, es decir es un array finv[i][valor']=valor
def getFuncEQinv(img):

    feq = getFuncEQ(img)
    feq = denoise(feq)

    finv = np.zeros(256 * 3, dtype=int)
    finv.shape = (3, 256)
    for color in (0, 1, 2):
        last_result = 0
        for i in range(0, 256):
            suma = 0
            count = 0
            # hay una ambiguedad. pueden varios bins acabar en el mismo. ¿cual coger?
            # cuando esto ocurre se pierde informacion
            a = np.zeros(256)
            for j in range(0, 256):
                if (feq[color][j] == i):  # j pasa a i en la funcion de ecualizacion
                    suma = suma + j
                    a[count] = j
                    count = count + 1
            if (count > 0):
                # lo ideal seria la media pero no de los bins que acaban en el bin i, sino
                # ponderados por el numero de pixeles de dichos bins y eso no lo tenemos (podriamos calcularlo)
                # una aproximacion es la media de esos bins o el "bin medio"
                finv[color][i] = suma / (count)  # la media de los que cumplen
            else:  # si no encuentra ningun valor que se transforme a j usamos el ultimo (sera parecido)
                finv[color][i] = last_result
            last_result = finv[color][i]
    return finv


# ===============================================================================
def yuv_to_rgb(yuv):
    y, u, v = yuv
    r = max(0, min(round(y + 1.402 * (v - 128)), 255))
    g = max(0, min(round(y - 0.343 * (u - 128) - 0.71136 * (v - 128)), 255))
    b = max(0, min(round(y + 1.765 * (u - 128)), 255))
    return r, g, b


# ===============================================================================
def ycbcr_to_rgb(yuv):
    y, u, v = yuv
    r = max(0, min(round(y + 1.403 * (v - 128)), 255))
    g = max(0, min(round(y - 0.344 * (u - 128) - 0.714 * (v - 128)), 255))
    b = max(0, min(round(y + 1.773 * (u - 128)), 255))
    return r, g, b


# ===============================================================================
def rgb_to_hsl(rgb):
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    hue = int(h * 240)  # Convertir el tono a grados (0-240)
    saturation = int(s * 240)  # Convertir la saturación a porcentaje (0-240)
    lightness = int(l * 240)  # Convertir la luminosidad a porcentaje (0-240)

    return hue, saturation, lightness
