"""
######################################################################
# dataTrainGen.py
#  este modulo produce  datos en un fichero numpy para entrenar
#
# funciones de uso externo
# -------------------------
#   dataGen(brutefilelist, retouchedfilelist)
#   dataGenImage(file)
#
# funciones de uso interno
# ------------------------
#   checkCoherence(file_1, file_1t)
######################################################################



"""
import cv2
import numpy as np
import os
from pathlib import Path
import libaux
import re


# ==========================================================================
def checkCoherence(file_1, file_1t):
    """

    Comprueba que el nombre de un fichero imagen bruto
    y transformados son coherentes en su numero de
    secuencia que se encuentra contenido en el nombre

    Paremeters
    - file_1: path fichero imagen bruto
    - file_1t: path fichero imagen transformada by fotografo

    Returns:
    0 : OK
    -1: incoherencia


    """
    cad_bruto = Path(file_1).stem
    cad_trans = Path(file_1t).stem

    numero1 = [int(s) for s in re.findall(r'-?\d+\.?\d*', cad_bruto)]
    numero2 = [int(s) for s in re.findall(r'-?\d+\.?\d*', cad_trans)]
    if (numero1 != numero2):
        return -1
    return 0


# ==========================================================================
def dataGen_old(brutefilelist, retouchedfilelist):
    """ produce los datos para que una RNA entrene

    retorna un array numpy para entrenar

    Parameters
      - brutefilelist: lista de fotos en bruto ya cribadas

      - retouchedfilelist: lista de fotos correspondientes retocadas


    Returns:
    ret, data :tupla de dos resultaos
    ret:
        -1 si no pudo encontrar alguno de los directorios
         0 si ok
      data: array numpy con datos de entrenamiento. None si algo fue mal
    """
    res_in_data = np.zeros(0)
    res_out_data = np.zeros(0)
    index = 0
    for file_1 in brutefilelist:
        print("-----photo brute & trans readings-------")
        print("  DataTrainGen: orig =", file_1)
        file_1t = retouchedfilelist[index]  # nombre
        index = index + 1
        print("  DataTrainGen: trans=", file_1t)

        # comprueba que el nombre retocado esta en el nombre bruto
        res = checkCoherence(file_1, file_1t)
        if(res == index - 1):
            print("warning: incoherencia de ficheros bruto y trans")
            # solucion
            for i in range(0, len(retouchedfilelist)):
                file_1t = retouchedfilelist[i]
                res = checkCoherence(file_1, file_1t)
                if(res == 0):
                    index = i + 1  # sumamos ya el 1
                    break
            if (res == -1):
                print("foto no encontrada en retouched:", file_1)
                continue  # nos olvidamos de esta foto
            # return -1, None

        # genera el vector de datos "input" desde la imagen original
        # ----------------------------------------------------
        res, input_data = dataGenImage(file_1)
        if (res == -1):
            return -1

        # datos deseados a producir:
        # ---------------------------
        img1 = cv2.imread(file_1, cv2.IMREAD_COLOR)  # necesitamos leer la imagen bruto
        img1t = cv2.imread(file_1t, cv2.IMREAD_COLOR)  # necesitamos imagen retocada by fotografo

        # ecualizacion de imagen bruto y ecualizacion inversa de retocada
        # ----------------------------------------------------------------
        func_eq = libaux.getFuncEQ(img1)
        func_inv = libaux.getFuncEQinv(img1t)

        # funcion compuesta para hacer el camino entre la original y la retocada
        # --------------------------------------------------------------------
        func_comp = libaux.compose(func_eq, func_inv)

        # insert elementos en los numpy arays de respuesta
        res_in_data = np.append(res_in_data, input_data)
        func_comp = np.array(func_comp / 255)
        res_out_data = np.append(res_out_data, func_comp)

    # construccion de la respuesta compuesta de dos arrays numpy (x,y)
    res_in_data = np.asarray(res_in_data).astype('float32')
    res_in_data.shape = (index, 774)  # antes era 772
    res_out_data = np.asarray(res_out_data).astype('float32')
    res_out_data.shape = (index, 768)
    ret = 0
    return ret, (res_in_data, res_out_data)


# ==========================================================================
def dataGenImage(file):
    """

    dada una imagen bruto, genera el vector de datos
    de entrada para la RNA.
    Es decir, esta funcion genera el vector de una sola imagen

    Parameters:
    - file : filepath de imagen en bruto

    Returns
    - array numpy con :
        6 valores de iluminacion 0..1
        histograma , valores 0..1
    """
    # foto bruto
    details_orig = libaux.getImgDetails(file)  # nombre
    # exposure,iso, flash, aperture
    # -----------------------------
    details1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    details1[0] = details_orig[0] / 10  # exposure <1 segundo siempre
    details1[1] = details_orig[1] / 3000  # iso, puede ser grande
    details1[2] = details_orig[2] / 16  # flash 0 o 16
    details1[3] = details_orig[3] / 10  # aperture max 1.35
    details1[4] = details_orig[4] / 10  # bright
    details1[5] = details_orig[5] / 16  # shutter speed
    # print(details1)
    img1 = cv2.imread(file, cv2.IMREAD_COLOR)  # imagen
    # datos input
    # ------------
    details = np.array(details1)  # 6 parametros
    inputdata = np.zeros(0)
    inputdata = np.append(inputdata, details)
    # ahora los 3 histogramas
    color = ('b', 'g', 'r')
    histrgb = np.zeros(0)
    for i, col in enumerate(color):
        hist, bins1 = np.histogram(img1[i].ravel(), bins=256, range=[0, 256], weights=None, density=None)  # OK
        hist = np.array(hist / sum(hist))
        # prueba para ver si mejora calidad azules
        # converge mejor al multiplicar por algo, hasta x10 es bueno aunque no imprescindible
        # hist = np.array(hist * 5)
        hist = np.array(hist * 2)  # esto es lo mas conservador y a la vez potencia
        hist[hist > 1.0] = 1.0
        # end prueba
        histrgb = np.append(histrgb, hist)

    histrgb.shape = (3 * 256)
    inputdata = np.append(inputdata, histrgb)
    inputdata = np.asarray(inputdata).astype('float32')
    inputdata.shape = (1, 774)  # antes era 772
    ret = 0
    return ret, inputdata


# ==========================================================================
def dataGen(brutefilelist, retouchedfilelist):
    """ produce los datos para que una RNA entrene

    retorna un array numpy para entrenar

    Parameters
      - brutefilelist: lista de fotos en bruto ya cribadas

      - retouchedfilelist: lista de fotos correspondientes retocadas


    Returns:
     ret, data :tupla de dos resultaos
      ret:
        -1 si no pudo encontrar alguno de los directorios
         0 si ok
      data: array numpy con datos de entrenamiento. None si algo fue mal

    """
    res_in_data = np.zeros(0)
    res_out_data = np.zeros(0)
    index = 0
    for file_1 in brutefilelist:
        print("-----photo brute & trans readings-------")
        print("  DataTrainGen: orig =", file_1)
        file_1t = retouchedfilelist[index]  # nombre
        index = index + 1
        print("  DataTrainGen: trans=", file_1t)

        # comprueba que el nombre retocado esta en el nombre bruto
        res = checkCoherence(file_1, file_1t)
        if (res == -1):
            print("warning: incoherencia de ficheros bruto y trans")
            # solucion
            for i in range(0, len(retouchedfilelist)):
                file_1t = retouchedfilelist[i]
                res = checkCoherence(file_1, file_1t)
                if(res == 0):
                    index = i + 1  # sumamos ya el 1
                    break

            if (res == -1):
                print("foto no encontrada en retouched:", file_1)
                continue  # nos olvidamos de esta foto
            # return -1, None

        # genera el vector de datos "input" desde la imagen original
        # ----------------------------------------------------
        res, input_data = dataGenImage(file_1)
        if (res == -1):
            return -1

        # datos deseados a producir:
        # ---------------------------
        img1 = cv2.imread(file_1, cv2.IMREAD_COLOR)  # necesitamos leer la imagen bruto
        img1t = cv2.imread(file_1t, cv2.IMREAD_COLOR)  # necesitamos imagen retocada by fotografo

        # calculo de las LUT RGB y YUV
        # ---------------------------------
        curvas = libaux.CalculateRGBYUVCurves(img1, img1t)

        # insert elementos en los numpy arays de respuesta
        res_in_data = np.append(res_in_data, input_data)
        curvas = np.array(curvas)
        curvas = np.array(curvas / 255.0)

        res_out_data = np.append(res_out_data, curvas)  # RGB y YUV
        # res_out_data = np.append(res_out_data, curvas[0])  # RGB
        # res_out_data = np.append(res_out_data, curvas[1])  # YUV
        # res_out_data = np.array(res_out_data/255) # asi son valores 0..1

    # construccion de la respuesta compuesta de dos arrays numpy (x,y)
    res_in_data = np.asarray(res_in_data).astype('float32')
    res_in_data.shape = (index, 774)  # antes era 772
    res_out_data = np.asarray(res_out_data).astype('float32')
    res_out_data.shape = (index, 2 * 3 * 256)  # 1536
    ret = 0
    return ret, (res_in_data, res_out_data)


# ==========================================================================
if __name__ == "__main__":
    folder_input = "C:\\proyectos\\proyectos09\\SKYNNET\\FOTOS\\Boda Sabela y Javi Web Brutos\\"
    folder_input = "mini/bruto/"
    folder_output = "C:\\proyectos\\proyectos09\\SKYNNET\\FOTOS\\Boda Sabela y Javi Web\\"
    folder_output = "mini/trans/"
    brute_file_list = os.listdir(folder_input)
    retouched_file_list = os.listdir(folder_output)

    # concatenamos los directorios
    for indice in range(len(brute_file_list)):
        brute_file_list[indice] = folder_input + brute_file_list[indice]
        retouched_file_list[indice] = folder_output + retouched_file_list[indice]
    res, datos = dataGen(brute_file_list, retouched_file_list)
    print("datos[0].shape:", datos[0].shape)
    print("datos[1].shape:", datos[1].shape)
    print("--------INPUT-------------")
    print(datos[0])
    print("--------outPUT-------------")
    print(datos[1])
