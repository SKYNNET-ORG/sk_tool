"""
############################################################
# tester.py
#
# carga e instancia una red neuronal usando el modelo
# almacenado del perfil y procesa con ella las fotos
# especificadas en un directorio. Como salida produce fotos
# retocadas en otro directorio de salida especificado por el usuario
#
# funciones para invocar externamente:
#   testRetouch(dir_bruto, brute_file_list, retouched_file_list, profiledir, debuglevel)
#   retouchSinglePhoto(profiledir, dir_bruto, brute_file)
#
# funciones internas
#   ajustaf(f)
#   getRNAname(file_path)
#########################################################
"""
import cv2
import os
from pathlib import Path
import numpy as np
import re
import libaux
import lrcatparser
import dataTrainGen
import cribado
from numpy.linalg import norm
from tensorflow.keras.models import load_model


LOGDIST = False  # poner a false si no queremos log de distancias


def retouchSinglePhoto(profiledir, dir_bruto, brute_file):
    """
    retoca una foto individual

    """
    # carga el modelo
    model = libaux.loadModel(profiledir)
    if(model is None):
        return -2
    # concatena directorio
    brute_file = dir_bruto + brute_file
    ret, x = dataTrainGen.dataGenImage(brute_file)
    if(ret != 0):
        print("error accediendo a la imagen:", brute_file)
        return -1
    # ejecutamos la RNA para sacar la funcion de transformacion
    f = model.predict(x)
    f = np.array(f * 255)
    f[0] = np.clip(f[0], a_min=0, a_max=255)  # topar a 0..255
    f = np.asarray(f).astype('int')
    f.shape = (2, 3, 256)
    # ahora aplicamos la funcion de transformacion a la imagen
    img1 = cv2.imread(brute_file, cv2.IMREAD_COLOR)
    img1t = libaux.transformRGBYUV(img1, f[0], f[1])
    img1tr = libaux.imgResize(img1t, 50)  # reduce al 50%
    # crea el directorio de imagenes si no existe
    directorio = profiledir + "/RNA_images/"
    try:
        os.stat(directorio)
    except FileNotFoundError:
        # Si el directorio no existe, crea uno nuevo
        os.mkdir(directorio)
        print(f"Directorio '{directorio}' creado exitosamente.")
    except Exception as e:
        # Maneja otras excepciones de manera general
        print(f"Error al crear o verificar el directorio: {e}")
    # salvamos la imagen
    file_rna = getRNAname(brute_file)
    print("file:", file_rna)
    file_rna = profiledir + "/RNA_images/" + file_rna
    print("salvando en:", file_rna)
    cv2.imwrite(file_rna, img1tr)


# ===============================================================
def retouchLrcat(dir_bruto, lrcat, profiledir, debuglevel):

    # Si el lrcat no es new.lrcat, creamos new.lrcat y lo usamos
    if os.path.basename(lrcat) == 'new.lrcat':
        print("Editando ", os.path.basename(lrcat))
    else:
        print("Creando nuevo lrcat...")
        lrcat = lrcatparser.create(lrcat)

    # carga el modelo
    model = libaux.loadModel(profiledir)
    if(model is None):
        return -2
    # si lista brute esta vacia, la construimos
    brute_file_list = os.listdir(dir_bruto)
    # concatenamos los directorios
    for indice in range(len(brute_file_list)):
        brute_file_list[indice] = dir_bruto + '/' + brute_file_list[indice]

    # retocamos una a una cada imagen
    for file_1 in brute_file_list:
        # ahora hay que generar los datos de entrada (ojo no son de entrenamiento)
        ret, x = dataTrainGen.dataGenImage(file_1)
        if(ret != 0):
            print("error accediendo a la imagen:", file_1)
            return -1
        # ejecutamos la RNA para sacar la funcion de transformacion
        f = model.predict(x)
        f = np.array(f * 255)
        f[0] = np.clip(f[0], a_min=0, a_max=255)  # topar a 0..255
        f = np.asarray(f).astype('int')
        f.shape = (2, 3, 256)

        # Los valores U' están en f[1][1] V' están en f[1][2]
        yuv_colors = {
            'Red': (128, f[1][1][0], f[1][2][192]),
            'Green': (128, f[1][1][64], f[1][2][64]),
            'Blue': (128, f[1][1][192], f[1][2][64])
        }

        hsl_colors_iniciales = {
            'Red': (0, 240, 120),
            'Green': (80, 240, 120),
            'Blue': (160, 240, 120),
            'Aqua': (120, 240, 120),
            'Yellow': (40, 240, 120),
            'Orange': (26, 240, 120),
            'Purple': (200, 240, 60),
            'Magenta': (200, 240, 120)
        }

        # Convertir los colores YUV a RGB
        rgb_colors = {color: libaux.ycbcr_to_rgb(yuv) for color, yuv in yuv_colors.items()}
        for color, rgb in rgb_colors.items():
            print(f"RGB: {color}: {rgb}")

        # Convertir los colores RGB a hsl
        hsl_colors_finales = {color: libaux.rgb_to_hsl(hsl) for color, hsl in rgb_colors.items()}
        for color, hsl in hsl_colors_finales.items():
            print(f"HSL_final: {color}: {hsl}")

        hsl_colors = [(color, hsl_colors_iniciales[color] + hsl_colors_finales[color]) for color in hsl_colors_iniciales]
        for color, hsl in hsl_colors:
            print(f"HSL: {color}: {hsl}")

        # carga el modelo para obtener HUE y SAT en Lightroom
        model_color = load_model('modelo_color.h5')

        if(model_color is None):
            return -2

        # Crear listas para almacenar los resultados de la predicción
        resultados_controlH = []
        resultados_controlS = []

        # Iterar sobre la lista de colores y aplicar la predicción del modelo
        for color, valores in hsl_colors:

            # Predecir controlH y controlS para cada color
            predictions = model_color.predict([valores])[0]
            controlH, controlS = predictions
            print(f"Predictions: {predictions}")

            # Almacenar los resultados de la predicción
            resultados_controlH.append(round(controlH))
            resultados_controlS.append(round(controlS))

        """
        y' = 0.299R' + 0.587G' + 0.114B'
        f transforma desde y' a y''
        x transforma desde y a y''
        R = G = B = 28 --> y = 28
        R' = f[0][0][R]
        G' = f[0][1][G]
        B' = f[0][2][B]
        y' = 0.299xf[0][0][R] + ...
        y'' = f[1][0][0.299*f[0][0][28] + 0.587*f[0][1][28] + 0.114*f[0][2][28]]
        """

        updates = [
            ('ToneCurveName2012', "Custom"),
            ('ToneCurvePV2012', [
                0,
                f[1][0][0],
                28,
                f[1][0][28],
                56,
                f[1][0][56],
                85,
                f[1][0][85],
                113,
                f[1][0][113],
                141,
                f[1][0][141],
                170,
                f[1][0][170],
                198,
                f[1][0][198],
                227,
                f[1][0][227],
                255,
                f[1][0][255]]),
            ('ToneCurvePV2012Red', [
                0,
                f[0][2][0],
                28,
                f[0][2][28],
                56,
                f[0][2][56],
                85,
                f[0][2][85],
                113,
                f[0][2][113],
                141,
                f[0][2][141],
                170,
                f[0][2][170],
                198,
                f[0][2][198],
                227,
                f[0][2][227],
                255,
                f[0][2][255]]),
            ('ToneCurvePV2012Green', [
                0,
                f[0][1][0],
                28,
                f[0][1][28],
                56,
                f[0][1][56],
                85,
                f[0][1][85],
                113,
                f[0][1][113],
                141,
                f[0][1][141],
                170,
                f[0][1][170],
                198,
                f[0][1][198],
                227,
                f[0][1][227],
                255,
                f[0][1][255]]),
            ('ToneCurvePV2012Blue', [
                0,
                f[0][0][0],
                28,
                f[0][0][28],
                56,
                f[0][0][56],
                85,
                f[0][0][85],
                113,
                f[0][0][113],
                141,
                f[0][0][141],
                170,
                f[0][0][170],
                198,
                f[0][0][198],
                227,
                f[0][0][227],
                255,
                f[0][0][255]])
        ]

        # Imprimir los resultados de la predicción
        for color, controlH, controlS in zip(hsl_colors, resultados_controlH, resultados_controlS):
            print(f"Color: {color[0]}, ControlH: {controlH}, ControlS: {controlS}")
            # Agregar las actualizaciones para este color a la lista
            updates.append(('HueAdjustment' + color[0], controlH))
            updates.append(('SaturationAdjustment' + color[0], controlS))
        print(os.path.basename(file_1))

        lrcatparser.edit(lrcat, os.path.basename(file_1), updates)


# ===============================================================
def testRetouch(dir_bruto, dir_retouch, profiledir, debuglevel, CRIBADAS=False):
    """

    Retoca las fotos de un directorio o de una lista
    usando el modelo que encuentre en profiledir

    Parameters:

    - brutedir: directorio donde estan las fotos
    - dirbruto: dire de las fotos en bruto
    - lista_brute lista de fotos en bruto cribadas. if None se cogen todas las fotos de dirbruto
    - lista_retouch lista de fotos retocadas por fotografo cribadas
    - profiledir: dir de profile
    - debuglevel:
        debuglevel = 0, fotos retocadas por rna
        debuglevel = 1, fotos dobles (bruta y RNA)
        debuglevel = 2, fotos triples (bruta, fotografo, RNA). Solo valido si listas !=None
        debuglevel = 3, fotos dobles con histogramas (4 elementos en una foto). NO programado
        debuglevel = 4, fotos triples con histogramas (6 elementos en una foto)

    Returns:
      deja en profiledir/testRetouch/ las fotos retocadas reducidas al 50%

      0 : exito
      -1 : no encuentra directorio o algun fichero
      -2 : no encuentra el modelo
      -3 : debuglevel incompatible

    """
    if(LOGDIST):
        import csv
        file_distlog = open(profiledir + "/logdist.txt", 'w')
        writer = csv.writer(file_distlog)
        header = ['filename', 'dist RNA-fotografo', 'dist fotografo-RGBYUV', 'dist bruto-fotografo', 'dist bruto-rna', 'brillo']
        writer.writerow(header)

    # carga el modelo
    model = libaux.loadModel(profiledir)
    if model is None:
        return -2

    if dir_retouch is not None:
        # nos aseguramos que los elementos de las carpetas Raw y Edited "coinciden"
        unique_file_list, brute_file_list, retouched_file_list = cribado.compare_folders(dir_bruto, dir_retouch)

        # compatibilidad del debug level
        if(retouched_file_list is None):
            if(debuglevel == 2 or debuglevel == 4):
                print("Tester: debuglevel incompatible, se necesita directorio Edited")
                return -3
    else:
        brute_file_list = [dir_bruto + '/' + file for file in os.listdir(dir_bruto)]

    # retocamos una a una cada imagen
    index = 0
    for file_1 in brute_file_list:
        # ahora hay que generar los datos de entrada (ojo no son de entrenamiento)
        ret, x = dataTrainGen.dataGenImage(file_1)
        if(ret != 0):
            print("error accediendo a la imagen:", file_1)
            return -1
        # ejecutamos la RNA para sacar la funcion de transformacion
        f = model.predict(x)

        f = np.array(f * 255)

        f[0] = np.clip(f[0], a_min=0, a_max=255)  # topar a 0..255
        # f = ajustaf(f[0])

        f = np.asarray(f).astype('int')

        f.shape = (2, 3, 256)

        # ahora aplicamos la funcion de transformacion a la imagen
        img1 = cv2.imread(file_1, cv2.IMREAD_COLOR)
        print(img1.shape)
        img1t = libaux.transformRGBYUV(img1, f[0], f[1])

        # --------------------------------------------

        if(LOGDIST):
            # begin prueba de distancias
            file_2 = retouched_file_list[index]
            # img1 = cv2.imread(file_1, cv2.IMREAD_COLOR)  # orig
            img2 = cv2.imread(file_2, cv2.IMREAD_COLOR)  # trans
            f2 = libaux.CalculateRGBYUVCurves(img1, img2)
            # for i in range (0,256):
            #    print(f2[0][0][i],"-->",f[0][0][i])
            img2t = libaux.transformRGBYUV(img1, f2[0], f2[1])
            # cv2.imwrite(profiledir+"/RNA_images/"+getRNAname(file_1)+"_RGBYUV.jpg",img2t)
            # cv2.imwrite(profiledir+"/RNA_images/"+getRNAname(file_1)+"_RNA.jpg",img1t)

            brillo = np.average(norm(img2t, axis=2))  # / np.sqrt(3)

            d = libaux.CumulatedDistance(img2, img1t)
            print("distancia RNA-trans", d)  # nos dice si enseñamos bien
            d2 = libaux.CumulatedDistance(img2, img2t)
            print("distancia RGBYUV-trans", d2)  # nos dice si aprende bien
            d3 = libaux.CumulatedDistance(img1, img2)
            print("distancia bruto-trans", d3)  # nos dice cuanto cambia
            d4 = libaux.CumulatedDistance(img1, img1t)
            print("distancia bruto-rna", d4)
            print("distancia brillo", brillo)
            data = [file_2, int(d * 100), int(d2 * 100), int(d3 * 100), int(d4 * 100), int(brillo)]
            writer.writerow(data)
            file_distlog.flush()
            # end prueba de distancias

        # --------------------------------------------

        # img1t = libaux.transform(img1, f)
        img1tr = libaux.imgResize(img1t, 50)  # reduce al 50%

        # crea el directorio de imagenes si no existe
        directorio = profiledir + "/RNA_images/"
        try:
            os.stat(directorio)
        except FileNotFoundError:
            # Si el directorio no existe, crea uno nuevo
            os.mkdir(directorio)
            print(f"Directorio '{directorio}' creado exitosamente.")
        except Exception as e:
            # Maneja otras excepciones de manera general
            print(f"Error al crear o verificar el directorio: {e}")
        # salvamos la imagen
        file_rna = getRNAname(file_1)
        print("file:", file_rna)
        file_rna = profiledir + "/RNA_images/" + file_rna

        # truco para distinguir cribadas
        # --------------------------
        if(CRIBADAS is True):
            file_rna = profiledir + "/RNA_images/cribada_" + getRNAname(file_1)

        """
        if(os.path.exists(file_rna)):
            index = index+1
            continue
        else:
            #no existe
            file_rna = profiledir+"/RNA_images/cribada_"+getRNAname(file_1)
            img1tr = libaux.imgResize(img1tr,50)
            file_2 = retouched_file_list[index]
            img2 = cv2.imread(file_2, cv2.IMREAD_COLOR) # trans
            img2r = libaux.imgResize(img2,25)
            img1r = libaux.imgResize(img1,25)
            img_h_resize = concat_tile_resize([[img1r,img2r, img1tr]])
            cv2.imwrite(file_rna,img_h_resize) # es concatenar
            index = index+1
            continue
        """
        print("salvando en:", file_rna)
        if(debuglevel == 0):  # 0 = la foto retocada y reducida 50%
            cv2.imwrite(file_rna, img1tr)

        elif (debuglevel == 1):  # 1 = foto orig + RNA  reducidas al 25%

            img1tr = libaux.imgResize(img1tr, 50)  # reduce al 50% otra vez
            img1r = libaux.imgResize(img1, 25)
            # Escribir texto
            cv2.putText(img1tr, "RNA", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img1r, "ORIGINAL", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            im_v = cv2.vconcat([img1r, img1tr])
            cv2.imwrite(file_rna, im_v)

        elif(debuglevel == 2):  # 2 = foto orig + fotografo + RNA  reducidas al 25%
            img1tr = libaux.imgResize(img1tr, 50)  # reduce al 50% otra vez
            img1r = libaux.imgResize(img1, 25)
            file_2 = retouched_file_list[index]
            img2 = cv2.imread(file_2, cv2.IMREAD_COLOR)
            img2r = libaux.imgResize(img2, 25)
            # Escribir texto
            cv2.putText(img1r, "ORIGINAL", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img2r, "FOTOGRAFO", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img1tr, "RNA", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # collage
            # im_v = cv2.hconcat([img1r,img2r, img1tr])
            img_h_resize = concat_tile_resize([[img1r, img2r, img1tr]])
            # cv2.imwrite(file_rna, im_v)
            cv2.imwrite(file_rna, img_h_resize)

        elif (debuglevel == 3):  # 3 = foto orig + RNA  + 2 histogramas reducidas al 25%
            img1tr = libaux.imgResize(img1tr, 50)  # reduce al 50% otra vez
            img1r = libaux.imgResize(img1, 25)
            file_2 = brute_file_list[index]
            # Escribir texto
            cad = Path(file_2).stem
            numerito = [int(s) for s in re.findall(r'-?\d+\.?\d*', cad)]
            numerito = abs(int(str(numerito[0])))
            yo = 0
            xo = 100
            cv2.rectangle(img1r, (xo, yo), (xo + 60, yo + 25), (0, 0, 0), -1)
            cv2.putText(img1r, str(numerito), (100, 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(img1r, "ORIGINAL", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img1tr, "RNA", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # collage
            # img_h = concat_tile_resize([[img1r,img2r, img1tr]])
            # im_h = cv2.hconcat([img1r,img2r, img1tr])

            # histogramas
            libaux.painthist(img1, "ORIGINAL", 50, profiledir + "/RNA_images/tmp1.jpg")
            libaux.painthist(img1t, "RNA", 50, profiledir + "/RNA_images/tmp3.jpg")
            hist1 = cv2.imread(profiledir + "/RNA_images/tmp1.jpg", cv2.IMREAD_COLOR)
            hist3 = cv2.imread(profiledir + "/RNA_images/tmp3.jpg", cv2.IMREAD_COLOR)
            # collage
            img_h2 = concat_tile_resize([[img1r, img1tr], [hist1, hist3]])
            # im_h2 = cv2.hconcat([hist1, hist2, hist3])
            # ancho = img1t.shape[1]
            # ancho_h1 = im_h.shape[1]
            # factor = im_h.shape[1]/im_h2.shape[1]
            # im_h2r = cv2.resize(im_h2,(int(im_h.shape[1]), int(factor*im_h2.shape[0])))
            # print("h1:",ancho_h1, "ancho2:",im_h2r.shape[1])
            # im_v = cv2.vconcat([im_h, im_h2r])

            # cv2.imwrite(file_rna, im_v)
            # cv2.putText(file_rna, file_2, (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255, 255, 255), 2)
            cv2.imwrite(file_rna, img_h2)

        elif (debuglevel == 4):  # 4 = foto orig + fotografo + RNA + 3 histogramas reducidas al 25%
            img1tr = libaux.imgResize(img1tr, 50)  # reduce al 50% otra vez
            img1r = libaux.imgResize(img1, 25)
            file_2 = retouched_file_list[index]
            img2 = cv2.imread(file_2, cv2.IMREAD_COLOR)
            img2r = libaux.imgResize(img2, 25)
            # Escribir texto
            cad = Path(file_2).stem
            numerito = [int(s) for s in re.findall(r'-?\d+\.?\d*', cad)]
            numerito = abs(int(str(numerito[0])))
            yo = 0
            xo = 100
            cv2.rectangle(img1r, (xo, yo), (xo + 60, yo + 25), (0, 0, 0), -1)
            cv2.putText(img1r, str(numerito), (100, 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img1r, "ORIGINAL", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img2r, "FOTOGRAFO", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img1tr, "RNA", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # collage
            # img_h = concat_tile_resize([[img1r,img2r, img1tr]])
            # im_h = cv2.hconcat([img1r,img2r, img1tr])

            # histogramas
            libaux.painthist(img1, "ORIGINAL", 50, profiledir + "/RNA_images/tmp1.jpg")
            libaux.painthist(img2, "FOTOGRAFO", 50, profiledir + "/RNA_images/tmp2.jpg")
            libaux.painthist(img1t, "RNA", 50, profiledir + "/RNA_images/tmp3.jpg")
            hist1 = cv2.imread(profiledir + "/RNA_images/tmp1.jpg", cv2.IMREAD_COLOR)
            hist2 = cv2.imread(profiledir + "/RNA_images/tmp2.jpg", cv2.IMREAD_COLOR)
            hist3 = cv2.imread(profiledir + "/RNA_images/tmp3.jpg", cv2.IMREAD_COLOR)
            # collage
            img_h2 = concat_tile_resize([[img1r, img2r, img1tr], [hist1, hist2, hist3]])
            # ancho = img1t.shape[1]
            # im_h2 = cv2.hconcat([hist1,hist2, hist3])
            # ancho_h1 = im_h.shape[1]
            # factor = im_h.shape[1]/im_h2.shape[1]
            # im_h2r = cv2.resize(im_h2,(int(im_h.shape[1]), int(factor*im_h2.shape[0])))
            # print("h1:",ancho_h1, "ancho2:",im_h2r.shape[1])
            # im_v = cv2.vconcat([im_h, im_h2r])

            # cv2.imwrite(file_rna,im_v)
            # cv2.putText(file_rna, file_2, (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,255,255), 2)
            cv2.imwrite(file_rna, img_h2)
        index = index + 1
    return 0


# ===============================================================
def getRNAname(file_path):
    """
    esta funcion genera un nombre de fichero
    a partir de un path completo de fichero de imagen original

    se queda sin directorio y le quita la extension sea cual sea
    y la cambia por <nombre>_rna.jpg

    Parameters
    - file_path: nombre del filepath completo

    Returns:
    string con el nombre de fichero de salida de RNA
    no es path completo, solo file name
    """
    cad = Path(file_path).stem  # le quita la extension y el directorio
    """
    print("cad:",cad)
    index = cad.rfind('/')
    print("index:", index)
    if(index==-1):
        index = cad.rfind('\\')

    cad2 = cad[index:]
    print("cad2:",cad2)
    """
    cad2 = cad + "_rna.jpg"
    return cad2


# ===============================================================
def ajustaf(f):
    """

    ajusta una funcion de transformacion por si algun valor
    es mayor 255 o menor que cero. Simplemente topa

    Parameters:
    - f : funcion de transformacion de 768 elementos

    Returns:
    g: funcion topada

    """
    print("fshape:", f.shape)
    for i in range(0, f.shape[0]):
        f[i] = min(255, max(f[i], 0))
    return f


# ===============================================================
# define a function for horizontally
# concatenating images of different
# heights
def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum hights
    h_min = min(img.shape[0]
                for img in img_list)

    # image resizing
    im_list_resize = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min),
                                 interpolation=interpolation) for img in img_list]
    # return final image
    return cv2.hconcat(im_list_resize)


# ===============================================================
# define a function for vertically
# concatenating images of different
# widths
def vconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum width
    w_min = min(img.shape[1] for img in img_list)

    # resizing images
    im_list_resize = [cv2.resize(img, (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation=interpolation) for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)


# ===============================================================
# define a function for concatenating
# images of different sizes in
# vertical and horizontal tiles
def concat_tile_resize(list_2d, interpolation=cv2.INTER_CUBIC):
    # function calling for every
    # list of images
    img_list_v = [hconcat_resize(list_h, interpolation=cv2.INTER_CUBIC)
                  for list_h in list_2d]

    # return final image
    return vconcat_resize(img_list_v, interpolation=cv2.INTER_CUBIC)


# ===============================================================
# Definir una función para normalizar los valores
def normalizar(valor):
    # Normalizar el valor al rango [-100, 100]
    return (valor - 0.5) * 200


# ===============================================================
def testRetouch_list(dir_bruto, brute_file_list, retouched_file_list, profiledir, debuglevel, CRIBADAS=False):
    """

    Retoca las fotos de un directorio o de una lista
    usando el modelo que encuentre en profiledir

    Parameters:

    - dirbruto: dire de las fotos en bruto
    - lista_brute lista de fotos en bruto cribadas. if None se cogen todas las fotos de dirbruto
    - lista_retouch lista de fotos retocadas por fotografo cribadas
    - profiledir: dir de profile
    - debuglevel:
        debuglevel=0 fotos retocadas por rna
        debuglevel=1 fotos dobles (bruta y RNA)
        debuglevel=2 fotos triples (bruta, fotografo, RNA). Solo valido si listas !=None
        debuglevel=3 fotos dobles con histogramas (4 elementos en una foto). NO programado
        debuglevel=4 fotos triplesretocadas y con histogramas (6 elementos en una foto)

    Returns:
      deja en profiledir/testRetouch/ las fotos retocadas reducidas al 50%

      0 : exito
      -1 : no encuentra directorio o algun fichero
      -2 : no encuentra el modelo
      -3 : debuglevel incompatible

    """
    if (LOGDIST):
        import csv
        file_distlog = open(profiledir + "/logdist.txt", 'w')
        writer = csv.writer(file_distlog)
        header = ['filename', 'dist RNA-fotografo', 'dist fotografo-RGBYUV', 'dist bruto-fotografo', 'dist bruto-rna', 'brillo']
        writer.writerow(header)

    # carga el modelo
    model = libaux.loadModel(profiledir)
    if (model is None):
        return -2
    # si lista brute esta vacia, la construimos
    if (brute_file_list is None):
        brute_file_list = os.listdir(dir_bruto)
        # concatenamos los directorios
        for indice in range(len(brute_file_list)):
            brute_file_list[indice] = dir_bruto + brute_file_list[indice]
    # compatibilidad del debug level
    if (retouched_file_list is None):
        if (debuglevel == 2 or debuglevel == 4):
            print("Tester: debuglevel incompatible con lista retouched a None")
            return -3

    # retocamos una a una cada imagen
    index = 0
    for file_1 in brute_file_list:
        # ahora hay que generar los datos de entrada (ojo no son de entrenamiento)
        ret, x = dataTrainGen.dataGenImage(file_1)
        if (ret != 0):
            print("error accediendo a la imagen:", file_1)
            return -1
        # ejecutamos la RNA para sacar la funcion de transformacion
        f = model.predict(x)

        f = np.array(f * 255)

        f[0] = np.clip(f[0], a_min=0, a_max=255)  # topar a 0..255
        # f = ajustaf(f[0])

        f = np.asarray(f).astype('int')

        f.shape = (2, 3, 256)

        # ahora aplicamos la funcion de transformacion a la imagen
        img1 = cv2.imread(file_1, cv2.IMREAD_COLOR)
        print(img1.shape)
        img1t = libaux.transformRGBYUV(img1, f[0], f[1])

        # --------------------------------------------

        if (LOGDIST):
            # begin prueba de distancias
            file_2 = retouched_file_list[index]
            # img1 = cv2.imread(file_1, cv2.IMREAD_COLOR) # orig
            img2 = cv2.imread(file_2, cv2.IMREAD_COLOR)  # trans
            f2 = libaux.CalculateRGBYUVCurves(img1, img2)
            # for i in range (0, 256):
            #    print(f2[0][0][i],"-->",f[0][0][i])
            img2t = libaux.transformRGBYUV(img1, f2[0], f2[1])
            # cv2.imwrite(profiledir+"/RNA_images/"+getRNAname(file_1)+"_RGBYUV.jpg",img2t)
            # cv2.imwrite(profiledir+"/RNA_images/"+getRNAname(file_1)+"_RNA.jpg",img1t)

            brillo = np.average(norm(img2t, axis=2))  # /np.sqrt(3)

            d = libaux.CumulatedDistance(img2, img1t)
            print("distancia RNA-trans", d)  # nos dice si enseñamos bien
            d2 = libaux.CumulatedDistance(img2, img2t)
            print("distancia RGBYUV-trans", d2)  # nos dice si aprende bien
            d3 = libaux.CumulatedDistance(img1, img2)
            print("distancia bruto-trans", d3)  # nos dice cuanto cambia
            d4 = libaux.CumulatedDistance(img1, img1t)
            print("distancia bruto-rna", d4)
            print("distancia brillo", brillo)
            data = [file_2, int(d * 100), int(d2 * 100), int(d3 * 100), int(d4 * 100), int(brillo)]
            writer.writerow(data)
            file_distlog.flush()

        # --------------------------------------------

        # img1t = libaux.transform(img1, f)
        img1tr = libaux.imgResize(img1t, 50)  # reduce al 50%

        # crea el directorio de imagenes si no existe
        directorio = profiledir + "/RNA_images/"
        try:
            os.stat(directorio)
        except:
            os.mkdir(directorio)
        # salvamos la imagen
        file_rna = getRNAname(file_1)
        print("file:", file_rna)
        file_rna = profiledir + "/RNA_images/" + file_rna

        # truco para distinguir cribadas
        # --------------------------
        if (CRIBADAS is True):
            file_rna = profiledir + "/RNA_images/cribada_" + getRNAname(file_1)

        print("salvando en:", file_rna, " con debug level", debuglevel)
        if (debuglevel == 0):  # 0= la foto retocada y reducida 50%
            cv2.imwrite(file_rna, img1tr)

        elif (debuglevel == 1):  # 1= foto orig + RNA  reducidas al 25%

            img1tr = libaux.imgResize(img1tr, 50)  # reduce al 50% otra vez
            img1r = libaux.imgResize(img1, 25)
            # Escribir texto
            cv2.putText(img1tr, "RNA", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img1r, "ORIGINAL", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            im_v = cv2.vconcat([img1r, img1tr])
            cv2.imwrite(file_rna, im_v)

        elif (debuglevel == 2):  # 2= foto orig + fotografo +RNA  reducidas al 25%
            img1tr = libaux.imgResize(img1tr, 50)  # reduce al 50% otra vez
            img1r = libaux.imgResize(img1, 25)
            file_2 = retouched_file_list[index]
            img2 = cv2.imread(file_2, cv2.IMREAD_COLOR)
            img2r = libaux.imgResize(img2, 25)
            # Escribir texto
            cv2.putText(img1r, "ORIGINAL", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img2r, "FOTOGRAFO", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img1tr, "RNA", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # collage
            # im_v = cv2.hconcat([img1r, img2r, img1tr])
            img_h_resize = concat_tile_resize([[img1r, img2r, img1tr]])
            # cv2.imwrite(file_rna, im_v)
            cv2.imwrite(file_rna, img_h_resize)

        elif (debuglevel == 4):  # 4= foto orig + fotografo +RNA  + 3 histogramas reducidas al 25%
            img1tr = libaux.imgResize(img1tr, 50)  # reduce al 50% otra vez
            img1r = libaux.imgResize(img1, 25)
            file_2 = retouched_file_list[index]
            print("file 2 es ", file_2)
            img2 = cv2.imread(file_2, cv2.IMREAD_COLOR)
            img2r = libaux.imgResize(img2, 25)
            # Escribir texto
            cad = Path(file_2).stem
            numerito = [int(s) for s in re.findall(r'-?\d+\.?\d*', cad)]
            numerito = abs(int(str(numerito[0])))
            yo = 0
            xo = 100
            cv2.rectangle(img1r, (xo, yo), (xo + 60, yo + 25), (0, 0, 0), -1)
            cv2.putText(img1r, str(numerito), (100, 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img1r, "ORIGINAL", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img2r, "FOTOGRAFO", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img1tr, "RNA", (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # collage
            # img_h = concat_tile_resize([[img1r, img2r, img1tr]])
            # im_h = cv2.hconcat([img1r, img2r, img1tr])

            # histogramas
            libaux.painthist(img1, "ORIGINAL", 50, profiledir + "/RNA_images/tmp1.jpg")
            libaux.painthist(img2, "FOTOGRAFO", 50, profiledir + "/RNA_images/tmp2.jpg")
            libaux.painthist(img1t, "RNA", 50, profiledir + "/RNA_images/tmp3.jpg")
            # ancho = img1t.shape[1]
            hist1 = cv2.imread(profiledir + "/RNA_images/tmp1.jpg", cv2.IMREAD_COLOR)
            hist2 = cv2.imread(profiledir + "/RNA_images/tmp2.jpg", cv2.IMREAD_COLOR)
            hist3 = cv2.imread(profiledir + "/RNA_images/tmp3.jpg", cv2.IMREAD_COLOR)
            # collage
            img_h2 = concat_tile_resize([[img1r, img2r, img1tr], [hist1, hist2, hist3]])
            # im_h2 = cv2.hconcat([hist1, hist2, hist3])
            # ancho_h1 = im_h.shape[1]
            # factor = im_h.shape[1]/im_h2.shape[1]
            # im_h2r = cv2.resize(im_h2,(int(im_h.shape[1]), int(factor*im_h2.shape[0])))
            # print("h1:",ancho_h1, "ancho2:",im_h2r.shape[1])
            # im_v = cv2.vconcat([im_h, im_h2r])
            # cv2.imwrite(file_rna, im_v)
            # cv2.putText(file_rna, file_2, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255, 255, 255), 2)
            print("salvando ", file_rna)
            cv2.imwrite(file_rna, img_h2)
        index = index + 1
    return 0
