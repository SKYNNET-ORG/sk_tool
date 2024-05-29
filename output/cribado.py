import cv2
import numpy as np
import os
import re
import libaux
from numpy.linalg import norm

BN_LIMIT = 3
RGBYUV_LIMIT = 3.5
BRILLO_LIMIT = 38
BRILLO_LIMIT2 = 54
ZOOM_LIMIT = 240
regex = r'\b(?:-)?(\d+)\b'

dir_profile = "./profiles/juanjo"

DEBUG_MODE = True
if DEBUG_MODE:
    # raw_dir = (r'C:\Users\guerrero\VisualStudio\source\SKYNNET\CRIBADO\Raw')
    raw_dir = (r'C:\Users\guerrero\Desktop\Fotos\raw')
    # edited_dir = (r'C:\Users\guerrero\VisualStudio\source\SKYNNET\CRIBADO\Edited')
    edited_dir = (r'C:\Users\guerrero\Desktop\Fotos\edited')
else:
    raw_dir = (r'C:\Users\guerrero\VisualStudio\source\SKYNNET\CRIBADO\Boda Sabela y Javi Web Brutos')
    edited_dir = (r'C:\Users\guerrero\VisualStudio\source\SKYNNET\CRIBADO\Boda Sabela y Javi Web')


# ==========================================================================
# Elimina de la lista todos los archivos que no tienen la extensión deseada
def filter_extensions(list):
    allowed_extensions = ['.jpeg', '.jpg', '.gif']
    list = [item for item in list if any(ext in item.lower() for ext in allowed_extensions)]
    return list


# ==========================================================================
# Busca los elementos de la lista1 en la lista2 y elimina los que no tienen "pareja"
def process_list(list1, list2, unique_list):
    new_list1 = []

    for i in range(len(list1)):
        name1 = list1[i]
        num1 = [int(s) for s in re.findall(regex, name1)]
        match = False

        for j in range(len(list2)):
            name2 = list2[j]
            num2 = [int(s) for s in re.findall(regex, name2)]

            if num1 == num2:
                new_list1.append(name1)
                match = True
                break

        if not match:
            # new_unique = os.path.join(
            unique_list.append(name1)

    return new_list1


# ==========================================================================
def compare_folders(raw_folder, edited_folder):

    # Eliminamos de las listas los elementos con una extensión no deseada
    raw_list = filter_extensions(os.listdir(raw_folder))
    edited_list = filter_extensions(os.listdir(edited_folder))

    unique_list = []

    # Comparamos los elementos de ambas listas y quitamos los elementos únicos
    new_raw = process_list(raw_list, edited_list, unique_list)
    new_edited = process_list(edited_list, raw_list, unique_list)

    # Aquí ya tengo las dos listas que quiero y los directorios.
    for i, (raw_image, edited_image) in enumerate(zip(new_raw, new_edited)):
        new_raw[i] = os.path.join(raw_folder, raw_image)
        new_edited[i] = os.path.join(edited_folder, edited_image)

    # Verificar si la lista unique_list está vacía
    if unique_list:
        # Si la lista no está vacía, convertir la lista a una cadena utilizando la función join()
        str_unique_list = ", ".join(unique_list)  # Cadena con los elementos únicos en ambas listas
    return unique_list, new_raw, new_edited


# ==========================================================================
def is_color(image):
    b, g, r = cv2.split(image)
    if np.mean(np.abs(r - g)) >= BN_LIMIT or np.mean(np.abs(g - b)) >= BN_LIMIT:
        return True
    else:
        return False


# ==========================================================================
def aspect_ratio(image1, image2):
    # Obtener las dimensiones de las imágenes
    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape

    # Calcular las relaciones de aspecto
    aspect_ratio1 = width1 - height1  # Si es negativo, la foto es vertical
    # print(aspect_ratio1)
    aspect_ratio2 = width2 - height2  # Si es positivo, la foto es horizontal
    # print(aspect_ratio2)

    # Comparar las relaciones de aspecto
    if (aspect_ratio1 >= 0 and aspect_ratio2 >= 0) or (aspect_ratio1 < 0 and aspect_ratio2 < 0):  # Las dos fotos son horizontales
        return True
    else:
        return False


# ===================================================================================
def calcular_distancia(imgRaw, imgTrans):
    # Calculamos las lookup tablas RGB y YUV para llegar de la original a la transformada.
    lutables = libaux.CalculateRGBYUVCurves(imgRaw, imgTrans)

    # Generamos la transformada RGBYUV de la original
    imgRGBYUV = libaux.transformRGBYUV(imgRaw, lutables[0], lutables[1])

    # Calculamos la distancia entre la transformada y la transformada RGBYUV
    distance = libaux.CumulatedDistance(imgTrans, imgRGBYUV)
    return distance


# ===================================================================================
def calcular_brillo(image1, image2):
    f2 = libaux.CalculateRGBYUVCurves(image1, image2)
    image1t = libaux.transformRGBYUV(image1, f2[0], f2[1])
    image3 = cv2.addWeighted(image1t, 1, image2, -1, 0)
    promedio_brillo = np.average(norm(image3, axis=2))

    return promedio_brillo


# ===================================================================================
def calcular_zoom(image1, image2):
    img_original = libaux.imgResize(image1, 20)
    img_transformada = libaux.imgResize(image2, 20)

    # Inicializar el detector de puntos clave (SIFT o SURF)
    detector = cv2.SIFT_create()  # Puedes cambiarlo a SURF si lo deseas

    # Detectar los puntos clave y descriptores en ambas imágenes
    keypoints_original, descriptors_original = detector.detectAndCompute(img_original, None)
    keypoints_transformada, descriptors_transformada = detector.detectAndCompute(img_transformada, None)

    # Inicializar el matcher de correspondencia de puntos clave (Fuerza bruta)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    # Realizar la correspondencia de puntos clave entre las imágenes
    matches = matcher.match(descriptors_original, descriptors_transformada)

    # Calcular la distancia promedio entre los puntos clave correspondientes
    promedio_distancia = sum([match.distance for match in matches]) / len(matches)

    return promedio_distancia


# ===================================================================================
def criba(unique_list, raw_list, edited_list, profile_dir):

    criba_list = []

    new_raw_list = []
    new_raw_list += raw_list
    new_edited_list = []
    new_edited_list += edited_list

    for i, (raw_image, edited_image) in enumerate(zip(raw_list, edited_list)):
        # Cargar las imágenes en bruto y editada en cada iteración
        image1 = cv2.imread(raw_image)
        image2 = cv2.imread(edited_image)

        if aspect_ratio(image1, image2):    # Las imágenes tienen la misma relación de aspecto
            if is_color(image2):   # Las imágenes son a color
                # distRGBYUV = calcular_distancia(image1, image2)
                brillo = calcular_brillo(image1, image2)
                zoom = calcular_zoom(image1, image2)
                # Comprobamos si el brillo y el zoom superan el umbral
                if ((brillo >= BRILLO_LIMIT and zoom >= ZOOM_LIMIT) or brillo > BRILLO_LIMIT2):
                    criba_list.append(os.path.basename(edited_image))
                    new_raw_list.remove(raw_image)
                    new_edited_list.remove(edited_image)
            # Las imágenes son en blanco y negro
            else:
                criba_list.append(os.path.basename(edited_image))
                new_raw_list.remove(raw_image)
                new_edited_list.remove(edited_image)
        else:   # Las imágenes tienen relación de aspecto diferente
            criba_list.append(os.path.basename(edited_image))
            new_raw_list.remove(raw_image)
            new_edited_list.remove(edited_image)
    print(profile_dir)
    with open(profile_dir + "/criba.txt", "w") as file:
        for string in unique_list:
            file.write(string + "\n")
        for string in criba_list:
            file.write(string + "\n")

    return new_raw_list, new_edited_list

# ===================================================================================
if __name__ == "__main__":

    unique_list, raw_list, edited_list = compare_folders(raw_dir, edited_dir)
    criba(unique_list, raw_list, edited_list, dir_profile)
