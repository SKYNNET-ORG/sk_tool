import cv2
import os
import time
import sys

# modulos de la app
import dataTrainGen
import trainer
import tester
import cribado
import tensorflow as tf

import keras
print(keras.__version__)


########################################################################################
# define a function for vertically
# concatenating images of different
# widths
def vconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum width
    w_min = min(img.shape[1] for img in img_list)
    # resizing images
    im_list_resize = [cv2.resize(img,
                                 (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation=interpolation) for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)


########################################################################################
def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum hights
    h_min = min(img.shape[0] for img in img_list)
    # image resizing
    im_list_resize = [cv2.resize(img,
                                 (int(img.shape[1] * h_min / img.shape[0]),
                                  h_min), interpolation=interpolation) for img in img_list]
    # return final image
    return cv2.hconcat(im_list_resize)


########################################################################################
def concat_tile_resize(list_2d, interpolation=cv2.INTER_CUBIC):
    # function calling for every
    # list of images
    img_list_v = [hconcat_resize(list_h, interpolation=cv2.INTER_CUBIC) for list_h in list_2d]
    # return final image
    return vconcat_resize(img_list_v, interpolation=cv2.INTER_CUBIC)

########################################################################################
#                    PROGRAMA MAIN
########################################################################################
ENTRENAR = True

# variables globales
# ------------------
dir_profile = "./mini"

dir_bruto = "mini/bruto/"
dir_retouch = "mini/trans/"

# ===================================COMIENZA MAIN =====================================
# obtencion de lista de fotos sin cribar
# --------------------------------------
brute_file_list = os.listdir(dir_bruto)
retouched_file_list = os.listdir(dir_retouch)

total_fotos = len(brute_file_list)
print("total fotos en bruto:", total_fotos)

# cribado
# ---------
start = time.time()
print("cribando fotos para entrenar en dir bruto=", dir_bruto)
unique_list, brute_file_list, retouched_file_list = cribado.compare_folders(dir_bruto, dir_retouch)
brute_file_list, retouched_file_list = cribado.criba(unique_list, brute_file_list, retouched_file_list, dir_profile)
print("cribando terminado")

end = time.time()
criba_time = end - start
print(" criba: tiempo transcurrido (segundos) =", criba_time)

# generacion de datos de entrenamiento
# --------------------------------------
total_fotos_cribadas = len(brute_file_list)
start = time.time()
if (ENTRENAR):
    res, data = dataTrainGen.dataGen(brute_file_list, retouched_file_list)  # genera data para entrenar
else:
    res, data = (0, 0)  # caso de no entrenar
end = time.time()
data_gen_time = end - start
print(" datagen: tiempo transcurrido (segundos) =", data_gen_time)
if (res != 0):
    sys.exit()

print("dataTrainGen.dataGen:", res)

# creacion y entrenamiento del modelo
# -------------------------------------
start = time.time()
if (ENTRENAR):
    # res = trainer.trainProfile(dir_profile, data)  # esto genera el modelo h5
    ################################################################################################
    # hacer comprobación de si existe el modelo previamente
    # model = libaux.loadModel(profiledir)
    cad = dir_profile+"/model.h5"
    model = tf.keras.models.load_model(cad)
    model.compile(optimizer='adam', loss='mse')
    model_created_now = 0
    if(model is None):
        print("El modelo no existe")
        ############################################################################################
        dim = 256 * 3 + 6  # 4
        # dim = 64
        input_data = tf.keras.layers.Input(shape=(256 * 3 + 6,))  # antes 4
        encoded = tf.keras.layers.Dense(dim, activation='relu6')(input_data)
        encoded = tf.keras.layers.Dense(dim * 2, activation='relu')(encoded)  # mejor que selu
        decoded = tf.keras.layers.Dense(256 * 3 * 2, activation='softplus')(encoded)
        autoencoder = tf.keras.models.Model(input_data, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        model_created_now = 1
        ############################################################################################
    print(model.summary())
    # para usar menos epocas
    inputs = data[0]
    desired_output = data[1]
    if(model_created_now == 1):
        epocas = 200
    else:
        epocas = 25
    # log_train = trainModel(model, (inputs, desired_output), epocas)
    trained_model = model.fit(data[0], data[1], epochs=epocas, batch_size=4)
    # salva el modelo en el directorio de profile
    # libaux.saveModel(model, profiledir)
    model.save(dir_profile + '\\model.h5')
    ################################################################################################
else:
    res = None
print("trainer.trainProfile:", res)
end = time.time()
train_time = end - start
print(" train: tiempo transcurrido (segundos) =", train_time)
print("")

# ejecucion de prueba del modelo entrenado
# ------------------------------------------
debuglevel = 4
# 0= la foto retocada y reducida 50%
# 1= foto orig + RNA  reducidas al 25%
# 2= foto orig + fotografo +RNA  reducidas al 25%
# 3= esta opcion no existe
# 4= foto orig + fotografo +RNA  + 3 histogramas reducidas al 25%

start = time.time()
# esta funcion retoca las fotos que esten en la lista brute_file_list, la cual no contiene cribadas.
res = tester.testRetouch_list(dir_bruto, brute_file_list, retouched_file_list, dir_profile, debuglevel, False)
end = time.time()
print("Ahora hacemos las cribadas...")
print("------------------------------")

# ahora hacemos las cribadas
# --------------------------
set_bruto_cribada = set(brute_file_list)
set_retouched_cribada = set(retouched_file_list)

brute_full_list = os.listdir(dir_bruto)
retouched_full_list = os.listdir(dir_retouch)

# concatenamos los directorios
for indice in range(len(brute_full_list)):
    brute_full_list[indice] = dir_bruto + brute_full_list[indice]
    retouched_full_list[indice] = dir_retouch + retouched_full_list[indice]

set_brute_full = set(brute_full_list)
set_retouched_full = set(retouched_full_list)

set_brute_dif = set_brute_full - set_bruto_cribada  # set_bruto_cribada es la lista cribada (no las cribadas)
set_retouched_dif = set_retouched_full - set_retouched_cribada

list_bruto_cribadas = list(set_brute_dif)
list_retouched_cribadas = list(set_retouched_dif)

# aqui falta quitar de retouched las que no esten en list bruto
# ------------------------------------------------------
lista_cribadas = []
lista_cribadas_retouch = []
for i in list_bruto_cribadas:
    for j in list_retouched_cribadas:
        if (dataTrainGen.checkCoherence(i, j) == 0):
            lista_cribadas.append(i)
            lista_cribadas_retouch.append(j)

list_bruto_cribadas = lista_cribadas
list_retouched_cribadas = lista_cribadas_retouch
# ------------------------------------------------------

list_bruto_cribadas.sort()
list_retouched_cribadas.sort()
print("-----------BRUTO CRIBADAS")
print(list_bruto_cribadas)

print("-------------RETOUCH CRIBADAS")
print(list_retouched_cribadas)
print()

# ejecucion de prueba del modelo entrenado con fotos cribadas
# -------------------------------------------------------------
res = tester.testRetouch_list(dir_bruto, list_bruto_cribadas, list_retouched_cribadas, dir_profile, debuglevel, True)

test_time = end - start
print(" test: tiempo transcurrido (segundos) =", test_time)
print("tester.testRetouch:", res)


# reporte de estadisticas
# -------------------------------------------------------------
print("====================================")
print("    PERFORMANCE SUMMARY:         ")
print("===================================")
print("Fase de criba : ")
print("  -tiempo cribado : ", criba_time)
print("  -total fotos: ", total_fotos)
print("  -tiempocriba por foto: ", criba_time / total_fotos)
print("------------------------------------")
print("Fase de data generation : ")
print("  -tiempo data gen : ", data_gen_time)
print("  -total fotos: ", total_fotos_cribadas)
print("  -tiempo datagen por foto: ", data_gen_time / total_fotos_cribadas)
print("------------------------------------")
print("Fase de training : ")
print("  -tiempo training (100 epoch) : ", train_time)
print("  -total fotos: ", total_fotos_cribadas)
print("  -tiempo training por foto: ", train_time / total_fotos_cribadas)
print("  -tiempo training por epoca: ", train_time / 100)
print("------------------------------------")
print("Fase de test : ")
print("  -tiempo test  : ", test_time)
print("  -total fotos: ", total_fotos_cribadas)
print("  -tiempo test por foto: ", test_time / total_fotos_cribadas)
print("------------------------------------")

