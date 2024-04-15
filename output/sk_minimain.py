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
import libaux
import numpy as np

from pathlib import Path
import re

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

#######################################################################
def gen_data_input_listbulk(dir_bruto, brute_file_list,retouched_file_list, profiledir, debuglevel, CRIBADAS=False):
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
	# si lista brute esta vacia, la construimos
	print("============================Enter in bulk")
	if (brute_file_list==None):
		brute_file_list=os.listdir(dir_bruto)
		#concatenamos los directorios
		for indice in range(len(brute_file_list)):
			brute_file_list[indice] = dir_bruto+brute_file_list[indice]
	# compatibilidad del debug level
	if (retouched_file_list==None):
		if (debuglevel==2 or debuglevel==4):
			print ("Tester: debuglevel incompatible con lista retouched a None")
			sys.exit()
				
	# retocamos una a una cada imagen
	index=0
	data_inputs=[]
	for file_1 in brute_file_list:
		# ahora hay que generar los datos de entrada (ojo no son de entrenamiento)
		print("leyendo imagen...", file_1, " y generando su data input")
		ret,x=dataTrainGen.dataGenImage(file_1)
		if (ret!=0):
			print("error accediendo a la imagen:", file_1)
			sys.exit()
		data_inputs.append(x)
		index=index+1

	print ("hemos terminado de leer y generar el array de input data")
	print();
	data_np=np.array(data_inputs)
	data_np.shape=(index,774)
	print("==================================Exit bulk")
	return (data_np,brute_file_list,retouched_file_list)

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

debuglevel = 4
data_np,brute_file_list,retouched_file_list = gen_data_input_listbulk(dir_bruto, brute_file_list,retouched_file_list, dir_profile, debuglevel, CRIBADAS=False)
# creacion y entrenamiento del modelo
# -------------------------------------
#SKYNNET:BEGIN_REGRESSION

#__CLOUDBOOK:LOCAL__
def dividir_array_categorias(array, n, m):
	#n = categorias iniciales
	#m = numero de arrays resultantes
	# Obtener las categorias unicas del array original
	categorias_unicas = np.unique(array)
	
	if n < m:
		raise ValueError(f"El numero de categorias original {n} debe ser mayor o igual al numero de arrays de destino {m}.")
	
	if m > len(categorias_unicas):
		raise ValueError(f"El numero de categorias unicas {len(categorias_unicas)} no es suficiente para dividirlas en los {m} arrays deseados.")
	
	# Calcular el numero de categorias en cada array de destino
	categorias_por_array = n // m    
	# Crear los m arrays de destino
	arrays_destino = []
	inicio_categoria = 0
	
	for i in range(m):
		fin_categoria = inicio_categoria + categorias_por_array
		
		if i < n % m:#
			#print(f"		Como {i} < n({n}) % m({m}), hacemos fin_categoria = {fin_categoria+1}")
			fin_categoria += 1
		
		categorias_array_actual = categorias_unicas[inicio_categoria:fin_categoria]
		arrays_destino.append(categorias_array_actual)
		inicio_categoria = fin_categoria
	return arrays_destino

#__CLOUDBOOK:LOCAL__
def combinar_arrays(arrays):
	from itertools import combinations
	if len(arrays) < 2:
		raise ValueError("Se requieren al menos dos arrays para realizar la combinacion.")
	
	combinaciones = list(combinations(arrays, 2))
	
	arrays_combinados = []
	
	for combo in combinaciones:
		array_1, array_2 = combo
		
		# Concatenar los dos arrays en uno solo
		array_combinado = np.concatenate((array_1, array_2))
		
		arrays_combinados.append(array_combinado)
	
	return arrays_combinados

#__CLOUDBOOK:GLOBAL__
predictions_0 = {}
#__CLOUDBOOK:NONSHARED__
model = [None, None, None, None]
to_predict_models = []
#__CLOUDBOOK:PARALLEL__
def skynnet_train_0(sk_i):
	global model
	global to_predict_models
	data_0 = data[0]
	data_1 = data[1]
	_DATA_TRAIN_X = data_0
	_DATA_TRAIN_Y = data_1
	_DATA_TEST_X = data_np
	_NEURON_1 = 194
	_NEURON_2 = 387
	_NEURON_3 = 384
	start = time.time()
	ENTRENAR = True
	if ENTRENAR:
		cad = dir_profile + '/' + 'model.h5'
		print('El modelo no existe.')
		dim = 256 * 3 + 6
		#Is not neccesary to divide data
		train_splits = np.array_split(_DATA_TRAIN_Y, 4, axis=1)
		_DATA_TRAIN_Y = train_splits[sk_i]
		#El tam de la ultima dimension
		_NEURON_3 = _DATA_TRAIN_Y.shape[-1]
		input_data = tf.keras.layers.Input(shape=(256 * 3 + 6,))
		encoded = tf.keras.layers.Dense(_NEURON_1, activation='relu6')(input_data)
		encoded = tf.keras.layers.Dense(_NEURON_2, activation='relu')(encoded)
		decoded = tf.keras.layers.Dense(_NEURON_3, activation='softplus')(encoded)
		model[sk_i] = tf.keras.models.Model(input_data, decoded)
		model[sk_i].compile(optimizer='adam', loss='mse')
		model_created_now = 1
		print('Modelo creado.')
		if model_created_now == 1:
			epocas = 200
		else:
			epocas = 25
		trained_model = model[sk_i].fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, epochs=epocas, batch_size=4)
		model[sk_i].save(dir_profile + '\\model.h5')
	end = time.time()
	train_time = end - start
	print(' train: tiempo transcurrido (segundos) =', train_time)
	print('')
	to_predict_models.append(sk_i)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_0():
	global predictions_0
	global to_predict_models
	global model
	_DATA_TEST_X = data_np
	#__CLOUDBOOK:BEGINREMOVE__
	__CLOUDBOOK__ = {}
	__CLOUDBOOK__['agent'] = {}
	__CLOUDBOOK__['agent']['id'] = 'agente_skynnet'
	#__CLOUDBOOK:ENDREMOVE__
	#__CLOUDBOOK:LOCK__
	for sk_i in to_predict_models[:]:
		to_predict_models.remove(sk_i)
		label = __CLOUDBOOK__['agent']['id'] + ('_' + str(sk_i))
		g = model[sk_i].predict(_DATA_TEST_X, verbose=1)
		resul = g.tolist()
		predictions_0[label] = resul
	#__CLOUDBOOK:UNLOCK__


#SKYNNET:END

#==============================================================


	










# ejecucion de prueba del modelo entrenado
# ------------------------------------------
'''debuglevel = 4
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
'''


#__CLOUDBOOK:DU0__
def skynnet_train_global_0():
	for i in range(4):
		skynnet_train_0(i)
	#__CLOUDBOOK:SYNC__
#__CLOUDBOOK:DU0__
def skynnet_prediction_global_0():
	_DATA_TEST_X = data_np
	for i in range(4):
		skynnet_prediction_0()
	#__CLOUDBOOK:SYNC__
	global predictions_0
	predictions_0 = dict(sorted(predictions_0.items(), key=lambda x: int(x[0].split('_')[-1])))
	g = np.concatenate(list(predictions_0.values()), axis=1)
	CRIBADAS = False
	index = 0
	for file_1 in brute_file_list:
		f = g[index]
		f = np.array(f * 255)
		f[0] = np.clip(f[0], a_min=0, a_max=255)
		f = np.asarray(f).astype('int')
		f.shape = (2, 3, 256)
		img1 = cv2.imread(file_1, cv2.IMREAD_COLOR)
		print(img1.shape)
		img1t = libaux.transformRGBYUV(img1, f[0], f[1])
		img1tr = libaux.imgResize(img1t, 50)
		directorio = dir_profile + '/RNA_images/'
		try:
			os.stat(directorio)
		except:
			os.mkdir(directorio)
		file_rna = tester.getRNAname(file_1)
		print('file:', file_rna)
		file_rna = dir_profile + '/RNA_images/' + file_rna
		if CRIBADAS == True:
			file_rna = dir_profile + '/RNA_images/cribada_' + tester.getRNAname(file_1)
		print('salvando en:', file_rna, ' con debug level', debuglevel)
		if debuglevel == 0:
			cv2.imwrite(file_rna, img1tr)
		elif debuglevel == 1:
			img1tr = libaux.imgResize(img1tr, 50)
			img1r = libaux.imgResize(img1, 25)
			cv2.putText(img1tr, 'RNA', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.putText(img1r, 'ORIGINAL', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			im_v = cv2.vconcat([img1r, img1tr])
			cv2.imwrite(file_rna, im_v)
		elif debuglevel == 2:
			img1tr = libaux.imgResize(img1tr, 50)
			img1r = libaux.imgResize(img1, 25)
			file_2 = retouched_file_list[index]
			img2 = cv2.imread(file_2, cv2.IMREAD_COLOR)
			img2r = libaux.imgResize(img2, 25)
			cv2.putText(img1r, 'ORIGINAL', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.putText(img2r, 'FOTOGRAFO', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.putText(img1tr, 'RNA', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			img_h_resize = concat_tile_resize([[img1r, img2r, img1tr]])
			cv2.imwrite(file_rna, img_h_resize)
		elif debuglevel == 4:
			img1tr = libaux.imgResize(img1tr, 50)
			img1r = libaux.imgResize(img1, 25)
			file_2 = retouched_file_list[index]
			print('file 2 es ', file_2)
			img2 = cv2.imread(file_2, cv2.IMREAD_COLOR)
			img2r = libaux.imgResize(img2, 25)
			cad = Path(file_2).stem
			numerito = [int(s) for s in re.findall('-?\\d+\\.?\\d*', cad)]
			numerito = abs(int(str(numerito[0])))
			yo = 0
			xo = 100
			cv2.rectangle(img1r, (xo, yo), (xo + 60, yo + 25), (0, 0, 0), -1)
			cv2.putText(img1r, str(numerito), (100, 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
			cv2.putText(img1r, 'ORIGINAL', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.putText(img2r, 'FOTOGRAFO', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.putText(img1tr, 'RNA', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			libaux.painthist(img1, 'ORIGINAL', 50, dir_profile + '/RNA_images/tmp1.jpg')
			libaux.painthist(img2, 'FOTOGRAFO', 50, dir_profile + '/RNA_images/tmp2.jpg')
			libaux.painthist(img1t, 'RNA', 50, dir_profile + '/RNA_images/tmp3.jpg')
			ancho = img1t.shape[1]
			hist1 = cv2.imread(dir_profile + '/RNA_images/tmp1.jpg', cv2.IMREAD_COLOR)
			hist2 = cv2.imread(dir_profile + '/RNA_images/tmp2.jpg', cv2.IMREAD_COLOR)
			hist3 = cv2.imread(dir_profile + '/RNA_images/tmp3.jpg', cv2.IMREAD_COLOR)
			img_h2 = concat_tile_resize([[img1r, img2r, img1tr], [hist1, hist2, hist3]])
			print('salvando ', file_rna)
			cv2.imwrite(file_rna, img_h2)
		index = index + 1


#__CLOUDBOOK:MAIN__
def main_sk():
	skynnet_train_global_0()
	skynnet_prediction_global_0()

if __name__ == '__main__':
	main()
	main_sk()

