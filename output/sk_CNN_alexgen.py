import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers,models
import time
import sys # para coger parametros de entrada

import cv2 # opencv para el reescalado y crop de imagenes
import math
import matplotlib.pyplot as plt
import os 



#__CLOUDBOOK:NONSHARED__
data = None
x_train = np.arange(0,102)
y_train = np.arange(0,102)

testx = None
testy = None

h2 = 128
w2 = 128
channels2 = 3

#__CLOUDBOOK:LOCAL__
def data_generator(batch_size, combinacion_arrays):
	categorias = os.listdir('./dataset128128')
	
	#print(categorias)
	url_fotos=[]
	#batch_size=32
	
	for x in categorias:
			if int(x) in combinacion_arrays:
				a=os.listdir('./dataset128128'+'/'+str(x))
				for url in a:
					url_fotos.append('./dataset128128'+'/'+str(x)+'/'+url)

	
	
	
	while True:
			np.random.shuffle(url_fotos)
			for start in range(0, len(url_fotos)-batch_size, batch_size):
				
				train_X=[]
				train_Y=[]
				for x in range(start,start+batch_size):
					
					imagen_cv2 = cv2.imread(url_fotos[x])
					imagen_numpy = np.array(imagen_cv2)
					train_X.append(imagen_numpy)
					train_Y.append(int(url_fotos[x].split('/')[2]))
					#print(imagen_numpy.shape())
					#print(foto_carpeta)
					#print(x%102)
				train_Y = np.searchsorted(combinacion_arrays, train_Y)
				train_X = np.array(train_X)
				train_X=train_X / 255.0
				train_Y=np.array(train_Y)
				yield train_X, train_Y

def load_test():
	global testy
	global testx
	categorias = os.listdir('./dataset128128')
	for x in categorias:
		a=os.listdir('./dataset128128'+'/'+str(x))[0]
		url_fotos='./dataset128128'+'/'+str(x)+'/'+str(a)
		imagen_cv2 = cv2.imread(url_fotos)
		imagen_numpy = np.array(imagen_cv2)
		testx.append(imagen_numpy)
		testy.append(int(url_fotos.split('/')[2]))
	testx = np.array(testx)



#SKYNNET:BEGIN_MULTICLASS_ACC_LOSS

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
model = [None, None, None]
to_predict_models = []
#__CLOUDBOOK:PARALLEL__
def skynnet_train_0(sk_i):
	global model
	global to_predict_models
	_DATA_TRAIN_GEN_Y = y_train
	_NEURON_1 = 2731
	_NEURON_2 = 68
	_FILTERS_1 = 256
	batch_size = 32
	grupos_de_categorias = dividir_array_categorias(_DATA_TRAIN_GEN_Y, 102, 3)
	combinacion_arrays = combinar_arrays(grupos_de_categorias)[sk_i]
	_DATA_TRAIN_GEN_Y = _DATA_TRAIN_GEN_Y[np.isin(_DATA_TRAIN_GEN_Y, combinacion_arrays)]
	print('======================================')
	print('Skynnet Info: Categorias de esta subred', np.unique(_DATA_TRAIN_GEN_Y))
	print('======================================')
	categorias_incluir = np.unique(_DATA_TRAIN_GEN_Y)
	etiquetas_consecutivas = np.arange(len(categorias_incluir))
	_NEURON_2 = len(combinacion_arrays)
	model[sk_i] = tf.keras.models.Sequential([layers.Conv2D(filters=int(_FILTERS_1 / 4), kernel_size=(11, 11), strides=4, activation='relu', input_shape=(h2, w2, channels2)), layers.MaxPooling2D((3, 3), strides=2), layers.Conv2D(filters=int(_FILTERS_1 / 1.5), kernel_size=(5, 5), strides=1, padding='same', activation='relu'), layers.MaxPooling2D((3, 3), strides=2), layers.Conv2D(filters=int(_FILTERS_1), kernel_size=(3, 3), strides=1, padding='same', activation='relu'), layers.Conv2D(filters=int(_FILTERS_1), kernel_size=(3, 3), strides=1, padding='same', activation='relu'), layers.Conv2D(filters=int(_FILTERS_1 / 1.5), kernel_size=(3, 3), strides=1, padding='same', activation='relu'), layers.MaxPooling2D((3, 3), strides=2), tf.keras.layers.Flatten(), tf.keras.layers.Dense(int(_NEURON_1), activation='relu'), tf.keras.layers.Dropout(rate=0.5), tf.keras.layers.Dense(int(_NEURON_1), activation='relu'), tf.keras.layers.Dropout(rate=0.5), tf.keras.layers.Dense(_NEURON_2, activation='softmax')])
	print(model[sk_i].summary())
	model[sk_i].compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	num_fotos = 0
	categorias = os.listdir('./dataset128128')
	for x in categorias:
		num_fotos += len(os.listdir('./dataset128128' + '/' + str(x)))
	steps_per_epoch = num_fotos // batch_size
	start = time.time()
	trained_model = model[sk_i].fit(data_generator(batch_size, _DATA_TRAIN_GEN_Y), steps_per_epoch=steps_per_epoch, epochs=1)
	end = time.time()
	print(' tiempo transcurrido (segundos) =', end - start)
	print('To predict model')
	print('Predicted model')
	to_predict_models.append(sk_i)

#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_0():
	pass



#SKYNNET:END

#__CLOUDBOOK:DU0__
def skynnet_train_global_0():
	for i in range(3):
		skynnet_train_0(i)
	#__CLOUDBOOK:SYNC__
#__CLOUDBOOK:DU0__
def skynnet_prediction_global_0():
	for i in range(3):
		skynnet_prediction_0()
	#__CLOUDBOOK:SYNC__
	#Error: There is no prediction in original code, make prediction=model.predict() in order to use it


#__CLOUDBOOK:MAIN__
def sk_main():
	try:
		main()
	except:
		pass
	skynnet_train_global_0()
	skynnet_prediction_global_0()

if __name__ == '__main__':
	sk_main()

