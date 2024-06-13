import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
import math
import time
import sys

#__CLOUDBOOK:NONSHARED__
mnist = tf.keras.datasets.mnist
x_train = mnist.load_data()[0][0]
y_train = mnist.load_data()[0][1]
x_test = mnist.load_data()[1][0]
y_test = mnist.load_data()[1][1]

x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = np.where(y_train % 2 == 0,0,1)
y_test = np.where(y_test % 2 == 0,0,1)


#SKYNNET:BEGIN_BINARYCLASS_ACC_LOSS

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
model = [None, None]
to_predict_models = []
#__CLOUDBOOK:PARALLEL__
def skynnet_train_0(sk_i):
	global model
	global to_predict_models
	_DATA_TRAIN_X = x_train
	_DATA_TRAIN_Y = y_train
	_DATA_TEST_X = x_test
	_DATA_TEST_Y = y_test
	_NEURON_1 = 16
	_NEURON_2 = 8
	_NEURON_3 = 4
	_EPOCHS = 4
	datos_train_x_1 = _DATA_TRAIN_X[:len(_DATA_TRAIN_X) // 2]
	datos_train_x_2 = _DATA_TRAIN_X[len(_DATA_TRAIN_X) // 2:]
	datos_train_y_1 = _DATA_TRAIN_Y[:len(_DATA_TRAIN_Y) // 2]
	datos_train_y_2 = _DATA_TRAIN_Y[len(_DATA_TRAIN_Y) // 2:]
	if sk_i == 1:
		_DATA_TRAIN_X = datos_train_x_1
		_DATA_TRAIN_Y = datos_train_y_1
	else:
		_DATA_TRAIN_X = datos_train_x_2
		_DATA_TRAIN_Y = datos_train_y_2
	_NEURON_3 = 2
	inputs = tf.keras.Input(shape=(28, 28))
	x = tf.keras.layers.Flatten()(inputs)
	x = tf.keras.layers.Dense(_NEURON_1, activation='relu')(x)
	x = tf.keras.layers.Dense(_NEURON_2, activation='relu')(x)
	outputs = tf.keras.layers.Dense(_NEURON_3, activation='softmax')(x)
	model[sk_i] = tf.keras.Model(inputs=inputs, outputs=outputs)
	print(model[sk_i].summary())
	model[sk_i].compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	start = time.time()
	model[sk_i].fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, validation_split=0.3, epochs=_EPOCHS)
	end = time.time()
	print(' original: tiempo transcurrido (segundos) =', end - start)
	to_predict_models.append(sk_i)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_0():
	global predictions_0
	global to_predict_models
	global model
	_DATA_TEST_X = x_test
	_DATA_TEST_Y = y_test
	#__CLOUDBOOK:BEGINREMOVE__
	__CLOUDBOOK__ = {}
	__CLOUDBOOK__['agent'] = {}
	__CLOUDBOOK__['agent']['id'] = 'agente_skynnet'
	#__CLOUDBOOK:ENDREMOVE__
	#__CLOUDBOOK:LOCK__
	for sk_i in to_predict_models[:]:
		to_predict_models.remove(sk_i)
		label = __CLOUDBOOK__['agent']['id'] + ('_' + str(sk_i))
		predicted = model[sk_i].predict(_DATA_TEST_X, verbose=1)
		categorias = [0, 1]
		resul = predicted.tolist()
		predictions_0[label] = resul
	#__CLOUDBOOK:UNLOCK__


#SKYNNET:END

def main():
	pass


#__CLOUDBOOK:DU0__
def skynnet_train_global_0():
	for i in range(2):
		skynnet_train_0(i)
	#__CLOUDBOOK:SYNC__
#__CLOUDBOOK:DU0__
def skynnet_prediction_global_0():
	_DATA_TEST_X = x_test
	_DATA_TEST_Y = y_test
	for i in range(2):
		skynnet_prediction_0()
	#__CLOUDBOOK:SYNC__
	global predictions_0
	precision_compuesta = []
	for (idx, i) in enumerate(predictions_0):
		if idx == 0:
			p1 = predictions_0[i]
		elif idx == 1:
			p2 = predictions_0[i]
	p1 = np.array(p1)
	p2 = np.array(p2)
	p1 = p1.reshape((-1, 2))
	p2 = p2.reshape((-1, 2))
	predicted = np.zeros(0)
	for i in range(0, p1.shape[0]):
		a = abs(p1[i][0] - p1[i][1])
		b = abs(p2[i][0] - p2[i][1])
		c = a - b
		if c >= 0:
			predicted = np.append(predicted, p1[i])
		else:
			predicted = np.append(predicted, p2[i])
	predicted.shape = p1.shape
	correctas = 0
	total = 0
	for i in range(len(_DATA_TEST_Y)):
		if _DATA_TEST_Y[i] == np.argmax(predicted[i]):
			correctas += 1
		total += 1
	precision_compuesta.append(correctas / total)
	print('============================================')
	print('Skynnet Info: La accuracy de la prediccion compuesta es: ', precision_compuesta)
	print('============================================')
	if _DATA_TEST_Y[0].shape != ():
		#La salida tiene mas de una dimension
		cce = tf.keras.losses.CategoricalCrossentropy()
		cce_orig = cce(_DATA_TEST_Y, predicted).numpy()
		print('============================================')
		print('Skynnet Info: La loss compuesta es (cce): ', cce_orig)
		print('============================================')
	else:
		scce = tf.keras.losses.SparseCategoricalCrossentropy()
		scce_orig = scce(_DATA_TEST_Y, predicted).numpy()
		print('============================================')
		print('Skynnet Info: La loss compuesta es (scce): ', scce_orig)
		print('============================================')


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

