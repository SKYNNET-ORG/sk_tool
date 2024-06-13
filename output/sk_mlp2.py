import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
import time

#__CLOUDBOOK:NONSHARED__
mnist = tf.keras.datasets.mnist
x_train = mnist.load_data()[0][0]
y_train = mnist.load_data()[0][1]
x_test = mnist.load_data()[1][0]
y_test = mnist.load_data()[1][1]

#Noramalize the pixel values by deviding each pixel by 255
x_train = x_train / 255.0
x_test =  x_test / 255.0


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
	_DATA_TRAIN_X = x_train
	_DATA_TRAIN_Y = y_train
	_DATA_TEST_X = x_test
	_DATA_TEST_Y = y_test
	_NEURON_1 = 86
	_NEURON_2 = 40
	_NEURON_3 = 7
	_EPOCHS = 7
	grupos_de_categorias = dividir_array_categorias(_DATA_TRAIN_Y, 10, 3)
	combinacion_arrays = combinar_arrays(grupos_de_categorias)[sk_i]
	_DATA_TRAIN_X = _DATA_TRAIN_X[np.isin(_DATA_TRAIN_Y, combinacion_arrays)]
	_DATA_TRAIN_Y = _DATA_TRAIN_Y[np.isin(_DATA_TRAIN_Y, combinacion_arrays)]
	print('======================================')
	print('Skynnet Info: Longitud de los datos de la subred (datos,etiquetas):', len(_DATA_TRAIN_X), len(_DATA_TRAIN_Y))
	print('Skynnet Info: Categorias de esta subred', np.unique(_DATA_TRAIN_Y))
	print('======================================')
	categorias_incluir = np.unique(_DATA_TRAIN_Y)
	etiquetas_consecutivas = np.arange(len(categorias_incluir))
	_DATA_TRAIN_Y = np.searchsorted(categorias_incluir, _DATA_TRAIN_Y)
	_NEURON_3 = len(np.unique(_DATA_TRAIN_Y))
	inputs = tf.keras.Input(shape=(28, 28))
	x = tf.keras.layers.Flatten()(inputs)
	x = tf.keras.layers.Dense(_NEURON_1, activation='relu')(x)
	x = tf.keras.layers.Dense(_NEURON_2, activation='relu')(x)
	outputs = tf.keras.layers.Dense(_NEURON_3, activation='softmax')(x)
	model[sk_i] = tf.keras.Model(inputs=inputs, outputs=outputs)
	print(model[sk_i].summary())
	start = time.time()
	model[sk_i].compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	model[sk_i].fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, validation_split=0.3, epochs=_EPOCHS)
	end = time.time()
	print(' tiempo de training transcurrido (segundos) =', end - start)
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
		grupos_de_categorias = dividir_array_categorias(_DATA_TEST_Y, 10, 3)
		categorias_incluir = combinar_arrays(grupos_de_categorias)[sk_i]
		aux = f'{categorias_incluir}'
		predicted = model[sk_i].predict(_DATA_TEST_X, verbose=1)
		categorias_str = aux[aux.find('[') + 1:aux.find(']')]
		categorias = np.fromstring(categorias_str, dtype=int, sep=' ')
		resul = []
		for (i, pred) in enumerate(predicted):
			array_final = np.ones(10)
			array_final[categorias] = pred
			resul.append(array_final.tolist())
		predictions_0[label] = resul
	#__CLOUDBOOK:UNLOCK__


#SKYNNET:END

def main():
	pass


#__CLOUDBOOK:DU0__
def skynnet_train_global_0():
	for i in range(3):
		skynnet_train_0(i)
	#__CLOUDBOOK:SYNC__
#__CLOUDBOOK:DU0__
def skynnet_prediction_global_0():
	_DATA_TEST_X = x_test
	_DATA_TEST_Y = y_test
	for i in range(3):
		skynnet_prediction_0()
	#__CLOUDBOOK:SYNC__
	global predictions_0
	precision_compuesta = []
	valores = np.array(list(predictions_0.values()))
	predicted = np.prod(valores, axis=0)
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

