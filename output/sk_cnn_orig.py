import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow import keras
import numpy as np
import time


#__CLOUDBOOK:NONSHARED__
loaded_dataset = keras.datasets.mnist.load_data()
x_train = loaded_dataset[0][0]
y_train = loaded_dataset[0][1]
x_test = loaded_dataset[1][0]
y_test = loaded_dataset[1][1]
#(x_train, y_train) , (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

#reshape data para red original
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)


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
cnn_orig = [None, None, None]
to_predict_models = []
#__CLOUDBOOK:PARALLEL__
def skynnet_train_0(sk_i):
	global cnn_orig
	global to_predict_models
	_DATA_TRAIN_X = x_train
	_DATA_TRAIN_Y = y_train
	_FILTERS_1 = 17
	_FILTERS_2 = 43
	_FILTERS_3 = 43
	_NEURON_1 = 43
	_NEURON_2 = 7
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
	_NEURON_2 = len(combinacion_arrays)
	cnn_orig[sk_i] = models.Sequential([layers.Conv2D(filters=_FILTERS_1, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)), layers.MaxPooling2D((2, 2)), layers.Conv2D(filters=_FILTERS_2, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D((2, 2)), layers.Conv2D(filters=_FILTERS_3, kernel_size=(3, 3), activation='relu'), layers.MaxPooling2D((2, 2)), layers.Flatten(), layers.Dense(_NEURON_1, activation='relu'), layers.Dense(_NEURON_2, activation='softmax')])
	print(cnn_orig[sk_i].summary())
	print('========= entrenamiento de red original ================')
	start = time.time()
	cnn_orig[sk_i].compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	cnn_orig[sk_i].fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, epochs=_EPOCHS, validation_split=0.3)
	end = time.time()
	print(' tiempo de training transcurrido (segundos) =', end - start)
	to_predict_models.append(sk_i)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_0():
	global predictions_0
	global to_predict_models
	global cnn_orig
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
		predicted = cnn_orig[sk_i].predict(_DATA_TEST_X, verbose=1)
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

