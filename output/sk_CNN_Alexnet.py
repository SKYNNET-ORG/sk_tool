import tensorflow_datasets as tfds
import tensorflow as tf, numpy as np
from tensorflow.keras import layers,models
import time
import sys # para coger parametros de entrada

import cv2 # opencv para el reescalado y crop de imagenes
import math
import matplotlib.pyplot as plt

def adaptImages(x_train,a):
	j=0
	for i in range(0,samples):
		h,w, channels=x_train[i].shape
		#print("h=",h, " w=",w, "channels=", channels)
		img=x_train[i]
		#print ("dtype=",img.dtype)
		# dos estrategias: cortar o meter franjas negras para llegar a cuadrado
		# si cortamos aprende pero no aprende bien porque muchas imagenes se corrompen
		estrategia=2 # 1 =crop, 2 = franjas, 3 =escalado con deformacion

		if (estrategia==1): 
			# crop
			#---------------------
			#img=cv2.resize(img, dsize=(w,h))
			# diferenciamos segun relacion de aspecto w>=h o w<h
			if (w>=h):
				# cortamos el lado mayor
				margen=int((w-h)/2)
				img2=img[0:h, margen:margen+h]
			
			else:
				margen=int((h-w)/2)
				img2=img[margen:margen+w, 0:w]
				
		elif (estrategia==2): 
			# aumentar con franjas negras
			#------------------------
			if (w>=h):#imagen horizontal w>h
				img2= np.zeros(w*w*channels,dtype=data_type)
				#img2[img2==0]=128 #probamos 128
				img2.shape=(w,w,channels)
				ini=int((w-h)/2)
				fin=int(h+ini)
				
				#print ("ini, fin, dif =", ini, fin, (fin-ini))
				#img3=img2[ini:fin,0:w,0:3];
				#print ("shape ",x_train[i].shape, img.shape, img2.shape)
				img2[ini:fin,0:w,0:3]=img[0:h, 0:w,0:3]
				#img2 = np.float32(img2)

				#franjas de repeticion
				#img2[0:ini,0:w,0:3]=img[0:1, 0:w,0:3]
				#img2[fin:w,0:w,0:3]=img[h-1:h, 0:w,0:3]
				
			else: # imagen vertical h>w
				img2= np.zeros(h*h*channels,dtype=data_type)
				#img2[img2==0]=128 #probamos 128
				img2.shape=(h,h,channels)
				
				ini=int((h-w)/2)
				fin=int(w+ini)
				
				img2[0:h,ini:fin,0:3]=img[0:h, 0:w,0:3]
				#img2 = np.float32(img2)

				#franjas de repeticion
				#img2[0:h,0:ini,0:3]=img[0:h, 0:1,0:3]
				#img2[0:h,fin:h,0:3]=img[0:h, w-1:w,0:3]
				
		elif (estrategia==3):
			# escalado con deformacion
			#------------------------
			maxl=max(w,h)
			minl=min(w,h)
			if (maxl/minl>4/3):
				continue # cribamos
			img2=img
				
		# creamos la imagen nueva reescalada
		if (w!=w2 or h!=h2): # optimizacion
			img2=cv2.resize(img2, dsize=(w2,h2), interpolation=cv2.INTER_LANCZOS4) # INTER_NEAREST INTER_LINEAR, INTER_CUBIC , INTER_LANCZOS4
		
		
		#la guardamos
		
		if (channels2==1):
			#print("cambio shape")
			if (channels==3):
				img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # convertimos a bn
			#b,g,r = cv2.split(img2)
			#img2=b #.shape=(h2,w2,1)
			img2.shape=(h2,w2,1)
			#printf("b shape =",b.shape)
			
		
		#print(f" item {i} xtrain shape: {x_train[i].shape}")
		a[j]=img2 # como  A es float, se copia desde uint8 a float. Es decir, a[i] es float
		j=j+1
		#print(f" item {i} a shape: {a[i].shape}")
		# las mostramos
		if (i<0):
			img_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # las magenes llegan en BGR y para hacer show necesitamos RGB
			cv2.imshow('orig', img_orig)
			print("img2.shape=", img2.shape)
			img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) # las magenes llegan en BGR y para hacer show necesitamos RGB
			cv2.imshow('Image', img3)
			#img3 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
			#cv2.imshow('Image', img3)
			
			cv2.waitKey(0);
	
	return j


ds_train = tfds.load('caltech101', 
					data_dir='caltech101',
					split='all',
					as_supervised=True,
					shuffle_files=False
					)

ds_train_npy=tfds.as_numpy(ds_train)
x_train = np.array(list(map(lambda x: x[0], ds_train_npy)))
y_train = np.array(list(map(lambda x: x[1], ds_train_npy)))

w2=227
h2=227
channels=x_train[0].shape[2] # todos los elementos tienen mismo num canales
samples=x_train.shape[0]
data_type= x_train[0].dtype
channels2=channels # es por si queremos pasar a bn, basta con poner channels2=1
a = np.zeros(h2*w2*channels2*samples, dtype=np.float32) #data_type)
a.shape=(samples,h2,w2,channels2)

valid_samples=adaptImages(x_train,a)
#adaptImages(x_test,b)
print("")
#Normalize the pixel values by deviding each pixel by 255
#x_train, x_test = x_train / 255.0, x_test / 255.0

a = a/255.0

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
	num_classes = 102
	n = num_classes
	filters = 384
	densas = 4096
	_DATA_TRAIN_X = a
	_DATA_TRAIN_Y = y_train
	_NEURON_1 = 2731
	_NEURON_2 = 68
	_FILTERS_1 = 256
	_EPOCHS = 1
	grupos_de_categorias = dividir_array_categorias(_DATA_TRAIN_Y, 102, 3)
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
	_NEURON_2 = len(np.unique(_DATA_TRAIN_Y))
	model[sk_i] = tf.keras.models.Sequential([layers.Conv2D(filters=int(_FILTERS_1 / 4), kernel_size=(11, 11), strides=4, activation='relu', input_shape=(h2, w2, channels2)), layers.MaxPooling2D((3, 3), strides=2), layers.Conv2D(filters=int(_FILTERS_1 / 1.5), kernel_size=(5, 5), strides=1, padding='same', activation='relu'), layers.MaxPooling2D((3, 3), strides=2), layers.Conv2D(filters=int(_FILTERS_1), kernel_size=(3, 3), strides=1, padding='same', activation='relu'), layers.Conv2D(filters=int(_FILTERS_1), kernel_size=(3, 3), strides=1, padding='same', activation='relu'), layers.Conv2D(filters=int(_FILTERS_1 / 1.5), kernel_size=(3, 3), strides=1, padding='same', activation='relu'), layers.MaxPooling2D((3, 3), strides=2), tf.keras.layers.Flatten(), tf.keras.layers.Dense(int(_NEURON_1), activation='relu'), tf.keras.layers.Dropout(rate=0.5), tf.keras.layers.Dense(int(_NEURON_1), activation='relu'), tf.keras.layers.Dropout(rate=0.5), tf.keras.layers.Dense(_NEURON_2, activation='softmax')])
	print(model[sk_i].summary())
	model[sk_i].compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	start = time.time()
	trained_model = model[sk_i].fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, validation_split=0.2, batch_size=32, epochs=_EPOCHS)
	end = time.time()
	print(' tiempo transcurrido (segundos) =', end - start)
	cosa = 'Acc. using' + ' DS=' + 'caltech101' + ', cat=' + str(102) + ' size:' + str(h2) + 'x' + str(w2)
	plt.title(cosa)
	plt.plot(trained_model.history['accuracy'], 'b-')
	plt.plot(trained_model.history['val_accuracy'], 'g-')
	plt.legend(['train', 'val'], loc='upper left')
	plt.xlabel('Epoch')
	plt.ylabel('Percent')
	plt.show()
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
def main():
	skynnet_train_global_0()
	skynnet_prediction_global_0()

if __name__ == '__main__':
	main()

