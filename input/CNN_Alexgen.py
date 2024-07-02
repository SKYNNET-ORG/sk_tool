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


_DATA_TRAIN_GEN_Y = y_train
#if testx==None:
    #load_data()
_DATA_TEST_X = testx
_DATA_TEST_Y = testy
_NEURON_1 = 4096
_NEURON_2 = 102
_FILTERS_1 = 384
batch_size=32



#ALEXNET
#---------
model = tf.keras.models.Sequential([
  
  #alexnet original
  layers.Conv2D(filters=int(_FILTERS_1/4), kernel_size=(11, 11),strides=4, activation='relu', input_shape=(h2, w2,channels2)),
  layers.MaxPooling2D((3, 3), strides=2),
  
  layers.Conv2D(filters=int(_FILTERS_1/1.5), kernel_size=(5, 5),strides=1, padding="same",activation='relu'),
  

  layers.MaxPooling2D((3, 3), strides=2),
  
  layers.Conv2D(filters=int(_FILTERS_1), kernel_size=(3, 3), strides=1,padding="same",activation='relu'),
  layers.Conv2D(filters=int(_FILTERS_1), kernel_size=(3, 3), strides=1,padding="same",activation='relu'),

  # si meto esta ultima capa, ya no converge. se estanca en 0.0016, es decir, aleatorio
  layers.Conv2D(filters=int(_FILTERS_1/1.5), kernel_size=(3, 3), strides=1,padding="same",activation='relu'), # si metemos esta, estamos ante alexnet
  #layers.MaxPooling2D((2, 2)),

  #este lo quito para adaptacion cifar100
  layers.MaxPooling2D((3, 3), strides=2),
  
  tf.keras.layers.Flatten(),
  
  tf.keras.layers.Dense(int(_NEURON_1), activation='relu'),
  tf.keras.layers.Dropout(rate=0.5),
  
  tf.keras.layers.Dense(int(_NEURON_1), activation='relu'),
  tf.keras.layers.Dropout(rate=0.5),
  tf.keras.layers.Dense(_NEURON_2, activation='softmax')
])
print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', #sparse es cuando hay muchas categorias
              metrics=['accuracy'])

# Finally, train or fit the model


num_fotos=0

categorias = os.listdir('./dataset128128')
for x in categorias:
      num_fotos+=len(os.listdir('./dataset128128'+'/'+str(x)))


steps_per_epoch=num_fotos//batch_size

start=time.time()
trained_model = model.fit(data_generator(batch_size,_DATA_TRAIN_GEN_Y), 
                          #validation_split=0.2,
                          steps_per_epoch=steps_per_epoch,
                          epochs=1
                          )
end=time.time()
print (" tiempo transcurrido (segundos) =", (end-start))

print("To predict model")
#predicted = model.predict(_DATA_TEST_X)
print("Predicted model")
#SKYNNET:END