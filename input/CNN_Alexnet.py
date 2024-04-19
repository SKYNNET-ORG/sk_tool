import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers,models
import time
import sys # para coger parametros de entrada

import cv2 # opencv para el reescalado y crop de imagenes
import math
import matplotlib.pyplot as plt
import dataloader


#__CLOUDBOOK:NONSHARED__
data = dataloader.load_data()
data_x = data[0]
data_y = data[1]

data_test_x = data[5]
data_test_y = data[6]

h2 = data[2]
w2 = data[3]
channels2 = data[4]

#SKYNNET:BEGIN_MULTICLASS_ACC_LOSS


_DATA_TRAIN_X = data_x
_DATA_TRAIN_Y = data_y
_DATA_TEST_X = data_test_x
_DATA_TEST_Y = data_test_y
_NEURON_1 = 4096
_NEURON_2 = 102
_FILTERS_1 = 384


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
start=time.time()

trained_model = model.fit(_DATA_TRAIN_X, 
                          _DATA_TRAIN_Y, 
                          validation_split=0.2,
                          batch_size=32,
                          epochs=1
                          )
end=time.time()
print (" tiempo transcurrido (segundos) =", (end-start))

# Visualize loss  and accuracy history
#--------------------------------------
#cosa="Acc. using"+" DS="+ 'caltech101'+", cat="+str(102)+" size:"+ str(h2)+"x"+str(w2)
'''cosa = "titulo"

plt.title(cosa)
plt.plot(trained_model.history['accuracy'], 'b-')
plt.plot(trained_model.history['val_accuracy'], 'g-')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('Epoch')
plt.ylabel('Percent')
plt.show();'''
print("To predict model")
predicted = model.predict(_DATA_TEST_X)
print("Predicted model")
#SKYNNET:END