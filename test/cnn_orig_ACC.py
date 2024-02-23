import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow import keras
import numpy as np
import time

#load data
(x_train, y_train) , (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

#reshape data para red original
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)


#SKYNNET:BEGIN_MULTICLASS_ACC

_DATA_TRAIN_X = x_train
_DATA_TRAIN_Y = y_train
#_DATA_VAL=(x_val,y_val) En este caso, se usa un validation split
_DATA_TEST_X = x_test
_DATA_TEST_Y = y_test

_FILTERS_1 = 25
_FILTERS_2 = 64
_FILTERS_3 = 64
_NEURON_1 = 64
_NEURON_2 = 10
_EPOCHS = 10

# red original
cnn_orig = models.Sequential([
    layers.Conv2D(filters=_FILTERS_1, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=_FILTERS_2, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=_FILTERS_3, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(_NEURON_1, activation='relu'),
    layers.Dense(_NEURON_2, activation='softmax')
])
print(cnn_orig.summary())



#training red original
print ("========= entrenamiento de red original ================")
start=time.time()
cnn_orig.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_orig.fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, epochs=_EPOCHS,validation_split=0.3)
end=time.time()
print (" tiempo de training transcurrido (segundos) =", (end-start))

predicted = cnn_orig.predict(_DATA_TEST_X)
#SKYNNET:END