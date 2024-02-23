import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
#from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

# cargamos los datos de entrenamiento
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#normalizamos a 0..1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# We will make a test set of 10 samples and use the other 9990 as validation data.
# esto quita los ultimos 10 elementos (se queda con 0..9990)
x_validate, y_validate = x_test[:-10], y_test[:-10]
# esto coge los 10 ultimos (se queda con 9990..9999)
x_test, y_test = x_test[-10:], y_test[-10:]


#SKYNNET:BEGIN_MULTICLASS

_DATA_TRAIN_X = x_train
_DATA_TRAIN_Y = y_train

_DATA_VAL_X = x_validate
_DATA_VAL_Y = y_validate

_DATA_TEST_X = x_test
_DATA_TEST_Y = y_test

_NEURON_1 = 64
_NEURON_2 = 10

_EPOCHS = 10


#Modelo funcional
#inputs = tf.keras.Input(shape=(28,28))
#x = tf.keras.layers.GRU(_NEURON_1)
#x = tf.keras.layers.BatchNormalization()(x)
#outputs = tf.keras.layers.Dense(_NEURON_2, activation='softmax')(x)
#model = tf.keras.Model(inputs=inputs, outputs=outputs)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(28,28))) # capa de entrada de 28x28 ( seria como 28 vectores de 28 dimensiones)
model.add(tf.keras.layers.GRU(_NEURON_1)) # 64 celdas GRU
# la capa de normalizacion no es imprescindible pero he constatado que ayuda a aprender mas rapido
model.add(tf.keras.layers.BatchNormalization()) # capa de normalizacion para centrar salida en cero
model.add(tf.keras.layers.Dense(_NEURON_2, activation='softmax')) #capa de salida , 10 categorias
print(model.summary())

model.compile(
    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # le que quitado el logits y le puse softmax. funciona igual de bien
    #  If your output layer has a 'softmax' activation, from_logits should be False.
    #  If your output layer doesn't have a 'softmax' activation, from_logits should be True
    loss='sparse_categorical_crossentropy',
    optimizer="sgd",
    metrics=["accuracy"],
)

#entrenamos
start=time.time()
model.fit(
    _DATA_TRAIN_X, _DATA_TRAIN_Y,
    validation_data=(_DATA_VAL_X, _DATA_VAL_Y),
    batch_size=32, #64, # como son 60.000 datos --> 60000/64 = 938 batches por epoca.
    epochs=_EPOCHS #10
)
end=time.time()
print (" tiempo de training transcurrido (segundos) =", (end-start))

predicted = model.predict(_DATA_TEST_X)

#SKYNNET:END