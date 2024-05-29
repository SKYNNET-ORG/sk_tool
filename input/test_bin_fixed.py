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


#SKYNNET:BEGIN_BINARYCLASS

_DATA_TRAIN_X = x_train
_DATA_TRAIN_Y = y_train
_DATA_TEST_X = x_test
_DATA_TEST_Y = y_test
_NEURON_1 = 32
_NEURON_2 = 16
_NEURON_3 = 2
_EPOCHS = 8

inputs = tf.keras.Input(shape=(28,28))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(_NEURON_1, activation='relu')(x)
x = tf.keras.layers.Dense(_NEURON_2, activation='relu')(x)
outputs =  tf.keras.layers.Dense(_NEURON_3, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(optimizer='adam',
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

start=time.time()

model.fit(_DATA_TRAIN_X, _DATA_TRAIN_Y,
        validation_split=0.3,
        epochs=_EPOCHS)
end=time.time()
print (" original: tiempo transcurrido (segundos) =", (end-start))

predicted = model.predict(_DATA_TEST_X)

correctas = 0
total = 0
for i in range(len(_DATA_TEST_Y)):
        if _DATA_TEST_Y[i] == np.argmax(predicted[i]):
                correctas += 1
        total += 1
precision = correctas / total
print('============================================')
print('La accuracy de la prediccion es: ', precision)
print('============================================')
scce = tf.keras.losses.SparseCategoricalCrossentropy()
scce_orig = scce(_DATA_TEST_Y, predicted).numpy()
print('============================================')
print('La loss es: ', scce_orig)
print('============================================')

#SKYNNET:END
