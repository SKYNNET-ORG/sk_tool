import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
# Load MNIST data using built-in datasets download function
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Noramalize the pixel values by deviding each pixel by 255
x_train, x_test = x_train / 255.0, x_test / 255.0

#SKYNNET:BEGIN_MULTICLASS_ACC_LOSS

_DATA_TRAIN_X = x_train
_DATA_TRAIN_Y = y_train
_DATA_TEST_X = x_test
_DATA_TEST_Y = y_test
_NEURON_1 = 128
_NEURON_2 = 60
_NEURON_3 = 10
_EPOCHS = 10


#Modelo funcional
inputs = tf.keras.Input(shape=(28,28))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(_NEURON_1, activation='relu')(x)
x = tf.keras.layers.Dense(_NEURON_2, activation='relu')(x)
outputs =  tf.keras.layers.Dense(_NEURON_3, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, validation_split=0.3, epochs=_EPOCHS)


predicted = model.predict(_DATA_TEST_X)


#SKYNNET:END

print("End of program")