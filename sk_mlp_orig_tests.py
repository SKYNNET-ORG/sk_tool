import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
# Load MNIST data using built-in datasets download function
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Noramalize the pixel values by deviding each pixel by 255
x_train, x_test = x_train / 255.0, x_test / 255.0


#SKYNNET:BEGIN_MULTICLASS_ACC_LOSS
_DATA_TRAIN = (x_train, y_train)
_DATA_TEST = (x_test, y_test)
_NEURON_1 = 32
_NEURON_2 = 15
_NEURON_3 = 2

def crea_modelo():
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(_NEURON_1, activation='relu'), tf.keras.layers.Dense(_NEURON_2, activation='relu'), tf.keras.layers.Dense(_NEURON_3, activation='softmax')])
    return model
crea_modelo()
print(model.summary())
print('bonito sumario')
model = 43
model.hola()
calculanumeropi()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, validation_split=0.3, epochs=2)
predicted = model.predict(x_test)

_DATA_TRAIN = (x_train, y_train)
_DATA_TEST = (x_test, y_test)

def crea_modelo():
    model_1 = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(_NEURON_1, activation='relu'), tf.keras.layers.Dense(_NEURON_2, activation='relu'), tf.keras.layers.Dense(_NEURON_3, activation='softmax')])
    return model_1
crea_modelo()
print(model_1.summary())
print('bonito sumario')
model = 43
model.hola()
calculanumeropi()
model_1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_1.fit(x_train, y_train, validation_split=0.3, epochs=2)
predicted = model_1.predict(x_test)

_DATA_TRAIN = (x_train, y_train)
_DATA_TEST = (x_test, y_test)

def crea_modelo_2():
    model_2 = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(_NEURON_1, activation='relu'), tf.keras.layers.Dense(_NEURON_2, activation='relu'), tf.keras.layers.Dense(_NEURON_3, activation='softmax')])
    return model_2
crea_modelo_2()
print(model_2.summary())
print('bonito sumario')
model = 43
model.hola()
calculanumeropi()
model_2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_2.fit(x_train, y_train, validation_split=0.3, epochs=2)
predicted = model_2.predict(x_test)

_DATA_TRAIN = (x_train, y_train)
_DATA_TEST = (x_test, y_test)

def crea_modelo_3():
    model_3 = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(_NEURON_1, activation='relu'), tf.keras.layers.Dense(_NEURON_2, activation='relu'), tf.keras.layers.Dense(_NEURON_3, activation='softmax')])
    return model_3
crea_modelo()
print(model_3.summary())
print('bonito sumario')
model = 43
model.hola()
calculanumeropi()
model_3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_3.fit(x_train, y_train, validation_split=0.3, epochs=2)
predicted = model_3.predict(x_test)


#SKYNNET:END

print("End of program")