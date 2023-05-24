import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
# Load MNIST data using built-in datasets download function
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Noramalize the pixel values by deviding each pixel by 255
x_train, x_test = x_train / 255.0, x_test / 255.0


#SKYNNET:BEGIN_MULTICLASS_ACC_LOSS
#__CLOUDBOOK:GLOBAL__
predictions_0_0 = {}
#__CLOUDBOOK:NONSHARED__
model = None
#__CLOUDBOOK:PARALLEL__
def skynnet_block_0():
    global model
    _DATA_TRAIN = (x_train, y_train)
    _DATA_TEST = (x_test, y_test)
    _NEURON_1 = 32
    _NEURON_2 = 15
    _NEURON_3 = 2
    _EPOCHS = 0

    def crea_modelo():
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(_NEURON_1, activation='relu'), tf.keras.layers.Dense(_NEURON_2, activation='relu'), tf.keras.layers.Dense(_NEURON_3, activation='softmax')])
        return model
    model = crea_modelo()
    print(model.summary())
    print('bonito sumario')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.3, epochs=_EPOCHS)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_block_0():
    global predictions_0_0
    global model
    #__CLOUDBOOK:BEGINREMOVE__
    __CLOUDBOOK__['agent']['id'] = 'agente_skynnet'
    #__CLOUDBOOK:ENDREMOVE__
    label = __CLOUDBOOK__['agent']['id']
    predictions_0_0[label] = model.predict(x_test)


#SKYNNET:END

print("End of program 1")

#SKYNNET:BEGIN_MULTICLASS_ACC_LOSS


#__CLOUDBOOK:GLOBAL__
predictions_1_0 = {}
predictions_1_1 = {}
#__CLOUDBOOK:NONSHARED__
model = None
model2 = None
#__CLOUDBOOK:PARALLEL__
def skynnet_block_1():
    global model2
    global model
    _DATA_TRAIN = (x_train, y_train)
    _DATA_TEST = (x_test, y_test)
    _NEURON_1 = 32
    _NEURON_2 = 15
    _NEURON_3 = 2
    _EPOCHS = 0

    def crea_modelo():
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(_NEURON_1, activation='relu'), tf.keras.layers.Dense(_NEURON_2, activation='relu'), tf.keras.layers.Dense(_NEURON_3, activation='softmax')])
        return model
    model = crea_modelo()
    print(model.summary())
    print('bonito sumario')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.3, epochs=_EPOCHS)
    _DATA_TRAIN = (x_train, y_train)
    _DATA_TEST = (x_test, y_test)
    _NEURON_1 = 32
    _NEURON_2 = 15
    _NEURON_3 = 2
    _EPOCHS = 0
    model2 = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(_NEURON_1, activation='relu'), tf.keras.layers.Dense(_NEURON_2, activation='relu'), tf.keras.layers.Dense(_NEURON_3, activation='softmax')])
    print(model2.summary())
    model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model2.fit(x_train, y_train, validation_split=0.3, epochs=_EPOCHS)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_block_1():
    global predictions_1_0
    global predictions_1_1
    global model
    global model2
    #__CLOUDBOOK:BEGINREMOVE__
    __CLOUDBOOK__['agent']['id'] = 'agente_skynnet'
    #__CLOUDBOOK:ENDREMOVE__
    label = __CLOUDBOOK__['agent']['id']
    predictions_1_0[label] = model.predict(x_test)
    predictions_1_1[label] = model2.predict(x_test)


#SKYNNET:END

print("End of program 2")





