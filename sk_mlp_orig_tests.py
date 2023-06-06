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
model = []
#__CLOUDBOOK:PARALLEL__
def skynnet_block_0(i):
    global model
    model.append(None)
    _DATA_TRAIN = (x_train, y_train)
    _DATA_TEST = (x_test, y_test)
    _NEURON_1 = 64
    _NEURON_2 = 30
    _NEURON_3 = 5
    _EPOCHS = 1

    def crea_modelo():
        model[i] = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(_NEURON_1, activation='relu'), tf.keras.layers.Dense(_NEURON_2, activation='relu'), tf.keras.layers.Dense(_NEURON_3, activation='softmax')])
        return model[i]
    model[i] = crea_modelo()
    print(model[i].summary())
    print('bonito sumario')
    model[i].compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model[i].fit(x_train, y_train, validation_split=0.3, epochs=_EPOCHS)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_block_0(i):
    global predictions_0_0
    global model
    #__CLOUDBOOK:BEGINREMOVE__
    __CLOUDBOOK__ = {}
    __CLOUDBOOK__['agent'] = {}
    __CLOUDBOOK__['agent']['id'] = 'agente_skynnet'
    #__CLOUDBOOK:ENDREMOVE__
    label = __CLOUDBOOK__['agent']['id'] + str(i)
    predictions_0_0[label] = model[i].predict(x_test)


#SKYNNET:END

print("End of program 1")

#SKYNNET:BEGIN_MULTICLASS_ACC_LOSS


#__CLOUDBOOK:DU0__
def skynnet_global_0():
    for i in range(2):
        skynnet_block_0(i)
    #__CLOUDBOOK:SYNC__
#__CLOUDBOOK:DU0__
def skynnet_prediction_global_0():
    for i in range(2):
        skynnet_prediction_block_0(i)
    #__CLOUDBOOK:SYNC__

#__CLOUDBOOK:GLOBAL__
predictions_1_0 = {}
predictions_1_1 = {}
#__CLOUDBOOK:NONSHARED__
model = []
model2 = []
#__CLOUDBOOK:PARALLEL__
def skynnet_block_1(i):
    global model2
    model2.append(None)
    global model
    model.append(None)
    _DATA_TRAIN = (x_train, y_train)
    _DATA_TEST = (x_test, y_test)
    _NEURON_1 = 64
    _NEURON_2 = 30
    _NEURON_3 = 5
    _EPOCHS = 1

    def crea_modelo():
        model[i] = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(_NEURON_1, activation='relu'), tf.keras.layers.Dense(_NEURON_2, activation='relu'), tf.keras.layers.Dense(_NEURON_3, activation='softmax')])
        return model[i]
    model[i] = crea_modelo()
    print(model[i].summary())
    print('bonito sumario')
    model[i].compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model[i].fit(x_train, y_train, validation_split=0.3, epochs=_EPOCHS)
    _DATA_TRAIN = (x_train, y_train)
    _DATA_TEST = (x_test, y_test)
    _NEURON_1 = 64
    _NEURON_2 = 30
    _NEURON_3 = 5
    _EPOCHS = 1
    model2[i] = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(_NEURON_1, activation='relu'), tf.keras.layers.Dense(_NEURON_2, activation='relu'), tf.keras.layers.Dense(_NEURON_3, activation='softmax')])
    print(model2[i].summary())
    model2[i].compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model2[i].fit(x_train, y_train, validation_split=0.3, epochs=_EPOCHS)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_block_1(i):
    global predictions_1_0
    global predictions_1_1
    global model
    global model2
    #__CLOUDBOOK:BEGINREMOVE__
    __CLOUDBOOK__ = {}
    __CLOUDBOOK__['agent'] = {}
    __CLOUDBOOK__['agent']['id'] = 'agente_skynnet'
    #__CLOUDBOOK:ENDREMOVE__
    label = __CLOUDBOOK__['agent']['id'] + str(i)
    predictions_1_0[label] = model[i].predict(x_test)
    predictions_1_1[label] = model2[i].predict(x_test)


#SKYNNET:END

print("End of program 2")

#__CLOUDBOOK:DU0__
def skynnet_global_1():
    for i in range(2):
        skynnet_block_1(i)
    #__CLOUDBOOK:SYNC__
#__CLOUDBOOK:DU0__
def skynnet_prediction_global_1():
    for i in range(2):
        skynnet_prediction_block_1(i)
    #__CLOUDBOOK:SYNC__


#__CLOUDBOOK:MAIN__
def main():
    skynnet_global_0()
    skynnet_prediction_global_0()
    skynnet_global_1()
    skynnet_prediction_global_1()

if __name__ == '__main__':
    main()

