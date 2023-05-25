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
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(_NEURON_1, activation='relu'), tf.keras.layers.Dense(_NEURON_2, activation='relu'), tf.keras.layers.Dense(_NEURON_3, activation='softmax')])
    print(model.summary())
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.3, epochs=_EPOCHS)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_block_0():
    global predictions_0_0
    global model
    #__CLOUDBOOK:BEGINREMOVE__
    __CLOUDBOOK__ = {}
    __CLOUDBOOK__['agent'] = {}
    __CLOUDBOOK__['agent']['id'] = 'agente_skynnet'
    #__CLOUDBOOK:ENDREMOVE__
    label = __CLOUDBOOK__['agent']['id']
    predictions_0_0[label] = model.predict(x_test)


#SKYNNET:END

print("End of program")

#__CLOUDBOOK:DU0__
def skynnet_global_0():
    for i in range(4):
        skynnet_block_0()
    #__CLOUDBOOK:SYNC__
#__CLOUDBOOK:DU0__
def skynnet_prediction_global_0():
    for i in range(4):
        skynnet_prediction_block_0()
    #__CLOUDBOOK:SYNC__


#__CLOUDBOOK:MAIN__
def main():
    skynnet_global_0()
    skynnet_prediction_global_0()

if __name__ == '__main__':
    main()

