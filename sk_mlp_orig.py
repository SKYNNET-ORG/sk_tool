import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
# Load MNIST data using built-in datasets download function
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Noramalize the pixel values by deviding each pixel by 255
x_train, x_test = x_train / 255.0, x_test / 255.0


#SKYNNET:BEGIN_MULTICLASS_ACC_LOSS

#__CLOUDBOOK:NONSHARED__
model = None
#__CLOUDBOOK:PARALLEL__
def skynnet_block_0():
    _DATA_TRAIN = (x_train, y_train)
    _DATA_TEST = (x_test, y_test)
    _NEURON_1 = 64
    _NEURON_2 = 30
    _NEURON_3 = 5
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(_NEURON_1, activation='relu'), tf.keras.layers.Dense(_NEURON_2, activation='relu'), tf.keras.layers.Dense(_NEURON_3, activation='softmax')])
    print(model.summary())
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_split=0.3, epochs=2)
    predicted = model.predict(x_test)


#SKYNNET:END

print("End of program")


#__CLOUDBOOK:DU0__
def skynnet_global_0():
    for i in range(2):
        skynnet_block_0()
		#__CLOUDBOOK:SYNC__


if __name__ == '__main__':
    skynnet_global_0()

