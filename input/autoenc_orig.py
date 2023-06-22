import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_out=x_train #salida igual a entrada 
x_test_out=x_test
#input shape
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))

#output shape
x_train_out = x_train_out.reshape((len(x_train_out),np.prod(x_train_out.shape[1:])))
x_test_out = x_test_out.reshape((len(x_test_out),np.prod(x_test_out.shape[1:])))

#SKYNNET:BEGIN_REGRESSION_ACC_LOSS
ratio=1
data_dim_input=784
bottleneck=32
data_dim_output=data_dim_input

_DATA_TRAIN_X = x_train
_DATA_TRAIN_Y = x_train_out
_DATA_VAL_X = x_test
_DATA_VAL_Y = x_test_out
_DATA_TEST_X = x_test
_DATA_TEST_Y = x_test_out
_NEURON_1 = 8
_NEURON_2 = 784
_EPOCHS = 10

# Placeholder for input
input_image = tf.keras.layers.Input(shape=(data_dim_input,))
encoded_input = tf.keras.layers.Dense(bottleneck, activation='relu')(input_image)
decoded_output = tf.keras.layers.Dense(data_dim_output, activation='sigmoid')(encoded_input)

# Autoencoder model to map an input to its output
autoencoder = tf.keras.models.Model(input_image, decoded_output)

autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

# VAMOS A ENTRENAR !!
start=time.time()
autoencoder.fit(_DATA_TRAIN_X,_DATA_TRAIN_Y,
                epochs = _EPOCHS, #30,
                batch_size = 256,
                shuffle = True,
                validation_data = (_DATA_VAL_X, _DATA_VAL_Y))
end=time.time()
print (" tiempo de training transcurrido (segundos) =", (end-start))


reconstructed_img = autoencoder.predict(_DATA_TEST_X)

#SKYNNET:END