import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
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
y_test = x_test_out
y_train = x_train_out

#SKYNNET:BEGIN_REGRESSION_ACC_LOSS
ratio=1
data_dim_input=784
bottleneck=32
data_dim_output=data_dim_input/4

_DATA_TRAIN_X = x_train
_DATA_TRAIN_Y = x_train_out
_DATA_VAL_X = x_test
_DATA_VAL_Y = x_test_out
_DATA_TEST_X = x_test
_DATA_TEST_Y = x_test_out
bottleneck = 32
_NEURON_3 = 784
_EPOCHS = 30

# Placeholder for input
input_image = tf.keras.layers.Input(shape=(data_dim_input,))
encoded_input = tf.keras.layers.Dense(bottleneck, activation='relu')(input_image)
decoded_output = tf.keras.layers.Dense(_NEURON_3, activation='sigmoid')(encoded_input)

# Autoencoder model to map an input to its output
autoencoder = tf.keras.models.Model(input_image, decoded_output)

autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
print(autoencoder.summary())

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
n = 20
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(_DATA_TEST_X[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_img[i].reshape(28, 28))
    
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#plt.show()
print("Todo terminado")
print("Adios")

#SKYNNET:END