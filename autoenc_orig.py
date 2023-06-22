import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time




#SKYNNET:BEGIN_REGRESSION_ACC_LOSS
ratio_in=1
ratio_out=2
bottleneck=8
data_dim_input=784
bottleneck=16
data_dim_output=784/4 #184/4=196

# Placeholder for input
input_image = tf.keras.layers.Input(shape=(data_dim_input,))
encoded_input = tf.keras.layers.Dense(bottleneck, activation='relu')(input_image)
decoded_output = tf.keras.layers.Dense(data_dim_output, activation='sigmoid')(encoded_input)

# Autoencoder model to map an input to its output
autoencoder = tf.keras.models.Model(input_image, decoded_output)

autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')


# tenemos una de ropa en fashion_mnist.load_data()
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
#(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

X_train_out = X_train[0:,14:28,14:28]
X_test_out = X_test[0:,14:28,14:28]

#input shape
X_train = X_train.reshape((len(X_train),np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test),np.prod(X_test.shape[1:])))

#output shape
X_train_out = X_train_out.reshape((len(X_train_out),np.prod(X_train_out.shape[1:])))
X_test_out = X_test_out.reshape((len(X_test_out),np.prod(X_test_out.shape[1:])))

# VAMOS A ENTRENAR !!
start=time.time()
trained_model=autoencoder.fit(X_train,X_train_out,
                epochs = 30, #30,
                batch_size = 256,
                shuffle = True,
                validation_data = (X_test, X_test_out))
end=time.time()
print (" tiempo de training transcurrido (segundos) =", (end-start))


reconstructed_img = autoencoder.predict(X_test)

