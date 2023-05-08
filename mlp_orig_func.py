import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
# Load MNIST data using built-in datasets download function
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Noramalize the pixel values by deviding each pixel by 255
x_train, x_test = x_train / 255.0, x_test / 255.0


#SKYNNET:BEGIN_MULTICLASS_ACC_LOSS

_DATA_TRAIN =(x_train,y_train)
#_DATA_VAL=(x_val,y_val) En este caso, se usa un validation split
_DATA_TEST=(x_test,y_test)
_NEURON_1 = 128
_NEURON_2 = 60
_NEURON_3 = 10


#Modelo funcional
inputs = tf.keras.Input(shape=(28,28))
x = tf.keras.layers.Dense(_NEURON_1, activation='relu')(inputs)
x = tf.keras.layers.Dense(_NEURON_2, activation='relu')(x)
outputs =  tf.keras.layers.Dense(_NEURON_3, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, validation_split=0.3, epochs=2)


predicted = model.predict(x_test)


#SKYNNET:END

print("End of program")