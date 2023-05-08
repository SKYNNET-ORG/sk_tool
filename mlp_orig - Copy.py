import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
# Load MNIST data using built-in datasets download function
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Noramalize the pixel values by deviding each pixel by 255
x_train, x_test = x_train / 255.0, x_test / 255.0

#cloudbook:parallel
def skynnet():#cada maquina ejecutara el model que tenga su id, cambiar los model por model[id]
  #SKYNNET:BEGIN_MULTICLASS_ACC_LOSS
  #filtra segun el numero del nombre, y la informacion de la etiqueta skynnet, y el numero de subredes
  que_hago(n,num_subredes,multiclass)
  _DATA_TRAIN =(x_train,y_train)
  #_DATA_VAL=(x_val,y_val) En este caso, se usa un validation split
  _DATA_TEST=(x_test,y_test)
  _NEURON_1 = 128
  _NEURON_2 = 60
  _NEURON_3 = 10

  #Modelo normal
  model[n] = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(_NEURON_1, activation='relu'),
    tf.keras.layers.Dense(_NEURON_2, activation='relu'),
    tf.keras.layers.Dense(_NEURON_3, activation='softmax')
  ])

  print(model.summary())

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


  model.fit(x_train, y_train, validation_split=0.3, epochs=2)
#SKYNNET:END

#cloudbook:parallel
def skynnet():#cada maquina ejecutara el model que tenga su id, cambiar los model por model[id], en la herramienta igual
  #SKYNNET:BEGIN_MULTICLASS_ACC_LOSS
  que_hago(...)
  _DATA_TRAIN =(x_train,y_train)
  #_DATA_VAL=(x_val,y_val) En este caso, se usa un validation split
  _DATA_TEST=(x_test,y_test)
  _NEURON_1 = 128
  _NEURON_2 = 60
  _NEURON_3 = 10

  #Modelo normal
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(_NEURON_1, activation='relu'),
    tf.keras.layers.Dense(_NEURON_2, activation='relu'),
    tf.keras.layers.Dense(_NEURON_3, activation='softmax')
  ])

  print(model.summary())

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


  model.fit(x_train, y_train, validation_split=0.3, epochs=2)
#SKYNNET:END


''' antes del skynnet end, el user pone predict de esta manera predicted = model.predict(x_test) que lo convierto en'''


#convierto el predict en esto, fuera de la funcion skynnet


#lo convertimos en:
#cloudbook_parallel
def predict():
  #i = get_id()
  #filtras datos de test con el que te toque
  for i in [get_id()]:
    predicted = model[i].predict(x_test)
  return i, predicted
'''al definir el modelo, se declara model como #__CLOUDBOOK:NONSHARED__
  model es un array, del que solo se rellena la posicion i, que es lo que te toca en la invocacion con parallel
  get_id() es una lista en la herramienta, y en cloudbook sera el id, se pone con cloudbook beginremove

'''
def get_id():
  return iden

iden = 0
'''iden es una vble global que le escribo yo al user en la herramienta'''

print("End of program")

###En el main del usuario
#el codigo skynnet entre las etiquetas


####en la du_0
#en la herramienta el model es global y es model = []
def skynnet_global():
  for i in subredes:
    assign_unique_id(i) #y filtrar datos 
    #en la herramienta no hace nada
  for i in subredes:
    skynnet()
    #cloudbook:sync

def skynnet_predict_global():
  for i in num_subredes:
    iden = i
    iden,prediccion = predict()
    predicted[iden] = prediccion
  res = compose_respuesta(predicted)
  return res

#para llamar al predicted correcto, hay q tener identificador unico, como se hacia en el modelo BOINC

skynnet_global()
predicted = skynnet_predict_global()