"""
############################################################
# trainer.py
#
# este modulo carga (si existe ) una red entrenada o crea una
# nueva si no existe y la entrena con los datos numpy de
# salida del modulo dataTrainGen
#
# Una vez que termina, almacena (o actualiza) el fichero de
# configuración de la red neuronal entrenada. Es decir, se va
# a sobreescribir siempre el fichero de configuración de la
# red del perfil
#
# el entrenamiento consume todos los ficheros que haya en el
# directorio de perfil y los renombrara o movera a un subdirectorio
# de ficheros procesados (quizas pongamos flag para borrado automatico)
#
# el modelo de red se almacenara en el directorio de perfil
# se sobreescribe, no se guarda el ultimo perfil.
#
# funciones de uso externo:
# -------------------------
#   trainProfile(profiledir)
#
#
# funciones internas
# -------------------
#   createModel()
#   loadTrainData(profiledir)
#   trainModel(model, data)
#
###############################################################
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import libaux


# ==========================================================================
def trainProfile(profiledir, data):
    """ crea o carga un modelo del directorio de profile
    y lo entrena con los ficheros numpy que encuentre en dicho directorio
    tras consumirlos, los movera a un subdirectorio procesados y
    actualizará el fichero que contiene el modelo

    esta funcion es para invocarla desde el modulo MAIN
    no es compatible con skynnet porque skynnet no permite cargar modelos
    para skynnet haremos un miniprograma en el que el modelo siempre se construya
    este tema hay que tratarlo tambien con juan


    Parameters:
    - profiledir: directorio del profile
    - data: data[0] es input y data[1] es salida deseada

    Returns:
    0: OK
    -1 : el directorio no existe

    """
    # esta funcion es para invocarla desde el modulo MAIN
    # no es compatible con skynnet porque skynnet no permite cargar modelos
    # para skynnet haremos un miniprograma en el que el modelo siempre se construya
    # este tema hay que tratarlo tambien con juan

    # primero entramos a cargar el modelo
    model = libaux.loadModel(profiledir)
    model_created_now = 0
    if(model is None):
        model = createModel()
        model_created_now = 1

    # carga los datos de entrenamiento
    # a,b= loadTrainData(profiledir)
    # imprime el modelo de RNA
    print(model.summary())

    # entrena el modelo. Tiene en cuenta si ya esta entrenado
    # para usar menos epocas
    inputs = data[0]
    desired_output = data[1]
    if(model_created_now == 1):
        epocas = 200
    else:
        epocas = 25

    log_train = trainModel(model, (inputs, desired_output), epocas)

    # salva la evolucion de la funcion loss durante entrenamiento
    if(model_created_now == 1):
        paintLoss(log_train, profiledir)

    # salva el modelo en el directorio de profile
    libaux.saveModel(model, profiledir)
    return 0


# ==========================================================================
def createModel():
    """
    crea el modelo y lo compila.

    Parameters:
    ninguno

    Returns:
    modelo compilado
        input :
        - 6 datos de iluminacion
        - 3 histogramas RGB
        output
        - 6 LUT de 256 (3 de RGB, 3 de YUV) = 1536 datos
    """
    dim = 256 * 3 + 6  # 4
    # dim = 64
    input_data = tf.keras.layers.Input(shape=(256 * 3 + 6))  # antes 4
    encoded = tf.keras.layers.Dense(dim, activation='relu6')(input_data)
    # encoded = tf.keras.layers.Dense(dim, activation='selu')(encoded)
    encoded = tf.keras.layers.Dense(dim * 2, activation='relu')(encoded)  # mejor que selu
    # encoded=  tf.keras.layers.BatchNormalization(encoded)
    decoded = tf.keras.layers.Dense(256 * 3 * 2, activation='softplus')(encoded)

    autoencoder = tf.keras.models.Model(input_data, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


# ==========================================================================
def loadTrainData(profiledir):
    """

    WARINING: ESTA FUNCION ESTA VACIA
    Esta funcion carga un fichero de training numpy desde
    el directorio de profile y retorna un array con entrada
    y salida esperada

    Parameters:
     - profiledir: directorio del profile

    Returns:
    input_data,output_data : tupla con todos los ejemplos
    None, None: si el directorio no contiene datos

    """
    input_data = np.zeros(0)
    output_data = np.zeros(0)
    return input_data, output_data


# ==========================================================================
def trainModel(model, data, epocas):
    """
    entrena un modelo con unos datos de entrada y salida

    Parameters:
    - model: el modelo compilado
    - data : las entradas data[0] y salidas data[1]

    Returns:
    modelo entrenado
    """

    # data = np.array(data)
    # np.random.shuffle(data)
    trained_model = model.fit(data[0], data[1], epochs=epocas, batch_size=4)
    # 64, # 16, # 256,
    # shuffle = True,
    # validation_data = (data[0], data[1]),
    # validation_split = 0.3
    return trained_model


# ==========================================================================
def paintLoss(log_train, profiledir):
    fig, ax = plt.subplots()
    plt.plot(log_train.history['loss'], color='r')
    # plt.legend(['Training Loss'])
    plt.title("Training Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Percent')
    # plt.show();
    filepath = profiledir + "/history_loss.jpg"
    plt.savefig(filepath)
    plt.close()
