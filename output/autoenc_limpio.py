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

#SKYNNET:BEGIN_REGRESSION_LOSS

#__CLOUDBOOK:LOCAL__
def dividir_array_categorias(array, n, m):
    #n = categorias iniciales
    #m = numero de arrays resultantes
    # Obtener las categorias unicas del array original
    categorias_unicas = np.unique(array)
    
    if n < m:
        raise ValueError(f"El numero de categorias original {n} debe ser mayor o igual al numero de arrays de destino {m}.")
    
    if m > len(categorias_unicas):
        raise ValueError(f"El numero de categorias unicas {len(categorias_unicas)} no es suficiente para dividirlas en los {m} arrays deseados.")
    
    # Calcular el numero de categorias en cada array de destino
    categorias_por_array = n // m    
    # Crear los m arrays de destino
    arrays_destino = []
    inicio_categoria = 0
    
    for i in range(m):
        fin_categoria = inicio_categoria + categorias_por_array
        
        if i < n % m:#
            #print(f"		Como {i} < n({n}) % m({m}), hacemos fin_categoria = {fin_categoria+1}")
            fin_categoria += 1
        
        categorias_array_actual = categorias_unicas[inicio_categoria:fin_categoria]
        arrays_destino.append(categorias_array_actual)
        inicio_categoria = fin_categoria
    return arrays_destino

#__CLOUDBOOK:LOCAL__
def combinar_arrays(arrays):
    from itertools import combinations
    if len(arrays) < 2:
        raise ValueError("Se requieren al menos dos arrays para realizar la combinacion.")
    
    combinaciones = list(combinations(arrays, 2))
    
    arrays_combinados = []
    
    for combo in combinaciones:
        array_1, array_2 = combo
        
        # Concatenar los dos arrays en uno solo
        array_combinado = np.concatenate((array_1, array_2))
        
        arrays_combinados.append(array_combinado)
    
    return arrays_combinados

#__CLOUDBOOK:GLOBAL__
predictions_0_0 = {}
#__CLOUDBOOK:NONSHARED__
autoencoder = []
precision_compuesta = []
#__CLOUDBOOK:PARALLEL__
def skynnet_block_0(sk_i):
    global autoencoder
    autoencoder.append(None)
    ratio = 1
    data_dim_input = 784
    bottleneck = 32
    data_dim_output = data_dim_input/4
    _DATA_TRAIN_X = x_train
    _DATA_TRAIN_Y = x_train_out
    _DATA_VAL_X = x_test
    _DATA_VAL_Y = x_test_out
    _DATA_TEST_X = x_test
    _DATA_TEST_Y = x_test_out
    _NEURON_2 = 8
    _EPOCHS = 3
    #Is not neccesary to divide data
    #divido los datos por n trozos
    _DATA_TRAIN_Y_splits = np.array_split(_DATA_TRAIN_Y,4,axis=1)
    _DATA_VAL_Y_splits = np.array_split(_DATA_VAL_Y,4,axis=1)
    _DATA_TRAIN_Y = _DATA_TRAIN_Y_splits[sk_i]
    _DATA_VAL_Y = _DATA_VAL_Y_splits[sk_i]
    print(f"Shapes de: train {_DATA_TRAIN_X.shape}, train_out {_DATA_TRAIN_Y.shape}, test {_DATA_TEST_X.shape}, test_out {_DATA_TEST_Y.shape}")
    #Is not neccesary to divide data
    input_image = tf.keras.layers.Input(shape=(data_dim_input,))
    encoded_input = tf.keras.layers.Dense(_NEURON_2, activation='relu')(input_image)
    decoded_output = tf.keras.layers.Dense(data_dim_output, activation='sigmoid')(encoded_input)
    autoencoder[sk_i] = tf.keras.models.Model(input_image, decoded_output)
    autoencoder[sk_i].compile(optimizer='adam', loss='binary_crossentropy')
    print(autoencoder[sk_i].summary())
    start = time.time()
    autoencoder[sk_i].fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, epochs=_EPOCHS, batch_size=256, shuffle=True, validation_data=(_DATA_VAL_X, _DATA_VAL_Y))
    end = time.time()
    print(' tiempo de training transcurrido (segundos) =', end - start)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_block_0(sk_i):
    global predictions_0_0
    global autoencoder
    _DATA_TEST_X = x_test
    _DATA_TEST_Y = x_test_out
    #__CLOUDBOOK:BEGINREMOVE__
    __CLOUDBOOK__ = {}
    __CLOUDBOOK__['agent'] = {}
    __CLOUDBOOK__['agent']['id'] = 'agente_skynnet'
    #__CLOUDBOOK:ENDREMOVE__
    label = __CLOUDBOOK__['agent']['id'] + str(sk_i)
    prediction = autoencoder[sk_i].predict(_DATA_TEST_X, verbose=1)
    resul = prediction
    predictions_0_0[label] = resul


#SKYNNET:END

#__CLOUDBOOK:DU0__
def skynnet_global_0():
    for i in range(4):
        skynnet_block_0(i)
    #__CLOUDBOOK:SYNC__
#__CLOUDBOOK:DU0__
def skynnet_prediction_global_0():
    for i in range(4):
        skynnet_prediction_block_0(i)
    #__CLOUDBOOK:SYNC__
    global precision_compuesta
    #valores = np.array(list(predictions_0_0.values()))
    #resultado = np.prod(valores, axis=0)

    resultado = np.concatenate(list(predictions_0_0.values()), axis=1)
    print(f"Combined prediction shape: {resultado.shape}")

    bce = tf.keras.losses.BinaryCrossentropy()
    bce_orig = bce(y_test, resultado).numpy()
    print('============================================')
    print('Skynnet Info: La loss compuesta es: ', bce_orig)
    print('============================================')


#__CLOUDBOOK:MAIN__
def main():
    skynnet_global_0()
    skynnet_prediction_global_0()

if __name__ == '__main__':
    main()

