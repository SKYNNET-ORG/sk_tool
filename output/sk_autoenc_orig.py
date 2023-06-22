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

#__CLOUDBOOK:LOCAL__
def get_combinacion(a, n):
    from math import comb
    pairs = []
    r = 2
    c = comb(n, r)
    if c == a:
        pairs.append((n, r))
    if n > 1:
        pairs += get_combinacion(a, n - 1)
    
    return pairs

#__CLOUDBOOK:LOCAL__
def get_categorias(subredes, categorias):
    results = {}
    for subred in range(subredes,1,-1):
        pairs = get_combinacion(subred,categorias)
        for i in pairs:
            #results.append(i)
            results[subred]=i
    #print(results)
    #devuelvo el numero deseado si es posible, o uno menor, para que quede una maquina libre
    if subredes in results:
        n_subredes = subredes
    else:
        n_subredes = max(results.keys())

    return n_subredes,results[n_subredes]

#__CLOUDBOOK:LOCAL__
def dividir_array_categorias(array, n, m):
    #n = categorias iniciales
    #m = numero de arrays resultantes
    # Obtener las categorias unicas del array original
    #print(f"Dividiendo un array de {n} categorias en {m} arrays con menos categorias")
    categorias_unicas = np.unique(array)
    #print(f"Tenemos {categorias_unicas} categorias unicas")
    
    if n < m:
        raise ValueError(f"El numero de categorias original {n} debe ser mayor o igual al numero de arrays de destino {m}.")
    
    if m > len(categorias_unicas):
        raise ValueError(f"El numero de categorias unicas {len(categorias_unicas)} no es suficiente para dividirlas en los {m} arrays deseados.")
    
    # Mezclar las categorias unicas de forma aleatoria
    #np.random.shuffle(categorias_unicas)
    
    # Calcular el numero de categorias en cada array de destino
    categorias_por_array = n // m
    #print(f"Categorias por array = {categorias_por_array}")
    
    # Crear los m arrays de destino
    arrays_destino = []
    inicio_categoria = 0
    
    for i in range(m):
        #print(f"Para el subarray {i}")
        fin_categoria = inicio_categoria + categorias_por_array
        #print(f"	Con incicio de categoria = {inicio_categoria} y fin de categoria = {fin_categoria}")
        
        if i < n % m:#
            #print(f"		Como {i} < n({n}) % m({m}), hacemos fin_categoria = {fin_categoria+1}")
            fin_categoria += 1
        
        categorias_array_actual = categorias_unicas[inicio_categoria:fin_categoria]
        #print(f"	Categorias array actual = {categorias_array_actual}")
        # Filtrar el array original para obtener los elementos de las categorias del array actual
        #array_actual = array[np.isin(array, categorias_array_actual)]
        #print(f"	Tras filtrar el aaray original para formar array actual queda: {array_actual}")
        #arrays_destino.append(array_actual)
        arrays_destino.append(categorias_array_actual)
        inicio_categoria = fin_categoria
        #print(f"	Se mete el array actual en arrays_destino quedando {arrays_destino}")
        #print(f"	Se hace inicio_categoria = fin_categoria: {inicio_categoria}={fin_categoria}")
    return arrays_destino

#__CLOUDBOOK:LOCAL__
def combinar_arrays(arrays):
    from itertools import combinations
    if len(arrays) < 2:
        raise ValueError("Se requieren al menos dos arrays para realizar la combinacion.")
    
    combinaciones = list(combinations(arrays, 2))
    #print(f"Tenemos una lista con todas las combinaciones de los arrays tomados de 2 en 2: {combinaciones}")
    
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
#__CLOUDBOOK:PARALLEL__
def skynnet_block_0(i):
    global autoencoder
    autoencoder.append(None)
    ratio = 1
    data_dim_input = 784
    bottleneck = 32
    data_dim_output = data_dim_input
    _DATA_TRAIN_X = x_train
    _DATA_TRAIN_Y = x_train_out
    _DATA_VAL_X = x_test
    _DATA_VAL_Y = x_test_out
    _DATA_TEST_X = x_test
    _DATA_TEST_Y = x_test_out
    _NEURON_1 = 2
    _NEURON_2 = 196
    _EPOCHS = 3
    #TODO
    #TODO
    input_image = tf.keras.layers.Input(shape=(data_dim_input,))
    encoded_input = tf.keras.layers.Dense(bottleneck, activation='relu')(input_image)
    decoded_output = tf.keras.layers.Dense(data_dim_output, activation='sigmoid')(encoded_input)
    autoencoder[i] = tf.keras.models.Model(input_image, decoded_output)
    autoencoder[i].compile(optimizer='adam', loss='binary_crossentropy')
    start = time.time()
    autoencoder[i].fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, epochs=_EPOCHS, batch_size=256, shuffle=True, validation_data=(_DATA_VAL_X, _DATA_VAL_Y))
    end = time.time()
    print(' tiempo de training transcurrido (segundos) =', end - start)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_block_0(i):
    global predictions_0_0
    global autoencoder
    #__CLOUDBOOK:BEGINREMOVE__
    _DATA_TEST_X = x_test
    _DATA_TEST_Y = x_test_out
    __CLOUDBOOK__ = {}
    __CLOUDBOOK__['agent'] = {}
    __CLOUDBOOK__['agent']['id'] = 'agente_skynnet'
    #__CLOUDBOOK:ENDREMOVE__
    label = __CLOUDBOOK__['agent']['id'] + str(i)
    #TODO
    predictions_0_0[label] = autoencoder[i].predict(_DATA_TEST_X)


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


#__CLOUDBOOK:MAIN__
def main():
    skynnet_global_0()
    skynnet_prediction_global_0()

if __name__ == '__main__':
    main()

