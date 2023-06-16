import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Noramalize the pixel values by deviding each pixel by 255
x_train, x_test = x_train / 255.0, x_test / 255.0

#SKYNNET:BEGIN_MULTICLASS_ACC_LOSS

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
        raise ValueError("El numero de categorias original (n) debe ser mayor o igual al numero de arrays de destino (m).")
    
    if m > len(categorias_unicas):
        raise ValueError("El numero de categorias unicas no es suficiente para dividirlas en los m arrays deseados.")
    
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
model = []
#__CLOUDBOOK:PARALLEL__
def skynnet_block_0(i):
    global model
    model.append(None)
    _DATA_TRAIN_X = x_train
    _DATA_TRAIN_Y = y_train
    _DATA_TEST_X = x_test
    _DATA_TEST_Y = y_test
    _NEURON_1 = 86
    _NEURON_2 = 40
    _NEURON_3 = 7
    _EPOCHS = 2
    grupos_de_categorias = dividir_array_categorias(_DATA_TRAIN_Y, 10, 3)
    _DATA_TRAIN_X = _DATA_TRAIN_X[np.isin(_DATA_TRAIN_Y, combinar_arrays(grupos_de_categorias)[i])]
    _DATA_TRAIN_Y = _DATA_TRAIN_Y[np.isin(_DATA_TRAIN_Y, combinar_arrays(grupos_de_categorias)[i])]
    print(len(_DATA_TRAIN_X), len(_DATA_TRAIN_Y))
    print(np.unique(_DATA_TRAIN_Y))
    categorias_incluir = np.unique(_DATA_TRAIN_Y)
    etiquetas_consecutivas = np.arange(len(categorias_incluir))
    _DATA_TRAIN_Y = np.searchsorted(categorias_incluir, _DATA_TRAIN_Y)
    model[i] = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(_NEURON_1, activation='relu'), tf.keras.layers.Dense(_NEURON_2, activation='relu'), tf.keras.layers.Dense(_NEURON_3, activation='softmax')])
    print(model[i].summary())
    model[i].compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model[i].fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, validation_split=0.3, epochs=_EPOCHS)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_block_0(i):
    global predictions_0_0
    global model
    #__CLOUDBOOK:BEGINREMOVE__
    _DATA_TEST_X = x_test
    _DATA_TEST_Y = y_test
    __CLOUDBOOK__ = {}
    __CLOUDBOOK__['agent'] = {}
    __CLOUDBOOK__['agent']['id'] = 'agente_skynnet'
    #__CLOUDBOOK:ENDREMOVE__
    label = __CLOUDBOOK__['agent']['id'] + str(i)
    grupos_de_categorias = dividir_array_categorias(_DATA_TEST_Y, 10, 3)
    _DATA_TEST_X = _DATA_TEST_X[np.isin(_DATA_TEST_Y, combinar_arrays(grupos_de_categorias)[i])]
    _DATA_TEST_Y = _DATA_TEST_Y[np.isin(_DATA_TEST_Y, combinar_arrays(grupos_de_categorias)[i])]
    print(len(_DATA_TEST_X), len(_DATA_TEST_Y))
    print(np.unique(_DATA_TEST_Y))
    categorias_incluir = np.unique(_DATA_TEST_Y)
    etiquetas_consecutivas = np.arange(len(categorias_incluir))
    _DATA_TEST_Y = np.searchsorted(categorias_incluir, _DATA_TEST_Y)
    predictions_0_0[label] = model[i].predict(_DATA_TEST_X)


#SKYNNET:END

print("End of program")

#__CLOUDBOOK:DU0__
def skynnet_global_0():
    for i in range(3):
        skynnet_block_0(i)
    #__CLOUDBOOK:SYNC__
#__CLOUDBOOK:DU0__
def skynnet_prediction_global_0():
    for i in range(3):
        skynnet_prediction_block_0(i)
    #__CLOUDBOOK:SYNC__


#__CLOUDBOOK:MAIN__
def main():
    skynnet_global_0()
    skynnet_prediction_global_0()

if __name__ == '__main__':
    main()

