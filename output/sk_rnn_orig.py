import tensorflow as tf
#from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

# cargamos los datos de entrenamiento
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#normalizamos a 0..1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# We will make a test set of 10 samples and use the other 9990 as validation data.
# esto quita los ultimos 10 elementos (se queda con 0..9990)
x_validate, y_validate = x_test[:-10], y_test[:-10]
# esto coge los 10 ultimos (se queda con 9990..9999)
x_test, y_test = x_test[-10:], y_test[-10:]


#SKYNNET:BEGIN_MULTICLASS_ACC

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
predictions_0 = {}
#__CLOUDBOOK:NONSHARED__
model = [None, None, None]
trained_models = []
precision_compuesta = []
#__CLOUDBOOK:PARALLEL__
def skynnet_train_0(sk_i):
    global model
    _DATA_TRAIN_X = x_train
    _DATA_TRAIN_Y = y_train
    _DATA_VAL_X = x_validate
    _DATA_VAL_Y = y_validate
    _DATA_TEST_X = x_test
    _DATA_TEST_Y = y_test
    _NEURON_1 = 43
    _NEURON_2 = 7
    _EPOCHS = 7
    grupos_de_categorias = dividir_array_categorias(_DATA_TRAIN_Y, 10, 3)
    combinacion_arrays = combinar_arrays(grupos_de_categorias)[sk_i]
    _DATA_TRAIN_X = _DATA_TRAIN_X[np.isin(_DATA_TRAIN_Y, combinacion_arrays)]
    _DATA_TRAIN_Y = _DATA_TRAIN_Y[np.isin(_DATA_TRAIN_Y, combinacion_arrays)]
    print('======================================')
    print('Skynnet Info: Longitud de los datos de la subred (datos,etiquetas):', len(_DATA_TRAIN_X), len(_DATA_TRAIN_Y))
    print('Skynnet Info: Categorias de esta subred', np.unique(_DATA_TRAIN_Y))
    print('======================================')
    categorias_incluir = np.unique(_DATA_TRAIN_Y)
    etiquetas_consecutivas = np.arange(len(categorias_incluir))
    _DATA_TRAIN_Y = np.searchsorted(categorias_incluir, _DATA_TRAIN_Y)
    _NEURON_2 = len(np.unique(_DATA_TRAIN_Y))
    grupos_de_categorias = dividir_array_categorias(_DATA_VAL_Y, 10, 3)
    combinacion_arrays = combinar_arrays(grupos_de_categorias)[sk_i]
    _DATA_VAL_X = _DATA_VAL_X[np.isin(_DATA_VAL_Y, combinacion_arrays)]
    _DATA_VAL_Y = _DATA_VAL_Y[np.isin(_DATA_VAL_Y, combinacion_arrays)]
    print('======================================')
    print('Skynnet Info: Longitud de los datos de la subred (datos,etiquetas):', len(_DATA_VAL_X), len(_DATA_VAL_Y))
    print('Skynnet Info: Categorias de esta subred', np.unique(_DATA_VAL_Y))
    print('======================================')
    categorias_incluir = np.unique(_DATA_VAL_Y)
    etiquetas_consecutivas = np.arange(len(categorias_incluir))
    _DATA_VAL_Y = np.searchsorted(categorias_incluir, _DATA_VAL_Y)
    _NEURON_2 = len(np.unique(_DATA_VAL_Y))
    model[sk_i] = tf.keras.Sequential()
    model[sk_i].add(tf.keras.layers.Input(shape=(28, 28)))
    model[sk_i].add(tf.keras.layers.GRU(_NEURON_1))
    model[sk_i].add(tf.keras.layers.BatchNormalization())
    model[sk_i].add(tf.keras.layers.Dense(_NEURON_2, activation='softmax'))
    print(model[sk_i].summary())
    model[sk_i].compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    start = time.time()
    model[sk_i].fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, validation_data=(_DATA_VAL_X, _DATA_VAL_Y), batch_size=32, epochs=_EPOCHS)
    end = time.time()
    print(' tiempo de training transcurrido (segundos) =', end - start)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_0(sk_i):
    global predictions_0
    global trained_models
    global model
    _DATA_TEST_X = x_test
    _DATA_TEST_Y = y_test
    #__CLOUDBOOK:BEGINREMOVE__
    __CLOUDBOOK__ = {}
    __CLOUDBOOK__['agent'] = {}
    __CLOUDBOOK__['agent']['id'] = 'agente_skynnet'
    #__CLOUDBOOK:ENDREMOVE__
    label = __CLOUDBOOK__['agent']['id'] + str(sk_i)
    grupos_de_categorias = dividir_array_categorias(_DATA_TEST_Y, 10, 3)
    categorias_incluir = combinar_arrays(grupos_de_categorias)[sk_i]
    label += f'{categorias_incluir}'
    predicted = model[sk_i].predict(_DATA_TEST_X, verbose=1)
    categorias_str = label[label.find('[') + 1:label.find(']')]
    categorias = np.fromstring(categorias_str, dtype=int, sep=' ')
    resul = []
    for (i, pred) in enumerate(predicted):
        array_final = np.ones(10)
        array_final[categorias] = pred
        resul.append(array_final)
    predictions_0[label] = resul


#SKYNNET:END

#__CLOUDBOOK:DU0__
def skynnet_train_global_0():
    for i in range(3):
        skynnet_train_0(i)
    #__CLOUDBOOK:SYNC__
#__CLOUDBOOK:DU0__
def skynnet_prediction_global_0():
    _DATA_TEST_X = x_test
    _DATA_TEST_Y = y_test
    for i in range(3):
        skynnet_prediction_0(i)
    #__CLOUDBOOK:SYNC__
    global precision_compuesta
    valores = np.array(list(predictions_0.values()))
    predicted = np.prod(valores, axis=0)
    correctas = 0
    total = 0
    for i in range(len(y_test)):
        if y_test[i] == np.argmax(predicted[i]):
            correctas += 1
        total += 1
    precision_compuesta.append(correctas / total)
    print('============================================')
    print('Skynnet Info: La accuracy de la prediccion compuesta es: ', precision_compuesta)
    print('============================================')


#__CLOUDBOOK:MAIN__
def main():
    skynnet_train_global_0()
    skynnet_prediction_global_0()

if __name__ == '__main__':
    main()

