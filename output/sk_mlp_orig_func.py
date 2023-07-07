import tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
# Load MNIST data using built-in datasets download function
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Noramalize the pixel values by deviding each pixel by 255
x_train, x_test = x_train / 255.0, x_test / 255.0

#SKYNNET:BEGIN_MULTICLASS_ACC_LOSS

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
model = []
#__CLOUDBOOK:PARALLEL__
def skynnet_block_0(sk_i):
    global model
    model.append(None)
    _DATA_TRAIN_X = x_train
    _DATA_TRAIN_Y = y_train
    _DATA_TEST_X = x_test
    _DATA_TEST_Y = y_test
    _NEURON_1 = 64
    _NEURON_2 = 30
    _NEURON_3 = 5
    _EPOCHS = 5
    grupos_de_categorias = dividir_array_categorias(_DATA_TRAIN_Y, 10, 4)
    _DATA_TRAIN_X = _DATA_TRAIN_X[np.isin(_DATA_TRAIN_Y, combinar_arrays(grupos_de_categorias)[sk_i])]
    _DATA_TRAIN_Y = _DATA_TRAIN_Y[np.isin(_DATA_TRAIN_Y, combinar_arrays(grupos_de_categorias)[sk_i])]
    print(len(_DATA_TRAIN_X), len(_DATA_TRAIN_Y))
    print(np.unique(_DATA_TRAIN_Y))
    categorias_incluir = np.unique(_DATA_TRAIN_Y)
    etiquetas_consecutivas = np.arange(len(categorias_incluir))
    _DATA_TRAIN_Y = np.searchsorted(categorias_incluir, _DATA_TRAIN_Y)
    _NEURON_3 = len(np.unique(_DATA_TRAIN_Y))
    inputs = tf.keras.Input(shape=(28, 28))
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(_NEURON_1, activation='relu')(x)
    x = tf.keras.layers.Dense(_NEURON_2, activation='relu')(x)
    outputs = tf.keras.layers.Dense(_NEURON_3, activation='softmax')(x)
    model[sk_i] = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model[sk_i].summary())
    model[sk_i].compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model[sk_i].fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, validation_split=0.3, epochs=_EPOCHS)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_block_0(sk_i):
    global predictions_0_0
    global model
    _DATA_TEST_X = x_test
    _DATA_TEST_Y = y_test
    #__CLOUDBOOK:BEGINREMOVE__
    __CLOUDBOOK__ = {}
    __CLOUDBOOK__['agent'] = {}
    __CLOUDBOOK__['agent']['id'] = 'agente_skynnet'
    #__CLOUDBOOK:ENDREMOVE__
    label = __CLOUDBOOK__['agent']['id'] + str(sk_i)
    grupos_de_categorias = dividir_array_categorias(_DATA_TEST_Y, 10, 4)
    #_DATA_TEST_X = _DATA_TEST_X[np.isin(_DATA_TEST_Y, combinar_arrays(grupos_de_categorias)[sk_i])]
    #_DATA_TEST_Y = _DATA_TEST_Y[np.isin(_DATA_TEST_Y, combinar_arrays(grupos_de_categorias)[sk_i])]
    #print(len(_DATA_TEST_X), len(_DATA_TEST_Y))
    #print(np.unique(_DATA_TEST_Y))
    categorias_incluir = combinar_arrays(grupos_de_categorias)[sk_i]
    #etiquetas_consecutivas = np.arange(len(categorias_incluir))
    #_DATA_TEST_Y = np.searchsorted(categorias_incluir, _DATA_TEST_Y)
    #_NEURON_3 = len(np.unique(_DATA_TEST_Y))
    label+=f"_{categorias_incluir}"
    prediction = model[sk_i].predict(_DATA_TEST_X)
    #predictions_0_0[label] = model[sk_i].predict(_DATA_TEST_X)
    categorias_str = label[label.find("[")+1:label.find("]")]
    categorias = np.fromstring(categorias_str, dtype=int,sep=' ')
    resul = []
    for i,pred in enumerate(prediction):
        #print(f"prediccion individual {pred}")
        array_final = np.ones(10)
        array_final[categorias] = pred
        resul.append(array_final)
    predictions_0_0[label] = resul

#SKYNNET:END

print("End of program")

#__CLOUDBOOK:DU0__
def skynnet_global_0():
    for i in range(6):
        skynnet_block_0(i)
    #__CLOUDBOOK:SYNC__
#__CLOUDBOOK:DU0__
def skynnet_prediction_global_0():
    for i in range(6):
        skynnet_prediction_block_0(i)
    #__CLOUDBOOK:SYNC__


#__CLOUDBOOK:MAIN__
def main():
    skynnet_global_0()
    skynnet_prediction_global_0()
    print("Empezamos con las predicciones")
    print("==============================")
    #resultado = None
    for prediction in predictions_0_0.keys():
        print(f"{len(y_test)}, {len(y_test.shape)}")
        print(f"prediction: {prediction}")
        #resultado = tf.multiply(prediction)
        for i in range(10):
            prediccion = predictions_0_0[prediction]
            print(f"Valor {y_test[i]} se predice {np.argmax(prediccion[i])}")
    valores = np.array(list(predictions_0_0.values()))
    resultado = np.prod(valores,axis=0)
    print(resultado)
    # Redondear las predicciones a la clase más probable
    predicciones_clases = np.argmax(resultado, axis=1)
    # Comparar con las etiquetas reales
    #etiquetas_reales = np.argmax(y_test, axis=1)
    # Calcular la precisión
    precision = np.mean(predicciones_clases == y_test)

    print("Precisión:", precision)
    '''lista_predicciones = list(predictions_0_0.keys())
    resultado = predictions_0_0[lista_predicciones[0]]
    lista_predicciones = lista_predicciones[1:]
    for prediccion in lista_predicciones:
        resultado = np.multiply(predictions_0_0[prediccion],resultado)
    print(f"{len(y_test)}, {len(y_test.shape)}")
    print(f"{len(resultado)}, {len(resultado)}")
    for i in range(10):
        print(f"Valor {y_test[i]} se predice {np.argmax(resultado[i])}")
        print(f"Valor {y_test[i]} se predice {resultado[i]}")
'''
if __name__ == '__main__':
    main()

