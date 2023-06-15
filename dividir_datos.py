from math import comb
from itertools import combinations
import numpy as np
import tensorflow as tf, numpy as np

def find_r_old(a, n):
    pairs = []
    r = n
    while r > 1: #r=1 puede dar falso positivo 
        c = comb(n, r)
        if c == a:
            pairs.append((n, r))
        r -= 1

    if n > 1:
        pairs += find_r(a, n - 1)
    
    return pairs

def get_combinacion(a, n):
    pairs = []
    r = 2
    c = comb(n, r)
    if c == a:
        pairs.append((n, r))
    if n > 1:
        pairs += get_combinacion(a, n - 1)
    
    return pairs

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

def dividir_array_categorias(array, n, m):
    #n = categorias iniciales
    #m = numero de arrays resultantes
    # Obtener las categorías únicas del array original
    #print(f"Dividiendo un array de {n} categorias en {m} arrays con menos categorias")
    categorias_unicas = np.unique(array)
    #print(f"Tenemos {categorias_unicas} categorias unicas")
    
    if n < m:
        raise ValueError("El número de categorías original (n) debe ser mayor o igual al número de arrays de destino (m).")
    
    if m > len(categorias_unicas):
        raise ValueError("El número de categorías únicas no es suficiente para dividirlas en los m arrays deseados.")
    
    # Mezclar las categorías únicas de forma aleatoria
    #np.random.shuffle(categorias_unicas)
    
    # Calcular el número de categorías en cada array de destino
    categorias_por_array = n // m
    #print(f"Categorias por array = {categorias_por_array}")
    
    # Crear los m arrays de destino
    arrays_destino = []
    inicio_categoria = 0
    
    for i in range(m):
        #print(f"Para el subarray {i}")
        fin_categoria = inicio_categoria + categorias_por_array
        #print(f"\tCon incicio de categoria = {inicio_categoria} y fin de categoria = {fin_categoria}")
        
        if i < n % m:#
            #print(f"\t\tComo {i} < n({n}) % m({m}), hacemos fin_categoria = {fin_categoria+1}")
            fin_categoria += 1
        
        categorias_array_actual = categorias_unicas[inicio_categoria:fin_categoria]
        #print(f"\tCategorias array actual = {categorias_array_actual}")
        # Filtrar el array original para obtener los elementos de las categorías del array actual
        #array_actual = array[np.isin(array, categorias_array_actual)]
        #print(f"\tTras filtrar el aaray original para formar array actual queda: {array_actual}")
        #arrays_destino.append(array_actual)
        arrays_destino.append(categorias_array_actual)
        inicio_categoria = fin_categoria
        #print(f"\tSe mete el array actual en arrays_destino quedando {arrays_destino}")
        #print(f"\tSe hace inicio_categoria = fin_categoria: {inicio_categoria}={fin_categoria}")
    return arrays_destino

def combinar_arrays(arrays):
    if len(arrays) < 2:
        raise ValueError("Se requieren al menos dos arrays para realizar la combinación.")
    
    combinaciones = list(combinations(arrays, 2))
    #print(f"Tenemos una lista con todas las combinaciones de los arrays tomados de 2 en 2: {combinaciones}")
    
    arrays_combinados = []
    
    for combo in combinaciones:
        array_1, array_2 = combo
        
        # Concatenar los dos arrays en uno solo
        array_combinado = np.concatenate((array_1, array_2))
        
        arrays_combinados.append(array_combinado)
    
    return arrays_combinados

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Noramalize the pixel values by deviding each pixel by 255
x_train, x_test = x_train / 255.0, x_test / 255.0



subredes = 7
categorias = 10
subredes_def,combinatorio = get_categorias(subredes,categorias)
print(f"Para {subredes} subredes y {categorias} categorias:")
print(f"  Subredes definitivas: {subredes_def}, {combinatorio[0]} grupos tomados de {combinatorio[1]}, cada subred tendra {(categorias/combinatorio[0])*combinatorio[1]} categorias")

print(f"Voy a dividir las {categorias} categorias iniciales en los {combinatorio[0]} grupos finales")
grupos_de_categorias = dividir_array_categorias(y_train,categorias,combinatorio[0])

for i, array in enumerate(grupos_de_categorias):
    print(f"Array {i + 1}: {array}, {len(array)}")

print(f"Ahora hay que combinar los grupos de categorias en las subredes finales")
categorias_subredes_finales = combinar_arrays(grupos_de_categorias)
for i, array_combinado in enumerate(categorias_subredes_finales):
    print(f"Array combinado {i + 1}: {array_combinado} con {len(np.unique(array_combinado))} categorias {np.unique(array_combinado)}")

print(f"Ahora hay que generar loas arrays correctamente filtrados")

_DATA_TRAIN =(x_train,y_train)
_DATA_TEST=(x_test,y_test)

for i in range(subredes_def):
    print(f"Para la subred {i} queremos las categorias {categorias_subredes_finales[i]}")
    _DATA_TRAIN_X_TEMP = _DATA_TRAIN[0][np.isin(_DATA_TRAIN[1],combinar_arrays(grupos_de_categorias)[i])]
    _DATA_TRAIN_Y_TEMP = _DATA_TRAIN[1][np.isin(_DATA_TRAIN[1],categorias_subredes_finales[i])]
    print(len(_DATA_TRAIN_X_TEMP),len(_DATA_TRAIN_Y_TEMP))
    _DATA_TRAIN_TEMP = _DATA_TRAIN_X_TEMP,_DATA_TRAIN_Y_TEMP

'''Para sk_tool copiar esto:
    grupos_de_categorias = dividir_array_categorias(y_train,10,3)
    print("================================================")
    print(f"Para la subred {i} queremos las categorias {combinar_arrays(grupos_de_categorias)[i]}")
    print(f"Estas son las {len(combinar_arrays(grupos_de_categorias)[i])} categorias que filtramos {combinar_arrays(grupos_de_categorias)[i]}")
    x_train_temp = x_train[np.isin(y_train,combinar_arrays(grupos_de_categorias)[i])]
    y_train_temp = y_train[np.isin(y_train,combinar_arrays(grupos_de_categorias)[i])]
    print(len(x_train_temp),len(y_train_temp))
    print(np.unique(y_train_temp))
    _NEURON_3 = len(np.unique(y_train_temp))
    print(f"la ultima capa tiene {_NEURON_3} categorias")
    categorias_incluir = np.unique(y_train_temp)
    etiquetas_consecutivas = np.arange(len(categorias_incluir))
    y_train_temp = np.searchsorted(categorias_incluir, y_train_temp)'''