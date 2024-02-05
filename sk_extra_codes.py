

funciones_combinatorias='''
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
            #print(f"\t\tComo {i} < n({n}) % m({m}), hacemos fin_categoria = {fin_categoria+1}")
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
'''

def division_datos_fit_multiclass(categorias,grupos,datos_x,datos_y,last_neuron):
    division_datos = f'''grupos_de_categorias = dividir_array_categorias({datos_y},{categorias},{grupos})
combinacion_arrays = combinar_arrays(grupos_de_categorias)[sk_i]
{datos_x} = {datos_x}[np.isin({datos_y},combinacion_arrays)]
{datos_y} = {datos_y}[np.isin({datos_y},combinacion_arrays)]
print("======================================")
print("Skynnet Info: Longitud de los datos de la subred (datos,etiquetas):",len({datos_x}),len({datos_y}))
print("Skynnet Info: Categorias de esta subred",np.unique({datos_y}))
print("======================================")
categorias_incluir = np.unique({datos_y})
etiquetas_consecutivas = np.arange(len(categorias_incluir))
{datos_y} = np.searchsorted(categorias_incluir, {datos_y})
{last_neuron[0]} = len(np.unique({datos_y}))
'''
    return division_datos

def division_datos_fit_binaryclass(tipo_datos,datos_x,datos_y,last_neuron):
    division_datos = f'''
datos_{tipo_datos}_x_1 = {datos_x}[:len({datos_x})//2]
datos_{tipo_datos}_x_2 = {datos_x}[len({datos_x})//2:]
datos_{tipo_datos}_y_1 = {datos_y}[:len({datos_y})//2]
datos_{tipo_datos}_y_2 = {datos_y}[len({datos_y})//2:]
if sk_i == 1:
    {datos_x} = datos_{tipo_datos}_x_1
    {datos_y} = datos_{tipo_datos}_y_1
else:
    {datos_x} = datos_{tipo_datos}_x_2
    {datos_y} = datos_{tipo_datos}_y_2
{last_neuron[0]} = 2
        '''
    return division_datos

def division_datos_fit_regression(grupos,tipo_datos,datos_y, last_neuron):
    division_datos = f'''#Is not neccesary to divide data
{tipo_datos}_splits = np.array_split({datos_y},{grupos},axis=1)
{datos_y} = {tipo_datos}_splits[sk_i]
{last_neuron[0]} = {datos_y}.shape[-1]#El tam de la ultima dimension'''
    return division_datos


def preparacion_datos_predict_multiclass(medida_compuesta,predict_name,model_name,datos_y,categorias,grupos):
    if medida_compuesta == "":
        preparacion_datos = f'''
'''+f"{predict_name}"+''' = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
resul = '''+f"{predict_name}"+'''
'''
    else:
        preparacion_datos = f'''grupos_de_categorias = dividir_array_categorias({datos_y},{categorias},{grupos})
categorias_incluir = combinar_arrays(grupos_de_categorias)[sk_i]
label+=f"'''+"{categorias_incluir}"+'''"
'''+f"{predict_name}"+''' = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
categorias_str = label[label.find("[")+1:label.find("]")]
categorias = np.fromstring(categorias_str, dtype=int,sep=' ')
resul = []
for i,pred in enumerate('''+f"{predict_name}"+'''):
    array_final = np.ones('''+f"{categorias}"+''')
    array_final[categorias] = pred
    resul.append(array_final)
'''
    return preparacion_datos

def preparacion_datos_predict_binaryclass(medida_compuesta,predict_name,categorias,model_name):
    if medida_compuesta == "":
        preparacion_datos = f'''
'''+f"{predict_name}"+''' = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
resul = '''+f"{predict_name}"+'''
'''
    else:
        preparacion_datos = f'''
'''+f"{predict_name}"+''' = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
categorias = [0,1]
resul = []
for i,pred in enumerate('''+f"{predict_name}"+'''):
    array_final = np.ones('''+f"{categorias}"+''')
    array_final[categorias] = pred
    resul.append(array_final)
'''
    return preparacion_datos

def preparacion_datos_predict_regression(predict_name,model_name,categorias):
    preparacion_datos = f'''
'''+f"{predict_name}"+''' = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
resul = '''+f"{predict_name}"+'''
'''
    return preparacion_datos

def backup_division_datos_predict(tipo_datos,categorias,grupos,last_neuron,tipo_red,model_name,medida_compuesta, predict_name):
    '''Ya no es division de datos, es la adaptacion del predict para sacar el compuesto
    Medida compuesta puede ser: "acc", "loss","acc,loss"'''
    if tipo_datos == "train":
        datos_x = "_DATA_TRAIN_X"
        datos_y = "_DATA_TRAIN_Y"
    elif tipo_datos== "validate":
        datos_x = "_DATA_VAL_X"
        datos_y = "_DATA_VAL_Y"
    elif tipo_datos == "test":
        datos_x = "_DATA_TEST_X"
        datos_y = "_DATA_TEST_Y"
    else:
        print("Warning unknown data type")
        return None
    if tipo_red == 'MULTICLASS':#MANDO SIEMPRE ACCURACY ADAPT PORQUE SIEMPRE HAY QUE EXPANDIR LA PREDICCION
        no_measure = f'''
'''+f"{predict_name}"+''' = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
resul = '''+f"{predict_name}"+'''
'''
        accuracy_adapt = f'''grupos_de_categorias = dividir_array_categorias({datos_y},{categorias},{grupos})
categorias_incluir = combinar_arrays(grupos_de_categorias)[sk_i]
label+=f"'''+"{categorias_incluir}"+'''"
'''+f"{predict_name}"+''' = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
categorias_str = label[label.find("[")+1:label.find("]")]
categorias = np.fromstring(categorias_str, dtype=int,sep=' ')
resul = []
for i,pred in enumerate('''+f"{predict_name}"+'''):
    array_final = np.ones('''+f"{categorias}"+''')
    array_final[categorias] = pred
    resul.append(array_final)
'''
        loss_adapt = f'''
#Loss adapt: Not neccesary in this function, is in the global
'''
        if medida_compuesta == "acc,loss":
            division_datos = accuracy_adapt+loss_adapt
        elif medida_compuesta == "acc":
            division_datos = accuracy_adapt
        elif medida_compuesta == "loss":
            division_datos = accuracy_adapt#no_measure+loss_adapt
        else:
            division_datos = no_measure
    elif tipo_red == 'BINARYCLASS':
        no_measure = f'''
'''+f"{predict_name}"+''' = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
resul = '''+f"{predict_name}"+'''
'''
        accuracy_adapt = f'''
'''+f"{predict_name}"+''' = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
categorias = [0,1]
resul = []
for i,pred in enumerate('''+f"{predict_name}"+'''):
    array_final = np.ones('''+f"{categorias}"+''')
    array_final[categorias] = pred
    resul.append(array_final)
'''
        loss_adapt = f'''
#MSSE measure to get
'''
        if medida_compuesta == "acc,loss":
            division_datos = accuracy_adapt#accuracy_adapt+loss_adapt
        elif medida_compuesta == "acc":
            division_datos = accuracy_adapt
        elif medida_compuesta == "loss":
            division_datos = accuracy_adapt#no_measure+loss_adapt
        else:
            division_datos = no_measure
    elif tipo_red == 'REGRESSION':
        division_datos = f'''
'''+f"{predict_name}"+''' = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
resul = '''+f"{predict_name}"+'''
'''
        division_datos_not = f'''
'''+f"{predict_name}"+''' = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
categorias = [0,1]
resul = []
for i,pred in enumerate('''+f"{predict_name}"+'''):
    array_final = np.ones('''+f"{categorias}"+''')
    array_final[categorias] = pred
    resul.append(array_final)
'''
    return fix_missing_locations(parse(division_datos))