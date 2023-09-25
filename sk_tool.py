from ast import *
import sys
import re
from ast_comments import *
from math import comb,ceil


#Variables globales
sk_vars_list = ["_EMBEDDING_","_NEURON_","_CELL_","_EPOCHS","_FILTERS_"]
sk_functions_list = ['summary','compile','fit','predict',]
sk_creation_model_list = ['Sequential','Model']

num_subredes = 0

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

def get_var_from_list(cadena, lista):
    '''Esta funcion se usa para reconocer las variables especificas para capas de neuronas que pedimos que ponga el diseñador
    Esta funcion recibe un string de una variable y una lista de variables
    comprueba si las variable pertenece a la cadena, como a veces la variable
    incluye un _numero para indicar el numero de la capa en la que esta, primero lo separa, para asegurar que la variable sin numero es una de las de la lista que nos interesan

    Devuelve una tupla (terna): 
    - True o False: Si la variable es de las de la lista
    - El nombre de la variable sin el numero
    - El numero de la variable, que puede ser:
        - n, un numero entero
        - 0, si la variable no tiene numero
        - -1 si la variable no era de la lista
    '''
    #Separo el numero con er y tengo var y numero (si hay)
    hay_num = re.search(r'\d+',cadena) 
    if hay_num:
        start_num = hay_num.start()
        end_num = hay_num.end()
        sk_var = cadena[:start_num]
        num = cadena[start_num:end_num]
        if sk_var in lista:
            return True,sk_var,int(num)
        else:
            return False,sk_var,int(num)
    else:
        sk_var = cadena
        if sk_var in lista:
            return True,sk_var,0
        else:
            return False,sk_var,-1

class TransformAssignSkVars(ast.NodeTransformer):
    '''Esta clase es la que se usa para detectar de forma automatica las variables de las 
    redes neuronales y reducir el valor que se les da para adecuarlas a las respectivas subredes
    por ejemplo: _NEURON_3 = 100 ------->  _NEURON_3 = 50, si se divide en dos subredes'''

    def __init__(self,reduccion=1):
        self.reduccion = reduccion #Indica por cuanto hay que dividir el valor de la asignacion

    def visit_Assign(self, node):
        '''Esta funcion es la que divide por un numero entero las variables de skynnet
        '''
        if len(node.targets)==1:
            variable_valida =  isinstance(node.targets[0], ast.Name) and node.value
            if variable_valida:
                variable_skynnet = get_var_from_list(node.targets[0].id, sk_vars_list)[0]==True
                if variable_valida and variable_skynnet:
                    new_value = ast.Constant(value=(ceil(node.value.value/self.reduccion)))
                    if new_value.value == 0:
                        print(f"WARNING: Deconstruction on variable {node.targets[0].id} is gonna be reduced to 0")
                    new_node = ast.Assign(targets=node.targets, value=new_value, lineno = node.lineno)
                    return new_node
                else:
                    # Si no es una asignación de un solo objetivo con un valor, simplemente devuelve el nodo original sin modificarlo
                    #print("Nodo erroneo", node.targets[0].id)
                    return node
            else:
                return node
        else:
            return node

class VisitModelFunctions(ast.NodeVisitor):
    '''Esta clase es la que permite obtener las asignaciones que se usan para crear modelos en skynnet,
    ya sea con el modelo normal o funcional. Ademas prepara el diccionario que tendra un resumen de los nodos
    para las funciones summary, creation, compile, fit, si añadimos alguna funcion nueva se añade al diccionario de forma automatica en otra clase'''

    def __init__(self):
        self.dict_modelos = {}

    @staticmethod
    def get_sk_model_creation(llamada):
        for function in sk_creation_model_list:
            if function in llamada:
                return True
            else:
                continue
        return False


    def visit_Assign(self,node):
        #nodo valido un nombre igual a una invocacion
        nodo_valido_izqda = len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        nodo_valido_dcha = isinstance(node.value, ast.Call)
        if nodo_valido_izqda and nodo_valido_dcha:
            llamada = node.value
            if self.get_sk_model_creation(ast.unparse(llamada)):
                model_name = node.targets[0].id
                self.dict_modelos[model_name] = {'creation': node }
            self.generic_visit(node)

    
    def visit_Call(self,node):#funciones tipo model.fit(lo que sea)
        #nodo valido, si node.func es un atributo (de model)
        #y esta en la lista de funciones que queremos para skynnet
        if isinstance(node.func, ast.Attribute) and node.func.attr in sk_functions_list:
            if isinstance(node.func.value, ast.Name) and node.func.value.id in self.dict_modelos.keys():
                self.dict_modelos[node.func.value.id][node.func.attr] = node
        
        self.generic_visit(node)

class getInputModel(ast.NodeVisitor):
    '''meto en el dict, el input del diccionario'''
    def __init__(self,model_name,dict_modelos):
        self.dict_modelos = dict_modelos
        self.model_name = model_name
        #self.dict_modelos[self.model_name] = {}
    
    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "Input":
                self.dict_modelos[self.model_name]["input"] = node
        
        self.generic_visit(node)

    '''def visit_Call(self,node):
        if isinstance(node.func, ast.Attribute) and node.func.attr =="Input":
            self.dict_modelos[self.model_name]["input"] = node'''

class RemovePredictionNode(ast.NodeTransformer):
    '''Elimino la asignacion predicted = model.predict(x), solo las que tengan esa
    forma especifica'''

    def __init__(self,dict_modelos):
        self.dict_modelos = dict_modelos

    def visit_Assign(self, node):
        nodo_valido_izqda = len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        nodo_valido_dcha = isinstance(node.value, ast.Call)
        if nodo_valido_izqda and nodo_valido_dcha:
            derecha = node.value
            if isinstance(derecha.func, ast.Attribute) and derecha.func.attr == 'predict':
                if isinstance(derecha.func.value, ast.Name) and derecha.func.value.id in self.dict_modelos.keys():
                    return None
                else:
                    return node
            else:
                return node
        else:
            return node

class ModelArrayTransform(ast.NodeTransformer):
    '''Esta clase es para cambiar los modelos por arrays de modelos, 
    cambia todas las apariciones del modelo'''
    def __init__(self, model_name):
        self.model_name = model_name

    def visit_Name(self, node):
        if node.id == self.model_name:
            return ast.Subscript(
                value=ast.Name(id= self.model_name, ctx=ast.Load()),
                slice=ast.Index(value=ast.Name(id='sk_i', ctx=ast.Load())),
                ctx=node.ctx
            )       
        return node#self.generic_visit(node)

class VisitLastNeuron(ast.NodeVisitor):
    '''Esta clase es para obtener el numero de categorias en las que se clasifica
    TODO: Mezclar esta clase con la que reduce los datos'''
    max_valor = 0
    n_categorias = 0
    last_neuron = ()

    def visit_Assign(self, node):
        '''Esta funcion es la que divide por un numero entero las variables de skynnet
        '''
        variable_valida =  (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.value)
        if variable_valida:
            existe_sk_var,variable_skynnet,valor = get_var_from_list(node.targets[0].id, sk_vars_list)          
            if existe_sk_var == True and variable_skynnet == '_NEURON_':
                #print(f"node.value es {node.value.value}")
                if valor>self.max_valor:
                    self.last_neuron = ()
                    self.max_valor = valor
                    self.n_categorias = node.value.value
                    self.last_neuron = (node.targets[0].id,self.n_categorias)

class GetModelDataVars(ast.NodeVisitor):
    '''Uso esta clase para guardar en el diccionario las variables de datos del modelo'''
    data_vars_train = ["_DATA_TRAIN_X","_DATA_TRAIN_Y"]
    data_vars_val = ["_DATA_VAL_X","_DATA_VAL_Y"]
    data_vars_test = ["_DATA_TEST_X","_DATA_TEST_Y"]

    def __init__(self,model_name, dict_modelos):
        self.dict_modelos = dict_modelos
        self.model_name = model_name
        self.dict_modelos[self.model_name]["data_train"] = []
        self.dict_modelos[self.model_name]["data_val"] = []
        self.dict_modelos[self.model_name]["data_test"] = []

    def visit_Assign(self,node):

        if len(node.targets)==1 and node.targets[0].id in self.data_vars_train:
            self.dict_modelos[self.model_name]["data_train"].append(node)
        if len(node.targets)==1 and node.targets[0].id in self.data_vars_val:
            self.dict_modelos[self.model_name]["data_val"].append(node)
        if len(node.targets)==1 and node.targets[0].id in self.data_vars_test:
            self.dict_modelos[self.model_name]["data_test"].append(node)

def division_datos_fit(tipo_datos,categorias,grupos,last_neuron,tipo_red):
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
    if tipo_red == 'MULTICLASS':
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
    elif tipo_red == 'BINARYCLASS':
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
    elif tipo_red == 'REGRESSION':
        division_datos = f'''#Is not neccesary to divide data'''
    return fix_missing_locations(parse(division_datos))

def division_datos_predict(tipo_datos,categorias,grupos,last_neuron,tipo_red,model_name,medida_compuesta):
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
prediction = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
resul = prediction
'''
        accuracy_adapt = f'''grupos_de_categorias = dividir_array_categorias({datos_y},{categorias},{grupos})
categorias_incluir = combinar_arrays(grupos_de_categorias)[sk_i]
label+=f"'''+"{categorias_incluir}"+'''"
prediction = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
categorias_str = label[label.find("[")+1:label.find("]")]
categorias = np.fromstring(categorias_str, dtype=int,sep=' ')
resul = []
for i,pred in enumerate(prediction):
    array_final = np.ones('''+f"{categorias}"+''')
    array_final[categorias] = pred
    resul.append(array_final)
'''
        loss_adapt = f'''
#MSSE measure to get
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
prediction = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
resul = prediction
'''
        accuracy_adapt = f'''
prediction = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
categorias = [0,1]
resul = []
for i,pred in enumerate(prediction):
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
prediction = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
resul = prediction
'''
        division_datos_not = f'''
prediction = '''+f"{model_name}"+'''[sk_i].predict(_DATA_TEST_X, verbose=1)
categorias = [0,1]
resul = []
for i,pred in enumerate(prediction):
    array_final = np.ones('''+f"{categorias}"+''')
    array_final[categorias] = pred
    resul.append(array_final)
'''
    return fix_missing_locations(parse(division_datos))

def inserta_nodo(sk_dict,model_name,to_insert_node,node_destiny,last_neuron):
    
    nodo_a_buscar = sk_dict[model_name]["creation"]
    nodo_a_buscar = sk_dict[model_name]["input"]
    if isinstance(node_destiny, ast.AST):
        for child in ast.iter_child_nodes(node_destiny):
            #print("child node es", child)
            if child == nodo_a_buscar:
                #print(f"ENCONTRADO en {unparse(child)}")
                index = node_destiny.body.index(child)
                node_destiny.body.insert(index, to_insert_node)
                #print("Nodo insertado")
                break
            else:
                inserta_nodo(sk_dict, model_name, to_insert_node, child,last_neuron)
    elif isinstance(node_destiny, list):
        for child in node_destiny:
            inserta_nodo(sk_dict, model_name, to_insert_node, child,last_neuron)



def inserta_filtro_datos(nodo_destino,tipo_funcion,sk_dict,categorias,grupos,last_neuron,tipo_red, medida_compuesta):
    '''tipo funcion: general(train y val) predict(test), general es la division para entrenar
    nodo_destino: nodo tipo funcion en el que hay que insertar'''
    if tipo_funcion=="general":
        for model_name in sk_dict.keys():
            if sk_dict[model_name]["data_train"] != []:
                to_insert_node = division_datos_fit("train",categorias,grupos,last_neuron,tipo_red)
                inserta_nodo(sk_dict,model_name,to_insert_node,nodo_destino,last_neuron)
            if sk_dict[model_name]["data_val"] != []:
                to_insert_node = division_datos_fit("validate",categorias,grupos,last_neuron,tipo_red)
                inserta_nodo(sk_dict,model_name,to_insert_node,nodo_destino,last_neuron)
    elif tipo_funcion == "predict":
        for model_name in sk_dict.keys():
            if sk_dict[model_name]["data_test"] != []:
                #to_insert_node = division_datos_fit("test",categorias,grupos,last_neuron,tipo_red)
                to_insert_node = division_datos_predict("test",categorias,grupos,last_neuron,tipo_red,model_name, medida_compuesta)
                nodo_destino.body.insert(8,to_insert_node) #En el predict es distinto, se exactamente donde insertar
    else:
        print(f"Error: el tipo de funcion no es valido ({tipo_funcion})")
                    
        
def create_new_file(file):
    '''Esta funcion, crea el fichero que vamos a devolver con la herramient sk_tool
    Es un fichero con el mismo nombre pero precedido de "sk_"'''
    if file.find(".py") == -1:
        sk_file = "output/sk_"+file+".py"
    else:
        sk_file = "output/sk_"+file
    open(sk_file, "w").close() #python deberia cerrarlo automaticamente, pero queda mas claro asi
    return sk_file

def get_skynnet_atributes(cadena):
    '''Esta funcion crea un diccionario que analiza la etiqueta de Skynnet
    y almacena el tipo de red y las opciones que permite, para luego calcular la
    reduccion de capas en cada caso'''
    skynnet_config = {}
    t_skynnet =r'(#)?(SKYNNET:BEGIN_)?(REGRESSION|MULTICLASS|BINARYCLASS)?(_ACC|_LOSS|_ACC_LOSS)?$'
    coincidencias = re.findall(t_skynnet,cadena)
    if len(coincidencias)==0:
        return skynnet_config
    coincidencias = list(coincidencias[0])
    nparams_correcto = (len(coincidencias)== 4 or len(coincidencias)==3)
    sk_begin_correcto = (coincidencias[0] == '#' and coincidencias[1] == 'SKYNNET:BEGIN_')
    if nparams_correcto and sk_begin_correcto:
        skynnet_config['Type'] = coincidencias[2]
        skynnet_config['Options'] = coincidencias[3] if len(coincidencias)==4 else ''
    return skynnet_config

def prepare_sk_file(file):
    ''' Esta funcion coge el fichero fuente, y apunta los indices de inicio y fin de 
    todas las parejas de etiquetas skynnet, ademas de guardar todo el codigo fente
    de skynnet que habrá que duplicar 
    Se almacena:
    -El codigo de inicio (antes de la etiqueta)
    -El codigo de skynnet (entre etiquetas)
    -El codigo de finalizacion (despues de la etiqueta end)

    Se permite que haya varias parejas de etiquetas skynnet, por eso los codigos de skynnet y finalizacion
    son listas
    '''

    sk_trees = []
    sk_tree = ""
    save_sk_code = False
    init_code = ""
    post_sk_codes = []
    post_sk_code = ""
    save_init_code = True
    save_post_sk_code = False
    skynnet_configs = []
    file = "input/"+file

    with open(file,'r') as fi:
        line = fi.readline()
        i = 1
        while line:
            #print(line)
            if save_init_code:
                init_code+=line
            if save_sk_code:
                sk_tree+=line
            if save_post_sk_code:
                post_sk_code+=line
            if "SKYNNET:BEGIN" in line:
                skynnet_configs.append(get_skynnet_atributes(line))
                save_init_code = False
                save_post_sk_code = False
                save_sk_code = True
                start_skynnet = i
                if len(sk_trees) != 0:#Antes del primer begin, no hay post_sk_code 
                    post_sk_codes.append(post_sk_code)
                post_sk_code = ""
            if "SKYNNET:END" in line:
                save_init_code = False #Por si alguien lo pone desordenado o le falta etiqueta begin
                save_sk_code = False 
                end_skynnet = i
                sk_trees.append(sk_tree)
                sk_tree = ""
                save_post_sk_code = True
                post_sk_code = '\n'+line #La linea del end la tienen sk_code, y post_sk, porque sk_code la ignora y quiero conservarla      
            i+=1
            line = fi.readline()
        post_sk_codes.append(post_sk_code)

    return(sk_trees,init_code,post_sk_codes,skynnet_configs)

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
    #devuelvo el numero deseado si es posible, o uno menor, para que quede una maquina libre
    if subredes in results:
        n_subredes = subredes
    else:
        n_subredes = max(results.keys())
    return n_subredes,results[n_subredes]

def process_skynnet_code(code, skynnet_config, fout, num_subredes, block_number):
    '''En esta funcion se aplican las transformaciones sobre el arbol de sintaxis de python
    que se han definido en las correspondientes clases
    1- Se reducen las variables de skynnet. Antes se calculan subredes y divison de datos
    2- Se almacenan las asignaciones en las que se crean los modelos
    3- Se almacenan las invocaciones que se aplican a los modelos
    4- Se indica el diccionario con la info recabada en los pasos 2 y 3
    5- Se crean las etiquetas de cloudbook necesarias para escribir la funcion skynnet
    Extra: Se añade el model number, porque puede haber varios modelos en un script'''

    #Primero calculan categorias y se reducen las variables por n y se escriben
    ast_code = ast.parse(code)
    #Lo primerisimo es saber las categorias 
    reduccion = num_subredes
    visit_categorias = VisitLastNeuron()
    visit_categorias.visit(ast_code)
    last_neuron = visit_categorias.last_neuron
    #print(last_neuron)
    if skynnet_config['Type'] == 'MULTICLASS':
        #categorias: las categorias totales, las de la ultima capa _NEURON
        #estas categorias se distribuyen en las subredes dividiendose en grupos tomados de dos en dos
        #grupos: grupos en los que divido las categorias
        #tomados: 2, porque por defecto se toman de 2 en 2
        #categorias_subred: la media de categorias que le tocan a una subred (las categorias se meten en grupos, y a cada subred le tocan dos grupos)
        #visit_categorias = VisitLastNeuron()
        #visit_categorias.visit(ast_code)
        categorias = visit_categorias.n_categorias
        print(f"Son {categorias} categorias originales")
        #Se calcula cuantas subredes quedaran
        num_subredes,combinatorio = get_categorias(num_subredes, categorias)
        grupos = combinatorio[0]
        tomados = combinatorio[1]
        print(f"Para formar subredes=C{combinatorio} es decir {grupos} categorias tomados de {tomados} en {tomados}")
        print(f"numero de grupos es {grupos}")
        print(f"El numero de subredes va a ser {num_subredes}")
        categorias_subred = ((categorias/grupos)*tomados)
        print(f"Categorias por subred: {categorias_subred}")
        reduccion = (categorias/categorias_subred)
        print(f"la reduccion sera por {reduccion} esto es por defecto")
    elif skynnet_config['Type'] == 'BINARYCLASS':
        reduccion = 2
        last_neuron = (last_neuron[0],2)
        num_subredes = 2
        grupos = 2
        categorias = 2
    elif skynnet_config['Type'] == 'REGRESSION':
        categorias = last_neuron[1]
        grupos = num_subredes
        reduccion = num_subredes
    #Se reducen los datos
    node_data_vars_reduced = TransformAssignSkVars(reduccion).visit(ast_code)
    #==========================================
    #Luego se escribe el resto retocando las invocaciones
    #visito asignaciones
    model_functions = VisitModelFunctions()
    model_functions.visit(node_data_vars_reduced)
    sk_dict = model_functions.dict_modelos

    #print(sk_dict)
    #sk_dict contiene todos los nodos de las funciones que busco
    #============================================
    #Una vez tengo el skdict, puedo guardar los datos de entrenamiento, test y validacion
    hay_prediccion = False #Variable para hacer la funcion predict o no, si no la hay poner un pass en la funcion
    for model_name in sk_dict.keys():
        hay_prediccion = 'predict' in sk_dict[model_name]
        GetModelDataVars(model_name,sk_dict).visit(node_data_vars_reduced)
        getInputModel(model_name,sk_dict).visit(node_data_vars_reduced) #aqui cojo el nodo antes de crear la red, para meter la division de datos y la ultima neurona antes
        #if not(sk_dict[model_name]["input"]):
        if "input" not in sk_dict[model_name]:
            sk_dict[model_name]["input"] = sk_dict[model_name]["creation"] #Si no usas el modelo funcional TODO: Eliminar en el futuro
    #print(sk_dict)
    #=========================================
    #Quito la prediccion y la guardo para meterla en una funcion nueva
    predictions =  RemovePredictionNode(sk_dict)
    node_data_vars_reduced = predictions.visit(node_data_vars_reduced)
    #=========================================
    #Escribo las variables globales, una por modelo que haya en el bloque
    number_of_models = len(sk_dict)
    global_predictions_declarations = []
    global_cloudbook_label = Expr(value=Comment(value='#__CLOUDBOOK:GLOBAL__'))
    global_predictions_declarations.append(global_cloudbook_label)
    for model_number in range(number_of_models):
        global_pred_assignment = Assign(targets=[ast.Name(id=f"predictions_{block_number}_{model_number}", ctx=ast.Store())],  # El objetivo de la asignación es el nombre "predictions"
                                value=ast.Dict(keys=[], values=[]),  # El valor asignado es un diccionario vacío {}
                                )
        global_pred_expr = Expr(value=global_pred_assignment)
        global_predictions_declarations.append(global_pred_expr)
    fixed_predictions = map(lambda x: unparse(fix_missing_locations(x)), global_predictions_declarations)
    fout.writelines(fixed_predictions)
    #TODO Ver si merece la pena llamarlo como el nombre del modelo en lugar de un indice, por si hay varios ficheros con bloques SKYNNET
    #=========================================
    #Escribo las variables nonshared que son las de los modelos = None
    nonshared_models_declarations = []
    nonshared_cloudbook_label = Expr(value = Comment(value='#__CLOUDBOOK:NONSHARED__'))
    nonshared_models_declarations.append(nonshared_cloudbook_label)
    for model_name in sk_dict.keys():
        nombre = ast.Name(id=model_name, ctx=ast.Store())
        #valor = ast.NameConstant(value=None) #Esto era cuando no teniamos array de modelos
        valor = ast.List(elts=[], ctx=ast.Load())
        model_name_declaration = ast.Assign(targets=[nombre], value=valor)
        model_name_expression = ast.Expr(value = model_name_declaration) #Como expresion para que separe las asignaciones en lineas distintas
        nonshared_models_declarations.append(model_name_expression)
    fixed_nonshared = map(lambda x: unparse(fix_missing_locations(x)), nonshared_models_declarations)
    fout.writelines(fixed_nonshared)
    #====================================== Añado la precision compuesta como una variable nonshared
    precision_compuesta= parse("precision_compuesta=[]")
    fout.write(unparse(fix_missing_locations(Expr(value = precision_compuesta))))
    #=========================================
    #Escribo la funcion skynnet block, que tiene todos los modelos del bloque
    parallel_cloudbook_label = Expr(value=Comment(value='#__CLOUDBOOK:PARALLEL__'))
    fout.write(unparse(fix_missing_locations(parallel_cloudbook_label)))
    #Aqui cambiamos los model por model[i]
    for model_name in sk_dict.keys():
        ModelArrayTransform(model_name).visit(node_data_vars_reduced)
    func_node = FunctionDef(
        name="skynnet_block_" + str(block_number),
        args=arguments(args=[ast.arg(arg='sk_i', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
        body=[node_data_vars_reduced],
        decorator_list=[]
    )
    for model_name in sk_dict.keys():
        global_node = Global(names=[model_name])
        func_node.body.insert(0,global_node)
        #Ademas de llamar a global hay que hacer model.append(None) para permitir los arrays
        append_node = ast.parse(model_name+".append(None)")
        func_node.body.insert(1,append_node)
    fout.write("\n")#Para evitar que el func_node se escriba en la misma linea que el comentario
    #Aqui cambiamos model por model[i]
    '''for model_name in sk_dict.keys():
        ModelArrayTransform(model_name).visit(func_node)'''
    #Meto antes del nodo de creacion del modelo, el codigo para calcular la division de los datos
    composed_measure = "" #Esta variable solo interesa en el predict, es para ver si calculo accuracy o loss o ambos
    if skynnet_config['Type'] == 'MULTICLASS':
        inserta_filtro_datos(func_node,"general",sk_dict,categorias,grupos,last_neuron,'MULTICLASS', composed_measure)
    elif skynnet_config['Type'] == 'BINARYCLASS':
        inserta_filtro_datos(func_node,"general",sk_dict,categorias,grupos,last_neuron,'BINARYCLASS', composed_measure)
    elif skynnet_config['Type'] == 'REGRESSION':
        inserta_filtro_datos(func_node,"general",sk_dict,categorias,grupos,last_neuron,'REGRESSION', composed_measure)
    fout.write(unparse(fix_missing_locations(func_node)))
    #=========================================
    if hay_prediccion: #si no la hay escribo la funcion con un pass
        #Ahora hay que escribir la funcion de la prediccion, parallel y todo eso
        fout.write(unparse(fix_missing_locations(parallel_cloudbook_label)))
        #cuerpo de la funcion: global predictions
        prediction_vars = []
        for model_number in range(number_of_models):
            prediction_vars.append(f"predictions_{block_number}_{model_number}")
        global_predictions_vars = []
        for prediction in prediction_vars:
            global_predictions_vars.append(Global(names=[prediction]))
        model_vars = []
        for model_name in sk_dict.keys():
            model_vars.append(Global(names=[model_name]))
        #fix_missing_locations(global_prediction_vars)
        #beginremove endremove
        #beginremove_cloudbook_label=Expr(value=Comment(value='#__CLOUDBOOK:BEGINREMOVE__'))
        beginremove_cloudbook_label=Comment(value='#__CLOUDBOOK:BEGINREMOVE__')
        cloudbook_var_prepare = ast.parse("__CLOUDBOOK__ = {}\n__CLOUDBOOK__['agent'] = {}")
        cloudbook_var = Subscript(
            value=Subscript(
                value=Name(id="__CLOUDBOOK__", ctx=Load()),
                slice=Index(value=Str(s="agent")),
                ctx=Load()
            ),
            slice=Index(value=Str(s="id")),
            ctx=Store()
        )
        value = Str(s="agente_skynnet")
        cloudbook_var_assig = Assign(targets=[cloudbook_var], value=value)
        #endremove_cloudbook_label=Expr(value=Comment(value='#__CLOUDBOOK:ENDREMOVE__'))
        endremove_cloudbook_label=Comment(value='#__CLOUDBOOK:ENDREMOVE__')
        label_var = Name(id="label",ctx=Load())
        #Para permitir varios modelos por agente tengo que añadir str(i) al label
        #assignation_cb_dict = Assign(targets=[label_var], value=cloudbook_var)
        value=ast.BinOp(
            left=cloudbook_var,     # Variable adios
            op=ast.Add(),                                  # Operador de suma
            right=ast.Call(
                func=ast.Name(id='str', ctx=ast.Load()),    # Función str
                args=[ast.Name(id='sk_i', ctx=ast.Load())],    # Argumento i
                keywords=[]
            )
        )
        assignation_cb_dict = Assign(targets=[label_var], value=value)
        predictions_assignements = []
        for i,model_name in enumerate(sk_dict.keys()):
            nombre = prediction_vars[i]
            #valor = sk_dict[model_name]['predict']
            #valor = ModelArrayTransform(model_name).visit(sk_dict[model_name]['predict']) #Ahora cambia por array de modelos
            valor = Name(id="resul", ctx=ast.Load())#ModelArrayTransform(model_name).visit(sk_dict[model_name]['predict']) #Ahora cambia por array de modelos
            prediction_var_target = Subscript(
                value=Name(id=nombre,ctx=Load()),
                slice=Index(value=label_var)
                )
            prediction_assignment = Assign(targets=[prediction_var_target], value=valor)
            fix_missing_locations(prediction_assignment)
            #print(unparse(prediction_assignment))
            predictions_assignements.append(prediction_assignment)
        #crear la funcion y meterle lo anterior en el body
        pred_func_node = FunctionDef(
            name="skynnet_prediction_block_" + str(block_number),
            args=arguments(args=[ast.arg(arg='sk_i', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
            body=[global_predictions_vars,model_vars,beginremove_cloudbook_label,cloudbook_var_prepare,cloudbook_var_assig, endremove_cloudbook_label,assignation_cb_dict, predictions_assignements],
            decorator_list=[]
        )
        fout.write("\n")
        #insertar el data_test
        #print(skynnet_config['Options'])
        measure_options = skynnet_config['Options']        
        if "ACC" in measure_options and "LOSS" in measure_options:
            composed_measure = "acc,loss"
        if "ACC" in measure_options and "LOSS" not in measure_options:
            composed_measure = "acc"
        if "ACC" not in measure_options and "LOSS" in measure_options:
            composed_measure = "loss"
        if "ACC" not in measure_options and "LOSS" not in measure_options:
            composed_measure = ""
        #print(composed_measure)
        for model_name in sk_dict.keys():
            to_insert_nodes = sk_dict[model_name]['data_test']
            pred_func_node.body.insert(2,to_insert_nodes)
        if skynnet_config['Type'] == 'MULTICLASS':
            inserta_filtro_datos(pred_func_node,"predict",sk_dict,categorias,grupos,last_neuron,'MULTICLASS', composed_measure)
        elif skynnet_config['Type'] == 'BINARYCLASS':
            inserta_filtro_datos(pred_func_node,"predict",sk_dict,categorias,grupos,last_neuron,'BINARYCLASS', composed_measure)
        elif skynnet_config['Type'] == 'REGRESSION':
            inserta_filtro_datos(pred_func_node,"predict",sk_dict,categorias,grupos,last_neuron,'REGRESSION', composed_measure)
        fout.write(unparse(fix_missing_locations(pred_func_node)))


    fout.write('\n\n')
    return num_subredes

def write_sk_block_invocation_code(block_number,fo, skynnet_config):
    '''Escribe la funcion con el bucle, va en la du_0
    #DU_0
    def skynnet_global_n():
      for i in subredes:
        assign_unique_id(i) #y filtrar datos 
        #en la herramienta no hace nada
      for i in subredes:
        skynnet()
        #cloudbook:sync
    '''
    #=====================================
    measure_options = skynnet_config['Options']        
    if "ACC" in measure_options and "LOSS" in measure_options:
        composed_measure = "acc,loss"
    if "ACC" in measure_options and "LOSS" not in measure_options:
        composed_measure = "acc"
    if "ACC" not in measure_options and "LOSS" in measure_options:
        composed_measure = "loss"
    if "ACC" not in measure_options and "LOSS" not in measure_options:
        composed_measure = ""
    #=====================================
    nodos_ast = []
    #creo nodo de comentario, y de funcion, y con el cuerpo, como es por defecto, lo puedo hacer con texto y parsearlo. y hacerle un fix missing locations o algo asi
    comment_du0 = Comment(value = "#__CLOUDBOOK:DU0__")
    fo.write(unparse(fix_missing_locations(comment_du0)))
    fo.write("\n")
    #hago el parametro subredes del bucle for
    range_call = Call(
        func=Name(id='range', ctx=Load()),
        args=[Num(num_subredes)],
        keywords=[],
    )
    #llamada a funcion skynnet_block_n
    skynnet_call = Expr(
        value=Call(
            func=Name(id='skynnet_block_'+str(block_number), ctx=Load()),
            args=[ast.arg(arg='i', annotation=None)],
            keywords=[],
        )
    )
    #bucle for
    for_loop = For(
        target=Name(id='i', ctx=Store()),
        iter=range_call,
        body=[skynnet_call],
        orelse=[],
    )
    #comentario sync
    comment_sync = Comment(value = "#__CLOUDBOOK:SYNC__")
    #funcion
    func_def = FunctionDef(
        name=f'skynnet_global_{block_number}',
        args=arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
        body=[for_loop,comment_sync],
        decorator_list=[],
        returns=None,
    )
    fo.write(unparse(fix_missing_locations(func_def)))
    fo.write("\n")
    #=====================================================
    fo.write(unparse(fix_missing_locations(comment_du0)))
    fo.write("\n")
    skynnet_pred_call = Expr(
        value=Call(
            func=Name(id='skynnet_prediction_block_'+str(block_number), ctx=Load()),
            args=[ast.arg(arg='i', annotation=None)],
            keywords=[],
        )
    )
    #bucle for
    for_pred_loop = For(
        target=Name(id='i', ctx=Store()),
        iter=range_call,
        body=[skynnet_pred_call],
        orelse=[],
    )
    #Prediccion compuesta
    codigo_medidas_extra='''
#No measures in pragma, nothing to add
'''
    codigo_prediccion_compuesta =f'''global precision_compuesta
valores = np.array(list(predictions_{block_number}_0.values()))
resultado = np.prod(valores,axis=0)
correctas = 0
total = 0
for i in range(len(y_test)):
    if y_test[i] == np.argmax(resultado[i]):
        correctas+=1
    total+=1
precision_compuesta.append(correctas/total)
print("============================================")
print('Skynnet Info: La accuracy de la prediccion compuesta es: ', precision_compuesta)
print("============================================")
'''
    prediccion_aux = f'''
global precision_compuesta
valores = np.array(list(predictions_{block_number}_0.values()))
resultado = np.prod(valores,axis=0)
'''
    codigo_loss_compuesta = f'''
scce = tf.keras.losses.SparseCategoricalCrossentropy()
scce_orig=scce(y_test, resultado).numpy()
print('============================================')
print('Skynnet Info: La loss compuesta es: ', scce_orig)
print('============================================')
'''
    codigo_loss_compuesta_regresion = f'''
bce = tf.keras.losses.BinaryCrossentropy()
bce_orig=bce(y_test, resultado)
print('============================================')
print('Skynnet Info: La loss compuesta es: ', bce_orig)
print('============================================')
'''
    if composed_measure == "acc,loss": #si es loss y regresion, la loss esta mal, pero peta en la prediccion, no afecta
        codigo_medidas_extra = codigo_prediccion_compuesta+codigo_loss_compuesta
    elif composed_measure == "acc":
        codigo_medidas_extra = codigo_prediccion_compuesta
    elif composed_measure == "loss" and skynnet_config['Type'] == 'MULTICLASS' :
        codigo_medidas_extra = prediccion_aux+codigo_loss_compuesta
    elif composed_measure == "loss" and skynnet_config['Type'] == 'BINARYCLASS' :
        codigo_medidas_extra = prediccion_aux+codigo_loss_compuesta
    elif composed_measure == "loss" and skynnet_config['Type'] == 'REGRESSION' :
        codigo_medidas_extra = prediccion_aux+codigo_loss_compuesta_regresion
    else:
        codigo_medidas_extra = codigo_medidas_extra
    codigo_medidas_extra = fix_missing_locations(parse(codigo_medidas_extra))
    #funcion
    func_pred_def = FunctionDef(
        name=f'skynnet_prediction_global_{block_number}',
        args=arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
        body=[for_pred_loop,comment_sync,codigo_medidas_extra],
        decorator_list=[],
        returns=None,
    )
    fo.write(unparse(fix_missing_locations(func_pred_def)))
    fo.write("\n")

def write_sk_global_code(number_of_sk_functions,fo):
    '''escribo un if name al final del fichero que invoca a las funciones de cada modelo necesarias, solo invocaciones, las definciones en 
    la funcion sk_model_code. Esta invocacion debería ir en la du_0
    if name = main:
        skynnet_global_0()
        skynnet_global_n()
        predicted_1 = bla bla'''
    
    # Creamos una lista de nombres de función con el patrón "skynnet_global_{i}"
    func_names = [f"skynnet_global_{i}" for i in range(number_of_sk_functions)]
    func_pred_names = [f"skynnet_prediction_global_{i}" for i in range(number_of_sk_functions)]
    # Creamos una lista de llamadas a función con los nombres generados y los índices del 0 a n
    func_calls = [Call(func=Name(id=name, ctx=ast.Load()), args=[], keywords=[]) for name in func_names]
    func_pred_calls = [Call(func=Name(id=name, ctx=ast.Load()), args=[], keywords=[]) for name in func_pred_names]

    
    #Hacemos una funcion cloudbook main, primero la etiqueta y luego la funcion
    comment_main = Comment(value = "#__CLOUDBOOK:MAIN__")
    fo.write(unparse(fix_missing_locations(comment_main)))
    fo.write("\n")
    #funcion
    main_def = FunctionDef(
        name="main",
        args=arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
        body=[],#Expr(value=call) for call in func_calls],
        decorator_list=[],
        returns=None,
    )
    for call in range(number_of_sk_functions):
        main_def.body.append(Expr(value=func_calls[call]))
        main_def.body.append(Expr(value=func_pred_calls[call]))
    fix_missing_locations(main_def)
    fo.write(unparse(main_def))
    fo.write('\n\n')
    # Creamos un bloque if __name__ == "__main__" que contiene todas las llamadas a función generadas
    # Creamos una llamada a la función main()
    main_call = Call(func=Name(id="main", ctx=Load()), args=[], keywords=[])
    main_block = If(
        test=Compare(left=Name("__name__", Load()), ops=[Eq()], comparators=[Str("__main__")]),
        body=[Expr(value=main_call)],
        orelse=[]
    )
    fix_missing_locations(main_block)
    fo.write(unparse(main_block))


def main():
    '''Procesa el fichero de entrada y genera el de salida'''
    global num_subredes
    print(f"num subredes original {num_subredes}")
    sk_file = create_new_file(file)
    sk_trees,init_code,post_sk_codes,skynnet_configs = prepare_sk_file(file)
    print(skynnet_configs)
    num_sk_blocks = len(skynnet_configs) #Cuantos bloques skynnet hay
    with open(sk_file,'w') as fo:
        #escribo el principio
        fo.write(init_code)
        fo.write(funciones_combinatorias)
        for block_number in range(num_sk_blocks):
            code = sk_trees[block_number]
            num_subredes = process_skynnet_code(code, skynnet_configs[block_number], fo, n, block_number)
            #escribo el final
            fo.write(post_sk_codes[block_number])
            fo.write("\n\n")
            #Escribo la llamada a los modelos del bloque
            write_sk_block_invocation_code(block_number,fo,skynnet_configs[block_number])
        fo.write("\n\n")
        #Escribo la llamada a todos los bloques
        write_sk_global_code(num_sk_blocks,fo)
        fo.write("\n\n")

if __name__=="__main__":
    if len(sys.argv)!=3:
        print("Usage: py sk_tool.py file num_machines")
        sys.exit()
    else:
        n = int(sys.argv[2])
        file = sys.argv[1]
        if file.find(".py") == -1:
            file = file+".py"
        print("Fichero: {} en {} subredes".format(file,n))
        num_subredes = n
        main()


