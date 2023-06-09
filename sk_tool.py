from ast import *
import sys
import re
from ast_comments import *
from math import comb,ceil


#Variables globales
sk_vars_list = ["_EMBEDDING_","_NEURON_","_CELL_","_EPOCHS","_BATCH"]
sk_functions_list = ['summary','compile','fit','predict']
sk_creation_model_list = ['Sequential','Model']

num_subredes = 0

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
					new_value = ast.Constant(value=node.value.value//self.reduccion)
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
		nodo_valido_dcha = len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
		nodo_valido_izqda = isinstance(node.value, ast.Call)
		if nodo_valido_dcha and nodo_valido_izqda:
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
				slice=ast.Index(value=ast.Name(id='i', ctx=ast.Load())),
				ctx=node.ctx
			)		
		return node#self.generic_visit(node)


class VisitLastNeuron(ast.NodeVisitor):
	'''Esta clase es para obtener el numero de categorias en las que se clasifica
	TODO: Mezclar esta clase con la que reduce los datos'''
	max_valor = 0
	n_categorias = 0

	def visit_Assign(self, node):
		'''Esta funcion es la que divide por un numero entero las variables de skynnet
		'''
		variable_valida =  (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.value)
		if variable_valida:
			existe_sk_var,variable_skynnet,valor = get_var_from_list(node.targets[0].id, sk_vars_list)			
			if existe_sk_var == True and variable_skynnet == '_NEURON_':
				#print(f"node.value es {node.value.value}")
				if valor>self.max_valor:
					self.max_valor = valor
					self.n_categorias = node.value.value

			

def create_new_file(file):
	'''Esta funcion, crea el fichero que vamos a devolver con la herramient sk_tool
	Es un fichero con el mismo nombre pero precedido de "sk_"'''
	if file.find(".py") == -1:
		sk_file = "sk_"+file+".py"
	else:
		sk_file = "sk_"+file
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
	if skynnet_config['Type'] == 'MULTICLASS':
		categorias = VisitLastNeuron()
		categorias.visit(ast_code)
		n_categorias = categorias.n_categorias
		print(f"Son {n_categorias} categorias originales")
		#Se calcula cuantas subredes quedaran
		num_subredes,combinatorio = get_categorias(num_subredes, n_categorias)
		print(f"Para formar subredes=C{combinatorio} es decir {combinatorio[0]} categorias tomados de {combinatorio[1]} en {combinatorio[1]}")
		num_grupos = combinatorio[0]
		print(f"numero de grupos es {num_grupos}")
	print(f"El numero de subredes va a ser {num_subredes}")
	reduccion = ceil(n_categorias/num_subredes)
	print(f"la reduccion sera por {reduccion}")
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
	#=========================================
	#Escribo la funcion skynnet block, que tiene todos los modelos del bloque
	parallel_cloudbook_label = Expr(value=Comment(value='#__CLOUDBOOK:PARALLEL__'))
	fout.write(unparse(fix_missing_locations(parallel_cloudbook_label)))
	#Aqui cambiamos los model por model[i]
	for model_name in sk_dict.keys():
		ModelArrayTransform(model_name).visit(node_data_vars_reduced)
	func_node = FunctionDef(
		name="skynnet_block_" + str(block_number),
		args=arguments(args=[ast.arg(arg='i', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
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
	fout.write(unparse(fix_missing_locations(func_node)))
	#=========================================
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
			args=[ast.Name(id='i', ctx=ast.Load())],    # Argumento i
			keywords=[]
		)
	)
	assignation_cb_dict = Assign(targets=[label_var], value=value)
	predictions_assignements = []
	for i,model_name in enumerate(sk_dict.keys()):
		nombre = prediction_vars[i]
		#valor = sk_dict[model_name]['predict']
		valor = ModelArrayTransform(model_name).visit(sk_dict[model_name]['predict']) #Ahora cambia por array de modelos
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
		args=arguments(args=[ast.arg(arg='i', annotation=None)], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
		body=[global_predictions_vars,model_vars, beginremove_cloudbook_label,cloudbook_var_prepare,cloudbook_var_assig, endremove_cloudbook_label,assignation_cb_dict, predictions_assignements],
		decorator_list=[]
	)
	fout.write("\n")
	fout.write(unparse(fix_missing_locations(pred_func_node)))


	fout.write('\n\n')
	return num_subredes

def write_sk_block_invocation_code(block_number,fo):
	'''Escribe la funcion con el bucle, va en la du_0
	#DU_0
	def skynnet_global_n():
	  for i in subredes:
		assign_unique_id(i) #y filtrar datos 
		#en la herramienta no hace nada
	  for i in subredes:
		skynnet()
		#cloudbook:sync
	TODO: El predicted, sera otra funcion de du0 que hace un bucle llamando 
	a las predicted y luego hace global predicted y return de ese predicted'''
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
	#funcion
	func_pred_def = FunctionDef(
		name=f'skynnet_prediction_global_{block_number}',
		args=arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
		body=[for_pred_loop,comment_sync],
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
		for block_number in range(num_sk_blocks):
			code = sk_trees[block_number]
			num_subredes = process_skynnet_code(code, skynnet_configs[block_number], fo, n, block_number)
			#escribo el final
			fo.write(post_sk_codes[block_number])
			fo.write("\n\n")
			#Escribo la llamada a los modelos del bloque
			write_sk_block_invocation_code(block_number,fo)
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


