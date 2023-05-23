import ast
import sys
import re
#from ast_comments import *
import ast_comments

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
	- La nombre de la variable sin en numero
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

	def __init__(self,reduccion=1, num_redes=1):
		self.reduccion = reduccion #Indica por cuanto hay que dividir el valor de la asignacion
		self.num_redes = num_redes #No se usa. DELETE

	def visit_Assign(self, node):
		'''Esta funcion es la que divide por un numero entero las variables de skynnet
		'''
		try:
			variable_valida =  (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.value)
			variable_skynnet = get_var_from_list(node.targets[0].id, sk_vars_list)[0]==True
			if variable_valida and variable_skynnet:
				new_value = ast.Constant(value=node.value.value//self.reduccion)
				new_node = ast.Assign(targets=node.targets, value=new_value, lineno = node.lineno)
				return new_node
			else:
				# Si no es una asignación de un solo objetivo con un valor, simplemente devuelve el nodo original sin modificarlo
				#print("Nodo erroneo", node.targets[0].id)
				return node
		except Exception as error:
			print("Nodo erroneo",error, node.targets[0].id)
			return node

class VisitSkFunctionsAssign(ast.NodeVisitor):
	'''Esta clase es la que permite obtener las asignaciones que se usan para crear modelos en skynnet,
	ya sea con el modelo normal o funcional. Ademas prepara el diccionario que tendra un resumen de los nodos
	para las funciones summary, creation, compile, fit, si añadimos alguna funcion nueva se añade al diccionario de forma automatica en otra clase'''
	def __init__(self):
		#self.lista_modelos = []
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
				#Diccionario TODO: Puedo quitar summary, compile y fit, ya que se añaden automaticamente en la clase VisitSkFunctionCalls
				self.dict_modelos[model_name] = {'creation': node,
												'summary': None,
												'compile': None,
												'fit': None}

class VisitSkFunctionsCalls(ast.NodeVisitor):
	'''Al igual que la clase que visita las asignaciones, esta clase sirve para visitar las invocaciones, de tipo model.algo
	esta funcion "algo", tiene que ser una de la lista de funciones skynnet que hay en la variable global. Una vez la encuentra la guarda 
	en el diccionario'''
	def __init__(self,dict_modelos):
		#self.lista_modelos = lista_modelos
		self.dict_modelos = dict_modelos

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
	def __init__(self,dict_modelos, prediction_node):
		self.prediction_node = None
		self.dict_modelos = dict_modelos

	def visit_Assign(self, node):
		nodo_valido_izqda = len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
		nodo_valido_dcha = isinstance(node.value, ast.Call)
		if nodo_valido_izqda and nodo_valido_dcha:
			derecha = node.value
			if isinstance(derecha.func, ast.Attribute) and derecha.func.attr == 'predict':
				if isinstance(derecha.func.value, ast.Name) and derecha.func.value.id in self.dict_modelos.keys():
					self.prediction_node = node
					return None
				else:
					return node
			else:
				return node
		else:
			return node
			#if isinstance(derecha.func, ast.Attribute) and derecha.func.attr == 'predict':
			#	if isinstance(derecha.func.value, ast.Name) and derecha.func.value.id in self.dict_modelos.keys():
			#		self.prediction_node = node
			#		return None
		#self.generic_visit(node)

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

	index_pragmas = []
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
				if len(index_pragmas) != 0:#Antes del primer begin, no hay post_sk_code 
					post_sk_codes.append(post_sk_code)
				post_sk_code = ""
			if "SKYNNET:END" in line:
				save_init_code = False #Por si alguien lo pone desordenado o le falta etiqueta begin
				save_sk_code = False 
				end_skynnet = i
				index_pragmas.append((start_skynnet,end_skynnet))
				sk_trees.append(sk_tree)
				sk_tree = ""
				save_post_sk_code = True
				post_sk_code = '\n'+line #La linea del end la tienen sk_code, y post_sk, porque sk_code la ignora y quiero conservarla		
			i+=1
			line = fi.readline()
		post_sk_codes.append(post_sk_code)

	return(index_pragmas, sk_trees,init_code,post_sk_codes,skynnet_configs)


def process_skynnet_code(code, skynnet_config, fout, num_subredes, model_number):
	'''En esta funcion se aplican las transformaciones sobre el arbol de sintaxis de python
	que se han definido en las correspondientes clases
	1- Se reducen las variables de skynnet
	2- Se almacenan las asignaciones en las que se crean los modelos
	3- Se almacenan las invocaciones que se aplican a los modelos
	4- Se indica el diccionario con la info recabada en los pasos 2 y 3
	5- Se crean las etiquetas de cloudbook necesarias para escribir la funcion skynnet
	Extra: Se añade el model number, porque puede haber varios modelos en un script'''
	#Primero se reducen las variables por n y se escriben
	ast_code = ast.parse(code)
	node_data_vars_reduced = TransformAssignSkVars(num_subredes).visit(ast_code)
	#==========================================
	#Luego se escribe el resto retocando las invocaciones
	#visito asignaciones
	assignations = VisitSkFunctionsAssign()
	assignations.visit(node_data_vars_reduced)
	#visito invocaciones de tipo atributo, o "de metodo"
	invocations = VisitSkFunctionsCalls(assignations.dict_modelos)
	invocations.visit(node_data_vars_reduced)
	#ahora puedo escribir como quiera el codigo
	sk_dict = invocations.dict_modelos
	print(sk_dict)
	print("Work in progress: Analizar la coherencia del diccionario, que esten todos los nodos al menos")
	#sk_dict contiene todos los nodos de las funciones que busco
	#=========================================
	#Quito la prediccion y la guardo para meterla en una funcion nueva
	prediction_node = None
	predictions =  RemovePredictionNode(sk_dict,prediction_node)
	node_data_vars_reduced = predictions.visit(node_data_vars_reduced)
	prediction_node = predictions.prediction_node	
	#=========================================	
	nodos_skynnet = []
	#variable global predictions
	global_pred_declaration = ast_comments.Comment(value = "\n#__CLOUDBOOK:GLOBAL__\n")
	nodos_skynnet.append(global_pred_declaration)
	pred_assignment = ast.Assign(
    targets=[ast.Name(id="predictions", ctx=ast.Store())],  # El objetivo de la asignación es el nombre "predictions"
    value=ast.Dict(keys=[], values=[]),  # El valor asignado es un diccionario vacío {}
	)
	nodos_skynnet.append(pred_assignment)
	#=========================================
	func_node = ast.FunctionDef(
		name="skynnet_block_" + str(model_number),
		args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
		body=[node_data_vars_reduced],
		decorator_list=[]
	)
	nonshared_declaration = ast_comments.Comment(value = "\n#__CLOUDBOOK:NONSHARED__")
	nodos_skynnet.append(nonshared_declaration)
	for model_name in sk_dict.keys():
		nombre_variable = model_name
		nombre = ast.Name(id=nombre_variable, ctx=ast.Store())
		valor = ast.NameConstant(value=None)
		model_name_declaration = ast.Assign(targets=[nombre], value=valor)
		model_name_expression = ast.Expr(value = model_name_declaration) #Como expresion para que separe las asignaciones en lineas distintas
		nodos_skynnet.append(model_name_expression)
	parallel_declaration =  ast_comments.Comment(value = "\n\n#__CLOUDBOOK:PARALLEL__\n")
	nodos_skynnet.append(parallel_declaration)
	nodos_skynnet.append(func_node)
	#=========================================
	
	for i in nodos_skynnet:
		ast.fix_missing_locations(i)
		if isinstance(i,ast_comments.Comment):
			fout.write(ast_comments.unparse(i))
		fout.write(ast.unparse(i))
	fout.write('\n')
	#=========================================
	#Aqui escribo la prediccion
	fout.write(ast_comments.unparse(parallel_declaration))
	# Crear el nodo Global
	global_node = ast.Global(names=['predictions'])
	pred_func_node = ast.FunctionDef(
		name="skynnet_prediction_block_" + str(model_number),
		args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
		body=[global_node],
		decorator_list=[]
	)
	ast.fix_missing_locations(pred_func_node)
	fout.write(ast.unparse(pred_func_node))
	beginremove_comment = ast_comments.Comment(value = "\n    #__CLOUDBOOK:BEGINREMOVE__\n")
	endremove_comment = ast_comments.Comment(value = "\n    #__CLOUDBOOK:ENDREMOVE__\n")
	# Crear el nodo de asignación
	target = ast.Subscript(
	    value=ast.Subscript(
	        value=ast.Name(id="__CLOUDBOOK__", ctx=ast.Load()),
	        slice=ast.Index(value=ast.Str(s="agent")),
	        ctx=ast.Load()
	    ),
	    slice=ast.Index(value=ast.Str(s="id")),
	    ctx=ast.Store()
	)
	# Crear el nodo de valor
	value = ast.Str(s="agente_skynnet")
	# Crear el nodo de asignación completa
	cloudbook_var_assig = ast.Assign(targets=[target], value=value)
	ast.fix_missing_locations(cloudbook_var_assig)
	fout.write(ast_comments.unparse(beginremove_comment))
	if (pred_func_node.col_offset == 0):
		fout.write("    ")
	fout.write(ast.unparse(cloudbook_var_assig))
	fout.write(ast_comments.unparse(endremove_comment))
	if (pred_func_node.col_offset == 0):
		fout.write("    ")
	# Crear el nodo Name con el nombre 'predictions'
	predictions_node = ast.Name(id='predictions', ctx=ast.Store())

	# Crear el nodo Name con el nombre 'label'
	label_node = ast.Name(id='label', ctx=ast.Load())

	# Crear el nodo Subscript con el objetivo predictions[label]
	subscript_node = ast.Subscript(value=predictions_node, slice=ast.Index(value=label_node), ctx=ast.Store())

	# Crear el nodo Name con el nombre 'cosa'
	cosa_node = ast.Name(id='cosa', ctx=ast.Load())

	# Crear el nodo Assign con el objetivo predictions[label] y el valor cosa
	prediction_assign_node = ast.Assign(targets=[subscript_node], value=prediction_node.value)
	ast.fix_missing_locations(prediction_assign_node)
	fout.write(ast.unparse(prediction_assign_node))


	fout.write('\n\n')

def write_sk_model_invocation_code(block_number,fo):
	'''Escribe la funcion con el bucle, va en la du_0
	#DU_0
	def skynnet_global_n():
	  for i in subredes:
		assign_unique_id(i) #y filtrar datos 
		#en la herramienta no hace nada
	  for i in subredes:
		skynnet()
		#cloudbook:sync
	TODO: El predicted'''
	nodos_ast = []
	#creo nodo de comentario, y de funcion, y con el cuerpo, como es por defecto, lo puedo hacer con texto y parsearlo. y hacerle un fix missing locations o algo asi
	comment_du0 = ast_comments.Comment(value = "\n#__CLOUDBOOK:DU0__\n")
	#hago el parametro subredes del bucle for
	range_call = ast.Call(
        func=ast.Name(id='range', ctx=ast.Load()),
        args=[ast.Num(num_subredes)],
        keywords=[],
    )
	#llamada a funcion skynnet_block_n
	skynnet_call = ast.Expr(
		value=ast.Call(
			func=ast.Name(id='skynnet_block_'+str(block_number), ctx=ast.Load()),
			args=[],
			keywords=[],
		)
	)
	#bucle for
	for_loop = ast.For(
		target=ast.Name(id='i', ctx=ast.Store()),
		iter=range_call,
		body=[skynnet_call],
		orelse=[],
	)
	#comentario sync
	comment_sync = ast_comments.Comment(value = "\n    #__CLOUDBOOK:SYNC__\n")
	#funcion
	func_def = ast.FunctionDef(
		name=f'skynnet_global_{block_number}',
		args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
		body=[for_loop],
		decorator_list=[],
		returns=None,
	)
	
	nodos_ast = [comment_du0,func_def,comment_sync]
	for i in nodos_ast:
		if isinstance(i,ast_comments.Comment):
			ast_comments.fix_missing_locations(i)
			fo.write(ast_comments.unparse(i))
		ast.fix_missing_locations(i)
		fo.write(ast.unparse(i))


def write_sk_global_code(number_of_sk_functions,fo):
	'''escribo un if name al final del fichero que invoca a las funciones de cada modelo necesarias, solo invocaciones, las definciones en 
	la funcion sk_model_code. Esta invocacion debería ir en la du_0
	if name = main:
		skynnet_global_0()
		skynnet_global_n()
		predicted_1 = bla bla'''
	
	# Creamos una lista de nombres de función con el patrón "skynnet_global_{i}"
	func_names = [f"skynnet_global_{i}" for i in range(number_of_sk_functions)]

	# Creamos una lista de llamadas a función con los nombres generados y los índices del 0 a n
	func_calls = [ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[], keywords=[]) for name in func_names]

	# Creamos un bloque if __name__ == "__main__" que contiene todas las llamadas a función generadas
	#main_block = ast.If(
	#    test=ast.Compare(left=ast.Name("__name__", ast.Load()), ops=[ast.Eq()], comparators=[ast.Str("__main__")]),
	#    body=[ast.Expr(value=call) for call in func_calls],
	#    orelse=[]
	#)
	#Hacemos una funcion cloudbook main, primero la etiqueta y luego la funcion
	comment_main = ast_comments.Comment(value = "\n#__CLOUDBOOK:MAIN__\n")
	fo.write(ast_comments.unparse(comment_main))
	#funcion
	main_def = ast.FunctionDef(
		name="main",
		args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
		body=[ast.Expr(value=call) for call in func_calls],
		decorator_list=[],
		returns=None,
	)

	ast.fix_missing_locations(main_def)
	fo.write(ast.unparse(main_def))
	fo.write('\n\n')
	# Creamos un bloque if __name__ == "__main__" que contiene todas las llamadas a función generadas
	# Creamos una llamada a la función main()
	main_call = ast.Call(func=ast.Name(id="main", ctx=ast.Load()), args=[], keywords=[])
	main_block = ast.If(
	    test=ast.Compare(left=ast.Name("__name__", ast.Load()), ops=[ast.Eq()], comparators=[ast.Str("__main__")]),
	    body=[ast.Expr(value=main_call)],
	    orelse=[]
	)
	ast.fix_missing_locations(main_block)
	fo.write(ast.unparse(main_block))


def main():
	'''Procesa el fichero de entrada y genera el de salida'''
	sk_file = create_new_file(file)
	index_pragmas,sk_trees,init_code,post_sk_codes,skynnet_configs = prepare_sk_file(file)
	print(skynnet_configs)
	num_sk_blocks = len(skynnet_configs) #Cuantos bloques skynnet hay
	print("Work in progress: Con los parametros de la configuracion, hacer las operaciones necesarias para sacar el valor de reduccion de cada subred")
	with open(sk_file,'w') as fo:
		#escribo el principio
		fo.write(init_code)
		for tree_number,indexes in enumerate(index_pragmas):
			#bucle for: funciona pero se entiende mal, es de la v00 que guardaba los indices de etiquetas, basicamente el for es sobre el numero de bloques de etiquetas skynnet
			code = sk_trees[tree_number]
			process_skynnet_code(code, skynnet_configs[tree_number], fo, n,tree_number)
			#escribo el final
			fo.write(post_sk_codes[tree_number])
			#Escribo en "main" la llamada al modelo
			fo.write("\n\n")
			write_sk_model_invocation_code(tree_number,fo)
		fo.write("\n\n")
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


