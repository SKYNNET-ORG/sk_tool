import ast
import sys
import re
#from ast_comments import *
import ast_comments

#Variables globales
sk_vars_list = ["_EMBEDDING_","_NEURON_","_CELL_","_EPOCH","_BATCH"]
sk_functions_list = ['summary','compile','fit','predict']
sk_creation_model_list = ['Sequential','Model']


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

class RemoveAssignSkVars(ast.NodeTransformer):
	'''Esta clase auxiliar se usa para eliminar las invocaciones de skynnet una vez las has escrito reducidas,
	no te hace falta escribir _NEURON_1 = 40 las 4 veces que hagas una subred 
	NO NECESARIA EN LA V1.0'''

	def __init__(self,reduccion=1, num_redes=1):
		self.reduccion = reduccion
		self.num_redes = num_redes

	def visit_Assign(self, node):
		try:
			variable_valida =  (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.value)
			variable_skynnet = get_var_from_list(node.targets[0].id, sk_vars_list)[0]==True
			if variable_valida and variable_skynnet:
				pass #En este caso no devolvemos el nodo, porque es de skynnet y ya se ha escrito una vez
			else:
				ast.fix_missing_locations(node)
				return node
		except Exception as error:
			#print("Nodo erroneo",error, node.targets[0].id)
			ast.fix_missing_locations(node)
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


class TransformModelName(ast.NodeTransformer):
	'''Esta clase es la que implementa los metodos de visita necesarios para cambiar automaticamente los nombres de modelo
	a cada submodelo, se visitan las asignaciones (creacion), las invocaciones (model.funcion), y los returns por si metes
	una funcionalidad en un metodo y devuelves el modelo'''
	def __init__(self,dict_modelos, number):
		self.dict_modelos = dict_modelos
		self.number = number

	@staticmethod
	def get_sk_model_creation(llamada):#Esta funcion se eliminara, porque se hace mejor con el dict.keys()
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
				node.targets[0].id = re.sub(r"_[0-9]+", "", node.targets[0].id)
				node.targets[0].id += "_"+str(self.number)
				return node
		self.generic_visit(node)
		return node

	def visit_Call(self,node):
		#TODO: con expresiones regulares, o con el ast cambiar las invocaciones a los datos, por las que definimos en skynnet DEBATIR ESTO
		if isinstance(node.func, ast.Attribute) and node.func.attr in sk_functions_list:
			aux_func_id = re.sub(r"_[0-9]+", "", node.func.value.id)
			if isinstance(node.func.value, ast.Name) and aux_func_id in self.dict_modelos.keys():
				#print(f"voy a cambiar: {node.func.value.id} de {ast.unparse(node)}")
				node.func.value.id = re.sub(r"_[0-9]+", "", node.func.value.id)
				node.func.value.id += '_'+str(self.number)
				return node
		self.generic_visit(node)
		return node

	def visit_Return(self,node):
		if isinstance(node.value, ast.Name):
			aux_func_id = re.sub(r"_[0-9]+", "", node.value.id)
			if aux_func_id in self.dict_modelos.keys():
				node.value.id = re.sub(r"_[0-9]+", "", node.value.id)
				node.value.id += '_'+str(self.number)
		return node

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


def process_skynnet_code_v00(code, skynnet_config, fout, num_subredes):
	'''En esta funcion se aplican las transformaciones sobre el arbol de sintaxis de python
	que se han definido en las correspondientes clases
	1- Se reducen las variables de skynnet
	2- Se escribe el codigo skynnet con los valores reducidos por el numero de subrredes. Se incluye el modelo numero 0
	3- Al codigo que se ha escrito, se le quitan las variables skynnet, para no repetirlas n veces
	4- Se almacenan las asignaciones en las que se crean ls modelos
	5- Se almacenan las invocaciones que se aplican a los modelos
	6- Se indica el diccionario con la info recabada en los pasos 4 y 5
	7- Se reescriben el resto de modelos con los nombres cambiados, si el nombre original es model, ahora sera model_n'''
	#Primero se reducen las variables por n y se escriben
	ast_code = ast.parse(code)
	node_data_vars_reduced = TransformAssignSkVars(num_subredes).visit(ast_code)
	fout.write(ast.unparse(node_data_vars_reduced))
	fout.write('\n\n')
	#una vez escrito, eliminar los nodos de las asignaciones que coincidan #WARNING Las asignaciones del user se repetiran, pero no las toco nunca
	node_no_data_vars = RemoveAssignSkVars().visit(node_data_vars_reduced)
	#==========================================
	#Luego se escribe el resto retocando las invocaciones
	#visito asignaciones
	assignations = VisitSkFunctionsAssign()
	assignations.visit(node_no_data_vars)
	#visito invocaciones de tipo atributo, o "de metodo"
	invocations = VisitSkFunctionsCalls(assignations.dict_modelos)
	invocations.visit(node_no_data_vars)
	#ahora puedo escribir como quiera el codigo
	sk_dict = invocations.dict_modelos
	print(sk_dict)
	print("Work in progress: Analizar la coherencia del diccionario, que esten todos los nodos al menos")
	#sk_dict contiene todos los nodos de las funciones que busco
	#==========================================
	#ahora escribo el codigo linea a linea, comparando cada nodo por si esta en el diccionario, si esta en el diccionario le cambio el nombre al modelo
	code_to_write = ast.unparse(node_no_data_vars).split('\n')
	for model_number in range(1,num_subredes): #desde 1 para hacer una menos, ya que la original se escribe por defecto
		TransformModelName(sk_dict,model_number).visit(node_no_data_vars)
		fout.write(ast.unparse(node_no_data_vars))
		fout.write('\n\n')


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
	func_node = ast.FunctionDef(
		name="skynnet_block_" + str(model_number),
		args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
		body=[node_data_vars_reduced],
		decorator_list=[]
	)
	nodos_skynnet = []
	nonshared_declaration = ast_comments.Comment(value = "\n#__CLOUDBOOK:NONSHARED__\n")
	nodos_skynnet.append(nonshared_declaration)
	for model_name in sk_dict.keys():
		nombre_variable = model_name
		nombre = ast.Name(id=nombre_variable, ctx=ast.Store())
		valor = ast.NameConstant(value=None)
		model_name_declaration = ast.Assign(targets=[nombre], value=valor)
		nodos_skynnet.append(model_name_declaration)
	parallel_declaration =  ast_comments.Comment(value = "\n#__CLOUDBOOK:PARALLEL__\n")
	nodos_skynnet.append(parallel_declaration)
	nodos_skynnet.append(func_node)
	for i in nodos_skynnet:
		ast.fix_missing_locations(i)
		if isinstance(i,ast_comments.Comment):
			fout.write(ast_comments.unparse(i))
		fout.write(ast.unparse(i))
	fout.write('\n\n')

def write_sk_model_invocation_code(model_number):
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

	pass

def write_sk_global_code(number_of_sk_functions):
	'''escribo un if name al final del fichero que invoca a las funciones de cada modelo necesarias, solo invocaciones, las definciones en 
	la funcion sk_model_code. Esta invocacion debería ir en la du_0
	if name = main:
		skynnet_global_0()
		skynnet_global_n()
		predicted_1 = bla bla'''
	# es hacer el if, y las invocaciones, casi se puede escribir como texto que se puede parsear y hacerle un fix missing locations o algo asi
	pass


def main():
	'''Procesa el fichero de entrada y genera el de salida'''
	sk_file = create_new_file(file)
	index_pragmas,sk_trees,init_code,post_sk_codes,skynnet_configs = prepare_sk_file(file)
	print(skynnet_configs)
	num_sk_blocks = len(skynnet_configs) #Cuantos bloques skynnet hay
	print("Work in progress: Con los parametros de la configuracion, hacer las operaciones necesarias para sacar el valor de reduccion de cada subred")
	with open(sk_file,'w') as fo:
		for tree_number,indexes in enumerate(index_pragmas):
			#escribo el principio
			fo.write(init_code)
			#escribo skynnet
			#fo.write(sk_trees[tree_number])
			code = sk_trees[tree_number]
			process_skynnet_code(code, skynnet_configs[tree_number], fo, n,tree_number)
			#escribo el final
			fo.write(post_sk_codes[tree_number])
			#Escribo en "main" la llamada al modelo
			write_sk_model_invocation_code(tree_number)
		write_sk_global_code(num_sk_blocks)

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
		main()


