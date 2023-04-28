import ast
import sys
import ply.lex as lex
import re

sk_vars_list = ["_EMBEDDING_","_NEURON_","_CELL_","_EPOCH","_BATCH"]
sk_functions_list = ['summary','compile','fit','predict']
sk_creation_model_list = ['Sequential','Model']

def get_var_from_list(cadena, lista):
	'''Corregir
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

	def __init__(self,reduccion=1, num_redes=1):
		self.reduccion = reduccion
		self.num_redes = num_redes

	def visit_Assign(self, node):
		'''
		'''
		try:
			variable_valida =  (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.value)
			variable_skynnet = get_var_from_list(node.targets[0].id, sk_vars_list)[0]==True
			if variable_valida and variable_skynnet:
				new_value = ast.Constant(value=node.value.value//self.reduccion)
				#node.value = new_value
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

	def __init__(self,reduccion=1, num_redes=1):
		self.reduccion = reduccion
		self.num_redes = num_redes

	def visit_Assign(self, node):
		'''
		'''
		try:
			variable_valida =  (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.value)
			variable_skynnet = get_var_from_list(node.targets[0].id, sk_vars_list)[0]==True
			if variable_valida and variable_skynnet:
				pass
			else:
				# Si no es una asignación de un solo objetivo con un valor, simplemente devuelve el nodo original sin modificarlo
				#print("Nodo erroneo", node.targets[0].id)
				ast.fix_missing_locations(node)
				return node
		except Exception as error:
			#print("Nodo erroneo",error, node.targets[0].id)
			ast.fix_missing_locations(node)
			return node

class VisitSkFunctionsAssign(ast.NodeVisitor):

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
				#print(node.targets[0].id)
				#self.lista_modelos.append(model_name)
				self.dict_modelos[model_name] = {'creation': node,
												'summary': None,
												'compile': None,
												'fit': None}

class VisitSkFunctionsCalls(ast.NodeVisitor):

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
	if file.find(".py") == -1:
		sk_file = "sk_"+file+".py"
	else:
		sk_file = "sk_"+file
	open(sk_file, "w").close() #python deberia cerrarlo automaticamente, pero queda mas claro asi
	return sk_file

def get_skynnet_atributes(cadena):
	##SKYNNET:BEGIN_[REGRESSION|MULTICLASS|BINARYCLASS]_[ACC]_[LOSS]
	skynnet_config = {}
	t_skynnet =r'^(#)?(SKYNNET:BEGIN_)?(REGRESSION|MULTICLASS|BINARYCLASS)?(_ACC|_LOSS|_ACC_LOSS)?$'
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
	'''
	'''
	TODO: Expresiones regulares para los parametros de skynnet
		Seguramente implique devolver un campo mas para conocer la etiqueta
		Comprobar que las parejas de begin end sean pares, y siempre menor begin que end
		mismo numero de arboles que de pares begin end 
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


def process_skynnet_code(code, skynnet_config, fout, num_subredes):
	#Primero se reducen las variables por n y se escriben
	ast_code = ast.parse(code)
	node_data_vars_reduced = TransformAssignSkVars(num_subredes).visit(ast_code)
	#print(ast.unparse(node_data_vars_reduced))
	fout.write(ast.unparse(node_data_vars_reduced))
	fout.write('\n\n')
	#una vez escrito, eliminar los nodos de las asignaciones que coincidan #WARNING Las asignaciones del user se repetiran, pero no las toco nunca
	node_no_data_vars = RemoveAssignSkVars().visit(node_data_vars_reduced)
	#fout.write(ast.unparse(node_no_data_vars))
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
	#sk_dict contiene todos los nodos de las funciones que busco
	#==========================================
	#ahora escribo el codigo linea a linea, comparando cada nodo por si esta en el diccionario, si esta en el diccionario le cambio el nombre al modelo
	code_to_write = ast.unparse(node_no_data_vars).split('\n')
	for model_number in range(1,num_subredes): #desde 1 para hacer una menos, ya que la original se escribe por defecto
		TransformModelName(sk_dict,model_number).visit(node_no_data_vars)
		fout.write(ast.unparse(node_no_data_vars))
		fout.write('\n\n')
			

	pass

def main():
	print(n)
	sk_file = create_new_file(file)
	index_pragmas,sk_trees,init_code,post_sk_codes,skynnet_configs = prepare_sk_file(file)
	'''print(index_pragmas)
	print("=================")
	print(sk_trees)
	print("=================")
	print(init_code)
	print("=================")
	print(post_sk_codes)'''
	print(skynnet_configs)
	with open(sk_file,'w') as fo:
		for tree_number,indexes in enumerate(index_pragmas):
			#escribo el principio
			fo.write(init_code)
			#escribo skynnet
			#fo.write(sk_trees[tree_number])
			code = sk_trees[tree_number]
			process_skynnet_code(code, skynnet_configs[tree_number], fo, n)
			#escribo el final
			fo.write(post_sk_codes[tree_number])

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


