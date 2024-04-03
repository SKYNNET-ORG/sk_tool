# sk_tool
0. Descarga de la herramienta sk_tool: https://github.com/SKYNNET-ORG/sk_tool
1. Requisitos:
	1. Si usas python básico:  Versión 1.0.1 de la librería ast_comments (descargar via pip >pip install ast-comments=1.0.1)
	2. Si usas Anaconda: El fichero del entorno se encuentra en la carpeta descargada sk_tool.yaml
2. En la carpeta descargada de git se encuentran los siguientes ficheros y carpetas:
	1. Ficheros: 
		1. sk_tool.py: Codigo fuente de la herramienta
		2. sk_extra_codes.py: Codigo auxiliar de la herramienta
		3. sk_tool.yaml: Entorno de anaconda
	2. Carpetas: 
		1. input: Ficheros de entrada de la herramienta
		2. output: Ficheros de salida de la herramienta
			1. Contiene un fichero llamado test_all.bat: Es un fichero para probar todos los scripts que haya en la carpeta output, útil para realizar pruebas
		3. test: Batería de ficheros para ejecutar pruebas
	Dentro de las carpetas se encuentran ficheros de pruebas de concepto a modo de ejemplo.
3. Para uso normal:
	1. Colocas el fichero original en la carpeta inputs, por ejemplo sk_tool/inputs/original.py
	2. Ejecutas: >py sk_tool.py original numero_subredes_deseadas
	3. En la carpeta output se genera el fichero: sk_tool/output/sk_original.py
	4. Ejecutas la red deconstruida en sk_tool/output/sk_original.py
4. Para uso en modo test de uno o varios ficheros:
	1. Colocas los ficheros originales en la carpeta test 
	2. Ejecutas: >py sk_tool.py test
	3. En la carpeta output deconstruye todos los ficheros que hay en la carpeta test para cuatro subredes.


