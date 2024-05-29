import os
import sqlite3
import shutil
import re

import tkinter as tk
from tkinter import filedialog


# ===================================================================================
# Esta función sirve para seleccionar el campo texto de la foto dado el identificador de Adobe Lightroom
def get_text(photo_id, cur):
    # Obtener el campo Adobe_ImageDevelopSettings.text de esa imagen
    sql = "SELECT Adobe_ImageDevelopSettings.text FROM Adobe_ImageDevelopSettings " \
          "INNER JOIN Adobe_Images ON Adobe_ImageDevelopSettings.image = Adobe_Images.id_local " \
          "WHERE Adobe_Images.rootFile = ?"
    cur.execute(sql, (photo_id,))
    resultado_texto = cur.fetchone()

    if resultado_texto:
        textoEntrada = resultado_texto[0]
    else:
        print("No se encontró el campo Adobe_ImageDevelopSettings.text para la foto:", photo_id)
        return None

    # Modificamos el texto para facilitar el procesamiento
    textoEntrada = textoEntrada[6:-2]

    # Convertimos el texto a diccionario
    diccionario = text_to_dict(textoEntrada)

    return diccionario


# ===================================================================================
# Esta función procesa los valores para poder pasarlos de texto a diccionario
def process_value(value):
    # Eliminamos los espacios en blanco al principio y al final del valor
    value = value.strip()
    if value.startswith('{') and value.endswith('}'):
        # Eliminamos los corchetes y dividimos los elementos separados por comas en una lista de Python
        value = value[1:-1].split(',')
        return [v.strip() for v in value]
    elif value.isdigit():
        return int(value)
    elif value.replace(".", "", 1).isdigit():
        return float(value)
    elif value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        return value.rstrip(',')


# ===================================================================================
# Método que recibe la columna Text de la tabla Adobe_ImageDevelopSettings y la convierte en un diccionario de Python.
def text_to_dict(text):
    result_dict = {}
    lines = text.strip().split('\n')
    key = None
    value_lines = []
    for line in lines:
        line = line.strip()
        if '=' in line:
            if key:
                value = '\n'.join(value_lines)
                result_dict[key] = process_value(value)
                value_lines = []
            key, value = line.split('=', 1)
            key = key.strip()
            value_lines.append(value.strip())
        else:
            value_lines.append(line)

    if key:
        value = '\n'.join(value_lines)
        result_dict[key] = process_value(value)

    return result_dict


# ===================================================================================
# Esta función procesa los valores para pasarlos a texto
def format_value(value):
    if isinstance(value, str):
        return f'{value}'
    elif isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, list):
        elements = ', '.join(str(e) for e in value)
        return '{ ' + elements + ' }'
    else:
        return str(value)


# ===================================================================================
# Esta función recibe un diccionario de Python y lo convierte a texto en el mismo formato que la columna Text de la tabla.
def dict_to_text(data):
    lines = []
    for key, value in data.items():
        formatted_value = format_value(value)
        lines.append(f"{key} = {formatted_value},")
    cadena = "\n".join(lines)
    cadena = cadena.replace(':', ' =')  # Reemplazar ":" por "=" en la cadena final
    cadena = "s = { " + cadena + " }"

    return cadena


# ===================================================================================
# Esta función nos permite actualizar el diccionario, añadiendo nuevos valores
def update_dict(cur, photo, updates):
    # Verificar si la foto existe en la tabla AgLibraryFile
    sql = "SELECT id_local FROM AgLibraryFile WHERE idx_filename = ?"
    cur.execute(sql, (photo,))
    resultado = cur.fetchone()

    if resultado:
        photo_id = resultado[0]
        diccionario = get_text(photo_id, cur)

        if diccionario is None:
            return None

        for key, value in updates:
            diccionario[key] = value

        return diccionario, photo_id
    else:
        print("La foto no existe en la tabla AgLibraryFile:", photo)
        exit()


# ===================================================================================
# Esta función crea un nuevo lrcat realizando una copia del lrcat que se le pasa como parámetro
def create(old_lrcat_path):
    conn = sqlite3.connect(old_lrcat_path)
    new_lrcat_name = "new" + ".lrcat"

    # Obtener la ruta del directorio del archivo seleccionado
    lrcat_dir = os.path.dirname(old_lrcat_path)

    # Crear la ruta completa del nuevo LRCAT en el mismo directorio
    new_lrcat_path = os.path.join(lrcat_dir, new_lrcat_name)

    # Copiar el archivo LRCAT original al nuevo archivo
    shutil.copy2(old_lrcat_path, new_lrcat_path)

    # Guardar los cambios en la base de datos
    conn.commit()

    # Cerrar la conexión
    conn.close()

    return new_lrcat_path


# ===================================================================================
# Esta función nos permite aplicar los cambios "updates" en la foto "photo"
def edit(lrcat_path, photo, updates):
    # Usar with para manejar la conexión a la base de datos
    with sqlite3.connect(lrcat_path) as conn:
        cur = conn.cursor()

        diccionario, photo_id = update_dict(cur, photo, updates)

        # Convertir el diccionario a texto en el formato deseado (entendible por Adobe Lightroom)
        textoSalida = dict_to_text(diccionario)

        # Definir la sentencia SQL de actualización
        sql = "UPDATE Adobe_ImageDevelopSettings " \
            "SET text = ? " \
            "WHERE image IN (SELECT id_local FROM Adobe_Images WHERE rootFile = ?)"

        # Ejecutar la sentencia SQL con el contenido de textoSalida
        cur.execute(sql, (textoSalida, photo_id))

        # Guardar los cambios en la base de datos
        conn.commit()


# ===================================================================================
# Esta función sirve para seleccionar los nombres de la fotos dado el lrcat
def get_names(lrcat_path):
    # Conectar a la base de datos SQLite
    conn = sqlite3.connect(lrcat_path)
    cur = conn.cursor()

    # Ejecutar la consulta SQL para obtener los nombres de las fotos
    query = "SELECT idx_filename FROM AgLibraryFile"
    cur.execute(query)

    # Obtener todos los nombres de las fotos
    nombres_de_fotos = cur.fetchall()

    # Cerrar la conexión a la base de datos
    conn.close()

    return nombres_de_fotos
 

# ====================================================================================
# Función que dado el lrcat genera un lrcat nuevo con las modificaciones que se especifican en el nombre del archivo
# Modifica la imagen blue_35_15.png poniéndole 35 en el control HUE y 15 en el control SATURATION de Lightroom
def generate_HSL_lrcat(lrcat):
    regex = r'([a-zA-Z0-9]+)_(-?\d+)_(-?\d+)\.jpg'
    names = get_names(lrcat)
    contador = 0
    for nombre in names:
        coincidencia = re.search(regex, nombre[0])
        if coincidencia:
            color, hue, sat = coincidencia.groups()
            hue, sat = map(int, (hue, sat))
            print(contador, nombre[0])
            updates = [
                ('HueAdjustment' + color, hue),
                ('SaturationAdjustment' + color, sat)
            ]
            edit(lrcat, nombre[0], updates)
            contador += 1
        else:
            print("No se encontraron coincidencias en el nombre del archivo.")


# ====================================================================================
if __name__ == "__main__":
    # Crear una ventana de tkinter invisible
    root = tk.Tk()
    root.withdraw()

    # Path al archivo LRCAT
    lrcat = None

    while not lrcat:
        # Abrir el cuadro de diálogo del explorador de archivos
        lrcat = filedialog.askopenfilename(title="Seleccione el archivo LRCAT")

        # Verificar si se seleccionó un archivo
        if not lrcat:
            print("Error: No se ha seleccionado ningún archivo. Por favor, selecciona un archivo.")
            exit()
        else:
            print("Editando ", os.path.basename(lrcat))
            generate_HSL_lrcat(lrcat)
