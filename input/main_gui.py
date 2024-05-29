import os
import shutil
import time
import sys

# modulos de la app
import dataTrainGen
import trainer
import tester
import cribado

import tkinter as tk
from tkinter import messagebox
import tkinter.simpledialog as sd
import tkinter.filedialog as fd


# definimos la ruta por defecto
PATH = r"C:\Users\guerrero\VisualStudio\source\SKYNNET\Profiles"


# ==========================================================================
def train_profile(profile, dir_bruto, dir_retouch):
    print(dir_bruto)
    print(dir_retouch)
    # sin cribar
    brute_file_list = os.listdir(dir_bruto)
    retouched_file_list = os.listdir(dir_retouch)

    # cribado
    start = time.time()
    print("cribando fotos para entrenar...")
    unique_list, brute_file_list, retouched_file_list = cribado.compare_folders(dir_bruto, dir_retouch)
    cribado.criba(unique_list, brute_file_list, retouched_file_list, profile)
    print("cribado terminado")
    end = time.time()
    criba_time = end - start
    print("criba: tiempo transcurrido (segundos) =", criba_time)

    # concatenamos los directorios
    for indice in range(len(brute_file_list)):
        brute_file_list[indice] = brute_file_list[indice]
        retouched_file_list[indice] = retouched_file_list[indice]

    # ahora comienza el procesamiento
    start = time.time()
    res, data = dataTrainGen.dataGen(brute_file_list, retouched_file_list)  # genera data para entrenar
    print(res)
    print(data)
    end = time.time()
    data_gen_time = end - start
    print(" datagen: tiempo transcurrido (segundos) =", data_gen_time)
    if (res != 0):
        sys.exit()
    print("dataTrainGen.dataGen:", res)
    # ahora pasamos a entrenar
    start = time.time()
    res = trainer.trainProfile(profile, data)  # aquí se genera el modelo h5
    print("trainer.trainProfile:", res)
    end = time.time()
    train_time = end - start
    print(" train: tiempo transcurrido (segundos) =", train_time)


# ==========================================================================
def debug_profile(bruto_path, trans_path, model, level):
    start = time.time()
    res = tester.testRetouch(bruto_path, trans_path, model, level)
    end = time.time()
    test_time = end - start
    print(" test: tiempo transcurrido (segundos) =", test_time)
    print("tester.testRetouch:", res)


# ==========================================================================
def execute_profile(bruto, lrcat, model, level):
    start = time.time()
    res = tester.retouchLrcat(bruto, lrcat, model, level)
    end = time.time()
    test_time = end - start
    print(" execute: tiempo transcurrido (segundos) =", test_time)
    print("tester.retouchLrcat:", res)


# ==========================================================================
def main_menu_opt():
    """
    Displays the main menu options and prompts the user for input.
    Options:
    0. Exit - Exits the program.
    1. Edit Default Path - Allows the user to edit the default path for file storage.
    2. Create/Edit Training Profile - Allows the user to create or edit a training profile.
    3. Train Training Profile - Trains the selected training profile.
    4. Execute - Executes the photo editing process.
    5. Debug - Enters the debug mode for troubleshooting.
    Returns:
    bool: True if a valid option is selected, False if the option is to exit.
    """
    # Funciones para manejar eventos de botones
    def exit_program():
        resultado = messagebox.askquestion("Salir", "¿Estás seguro de que quieres salir de la aplicación?")
        if resultado == "yes":
            # Lógica para salir de la aplicación
            messagebox.showinfo("Salir", "¡Hasta pronto!")
            window_main.destroy()
        else:
            # Lógica para cancelar la salida de la aplicación
            pass
    """
    def edit_default_path():
        # Mostrar cuadro de diálogo para editar la ruta por defecto
        global PATH
        new_path = fd.askdirectory(title="Editar Ruta", initialdir=PATH)
        if new_path:
            # Si se ingresó una nueva ruta, realizar la edición
            PATH = new_path
            messagebox.showinfo("Editar Ruta", "Ruta actualizada exitosamente")
        ruta.config(text=f"Ruta actual:\n{PATH}\n")

        # Calculamos la anchura total del contenido dentro del frame
        contenido_width = ruta.winfo_reqwidth()  # Obtén la anchura requerida del contenido

        # Ajustamos el tamaño de la ventana en función de la anchura total del contenido
        window_main_width = contenido_width + 20  # Agregamos un margen de 20 píxeles

        if window_main_width > 450:
            window_main.geometry(f"{window_main_width}x650")  # Mantenemos la altura de la ventana igual
    """

    def create_training_profile():
        # Mostrar cuadro de diálogo para crear/editar perfil de entrenamiento
        nuevo_perfil = sd.askstring("Nuevo perfil", "Introduce un nombre para el nuevo perfil")
        ruta_perfil = os.path.join(PATH, nuevo_perfil)
        if os.path.exists(ruta_perfil):
            print("El perfil ya existe.")
        else:
            os.makedirs(ruta_perfil)
            print(f"El perfil {nuevo_perfil} ha sido creado exitosamente en {ruta_perfil}.")

    def edit_training_profile():
        global PATH

        # Función para eliminar un perfil seleccionado
        def eliminar_perfil():
            if messagebox.askquestion("Eliminar perfil", f"¿Estás seguro de que quieres eliminar el perfil seleccionado en {directorio}?") == "yes":
                shutil.rmtree(directorio)  # Elimina el directorio seleccionado
                messagebox.showinfo("Eliminar perfil", f"Perfil seleccionado eliminado: {directorio}")
            window_profile.destroy()  # Cierra la ventana

        # Función para editar un perfil seleccionado
        def editar_perfil():
            nuevo_nombre = sd.askstring("Editar perfil", "Introduce un nuevo nombre para el perfil seleccionado:")
            if nuevo_nombre:
                nuevo_path = os.path.join(os.path.dirname(directorio), nuevo_nombre)
                os.rename(directorio, nuevo_path)  # Renombra el directorio seleccionado
                messagebox.showinfo("Editar perfil", f"Perfil seleccionado renombrado como {nuevo_nombre}")
            window_profile.destroy()  # Cierra la ventana

        # Función para clonar un perfil seleccionado
        def clonar_perfil():
            perfil_clonado = sd.askstring("Clonar perfil", "Introduce un nombre para el nuevo perfil:")
            if perfil_clonado:
                ruta_clonada = os.path.join(os.path.dirname(directorio), perfil_clonado)
                if os.path.exists(ruta_clonada):
                    if messagebox.askquestion("Clonar perfil", f"El perfil {perfil_clonado} ya existe. ¿Desea sobrescribirlo?") == "yes":
                        shutil.rmtree(ruta_clonada)
                        shutil.copytree(directorio, ruta_clonada)  # Crea una copia del directorio seleccionado con el nuevo nombre
                        messagebox.showinfo("Clonar perfil", f"Perfil seleccionado clonado como {perfil_clonado}")
                else:
                    shutil.copytree(directorio, ruta_clonada)  # Crea una copia del directorio seleccionado con el nuevo nombre
                    messagebox.showinfo("Clonar perfil", f"Perfil seleccionado clonado como {perfil_clonado}")
            window_profile.destroy()  # Cierra la ventana

        # Abrir cuadro de diálogo para seleccionar directorio
        directorio = fd.askdirectory(title="Seleccionar directorio", initialdir=PATH)
        if directorio:
            # Crear ventana
            window_profile = tk.Tk()
            window_profile.title("Opciones de perfil")
            # Configuramos el tamaño de la ventana
            window_profile.geometry("280x120")

            # Configuramos el fondo de la ventana
            window_profile.configure(bg='white')

            # Hacemos que la ventana no sea redimensionable
            window_profile.resizable(False, False)
            # Crear botones de opciones
            btn_eliminar = tk.Button(window_profile, text="Eliminar perfil", command=eliminar_perfil)
            btn_editar = tk.Button(window_profile, text="Cambiar nombre del perfil", command=editar_perfil)
            btn_clonar = tk.Button(window_profile, text="Clonar perfil", command=clonar_perfil)

            # Agregar botones a la ventana
            btn_eliminar.pack(pady=5)
            btn_editar.pack(pady=5)
            btn_clonar.pack(pady=5)

            # Iniciar el bucle de eventos de la ventana
            window_profile.mainloop()

    def train_training_profile():
        # Mostrar cuadro de diálogo para seleccionar directorio
        profile_path = fd.askdirectory(title="Seleccionar perfil", initialdir=PATH)
        # Verificar si se seleccionó un directorio
        if profile_path:
            bruto_path = fd.askdirectory(title="Seleccionar directorio fotos en bruto", initialdir=PATH)
            if bruto_path:
                retouch_path = fd.askdirectory(title="Seleccionar directorio fotos editadas", initialdir=PATH)
                if retouch_path:
                    # Obtener el nombre del directorio a partir de la ruta seleccionada
                    train_profile(profile_path, bruto_path, retouch_path)
                else:
                    # Si no existe mostrar error
                    messagebox.showerror("Error", f"No se seleccionó directorio para fotos editadas.")
            else:
                # Si no existe mostrar error
                messagebox.showerror("Error", f"No se seleccionó directorio para fotos en bruto.")
        else:
            # Si no existe mostrar error
            messagebox.showerror("Error", f"No se seleccionó ningún perfil.")

    def execute():
        # Mostrar cuadro de diálogo para seleccionar directorio
        profile_path = fd.askdirectory(title="Seleccionar perfil", initialdir=PATH)
        # Verificar si se seleccionó un directorio
        if profile_path:
            bruto_path = fd.askdirectory(title="Seleccionar directorio fotos en bruto", initialdir=PATH)
            if bruto_path:
                lrcat_path = fd.askopenfilename(title="Seleccionar lrcat", initialdir=PATH)
                if lrcat_path:
                    execute_profile(bruto_path, lrcat_path, profile_path, 0)
                else:
                    # Si no existe mostrar error
                    messagebox.showerror("Error", f"No se seleccionó ningún lrcat.")
            else:
                # Si no existe mostrar error
                messagebox.showerror("Error", f"No se seleccionó directorio para fotos en bruto.")
        else:
            # Si no existe mostrar error
            messagebox.showerror("Error", f"No se seleccionó ningún perfil.")

    def debug_menu(level):
        profile_path = fd.askdirectory(title="Seleccionar perfil", initialdir=PATH)
        if profile_path:
            bruto_path = fd.askdirectory(title="Seleccionar directorio fotos en bruto", initialdir=PATH)
            if bruto_path:
                # Solicitar trans_path solo para los niveles 2 y 4
                if level == 2 or level == 4:
                    trans_path = fd.askdirectory(title="Seleccionar directorio fotos transformadas", initialdir=PATH)
                    if not trans_path:
                        messagebox.showerror("Error", "No se seleccionó directorio para fotos transformadas.")
                        return  # Salir de la función si no se selecciona trans_path para niveles 2 y 4
                # Realizar acciones específicas para cada nivel
                if level == 0:
                    debug_profile(bruto_path, None, profile_path, 0)
                    pass
                elif level == 1:
                    debug_profile(bruto_path, None, profile_path, 1)
                    pass
                elif level == 2:
                    debug_profile(bruto_path, trans_path, profile_path, 2)
                    pass
                elif level == 3:
                    debug_profile(bruto_path, None, profile_path, 3)
                    pass
                elif level == 4:
                    debug_profile(bruto_path, trans_path, profile_path, 4)
                    pass
            else:
                messagebox.showerror("Error", "No se seleccionó directorio para fotos en bruto.")
        else:
            messagebox.showerror("Error", "No se seleccionó ningún perfil.")

    def debug():
        # Crear ventana
        window_debug = tk.Tk()
        window_debug.title("Nivel de debug")
        # Configuramos el tamaño de la ventana
        window_debug.geometry("280x220")  # Aumenté la altura de la ventana para acomodar los botones y mensajes

        # Configuramos el fondo de la ventana
        window_debug.configure(bg='white')

        # Hacemos que la ventana no sea redimensionable
        window_debug.resizable(False, False)

        # Crear botones de opciones
        btn_level0 = tk.Button(window_debug, text="Nivel 0: Fotos RNA", command=lambda: [window_debug.destroy(), debug_menu(0)])
        btn_level1 = tk.Button(window_debug, text="Nivel 1: Fotos Raw y RNA", command=lambda: [window_debug.destroy(), debug_menu(1)])
        btn_level2 = tk.Button(window_debug, text="Nivel 2: Fotos Raw, Edited y RNA", command=lambda: [window_debug.destroy(), debug_menu(2)])
        btn_level3 = tk.Button(window_debug, text="Nivel 3: Fotos Raw y RNA + Hist", command=lambda: [window_debug.destroy(), debug_menu(3)])
        btn_level4 = tk.Button(window_debug, text="Nivel 4: Fotos Raw, Edited y RNA + Hist", command=lambda: [window_debug.destroy(), debug_menu(4)])

        # Agregar botones a la ventana
        btn_level0.pack(pady=5)
        btn_level1.pack(pady=5)
        btn_level2.pack(pady=5)
        btn_level3.pack(pady=5)
        btn_level4.pack(pady=5)

        # Iniciar el bucle de eventos de la ventana
        window_debug.mainloop()

    # Crear ventana
    window_main = tk.Tk()
    window_main.title("Menú de Retoque Fotográfico")

    # Configuramos el tamaño de la ventana
    window_main.geometry("450x650")

    # Configuramos el fondo de la ventana
    window_main.configure(bg='white')

    # Hacemos que la ventana no sea redimensionable
    window_main.resizable(False, False)

    # Creamos un widget Canvas para el contenido desplazable
    canvas = tk.Canvas(window_main, bg='white')
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Creamos un frame dentro del canvas para el contenido
    frame = tk.Frame(canvas, bg='white')
    canvas.create_window((0, 0), window=frame, anchor=tk.NW)

    # Creamos una etiqueta con el logo
    logo = tk.Label(frame, text="    ____       __                   __  \n   / __ \___  / /_____  __  _______/ /_ \n  / /_/ / _ \/ __/ __ \/ / / / ___/ __ \\\n / _, _/  __/ /_/ /_/ / /_/ / /__/ / / /\n/_/ |_|\___/\__/\____/\__,_/\___/_/ /_/ \n                /   |  ____  ____       \n               / /| | / __ \/ __ \      \n              / ___ |/ /_/ / /_/ /      \n             /_/  |_/ .___/ .___/       \n                   /_/   /_/            ", font=("Courier", 12), fg="blue", bg="white", justify="left", anchor="c")
    logo.pack(pady=10)

    """
    # Creamos una etiqueta con la ruta
    ruta = tk.Label(frame, text=f"Ruta actual:\n{PATH}\n", font=("Courier", 10), fg="blue", bg="white", justify="left", anchor="w")
    ruta.pack(pady=10)

    # Crear botones para las opciones del menú
    boton_editar_ruta = tk.Button(frame, text="Editar Ruta", command=edit_default_path, width=15)
    boton_editar_ruta.pack(pady=10)
    """

    boton_crear_perfil = tk.Button(frame, text="Crear Perfil", command=create_training_profile, width=15)
    boton_crear_perfil.pack(pady=10)

    boton_editar_perfil = tk.Button(frame, text="Editar Perfil", command=edit_training_profile, width=15)
    boton_editar_perfil.pack(pady=10)

    boton_entrenar_perfil = tk.Button(frame, text="Entrenar Perfil", command=train_training_profile, width=15)
    boton_entrenar_perfil.pack(pady=10)

    boton_ejecutar = tk.Button(frame, text="Ejecutar", command=execute, width=15)
    boton_ejecutar.pack(pady=10)

    boton_debug = tk.Button(frame, text="Debug", command=debug, width=15)
    boton_debug.pack(pady=10)

    boton_salir = tk.Button(frame, text="Salir", command=exit_program, width=15)
    boton_salir.pack(pady=10)
    # Ejecutar bucle de eventos de la ventana
    window_main.mainloop()

# ==========================================================================
if __name__ == "__main__":
    # llamamos a main_menu() solo una vez
    while main_menu_opt():
        pass
