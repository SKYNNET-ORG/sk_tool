import cv2
import os
import time
import sys

import glob
import shutil
import os

# modulos de la app
import dataTrainGen
import trainer
import tester
import cribado
import tensorflow as tf
import libaux
import numpy as np

from pathlib import Path
import re

import keras

#__CLOUDBOOK:GLOBAL__
dir_profile = ""
ENTRENAR = False
CREAR_MODELO = False
CARGAR_MODELO = False
#__CLOUDBOOK:NONSHARED__
debuglevel = 4
data = None
data_np = None
dir_profile = ""
brute_file_list = []
retouched_file_list = []



########################################################################################
# define a function for vertically
# concatenating images of different
# widths
#__CLOUDBOOK:LOCAL__
def vconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum width
    w_min = min(img.shape[1] for img in img_list)
    # resizing images
    im_list_resize = [cv2.resize(img,
                                 (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation=interpolation) for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)


########################################################################################
#__CLOUDBOOK:LOCAL__
def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    # take minimum hights
    h_min = min(img.shape[0] for img in img_list)
    # image resizing
    im_list_resize = [cv2.resize(img,
                                 (int(img.shape[1] * h_min / img.shape[0]),
                                  h_min), interpolation=interpolation) for img in img_list]
    # return final image
    return cv2.hconcat(im_list_resize)


########################################################################################
#__CLOUDBOOK:LOCAL__
def concat_tile_resize(list_2d, interpolation=cv2.INTER_CUBIC):
    # function calling for every
    # list of images
    img_list_v = [hconcat_resize(list_h, interpolation=cv2.INTER_CUBIC) for list_h in list_2d]
    # return final image
    return vconcat_resize(img_list_v, interpolation=cv2.INTER_CUBIC)

#######################################################################
#__CLOUDBOOK:LOCAL__
def gen_data_input_listbulk(dir_bruto, brute_file_list,retouched_file_list, profiledir, debuglevel, CRIBADAS=False):
    """

    Retoca las fotos de un directorio o de una lista
    usando el modelo que encuentre en profiledir

    Parameters:

    
    - dirbruto: dire de las fotos en bruto
    - lista_brute lista de fotos en bruto cribadas. if None se cogen todas las fotos de dirbruto
    - lista_retouch lista de fotos retocadas por fotografo cribadas
    - profiledir: dir de profile
    - debuglevel:
        debuglevel=0 fotos retocadas por rna
        debuglevel=1 fotos dobles (bruta y RNA)
        debuglevel=2 fotos triples (bruta, fotografo, RNA). Solo valido si listas !=None
        debuglevel=3 fotos dobles con histogramas (4 elementos en una foto). NO programado
        debuglevel=4 fotos triplesretocadas y con histogramas (6 elementos en una foto)
        
    Returns:
      deja en profiledir/testRetouch/ las fotos retocadas reducidas al 50%
    
      0 : exito
      -1 : no encuentra directorio o algun fichero
      -2 : no encuentra el modelo
      -3 : debuglevel incompatible
    
    """
    # si lista brute esta vacia, la construimos
    print("============================Enter in bulk")
    if (brute_file_list==None):
        brute_file_list=os.listdir(dir_bruto)
        #concatenamos los directorios
        for indice in range(len(brute_file_list)):
            brute_file_list[indice] = dir_bruto+brute_file_list[indice]
    # compatibilidad del debug level
    if (retouched_file_list==None):
        if (debuglevel==2 or debuglevel==4):
            print ("Tester: debuglevel incompatible con lista retouched a None")
            sys.exit()
       
    # retocamos una a una cada imagen
    index=0
    data_inputs=[]
    for file_1 in brute_file_list:
        # ahora hay que generar los datos de entrada (ojo no son de entrenamiento)
        print("leyendo imagen...", file_1, " y generando su data input")
        ret,x=dataTrainGen.dataGenImage(file_1)
        if (ret!=0):
            print("error accediendo a la imagen:", file_1)
            sys.exit()
        data_inputs.append(x)
        index=index+1

    print ("hemos terminado de leer y generar el array de input data")
    print();
    data_np=np.array(data_inputs)
    data_np.shape=(index,774)
    print("==================================Exit bulk")
    return (data_np,brute_file_list,retouched_file_list)

########################################################################################
#                    PROGRAMA MAIN
########################################################################################

#__CLOUDBOOK:DU0__
def main():
    if hasattr(main,'executed'):
        return
    else:
        setattr(main,'executed',True)
    global dir_profile
    global ENTRENAR
    global CREAR_MODELO
    global CARGAR_MODELO

    sk_local_main_executed = True


    perfil = input("Que perfil quieres usar? ")
    dir_aux = "./"+perfil
    dir_profile = dir_aux
     
    # Crear el directorio si no existe
    if not os.path.exists(dir_profile):
        os.mkdir(dir_profile)
        print(f"Directorio '{dir_profile}' creado exitosamente.")
        #os.chdir(dir_profile)
        #print(f"La ubicación ahora es: {os.getcwd()}")
    #else:
        #os.chdir(dir_profile)
        #print(f"La ubicación ahora es: {os.getcwd()}")

    # Preguntar qué quiere hacer
    print("====App fotografica=====")
    print("\n¿Qué quieres hacer?")
    print("1. Crear modelo y entrenar")
    print("2. Cargar modelo y reentrenar")
    print("3. Cargar modelo y predecir")
     
    opcion = input("Selecciona una opción (1/2/3): ")
    if opcion == "1":
        CREAR_MODELO = True
        CARGAR_MODELO = False
        ENTRENAR = True
    elif opcion == "2":
        CREAR_MODELO = False
        CARGAR_MODELO = True
        ENTRENAR = True
    elif opcion == "3":
        CREAR_MODELO = False
        CARGAR_MODELO = True
        ENTRENAR = False
    else:
        print("Opción no válida. Por favor, selecciona 1, 2 o 3.")
    load_data()

#__CLOUDBOOK:LOCAL__
def load_data():
    global ENTRENAR
    global CREAR_MODELO
    global CARGAR_MODELO
    global dir_profile
    global data
    global data_np
    
    global brute_file_list
    global retouched_file_list
     
    # variables globales
    # ------------------
    # dir_profile = "./mini"
    
    dir_bruto = "./bruto"#os.path.join(dir_profile, "bruto")
    dir_retouch = "./trans"#os.path.join(dir_profile, "trans")

     
    
    # ===================================COMIENZA MAIN =====================================
    # obtencion de lista de fotos sin cribar
    # --------------------------------------
    brute_file_list = os.listdir(dir_bruto)
    retouched_file_list = os.listdir(dir_retouch)

    total_fotos = len(brute_file_list)
    print("total fotos en bruto:", total_fotos)

    # cribado
    # ---------
    start = time.time()
    print("cribando fotos para entrenar en dir bruto=", dir_bruto)
    unique_list, brute_file_list, retouched_file_list = cribado.compare_folders(dir_bruto, dir_retouch)
    brute_file_list, retouched_file_list = cribado.criba(unique_list, brute_file_list, retouched_file_list, dir_profile)
    print("cribando terminado")

    end = time.time()
    criba_time = end - start
    print(" criba: tiempo transcurrido (segundos) =", criba_time)

    # generacion de datos de entrenamiento
    # --------------------------------------
    total_fotos_cribadas = len(brute_file_list)
    start = time.time()
    if (ENTRENAR):
        res, data = dataTrainGen.dataGen(brute_file_list, retouched_file_list)  # genera data para entrenar
    else:
        res, data = (0, [0,0])  # caso de no entrenar
    end = time.time()
    data_gen_time = end - start
    print(" datagen: tiempo transcurrido (segundos) =", data_gen_time)
    if (res != 0):
        sys.exit()

    print("dataTrainGen.dataGen:", res)

    data_np,brute_file_list,retouched_file_list = gen_data_input_listbulk(dir_bruto, brute_file_list,retouched_file_list, dir_profile, debuglevel, CRIBADAS=False)


main()

def refresh_ENTRENAR():
    global ENTRENAR
    return ENTRENAR

def refresh_CREAR_MODELO():
    global CREAR_MODELO
    return CREAR_MODELO

def refresh_CARGAR_MODELO():
    global CARGAR_MODELO
    return CARGAR_MODELO

def refresh_dir_profile():
    global dir_profile
    return dir_profile

def refresh_flag():
    global flag
    return flag

#SKYNNET:BEGIN_REGRESSION

entrenar = refresh_ENTRENAR()
crear_modelo = refresh_CREAR_MODELO()
cargar_modelo = refresh_CARGAR_MODELO()
dir_profile = refresh_dir_profile()
load_data()
data_0 = data[0]
data_1 = data[1]
_DATA_TRAIN_X = data_0
_DATA_TRAIN_Y = data_1
_DATA_TEST_X = data_np
_NEURON_1 = 774
_NEURON_2 = 1548
_NEURON_3 = 1536

epocas = 0
if cargar_modelo:
    model = tf.keras.models.load_model('model.h5')

if (entrenar):
    ############################################################################################
    input_data = tf.keras.layers.Input(shape=(256 * 3 + 6,))  # antes 4
    encoded = tf.keras.layers.Dense(_NEURON_1, activation='relu6')(input_data)
    encoded = tf.keras.layers.Dense(_NEURON_2, activation='relu')(encoded)  # mejor que selu
    decoded = tf.keras.layers.Dense(_NEURON_3, activation='softplus')(encoded)
    if crear_modelo:
        model = tf.keras.models.Model(input_data, decoded)
        model.compile(optimizer='adam', loss='mse')
        print("Modelo creado.")
        epocas = 200
        print(model.summary())
    else:
        epocas = 25

    # log_train = trainModel(model, (inputs, desired_output), epocas)
    start = time.time()
    trained_model = model.fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, epochs=epocas, batch_size=4)
    end = time.time()   
    train_time = end - start
    print(" train: tiempo transcurrido (segundos) =", train_time)
    # salva el modelo en el directorio de profile
    # libaux.saveModel(model, profiledir)
    model.save('model.h5')            
    
    ################################################################################################


print("")
g=model.predict(_DATA_TEST_X)

dir_profile2 = refresh_dir_profile()
dir_profile = dir_profile2
print("Prediction done")
print(brute_file_list)
print(dir_profile)
#print(f"shape de data_np= {data_np.shape}, de data_test={_DATA_TEST_X.shape}")
#g=model.predict(data_np)
CRIBADAS=False
index=0;
for file_1 in brute_file_list:
    #ejecutamos la RNA para sacar la funcion de transformacion
    f=g[index] #model.predict(x)
    
    f=np.array(f*255)
    
    f[0]=np.clip(f[0], a_min = 0, a_max = 255) #topar a 0..255
    #f=ajustaf(f[0])
    
    f=np.asarray(f).astype('int')

    f.shape=(2,3,256)

    
    #ahora aplicamos la funcion de transformacion a la imagen
    img1=cv2.imread(file_1, cv2.IMREAD_COLOR )
    print (img1.shape)
    img1t=libaux.transformRGBYUV(img1, f[0], f[1])

    
    #img1t=libaux.transform(img1, f)
    img1tr=libaux.imgResize(img1t,50)# reduce al 50%
    
    #crea el directorio de imagenes si no existe
    directorio = dir_profile+"/RNA_images/"
    try:
        os.stat(directorio)
    except:
        os.mkdir(directorio)
    # salvamos la imagen
    file_rna=tester.getRNAname(file_1)
    print ("file:", file_rna)
    file_rna=dir_profile+"/RNA_images/"+file_rna
    
    # truco para distinguir cribadas
    # --------------------------
    if (CRIBADAS==True):
        file_rna=dir_profile+"/RNA_images/cribada_"+tester.getRNAname(file_1)
    
   
    print ("salvando en:", file_rna, " con debug level", debuglevel)
    if (debuglevel==0): # 0= la foto retocada y reducida 50%
        cv2.imwrite(file_rna,img1tr)
        
    elif (debuglevel==1):# 1= foto orig + RNA  reducidas al 25% 

        img1tr=libaux.imgResize(img1tr,50)# reduce al 50% otra vez
        img1r=libaux.imgResize(img1,25)
        #Escribir texto
        cv2.putText(img1tr, "RNA", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(img1r, "ORIGINAL", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,255,255), 2)
        im_v = cv2.vconcat([img1r, img1tr])
        cv2.imwrite(file_rna,im_v)

    elif (debuglevel==2): #2= foto orig + fotografo +RNA  reducidas al 25% 
        img1tr=libaux.imgResize(img1tr,50)# reduce al 50% otra vez
        img1r=libaux.imgResize(img1,25)
        file_2=retouched_file_list[index]
        img2=cv2.imread(file_2, cv2.IMREAD_COLOR )
        img2r=libaux.imgResize(img2,25)
        #Escribir texto
        cv2.putText(img1r, "ORIGINAL", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,255,255), 2)
        cv2.putText(img2r, "FOTOGRAFO", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,255,255), 2)
        cv2.putText(img1tr, "RNA", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        img_h_resize = concat_tile_resize([[img1r,img2r, img1tr]])
        cv2.imwrite(file_rna,img_h_resize)

    elif (debuglevel==4):#4= foto orig + fotografo +RNA  + 3 histogramas reducidas al 25% 
        img1tr=libaux.imgResize(img1tr,50)# reduce al 50% otra vez
        img1r=libaux.imgResize(img1,25)
        file_2=retouched_file_list[index]
        print("file 2 es ", file_2)
        img2=cv2.imread(file_2, cv2.IMREAD_COLOR )
        img2r=libaux.imgResize(img2,25)
        #Escribir texto
        cad=Path(file_2).stem
        numerito=[int(s) for s in re.findall(r'-?\d+\.?\d*', cad)]
        numerito=abs(int(str(numerito[0])))
        yo=0
        xo=100
        cv2.rectangle(img1r, (xo ,yo), (xo + 60, yo + 25), (0,0,0), -1)
        cv2.putText(img1r, str(numerito), (100,23), cv2.FONT_HERSHEY_SIMPLEX, 1,  (255,255,255), 2)
        
        cv2.putText(img1r, "ORIGINAL", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,255,255), 2)
        cv2.putText(img2r, "FOTOGRAFO", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255,255,255), 2)
        cv2.putText(img1tr, "RNA", (0,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        #histogramas
        libaux.painthist(img1,"ORIGINAL", 50,dir_profile+"/RNA_images/tmp1.jpg")
        libaux.painthist(img2,"FOTOGRAFO", 50,dir_profile+"/RNA_images/tmp2.jpg")
        libaux.painthist(img1t,"RNA", 50,dir_profile+"/RNA_images/tmp3.jpg")
        ancho=img1t.shape[1]
        hist1=cv2.imread(dir_profile+"/RNA_images/tmp1.jpg", cv2.IMREAD_COLOR )
        hist2=cv2.imread(dir_profile+"/RNA_images/tmp2.jpg", cv2.IMREAD_COLOR )
        hist3=cv2.imread(dir_profile+"/RNA_images/tmp3.jpg", cv2.IMREAD_COLOR )
        #collage
        img_h2 = concat_tile_resize([[img1r,img2r, img1tr],
                                     [hist1,hist2, hist3] ])
        print("salvando ", file_rna)
        cv2.imwrite(file_rna,img_h2)

       

        # Ruta de la carpeta de origen
        ruta_origen = '.'

        # Ruta de la carpeta de destino
        ruta_destino = dir_profile

        # Obtener la lista de archivos .h5 en la carpeta de origen
        archivos_h5 = glob.glob(os.path.join(ruta_origen, '*.h5'))

        # Copiar cada archivo .h5 a la carpeta de destino
        for archivo in archivos_h5:
            shutil.copy(archivo, ruta_destino)



#SKYNNET:END

#==============================================================
