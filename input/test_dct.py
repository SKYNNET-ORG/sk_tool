import tensorflow_datasets as tfds
import tensorflow as tf, numpy as np
from tensorflow.keras import layers,models
import time
import sys # para coger parametros de entrada

import cv2 # opencv para el reescalado y crop de imagenes
import math
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator


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


###########################################################################################################################
#__CLOUDBOOK:DU0__
def generaImages(predicted):
    global x_train
    global y_train
    global x_test
    global y_test
    _DATA_TEST_X = x_test
    _DATA_TEST_Y = y_test
    
    # crea el directorio de imagenes si no existe
    directorio = "./OUT_images/"
    try:
        os.stat(directorio)
    except os.error:
        os.mkdir(directorio)
        
    for i in range(0,4): #4 fotos concretas. el nombre tiene un numero aunque la cero es la uno
        print ("componiendo foto", i) 
        print("input shape:",_DATA_TEST_X[i-1].shape) #i-1 porque comienzan numeracion en cero
        # ESTA ES LA ORIGINAL
        orig=cv2.resize(_DATA_TEST_X[i-1],(256,256))
        #cv2.imshow("orig", orig) 
        #cv2.waitKey(0);
        #file1 = "./OUT_images/orig.jpg"
        orig=orig*255
        #orig = cv2.cvtColor(orig,cv2.COLOR_GRAY2BGR)
        #cv2.imwrite(file1, orig)
        print ("  orig ok")
        
        #ESTA ES LA Y
        pruebaY=(_DATA_TEST_Y[i-1]-0.5)*100 # la dct estaba "comprimida"
        prueba3=np.zeros(64*64)
        prueba3.shape=(64,64)
        pruebaY.shape=(32,32)
        prueba3[0:32,0:32]=pruebaY[0:32,0:32]
        prueba3 = cv2.idct(prueba3)
        prueba3 = cv2.resize(prueba3,(256,256))
        prueba3=prueba3*255
        #file2 = "./OUT_images/orig_y.jpg" 
        #cv2.imwrite(file2, prueba3)
        #cv2.imshow("origY", prueba3) 
        #cv2.waitKey(0);
        print ("  Y ok")
       
        # ESTA ES LA RECONSTRUIDA
        prueba=predicted[i-1]
        prueba=(prueba-0.5)*100 # la dct estaba "comprimida"
        #prueba=prueba[0] # primer elemento de la lista
        prueba.shape=(32,32)
        prueba2=np.zeros(64*64)
        prueba2.shape=(64,64)
        prueba2[0:32,0:32]=prueba[0:32,0:32] #copia de 1er cuadrante
        prueba2 = cv2.idct(prueba2)
        prueba2=cv2.resize(prueba2,(256,256))
        prueba2=prueba2*255
        #file3 = "./OUT_images/rebuilt.jpg" 
        #cv2.imwrite(file3, prueba2)
        #cv2.imshow("rebuilt", prueba2)  
        #cv2.waitKey(0);
        
        print ("  rebuilt ok")

        name="./OUT_images/img_"+str(i)+"_.jpg"
        print ("file name ", name)
        img_h_resize = cv2.hconcat([orig, prueba3, prueba2]) #hconcat_resize([orig, prueba3, prueba2])
        cv2.imwrite(name, img_h_resize)
    
###########################################################################################################################
#__CLOUDBOOK:LOCAL__
def prepareImages(num_freq):
    y_final=np.zeros(0)
    y_final2=np.zeros(0)
    #y_final.shape=(total_images, 64,64,1)
    contenido = os.listdir('./orig_images/')
    print (contenido)
    idx=0
    for img_name in contenido:
        idx=idx+1
        print(img_name)
        name='./orig_images/'+img_name
        img=cv2.imread(name)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(resolution,resolution)) # reduce a 64x64
        #la paso a BN
        name2='./train/all/'+img_name
        #cv2.imshow("orig", img)
        #cv2.waitKey(0);

        #cv2.imwrite(name2,img) # guarda version en BN y 64x64  --> quito la escritura
        
        
        #print("------float orig-----")
        #print (img)
        img=np.float32(img)# ahora ya es un decimal
        img=img/255.0 # ahora ya entre 0 y 1
        """
        print ("  img:")
        print ("el maximo es:",np.max(img))
        print ("el minimo es:",np.min(img))
        print ("el media es:",np.mean(img))
        """
        h,w=img.shape[:2]
        img_dct=cv2.dct(img)#,cv2.DCT_INVERSE)

        img_dct[16:64,0:64]=0
        img_dct[0:64,16:64]=0
        img_dct2=img_dct[0:16,0:16]
        y_final2=np.append(y_final2,img_dct2/100+0.5)
        y_final2.shape=(idx,16,16)
        
        
        #print ("img_dct shape=", img_dct.shape)
        y_final=np.append(y_final,img_dct/100+0.5)
        y_final.shape=(idx,64,64)
        
        #print ("yfinal shape=", y_final[-1].shape)

        #img_dct=img_dct/100.0 # ajuste
        """
        print ("  DCT:")
        print ("el maximo es:",np.max(img_dct)) #supongo max=64
        print ("el minimo es:",np.min(img_dct))#supongo min=-64
        print ("el media es:",np.mean(img_dct)) #supongo max=64
        """
        #elmax=np.max(img_dct)
        
        #k=128/np.max(img_dct)
        #hay que elegir un K, no vale escoger el mejor
        k=4 #4 #2048/128
        off=128
        #img_dct[img_dct>255]=255
        #img_dct[img_dct<0]=0
        
        img_dct=(img_dct*k+off)
        #control de limites
        img_dct[img_dct>255]=255
        img_dct[img_dct<0]=0
        #print("------float -----")
        #print (img_dct)
        #img_dct=img_dct*255
        img_dct = np.uint8(img_dct)
        #img_dct8 =np.array(img_dct, dtype=np.uint8)
        #print("------8 bit-----")
        #print (img_dct)
        

        #img_dct=cv2.dft(img)#,cv2.DCT_INVERSE)
        #img_dct_crop=img_dct[0:num_freq,0:num_freq]#filtro paso bajo
        
        #img_dct[num_freq:h,0:w]=0#filtro paso bajo
        #img_dct[0:h,num_freq:w]=0#filtro paso bajo
        #img_dct[num_freq:h,num_freq:w]=0#filtro paso bajo
        # convert to uint8 para guardar la DCT como imagen
        #img_dct=img_dct*255.0
        #img_dct[img_dct>255]=255
        #img_dct[img_dct<0]=0
        
        #img_dct8 = np.uint8(img_dct)
        #img_dct8 =np.array(img_dct, dtype=np.uint8)
        #img_dct_crop8 = np.uint8(img_dct_crop)
        #img_dct8crop=img_dct8[0:num_freq,0:num_freq]#crop =filtro paso bajo
        name_dct="./trainDCT/all/dct_"+img_name
        #cv2.imshow("DCT", img_dct)
        #cv2.waitKey(0);
        
        #cv2.imwrite(name_dct,img_dct)--> quito la escritura


        #reconstruccion
        img3= np.float32(img_dct)
        #img_dct32 =np.array(img_dct8, dtype=np.float32)
        #img_dct=img_dct/255.0
        img3=(img3-off)/k
        #img3=img3*100.0
        img_dct_inv=cv2.idct(img3)

        img_dct_inv=img_dct_inv *255
        #control de limites
        img_dct_inv[img_dct_inv>255]=255
        img_dct_inv[img_dct_inv<0]=0
        img_dct_inv = np.uint8(img_dct_inv)
        #img_dct_inv=cv2.resize(img_dct_inv,(256,256))
        #print (img_dct_crop8)

        a=(y_final[idx-1]-0.5)*100.0

        #begin con solo 16x16 frecuencias
        a=(y_final2[idx-1]-0.5)*100.0
        b=np.zeros(64*64)
        b.shape=(64,64)
        b[0:16,0:16]=a
        a=b
        #end con solo 16
        #cv2.imshow("dct", a)
        #cv2.waitKey(0)
        #print("a shape:",a.shape)
        a=cv2.idct(a)
        #a=a*255
        #cv2.imshow("inv", a)
        #cv2.waitKey(0)
        #cv2.imshow("inv", img_dct_inv)
        #cv2.waitKey(0);
        name_idct='./idct/idct_'+str(num_freq)+"_"+img_name
        #cv2.imwrite(name_idct,img_dct_inv)---> quito la escritura
    #return y_final
    return y_final2
###########################################################################################################################
#__CLOUDBOOK:LOCAL__
def computex():
    resolution=64
    x_final=np.zeros(0)
    contenido = os.listdir('./orig_images/')
    #print (contenido)
    idx=0
    for img_name in contenido:
        idx=idx+1
        print("loading ", img_name,"   ", end='\r')
        name='./orig_images/'+img_name
        img=cv2.imread(name)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(resolution,resolution)) # reduce a 64x64
        img=np.float32(img)# ahora ya es un decimal
        img=img/255.0 # ahora ya entre 0 y 1
        x_final=np.append(x_final,img)
        #cv2.imshow("orig", img)
        #cv2.waitKey(0)
    #x_final.shape=(idx,resolution,resolution,1)
    x_final.shape=(idx,resolution,resolution)
    print("", end="\n")
    return x_final
###########################################################################################################################
#__CLOUDBOOK:LOCAL__
def computey16(x_train):   # esta funcion no se usa
    y_final2=np.zeros(0)
    idx=0
    for img in x_train:
        idx=idx+1
        img.shape=(64,64)
        #cv2.imshow("orig", img)
        #cv2.waitKey(0)
        img_dct=cv2.dct(img)#,cv2.DCT_INVERSE)
        img_dct[16:64,0:64]=0
        img_dct[0:64,16:64]=0
        img_dct2=img_dct[0:16,0:16]
        y_final2=np.append(y_final2,img_dct2/100+0.5)
        y_final2.shape=(idx,16,16)
        #cv2.imshow("orig", img)
        #cv2.waitKey(0)
    return y_final2
###########################################################################################################################
#__CLOUDBOOK:LOCAL__
def computey(x_train):
    y_final2=np.zeros(0)
    idx=0
    for img in x_train:
        idx=idx+1
        img.shape=(64,64)
        #prueba=cv2.resize(img,(256,256)) 
        #cv2.imshow("orig", prueba)
        #cv2.waitKey(0)
        img_dct=cv2.dct(img)#,cv2.DCT_INVERSE)
        #prueba=cv2.resize(img_dct,(256,256)) 
        #cv2.imshow("dct", prueba)
        #cv2.waitKey(0)

        #img_dct[freq:64,0:64]=0
        #img_dct[0:64,freq:64]=0

        img_dct[32:64,0:64]=0
        img_dct[0:64,32:64]=0
        
        
        #prueba=cv2.resize(img_dct,(256,256)) 
        #cv2.imshow("filter", prueba)
        #cv2.waitKey(0)

        #img_dct2=img_dct[0:,0:freq]
        img_dct2=img_dct[0:32,0:32]

        
        y_final2=np.append(y_final2,img_dct2/100+0.5)

        #y_final2.shape=(idx,freq,freq)
        y_final2.shape=(idx,32,32)

        #cv2.imshow("orig", img)
        #cv2.waitKey(0)
    return y_final2

###########################################################################################################################

#__CLOUDBOOK:LOCAL__
def inicializaNonsharedVars():
    global x_train
    global y_train
    global x_test
    global y_test
    global total_images
    global countx
    global resolution
    global freq
    x_train = computex()
    #x_train = x_train[:countx,:resolution,:resolution,:1]
    x_train = x_train[:countx,:resolution,:resolution] 
    y_train = computey(x_train)
    #y_train = np.reshape(y_train,(countx,freq*freq,1)) #aplanamos la salida
    y_train = np.reshape(y_train,(countx,freq*freq)) #aplanamos la salida
    x_test=x_train
    y_test=y_train 
    total_images=(500//2)*2 
    #total_images=(total_images//batch_size)*batch_size # asi ajustamos a numero entero de batches
    print ("TOTAL images=", total_images)
    print ("train x shape:",x_train.shape)
    print ("train y shape:",y_train.shape)
    print ("non shared vars initialized")



    


#__CLOUDBOOK:NONSHARED__
x_train = np.zeros(0)
y_train = np.zeros(0)
x_test = np.zeros(0)
y_test = np.zeros(0)
batch_size = 2
freq=32
salida=int((freq*freq)/1)
resolution=64
total_images=500
countx=(total_images//batch_size)*batch_size
myshufle=True

#__CLOUDBOOK:DU0__
def main():
    if hasattr(main, 'executed'):
        return
    else:
        setattr(main, 'executed', True)
    print ("**************************************")
    print ("*     REGRESION NO LOCAL MLP         *")
    print ("*   programa de test                 *")
    print ("*                                    *")
    print ("**************************************")
    inicializaNonsharedVars()
    
main()    


#SKYNNET:BEGIN_REGRESSION_LOSS
inicializaNonsharedVars()
_DATA_TRAIN_X = x_train
_DATA_TRAIN_Y = y_train
_DATA_TEST_X = x_test
_DATA_TEST_Y = y_test
_FILTERS_1 = 64 #64
_FILTERS_2 = 64 #64
_NEURON_1 = 4096 #4096 # 4 *freq*freq
_NEURON_2 = 1024 # freq*freq
#_EPOCHS= 25
EPOCAS= 25
print ("train x shape:",x_train.shape)
print ("train y shape:",y_train.shape)
print ("_DATA_TRAIN_X shape:",_DATA_TRAIN_X.shape)
print ("_DATA_TRAIN_Y shape:",_DATA_TRAIN_Y.shape)

print("PARAM:")
print("  batch:",batch_size )
print("  images:",countx )
print("  shuffle:",myshufle )
print("  split: 0.2")
print("  epoc:", _EPOCHS)
print("  freq", freq)
print("  original output:", freq*freq)
print("  subnet output:", salida)
print("")

# red original
input_shape = (int(resolution), int(resolution), 1)
#input_shape = (int(resolution), int(resolution))
input_data = tf.keras.layers.Input(shape=input_shape)


x = tf.keras.layers.Conv2D(int(_FILTERS_1), (16, 16), activation='relu', padding='same')(input_data)
x = tf.keras.layers.MaxPooling2D((2,2 ), padding='same')(x)

x = tf.keras.layers.Conv2D(int(_FILTERS_2), (4, 4), activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D((2,2 ), padding='same')(x)

x=tf.keras.layers.Flatten()(x) 
x=tf.keras.layers.Dense(int(_NEURON_1),activation='relu')(x)
output=tf.keras.layers.Dense(int(_NEURON_2),activation='sigmoid')(x)
regresor = tf.keras.models.Model(input_data, output)

print(regresor.summary())


#optimizer = tf.keras.optimizers.Adam() #lr=lr)
#regresor.compile(loss='mse', optimizer=optimizer)
regresor.compile( optimizer='adam', loss='mse')

start=time.time()
regresor.fit(_DATA_TRAIN_X, _DATA_TRAIN_Y,
                epochs = EPOCAS,
                batch_size = batch_size,
                shuffle = myshufle,
                validation_split = 0.2,
                verbose=1)
end=time.time()
print (" tiempo de training transcurrido (segundos) =", (end-start))
print("----------------------")
predicted = regresor.predict(_DATA_TEST_X,verbose=1)


mse = tf.keras.losses.MeanSquaredError()
mse_value = mse(_DATA_TEST_Y, predicted).numpy()
print('============================================')
print('La loss es: ', mse_value)
print('============================================')

print("generando imagenes de ejemplo...")
print("--------------------------------")
generaImages(predicted)


#SKYNNET:END

