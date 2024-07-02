
from typing import List
import time
import sys # para coger parametros de entrada
import cv2 # opencv para el reescalado y crop de imagenes
import math
import matplotlib.pyplot as plt
import tensorflow as tf, numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
import pandas as pd

mezclas = 0
#######################################################################################
#__CLOUDBOOK:LOCAL__
def adaptImages(x_train,samples,data_type,a, h2,w2, channels2):
    j=0
    for i in range(0,samples):
        h,w, channels=x_train[i].shape
        #print("h=",h, " w=",w, "channels=", channels)
        img=x_train[i]
        #print ("dtype=",img.dtype)
        # dos estrategias: cortar o meter franjas negras para llegar a cuadrado
        # si cortamos aprende pero no aprende bien porque muchas imagenes se corrompen
        estrategia=2 # 1 =crop, 2 = franjas, 3 =escalado con deformacion

        if (estrategia==1): 
            # crop
            #---------------------
            #img=cv2.resize(img, dsize=(w,h))
            # diferenciamos segun relacion de aspecto w>=h o w<h
            if (w>=h):
                # cortamos el lado mayor
                margen=int((w-h)/2)
                img2=img[0:h, margen:margen+h]
            
            else:
                margen=int((h-w)/2)
                img2=img[margen:margen+w, 0:w]
                
        elif (estrategia==2): 
            # aumentar con franjas negras
            #------------------------
            if (w>=h):#imagen horizontal w>h
                img2= np.zeros(w*w*channels,dtype=data_type)
                #img2[img2==0]=128 #probamos 128
                img2.shape=(w,w,channels)
                ini=int((w-h)/2)
                fin=int(h+ini)
                
                #print ("ini, fin, dif =", ini, fin, (fin-ini))
                #img3=img2[ini:fin,0:w,0:3];
                #print ("shape ",x_train[i].shape, img.shape, img2.shape)
                img2[ini:fin,0:w,0:3]=img[0:h, 0:w,0:3]
                #img2 = np.float32(img2)

                #franjas de repeticion
                #img2[0:ini,0:w,0:3]=img[0:1, 0:w,0:3]
                #img2[fin:w,0:w,0:3]=img[h-1:h, 0:w,0:3]
                
            else: # imagen vertical h>w
                img2= np.zeros(h*h*channels,dtype=data_type)
                #img2[img2==0]=128 #probamos 128
                img2.shape=(h,h,channels)
                
                ini=int((h-w)/2)
                fin=int(w+ini)
                
                img2[0:h,ini:fin,0:3]=img[0:h, 0:w,0:3]
                #img2 = np.float32(img2)

                #franjas de repeticion
                #img2[0:h,0:ini,0:3]=img[0:h, 0:1,0:3]
                #img2[0:h,fin:h,0:3]=img[0:h, w-1:w,0:3]
                
        elif (estrategia==3):
            # escalado con deformacion
            #------------------------
            maxl=max(w,h)
            minl=min(w,h)
            if (maxl/minl>4/3):
                continue # cribamos
            img2=img
             
        # creamos la imagen nueva reescalada
        if (w!=w2 or h!=h2): # optimizacion
            img2=cv2.resize(img2, dsize=(w2,h2), interpolation=cv2.INTER_LINEAR) # INTER_NEAREST INTER_LINEAR, INTER_CUBIC , INTER_LANCZOS4
        
        
        #la guardamos
        
        if (channels2==1):
            #print("cambio shape")
            if (channels==3):
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # convertimos a bn
            #b,g,r = cv2.split(img2)
            #img2=b #.shape=(h2,w2,1)
            img2.shape=(h2,w2,1)
            #printf("b shape =",b.shape)
            
        
        #print(f" item {i} xtrain shape: {x_train[i].shape}")
        a[j]=img2 # como  A es float, se copia desde uint8 a float. Es decir, a[i] es float
        j=j+1
        #print(f" item {i} a shape: {a[i].shape}")
        # las mostramos
        if (i<0):
            img_orig = img #cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # las magenes llegan en BGR y para hacer show necesitamos RGB
            cv2.imshow('orig', img_orig)
            print("img2.shape=", img2.shape)
            img3 = img2 #cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) # las magenes llegan en BGR y para hacer show necesitamos RGB
            cv2.imshow('Image', img3)
            #img3 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
            #cv2.imshow('Image', img3)
            
            cv2.waitKey(0);
    
    return j




##################################################################################
class Patches(layers.Layer):
    """Create a a set of image patches from input. The patches all have
    a size of patch_size * patch_size.
    """

    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    """The `PatchEncoder` layer will linearly transform a patch by projecting it into a
    vector of size `projection_dim`. In addition, it adds a learnable position
    embedding to the projected vector.
    """
    
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
############################################################################################
#__CLOUDBOOK:LOCAL__
def get_data_augmentation_layer(image_size: int, normalization: bool=True) -> keras.layers.Layer:
    list_layers = []
    
    if normalization:
        list_layers.append(layers.Normalization()) # TODO: Implement adapt in `create_vit_classifier`
    

    
    data_augmentation = keras.Sequential(
        list_layers,
        name="data_augmentation",
    )
    return data_augmentation

#__CLOUDBOOK:LOCAL__
def loadDataset(tipo, ds_name, directorio):
    # tipo :
    #   "local"  si es un directorio con subdirectorios de imagenes por ejemplo si directorio == "./disney/
    #        ./disney/mickey/ contiene imagenes de mickey ( es decir, en todos ellos Y_train será el nombre del subdirectorio "mickey" )
    #        ./disney/donald/ contiene imagenes de donald ( es decir, en todos ellos Y_train será "donald")
    #        ./disney/pluto/ contiene imagenes de pluto ( es decir, en todos ellos Y_train será "pluto")
    #
    #   "standar" si es un dataset de los que se pueden cargar con el modulo tdfs de tensorflow
    #
    #  esta funcion retorna x_train, y_train+
    global mezclas
    ds_dir=ds_name #directorio de descarga
    lote_size=1 
    if (tipo=="standar"):
        ds_train,  info= tfds.load(ds_name,
                     data_dir=ds_dir, #directorio de descarga
                     #en el campo splits del datasetinfo vemos los splits que contiene y aqui cogemos uno
                     # si usamos split, ya no retorna info y hay que sacarla
                     #split=['train','test'],
                     #split=["train[:10%]","test[:10%]"],
                     #split=["train[:10%]","train[:5%]"],
                     #split=["train","test"], #para caltech101
                     split="all", #para caltech101. es train + test +...
                     #split="train[:99%]", #imagenet
                     #split="train[:10%]", #prueba inicial
                     #supervised: retorna una estructura con dos tuplas input, label according to builder.info.supervised_keys
                     # si false, el dataset "tendra un diccionario con todas las features"
                     as_supervised=True,
                     shuffle_files=False, #True, # desordenar. lo quito para que funcione siempre igual y no dependa de ejecucion
                     #  if "batch_size" isset, add a batch dimension to examples
                     #batch_size=lote_size, #por ejemplo en lugar de (28,28,3) sera (10,28,28,3)
                     with_info=True # descarga info (tfds.core.DatasetInfo)
                     )
        # Extract informative features
        print("informative features")
        print("--------------------")
        class_names = info.features["label"].names
        n_classes = info.features["label"].num_classes
        print("  class names:", class_names) 
        print("  num clases:", n_classes)
        print("")

        #tamanos de datasets
        print("datasets sizes. IF batch_size is used THEN this is the number of batches")
        print("-------------------------------------------------------------------------")
        print("  Train set size: ", len(ds_train), " batches of ",lote_size, " elements") # Train set size
        print()
        
        #contenido dataset
        print("ds_train contents (", ds_name,")")
        print("--------------------------------")
        print(ds_train)
        print()
        print("dataset to numpy conversion (", ds_name,")")
        print("------------------------------------------")
        ds_train_npy=tfds.as_numpy(ds_train)

        print("  convirtiendo DS en arrays numpy...(", ds_name,")")
        x_train = np.array(list(map(lambda x: x[0], ds_train_npy)))
        y_train = np.array(list(map(lambda x: x[1], ds_train_npy)))


        print("  conversion OK")
        print("   x_train numpy shape:",x_train.shape)
        print("   y_train numpy shape:",y_train.shape)
        print("")

        print()

        return n_classes, x_train, y_train
    
    else : # tipo local
        print(" dataset local")
        #resolution=1000 # resolucion igual para todos pero en numero total de pixeles (al cuadrado)
        x_train=[] #np.zeros(0,dtype=np.uint8)
        y_train=[]#np.zeros(0,dtype=np.float32)
        carpetas = sorted(os.listdir(directorio))
        idx=0
        n_classes=0
        for subdir in carpetas:
            n_classes=n_classes+1
            categoria=n_classes-1
            imagenes = os.listdir(directorio+"/"+subdir)
            for img_name in imagenes:
                idx=idx+1
                name=directorio+"/"+subdir+"/"+img_name
                img=cv2.imread(name)
                h_orig,w_orig, channels_orig=img.shape
               
                #print("shape orig=",img.shape)
                #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #factor=math.sqrt((resolution*resolution)/(h_orig*w_orig))
                #h_fin=int(h_orig*factor)
                #w_fin=int(w_orig*factor)
                #img=cv2.resize(img,(h_fin,w_fin)) # reduce a resolicion fija para todas las imagenes

                #img=np.float32(img)# ahora ya es un decimal
                #img=img/255.0 # ahora ya entre 0 y 1

                #x_train=np.append(x_train,img)
                #y_train=np.append(y_train,categoria)

                x_train.append(img)
                y_train.append(categoria)

                
                #cv2.imshow("orig", img)
                #cv2.waitKey(0)
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        ind = np.arange(len(x_train)) #ABM, to shuffle
        np.random.shuffle(ind)
        x_train = x_train[ind]
        y_train = y_train[ind]
        print("x_train.shape", x_train.shape)
        print("number of classes", n_classes)
        mezclas += 1
        #x_train.shape=(idx,resolution,resolution,3)

        print("", end="\n")
        return n_classes, x_train, y_train
        
##################################################################################
#__CLOUDBOOK:NONSHARED__
x_train = None
y_train = None
w2=1368
h2=1368 
channels2 = None

#__CLOUDBOOK:LOCAL__
def load_data():
    global x_train
    global y_train
    global channels2
    global h2
    global w2
    loaded_dataset = loadDataset("local", "cell", "./TRAIN_dataset_cells/")
    
    x_train = loaded_dataset[1]
    y_train = loaded_dataset[2]

    w2=320
    h2=320

    channels=x_train[0].shape[2] # todos los elementos tienen mismo num canales
    samples=x_train.shape[0]

    data_type= x_train[0].dtype
    channels2=channels # es por si queremos pasar a bn, basta con poner channels2=1

    a = np.zeros(h2*w2*channels2*samples, dtype=np.float32) #data_type)
    #a.shape=(samples,h2,w2,channels2)
    a = a.reshape(samples, h2, w2, channels2)


    valid_samples=adaptImages(x_train,samples,data_type, a, h2,w2, channels2)
    a=a / 255.0
    x_train = a

def main():
    if hasattr(main,'executed'):
        return
    else:
        setattr(main,'executed',True)
    load_data()

main()

#SKYNNET:BEGIN_MULTICLASS_ACC_LOSS
if x_train is None:
    load_data()
_DATA_TRAIN_X = x_train
_DATA_TRAIN_Y = y_train
_DATA_TEST_X = x_train[-2000:]
_DATA_TEST_Y = y_train[-2000:]




_NEURON_2 = 4 #n_classes 
_EMBEDDING_ = 32 #dimensión del espacio latente
epocas = 150

batch_size= 64 #8 #32# 32#64 para disney, usar 8 porque hay pocas imagenes. para caltech usar 32
input_shape=[h2, w2, channels2] # alto final, ancho final, canales.
image_size=w2 # lado del tamaño final de las imagenes
embeddings_dim=32#_EMBEDDING_ # bytes de cada vector
transformer_layers=4 # 3 encoders
num_heads=4 
transformer_units =[128, embeddings_dim] # Size of the transformer layers
mlp_head_units=[256,128]#[128,128]  # Size of the dense layers of the final classifier. 
num_epocas=epocas #10
my_split=0.2 # un 20% del conjunto de train lo usamos para validar
lr=1e-3#2e-3 #1e-2 #1e-4 # learning rate  (default de keras es 0.001 es decir 1e-3)
num_patches_lado=6#2 # los patches que caben por lado
num_patches=num_patches_lado*num_patches_lado # numero total de patches
patch_size=w2//num_patches_lado # lado de un patch
if (patch_size!=w2/num_patches_lado):
    print(" imagen no puede dividirse en numero entero de patches")
    exit()
n_classes = _NEURON_2


def mlp(x: tf.Tensor, hidden_units: List[int], dropout_rate: float) -> tf.Tensor:
    """Multi-Layer Perceptron

    Args:
        x (tf.Tensor): Input
        hidden_units (List[int])
        dropout_rate (float)

    Returns:
        tf.Tensor: Output
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def create_vit_classifier(input_shape,
                          num_classes: int,
                          image_size: int,
                          patch_size: int,
                          num_patches: int,
                          projection_dim: int,
                          dropout: float,
                          n_transformer_layers: int,
                          num_heads: int,
                          transformer_units: List[int],
                          mlp_head_units: List[int],
                          normalization: bool=False):
    
    inputs = layers.Input(shape=input_shape)
    print ("num patches= ", num_patches)
    # Augment data.
    data_augmentation = get_data_augmentation_layer(image_size=image_size, normalization=normalization)
    augmented = data_augmentation(inputs)
    
    # Create patches.
    patches = Patches(patch_size)(augmented)
    
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(n_transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout)(representation)
    
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout)
    
    # Classify outputs.
    logits = layers.Dense(_NEURON_2, activation='softmax')(features)
    
    # Create the Keras model.
    vit_classifier = keras.Model(inputs=inputs, outputs=logits)

    
    return vit_classifier


print('ABM: n_classes=', n_classes)
vit_classifier = create_vit_classifier(input_shape=input_shape, 
                                       num_classes=n_classes, 
                                       image_size=image_size, 
                                       patch_size=patch_size,
                                       num_patches=num_patches, 
                                       projection_dim=embeddings_dim, 
                                       dropout=0.2,
                                       n_transformer_layers=transformer_layers, 
                                       num_heads=num_heads,
                                       transformer_units=transformer_units,
                                       mlp_head_units=mlp_head_units, 
                                       )

    
   
#print(vit_classifier.summary())

#history = run_experiment(vit_classifier, _DATA_TRAIN_X, _DATA_TRAIN_Y,num_epocas,my_split, batch_size,lr)
optimizer = tf.optimizers.Adam(learning_rate=lr)
vit_classifier.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
)
print(f"Parametros del modelo: {vit_classifier.count_params()}")
start = time.time()
# --- TRAINING ---
history = vit_classifier.fit(
    _DATA_TRAIN_X,
    _DATA_TRAIN_Y,
    epochs=num_epocas,
    batch_size=batch_size,
    shuffle=True,
    validation_split=my_split,
)
end = time.time()
print(' tiempo de training transcurrido (segundos) =', end - start)


#filename="learning_curve_VIT_"+ds_name+".png"
#plot_learning_curve(history=history, filepath="figs/learning_curve_mnist-VIT.png")
#plot_learning_curve(history=history, filepath=filename)

titulo="learning_curve_VIT"
plt.title(titulo)
plt.plot(history.history['accuracy'], 'b-')
plt.plot(history.history['val_accuracy'], 'g-')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('Epoch')
plt.ylabel('Percent')
#plt.show();#ABM
plt.savefig('learning_curves.png', dpi=200) #ABM

prediction = vit_classifier.predict(_DATA_TEST_X)
for i in [0,1,2,3]:
#    print(np.where(_DATA_TEST_Y==i))
    print('class ', i, ', num_elements=', len(np.where(_DATA_TEST_Y==i)[0]), '  (',len(np.where(_DATA_TEST_Y==i)[0])/len(_DATA_TEST_Y)*100., '%)' )
#print('prediction',prediction)
prediction = [np.argmax(p) for p in prediction]

print('acc=',sum(prediction==_DATA_TEST_Y)/len(prediction))
print(f"Mezclas de datos: {mezclas}")
#df = pd.DataFrame( {'accuracy':history.history['accuracy'],'val_accuracy':history.history['val_accuracy'], 'loss':history.history['loss'],'val_loss':history.history['val_loss']} )
#df.index.rename('epoch', inplace=True)
#df.to_csv('learning_curves.csv')
#print('Al final ', _DATA_TEST_X.shape)
#print('input/output shape: ', _DATA_TEST_X.shape, _DATA_TEST_Y.shape, set(_DATA_TEST_Y) )



#SKYNNET:END

