import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import numpy as np

#Implement a Transformer block as a layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

#Implement embedding layer
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# Download and prepare dataset
vocab_size = 20000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

#barajamos datos por si vienen ordenados
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]
x_test = x_train
y_test = y_train

#SKYNNET:BEGIN_BINARYCLASS_ACC_LOSS

#__CLOUDBOOK:LOCAL__
def dividir_array_categorias(array, n, m):
    #n = categorias iniciales
    #m = numero de arrays resultantes
    # Obtener las categorias unicas del array original
    categorias_unicas = np.unique(array)
    
    if n < m:
        raise ValueError(f"El numero de categorias original {n} debe ser mayor o igual al numero de arrays de destino {m}.")
    
    if m > len(categorias_unicas):
        raise ValueError(f"El numero de categorias unicas {len(categorias_unicas)} no es suficiente para dividirlas en los {m} arrays deseados.")
    
    # Calcular el numero de categorias en cada array de destino
    categorias_por_array = n // m    
    # Crear los m arrays de destino
    arrays_destino = []
    inicio_categoria = 0
    
    for i in range(m):
        fin_categoria = inicio_categoria + categorias_por_array
        
        if i < n % m:#
            #print(f"		Como {i} < n({n}) % m({m}), hacemos fin_categoria = {fin_categoria+1}")
            fin_categoria += 1
        
        categorias_array_actual = categorias_unicas[inicio_categoria:fin_categoria]
        arrays_destino.append(categorias_array_actual)
        inicio_categoria = fin_categoria
    return arrays_destino

#__CLOUDBOOK:LOCAL__
def combinar_arrays(arrays):
    from itertools import combinations
    if len(arrays) < 2:
        raise ValueError("Se requieren al menos dos arrays para realizar la combinacion.")
    
    combinaciones = list(combinations(arrays, 2))
    
    arrays_combinados = []
    
    for combo in combinaciones:
        array_1, array_2 = combo
        
        # Concatenar los dos arrays en uno solo
        array_combinado = np.concatenate((array_1, array_2))
        
        arrays_combinados.append(array_combinado)
    
    return arrays_combinados

#__CLOUDBOOK:GLOBAL__
predictions_0 = {}
#__CLOUDBOOK:NONSHARED__
model = [None, None]
to_predict_models = []
#__CLOUDBOOK:PARALLEL__
def skynnet_train_0(sk_i):
    global model
    global to_predict_models
    _DATA_TRAIN_X = x_train
    _DATA_TRAIN_Y = y_train
    _DATA_VAL_X = x_val
    _DATA_VAL_Y = y_val
    _DATA_TEST_X = x_train
    _DATA_TEST_Y = y_train
    _EMBEDDING_ = 16
    _NEURON_1 = 10
    _NEURON_2 = 1
    _EPOCHS = 1
    _BATCH = 32
    num_heads = 2
    ff_dim = 32
    datos_train_x_1 = _DATA_TRAIN_X[:len(_DATA_TRAIN_X) // 2]
    datos_train_x_2 = _DATA_TRAIN_X[len(_DATA_TRAIN_X) // 2:]
    datos_train_y_1 = _DATA_TRAIN_Y[:len(_DATA_TRAIN_Y) // 2]
    datos_train_y_2 = _DATA_TRAIN_Y[len(_DATA_TRAIN_Y) // 2:]
    if sk_i == 1:
        _DATA_TRAIN_X = datos_train_x_1
        _DATA_TRAIN_Y = datos_train_y_1
    else:
        _DATA_TRAIN_X = datos_train_x_2
        _DATA_TRAIN_Y = datos_train_y_2
    _NEURON_2 = 2
    datos_validate_x_1 = _DATA_VAL_X[:len(_DATA_VAL_X) // 2]
    datos_validate_x_2 = _DATA_VAL_X[len(_DATA_VAL_X) // 2:]
    datos_validate_y_1 = _DATA_VAL_Y[:len(_DATA_VAL_Y) // 2]
    datos_validate_y_2 = _DATA_VAL_Y[len(_DATA_VAL_Y) // 2:]
    if sk_i == 1:
        _DATA_VAL_X = datos_validate_x_1
        _DATA_VAL_Y = datos_validate_y_1
    else:
        _DATA_VAL_X = datos_validate_x_2
        _DATA_VAL_Y = datos_validate_y_2
    _NEURON_2 = 2
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, _EMBEDDING_)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(_EMBEDDING_, 2, 32)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(_NEURON_1, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(_NEURON_2, activation='softmax')(x)
    model[sk_i] = keras.Model(inputs=inputs, outputs=outputs)
    print('-----ESTRUCTURA RED ORIGINAL------')
    print(model[sk_i].summary())
    print('----------------------------------')
    print('Entrenamos sin someter a validacion pues eso lo haremos despues')
    model[sk_i].compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    start = time.time()
    print('training orig model...')
    model[sk_i].fit(_DATA_TRAIN_X, _DATA_TRAIN_Y, batch_size=32, epochs=_EPOCHS, validation_data=(_DATA_VAL_X, _DATA_VAL_Y))
    end = time.time()
    print(' original: tiempo transcurrido (segundos) =', end - start)
    to_predict_models.append(sk_i)
#__CLOUDBOOK:PARALLEL__
def skynnet_prediction_0():
    global predictions_0
    global to_predict_models
    global model
    _DATA_TEST_X = x_train
    _DATA_TEST_Y = y_train
    #__CLOUDBOOK:BEGINREMOVE__
    __CLOUDBOOK__ = {}
    __CLOUDBOOK__['agent'] = {}
    __CLOUDBOOK__['agent']['id'] = 'agente_skynnet'
    #__CLOUDBOOK:ENDREMOVE__
    #__CLOUDBOOK:LOCK__
    for sk_i in to_predict_models[:]:
        to_predict_models.remove(sk_i)
        label = __CLOUDBOOK__['agent']['id'] + str(sk_i)
        predicted = model[sk_i].predict(_DATA_TEST_X, verbose=1)
        categorias = [0, 1]
        resul = []
        for (i, pred) in enumerate(predicted):
            array_final = np.ones(2)
            array_final[categorias] = pred
            resul.append(array_final.tolist())
        predictions_0[label] = resul
    #__CLOUDBOOK:UNLOCK__


#SKYNNET:END

#__CLOUDBOOK:DU0__
def skynnet_train_global_0():
    for i in range(2):
        skynnet_train_0(i)
    #__CLOUDBOOK:SYNC__
#__CLOUDBOOK:DU0__
def skynnet_prediction_global_0():
    _DATA_TEST_X = x_train
    _DATA_TEST_Y = y_train
    for i in range(2):
        skynnet_prediction_0()
    #__CLOUDBOOK:SYNC__
    global predictions_0
    precision_compuesta = []
    valores = np.array(list(predictions_0.values()))
    predicted = np.prod(valores, axis=0)
    correctas = 0
    total = 0
    for i in range(len(y_test)):
        if y_test[i] == np.argmax(predicted[i]):
            correctas += 1
        total += 1
    precision_compuesta.append(correctas / total)
    print('============================================')
    print('Skynnet Info: La accuracy de la prediccion compuesta es: ', precision_compuesta)
    print('============================================')
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    scce_orig = scce(y_test, predicted).numpy()
    print('============================================')
    print('Skynnet Info: La loss compuesta es: ', scce_orig)
    print('============================================')


#__CLOUDBOOK:MAIN__
def main():
    skynnet_train_global_0()
    skynnet_prediction_global_0()

if __name__ == '__main__':
    main()

