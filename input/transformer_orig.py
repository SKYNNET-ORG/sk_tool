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

_DATA_TRAIN_X = x_train
_DATA_TRAIN_Y = y_train
_DATA_VAL_X = x_val
_DATA_VAL_Y = y_val
_DATA_TEST_X = x_train
_DATA_TEST_Y = y_train

_EMBEDDING_ = 32
_NEURON_1 = 20
_NEURON_2 = 2
_EPOCHS = 2
_BATCH = 32

# esta red tiene un solo bloque transformer. es "minimalista"
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, _EMBEDDING_)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(_EMBEDDING_, 2, 32)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(_NEURON_1, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(_NEURON_2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
print("-----ESTRUCTURA RED ORIGINAL------")
print(model.summary())
print("----------------------------------")


print ("Entrenamos sin someter a validacion pues eso lo haremos despues")

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

start=time.time()

print ("training orig model...")

model.fit(
    _DATA_TRAIN_X, _DATA_TRAIN_Y, batch_size=32, epochs=_EPOCHS, validation_data=(_DATA_VAL_X, _DATA_VAL_Y)
)


end=time.time()
print (" original: tiempo transcurrido (segundos) =", (end-start))
predicted = model.predict(_DATA_TEST_X)
#SKYNNET:END