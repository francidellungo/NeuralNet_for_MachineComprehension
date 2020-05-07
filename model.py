import tensorflow as tf
from tensorflow import keras
from layers import CharacterEmbedding
from tensorflow.keras import models, layers
from preprocessing import create_glove_matrix
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


class BiDafModel(tf.keras.Model):
    def __init__(self, conv_filters):
        super(BiDafModel, self).__init__()
        self.conv_filters = conv_filters
        self.char_emb = CharacterEmbedding(self.conv_filters)

    def build(self, input_shape):
        super(BiDafModel, self).build(input_shape)

    def call(self, input_):
        x = self.char_emb(input_)
        return x


# x = tf.ones((10, 10))
# # print(x)
# model = BiDafModel(100)

# print(model.char_emb.trainable_weights)
# model.build(x.shape)
# y = model(x)
# model.summary()


# def build_model(conv_filters=100):
#     model = keras.Sequential([
#         # 1 Character Embedding Layer
#         # tf.keras.layers.InputLayer([10]),
#         CharacterEmbedding()
#     ])
#
#     return model


# n_words = 10
# model = models.Sequential()
# model.add(layers.Conv2D(100, (n_words, 1), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
# model.summary()


# model = build_model()
# model.build()
# model.summary()

