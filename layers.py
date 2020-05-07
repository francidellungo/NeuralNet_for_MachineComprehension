import tensorflow as tf
from tensorflow import keras
from preprocessing import create_glove_matrix


class MaxOverTimePoolLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MaxOverTimePoolLayer, self).__init__()

    # def build(self):
    # self.kernel = self.add_weight("kernel",
    #                               shape=[int(input_shape[-1]),
    #                                      self.num_outputs])
    def call(self, input_tensor):
        return tf.math.reduce_max(input_tensor, axis=1)


class CharacterEmbedding(tf.keras.layers.Layer):
    def __init__(self, conv_filters=100):
        super(CharacterEmbedding, self).__init__()
        # num_channels = input_.get_shape()[-1]
        self.emb = tf.keras.layers.Embedding(100, 10),  # TODO adjust dimensions (input, output)
        self.conv = tf.keras.layers.Conv1D(conv_filters, 5, activation='relu'),  # , input_shape=(None, char_emb_dim)),
        self.max_pool = MaxOverTimePoolLayer()

    def build(self, input_shape):
        super(CharacterEmbedding, self).build(input_shape)
        # TODO

    def call(self, input_):
        x = self.emb(input_)
        x = self.conv(x)
        x = self.max_pool(x)
        return x

