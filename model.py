import tensorflow as tf
from tensorflow import keras
from layers import CharacterEmbedding, HighwayLayer, AttentionLayer
from preprocessing import read_data
# from tensorflow.keras import models, layers
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


class BiDafModel(tf.keras.Model):
    def __init__(self, char_vocab_size, conv_filters=100, conv_filter_width=5, emb_dim=None):
        super(BiDafModel, self).__init__()
        self.emb_dim = char_vocab_size
        self.conv_filters = conv_filters  # 100
        self.lstm_units = conv_filters * 2  # 200

        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        self.char_emb = CharacterEmbedding(self.emb_dim, self.conv_filters, filter_width=conv_filter_width, emb_dim=emb_dim)
        self.highway1 = HighwayLayer()
        self.highway2 = HighwayLayer()

        self.BiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True), merge_mode='concat')

        self.Attention = AttentionLayer()

    def train(self, c_words, c_chars, q_words, q_chars, y, epochs, batch_size):
        print("train ")
        self.train_step(c_words[:10], c_chars[:10], q_words[:10], q_chars[:10], y[:10])

        # self.train_step(X[:10], y[:10])

        # num_batches = int(np.ceil(X.shape[0] / batch_size))
        # num_batches = ...
        # for epoch in range(epochs):
            # shuffle X, y
            # for batch in range(num_batches)
                # train_step

        #     loss, y_pred = self.train_step(X, y)
        #     tqdm.write("epoch: {}/{}, loss: {:.4f}, accuracy: {:.4f}".format(epoch, epochs, loss,
        #                                                                      float(ChebNet.accuracy_mask(y, y_pred))))

    def train_step(self, c_words, c_chars, q_words, q_chars, y):
        print("train step")
        with tf.GradientTape() as tape:
            y_pred = self.call(c_words, c_chars, q_words, q_chars)  # forward pass
            loss = self.loss(y, y_pred)

        #     loss = self.non_zero_loss(y, y_pred)  # calculate loss
        grads = tape.gradient(loss, self.trainable_weights)  # backpropagation
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))  # optimizer step
        return loss, y_pred

    def loss(self):
        # ...
        pass

    def build(self, input_shape):
        pass

    # def call(self, input):
    def call(self, c_word_input, c_char_input, q_word_input, q_char_input):
        # first c stands for context second for char, q stands for query
        # TODO use tf.split()
        # query, context = input1, input2
        # c_char_input = tf.convert_to_tensor(c_char_input)

        for i in range(len(c_word_input)):
            cc = self.char_emb(c_char_input[i])
            qc = self.char_emb(q_char_input[i])

            # concatenate word representation vector given by chars and word representation vector by GloVe
            context0 = tf.concat([cc, c_word_input[i]], axis=-1)
            query0 = tf.concat([qc, q_word_input[i]], axis=-1)

            # highway layers
            context = self.highway2(self.highway1(context0))
            query = self.highway2(self.highway1(query0))

            # Contextual Embedding Layer (bidirectional LSTM)
            H = self.BiLSTM(tf.expand_dims(context, 0))
            U = self.BiLSTM(tf.expand_dims(query, 0))
            # context matrix (H) dimension: 2d x T, query matrix (U) dimension: 2d x J

            H = tf.squeeze(H, [0])  # Removes dimensions of size 1 from the shape of a tensor. (in position 0)
            U = tf.squeeze(U, [0])

            # Attention Layer -> G matrix (T x 8d)
            G = self.Attention(H, U)

        return


# tf.keras.layers.Attention()

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

# source = "dataset/qa/web-example.json"
# evidence = "dataset/evidence"
# data = read_data(source, evidence)
#
# model = BiDafModel(70, 100, 50, 50)
# model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
#
# model.fit(data)
# model.build(10)
# model.summary()
