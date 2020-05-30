import tensorflow as tf
from tensorflow import keras
from preprocessing import create_glove_matrix
from preprocessing import read_data, create_char_dict
import numpy as np


class MaxOverTimePoolLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MaxOverTimePoolLayer, self).__init__()

    # def build(self):
    # self.kernel = self.add_weight("kernel",
    #                               shape=[int(input_shape[-1]),
    #                                      self.num_outputs])
    def call(self, input_tensor):
        return tf.math.reduce_max(input_tensor, axis=1)
        # return tf.math.reduce_max(input_tensor, axis=0)


class CharacterEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, conv_filters, filter_width=5, emb_dim=None):
        super(CharacterEmbedding, self).__init__()
        # num_channels = input_.get_shape()[-1]
        self.vocab_size = vocab_size + 1
        self.out_emb_dim = emb_dim if emb_dim is not None else (self.vocab_size - (self.vocab_size // 5))
        # print('vocab_size: ', self.vocab_size, ', out_emb dim: ', self.out_emb_dim)
        self.conv_filters = conv_filters
        self.filter_width = filter_width

        self.emb = tf.keras.layers.Embedding(self.vocab_size, self.out_emb_dim)
        self.conv = tf.keras.layers.Conv1D(self.conv_filters, self.filter_width, activation='relu',
                                           padding='valid')  # , input_shape=(None, char_emb_dim)),
        self.max_pool = MaxOverTimePoolLayer()

    # def build(self, input_shape):
    #     super(CharacterEmbedding, self).build(input_shape)

    def call(self, x):
        x = self.emb(x)
        x = self.conv(x)
        x = self.max_pool(x)
        return x


class HighwayLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(HighwayLayer, self).__init__()
        # self.units1, self.units2 = units_trans, units_gate
        #
        # self.trans = tf.keras.layers.Dense(self.units1, activation='relu')
        # self.gate = tf.keras.layers.Dense(self.units2, activation='sigmoid')

    def build(self, input_shape):
        self.Wh = self.add_weight(shape=(input_shape[1], input_shape[1]))
        self.Wt = self.add_weight(shape=(input_shape[1], input_shape[1]))
        self.bh = self.add_weight(shape=(input_shape[1],))
        self.bt = self.add_weight(shape=(input_shape[1],))

    def call(self, x):
    #     gate = self.gate(input_)
    #     trans = self.trans(input_)
    #     out = tf.add(tf.matmul(gate, trans), tf.matmul((1 - gate), input_))
    #     # out = gate * trans + (1 - gate) * input
    #     return out
    #
    # def highway_layer(self, x):
        H = tf.nn.relu(tf.matmul(x, self.Wh) + self.bh, name='activation')
        T = tf.sigmoid(tf.matmul(x, self.Wt) + self.bt, name='transform_gate')
        # C = tf.sub(1.0, T, name="carry_gate")

        y = tf.add(tf.multiply(H, T), tf.multiply(x, 1-T), name='y')  # y = (H * T) + (x * C)
        return y


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        pass

    def build(self, input_shape):
        #  dim W = (T * J, 3*dim)
        # print("input_shape (Attention Layer): ", input_shape)
        self.Ws = self.add_weight(shape=(input_shape[-1]*3, 1))  # opzione1
        # self.Ws = self.add_weight(shape=(H.shape[0] * U.shape[0], input_shape[1]*3))  # opzione2 TODO: U.shape, T.shape -> input_shape

    def computeSimilarity(self, H, U):
        """
        Create similarity matrix S between context (H) and query (U)
        :param H: context matrix (T x 2d)
        :param U: query matrix (J x 2d)
        :return: S: similarity matrix (T x J)
        """
        print("H shape: ", H.shape, "; U shape: ", U.shape, "; W shape: ", self.Ws.shape)
        duplicateH = tf.keras.backend.repeat_elements(H, rep=U.shape[-2], axis=0)
        # print("new H dim: ", duplicateH.shape)
        duplicateU = tf.tile(U, [H.shape[-2], 1])
        # print(" new U dim: ", duplicateU.shape)
        C = tf.concat([duplicateH, duplicateU, tf.multiply(duplicateH, duplicateU)], axis=-1)
        # print("C dim: ", C.shape)

        # opzione giusta: stesso vettore di pesi Ws (funzione \alpha) che moltiplica ogni riga della matrice creata
        S = tf.matmul(C, self.Ws)
        # print("S shape: ", S.shape)
        S = tf.reshape(S, [H.shape[-2], U.shape[-2]])
        # print("S reshaped: ", S.shape)
        return S

    def computeContext2QueryAttention(self, S, U):
        """
        Create C2Q attention matrix: which query words are most relevant to each context word.
        :param S: similarity matrix (T x J)
        :param U: query matrix (J x 2d)
        :return: AttendedQuery: C2Q matrix (Ũ) (T x 2d)
        """
        C2Q = tf.nn.softmax(S, axis=-1)  # attention weights on the query words
        # print("C2Q shape (T x J): ", C2Q.shape)
        AttendedQuery = tf.matmul(C2Q, U)
        # print("AttendedQuery shape (T x d): ", AttendedQuery.shape)
        return AttendedQuery

    def computeQuery2ContextAttention(self, S, H):
        """
        Create Q2C attention matrix: which context words have the closest similarity to one of the query words
        and are therefore critical for answering the query.
        :param S: similarity matrix (T x J)
        :param H: context matrix (T x 2d)
        :return: AttendedContext: Q2C matrix (H̃) (T x 2d)
        """
        Q2C = tf.nn.softmax(tf.reduce_max(S, axis=-1))
        # print("Q2C shape (T x 1): ", Q2C.shape)
        Q2C = tf.expand_dims(Q2C, -1)
        AttendedContext = tf.matmul(Q2C, H, transpose_a=True)
        # print("AttendedContext shape (1 x d): ", AttendedContext.shape)
        AttendedContext = tf.tile(AttendedContext, [H.shape[-2], 1])
        # print("AttendedContext shape (T x d): ", AttendedContext.shape)
        return AttendedContext

    def merge(self, H, AttendedQuery, AttendedContext):
        """
        Combine the information obtained by the C2Q and Q2C attentions.
        Each column vector of G can be considered as the query-aware representation of each context word.
        :param H: context matrix (T x 2d)
        :param AttendedQuery: C2Q matrix (Ũ) (T x 2d)
        :param AttendedContext: Q2C matrix (H̃) (T x 2d)
        :return: G: matrix (T x 8d)
        """
        G = tf.concat([H, AttendedQuery, tf.multiply(H, AttendedQuery), tf.multiply(H, AttendedContext)], axis=-1)
        # print(" G shape (T X 8d): ", G.shape)
        return G

    def call(self, H, U):

        # Similarity matrix (S) dimension: TxJ
        S = self.computeSimilarity(H, U)

        # C2Q attention
        C2Q = self.computeContext2QueryAttention(S, U)

        # Q2C attention
        Q2C = self.computeQuery2ContextAttention(S, H)

        # Merge C2Q (Ũ) and Q2C (H̃) to obtain G
        G = self.merge(H, C2Q, Q2C)

        return G


# source = "dataset/qa/web-example.json"
# evidence = "dataset/evidence"
# data = read_data(source, evidence)

# char_dict = create_char_dict()
# print("len(char_dict): ", len(char_dict))
# model = tf.keras.Sequential()
# # model.add(CharacterEmbedding(7, 5, emb_dim=4))
# model.add(tf.keras.layers.Embedding(7, 4, input_shape=(10,)))
# model.add(tf.keras.layers.Conv1D(100, 5, activation='relu',
#                                            padding='valid'))
# # model.add(MaxOverTimePoolLayer())
# # # model.add(tf.keras.layers.Embedding(7, 5))
#
# vec = np.array([[1, 2, 3, 0, 5, 5, 3,6,6,6], [2, 3, 4, 5, 5, 5, 3,7,8,9]])
# y = [1, 0]
# # print("vec: ", vec, len(vec), len(vec[0]))
# # model(vec)
# # # model.call(vec)
# model.build(vec.shape)
# model.summary()

# model = tf.keras.Sequential()
# model.add(CharacterEmbedding(7, 30))
# model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
# model.fit(vec, y)