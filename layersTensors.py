import tensorflow as tf


class MaxOverTimePoolLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(MaxOverTimePoolLayer, self).__init__()

    def call(self, input_tensor):
        return tf.math.reduce_max(input_tensor, axis=2)
        # return tf.math.reduce_max(input_tensor, axis=0)


class CharacterEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, conv_filters, filter_width=5, emb_dim=8):
        super(CharacterEmbedding, self).__init__()
        # num_channels = input_.get_shape()[-1]
        self.vocab_size = vocab_size + 1
        # self.out_emb_dim = emb_dim if emb_dim is not None else (self.vocab_size - (self.vocab_size // 2))  # default 35
        self.out_emb_dim = emb_dim

        self.conv_filters = conv_filters
        self.filter_width = filter_width

        self.dropout_conv = tf.keras.layers.Dropout(.2)

        self.emb = tf.keras.layers.Embedding(self.vocab_size, self.out_emb_dim)
        self.conv = tf.keras.layers.Conv1D(self.conv_filters, self.filter_width, activation='relu',
                                           padding='valid')  # , input_shape=(None, char_emb_dim)),
        self.max_pool = MaxOverTimePoolLayer()

    def call(self, x, training=False):
        # x = tf.keras.layers.Flatten()(x)
        x = self.emb(x)  # Embedding
        x = self.dropout_conv(x, training=training)  # Dropout
        x = tf.stack([self.conv(xi) for xi in x])  # 1D convolutions
        x = self.max_pool(x)  # Max Pooling over time
        return x


class HighwayLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(HighwayLayer, self).__init__()
        # self.units1, self.units2 = units_trans, units_gate
        #
        # self.trans = tf.keras.layers.Dense(self.units1, activation='relu')
        # self.gate = tf.keras.layers.Dense(self.units2, activation='sigmoid')

    def build(self, input_shape):
        self.Wh = self.add_weight(shape=(input_shape[-1], input_shape[-1]), trainable=True)
        self.Wt = self.add_weight(shape=(input_shape[-1], input_shape[-1]), trainable=True)
        self.bh = self.add_weight(shape=(input_shape[-1],), trainable=True)
        self.bt = self.add_weight(shape=(input_shape[-1],), trainable=True)
        super(HighwayLayer, self).build(input_shape)

    def call(self, x):
        n, m = x.shape[-2], x.shape[-1]

        x = tf.reshape(x, [-1, m])  # x.shape [b, n, m] --> [b*n, m]
        H = tf.nn.relu(tf.matmul(x, self.Wh) + self.bh,
                       name='activation')  # x.shape [b*n, m] , Wh.shape [m, m] --> matmul: [b*n , m]
        T = tf.sigmoid(tf.matmul(x, self.Wt) + self.bt, name='transform_gate')
        # C = tf.sub(1.0, T, name="carry_gate")
        H = tf.reshape(H, [-1, n, m])  # [b*n, m] --> [b, n, m]
        T = tf.reshape(T, [-1, n, m])
        x = tf.reshape(x, [-1, n, m])
        y = tf.add(tf.multiply(H, T), tf.multiply(x, 1 - T), name='y')  # y = (H * T) + (x * C)
        return y


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        pass

    def build(self, input_shape):
        #  dim W = (T * J, 3*dim)
        # print("input_shape (Attention Layer): ", input_shape)
        self.Ws = self.add_weight(shape=(input_shape[-1] * 3, 1), trainable=True)
        # self.Ws = self.add_weight(shape=(H.shape[0] * U.shape[0], input_shape[1]*3))  # opzione2 TODO: U.shape, T.shape -> input_shape
        super(AttentionLayer, self).build(input_shape)

    def computeSimilarity(self, H, U):
        """
        Create similarity matrix S between context (H) and query (U)
        :param H: context matrix (T x 2d)
        :param U: query matrix (J x 2d)
        :return: S: similarity matrix (T x J)
        """
        # print(" ---------> H shape: ", H.shape, "; U shape: ", U.shape, "; W shape: ", self.Ws.shape)
        duplicateH = tf.keras.backend.repeat_elements(H, rep=U.shape[-2], axis=1)  # repeats each rows J times
        # print("new H dim: ", duplicateH.shape)
        duplicateU = tf.tile(U, [1, H.shape[-2], 1])  # repeats matrix U T times
        # print(" new U dim: ", duplicateU.shape)
        C = tf.concat([duplicateH, duplicateU, tf.multiply(duplicateH, duplicateU)], axis=-1)
        # print("C dim: ", C.shape)

        # opzione giusta: stesso vettore di pesi Ws (funzione \alpha) che moltiplica ogni riga della matrice creata
        S = tf.matmul(C, self.Ws)
        # print("S shape: ", S.shape)
        S = tf.reshape(S, [H.shape[0], H.shape[-2], U.shape[-2]])
        # print("S reshaped: ", S.shape)
        return S

    def computeContext2QueryAttention(self, S, U, mask=None):
        """
        Create C2Q attention matrix: which query words are most relevant to each context word.
        :param S: similarity matrix (T x J)
        :param U: query matrix (J x 2d)
        :return: attended_query: C2Q matrix (Ũ) (T x 2d)
        """
        C2Q = tf.nn.softmax(S, axis=-1)  # attention weights on the query words
        # print("C2Q shape (T x J): ", C2Q.shape)

        #  old one : attended_query = tf.matmul(C2Q, U)

        # print("attended_query shape (T x d): ", attended_query.shape)
        U = tf.transpose(U, perm=[0, 2, 1])  # change U dims: from J x 2d to 2d x J

        """ alternative way (maybe this is the good one) """

        # for c2q in C2Q[0]:
        #     matrix = tf.tile(tf.expand_dims(c2q, 0), [U.shape[-2], 1])  # repeat first row of C2Q 2d times --> 2d x J
        #     matrix = tf.multiply(matrix, U[0])  # 2d x J (elementwise multiplication)
        #     row = tf.reduce_sum(matrix, -1)  # 2d,
        #     rows.append(row)
        # final = tf.stack(rows)
        T = S.shape[-2]
        J = S.shape[-1]
        C2Q = tf.reshape(C2Q, [-1, T*J])
        attended_query = []
        for i in range(C2Q.shape[0]):
            a = tf.reduce_sum(tf.stack(tf.split(C2Q[i] * tf.tile(U[i], [1, T]), num_or_size_splits=T, axis=-1)), axis=-1)
            attended_query.append(a)
        # attended_query = tf.reshape(tf.reduce_sum(tf.stack(tf.split(C2Q * tf.tile(U, [1, 1, T]), num_or_size_splits=T, axis=-1)), axis=-1), [-1, T, U.shape[-2]])  # batch, T x 2d
        attended_query = tf.stack(attended_query)
        # """ """
        # tensor = []
        # for batch, C in enumerate(C2Q):
        #     rows = []
        #     for c2q in C:
        #         row = c2q * U[batch]  # 2d x J
        #         row = tf.reduce_sum(row, -1)  # 2d,
        #         rows.append(row)
        #     final = tf.stack(rows)  # T x 2d
        #     tensor.append(final)
        # attended_query = tf.stack(tensor)  # [batches, T, 2d]
        return attended_query

    def computeQuery2ContextAttention(self, S, H, mask=None):
        """
        Create Q2C attention matrix: which context words have the closest similarity to one of the query words
        and are therefore critical for answering the query.
        :param S: similarity matrix (T x J)
        :param H: context matrix (T x 2d)
        :return: attended_context: Q2C matrix (H̃) (T x 2d)
        """
        # Q2C = tf.nn.softmax(addMask(tf.reduce_max(S, axis=-1), mask))
        # # print("Q2C shape (T x 1): ", Q2C.shape)
        # Q2C = tf.expand_dims(Q2C, -1)

        # attended_context = tf.matmul(Q2C, H, transpose_a=True)
        # # print("attended_context shape (1 x d): ", AttendedContext.shape)
        # attended_context = tf.tile(attended_context, [1, H.shape[-2], 1])
        # # print("attended_context shape (T x d): ", attended_context.shape)
        """ from here new"""
        H = tf.transpose(H, perm=[0, 2, 1])  # change U dims: from J x 2d to 2d x J
        fin = []
        for i in range(H.shape[0]):
            h = tf.nn.softmax(tf.reduce_max(S[i], -1)) * H[i]
            h = tf.reduce_sum(h, axis=1)
            h = tf.tile(tf.expand_dims(h, 0), [S.shape[1], 1])  # tile T times --> final shapes == Tx 2d
            fin.append(h)
        attended_context = tf.stack(fin)

        return attended_context

    def merge(self, H, attended_query, attended_context):
        """
        Combine the information obtained by the C2Q and Q2C attentions.
        Each column vector of G can be considered as the query-aware representation of each context word.
        :param H: context matrix (T x 2d)
        :param attended_query: C2Q matrix (Ũ) (T x 2d)
        :param attended_context: Q2C matrix (H̃) (T x 2d)
        :return: G: matrix (T x 8d)
        """
        if attended_context is not None:
            G = tf.concat([H, attended_query, tf.multiply(H, attended_query), tf.multiply(H, attended_context)],
                          axis=-1)
        else:
            G = tf.concat([H, attended_query, tf.multiply(H, attended_query)], axis=-1)  # q2c ablation
            # to be fixed FIXME
        # print(" G shape (T X 8d) / (T x 6d) in case of q2c ablation : ", G.shape)
        return G

    def call(self, H, U, q2c_attention, c2q_attention, mask=None):
        # Similarity matrix (S) dimension: TxJ
        S = self.computeSimilarity(H, U)

        # C2Q attention
        if c2q_attention:
            C2Q = self.computeContext2QueryAttention(S, U, mask)
        else:
            C2Q = tf.tile(tf.expand_dims(tf.reduce_mean(U, 1), 1), [1, 256, 1])  # TODO check if it's ok
        # Q2C attention
        if q2c_attention:
            Q2C = self.computeQuery2ContextAttention(S, H, mask)
        else:
            Q2C = None  # to be fixed FIXME
        # Merge C2Q (Ũ) and Q2C (H̃) to obtain G
        G = self.merge(H, C2Q, Q2C)

        return G, S


class OutputLayer(tf.keras.layers.Layer):
    def __init__(self, lstm_units, dropout):
        super(OutputLayer, self).__init__()
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=True, dropout=dropout), merge_mode='concat')

        self.dropout_out1 = tf.keras.layers.Dropout(.2)
        self.dropout_out2 = tf.keras.layers.Dropout(.2)

    def build(self, input_shape):
        # M input_shape = (T X 2d) -> W dim: (10d, 1)
        self.W_start = self.add_weight(shape=(input_shape[-1] * 5, 1), trainable=True)
        self.W_end = self.add_weight(shape=(input_shape[-1] * 5, 1), trainable=True)
        super(OutputLayer, self).build(input_shape)

    def call(self, M, G, mask, training=False):
        # qui dropout
        G = self.dropout_out1(G, training=training)
        M = self.dropout_out2(M, training=training)

        p_start = tf.matmul(tf.concat([G, M], axis=-1), self.W_start)
        p_start = addMask(tf.squeeze(p_start, axis=-1), mask)
        # subtract by a very big number each element of the mask before apply softmax -> masked (padded) elements will be close to zero (zero prob)
        p_start = tf.nn.softmax(p_start, axis=-1)

        M = self.bi_lstm(M, training=training)
        # M = tf.squeeze(M, [0])  # Removes dimensions of size 1 from the shape of a tensor. (in position 0)
        # print(" new M shape ( Tx 2d): ", M.shape)

        # qui dropout
        p_end = tf.matmul(tf.concat([G, M], axis=-1), self.W_end)
        p_end = addMask(tf.squeeze(p_end, axis=-1), mask)
        p_end = tf.nn.softmax(p_end, axis=-1)

        return p_start, p_end


def addMask(tensor, mask):
    if mask is None:
        return tensor
    mask = (tf.cast(mask, 'float32') + 1) % 2
    mask *= -1000
    tensor += mask
    return tensor
