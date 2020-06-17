import tensorflow as tf
from tensorflow import keras
from layers import CharacterEmbedding, HighwayLayer, AttentionLayer, OutputLayer
from preprocessing import read_data
import os
import numpy as np
from tqdm import tqdm
import datetime
from sklearn.utils import shuffle
from utils import em_metric, f1_metric, computeLoss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


class BiDafModel(tf.keras.Model):
    def __init__(self, char_vocab_size, conv_filters=100, conv_filter_width=5, dropout_lstm=.2, emb_dim=None):
        super(BiDafModel, self).__init__()
        self.d = 100
        self.emb_dim = char_vocab_size
        self.conv_filters = conv_filters  # output dim for character embedding layer
        self.lstm_units = self.d

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        self.char_emb = CharacterEmbedding(self.emb_dim, self.conv_filters, filter_width=conv_filter_width, emb_dim=emb_dim)
        self.highway1 = HighwayLayer()
        self.highway2 = HighwayLayer()

        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, dropout=dropout_lstm), merge_mode='concat')

        self.attention = AttentionLayer()

        self.modeling_layer1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, dropout=dropout_lstm), merge_mode='concat')
        self.modeling_layer2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, dropout=dropout_lstm), merge_mode='concat')

        self.output_layer_ = OutputLayer(self.lstm_units, dropout_lstm)

    def train(self, c_words, c_chars, q_words, q_chars, y, epochs, batch_size, training, verbose=False):
        # print("train")
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # create tensorboard log dir
        train_log_dir = os.path.abspath(os.path.join('logs/gradient_tape/train/', current_time))
        # tb_path = os.path.abspath(os.path.join(self.folder, "tensorboard"))
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        print('Visualize training progress:\n`tensorboard --logdir="{}"`'.format(train_log_dir))

        # train_loss_results = []
        # train_accuracy_results = []
        num_batches = int(np.ceil(len(c_words) / batch_size))
        # print("num_batches {}, len(c_words) {}".format(num_batches, len(c_words)))

        for epoch in tqdm(range(epochs), desc='Train epochs'):
            ''' init metrics (loss, EM score, F1 score) '''
            epoch_loss = 0.
            epoch_em_score = 0.
            epoch_f1_score = 0.

            # shuffle data
            c_words, c_chars, q_words, q_chars, y = shuffle(c_words, c_chars, q_words, q_chars, y)

            # epoch_loss_avg = tf.keras.metrics.Mean()
            # epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

            # Training loop
            for batch in range(num_batches):
                if verbose: print("{} Batch number {} / {}".format(epoch, batch, num_batches))
                start_idx = batch * batch_size
                end_idx = ((batch + 1) * batch_size if (batch + 1) * batch_size < len(c_words) else len(c_words))
                y_true = y[start_idx:end_idx]
                # print("epoch : {}, batch num {} / {}; batch start-end idx {}-{} , batch dim: {}".format(epoch, batch, num_batches - 1, start_idx, end_idx, len(c_words[start_idx:end_idx])))
                batch_loss, y_pred = self.train_step(c_words[start_idx:end_idx], c_chars[start_idx:end_idx], q_words[start_idx:end_idx], q_chars[start_idx:end_idx], y_true, training, verbose)
                epoch_loss += batch_loss
                epoch_em_score += em_metric(y_true, y_pred)
                epoch_f1_score += f1_metric(y_true, y_pred)

                # epoch_loss_avg.update_state(loss)
                # epoch_accuracy.update_state(y_true, y_pred)

            # End epoch
            # train_loss_results.append(epoch_loss_avg.result())
            # train_accuracy_results.append(epoch_accuracy.result())

            # if epoch % 50 == 0:
            #     print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
            #                                                                 epoch_loss_avg.result(),
            #                                                                 epoch_accuracy.result()))

            # if epoch % 2 == 0:
            #     print("Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

            epoch_loss = epoch_loss / num_batches
            epoch_em_score = epoch_em_score / num_batches
            epoch_f1_score = epoch_f1_score / num_batches

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', epoch_loss, step=epoch)
                tf.summary.scalar('exact match', epoch_em_score, step=epoch)
                tf.summary.scalar('f1', epoch_f1_score, step=epoch)

            print("epoch:{epoch_num}, train_loss: {train_loss}, EM score: {em_score}, F1 score: {f1_score} "
                  .format(epoch_num=epoch, train_loss=epoch_loss, em_score=epoch_em_score, f1_score=epoch_f1_score))

# ---------------------------------------- #
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

    def train_step(self, c_words, c_chars, q_words, q_chars, y, training, verbose):
        if verbose: print("train step")
        with tf.GradientTape() as tape:
            if verbose: print("forward pass")
            y_pred = self.call(c_words, c_chars, q_words, q_chars, training, verbose)  # forward pass
            if verbose: print("compute loss")
            loss = computeLoss(y, y_pred)  # calculate loss
        if verbose: print("compute grads, number of trainable weights: ", len(self.trainable_weights))
        grads = tape.gradient(loss, self.trainable_weights)  # backpropagation
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))  # optimizer step
        return loss, y_pred

    def call(self, c_word_input, c_char_input, q_word_input, q_char_input, training, verbose):
        # first c stands for context second for char, q stands for query
        # TODO use tf.split()
        # query, context = input1, input2
        # c_char_input = tf.convert_to_tensor(c_char_input)
        y_pred = []
        for i in range(len(c_word_input)):
            if verbose: print("{}: char embedding".format(i))
            cc = self.char_emb(c_char_input[i], training)
            qc = self.char_emb(q_char_input[i], training)

            # concatenate word representation vector given by chars and word representation vector by GloVe
            context0 = tf.concat([cc, c_word_input[i]], axis=-1)
            query0 = tf.concat([qc, q_word_input[i]], axis=-1)

            if verbose: print("{}: highway".format(i))
            # highway layers
            context = self.highway2(self.highway1(context0))
            query = self.highway2(self.highway1(query0))

            if verbose: print("{}: biLSTM".format(i))
            # Contextual Embedding Layer (bidirectional LSTM)
            H = self.bi_lstm(tf.expand_dims(context, 0))
            U = self.bi_lstm(tf.expand_dims(query, 0))
            # context matrix (H) dimension: 2d x T, query matrix (U) dimension: 2d x J

            H = tf.squeeze(H, [0])  # Removes dimensions of size 1 from the shape of a tensor. (in position 0)
            U = tf.squeeze(U, [0])

            if verbose: print("{}: attention".format(i))
            # Attention Layer -> G matrix (T x 8d)
            G = self.attention(H, U)

            if verbose: print("{}: modeling layer".format(i))
            M = self.modeling_layer2(self.modeling_layer1(tf.expand_dims(G, 0)))

            # print("M shape (T x 2d): ", M.shape)
            M = tf.squeeze(M, [0])  # Removes dimensions of size 1 from the shape of a tensor. (in position 0)

            if verbose: print("{}: output layer".format(i))
            p_start, p_end = self.output_layer_(M, G)
            # print("output dims: (T, 1) :", p_start.shape, p_end.shape)
            y_pred.append([p_start, p_end])
        return y_pred



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
