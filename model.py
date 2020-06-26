import tensorflow as tf
from tensorflow import keras
from layers import CharacterEmbedding, HighwayLayer, AttentionLayer, OutputLayer
from preprocessing import read_data
import os
import numpy as np
from tqdm import tqdm
import datetime
from sklearn.utils import shuffle
from utils import em_metric, f1_metric, computeLoss, get_answer
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.summary.v1 import custom_scalar_pb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


class BiDafModel(tf.keras.Model):
    def __init__(self, char_vocab_size, conv_filters=100, conv_filter_width=5, dropout_lstm=.2, emb_dim=None):
        super(BiDafModel, self).__init__()
        self.d = 100
        self.emb_dim = char_vocab_size
        self.conv_filters = conv_filters  # output dim for character embedding layer
        self.lstm_units = self.d

        # set optimizer
        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.5)
        # self.ema = tf.train.ExponentialMovingAverage(decay=0.999)

        self.char_emb = CharacterEmbedding(self.emb_dim, self.conv_filters, filter_width=conv_filter_width,
                                           emb_dim=emb_dim)
        self.highway1 = HighwayLayer()
        self.highway2 = HighwayLayer()

        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, dropout=dropout_lstm), merge_mode='concat')

        self.attention = AttentionLayer()

        self.modeling_layer1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, dropout=dropout_lstm), merge_mode='concat')
        self.modeling_layer2 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, dropout=dropout_lstm), merge_mode='concat')

        self.output_layer_ = OutputLayer(self.lstm_units, dropout_lstm)

        self.checkpoints_dir = "checkpoints/my_checkpoint"
        self.saved_model_dir = "saved_model/my_model"

    def train(self, X_train, y_train, X_val, y_val, use_char_emb, use_word_emb, q2c_attention, c2q_attention, epochs, batch_size, training, verbose=False):
        # print("train")
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # create tensorboard log dir
        train_log_dir = os.path.abspath(os.path.join('logs/gradient_tape/train/', current_time))
        # tb_path = os.path.abspath(os.path.join(self.folder, "tensorboard"))
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # print('Visualize training progress:\n`tensorboard --logdir="{}"`'.format(train_log_dir))  # absolute path
        print('Visualize training progress:\n`tensorboard --logdir="{}"`'.format(
            os.path.join('logs/gradient_tape/train/', current_time)))  # relative path

        # dataset preparation
        cw_train, cc_train, qw_train, qc_train = X_train
        cw_val, cc_val, qw_val, qc_val = X_val

        best_f1_acc = 0.
        num_batches_t = int(np.ceil(len(cw_train) / batch_size))  # number batches for training
        # print("num_batches {}, len(c_words) {}".format(num_batches, len(c_words)))

        for epoch in tqdm(range(epochs), desc='Train epochs'):
            ''' init metrics (loss, EM score, F1 score) '''
            epoch_loss = []
            epoch_em_score = []
            epoch_f1_score = []

            # shuffle data
            cw_train, cc_train, qw_train, qc_train, y_train = shuffle(cw_train, cc_train, qw_train, qc_train, y_train)
            cw_val, cc_val, qw_val, qc_val, y_val = shuffle(cw_val, cc_val, qw_val, qc_val, y_val)  # , random_state=0) for reproducibility
            # c_words, c_chars, q_words, q_chars, y = shuffle(c_words, c_chars, q_words, q_chars, y)

            # Training loop
            for batch in range(num_batches_t):
                if verbose: print("{} Batch number {} / {}".format(epoch, batch, num_batches_t))
                start_idx = batch * batch_size
                end_idx = ((batch + 1) * batch_size if (batch + 1) * batch_size < len(cw_train) else len(cw_train))
                y_true = y_train[start_idx:end_idx]
                if verbose: print("batch dimension: ", len(y_true))
                # print("epoch : {}, batch num {} / {}; batch start-end idx {}-{} , batch dim: {}".format(epoch, batch, num_batches - 1, start_idx, end_idx, len(c_words[start_idx:end_idx])))
                batch_loss, y_pred, keras_loss = self.train_step(cw_train[start_idx:end_idx], cc_train[start_idx:end_idx],
                                                     qw_train[start_idx:end_idx], qc_train[start_idx:end_idx], y_true, use_char_emb, use_word_emb, q2c_attention, c2q_attention,
                                                     training, verbose)

                epoch_loss.append(batch_loss)
                epoch_em_score.append(em_metric(y_true, y_pred))
                epoch_f1_score.append(f1_metric(y_true, y_pred))

            # Validation
            val_loss, y_pred, keras_loss_v = self.validation_step(cw_val, cc_val, qw_val, qc_val, y_val, use_char_emb, use_word_emb, q2c_attention, c2q_attention, verbose)
            # validation metrics
            val_em = em_metric(y_val, y_pred)
            val_f1 = f1_metric(y_val, y_pred)

            # training metrics mean
            epoch_loss = np.mean(epoch_loss)
            epoch_em_score = np.mean(epoch_em_score)
            epoch_f1_score = np.mean(epoch_f1_score)

            # model checkpoint - to track the best validation accuracy (maybe with accuracy not with loss)
            # if val_f1 > best_f1_acc:
            #     if not os.path.isdir(self.checkpoints_dir):
            #         os.makedirs(self.checkpoints_dir)
            #     self.checkpoints_dir = os.path.join(self.checkpoints_dir, "cp-{epoch:03d}.ckpt".format(epoch=epoch))
            #     print("checkpoints_dir", self.checkpoints_dir)
            #     best_weights = self.get_weights()
            #     import pickle
            #     pickle.dump(best_weights, open(self.checkpoints_dir, 'wb'))
            #     weights = pickle.load(open(self.checkpoints_dir, 'rb'))
            #     self.set_weights(weights)
            #     # FIXME doesn't work
            #     # self.get_weights()
            #     # save also optimizer state
            #
            #     # self.save_weights(self.checkpoints_dir)
            #     # self.save(self.saved_model_dir)
            #     print("weights saved")
            #     best_f1_acc = val_f1

            with train_summary_writer.as_default():
                # training
                tf.summary.scalar("loss/train", epoch_loss, step=epoch)
                tf.summary.scalar('em/train', epoch_em_score, step=epoch)
                tf.summary.scalar('f1/train', epoch_f1_score, step=epoch)
                # tf.summary.scalar('custom loss (train)', keras_loss_t, step=epoch)

                # validation
                tf.summary.scalar("loss/val", val_loss, step=epoch)
                tf.summary.scalar('em/val', val_em, step=epoch)
                tf.summary.scalar('f1/val', val_f1, step=epoch)

            print(
                "epoch:{epoch_num}, train_loss: {train_loss}, EM score: {em_score}, F1 score: {f1_score}"
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

    # @tf.function
    def train_step(self, c_words, c_chars, q_words, q_chars, y, use_char_emb, use_word_emb, q2c_attention, c2q_attention, training, verbose):
        if verbose: print("train step")
        with tf.GradientTape() as tape:
            if verbose: print("forward pass")
            y_pred = self.call(c_words, c_chars, q_words, q_chars, use_char_emb, use_word_emb, q2c_attention, c2q_attention, training, verbose)  # forward pass
            if verbose: print("compute loss")
            loss, custom_loss = computeLoss(y, y_pred)  # calculate loss
        if verbose: print("backward pass; number of trainable weights: ", len(self.trainable_weights))
        grads = tape.gradient(loss, self.trainable_weights)  # backpropagation
        if verbose: print("optimizer.apply_gradients")
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))  # optimizer step
        # with tf.control_dependencies(grads):
        #     ema_op = self.ema.apply(grads)

        return loss, y_pred, custom_loss

    def validation_step(self, c_words, c_chars, q_words, q_chars, y, use_char_emb, use_word_emb, q2c_attention, c2q_attention, verbose):
        y_pred = self.call(c_words, c_chars, q_words, q_chars, use_char_emb, use_word_emb, q2c_attention, c2q_attention, training=False, verbose=verbose)
        loss, keras_loss = computeLoss(y, y_pred)
        return loss, y_pred, keras_loss

    def test(self, c_words, c_chars, q_words, q_chars, y, verbose=False):
        # FIXME
        if verbose: print("test")
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        test_log_dir = os.path.abspath(os.path.join('logs/gradient_tape/test/', current_time))
        file_writer = tf.summary.create_file_writer(test_log_dir)

        # all the dataset is passed to call function to get y predicted
        y_pred = self.call(c_words, c_chars, q_words, q_chars, training=False, verbose=verbose)
        loss = computeLoss(y, y_pred)
        em_score = em_metric(y, y_pred)
        f1_score = f1_metric(y, y_pred)

        with file_writer.set_as_default():
            tf.summary.scalar("test loss", loss)
            tf.summary.scalar("test em accuracy", em_score)
            tf.summary.scalar("test f1 accuracy", f1_score)


    def call(self, c_word_input, c_char_input, q_word_input, q_char_input, use_char_emb, use_word_emb, q2c_attention, c2q_attention, training=False, verbose=False):
        # first c stands for context second for char, q stands for query
        # TODO use tf.split()
        # query, context = input1, input2
        # c_char_input = tf.convert_to_tensor(c_char_input)
        y_pred = []
        # ragged = None
        assert use_char_emb or use_word_emb, "one of the two embedding must be used"

        for i in range(len(c_word_input)):

            if use_char_emb:
                # get character level representation of each word
                cc = self.char_emb(c_char_input[i], training)
                qc = self.char_emb(q_char_input[i], training)

            if use_char_emb and use_word_emb:
                # concatenate word representation vector given by chars and word representation vector by GloVe
                context = tf.concat([cc, c_word_input[i]], axis=-1)
                query = tf.concat([qc, q_word_input[i]], axis=-1)
            elif not use_char_emb:
                # ablation on char embedding
                context = tf.convert_to_tensor(c_word_input[i])
                query = tf.convert_to_tensor(q_word_input[i])
            else:
                # ablation on word embedding
                context = cc
                query = qc

            # if verbose: print("{}: highway".format(i))
            # highway layers
            context = self.highway2(self.highway1(context))
            query = self.highway2(self.highway1(query))

            # if verbose: print("{}: biLSTM".format(i))
            # Contextual Embedding Layer (bidirectional LSTM)
            H = self.bi_lstm(tf.expand_dims(context, 0), training=training)
            U = self.bi_lstm(tf.expand_dims(query, 0), training=training)
            # context matrix (H) dimension: 2d x T, query matrix (U) dimension: 2d x J

            H = tf.squeeze(H, [0])  # Removes dimensions of size 1 from the shape of a tensor. (in position 0)
            U = tf.squeeze(U, [0])

            # if verbose: print("{}: attention".format(i))
            # Attention Layer -> G matrix (T x 8d)
            G = self.attention(H, U, q2c_attention, c2q_attention)

            # if verbose: print("{}: modeling layer".format(i))
            M = self.modeling_layer2(self.modeling_layer1(tf.expand_dims(G, 0), training=training), training=training)

            # print("M shape (T x 2d): ", M.shape)
            M = tf.squeeze(M, [0])  # Removes dimensions of size 1 from the shape of a tensor. (in position 0)

            # if verbose: print("{}: output layer".format(i))
            p_start, p_end = self.output_layer_(M, G, training)
            # print("output dims: (T, 1) :", p_start.shape, p_end.shape)
            # print("output layer: p start: {}, p end: {}".format(p_start, p_end))
            y_pred.append([p_start, p_end])

            ## versione ragged tensor:
            # y_predi = [p_start, p_end]
            # if ragged is not None:
            #     ragged = tf.concat([ragged, [[tf.squeeze(tf.transpose(p_start)), tf.squeeze(tf.transpose(p_end))]]], axis=2)
            # else:
            #     ragged = tf.RaggedTensor.from_tensor([[tf.squeeze(tf.transpose(p_start)), tf.squeeze(tf.transpose(p_end))]])

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
