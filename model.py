import tensorflow as tf
from layersTensors import CharacterEmbedding, HighwayLayer, AttentionLayer, OutputLayer
from preprocessing import read_data, readSquadDataPadding
import os
import numpy as np
from tqdm import tqdm
import datetime
# from sklearn.utils import shuffle
from utils import em_metric, f1_metric, computeLoss, get_answer, computeLossTensors, plotAttentionMatrix, shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  #FIXME remove


class BiDafModel(tf.keras.Model):
    def __init__(self, char_vocab_size, conv_filters=100, conv_filter_width=5, dropout_lstm=.2, emb_dim=8):
        super(BiDafModel, self).__init__()
        self.d = 100
        self.emb_dim = char_vocab_size
        self.conv_filters = conv_filters  # output dim for character embedding layer
        self.lstm_units = self.d

        # set optimizer
        # self.lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
        # tf.keras.callbacks.

        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.5)
        # self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.5, decay=0.999)
        # self.ema = tf.train.ExponentialMovingAverage(decay=0.999)

        self.char_emb = CharacterEmbedding(self.emb_dim, self.conv_filters, filter_width=conv_filter_width,
                                           emb_dim=emb_dim)
        self.highway1 = HighwayLayer()
        self.highway2 = HighwayLayer()
        # self.highway1_q = HighwayLayer()
        # self.highway2_q = HighwayLayer()

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

    def train(self, X_train, y_train, X_val, y_val, use_char_emb, use_word_emb, q2c_attention, c2q_attention, epochs,
              batch_size, training, verbose=False):
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
        num_batches_v = int(np.ceil(len(cw_val) / batch_size))  # number batches for validation
        # print("num_batches {}, len(c_words) {}".format(num_batches, len(c_words)))

        for epoch in tqdm(range(epochs), desc='Train epochs'):
            ''' init metrics (loss, EM score, F1 score) '''
            # training
            epoch_loss, epoch_em_score, epoch_f1_score = [], [], []
            # validation
            epoch_loss_v, epoch_em_score_v, epoch_f1_score_v = [], [], []

            """ shuffle data """
            # shuffle training data
            assert (len(cw_train) == len(cc_train) == len(qw_train) == len(qc_train) == len(y_train))
            cw_train, cc_train, qw_train, qc_train, y_train = shuffle(cw_train, cc_train, qw_train, qc_train, y_train)

            # shuffle validation data
            assert (len(cw_val) == len(cc_val) == len(qw_val) == len(qc_val) == len(y_val))
            cw_val, cc_val, qw_val, qc_val, y_val = shuffle(cw_val, cc_val, qw_val, qc_val, y_val)

            # learning rate scheduler
            # self.lr_scheduler(epoch)

            """ Training """
            # training loop
            if verbose: print('Training')
            for batch in range(num_batches_t):
                if verbose: print("{} Batch number {} / {}".format(epoch, batch, num_batches_t))
                start_idx = batch * batch_size
                end_idx = ((batch + 1) * batch_size if (batch + 1) * batch_size < len(cw_train) else len(cw_train))
                y_true = y_train[start_idx:end_idx]
                if verbose: print("batch dimension: ", len(y_true))
                batch_loss, y_pred = self.train_step(cw_train[start_idx:end_idx],
                                                     cc_train[start_idx:end_idx],
                                                     qw_train[start_idx:end_idx],
                                                     qc_train[start_idx:end_idx], y_true, use_char_emb,
                                                     use_word_emb, q2c_attention, c2q_attention,
                                                     training, verbose, batch, end_idx - start_idx)

                epoch_loss.append(batch_loss)
                epoch_em_score.append(em_metric(y_true, y_pred))
                # epoch_f1_score.append(accuracy(y_true, y_pred))
                epoch_f1_score.append(f1_metric(y_true, y_pred))

            # Validation
            if verbose: print('Validation')
            # val_loss, y_pred = self.validation_step(cw_val, cc_val, qw_val, qc_val, y_val, use_char_emb,
            #                                         use_word_emb, q2c_attention, c2q_attention, verbose)
            for batch in range(num_batches_v):
                if verbose: print("{} Batch number {} / {}".format(epoch, batch, num_batches_v))
                start_idx = batch * batch_size
                end_idx = ((batch + 1) * batch_size if (batch + 1) * batch_size < len(cw_val) else len(cw_val))
                y_true = y_val[start_idx:end_idx]
                if verbose: print("batch dimension: ", len(y_true))
                val_batch_loss, y_pred = self.validation_step(cw_val[start_idx:end_idx],
                                                              cc_val[start_idx:end_idx],
                                                              qw_val[start_idx:end_idx],
                                                              qc_val[start_idx:end_idx], y_true, use_char_emb,
                                                              use_word_emb, q2c_attention, c2q_attention,
                                                              verbose)  # , batch, end_idx - start_idx)

                epoch_loss_v.append(val_batch_loss)
                epoch_em_score_v.append(em_metric(y_true, y_pred))
                epoch_f1_score_v.append(f1_metric(y_true, y_pred))

                # # validation metrics
                # val_em = em_metric(y_val, y_pred)
                # val_f1 = f1_metric(y_val, y_pred)

            # training metrics mean
            epoch_loss = np.mean(epoch_loss)
            epoch_em_score = np.mean(epoch_em_score)
            epoch_f1_score = np.mean(epoch_f1_score)

            # val metrics mean
            val_loss = np.mean(epoch_loss_v)
            val_em = np.mean(epoch_em_score_v)
            val_f1 = np.mean(epoch_f1_score_v)

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

                # learning rate
                print(float(tf.keras.backend.get_value(self.optimizer.learning_rate)))
                tf.summary.scalar("learning rate", float(tf.keras.backend.get_value(self.optimizer.learning_rate)),
                                  step=epoch)
                # tf.summary.scalar("learning rate", self.optimizer.lr.numpy(), step=epoch)

            print(
                "epoch:{epoch_num}, train_loss: {train_loss}, EM score: {em_score}, F1 score: {f1_score}"
                    .format(epoch_num=epoch, train_loss=epoch_loss, em_score=epoch_em_score, f1_score=epoch_f1_score))

    def train_step(self, c_words, c_chars, q_words, q_chars, y, use_char_emb, use_word_emb, q2c_attention,
                   c2q_attention, training, verbose, batch_idx, batch_dim):
        if verbose: print("train step")
        with tf.GradientTape() as tape:
            if verbose: print("forward pass")
            y_pred = self.call_t(c_words, c_chars, q_words, q_chars, use_char_emb, use_word_emb, q2c_attention,
                                 c2q_attention, training, verbose, batch_idx, batch_dim)  # forward pass
            if verbose: print("compute loss")
            loss = computeLossTensors(y, y_pred)  # calculate loss
            # loss = negative_avg_log_error(y, y_pred)  # calculate loss

        if verbose: print("backward pass; number of trainable weights: ", len(self.trainable_weights))
        grads = tape.gradient(loss, self.trainable_weights)  # backpropagation
        if verbose: print("optimizer.apply_gradients")
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))  # optimizer step
        # versione equivalente (ma non funzionante) con minimize()
        # opt_op = self.optimizer.minimize(computeLoss(y, y_pred), self.trainable_weights)
        # with tf.control_dependencies([opt_op]):
        #     training_op = self.

        return loss, y_pred

    def validation_step(self, c_words, c_chars, q_words, q_chars, y, use_char_emb, use_word_emb, q2c_attention,
                        c2q_attention, verbose):
        y_pred = self.call_t(c_words, c_chars, q_words, q_chars, use_char_emb, use_word_emb, q2c_attention,
                             c2q_attention, training=False, verbose=verbose)
        loss = computeLossTensors(y, y_pred)
        return loss, y_pred

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

    def call(self, c_word_input, c_char_input, q_word_input, q_char_input, use_char_emb, use_word_emb, q2c_attention,
             c2q_attention, training=False, verbose=False):
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
            context = self.highway2(self.highway1(context, training=training))
            query = self.highway2(self.highway1(query, training=training))

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

        return y_pred

    def call_t(self, c_word_input, c_char_input, q_word_input, q_char_input, use_char_emb, use_word_emb,
               q2c_attention, c2q_attention, training=False, verbose=False, batch_idx=0, batch_dim=0):
        # first c stands for context second for char, q stands for query

        assert use_char_emb or use_word_emb, "at least one of the two embedding must be used"
        c_word_input = tf.keras.layers.Masking(mask_value=0)(c_word_input)
        q_word_input = tf.keras.layers.Masking(mask_value=0)(q_word_input)

        if use_char_emb:
            # get character level representation of each word
            cc = self.char_emb(c_char_input, training)
            qc = self.char_emb(q_char_input, training)

        if use_char_emb and use_word_emb:
            # concatenate word representation vector given by chars and word representation vector by GloVe
            context = tf.concat([cc, c_word_input], axis=-1)  # final shape: [batch, num_words, 200]
            query = tf.concat([qc, q_word_input], axis=-1)

        elif not use_word_emb:
            # ablation on word embedding
            context = cc
            query = qc

        elif not use_char_emb:
            # no char embedding
            context = c_word_input
            query = q_word_input

        # highway layers
        context = self.highway2(self.highway1(context))
        query = self.highway2(self.highway1(query))

        # if verbose: print("{}: biLSTM".format(i))
        # Contextual Embedding Layer (bidirectional LSTM)
        H = self.bi_lstm(context, training=training) #, mask=c_word_input._keras_mask)  # use mask here
        U = self.bi_lstm(query, training=training) # , mask=q_word_input._keras_mask)
        # context matrix (H) dimension: 2d x T, query matrix (U) dimension: 2d x J

        # H = tf.squeeze(H, [0])  # Removes dimensions of size 1 from the shape of a tensor. (in position 0)
        # U = tf.squeeze(U, [0])

        # if verbose: print("{}: attention".format(i))
        # Attention Layer -> G matrix (T x 8d)
        # TODO use mask here maybe
        G, similarity_matrix = self.attention(H, U, q2c_attention, c2q_attention)

        # TODO fixme
        # if training is False:
        #     # save similarity matrix in file
        #     for i, el in enumerate(similarity_matrix):
        #         filename = './similarity/sim_' + str(i + batch_idx * batch_dim)
        #         with open(filename, 'wb') as outfile:
        #             np.save(outfile , similarity_matrix[i])
        #         outfile.close()

        # plotAttentionMatrix(tf.transpose(similarity_matrix), i + batch_idx*batch_dim, )

        # if verbose: print("{}: modeling layer".format(i))
        # TODO use mask here
        M = self.modeling_layer2(self.modeling_layer1(G, training=training), training=training) #, mask=c_word_input._keras_mask)

        # print("M shape (T x 2d): ", M.shape)
        # M = tf.squeeze(M, [0])  # Removes dimensions of size 1 from the shape of a tensor. (in position 0)

        # if verbose: print("{}: output layer".format(i))
        p_start, p_end = self.output_layer_(M, G, c_word_input._keras_mask, training)
        # print("output dims: (T, 1) :", p_start.shape, p_end.shape)
        # print("output layer: p start: {}, p end: {}".format(p_start, p_end))
        # y_pred.append()

        return [p_start, p_end]

    def lr_scheduler(self, epoch, logs=None):
        if not hasattr(self.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = 0.5
        if epoch == 12:
            scheduled_lr = lr / 100
            tf.keras.backend.set_value(self.optimizer.lr, scheduled_lr)
        elif epoch == 7:
            scheduled_lr = lr / 10
            tf.keras.backend.set_value(self.optimizer.lr, scheduled_lr)
        # Set the value back to the optimizer before this epoch starts
        # print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))
