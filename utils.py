import pickle

import tensorflow as tf
import sys
import os
import numpy as np
import sklearn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')), tf.test.is_gpu_available())
epsilon = 10e-5


def em_metric(y_true, y_pred, dev_set=False, dataset="squad"):
    # print("compute em score")
    if dataset == "squad":
        em_score = emSquad(y_true, y_pred, dev_set=dev_set)
    else:
        em_score = emTrivia(y_true, y_pred)
    return em_score


def emSquad(y_true, y_pred, dev_set=False):
    # print("compute em score")
    start, end = 0, 1
    em_count = 0
    assert len(y_true) > 0

    # dev set
    if dev_set:
        for batch_idx, y_true_i in enumerate(y_true):
            start_idx, end_idx = get_answer(y_pred[start][batch_idx], y_pred[end][batch_idx])
            count = [1 if (start_idx == y[0] and end_idx == y[1]) else 0 for y in y_true_i]
            if max(count):
                em_count += 1

    # training set
    else:
        for batch_idx, (y_true_start, y_true_end) in enumerate(y_true):
            # start_idx, end_idx = get_answer(y_pred[batch_idx][0], y_pred[batch_idx][1])
            start_idx, end_idx = get_answer(y_pred[start][batch_idx], y_pred[end][batch_idx])
            # print("{}-> start (true-pred):{}-{}; end (true-pred):{}-{}".format(batch_idx, y_true_start, start_idx, y_true_end, end_idx))
            if start_idx == y_true_start and end_idx == y_true_end:
                em_count += 1

    return 100 * em_count / len(y_true)


def emTrivia(y_true, y_pred):
    # print("compute em score")
    start, end = 0, 1
    em_count = 0
    assert len(y_true) > 0

    for batch_idx, (y_true_start, y_true_end) in enumerate(y_true):
        start_idx, end_idx = get_answer(y_pred[start][batch_idx], y_pred[end][batch_idx])
        if start_idx == y_true_start and end_idx == y_true_end:
            em_count += 1

    return 100 * em_count / len(y_true)


def f1_metric(y_true, y_pred, dev_set=False, dataset="squad"):
    # print("f1 score")
    if dataset == "squad":
        f1_score = f1Squad(y_true, y_pred, dev_set=dev_set)
    else:
        f1_score = f1Trivia(y_true, y_pred)
    return f1_score


def f1Squad(y_true, y_pred, dev_set=False):
    # print("compute f1 score")
    start, end = 0, 1
    f1_score = 0

    # dev set
    if dev_set:
        for batch_idx, y_true_i in enumerate(y_true):
            start_idx, end_idx = get_answer(y_pred[start][batch_idx], y_pred[end][batch_idx])

            f1_temp = []
            f1_temp.append(0)  # initialize if num same == 0
            for i in y_true_i:
                list_true = list(range(i[0], i[1] + 1))
                list_pred = list(range(start_idx, end_idx + 1))
                num_same = len(set(list_true).intersection(list_pred))  # num common tokens predicted
                assert len(list_pred) > 0 and len(list_true) > 0, "lists of predicted elements is empty, in f1 metric"
                if num_same != 0:
                    precision = 1.0 * num_same / len(list_pred)
                    recall = 1.0 * num_same / len(list_true)
                    f1 = (2 * precision * recall) / (precision + recall)
                    f1_temp.append(f1)

            f1_score += max(f1_temp)

    # training set
    else:
        for batch_idx, (y_true_start, y_true_end) in enumerate(y_true):
            start_idx, end_idx = get_answer(y_pred[start][batch_idx], y_pred[end][batch_idx])

            list_true = list(range(y_true_start, y_true_end + 1))
            list_pred = list(range(start_idx, end_idx + 1))
            num_same = len(set(list_true).intersection(list_pred))  # num common tokens predicted
            assert len(list_pred) > 0 and len(list_true) > 0, "lists of predicted elements is empty, in f1 metric"
            if num_same != 0:
                precision = 1.0 * num_same / len(list_pred)
                recall = 1.0 * num_same / len(list_true)
                f1 = (2 * precision * recall) / (precision + recall)
                f1_score += f1
    return 100 * f1_score / len(y_true)


def f1Trivia(y_true, y_pred):
    # print("compute f1 score")
    start, end = 0, 1
    f1_score = 0

    for batch_idx, (y_true_start, y_true_end) in enumerate(y_true):
        start_idx, end_idx = get_answer(y_pred[start][batch_idx], y_pred[end][batch_idx])

        list_true = list(range(y_true_start, y_true_end + 1))
        list_pred = list(range(start_idx, end_idx + 1))
        num_same = len(set(list_true).intersection(list_pred))  # num common tokens predicted
        assert len(list_pred) > 0 and len(list_true) > 0, "lists of predicted elements is empty, in f1 metric"
        if num_same != 0:
            precision = 1.0 * num_same / len(list_pred)
            recall = 1.0 * num_same / len(list_true)
            f1 = (2 * precision * recall) / (precision + recall)
            f1_score += f1
    return 100 * f1_score / len(y_true)


def computeLoss2(y_true, y_pred):
    # print("compute loss")
    start, end = 0, 1
    # TODO: do it with tensors instead of lists
    p_starts = [y_pred[idx][start][yi[start]] for idx, yi in enumerate(y_true)]
    p_ends = [y_pred[idx][end][yi[end]] for idx, yi in enumerate(y_true)]
    log_s = tf.math.log(p_starts) + tf.math.log(p_ends)
    mean = - tf.reduce_mean(log_s)

    # keras loss
    probs = p_starts + p_ends  # concatenate lists
    keras_loss = tf.keras.losses.binary_crossentropy(np.ones(len(probs)), tf.squeeze(probs)) * 2  # compute loss
    return mean, keras_loss


def computeLoss(y_true, y_pred):
    start, end = 0, 1
    p_starts = [y_pred[idx][start][yi[start]] for idx, yi in enumerate(y_true)]
    p_ends = [y_pred[idx][end][yi[end]] for idx, yi in enumerate(y_true)]
    # print("loss p_starts{}, p_ensd{}, \n p starts dims {} {}, p ends dims{} {}".format(p_starts,p_ends, len(p_starts), len(p_starts[0]), len(p_ends), len(p_ends[0])))
    probs = p_starts + p_ends  # concatenate lists
    # print("len probs vector :", len(probs))
    loss = tf.keras.losses.binary_crossentropy(np.ones(len(probs)), tf.squeeze(probs)) * 2  # compute loss

    # custom loss
    log_s = tf.math.log(p_starts) + tf.math.log(p_ends)
    mean = - tf.reduce_mean(log_s)
    return loss, mean


def computeLossTensors(y_true, y_pred):
    start, end = 0, 1
    y_true_start_idx = []
    y_true_end_idx = []

    # fixed for dev set
    if len(y_true.shape) == 3:
        y_true = y_true[:, 0]

    for batch_idx in range(len(y_true)):
        y_true_start_idx.append([batch_idx, y_true[batch_idx][start]])
        y_true_end_idx.append([batch_idx, y_true[batch_idx][end]])

    p_starts = y_pred[start]
    p_ends = y_pred[end]

    if p_starts.shape.rank == 1:
        p_starts = tf.expand_dims(p_starts, 0)
        p_ends = tf.expand_dims(p_ends, 0)

    p_starts = tf.gather_nd(p_starts, y_true_start_idx)
    p_ends = tf.gather_nd(p_ends, y_true_end_idx)

    probs = tf.concat([p_starts, p_ends], axis=-1)
    # print("len probs vector :", len(probs))
    loss = tf.keras.losses.binary_crossentropy(np.ones(len(probs)), probs) * 2  # compute loss

    # custom loss
    # log_s = tf.math.log(p_starts) + tf.math.log(p_ends)
    # mean = - tf.reduce_mean(log_s)
    return loss


def get_answer(y_start, y_end):
    # predicted vector values for start and end
    y_end = tf.transpose(tf.expand_dims(y_end, 0))
    prod_matrix = tf.multiply(y_start, y_end)  # FIXME control if it's correct
    prod_matrix = np.tril(prod_matrix)  # get upper triangle matrix of prod_matrix
    end_idx, start_idx = np.unravel_index(prod_matrix.argmax(), prod_matrix.shape)
    assert start_idx <= end_idx, "Start answer idx must be minor than end answer idx"
    return start_idx, end_idx


def pad3dSequence(seq, max_words=None, chars_maxlen=None, padding='pre', trunc='pre'):
    t = []
    if chars_maxlen is None:
        chars_maxlen = max([len(contexts[0]) for contexts in seq])
    if max_words is None:
        max_words = max([len(contexts) for contexts in seq])

    # padding word_i and context_i if necessary
    for context_i in seq:
        pad_word_i = 0
        pad_context_i = 0
        if len(context_i[0]) < chars_maxlen:
            pad_word_i = max(len(context_i[0]), chars_maxlen) - len(context_i[0])
        if len(context_i) < max_words:
            pad_context_i = max(len(context_i), max_words) - len(context_i)

        if padding == 'pre':
            context_i = np.pad(context_i, ((pad_context_i, 0), (pad_word_i, 0)))  # padding pre
        else:
            context_i = np.pad(context_i, ((0, pad_context_i), (0, pad_word_i)))  # padding post

        # pad = max(len(context_i[0]), chars_maxlen) - len(context_i[0])
        # pad = len(i[0]) - min(len(i[0]), chars_maxlen)

        # truncating words
        start_word = 0
        start_context = 0

        if len(context_i[0]) > chars_maxlen:
            start_word = len(context_i[0]) - min(len(context_i[0]), chars_maxlen)
        if len(context_i) > max_words:
            start_context = len(context_i) - min(len(context_i), max_words)

        if trunc == 'pre':
            context_i = context_i[start_context:, start_word:]
        else:
            if start_context != 0 and start_word != 0:
                context_i = context_i[:-start_context, :-start_word]
            elif start_context == 0 and start_word == 0:
                context_i = context_i[:, :]
            elif start_context == 0:
                context_i = context_i[:, :-start_word]
            else:  # word start == 0:
                context_i = context_i[:-start_context, :]

        t.append(context_i)

    return tf.convert_to_tensor(t)


# def scheduler(epoch, current_learning_rate):
#     if epoch > 2:
#         return current_learning_rate / 10
#     else:
#         return current_learning_rate
#         # return min(current_learning_rate, 0.001)


def shuffle(context_words, context_chars, query_words, query_chars, answer_start_end_idx):
    context_words = context_words if type(context_words) == np.ndarray else context_words.numpy()
    context_chars = context_chars if type(context_chars) == np.ndarray else context_chars.numpy()
    query_words = query_words if type(query_words) == np.ndarray else query_words.numpy()
    query_chars = query_chars if type(query_chars) == np.ndarray else query_chars.numpy()
    answer_start_end_idx = answer_start_end_idx if type(
        answer_start_end_idx) == np.ndarray else answer_start_end_idx.numpy()

    return sklearn.utils.shuffle(context_words, context_chars, query_words, query_chars, answer_start_end_idx)


def getMaxKValues(x, k):
    return np.partition(x, -k)[-k:]


def savePickle(filename, obj):
    max_bytes = 2 ** 31  # 2.14 GiB
    max_bytes = 10  # 2.14 GiB
    bytes_out = pickle.dumps(obj)
    with open(filename, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])
    f_out.close()
    # outfile = open(filename, 'wb')
    # pickle.dump(obj, outfile)
    # outfile.close()


def loadPickle(filename):
    bytes_in = bytearray(0)
    # max_bytes = 2**31  # 2.14 GiB
    max_bytes = 10  # 2.14 GiB
    input_size = os.path.getsize(filename)
    with open(filename, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    f_in.close()
    return pickle.loads(bytes_in)

    # infile = open(filename, 'rb')
    # obj = pickle.load(infile)
    # infile.close()
    # return obj
