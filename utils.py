import tensorflow as tf
import sys
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')), tf.test.is_gpu_available())
epsilon = 10e-5


def em_metric(y_true, y_pred):
    # print("compute em score")
    em_count = 0
    assert len(y_true) > 0
    for y_idx, (y_true_start, y_true_end) in enumerate(y_true):
        start_idx, end_idx = get_answer(y_pred[y_idx][0], y_pred[y_idx][1])
        # print("{}-> start (true-pred):{}-{}; end (true-pred):{}-{}".format(y_idx, y_true_start, start_idx, y_true_end, end_idx))
        if start_idx == y_true_start and end_idx == y_true_end:
            em_count += 1
    return 100 * em_count / len(y_true)


def f1_metric(y_true, y_pred):
    # print("compute f1 score")
    f1_score = 0
    for y_idx, (y_true_start, y_true_end) in enumerate(y_true):
        start_idx, end_idx = get_answer(y_pred[y_idx][0], y_pred[y_idx][1])
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
    probs = p_starts + p_ends  #concatenate lists
    keras_loss = tf.keras.losses.binary_crossentropy(np.ones(len(probs)), tf.squeeze(probs)) * 2  # compute loss
    return mean, keras_loss


def computeLoss(y_true, y_pred):
    start, end = 0, 1
    p_starts = [y_pred[idx][start][yi[start]] for idx, yi in enumerate(y_true)]
    p_ends = [y_pred[idx][end][yi[end]] for idx, yi in enumerate(y_true)]
    # print("loss p_starts{}, p_ensd{}, \n p starts dims {} {}, p ends dims{} {}".format(p_starts,p_ends, len(p_starts), len(p_starts[0]), len(p_ends), len(p_ends[0])))
    probs = p_starts + p_ends  #concatenate lists
    # print("len probs vector :", len(probs))
    loss = tf.keras.losses.binary_crossentropy(np.ones(len(probs)), tf.squeeze(probs)) * 2  # compute loss

    # custom loss
    log_s = tf.math.log(p_starts) + tf.math.log(p_ends)
    mean = - tf.reduce_mean(log_s)
    return loss, mean


def get_answer(y_start, y_end):
    # predicted vector values for start and end
    y_end = tf.transpose(y_end)
    prod_matrix = tf.multiply(y_start, y_end)
    prod_matrix = np.triu(prod_matrix)  # get upper triangle matrix of prod_matrix
    start_idx, end_idx = np.unravel_index(prod_matrix.argmax(), prod_matrix.shape)
    return start_idx, end_idx


# def scheduler(epoch, current_learning_rate):
#     if epoch > 2:
#         return current_learning_rate / 10
#     else:
#         return current_learning_rate
#         # return min(current_learning_rate, 0.001)


def plotHistogramOfLengths(data_sets, data_labels):
    import matplotlib.pyplot as plt
    number_of_bins = 10
    for idx, data in enumerate(data_sets):
        plt.clf()
        plt.hist(data, number_of_bins, alpha=0.5, label=data_labels[idx])
        print('{}: max: {}, min: {}, mean:{}'.format(data_labels[idx], np.max(data_sets[idx]), np.min(data_sets[idx]), np.mean(data_sets[idx])))
        # plt.gca().set(title='Length Histogram - context words', ylabel='Frequency', xlabel='Words length')
        plt.legend(loc='upper right')
        plt.show()
        # eventually save figures with plt.savefig(path/image.png)

