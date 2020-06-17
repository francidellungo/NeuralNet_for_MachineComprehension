import tensorflow as tf
import sys
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')), tf.test.is_gpu_available())


def em_metric(y_true, y_pred):
    em_count = 0
    for y_idx, (y_true_start, y_true_end) in enumerate(y_true):
        start_idx, end_idx = get_answer(y_pred[y_idx][0], y_pred[y_idx][1])
        # print("{}-> start (true-pred):{}-{}; end (true-pred):{}-{}".format(y_idx, y_true_start, start_idx, y_true_end, end_idx))
        if start_idx == y_true_start and end_idx == y_true_end:
            em_count += 1
    return 100 * em_count / len(y_true)


def f1_metric(y_true, y_pred):
    f1_score = 0
    for y_idx, (y_true_start, y_true_end) in enumerate(y_true):
        start_idx, end_idx = get_answer(y_pred[y_idx][0], y_pred[y_idx][1])
        list_true = list(range(y_true_start, y_true_end + 1))
        list_pred = list(range(start_idx, end_idx + 1))
        num_same = len(set(list_true).intersection(list_pred))  # num common tokens predicted
        if num_same is not 0:
            precision = 1.0 * num_same / len(list_pred)
            recall = 1.0 * num_same / len(list_true)
            f1 = (2 * precision * recall) / (precision + recall)
            f1_score += f1
    return 100 * f1_score / len(y_true)


def computeLoss(y_true, y_pred):
    start, end = 0, 1
    # TODO: do it with tensors instead of lists
    p_starts = [y_pred[idx][start][yi[start]] for idx, yi in enumerate(y_true)]
    p_ends = [y_pred[idx][end][yi[end]] for idx, yi in enumerate(y_true)]
    log_s = tf.math.log(p_starts) + tf.math.log(p_ends)
    mean = - tf.reduce_mean(log_s)
    return mean

def get_answer(y_start, y_end):
    # predicted vector values for start and end
    y_end = tf.transpose(y_end)
    prod_matrix = tf.multiply(y_start, y_end)
    prod_matrix = np.triu(prod_matrix)  # get upper triangle matrix of prod_matrix
    start_idx, end_idx = np.unravel_index(prod_matrix.argmax(), prod_matrix.shape)
    return start_idx, end_idx
