import argparse
import json
import os
import datetime
import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
from preprocessing import preprocessingSquad, read_squad_data_v2
from model import BiDafModel
# from layers import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main(epochs, batch_size, dataset_dim, training_set_dim, validation_set_dim, load_model, verbose, use_char_emb, use_word_emb, q2c_attention, c2q_attention, dynamic_attention):
    # gpu settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("verb: {}, q2c_attention: {},  use char emb: {},  use word emb: {}".format(verbose, q2c_attention, use_char_emb, use_word_emb))

    # reproducibility
    np.random.seed(0)
    tf.random.set_seed(0)

    # local dataset locations
    train_source_path = "./dataset/squad/downloads/rajpurkar_SQuAD-explorer_train-v1.1NSdmOYa4KVr09_zf8bof8_ctB9YaIPSHyyOKbvkv2VU.json"
    # source_path = "./DATA/squad/train-v1.1.json"
    # source_path = "./DATA/squad/dev-v1.1.json"
    dev_source_path = "./dataset/squad/downloads/rajpurkar_SQuAD-explorer_dev-v1.1lapqUtXWpzVWM2Z1PKUEkqZYAx2nTzAaxSOLA5Zpcsk.json"

    # get squad dataset
    """ training set """
    # c_words, c_chars, q_words, q_chars, answer_start_end_idx, vocab_size_t, _ = preprocessingSquad(train_source_path, dataset_len=training_set_dim, pre_batch_size=10)
    c_words, c_chars, q_words, q_chars, answer_start_end_idx, vocab_size_t, _ = read_squad_data_v2(train_source_path)

    # get a portion of dataset
    if dataset_dim != 0:
        c_words, c_chars, q_words, q_chars, answer_start_end_idx = c_words[:dataset_dim], c_chars[:dataset_dim], q_words[:dataset_dim], q_chars[:dataset_dim], answer_start_end_idx[:dataset_dim]

    cw_train, cc_train, qw_train, qc_train, y_train = c_words, c_chars, q_words, q_chars, answer_start_end_idx

    """ validation set """
    # c_words, c_chars, q_words, q_chars, answer_start_end_idx, vocab_size_v, _ = preprocessingSquad(dev_source_path, dataset_len=validation_set_dim, is_validation_set=True, pre_batch_size=10)
    c_words, c_chars, q_words, q_chars, answer_start_end_idx, vocab_size_v, _ = read_squad_data_v2(dev_source_path)
    cw_val, cc_val, qw_val, qc_val, y_val = c_words, c_chars, q_words, q_chars, answer_start_end_idx

    X_train, y_train = (cw_train, cc_train, qw_train, qc_train), y_train
    X_val, y_val = (cw_val, cc_val, qw_val, qc_val), y_val
    vocab_size = max(vocab_size_t, vocab_size_v)
    print("vocab_size {}, dataset dimension -> train {}: ; validation: {}".format(vocab_size, len(y_train), len(y_val)))
    model = BiDafModel(vocab_size)
    print(model.trainable_variables)
    model.train(X_train, y_train, X_val, y_val, use_char_emb, use_word_emb, q2c_attention=q2c_attention, c2q_attention=c2q_attention, epochs=epochs, batch_size=batch_size, training=True, verbose=verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # epochs
    parser.add_argument("-e", "--epochs", help="number of epochs for training", default=5, type=int)
    # batch_size
    parser.add_argument("-b", "--batch_size", help="batch dimension", default=5, type=int)
    # dataset dimension
    parser.add_argument("-d", "--dataset-dim", help="dimension of the dataset for training ", default=0, type=int)

    # dataset training dim
    parser.add_argument("-td", "--training-set-dim", help="dimension of the dataset for training ", default=2, type=int)  # , default=float('inf')
    parser.add_argument("-vd", "--validation-set-dim", help="dimension of the dataset for validation ", default=2, type=int)

    # save model path
    parser.add_argument("-cp", "--checkpoint-path", help="path where to save the weights", default=None)

    # verbose
    verbose_parser = parser.add_mutually_exclusive_group(required=False)
    verbose_parser.add_argument("-v", "--verbose", help="add comments", dest='verbose', action='store_true')
    verbose_parser.add_argument("-no-v", '--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

    # Ablations
    # TODO adjust cmd line arguments True/ False
    # 1) character embedding
    parser.add_argument("-no-cemb", "--char_emb", help="ablating char embedding", action="store_false", default=True)
    # 2) word embedding
    parser.add_argument("-no-wemb", "--word_emb", help="ablating word embedding", action="store_false", default=True)
    # 3) Q2C attention
    parser.add_argument("-no-q2c-att", "--q2c_attention", help="ablating question-to-context attention", action="store_false", default=True)
    # 4) C2Q attention
    parser.add_argument("-no-c2q-att", "--c2q_attention", help="ablating context-to-question attention", action="store_false", default=True)
    # 5) dynamic attention
    parser.add_argument("-dyn-att", "--dynamic_attention", help="use dynamic attention?", action="store_true", default=False)

    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.dataset_dim, args.training_set_dim, args.validation_set_dim, args.checkpoint_path, args.verbose, args.char_emb, args.word_emb, args.q2c_attention, args.c2q_attention, args.dynamic_attention)

