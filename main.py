import argparse
import json
import os
import datetime
import numpy as np
import tensorflow as tf
# import tensorflow_datasets as tfds
from preprocessing import preprocessingSquad
from model import BiDafModel
# from layers import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main(epochs, batch_size, dataset_dim, training_set_dim, validation_set_dim, load_model, verbose, use_char_emb, use_word_emb, q2c_attention, c2q_attention, dynamic_attention):
    # gpu settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("q2c_attention: {},  use char emb: {},  use word emb: {}".format(q2c_attention, use_char_emb, use_word_emb))

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
    c_words, c_chars, q_words, q_chars, answer_start_end_idx, vocab_size_t, _ = preprocessingSquad(train_source_path, dataset_len=training_set_dim, pre_batch_size=10)

    # get a portion of dataset
    if dataset_dim != 0:
        c_words, c_chars, q_words, q_chars, answer_start_end_idx = c_words[:dataset_dim], c_chars[:dataset_dim], q_words[:dataset_dim], q_chars[:dataset_dim], answer_start_end_idx[:dataset_dim]

    cw_train, cc_train, qw_train, qc_train, y_train = c_words, c_chars, q_words, q_chars, answer_start_end_idx

    """ validation set """
    c_words, c_chars, q_words, q_chars, answer_start_end_idx, vocab_size_v, _ = preprocessingSquad(dev_source_path, dataset_len=validation_set_dim, is_validation_set=True, pre_batch_size=10)
    cw_val, cc_val, qw_val, qc_val, y_val = c_words, c_chars, q_words, q_chars, answer_start_end_idx

    X_train, y_train = (cw_train, cc_train, qw_train, qc_train), y_train
    X_val, y_val = (cw_val, cc_val, qw_val, qc_val), y_val
    vocab_size = max(vocab_size_t, vocab_size_v)
    print("vocab_size {}, dataset dimension -> train {}: ; validation: {}".format(vocab_size, len(y_train), len(y_val)))
    model = BiDafModel(vocab_size)
    print(model.trainable_variables)
    model.train(X_train, y_train, X_val, y_val, use_char_emb, use_word_emb, q2c_attention=q2c_attention, c2q_attention=c2q_attention, epochs=epochs, batch_size=batch_size, training=True, verbose=verbose)


    # model.compile(optimizer='Adam', metrics=['accuracy'])
    # model.fit(x[0])
    # checkpoints_dir = "./checkpoints/my_checkpoint"
    # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # save_model_dir = "saved_model/my_model"
    # load_model = False

    # if not os.path.exists(checkpoints_dir):
    #     os.makedirs(checkpoints_dir)
    # model_path = os.path.join(checkpoints_dir, "1")

    # if not load_model:
    #     print("training")
    #     model.train(c_words, c_chars, q_words, q_chars, answer_start_end_idx, epochs=2, batch_size=4, training=True)
        # checkpoints_dir = os.path.join(checkpoints_dir, current_time)
        # print("model.weights", len(model.weights))
        # print("model.trainable_weights", len(model.trainable_weights))
        # model.save_weights(model_path)
        # print("weights saved")
        # model.save(save_model_dir)
    # else:
    #     model.load_weights(checkpoints_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # epochs
    parser.add_argument("-e", "--epochs", help="number of epochs for training", default=5, type=int)
    # batch_size
    parser.add_argument("-b", "--batch_size", help="batch dimension", default=1, type=int)
    # dataset dimension
    parser.add_argument("-d", "--dataset-dim", help="dimension of the dataset for training ", default=0, type=int)

    # dataset training dim
    parser.add_argument("-td", "--training-set-dim", help="dimension of the dataset for training ", default=float('inf'), type=int)  # , default=float('inf')
    parser.add_argument("-vd", "--validation-set-dim", help="dimension of the dataset for validation ", default=float('inf'), type=int)

    # save model path
    parser.add_argument("-cp", "--checkpoint-path", help="path where to save the weights", default=None)

    # verbose
    parser.add_argument("-v", "--verbose", help="add comments", default=True, type=bool)

    # Ablations
    # TODO adjust cmd line arguments True/ False
    parser.add_argument("-cemb", "--use_char_emb", help="use char embedding?", default=True, type=bool, choices=[False, True])
    parser.add_argument("-wemb", "--use_word_emb", help="use word embedding?", default=True, type=bool)
    parser.add_argument("-q2c_att", "--q2c-attention", help="use question-to-context attention?", default=True, type=bool)
    parser.add_argument("-c2q_att", "--c2q-attention", help="use context-to-question attention?", default=True, type=bool)
    parser.add_argument("-dyn_att", "--dynamic_attention", help="use dynamic attention?", default=False, type=bool)

    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.dataset_dim, args.training_set_dim, args.validation_set_dim, args.checkpoint_path, args.verbose, args.use_char_emb, args.use_word_emb, args.q2c_attention, args.c2q_attention, args.dynamic_attention)

