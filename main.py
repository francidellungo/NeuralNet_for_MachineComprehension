import argparse
import json
import os
import datetime
import tensorflow as tf
from sklearn.utils import shuffle
# import tensorflow_datasets as tfds
from preprocessing import read_squad_data, read_squad_data_v2
from layers import *
from model import BiDafModel

# pydot needed only to plot_model
# source = "dataset/qa/web-example.json"
# evidence = "dataset/evidence"
# train_data = read_data(source, evidence)
# print(len(train_data), type(train_data), type(train_data[0]), type(train_data[0][0]))
# y = np.array([0., 3., 1.])
# model = tf.keras.Sequential()
# model.add(CharacterEmbedding(70, 100))
# model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(train_data, y, epochs=10, batch_size=2)
# model = BiDafModel(68, 100, 100, 100)
# model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(train_data, epochs=10)
# model = tf.keras.Sequential()
# inputs = keras.Input(shape=(100,))
#
# # char embedding
# model.add(tf.keras.layers.Embedding(12, 8))
# model.add(tf.keras.layers.Conv1D(100, 5, activation='relu', padding='valid'))
# model.add(MaxOverTimePoolLayer())
# # draw model in model.png file
#
# model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
# v = [[1, 2, 3, 4, 1, 1, 6, 1, 8, 1, 10, 11], [5, 11, 10, 4, 5, 5, 6, 7, 8, 9, 10, 11], [1, 2, 3, 4, 2, 6, 2, 7, 2, 9, 2, 11], [1, 2, 3, 4, 1, 1, 6, 1, 8, 1, 10, 11]]
# y = [0, 1, 2, 0]
#
# model.fit(v, y, epochs=20, batch_size=2)
# model.summary()
# plot_model(model, to_file='model.png', show_shapes=True)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# project_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/squad")
# ds = tfds.load('squad', split='train', shuffle_files=False, data_dir=project_dir_path)
# assert isinstance(ds, tf.data.Dataset)
# # source_path = "./dataset/squad/downloads/squad-example.json"
# source_path = "./dataset/squad/downloads/rajpurkar_SQuAD-explorer_train-v1.1NSdmOYa4KVr09_zf8bof8_ctB9YaIPSHyyOKbvkv2VU.json"
# data = json.load(open(source_path, 'r'))
# print(data['data'][0].keys())
# print(data['data'][0]['paragraphs'][0].keys())
# print(data['data'][0]['paragraphs'][0]['qas'])

# ---

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def main(epochs, batch_size, dataset_dim, load_model, verbose, use_char_emb, use_word_emb, q2c_attention, c2q_attention, dynamic_attention):
    # gpu settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # print("q2c_attention: {},  use char emb: {}".format(q2c_attention, use_char_emb))

    # reproducibility
    np.random.seed(0)
    tf.random.set_seed(0)

    # local dataset locations
    train_source_path = "./dataset/squad/downloads/rajpurkar_SQuAD-explorer_train-v1.1NSdmOYa4KVr09_zf8bof8_ctB9YaIPSHyyOKbvkv2VU.json"
    # source_path = "./DATA/squad/train-v1.1.json"
    # source_path = "./DATA/squad/dev-v1.1.json"
    dev_source_path = "./dataset/squad/downloads/rajpurkar_SQuAD-explorer_dev-v1.1lapqUtXWpzVWM2Z1PKUEkqZYAx2nTzAaxSOLA5Zpcsk.json"
    # example_source_path = "./dataset/squad/downloads/squad-example.json"

    # get squad dataset
    """ training set """
    data, c_words, c_chars, q_words, q_chars, answer_start_end_idx, vocab_size_t, _ = read_squad_data_v2(train_source_path)

    # remove corrupted elements -> FIXME
    # c_words.pop(400), c_chars.pop(400), q_words.pop(400), q_chars.pop(400), answer_start_end_idx.pop(400)
    # corrupted_el_idx = [12732, 12726, 12685, 12054, 10559, 10441, 9058, 8011, 6640, 5797, 5172, 4648, 3581, 3578, 3147, 3049, 2891, 2853, 676, 660]
    # corrupted_el_idx.sort(reverse=True)
    # for i in corrupted_el_idx:
    #     c_words.pop(i), c_chars.pop(i), q_words.pop(i), q_chars.pop(i), answer_start_end_idx.pop(i)
    # shuffle data (random state = 0 for reproducibility)
    # c_words, c_chars, q_words, q_chars, answer_start_end_idx = shuffle(c_words, c_chars, q_words, q_chars, answer_start_end_idx, random_state=0)

    # get a portion of dataset
    if dataset_dim != 0:
        c_words, c_chars, q_words, q_chars, answer_start_end_idx = c_words[:dataset_dim], c_chars[:dataset_dim], q_words[:dataset_dim], q_chars[:dataset_dim], answer_start_end_idx[:dataset_dim]

    cw_train, cc_train, qw_train, qc_train, y_train = c_words, c_chars, q_words, q_chars, answer_start_end_idx
    # cw_train, cc_train, qw_train, qc_train, y_train = cw_train[659:], cc_train[659:], qw_train[659:], qc_train[659:], y_train[659:]

    """ validation set """
    data, c_words, c_chars, q_words, q_chars, answer_start_end_idx, vocab_size_v, _ = read_squad_data_v2(dev_source_path)
    # c_words.pop(291), c_chars.pop(291), q_words.pop(291), q_chars.pop(291), answer_start_end_idx.pop(291)
    cw_val, cc_val, qw_val, qc_val, y_val = c_words, c_chars, q_words, q_chars, answer_start_end_idx


    X_train, y_train = (cw_train, cc_train, qw_train, qc_train), y_train
    X_val, y_val = (cw_val, cc_val, qw_val, qc_val), y_val
    vocab_size = max(vocab_size_t, vocab_size_v)
    print("vocab_size {}, dataset dimension -> train {}: ; validation: {}".format(vocab_size, len(y_train), len(y_val)))
    model = BiDafModel(vocab_size)

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
    parser.add_argument("-e", "--epochs", help="number of epochs for training", default=50, type=int)
    # batch_size
    parser.add_argument("-b", "--batch_size", help="batch dimension", default=3, type=int)
    # dataset dimension
    parser.add_argument("-d", "--dataset-dim", help="dimension of the dataset for training ", default=0, type=int)

    # save model path
    parser.add_argument("-cp", "--checkpoint-path", help="path where to save the weights", default=None)

    # verbose
    parser.add_argument("-v", "--verbose", help="add comments", default=False, type=bool)

    # Ablations
    parser.add_argument("-cemb", "--use_char_emb", help="use char embedding?", default=True, type=bool)
    parser.add_argument("-wemb", "--use_word_emb", help="use word embedding?", default=True, type=bool)
    parser.add_argument("-q2c_att", "--q2c-attention", help="use question-to-context attention?", default=True, type=bool)
    parser.add_argument("-c2q_att", "--c2q-attention", help="use context-to-question attention?", default=True, type=bool)
    parser.add_argument("-dyn_att", "--dynamic_attention", help="use dynamic attention?", default=False, type=bool)

    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.dataset_dim, args.checkpoint_path, args.verbose, args.use_char_emb, args.use_word_emb, args.q2c_attention, args.c2q_attention, args.dynamic_attention)

