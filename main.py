import argparse
import json
import os
import datetime
import tensorflow as tf
from sklearn.utils import shuffle
# import tensorflow_datasets as tfds
from preprocessing import read_squad_data, normalize_text, get_words_tokens, word2vec
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


def main(epochs, batch_size, dataset_dim, load_model, verbose):
    # gpu settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    # reproducibility
    np.random.seed(0)
    tf.random.set_seed(0)

    # local dataset locations
    source_path = "./dataset/squad/downloads/rajpurkar_SQuAD-explorer_train-v1.1NSdmOYa4KVr09_zf8bof8_ctB9YaIPSHyyOKbvkv2VU.json"
    # source_path = "./DATA/squad/train-v1.1.json"
    example_source_path = "./dataset/squad/downloads/squad-example.json"

    # get squad dataset
    data, c_words, c_chars, q_words, q_chars, answer_start_end_idx, vocab_size = read_squad_data(source_path)

    # remove corrupted elements -> FIXME
    c_words.pop(676), c_words.pop(660), c_chars.pop(676), c_chars.pop(660), q_words.pop(676), q_words.pop(660), q_chars.pop(676), q_chars.pop(660) ,answer_start_end_idx.pop(676), answer_start_end_idx.pop(660)
    # shuffle data (random state = 0 for reproducibility)    
    # c_words, c_chars, q_words, q_chars, answer_start_end_idx = shuffle(c_words, c_chars, q_words, q_chars, answer_start_end_idx, random_state=0)

    # get a portion of dataset
    if dataset_dim != 0:
        c_words, c_chars, q_words, q_chars, answer_start_end_idx = c_words[:dataset_dim], c_chars[:dataset_dim], q_words[:dataset_dim], q_chars[:dataset_dim], answer_start_end_idx[:dataset_dim]

    # split dataset train/val
    end_train_idx = len(c_words) * 71 // 100  # 70-30
    cw_train, cc_train, qw_train, qc_train, y_train = c_words[:end_train_idx], c_chars[:end_train_idx], q_words[
                                                                                                        :end_train_idx], q_chars[
                                                                                                                         :end_train_idx], answer_start_end_idx[
                                                                                                                                          :end_train_idx]
    
    cw_val, cc_val, qw_val, qc_val, y_val = c_words[end_train_idx:], c_chars[end_train_idx:], q_words[
                                                                                              end_train_idx:], q_chars[
                                                                                                               end_train_idx:], answer_start_end_idx[
                                                                                                                                end_train_idx:]

    X_train, y_train = (cw_train, cc_train, qw_train, qc_train), y_train
    X_val, y_val = (cw_val, cc_val, qw_val, qc_val), y_val
    print("vocab_size {}, dataset dimension: {}-> train {}: ; validation: {}".format(vocab_size, len(c_words), len(y_train), len(y_val)))
    model = BiDafModel(vocab_size)

    model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, training=True, verbose=verbose)


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
    parser.add_argument("-b", "--batch_size", help="batch dimension", default=5, type=int)
    # dataset dimension
    parser.add_argument("-d", "--dataset-dim", help="dimension of the dataset for training ", default=0, type=int)

    # save model path
    parser.add_argument("-cp", "--checkpoint-path", help="path where to save the weights", default=None)

    # verbose
    parser.add_argument("-v", "--verbose", help="add comments", default=False, type=bool)

    args = parser.parse_args()
    main(args.epochs, args.batch_size, args.dataset_dim, args.checkpoint_path, args.verbose)

