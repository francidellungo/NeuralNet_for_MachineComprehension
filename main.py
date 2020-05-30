import json
import os

import tensorflow_datasets as tfds
from preprocessing import read_squad_data, normalize_text, get_words_tokens, char2vec, word2vec
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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

project_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/squad")
ds = tfds.load('squad', split='train', shuffle_files=False, data_dir=project_dir_path)
assert isinstance(ds, tf.data.Dataset)
# # source_path = "./dataset/squad/downloads/squad-example.json"
# source_path = "./dataset/squad/downloads/rajpurkar_SQuAD-explorer_train-v1.1NSdmOYa4KVr09_zf8bof8_ctB9YaIPSHyyOKbvkv2VU.json"
# data = json.load(open(source_path, 'r'))
# print(data['data'][0].keys())
# print(data['data'][0]['paragraphs'][0].keys())
# print(data['data'][0]['paragraphs'][0]['qas'])

# ---
source_path = "./dataset/squad/downloads/rajpurkar_SQuAD-explorer_train-v1.1NSdmOYa4KVr09_zf8bof8_ctB9YaIPSHyyOKbvkv2VU.json"
example_source_path = "./dataset/squad/downloads/squad-example.json"

data, c_words, c_chars, q_words, q_chars = read_squad_data(example_source_path)

# c_words = tf.ragged.constant(c_words)
# c_chars = tf.ragged.constant(c_chars)
# # print('len c_words: ', c_words.shape, len(c_words), len(c_words[0]), len(c_words[1]))
x = [d[0] for d in data]
y = [d[1] for d in data]


#
# x = np.array(x)
# print(x.shape)

# -----

# print('x[0][0] (context_word_vec): ', len(x[0][0]), len(x[0][1]), len(x[0][2]), len(x[0][3]))
#
#
# print(x[0][1])
# for i in x[0][1]:
#     print(len(i))

# text = "Hello, my name is Francesca and I came from Florence. I don't like vegetables. Bye bye!!"
# text = normalize_text(text)
# words = get_words_tokens(text)
# glove_matrix = create_glove_matrix('glove.6B.100d.txt')
# word_vec = [word2vec(word, glove_matrix) for word in words]
# chars = char2vec(words)

# words = np.array(word_vec)
# chars = np.array(chars)

model = BiDafModel(80)
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(x[0])

model.train(c_words, c_chars, q_words, q_chars, y, 10, 4)
