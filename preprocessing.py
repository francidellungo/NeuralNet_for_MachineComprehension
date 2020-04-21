import os
import json
import tensorflow as tf
import gensim
import numpy as np
from gensim.models import word2vec
import logging

dataset_dir = './dataset/triviaqa-rc'
qa_dir = 'qa'
file_example = 'web-example.json'

w2v_file = 'GoogleNews-vectors-negative300.bin'


def word_2_vec(model, word):
    assert type(word) == str, "word must be a string"
    try:
        vec = model[word]
    except KeyError:
        # word not in vocabulary
        print('{} not in vocabulary'.format(word))
        # vec = 0
        vec = (np.random.rand(300) - 0.5) ** 3  # get vector with random numbers

    return vec


def read_data():
    # path = os.path.join(dataset_dir, qa_dir, file_example)
    # with open(path, 'r') as f:
    #     data = json.load(f)
    #
    # print(data['Data'][0]['Answer']['NormalizedAliases'])
    # print('\n')
    # print(data['Data'][0]['Question'])
    #
    # sequences = [['io', 'sono', 'francesca'], ['ciao', ' mi ', ' chiamo', 'francesca']]
    # # print(sequences)
    #
    # question = tf.keras.preprocessing.text.text_to_word_sequence(
    #     data['Data'][0]['Question'], filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '
    # )
    #
    # print(question)

    model = gensim.models.KeyedVectors.load_word2vec_format('./dataset/GoogleNews-vectors-negative300.bin', binary=True)
    for i, word in enumerate(model.vocab):
        if i == 10:
            break
        print(word, len(model[str(word)]))
    king = word_2_vec(model, 'king')
    cici = word_2_vec(model, 'cicicic')
    print(king, '\n', cici)
    print('\n ', len(cici) == len(king))


read_data()
