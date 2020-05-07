import os
import json
import tensorflow as tf
import gensim
import numpy as np
import subprocess
import nltk

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}


# le 2 righe sotto servono la prima volta altrimenti il tokenizer non funziona
# nltk.download('punkt')
# SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')


# java PTB tokenizer

# project_dir_path = os.path.dirname(
#     os.path.abspath(__file__))  # /home/francesca/PycharmProjects/NeuralNet_for_MachineComprehension
#
# PTB_tokenizer_path = 'stanford-parser-full-2018-10-17/stanford-parser.jar'
# qa_dir_path = "dataset/triviaqa-rc/qa"
# file_name = "sample.txt"

# batcmd = "java -cp " + os.path.join(project_dir_path,
#                                     PTB_tokenizer_path) + " edu.stanford.nlp.process.PTBTokenizer " + os.path.join(
#     project_dir_path, qa_dir_path, file_name)
# result = subprocess.check_output(batcmd, shell=True, text=True)
# # result1 = subprocess.check_output([batcmd], stderr=subprocess.STDOUT, text=True)
# print(result)
# print(type(result))

def normalize_text(text):
    return text.lower()


def get_words_tokens(text):
    words = nltk.word_tokenize(text)  # PTB tokenizer
    sentences = nltk.sent_tokenize(text)
    word_index = nltk.word_index
    return words, sentences


def char2vec(words):
    # def char2vec(text):  # which version is better ?
    chars = [list(c) for c in words]
    char_dict = create_char_dict()
    # create characters vector from char_dictionary (if key not in dict set len dict value)
    c_vect = [[char_dict[c] if c in char_dict else len(char_dict) for c in char] for char in chars]
    # pad sequences and convert to tf Tensor
    c_vect = tf.keras.preprocessing.sequence.pad_sequences(c_vect, padding='post', maxlen=None)  # TODO maxlen must be fixed?
    c_vec = tf.convert_to_tensor(c_vect, np.float32)
    return c_vec


def create_char_dict():
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1
    return char_dict


def create_glove_matrix(glove_filename):
    assert os.path.isfile(os.path.join('glove', glove_filename)), 'Invalid glove filename'
    # create dictionary for word embedding
    embeddings_dict = {}
    with open(os.path.join('glove', glove_filename), 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    f.close()
    # print('Loaded {} word vectors.'.format(len(embeddings_dict)))
    return embeddings_dict


def word2vec(word, glove_dict):
    return glove_dict.get(word)

# text = "Hello, my name is Francesca and I came from Florence. I don't like vegetables. Bye bye!!"
# text = normalize_text(text)
# words, _ = get_words_tokens(text)
# print(words)
# chars = char2vec(words)
# print(chars)


emb_dict = create_glove_matrix('glove.6B.50d.txt')
print(emb_dict['hello'])
print(word2vec('hello', emb_dict))
