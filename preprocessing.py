import os
import json
import sys

import tensorflow as tf
# import gensim
import numpy as np
# import subprocess
import nltk
import re
from tqdm import tqdm
from utils import plotHistogramOfLengths, pad3dSequence, savePickle, loadPickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
verbose = True


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
    return text.lower().replace("''", '"').replace("``", '"')  # .replace("  ", " ").replace("-", " ")


def get_words_tokens(text):
    text = normalize_text(text)
    tokens = nltk.word_tokenize(text)  # PTB tokenizer
    tokens = [word.replace("''", '"').replace("``", '"') for word in tokens]
    # sentences = nltk.sent_tokenize(text)
    return tokens  # , sentences


def char2vec(words, chars_dict):
    chars = [list(c) for c in words]
    # create characters vector from char_dictionary (if key not in dict set len dict value)
    c_vect = [[chars_dict[c] if c in chars_dict else len(chars_dict) for c in char] for char in chars]
    # pad sequences and convert to tf Tensor
    c_vect = tf.keras.preprocessing.sequence.pad_sequences(c_vect, value=0, padding='pre',
                                                           maxlen=None)  # TODO maxlen must be fixed?
    # if words length is minor than 5-> pad
    if c_vect.shape[-1] < 5:
        # print("  ")
        c_vect = np.pad(c_vect, ((0, 0), (5 - len(c_vect[0]) % 5, 0)), 'constant', constant_values=(0))  # padding "pre"
    return c_vect


def create_char_dict():
    alphabet = "Aabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1
    return char_dict


def char2vec_v2(words):
    """ create list of vectors representing words at character level
    ( ord() function used: return an integer representing the Unicode code)"""
    chars = [list(c) for c in words]
    c_vect = [[float(ord(c)) for c in word] for word in chars]
    # TODO do it with ragged tensors instead of padding
    c_vect = tf.keras.preprocessing.sequence.pad_sequences(c_vect, padding='post', maxlen=None)
    max_value = c_vect.max()  # vocabulary size
    return c_vect, max_value


def create_glove_matrix(glove_filename='glove.6B.100d.txt'):
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
    # if word not in glove dict return None -> now generate random vector
    vec = glove_dict.get(word)
    if vec is None:
        vec = np.random.rand(100)
    return vec


def process_tokens(temp_tokens):
    tokens = []
    for token in temp_tokens:
        flag = False
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        # \u2013 is en-dash. Used for number to nubmer
        # l = ("-", "\u2212", "\u2014", "\u2013")
        # l = ("\u2013",)
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens


def read_data(source_path, evidence_path, wikipedia=True, train=True):
    # source_path = path to qa dir in dataset, wikipedia if wikipedia entity used
    if verbose: print("Preprocessing data...")
    source_data = json.load(open(source_path, 'r'))
    # data_list = []
    # data_vec = {}
    datas = []
    glove_matrix = create_glove_matrix('glove.6B.100d.txt')
    for data in source_data['Data']:
        # question
        question = data['Question']
        q_word_tokens = get_words_tokens(question)
        q_word_vec = [word2vec(word, glove_matrix) for word in q_word_tokens]
        q_char_vec = char2vec(q_word_tokens)
        # data_vec['Question'] = {'Char_emb': q_char_vec, 'Word_emb': q_word_vec}
        query = [q_word_vec, q_char_vec]
        if verbose: print(q_word_tokens)  # , len(q_word_vec[0]), len(q_char_vec[0]))

        # answer
        # TODO answer
        answer = data['Answer']['NormalizedValue']
        a_word_tokens = get_words_tokens(answer)
        a_word_vec = [word2vec(word, glove_matrix) for word in a_word_tokens]
        a_char_vec = char2vec(a_word_tokens)
        # data_vec['Answer'] = {'Char_emb': a_char_vec, 'Word_emb': a_word_vec}
        # TODO find answer in context with find() function

        # context
        contexts = []
        # for wiki_file in data['EntityPages']:
        #     contexts.append(wiki_file['Filename'])  # TODO wikipedia files

        # iterate over context files (web)
        for web_filename in data['SearchResults']:
            # web_file = json.load(open(os.path.join(evidence_path, 'web', web_filename['Filename']), 'r'))
            with open(os.path.join(evidence_path, 'web', web_filename['Filename']), 'r') as web_file:
                web_file = web_file.read()

                c_word_tokens = get_words_tokens(web_file)
                c_word_vec = [word2vec(word, glove_matrix) for word in c_word_tokens]
                c_char_vec = char2vec(c_word_tokens)
                # data_vec['Context'] = {'Char_emb': c_char_vec, 'Word_emb': c_word_vec}
                context = [c_word_vec, c_char_vec]
            contexts.append(context)
        # data_list.append(data_vec)
        datas.append([query, context])

    if verbose: print("Finish preprocessing")
    return np.array(datas)
    # return train_data


def read_squad_data(source_path):
    # if verbose: print("Preprocessing squad data...")
    dataset = json.load(open(source_path, 'r'))
    glove_matrix = create_glove_matrix('glove.6B.100d.txt')

    examples = []
    context_words = []
    context_chars = []
    query_words = []
    query_chars = []
    answer_start_end_idx = []
    skipped_count = 0
    qa_skipped = 0

    max_vocab_size = 0

    for articles_id in tqdm(range(len(dataset['data'])), desc='Preprocessing squad dataset'):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']

        for par_id in range(len(article_paragraphs)):
            context = article_paragraphs[par_id]['context']
            # context = context.replace("''", '" ')
            # context = context.replace("``", '" ')
            # context = context.replace("-", ' ')
            context = normalize_text(context)
            context_tokens = get_words_tokens(context)
            # print('context_tokens: ', len(context_tokens))
            # TODO:
            if articles_id == 21 and par_id == 69:
                print(" ")
            answer_map = get_char_word_loc_mapping(context, context_tokens)
            if answer_map is None:
                skipped_count += 1
                qa_skipped += len(article_paragraphs[par_id]['qas'])
                break
            context_word_vec = [word2vec(word, glove_matrix) for word in context_tokens]
            # context_char_vec = char2vec(context_tokens)
            context_char_vec, max_emb = char2vec_v2(context_tokens)
            if max_emb > max_vocab_size: max_vocab_size = max_emb

            qas = article_paragraphs[par_id]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = get_words_tokens(question)
                question_word_vec = [word2vec(word, glove_matrix) for word in question_tokens]
                question_char_vec, max_emb = char2vec_v2(question_tokens)
                if max_emb > max_vocab_size: max_vocab_size = max_emb
                # print('question_word_vec: ', len(question_word_vec), len(question_word_vec[0]))
                query_words.append(question_word_vec)
                query_chars.append(question_char_vec)

                ans_id = 0
                answer = qas[qid]['answers'][ans_id]['text']
                answer_start = qas[qid]['answers'][ans_id]['answer_start']
                answer_end = answer_start + len(answer)

                answer_tokens = get_words_tokens(answer)
                last_word_answer = len(answer_tokens[-1])
                a_start_idx = int(answer_map[answer_start][1])
                a_end_idx = int(answer_map[answer_end - last_word_answer][1])
                answer_start_end_idx.append([a_start_idx, a_end_idx])

                context_words.append(context_word_vec)
                context_chars.append(context_char_vec)

                examples.append(([context_word_vec, context_char_vec, question_word_vec, question_char_vec],
                                 [answer_start, answer_end]))
    print("skipped elements:", skipped_count)
    print("total qa skipped: ", qa_skipped)
    return examples, context_words, context_chars, query_words, query_chars, answer_start_end_idx, max_vocab_size + 1, skipped_count


def read_squad_data_v2(source_path):
    # if verbose: print("Preprocessing squad data...")
    dataset = json.load(open(source_path, 'r'))
    glove_matrix = create_glove_matrix('glove.6B.100d.txt')

    # examples = []
    context_words = []
    context_chars = []
    query_words = []
    query_chars = []
    answer_start_end_idx = []
    skipped_count = 0

    # max_vocab_size = 0
    chars_dict = create_char_dict()
    # error = False

    for articles_id in tqdm(range(len(dataset['data'])), desc='Preprocessing squad dataset'):
    # for articles_id in tqdm(range(4), desc='Preprocessing squad dataset'):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']

        for par_id in range(len(article_paragraphs)):
            context = article_paragraphs[par_id]['context']
            # context = context.replace("''", '" ')
            # context = context.replace("``", '" ')
            # context = context.replace("-", ' ')
            context = normalize_text(context)
            # context.replace("''", '" ').replace("``", '" ')
            context_tokens = get_words_tokens(context)
            # context_tokens = process_tokens(context_tokens)

            answer_map = get_char_word_loc_mapping(context, context_tokens)
            context_word_vec = [word2vec(word, glove_matrix) for word in context_tokens]
            # context_char_vec = char2vec(context_tokens)
            context_char_vec = char2vec(context_tokens, chars_dict)

            qas = article_paragraphs[par_id]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = get_words_tokens(question)
                question_word_vec = [word2vec(word, glove_matrix) for word in question_tokens]
                question_char_vec = char2vec(question_tokens, chars_dict)

                ans_id = 0
                answer = qas[qid]['answers'][ans_id]['text']
                answer_start = qas[qid]['answers'][ans_id]['answer_start']
                answer_end = answer_start + len(answer)

                # answer_tokens = get_words_tokens(answer)
                # last_word_answer = len(answer_tokens[-1])
                try:
                    a_start_idx = int(answer_map[answer_start][1])
                    a_end_idx = int(answer_map[answer_end - 1][1])
                except KeyError:
                    # print("answer problems: key not in dictionary")
                    skipped_count += 1
                    continue
                except TypeError:
                    skipped_count += 1
                    continue
                # a_end_idx = int(answer_map[answer_end - last_word_answer][1])

                if question_char_vec.shape[-1] < 5:  # if word
                    print('')
                    continue
                answer_start_end_idx.append([a_start_idx, a_end_idx])

                query_words.append(question_word_vec)
                query_chars.append(np.array(question_char_vec, dtype='float32'))

                context_words.append(context_word_vec)
                context_chars.append(np.array(context_char_vec, dtype='float32'))


    print("skipped elements:", skipped_count)
    max_words_context = 256
    max_chars_context = 30

    # adjust start end answer indexes
    context_words_lens = np.array([len(c) for c in context_words])  # list of lengths
    words_dist = max_words_context - context_words_lens
    words_dist = tf.keras.backend.repeat_elements(tf.convert_to_tensor(words_dist), 2, 0)
    words_dist = tf.reshape(words_dist, [-1, 2])
    new_answer_start_end_idx = (tf.constant(answer_start_end_idx, dtype='int64') + words_dist).numpy()

    # check if new answers indexes are less than zero
    minor = tf.math.less(new_answer_start_end_idx[:, 0], 0)  # check if new answ indexes are less than zero
    minor = tf.cast(minor, 'int32')
    minor = tf.squeeze(tf.where(minor)).numpy()  # get indexes of elements less than the threshold
    for idx in minor:
        start = max(0, answer_start_end_idx[idx][0] - 20)
        end = start + max_words_context
        context_words[idx] = context_words[idx][start: end]
        context_chars[idx] = context_chars[idx][start: end]
        new_answer_start_end_idx[idx] = [answer_start_end_idx[idx][0] - start, answer_start_end_idx[idx][1] - start]

    # do paddings and trunc
    pad_context_words = tf.keras.preprocessing.sequence.pad_sequences(context_words, dtype="float32",
                                                                      maxlen=max_words_context)

    context_chars = pad3dSequence(context_chars, max_words=max_words_context, chars_maxlen=max_chars_context)
    print('context_chars shape:', context_chars.shape)
    # context_chars = tf.keras.preprocessing.sequence.pad_sequences(context_chars, dtype="float32")
    query_words = tf.keras.preprocessing.sequence.pad_sequences(query_words, dtype="float32")
    # query_chars = tf.ragged.constant(query_chars, dtype='float32').to_tensor()
    query_chars = pad3dSequence(query_chars)

    # # check if answer idx is > than max_words
    # idx_to_remove = [idx for idx, el in enumerate(new_answer_start_end_idx) if
    #                  el[0] > max_words_context or el[1] > max_words_context or el[0] < 0 or el[1] < 0]
    # print('{} elements removed'.format(len(idx_to_remove)))
    # assert idx_to_remove == []
    # # idx_to_remove.reverse()
    # #
    # # pad_context_words = np.delete(pad_context_words, idx_to_remove, 0)
    # # new_answer_start_end_idx = np.delete(new_answer_start_end_idx, idx_to_remove, 0)
    # # context_chars = np.delete(context_chars, idx_to_remove, 0)
    # # query_words = np.delete(query_words, idx_to_remove, 0)
    # # query_chars = np.delete(query_chars, idx_to_remove, 0)

    return pad_context_words, context_chars, query_words, query_chars, new_answer_start_end_idx, len(
        chars_dict), skipped_count


def preprocessingSquad(source_path, save_path='./save', dataset_len=float('inf'), pre_batch_size=200,
                       is_validation_set=False):
    # TODO complete
    # check if dataset already preprocessed
    print('dataset len: ', dataset_len)
    filename = os.path.join(save_path,
                            'training_set/{}'.format(dataset_len)) if not is_validation_set else os.path.join(save_path,
                                                                                                              'validation_set/{}'.format(
                                                                                                                  dataset_len))

    if os.path.exists(filename):
        print('Dataset already preprocessed')
        # context_words = loadPickle(os.path.join(filename, 'context_words'))
        # context_chars = loadPickle(os.path.join(filename, 'context_chars'))
        # query_words = loadPickle(os.path.join(filename, 'query_words'))
        # query_chars = loadPickle(os.path.join(filename, 'query_chars'))
        # answer_start_end_idx = loadPickle(os.path.join(filename, 'answer_start_end_idx'))
        # vocab_size = loadPickle(os.path.join(filename, 'vocab_size'))
        #
        # return context_words, context_chars, query_words, query_chars, answer_start_end_idx, vocab_size, None
        context_words, context_chars, query_words, query_chars, answer_start_end_idx, vocab_size = getPreprocessedDataset(
            dataset_len, training_set=(not is_validation_set))
        return context_words, context_chars, query_words, query_chars, answer_start_end_idx, vocab_size, None

    dataset = json.load(open(source_path, 'r'))
    glove_matrix = create_glove_matrix('glove.6B.100d.txt')

    context_words, context_chars, query_words, query_chars, answers_idx, vocab_size, skipped_count, num_context_words, \
    num_query_words, context_chars_lens, query_chars_lens = [], [], [], [], [], [], [], [], [], [], []

    dataset_len = min(dataset_len, len(dataset['data']))
    num_batches = int(np.ceil(dataset_len / pre_batch_size))

    # FIXME check if it's ok
    for batch in tqdm(range(num_batches)):
        end_idx = ((batch + 1) * pre_batch_size if (batch + 1) * pre_batch_size < dataset_len else dataset_len)
        cw, cc, qw, qc, ai, vs, skipc, ncw, nqw, ccl, qcl = readSquadDataPadding(
            dataset, glove_matrix, is_validation_set=is_validation_set, article_start=batch * pre_batch_size,
            article_end=end_idx)
        context_words += cw
        context_chars += cc
        query_words += qw
        query_chars += qc
        answers_idx += ai
        vocab_size += [vs]
        skipped_count += [skipc]
        num_context_words += ncw
        num_query_words += nqw
        context_chars_lens += ccl
        query_chars_lens += qcl

    vocab_size = max(vocab_size)
    # print('sys.getsizeof(context_words): ', sys.getsizeof(context_words))

    statistics = plotHistogramOfLengths([num_context_words, num_query_words, context_chars_lens, query_chars_lens],
                                        ['num_context_words', 'num_query_words', 'context_chars_lens',
                                         'query_chars_lens'], is_validation_set)

    # pad all to create tensors FIXME
    max_words_context = int((statistics['num_context_words']['max'] + statistics['num_context_words']['mean']) / 2)
    # max_words_context = int(statistics['num_context_words']['mean'])
    max_chars_context = int((statistics['context_chars_lens']['max'] + statistics['context_chars_lens']['mean']) / 2)
    # max_chars_context = int(statistics['context_chars_lens']['mean'])
    max_words_context = 256
    print('max_words_context: ', max_words_context)
    print('max_chars_context: ', max_chars_context)

    pad_context_words = tf.keras.preprocessing.sequence.pad_sequences(context_words, dtype="float32",
                                                                      maxlen=max_words_context)

    # adjust start end answer indexes
    context_words_lens = np.array([len(c) for c in context_words])  # list of lengths
    words_dist = pad_context_words.shape[-2] - context_words_lens
    words_dist = tf.keras.backend.repeat_elements(tf.convert_to_tensor(words_dist), 2, 0)
    words_dist = tf.reshape(words_dist, [-1, 2])
    answers_idx = tf.constant(answers_idx, dtype='int64') + words_dist

    # check if answer idx is > than max_words
    idx_to_remove = [idx for idx, el in enumerate(answers_idx) if
                     el[0] > max_words_context or el[1] > max_words_context or el[0] < 0 or el[1] < 0]
    print('{} elements removed'.format(len(idx_to_remove)))
    idx_to_remove.reverse()
    pad_context_words = np.delete(pad_context_words, idx_to_remove, 0)
    answers_idx = np.delete(answers_idx, idx_to_remove, 0)
    for i in idx_to_remove:
        # pad_context_words = np.delete(pad_context_words, i, 0)
        del context_chars[i]
        del query_words[i]
        del query_chars[i]

    print('context_words shape:', pad_context_words.shape)
    # context_chars = tf.ragged.constant(context_chars, dtype='float32').to_tensor()
    context_chars = pad3dSequence(context_chars, max_words=max_words_context, chars_maxlen=max_chars_context)
    print('context_chars shape:', context_chars.shape)
    # context_chars = tf.keras.preprocessing.sequence.pad_sequences(context_chars, dtype="float32")
    query_words = tf.keras.preprocessing.sequence.pad_sequences(query_words, dtype="float32")
    # query_chars = tf.ragged.constant(query_chars, dtype='float32').to_tensor()
    query_chars = pad3dSequence(query_chars)

    print('conversion to tensors done, len context_words: {}, numbers words in each context: {}'.format(
        len(context_words), max_words_context))

    # save preprocessed data
    if verbose: print('Saving data...')
    # filename = './save/training_set/{}'.format(
    #     dataset_len) if not is_validation_set else './save/validation_set/{}'.format(dataset_len)

    if not os.path.exists(filename):
        os.makedirs(filename)

    # divide pickles to be saved
    step_save = 10000

    for id, e in enumerate(range(0, len(pad_context_words), step_save)):
        start = e
        end = min(e + step_save, len(pad_context_words))

        savePickle(os.path.join(filename, 'context_words' + '_' + str(id)), pad_context_words[start:end])
        savePickle(os.path.join(filename, 'context_chars' + '_' + str(id)), context_chars[start:end])
        savePickle(os.path.join(filename, 'query_words' + '_' + str(id)), query_words[start:end])
        savePickle(os.path.join(filename, 'query_chars' + '_' + str(id)), query_chars[start:end])
        savePickle(os.path.join(filename, 'answer_start_end_idx' + '_' + str(id)), answers_idx[start:end])

    savePickle(os.path.join(filename, 'vocab_size'), vocab_size)

    return pad_context_words, context_chars, query_words, query_chars, answers_idx, vocab_size, skipped_count


def readSquadDataPadding(dataset, glove_matrix, article_start=0, article_end=None, is_validation_set=False):
    # if verbose: print("Preprocessing squad data...")

    # f = open("dataset/dataset-qa", "a")  # file with info of qa
    dataset_info_list = []

    # examples = []
    context_words = []
    context_chars = []
    query_words = []
    query_chars = []
    answer_start_end_idx = []
    skipped_count = 0

    # create lengths statistics for padding
    num_context_words = []
    num_query_words = []
    context_chars_lens = []
    query_chars_lens = []

    chars_dict = create_char_dict()
    if article_end is None:
        article_end = len(dataset['data']) - article_start

    print(article_start, ' ', article_end)
    for articles_id in tqdm(range(article_start, article_end), desc='Preprocessing squad dataset'):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for par_id in range(len(article_paragraphs)):
            context = article_paragraphs[par_id]['context']
            # if articles_id == 105 and par_id == 24:
            #     print(" ")
            # context = context.replace("''", '" ')
            # context = context.replace("``", '" ')
            # context = context.replace("-", ' ')
            context = normalize_text(context)
            # context.replace("''", '" ').replace("``", '" ')
            context_tokens = get_words_tokens(context)
            # context_tokens = process_tokens(context_tokens)
            # context = context.lower()
            # print('context_tokens: ', len(context_tokens))

            # print("parid {} / {} ".format(par_id, len(article_paragraphs)))
            answer_map = get_char_word_loc_mapping(context, context_tokens)
            # if answer_map is not None:
            context_word_vec = [word2vec(word, glove_matrix) for word in context_tokens]
            # context_char_vec = char2vec(context_tokens)
            context_char_vec = char2vec(context_tokens, chars_dict)
            # if max_emb > max_vocab_size: max_vocab_size = max_emb

            qas = article_paragraphs[par_id]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = get_words_tokens(question)
                question_word_vec = [word2vec(word, glove_matrix) for word in question_tokens]
                question_char_vec = char2vec(question_tokens, chars_dict)

                # update file dataset-qa
                # TODO create dictionary and write to file, needed when print attention matrix to get words
                if is_validation_set:
                    dataset_info_list.append({'c': context_tokens, 'q': question_tokens})

                ans_id = 0
                answer = qas[qid]['answers'][ans_id]['text']
                answer_start = qas[qid]['answers'][ans_id]['answer_start']
                answer_end = answer_start + len(answer)

                # answer_tokens = get_words_tokens(answer)
                # last_word_answer = len(answer_tokens[-1])
                try:
                    a_start_idx = int(answer_map[answer_start][1])
                    a_end_idx = int(answer_map[answer_end - 1][1])
                except KeyError:
                    # print("answer problems: key not in dictionary")
                    skipped_count += 1
                    continue
                except TypeError:
                    # print("answer problems: answer map == None")
                    skipped_count += 1
                    continue
                # a_end_idx = int(answer_map[answer_end - last_word_answer][1])

                if question_char_vec.shape[-1] < 5:  # if word
                    continue
                answer_start_end_idx.append([a_start_idx, a_end_idx])

                # create lists of ord and chars for queries and context
                query_words.append(question_word_vec)
                query_chars.append(question_char_vec)

                context_words.append(context_word_vec)
                context_chars.append(context_char_vec)

                # lengths statistics
                num_context_words.append(len(context_word_vec))
                num_query_words.append(len(question_word_vec))
                context_chars_lens = context_chars_lens + [len(q[0]) for q in context_chars]
                query_chars_lens = query_chars_lens + [len(q[0]) for q in query_chars]

                # if len(answer_start_end_idx) >= dataset_len:
                #     return context_words, context_chars, query_words, query_chars, answer_start_end_idx, len(
                #         chars_dict), skipped_count, num_context_words, num_query_words, context_chars_lens, query_chars_lens
        # if len(answer_start_end_idx) > 500:
        #     break
        # print('Article {}, dim: {}, time: {}'.format(articles_id, len(article_paragraphs), time.time() - t1))
    # f.close()

    print("skipped elements:", skipped_count)
    return context_words, context_chars, query_words, query_chars, answer_start_end_idx, len(
        chars_dict), skipped_count, num_context_words, num_query_words, context_chars_lens, query_chars_lens

    # write validation elements to file
    # if is_validation_set:
    #     with open('data.txt', 'w') as outfile:
    #         json.dump(dataset_info_list, outfile)

    # statistics = plotHistogramOfLengths([num_context_words, num_query_words, context_chars_lens, query_chars_lens],
    #                                     ['num_context_words', 'num_query_words', 'context_chars_lens',
    #                                      'query_chars_lens'], is_validation_set)
    #
    # # pad all to create tensors FIXME
    # max_words_context = int((statistics['num_context_words']['max'] + statistics['num_context_words']['mean']) / 2)
    # max_chars_context = int((statistics['context_chars_lens']['max'] + statistics['context_chars_lens']['mean']) / 2)
    # print('max_words_context: ', max_words_context)
    # print('max_chars_context: ', max_chars_context)
    #
    # # check if answer idx is > than max_words
    # idx_to_remove = [idx for idx, el in enumerate(answer_start_end_idx) if
    #                  el[0] > max_words_context or el[1] > max_words_context]
    # for i in idx_to_remove:
    #     del context_words[i]
    #     del context_chars[i]
    #     del query_words[i]
    #     del query_chars[i]
    #     del answer_start_end_idx[i]
    #
    # context_words = tf.keras.preprocessing.sequence.pad_sequences(context_words, dtype="float32",
    #                                                               maxlen=max_words_context)
    # print('context_words shape:', context_words.shape)
    # # context_chars = tf.ragged.constant(context_chars, dtype='float32').to_tensor()
    # context_chars = pad3dSequence(context_chars, max_words=max_words_context, chars_maxlen=max_chars_context)
    # print('context_chars shape:', context_chars.shape)
    # # context_chars = tf.keras.preprocessing.sequence.pad_sequences(context_chars, dtype="float32")
    # query_words = tf.keras.preprocessing.sequence.pad_sequences(query_words, dtype="float32")
    # # query_chars = tf.ragged.constant(query_chars, dtype='float32').to_tensor()
    # query_chars = pad3dSequence(query_chars)

    # print('conversion to tensors done, len context_words: {}, numbers words in each context: {}'.format(
    #     len(context_words), max_words_context))
    #
    # # save preprocessed data
    # if verbose: print('Saving data')
    # # TODO adjust len(...)
    # filename = './save/{}/training_set'.format(len(context_words)) if not is_validation_set else './save/{}/validation_set'.format(len(context_words))
    # if not os.path.exists(filename):
    #     os.makedirs(filename)
    # savePickle(os.path.join(filename, 'context_words'), context_words)
    # savePickle(os.path.join(filename, 'context_chars'), context_chars)
    # savePickle(os.path.join(filename, 'query_words'), query_words)
    # savePickle(os.path.join(filename, 'query_chars'), query_chars)
    # savePickle(os.path.join(filename, 'answer_start_end_idx'), answer_start_end_idx)
    #
    #
    # return context_words, context_chars, query_words, query_chars, answer_start_end_idx, len(chars_dict), skipped_count


def getPreprocessedDataset(dim, training_set=True):
    path = os.path.join('save', 'training_set' if training_set else 'validation_set', str(dim))
    files = ['context_words', 'context_chars', 'query_words', 'query_chars', 'answer_start_end_idx']

    different_files = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    context_words, context_chars, query_words, query_chars, answer_start_end_idx = [], [], [], [], []
    for i in range(int(different_files / len(files))):
        context_words.append(loadPickle(os.path.join(path, 'context_words' + '_' + str(i))))
        context_chars.append(loadPickle(os.path.join(path, 'context_chars' + '_' + str(i))))
        query_words.append(loadPickle(os.path.join(path, 'query_words' + '_' + str(i))))
        query_chars.append(loadPickle(os.path.join(path, 'query_chars' + '_' + str(i))))
        answer_start_end_idx.append(loadPickle(os.path.join(path, 'answer_start_end_idx' + '_' + str(i))))

    # context_words = np.concatenate(context_words)  # tf.concat(context_words, axis=0)
    # context_chars = np.concatenate(context_chars)  # tf.concat(context_chars, axis=0)
    # query_words = np.concatenate(query_words)  # tf.concat(query_words, axis=0)
    # query_chars = np.concatenate(query_chars)  # tf.concat(query_chars, axis=0)
    # answer_start_end_idx = np.concatenate(answer_start_end_idx)  # tf.concat(answer_start_end_idx, axis=0)
    vocab_size = loadPickle(os.path.join(path, 'vocab_size'))

    context_words = tf.concat(context_words, axis=0)  # np.concatenate(context_words)
    context_chars = tf.concat(context_chars, axis=0)  # np.concatenate(context_chars)
    query_words = tf.concat(query_words, axis=0)  # np.concatenate(query_words)  #
    query_chars = tf.concat(query_chars, axis=0)  # np.concatenate(query_chars)  #
    answer_start_end_idx = tf.concat(answer_start_end_idx, axis=0)  # np.concatenate(answer_start_end_idx)  #

    return context_words, context_chars, query_words, query_chars, answer_start_end_idx, vocab_size


# ------------------------------------------------------------------------------------------ ##article_paragraphs

def list_topics(data):
    # for squad dataset
    list_topics = [data['data'][idx]['title'] for idx in range(0, len(data['data']))]
    return list_topics


def data_from_json(filename):
    """Loads JSON data from filename and returns"""
    with open(filename) as data_file:
        data = json.load(data_file)
    return data


def tokenize(sequence):
    tokens = [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(sequence)]
    return list(map(lambda x: x.encode('utf8'), tokens))


# def split_by_whitespace(sentence):
#     """
#     given a sentence return a list of words: 'hello my name is' -> ['hello', 'my', 'name', 'is']
#     :param sentence: string
#     :return: list of strings
#     """
#     words = []
#     for space_separated_fragment in sentence.strip().split():
#         words.extend(re.split(" ", space_separated_fragment))
#     return [w for w in words if w]


def get_char_word_loc_mapping(context, context_tokens):
    """
    Return a mapping that maps from character locations to the corresponding token locations.
    If we're unable to complete the mapping e.g. because of special characters, we return None.
    Inputs:
      context: string (unicode)
      context_tokens: list of strings (unicode)
    Returns:
      mapping: dictionary from ints (character locations) to (token, token_idx) pairs
        Only ints corresponding to non-space character locations are in the keys
        e.g. if context = "hello world" and context_tokens = ["hello", "world"] then
        0,1,2,3,4 are mapped to ("hello", 0) and 6,7,8,9,10 are mapped to ("world", 1)
    """
    acc = ''  # accumulator
    current_token_idx = 0  # current word loc
    mapping = dict()

    for char_idx, char in enumerate(context):  # step through original characters
        if char != u' ' and char != u'\n':  # if it's not a space:
            acc += char  # add to accumulator
            context_token = context_tokens[current_token_idx]  # current word token
            if acc == context_token:  # if the accumulator now matches the current word token
                syn_start = char_idx - len(acc) + 1  # char loc of the start of this word
                for char_loc in range(syn_start, char_idx + 1):
                    mapping[char_loc] = (acc, current_token_idx)  # add to mapping
                acc = ''  # reset accumulator
                current_token_idx += 1

    if current_token_idx != len(context_tokens):
        return None
    else:
        return mapping


# def token_idx_map(context, context_tokens):
#     """
#     Create dictionary with index start of each word in the context
#     :param context: sentence eg. 'hello my name is'
#     :param context_tokens: tokens of context sentence eg. ['hello', 'my', 'name', 'is']
#     :return: dictionary {token_start_idx_in_context: [token, token_idx], ...}
#             eg. {0: ['hello', 0], 6: ['my', 1], 9: ['name', 2], 14: ['is', 3]}
#     """
#     # TODO fix
#     acc = ''
#     current_token_idx = 0
#     token_map = dict()
#
#     for char_idx, char in enumerate(context):
#         if char != u' ':
#             # char.replace("``", '"').replace("''", '"')
#             acc += char.lower()
#             context_token = context_tokens[current_token_idx]
#             if acc == context_token:
#                 syn_start = char_idx - len(acc) + 1
#                 token_map[syn_start] = [acc, current_token_idx]
#                 acc = ''
#                 current_token_idx += 1
#     return token_map


def write_to_file(out_file, line):
    out_file.write(line.encode('utf8') + '\n')

# def read_write_dataset(dataset, tier, prefix):
#     """Reads the dataset, extracts context, question, answer,
#     and answer pointer in their own file. Returns the number
#     of questions and answers processed for the dataset"""
#     qn, an = 0, 0
#     skipped = 0
#
#     with open(os.path.join(prefix, tier + '.context'), 'w') as context_file, \
#             open(os.path.join(prefix, tier + '.question'), 'w') as question_file, \
#             open(os.path.join(prefix, tier + '.answer'), 'w') as text_file, \
#             open(os.path.join(prefix, tier + '.span'), 'w') as span_file:
#
#         for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
#             article_paragraphs = dataset['data'][articles_id]['paragraphs']
#             for pid in range(len(article_paragraphs)):
#                 context = article_paragraphs[pid]['context']
#                 # The following replacements are suggested in the paper
#                 # BidAF (Seo et al., 2016)
#                 context = context.replace("''", '" ')
#                 context = context.replace("``", '" ')
#
#                 context_tokens = tokenize(context)
#                 answer_map = token_idx_map(context, context_tokens)
#
#                 qas = article_paragraphs[pid]['qas']
#                 for qid in range(len(qas)):
#                     question = qas[qid]['question']
#                     question_tokens = tokenize(question)
#
#                     answers = qas[qid]['answers']
#                     qn += 1
#
#                     num_answers = range(1)
#
#                     for ans_id in num_answers:
#                         # it contains answer_start, text
#                         text = qas[qid]['answers'][ans_id]['text']
#                         a_s = qas[qid]['answers'][ans_id]['answer_start']
#
#                         text_tokens = tokenize(text)
#
#                         answer_start = qas[qid]['answers'][ans_id]['answer_start']
#
#                         answer_end = answer_start + len(text)
#
#                         last_word_answer = len(text_tokens[-1])  # add one to get the first char
#
#                         try:
#                             a_start_idx = answer_map[answer_start][1]
#
#                             a_end_idx = answer_map[answer_end - last_word_answer][1]
#
#                             # remove length restraint since we deal with it later
#                             context_file.write(' '.join(context_tokens) + '\n')
#                             question_file.write(' '.join(question_tokens) + '\n')
#                             text_file.write(' '.join(text_tokens) + '\n')
#                             span_file.write(' '.join([str(a_start_idx), str(a_end_idx)]) + '\n')
#
#                         except Exception as e:
#                             skipped += 1
#
#                         an += 1
#
#     print("Skipped {} question/answer pairs in {}".format(skipped, tier))
#     return qn, an


# text = "the top-five 4.5$ legend \"Venite Ad Me Omnes\" super. z."

# text2 = "jay z."
# # text= 'Hello, my name is "Francesca" and i don't like vegetables'
# # text = text.replace("-", ' ')
# textnorm = normalize_text(text)
# tokens = get_words_tokens(textnorm)
# mapidx = get_char_word_loc_mapping(textnorm, tokens)
# print(tokens, '\n', mapidx)
# print(nltk.word_tokenize(text))
#
# textnorm = normalize_text(text2)
# tokens = get_words_tokens(textnorm)
# mapidx = get_char_word_loc_mapping(textnorm, tokens)
# print(tokens, '\n', mapidx)
#
# print(nltk.word_tokenize(text2))

# source_path = "./dataset/squad/downloads/rajpurkar_SQuAD-explorer_train-v1.1NSdmOYa4KVr09_zf8bof8_ctB9YaIPSHyyOKbvkv2VU.json"
# example_source_path = "./dataset/squad/downloads/squad-example.json"
#
# x, y = read_squad_data(example_source_path)


# data_prefix = os.path.join("data", "squad")
# if not os.path.exists(data_prefix):
#     os.makedirs(data_prefix)
#
# train_num_questions, train_num_answers = read_write_dataset(data, 'train', data_prefix)
# print(train_num_questions, train_num_answers)

# qn, an = read_write_dataset()
# text = "Hello, my name is Francesca and I came from Florence. I don't like vegetables. Bye bye!!"
# text = normalize_text(text)
# words, _ = get_words_tokens(text)
# print(words)
# chars = char2vec(words)
# print(chars)


# emb_dict = create_glove_matrix('glove.6B.50d.txt')
# print(emb_dict['hello'])
# print(word2vec('hello', emb_dict))

# source_path = './dataset/qa/web-example.json'
# data_list = read_data(source_path)
# print("Where in England was Dame Judi Dench born?\n")
# print(data_list)
# print(len(data_list['Question']['Word_emb']))

# text = "Wilhelm W\u00fcrfel and jay-z Where in England was Dame Judi Dench born? Zeitung praised his \"wealth of musical ideas\ "
# tokens = get_words_tokens(text)
# print(tokens)
# print(process_tokens(tokens))
# print([process_tokens(token) for token in tokens])

# source = "dataset/qa/web-example.json"
# evidence = "dataset/evidence"
# data = read_data(source, evidence)
# print(len(data), len(data[0]), len(data[0][0]), len(data[0][0][0]), len(data[0][0][0][0]))
# print(len(data), len(data[0]), len(data[0][0]), len(data[0][0][1]), len(data[0][0][1][0]))
# data = np.array(data)
# print(data.shape)
