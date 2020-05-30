import os
import json
import tensorflow as tf
import gensim
import numpy as np
import subprocess
import nltk
import re
from tqdm import tqdm

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
    return text.lower()


def get_words_tokens(text):
    text = normalize_text(text)
    words = nltk.word_tokenize(text)  # PTB tokenizer
    # sentences = nltk.sent_tokenize(text)
    return words  # , sentences


def char2vec(words):
    # def char2vec(text):  # which version is better ?
    chars = [list(c) for c in words]
    char_dict = create_char_dict()  # TODO optimize it-> create only once
    # create characters vector from char_dictionary (if key not in dict set len dict value)
    c_vect = [[char_dict[c] if c in char_dict else len(char_dict) for c in char] for char in chars]
    # pad sequences and convert to tf Tensor
    c_vect = tf.keras.preprocessing.sequence.pad_sequences(c_vect, padding='post',
                                                           maxlen=None)  # TODO maxlen must be fixed?
    # c_vec = tf.convert_to_tensor(c_vect, np.float32)
    return c_vect


def create_char_dict():
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1
    return char_dict


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
        # TODO check if ok: generate random vector if word not in glove dict
        vec = np.random.rand(100)
    return vec


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
    for articles_id in tqdm(range(len(dataset)), desc='Preprocessing squad dataset'):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for par_id in range(len(article_paragraphs)):
            context = article_paragraphs[par_id]['context']
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')
            context_tokens = get_words_tokens(context)
            # print('context_tokens: ', len(context_tokens))
            # TODO: answer_map = token_idx_map(context, context_tokens)
            context_word_vec = [word2vec(word, glove_matrix) for word in context_tokens]
            context_char_vec = char2vec(context_tokens)

            # context_words.append(context_word_vec)
            # context_chars.append(context_char_vec)

            qas = article_paragraphs[par_id]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = get_words_tokens(question)
                question_word_vec = [word2vec(word, glove_matrix) for word in question_tokens]
                question_char_vec = char2vec(question_tokens)
                # print('question_word_vec: ', len(question_word_vec), len(question_word_vec[0]))
                query_words.append(question_word_vec)
                query_chars.append(question_char_vec)

                ans_id = 0
                answer = qas[qid]['answers'][ans_id]['text']
                answer_start = qas[qid]['answers'][ans_id]['answer_start']
                answer_end = answer_start + len(answer)
                # answer_tokens = get_words_tokens(answer)
                # last_word_answer = len(answer_tokens[-1])
                # a_start_idx = answer_map[answer_start][1]
                # a_end_idx = answer_map[answer_end - last_word_answer][1]

                context_words.append(context_word_vec)
                context_chars.append(context_char_vec)
                if qid == 3:
                    print("  W ")

                examples.append(([context_word_vec, context_char_vec, question_word_vec, question_char_vec], [answer_start, answer_end]))
    return examples, context_words, context_chars, query_words, query_chars

# ------------------------------------------------------------------------------------------ ##


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


def token_idx_map(context, context_tokens):
    """
    Create dictionary with index start of each word in the context
    :param context: sentence eg. 'hello my name is'
    :param context_tokens: tokens of context sentence eg. ['hello', 'my', 'name', 'is']
    :return: dictionary {token_start_idx_in_context: [token, token_idx], ...}
            eg. {0: ['hello', 0], 6: ['my', 1], 9: ['name', 2], 14: ['is', 3]}
    """
    # TODO fix
    acc = ''
    current_token_idx = 0
    token_map = dict()

    for char_idx, char in enumerate(context):
        if char != u' ':
            acc += char.lower()
            context_token = context_tokens[current_token_idx]
            if acc == context_token:
                syn_start = char_idx - len(acc) + 1
                token_map[syn_start] = [acc, current_token_idx]
                acc = ''
                current_token_idx += 1
    return token_map


def write_to_file(out_file, line):
    out_file.write(line.encode('utf8') + '\n')


def read_write_dataset(dataset, tier, prefix):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""
    qn, an = 0, 0
    skipped = 0

    with open(os.path.join(prefix, tier + '.context'), 'w') as context_file, \
            open(os.path.join(prefix, tier + '.question'), 'w') as question_file, \
            open(os.path.join(prefix, tier + '.answer'), 'w') as text_file, \
            open(os.path.join(prefix, tier + '.span'), 'w') as span_file:

        for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
            article_paragraphs = dataset['data'][articles_id]['paragraphs']
            for pid in range(len(article_paragraphs)):
                context = article_paragraphs[pid]['context']
                # The following replacements are suggested in the paper
                # BidAF (Seo et al., 2016)
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')

                context_tokens = tokenize(context)
                answer_map = token_idx_map(context, context_tokens)

                qas = article_paragraphs[pid]['qas']
                for qid in range(len(qas)):
                    question = qas[qid]['question']
                    question_tokens = tokenize(question)

                    answers = qas[qid]['answers']
                    qn += 1

                    num_answers = range(1)

                    for ans_id in num_answers:
                        # it contains answer_start, text
                        text = qas[qid]['answers'][ans_id]['text']
                        a_s = qas[qid]['answers'][ans_id]['answer_start']

                        text_tokens = tokenize(text)

                        answer_start = qas[qid]['answers'][ans_id]['answer_start']

                        answer_end = answer_start + len(text)

                        last_word_answer = len(text_tokens[-1])  # add one to get the first char

                        try:
                            a_start_idx = answer_map[answer_start][1]

                            a_end_idx = answer_map[answer_end - last_word_answer][1]

                            # remove length restraint since we deal with it later
                            context_file.write(' '.join(context_tokens) + '\n')
                            question_file.write(' '.join(question_tokens) + '\n')
                            text_file.write(' '.join(text_tokens) + '\n')
                            span_file.write(' '.join([str(a_start_idx), str(a_end_idx)]) + '\n')

                        except Exception as e:
                            skipped += 1

                        an += 1

    print("Skipped {} question/answer pairs in {}".format(skipped, tier))
    return qn, an


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

# text = "Where in England was Dame Judi Dench born?"
# tokens = get_words_tokens(text)
# words =

# source = "dataset/qa/web-example.json"
# evidence = "dataset/evidence"
# data = read_data(source, evidence)
# print(len(data), len(data[0]), len(data[0][0]), len(data[0][0][0]), len(data[0][0][0][0]))
# print(len(data), len(data[0]), len(data[0][0]), len(data[0][0][1]), len(data[0][0][1][0]))
# data = np.array(data)
# print(data.shape)
