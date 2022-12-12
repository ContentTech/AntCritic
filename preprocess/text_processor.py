import copy
from collections import Counter

import nltk
import numpy as np

import re
import torch
import codecs

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
END_TOKEN = '<eos>'
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, a):
        if isinstance(a, int):
            return self.idx2word[a]
        elif isinstance(a, list):
            return [self.idx2word[x] for x in a]
        elif isinstance(a, str):
            return self.word2idx[a]
        else:
            raise TypeError("Query word/index argument must be int or str")

    def __contains__(self, word):
        return word in self.word2idx


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def set_max_len(self, value):
        self.max_len = value

    def load_file(self, filename):
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                self.add_to_corpus(line)
        self.dictionary.add_word(UNK_TOKEN)
        self.dictionary.add_word(PAD_TOKEN)

    def add_to_corpus(self, line):
        """Tokenizes a text line."""
        # Add words to the dictionary
        words = line.split()
        # tokens = len(words)
        for word in words:
            word = word.lower()
            self.dictionary.add_word(word)

    def tokenize(self, line, max_len=20):
        # Tokenize line contents
        words = SENTENCE_SPLIT_REGEX.split(line.strip())
        # words = [w.lower() for w in words if len(w) > 0]
        words = [w.lower() for w in words if (len(w) > 0 and w != ' ')]  # do not include space as a token

        if words[-1] == '.':
            words = words[:-1]

        if max_len > 0:
            if len(words) > max_len:
                words = words[:max_len]
            elif len(words) < max_len:
                # words = [PAD_TOKEN] * (max_len - len(words)) + words
                words = words + [END_TOKEN] + [PAD_TOKEN] * (max_len - len(words) - 1)

        tokens = len(words)  # for end token
        ids = torch.LongTensor(tokens)
        token = 0
        for word in words:
            if word not in self.dictionary:
                word = UNK_TOKEN
            # print(word, type(word), word.encode('ascii','ignore').decode('ascii'), type(word.encode('ascii',
            # 'ignore').decode('ascii')))
            if type(word) != type('a'):
                print(word, type(word), word.encode('ascii', 'ignore').decode('ascii'),
                      type(word.encode('ascii', 'ignore').decode('ascii')))
                word = word.encode('ascii', 'ignore').decode('ascii')
            ids[token] = self.dictionary[word]
            token += 1
        # ids[token] = decoder_layer.dictionary[END_TOKEN]
        return ids

    def __len__(self):
        return len(self.dictionary)


def tokenize(sentence, use_prototype=False, word2vec=None):
    if type(sentence) is not str:
        return []
    punctuations = ['.', '?', ',', '', '(', ')', '!', ':', '…']
    raw_text = sentence.lower()
    words = nltk.word_tokenize(raw_text)
    if use_prototype:
        words = [nltk.WordNetLemmatizer().lemmatize(word) for word in words if word not in punctuations]
    else:
        words = [word for word in words if word not in punctuations]
    if word2vec is None:
        return words
    return [word for word in words if word in word2vec]


def is_noun(word):
    word_tuple = nltk.pos_tag(word)
    if word_tuple[1] in {'NN', 'NNS', 'NNP', 'NNPS'}:
        return True
    return False


def is_predicate(word):
    word_tuple = nltk.pos_tag(word)
    if word_tuple[1] in {'VB', 'VBD'}:
        return True
    return False


def get_stem(word):
    return nltk.PorterStemmer().stem_word(word)


class Vocabulary(object):
    def __init__(self, sentences, vocab_size=10000):
        super(Vocabulary, self).__init__()
        self.word2ind = {
            '<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '<MASK>': 4
        }
        self.ind2word = {
            0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>', 4: '<MASK>'
        }
        word_list = []
        for sentence in sentences:
            word_list.extend(tokenize(sentence))
        vocab_counter = Counter(word_list)
        vocab_count = 5
        vocab_list = vocab_counter.most_common(vocab_size - vocab_count)
        for vocab in vocab_list:
            self.word2ind[vocab[0]] = vocab_count
            self.ind2word[vocab_count] = vocab[0]
            vocab_count += 1
        self.max_word_id = max(list(self.ind2word.keys()))
        self.word_num = self.max_word_id + 1

    def digitize(self, sentences):
        if type(sentences[0]) is str:
            sentence = sentences
            return [self.stoi(word) for word in tokenize(sentence)]
        return [[self.stoi(word) for word in tokenize(sentence)] for sentence in sentences]

    def pad(self, digitized_sentences, max_len):
        if type(digitized_sentences[0]) is int:
            sentence = digitized_sentences
            return sentence + [self.stoi("<PAD>")] * (max_len - len(sentence))
        return [(sentence + [self.stoi("<PAD>")] * (max_len - len(sentence))) for sentence in digitized_sentences]

    def score2sentence(self, score):
        # score in shape (max_len, vocab_size)
        _, words_idx = torch.max(score, dim=-1)
        sentence = []
        for word_idx in words_idx:
            sentence.append(self.itos(int(word_idx)))
        return ' '.join(sentence)

    def id2sentence(self, id):
        sentence = []
        for word_idx in id:
            sentence.append(self.itos(int(word_idx)))
        return ' '.join(sentence)

    def stoi(self, word):
        if word in self.word2ind:
            return self.word2ind[word]
        else:
            return self.word2ind['<UNK>']

    def itos(self, index):
        if index in self.ind2word:
            return self.ind2word[index]
        else:
            return '<UNK>'

    def itoa(self, index):
        if index > self.max_word_id:
            return self.itoa(self.stoi('<UNK>'))
        one_hot = np.zeros(self.max_word_id)
        one_hot[index] = 1
        return one_hot

    def stoa(self, word):
        return self.itoa(self.stoi(word))

    @property
    def MASK(self):
        return self.stoi('<MASK>')

    @property
    def PAD(self):
        return self.stoi('<PAD>')

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    line = input_line  # reader.readline()
    line = line.strip()
    text_a = line
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=None))
    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


# Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


def bert_input2output(word_id):
    result = copy.deepcopy(word_id)
    result[result == 101] = 1
    result[result == 102] = 2
    return result
