# coding=utf8

import numpy as np
import random
from zutil.config import Config

import jieba
jieba.dt.tmp_dir = './'
jieba.initialize()


class Vocabulary(object):
    def __init__(self, max_lines=10000000, min_count=10):
        self.max_lines = max_lines
        self.min_count = min_count
        self.line_counter = 0
        self.closed = False     # if closed, should stop updating
        self.special_words = [u"PAD", u"START", u"END", u"UNK"]
        self.words = []
        self.word2id = {self.words[i]: i for i in range(len(self.words))}
        self.wordcount = {k: 1 for k, _ in self.word2id.iteritems()}

    def tokenize(self, sentence):
        return jieba.cut(sentence, cut_all=False)   # using jieba cut here

    def update(self, sentence):
        assert not self.closed, "already closed, should stop update~"
        for word in self.tokenize(sentence):
            if word in self.word2id:
                self.wordcount[word] += 1
            else:
                self.words.append(word)
                self.word2id[word] = len(self.words) - 1
                self.wordcount[word] = 1

    def build_from_file(self, filepath, verbose=True):
        # support filepaths ? [filename1, filename2,...]
        for line in open(filepath, 'r'):
            self.update(line)
            self.line_counter += 1
            if verbose and self.line_counter % 10000 == 0:
                print self.line_counter
            if (self.line_counter > self.max_lines):
                break
        self.shrink()
        self.close()

    def shrink(self):
        # shrink the dict by min_count
        self.words = [word for word, freq in self.wordcount.iteritems()
                      if freq >= self.min_count]
        self.word2id = {self.words[i]: i for i in range(len(self.words))}

    def close(self):
        # you should stop update vocabulary after calling this function
        self.closed = True
        self.words = self.special_words + self.words  # merge special words
        self.word2id = {self.words[i]: i for i in range(len(self.words))}

    def sentence_to_word_ids(self, sentence):
        assert self.closed, 'should close before use this vocabulary'
        # it is ugly, but works...
        return map(self.word2id.get,
                   [word if word in self.word2id else u"UNK"
                    for word in self.tokenize(sentence)])

    def word_ids_to_sentence(self, word_ids):
        assert self.closed, 'should be closed before using this vocabulary'
        return u"".join([self.words[word_id] if word_id < len(self.words)
                         and word_id >= 0 else u"UNK" for word_id in word_ids])

    @property
    def size(self):
        return len(self.words)

    @property
    def details(self):
        # print u"words: " + u", ".join(self.vocab[0:50]) + u"..." + u"\n"
        info = u"\n<class Vocabulary object>,got %d words\n" % (len(self.words))
        info += u"min_count = %d, max_lines = %d\n" % (self.min_count,
                                                       self.max_lines)
        info += u"words =【" + u" / ".join(self.words[0:10]) + u"】..." + u"\n"
        info += u"line counter = " + unicode(self.line_counter) + u"\n"
        return info

    def __repr__(self):
        return self.details.encode('utf-8')  # repr should return str


class BatchData(object):
    def __init__(self, vocabulary, config):
        self.vocabulary = vocabulary
        self.config = config
        # prepare for seq2seq model
        self.encoder_inputs = self._init_array(config.encoder_max_seq_len)
        self.decoder_inputs = self._init_array(config.decoder_max_seq_len)
        self.decoder_outputs = self._init_array(config.decoder_max_seq_len)
        # record actual length
        self.encoder_inputs_lengths = []
        self.decoder_inputs_lengths = []
        self.decoder_outputs_lengths = []
        # counter
        self.counter = 0

    def _init_array(self, length):
        return np.ones((self.config.batch_size, length), dtype='int32'
                       ) * self.vocabulary.word2id[u"PAD"]

    def append(self, data):
        '''
        update encoder/decoder inputs/outputs
        '''
        assert self.counter < self.config.batch_size, "out of batch_size"
        self.counter += 1
        source, target = data
        # truncate and reverse source
        source = source[0:self.config.encoder_max_seq_len]
        source.reverse()
        self.encoder_inputs_lengths.append(len(source))
        self.encoder_inputs[self.counter - 1][0:len(source)] = source

        # Add u"START" to target for input
        target = [self.vocabulary.word2id[u"START"]] + target
        target = target[0:self.config.decoder_max_seq_len]
        self.decoder_inputs_lengths.append(len(target))
        self.decoder_inputs[self.counter - 1][0:len(target)] = target

        # Remove u"START" and Add u"END" to target for output
        target = target[1:] + [self.vocabulary.word2id[u"END"]]
        self.decoder_outputs_lengths.append(len(target))
        self.decoder_outputs[self.counter - 1][0:len(target)] = target

    def shrink(self):
        # truncate encoder/decoder inputs/outputs
        encoder_max_seq_len = max(self.encoder_inputs_lengths)
        decoder_max_seq_len = max(self.decoder_inputs_lengths)
        self.encoder_inputs = self.encoder_inputs[:, 0:encoder_max_seq_len]
        self.decoder_inputs = self.decoder_inputs[:, 0:decoder_max_seq_len]
        self.decoder_outputs = self.decoder_outputs[:, 0:decoder_max_seq_len]


class Dataset(object):
    def __init__(self, vocabulary, config):
        self.vocabulary = vocabulary
        self.config = config
        self.dataset = np.array([
            map(vocabulary.sentence_to_word_ids, line.split('\t')[0:2])
            for line in open(self.config.dataset_filepath, 'r')])
        random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    @property
    def size(self):
        return self.__len__()

    def __getitem__(self, index):
        assert index >= 0 and index < self.size, 'invalid index'
        return self.dataset[index]

    def __iter__(self):
        index = 0
        while index < self.size:
            yield self.__getitem__(index)
            index += 1

    def next_batch(self):
        # padding, add EOF symbol to decoder inputs
        batchdata = BatchData(self.vocabulary, self.config)
        for i in range(self.config.batch_size):
            index = random.randint(0, self.size - 1)
            batchdata.append(self.dataset[index])
        batchdata.shrink()
        return batchdata

    @property
    def details(self):
        info = '\n<class Dataset object>, \n'
        info += 'size = %d, batch_size = %d,\nfilepath = %s\n' % (
            self.size, self.config.batch_size, self.config.dataset_filepath)
        return info

    def __repr__(self):
        return self.details


def test_vocab():
    vocab = Vocabulary()
    vocab.build_from_file('small.txt')
    print vocab

    word_ids = vocab.sentence_to_word_ids("突然宣布在香港实习？")
    print word_ids
    sentence = vocab.word_ids_to_sentence(word_ids)
    print sentence
    word_ids = vocab.sentence_to_word_ids(sentence)
    print word_ids


def test_dataset():
    vocab = Vocabulary()
    vocab.build_from_file('small.txt')
    print vocab

    config = Config('parameters.json')
    dataset = Dataset(vocab, config)
    print dataset, '\n'

    print 'dataset[0] = '
    print type(dataset[0])
    print dataset[0], '\n'

    print 'next batch: '
    print dataset.next_batch()


if __name__ == '__main__':
    test_dataset()
    # test_vocab()
