#encoding=utf8

import numpy as np
import random

import jieba
jieba.dt.tmp_dir = './'
jieba.initialize()

class Vocabulary(object):
    def __init__(self, max_lines=10000, min_count=10):
        self.max_lines = max_lines
        self.min_count = min_count
        self.line_counter = 0
        self.closed = False # if closed, should stop updating
        self.special_words = [u"PAD", u"START", u"EOS", u"UNK"]
        self.words = []
        self.word2id = {self.words[i]:i for i in range(len(self.words))}
        self.wordcount = {k:1 for k, _ in self.word2id.iteritems()}
        
    def tokenize(self, sentence):
        return jieba.cut(sentence, cut_all=False) # using jieba cut here
    
    def update(self, sentence):
        assert not self.closed, "already closed, should stop update~"
        for word in self.tokenize(sentence):
            if self.word2id.has_key(word):
                self.wordcount[word] += 1
            else:
                self.words.append(word)
                self.word2id[word] = len(self.words) - 1
                self.wordcount[word] = 1
    
    def build_from_file(self, filepath):
        # support filepaths ? [filename1, filename2,...]
        for line in open(filepath, 'r'):
            self.update(line)
            self.line_counter += 1
            if (self.line_counter > self.max_lines):
                break
        self.shrink()
        self.close()
                
    def shrink(self):
        # shrink the dict by min_count
        self.words = [word for word, freq in self.wordcount.iteritems() 
                if freq >= self.min_count]
        self.word2id = {self.words[i]:i for i in range(len(self.words))}
    
    def close(self):
        # you should stop update vocabulary after calling this function
        self.closed = True
        self.words = self.special_words + self.words # merge special words
        self.word2id = {self.words[i]:i for i in range(len(self.words))}
    
    def sentence_to_word_ids(self, sentence):
        assert self.closed, 'should close before use this vocabulary'
        # it is ugly, but works...
        return map(self.word2id.get, 
                   [word if self.word2id.has_key(word) else u"UNK" 
                       for word in self.tokenize(sentence)])
    
    def word_ids_to_sentence(self, word_ids):
        assert self.closed, 'should close before use this vocabulary'
        return u"".join(
            [self.words[word_id] if word_id < len(self.words) and word_id >= 0 else u"UNK" 
                for word_id in word_ids])
    
    @property
    def details(self):
        #print u"words: " + u", ".join(self.vocab[0:50]) + u"..." + u"\n"
        info = u"\n<class Vocabulary object>,  got %d words\n" %(len(self.words))
        info += u"min_count = %d, max_lines = %d\n" %(self.min_count, self.max_lines)
        info += u"words =【" + u" / ".join(self.words[0:10]) + u"】..." + u"\n"
        info += u"line counter = " + unicode(self.line_counter) + u"\n"
        return info

    def __repr__(self):
        return self.details.encode('utf-8') # repr should return str


class Dataset(object):
    def __init__(self, vocabulary, filepath, name='dataset', batch_size=3):
        self.filepath = filepath
        self.name = name
        self.batch_size = batch_size
        self.dataset = np.array([map(vocabulary.sentence_to_word_ids, line.split('\t')[0:2])
                        for line in open(filepath, 'r')])
        random.shuffle(self.dataset)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        assert index >= 0 and index < self.__len__(), 'invalid index'
        return self.dataset[index]
    
    def __iter__(self):
        index = 0
        while index < self.__len__():
            yield self.__getitem__(index)
            index += 1
    
    def next_batch(self):
        indices = [random.randint(0, self.__len__() - 1) for i in range(self.batch_size)]
        return self.dataset[indices]

    @property
    def details(self):
        info = '\n<class Dataset object>, \n'
        info += 'size = %d, batch_size = %d,\nfilepath = %s\n' %(
                self.__len__(), self.batch_size, self.filepath)
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

    dataset = Dataset(vocab, 'small.txt')
    print dataset, '\n'
    
    print type(dataset[0])
    print dataset[0], '\n'
    
    print type(dataset.next_batch())
    print dataset.next_batch()


if __name__ == '__main__':
    test_dataset()
    #test_vocab()
