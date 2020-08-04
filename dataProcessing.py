import numpy as np
from gensim.models import KeyedVectors
import pandas as pd
from multiprocessing import Pool
import jieba


class dataset(object):
    def __init__(self, s1, s2, label):
        self.index_in_epoch = 0
        self.s1 = s1
        self.s2 = s2
        self.label = label
        self.example_nums = len(label)
        self.epochs_completed = 0

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > self.example_nums:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.example_nums)
            np.random.shuffle(perm)
            self.s1 = self.s1[perm]
            self.s2 = self.s2[perm]
            self.label = self.label[perm]
            # Start next epoch
            start = 0
            self.index_in_epoch = batch_size
            assert batch_size <= self.example_nums
        end = self.index_in_epoch
        return np.array(self.s1[start:end]), np.array(self.s2[start:end]), np.array(self.label[start:end])


def build_wiki_dict():
    # # load word vectors
    # modelPath = 'chinese_wiki_embeding8000.txt'
    # model = word2vec.Word2Vec.load(modelPath)
    # # get word dict in the model
    # vocab = model.wv.vocab
    # # assign each word in the dict a id: '我'-1
    # word2id = pd.Series(range(1, len(vocab) + 1), index=vocab)
    # # for words not in 'vocab', name it '<unk>', its id is 0
    # word2id['<unk>'] = 0
    # # get word vectors in the model
    # wordEmbedding = model.wv.vectors
    # # get a mean word vector for all words
    # wordMean = np.mean(wordEmbedding, axis=0)
    # # for words not in 'vocab', use mean vector as their word vectors
    # wordEmbedding = np.vstack((wordMean, wordEmbedding))
    # # return words and their index, and word vectors
    # return word2id, wordEmbedding
    modelPath = 'sgns.wiki.bigram-char'
    # modelPath = 'chinese_wiki_embeding8000.txt'
    model = KeyedVectors.load_word2vec_format(modelPath)
    # model = KeyedVectors.load_word2vec_format(modelPath, encoding='utf-8')
    vocab = model.vocab
    word2id = pd.Series(range(1, len(vocab) + 1), index=vocab)
    word2id['<unk>'] = 0
    wordVectors = model.vectors
    meanVector = np.mean(wordVectors, axis=0)
    wordVectors = np.vstack([meanVector, wordVectors])
    print('word index and word vectors done')
    return word2id, wordVectors


def readData(path):
    # path = 'train_pointwise'
    data = pd.read_csv(path, sep='\t', names=['s1', 's2', 'similarity'], quoting=3)
    punc = "，。、【 】:“”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥"
    # data['s1'].replace(r'\s', '', regex=True, inplace=True)
    # data['s2'].replace(r'\s', '', regex=True, inplace=True)
    s1 = data['s1'].values
    s2 = data['s2'].values
    similarity = np.asarray(list(map(float, data['similarity'].values)), dtype=np.float32)
    sentenceNum = len(similarity)

    global word2id, wordEmbedding
    word2id, wordEmbedding = build_wiki_dict()

    # use multiprocessing to turn word to id
    p = Pool()
    s1 = np.asarray(list(p.map(sentence2id, s1)))
    s2 = np.asarray(list(p.map(sentence2id, s2)))
    p.close()
    p.join()

    # pad and shuffle sentences
    for i in range(sentenceNum):
        s1[i] = list(s1[i])
        s2[i] = list(s2[i])
    s1, s2 = sentencePadding(s1, s2)
    shuffleIndex = np.random.permutation(sentenceNum)
    s1 = s1[shuffleIndex]
    s2 = s2[shuffleIndex]
    similarity = similarity[shuffleIndex]

    # print(s1)
    # print(s2)
    # print(similarity)
    return s1, s2, similarity


# turn each word to their id
def getID(word):
    if word in word2id:
        return word2id[word]
    else:
        return word2id['<unk>']


# turn sentences made of words to sentences made of ids
def sentence2id(sen):
    sen = sen.split()
    sentence2id = map(getID, sen)
    return sentence2id


# find the max length of all the sentences
# sentences smaller than this length will be padded by <unk>
def sentencePadding(s1, s2):
    # s1MaxLen = max([len(sentence) for sentence in s1])
    # s2MaxLen = max([len(sentence) for sentence in s2])
    # maxLen = max(s1MaxLen, s2MaxLen)
    maxLen = 27
    sentenceNum = s1.shape[0]
    s1Padding = np.zeros((sentenceNum, maxLen))
    s2Padding = np.zeros((sentenceNum, maxLen))
    for index, sentence in enumerate(s1):
        s1Padding[index][:len(sentence)] = sentence
    for index, sentence in enumerate(s2):
        s2Padding[index][:len(sentence)] = sentence
    print('sentence padding done')
    return s1Padding, s2Padding


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    # Generates a batch iterator for a dataset
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# global word2id, wordEmbedding

# global word2id_dict, wordEmbedding_dict
# word2id_dict = {}
# wordEmbedding_dict = {}
# for word in word2id:
#     word2id_dict[word] = word2id[word]
# for i in range(len(word2id)):
#     wordEmbedding_dict[i] = wordEmbedding[i]
if __name__ == '__main__':
    # a = build_wiki_dict()
    # print(a)
    readData('train_pointwise')
    # print(word2id[1000:2000])