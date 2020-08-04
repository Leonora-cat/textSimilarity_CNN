import tensorflow as tf
import numpy as np
from dataProcessing import build_wiki_dict
tf.compat.v1.disable_eager_execution()


class TextSimilarityCNN(object):
    def __init__(self, sequenceLength, numClasses, filterSizes, numFilters, l2RegLambda=0.0):

        # sequenceLength: max len of sentences
        # numClasses: number of classification, i.e., the dimension of the output y
        # filterSizes: convolution core size
        # numFilters: number of convolution cores
        # l2RegLambda: l2 regularization coefficient

        self.input_s1 = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, sequenceLength], name='input_s1')
        self.input_s2 = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, sequenceLength], name='input_s2')
        self.input_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1], name='input_y')
        self.dropoutProb = tf.compat.v1.placeholder(dtype=tf.float32, name='dropoutProb')
        self.numClasses = numClasses
        self.numFilters = numFilters
        self.filterSizes = filterSizes
        self.sequenceLength = sequenceLength
        self.l2RegLambda = l2RegLambda
        self.l2_loss = tf.constant(0.0)

        self.initWeight()
        self.inference()
        self.dropoutLayer()
        self.outputLayer()
        self.lossAccuracy()

    def initWeight(self):
        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            word2id, self.wordEmbedding = build_wiki_dict()
            self.embeddingSize = self.wordEmbedding.shape[1]
            self.w = tf.Variable(initial_value=self.wordEmbedding, trainable=True, name='wordEmbedding', dtype=tf.float32, shape=self.wordEmbedding.shape)
            self.s1 = tf.nn.embedding_lookup(self.w, self.input_s1)
            self.s2 = tf.nn.embedding_lookup(self.w, self.input_s2)
            self.x = tf.concat([self.s1, self.s2], axis=1)
            self.x = tf.expand_dims(self.x, -1)

    def inference(self):
        # create a convolution + max pooling layer for each filter size
        poolingOutputs = []
        for index, filterSize in enumerate(self.filterSizes):
            with tf.name_scope('conv-maxpooling-'+str(filterSize)):
                # convolution layer
                # size of convolution core matrix: numFilters cores with shape filterSizw*embeddingSize, channel=1
                filterShape = [filterSize, self.embeddingSize, 1, self.numFilters]
                w = tf.Variable(tf.random.truncated_normal(filterShape, stddev=0.1), name='w')
                b = tf.Variable(tf.constant(0.1, shape=[self.numFilters]), name='b')
                # horizontal and vertical stride = 1
                conv = tf.nn.conv2d(self.x, w, strides=[1, 1, 1, 1], padding='VALID', name='conv')
                # activation function
                relu = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                # output feature map: shape=[batch,height,width,channels]

                # max pooling over outputs
                # ksize: pooling window size = [1, height, width, 1], batch+channels often=1
                pooling = tf.nn.max_pool(input=relu, ksize=[1, self.sequenceLength * 2 - filterSize + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name='pool')
                poolingOutputs.append(pooling)
                # poolingOutputs: shape=[len(filterSizes), batch, height, width, channels=1]

        # combine all the pooling features
        self.numFiltersTotal = self.numFilters * len(self.filterSizes)
        # concat the third dimension(width)
        # for each word in a sentence, concat the computation results of different cores
        self.reluPooling = tf.concat(poolingOutputs, 3)
        self.reluPoolingFlat = tf.reshape(self.reluPooling, [-1, self.numFiltersTotal])

    def dropoutLayer(self):
        # hidden layer
        # add dropout
        with tf.name_scope('dropout'):
            self.reluDrop = tf.nn.dropout(self.reluPoolingFlat, self.dropoutProb)

    def outputLayer(self):
        # final (unnormalized) scores and predictions
        with tf.name_scope('output'):
            w = tf.compat.v1.get_variable('w', shape=[self.numFiltersTotal, self.numClasses], initializer=tf.keras.initializers.GlorotUniform())
            b = tf.Variable(tf.constant(0.1, shape=[self.numClasses]), name='b')
            self.l2_loss += tf.nn.l2_loss(w)
            self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.compat.v1.nn.xw_plus_b(self.reluDrop, w, b, name='scores')

    def lossAccuracy(self):
        # calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.square(self.scores - self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2RegLambda * self.l2_loss

        # accuracy (pearson correlation coefficient)
        with tf.name_scope('pearson'):
            cov_Score_Input_y = tf.reduce_mean(self.scores * self.input_y) - tf.reduce_mean(self.scores) * tf.reduce_mean(self.input_y)
            d_Score_Input_y = tf.sqrt(tf.reduce_mean(tf.square(self.scores)) - tf.square(tf.reduce_mean(self.scores))) * tf.sqrt(tf.reduce_mean(tf.square(self.input_y)) - tf.square(tf.reduce_mean(self.input_y)))
            self.pearson = cov_Score_Input_y / d_Score_Input_y







