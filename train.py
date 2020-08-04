import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
import dataProcessing
from modelCNN import TextSimilarityCNN
import time
import os
import datetime


def train():
    # command line parameters
    parser = ArgumentParser()
    # data loading
    parser.add_argument("--trainData", default='data/train_pointwise', type=str)
    parser.add_argument("--devData", default='data/dev', type=str)
    parser.add_argument("--testData", default='data/test', type=str)

    # model hyperparameters
    parser.add_argument("--filterSizes", default=[3,4,5], type=list)
    parser.add_argument("--numFilters", default=128, type=int)
    parser.add_argument("--sequenceLength", default=84, type=int)
    parser.add_argument("--numClasses", default=1, type=int)
    parser.add_argument("--dropoutProb", default=0.5, type=float)
    parser.add_argument("--l2RegLambda", default=1, type=float)

    # training parameters
    parser.add_argument("--batchSize", default=64, type=int)
    parser.add_argument("--numEpochs", default=200, type=int)
    parser.add_argument("--evaluateEvery", default=100, type=int)
    parser.add_argument("--checkpointEvery", default=100, type=int)
    parser.add_argument("--numCheckpoints", default=5, type=int)

    # Misc Parameters
    parser.add_argument("--allowSoftPlacement", default=False, action='store_true')
    parser.add_argument("--logDevicePlacement", default=False, action='store_true')

    args = parser.parse_args()

    # process data
    print('start loading data...')
    s1_train, s2_train, score_train = dataProcessing.readData(args.trainData)
    s1_dev, s2_dev, score_dev = dataProcessing.readData(args.devData)
    s1_test, s2_test, score_test = dataProcessing.readData(args.testData)

    score_train = np.asarray([[score] for score in score_train])
    score_dev = np.asarray([[score] for score in score_dev])
    score_test = np.asarray([[score] for score in score_test])

    # train
    print('start training...')
    with tf.Graph().as_default():
        session_config = tf.compat.v1.ConfigProto(allow_soft_placement=args.allowSoftPlacement)
        sess = tf.compat.v1.Session(config=session_config)
        with sess.as_default():
            cnn = TextSimilarityCNN(sequenceLength=args.sequenceLength, numClasses=args.numClasses, filterSizes=args.filterSizes, numFilters=args.numFilters, l2RegLambda=args.l2RegLambda)
            # define training hyperparameters
            globalStep = tf.Variable(0, name='globalStep', trainable=False)
            optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            # auto update gradients and parameters
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=globalStep)

            # keep track of gradient values and sparsity
            # gradSumary = []
            # for gradient, variable in grads_and_vars:
            #     if gradient is not None:
            #         grad_hist_summary = tf.compat.v1.summary.histogram("{}/grad/hist".format(variable.name), gradient)
            #         sparsity_summary = tf.compat.v1.summary.scalar("{}/grad/sparsity".format(variable.name), tf.nn.zero_fraction(gradient))
            #         gradSumary.append(grad_hist_summary)
            #         gradSumary.append(sparsity_summary)
            # gradSumaryMerged = tf.compat.v1.summary.merge(gradSumary)
            for gradient, variable in grads_and_vars:
                if gradient is not None:
                    tf.compat.v1.summary.histogram("{}/grad/hist".format(variable.name), gradient)
                    tf.compat.v1.summary.scalar("{}/grad/sparsity".format(variable.name), tf.nn.zero_fraction(gradient))
            gradSumaryMerged = tf.compat.v1.summary.merge_all()

            # outpur directory for models and summary
            localtime = time.localtime()
            outputDir = str(localtime.tm_year) + "_" + str(localtime.tm_mon) + "_" + str(localtime.tm_mday) + "_" + str(localtime.tm_hour) + "_" + str(localtime.tm_min)

            # summaries for loss and pearson
            lossSummary = tf.compat.v1.summary.scalar("loss", cnn.loss)
            accuracySummary = tf.compat.v1.summary.scalar("pearson", cnn.pearson)

            # train summary
            trainSummary_op = tf.compat.v1.summary.merge([lossSummary, accuracySummary, gradSumaryMerged])
            trainSummary_dir = os.path.join(outputDir, "summaries", "train")
            trainSummary_writer = tf.compat.v1.summary.FileWriter(trainSummary_dir, sess.graph)

            # dev summary
            devSummary_op = tf.compat.v1.summary.merge([lossSummary, accuracySummary])
            devSummary_dir = os.path.join(outputDir, "summaries", "dev")
            devSummary_writer = tf.compat.v1.summary.FileWriter(devSummary_dir, sess.graph)

            checkpoint_dir = os.path.join(outputDir, "checkpoints")
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=args.numCheckpoints)

            # initialize all variables
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

            def trainStep(s1, s2, score):
                # a single training step
                feed_dict = {cnn.input_s1: s1, cnn.input_s2: s2, cnn.input_y: score, cnn.dropoutProb: args.dropoutProb}
                _, step, summary, loss, pearson = sess.run([train_op, globalStep, trainSummary_op, cnn.loss, cnn.pearson], feed_dict=feed_dict)
                timeForLog = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, pearson {:g}".format(timeForLog, step, loss, pearson))
                trainSummary_writer.add_summary(summary, step)

            def devStep(s1, s2, score, writer=None):
                # evalute model on a dev set
                feed_dict = {cnn.input_s1: s1, cnn.input_s2: s2, cnn.input_y: score, cnn.dropoutProb: 1.0}
                step, summary, loss, pearson = sess.run([globalStep, devSummary_op, cnn.loss, cnn.pearson], feed_dict=feed_dict)
                timeForLog = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, accuracy {:g}".format(timeForLog, step, loss, pearson))
                if writer:
                    writer.add_summary(summary, step)

            # generate batches
            STS_train = dataProcessing.dataset(s1=s1_train, s2=s2_train, label=score_train)

            # training loop for each batch
            for i in range(4000):
                batch_train = STS_train.next_batch(args.batchSize)
                trainStep(batch_train[0], batch_train[1], batch_train[2])
                currentStep = tf.compat.v1.train.global_step(sess, globalStep)
                if currentStep % args.evaluateEvery == 0:
                    print("\nEvaluation:")
                    devStep(s1_dev, s2_dev, score_dev, writer=devSummary_writer)
                    print("")
                if currentStep % args.checkpointEvery == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=currentStep)
                    print("Saved model checkpoint to {}\n".format(path))


def main():
    train()


if __name__ == '__main__':
    main()

