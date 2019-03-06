import tensorflow as tf
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import classification_report as cr
from sklearn.metrics import roc_curve as rc

import Utility


class NeuralNet:
    # Encapsulates a neural net trainer
    def __init__(self, inputNodes, outputNodes):
        # Useful numbers
        self.numHiddenNode1 = 256
        self.numHiddenNode2 = 256
        self.numHiddenNode3 = 256
        self.numHiddenNode4 = 256
        self.numInputNodes = inputNodes
        self.numOutputNodes = outputNodes
        self.epoch = 1
        self.learnRate = 0.00001
        self.ops = {}

    def TrainNet(self, imageset, epoch, basename):
        sess = tf.Session()
        # graph inputs
        x_inputs = tf.placeholder(tf.float32, shape=[None, self.numInputNodes], name = 'x_inputs')
        y_label = tf.placeholder(tf.float32, shape=[None, self.numOutputNodes], name = 'y_label')

        # weights
        weightOne = tf.Variable(tf.random_normal([self.numInputNodes,
                                                  self.numHiddenNode1]), name='weightOne')
        weightTwo = tf.Variable(tf.random_normal([self.numHiddenNode1,
                                                  self.numHiddenNode2]), name='weightTwo')
        weightOut = tf.Variable(tf.random_normal([self.numHiddenNode2,
                                                  self.numOutputNodes]), name='weightOut')
        # biases
        biasOne = tf.Variable(tf.random_normal([self.numHiddenNode1]), name='biasOne')
        biasTwo = tf.Variable(tf.random_normal([self.numHiddenNode2]), name='biasTwo')
        biasOut = tf.Variable(tf.random_normal([self.numOutputNodes]), name='biasOut')

        # forward propagation
        predictionOne = tf.add(tf.matmul(x_inputs, weightOne), biasOne, name='predictionOne')
        predictionTwo = tf.add(tf.matmul(predictionOne, weightTwo), biasTwo, name='predictionTwo')
        predOut = tf.add(tf.matmul(predictionTwo, weightOut), biasOut, name='predOut')

        prediction = tf.nn.softmax(predOut, name='prediction')

        # backpropagation
        theLogits = tf.nn.softmax_cross_entropy_with_logits(logits=predOut, labels=y_label, name='theLogits')
        loss = tf.reduce_mean(theLogits, name='loss')
        optimiser = tf.train.AdamOptimizer(self.learnRate).minimize(loss, name='optimiser')
        correct_prediction = tf.equal(tf.argmax(prediction, 1),
                                      tf.argmax(y_label, 1), name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # actual
        init = tf.global_variables_initializer()


        x_train = imageset['data']
        y_train = imageset['labels']

        x_inputs = x_inputs
        y_label = y_label

        sess.run(init)

        for ii in range(epoch):
            sess.run(optimiser,
                     feed_dict={x_inputs: x_train,
                                y_label: y_train})
            if ii % 100 == 0:
                # Analyse progress so far
                lossy = sess.run(loss,
                                feed_dict={x_inputs: x_train,
                                           y_label: y_train})
                acc = sess.run(accuracy,
                                    feed_dict={x_inputs: x_train,
                                               y_label: y_train})
                print('Training Step:' + str(ii) + ' out of ' +
                      str(epoch) + '  Accuracy =  ' + str(acc) +
                      '  Loss = ' + str(lossy))

        print("Training finished...")


        saver = tf.train.Saver()
        save_path = saver.save(sess, basename)
        #chkp.print_tensors_in_checkpoint_file(save_path, all_tensors=True, tensor_name='')
        return save_path

    def RestoreState(self, metafile):

        tf.reset_default_graph()
        self.saver = tf.train.import_meta_graph(metafile)

        # for tensor in tf.get_default_graph().get_operations():
        #     print(tensor.name)

    def TestNet(self, imageset, filepath):

        x_test = imageset['data']
        y_test = imageset['labels']



        with tf.Session() as sess:
            x_inputs = tf.placeholder(tf.float32, shape=[None, self.numInputNodes], name='x_inputs')
            y_label = tf.placeholder(tf.float32, shape=[None, self.numOutputNodes], name='y_label')

            # weights
            weightOne = tf.Variable(tf.random_normal([self.numInputNodes,
                                                      self.numHiddenNode1]), name='weightOne')
            weightTwo = tf.Variable(tf.random_normal([self.numHiddenNode1,
                                                      self.numHiddenNode2]), name='weightTwo')
            weightOut = tf.Variable(tf.random_normal([self.numHiddenNode2,
                                                      self.numOutputNodes]), name='weightOut')
            # biases
            biasOne = tf.Variable(tf.random_normal([self.numHiddenNode1]), name='biasOne')
            biasTwo = tf.Variable(tf.random_normal([self.numHiddenNode2]), name='biasTwo')
            biasOut = tf.Variable(tf.random_normal([self.numOutputNodes]), name='biasOut')

            # forward propagation
            predictionOne = tf.add(tf.matmul(x_inputs, weightOne), biasOne, name='predictionOne')
            predictionTwo = tf.add(tf.matmul(predictionOne, weightTwo), biasTwo, name='predictionTwo')
            predOut = tf.add(tf.matmul(predictionTwo, weightOut), biasOut, name='predOut')

            prediction = tf.nn.softmax(predOut, name='prediction')

            # backpropagation
            theLogits = tf.nn.softmax_cross_entropy_with_logits(logits=predOut, labels=y_label, name='theLogits')
            loss = tf.reduce_mean(theLogits, name='loss')
            optimiser = tf.train.AdamOptimizer(self.learnRate).minimize(loss, name='optimiser')
            correct_prediction = tf.equal(tf.argmax(prediction, 1),
                                          tf.argmax(y_label, 1), name='correct_prediction')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

            # actual
            init = tf.global_variables_initializer()
            sess.run(init)
            self.saver.restore(sess, tf.train.latest_checkpoint(filepath))

            # actual testing
            check = sess.run(prediction, feed_dict={x_inputs:x_test, y_label:y_test})
            print(check.argmax())

            lossy = sess.run(loss, feed_dict={x_inputs: x_test, y_label: y_test})
            acc = sess.run(accuracy, feed_dict={x_inputs: x_test, y_label: y_test})

            print('Accuracy = ' + str(acc) + ' Loss = ' + str(lossy))




    def CheckNet(self, testset, imageset, epoch, basename, matrixFile, reportFile):
        sess = tf.Session()
        # graph inputs
        x_inputs = tf.placeholder(tf.float32, shape=[None, self.numInputNodes], name='x_inputs')
        y_label = tf.placeholder(tf.float32, shape=[None, self.numOutputNodes], name='y_label')

        # weights
        weightOne = tf.Variable(tf.random_normal([self.numInputNodes,
                                                  self.numHiddenNode1]), name='weightOne')
        weightTwo = tf.Variable(tf.random_normal([self.numHiddenNode1,
                                                  self.numHiddenNode2]), name='weightTwo')
        weightThree = tf.Variable(tf.random_normal([self.numHiddenNode2,
                                                  self.numHiddenNode3]), name='weightThree')
        weightFour = tf.Variable(tf.random_normal([self.numHiddenNode3,
                                                    self.numHiddenNode4]), name='weightFour')
        weightOut = tf.Variable(tf.random_normal([self.numHiddenNode4,
                                                  self.numOutputNodes]), name='weightOut')
        # biases
        biasOne = tf.Variable(tf.random_normal([self.numHiddenNode1]), name='biasOne')
        biasTwo = tf.Variable(tf.random_normal([self.numHiddenNode2]), name='biasTwo')
        biasThree = tf.Variable(tf.random_normal([self.numHiddenNode3]), name='biasThree')
        biasFour = tf.Variable(tf.random_normal([self.numHiddenNode3]), name='biasFour')
        biasOut = tf.Variable(tf.random_normal([self.numOutputNodes]), name='biasOut')

        # forward propagation
        predictionOne = tf.add(tf.matmul(x_inputs, weightOne), biasOne, name='predictionOne')
        predictionTwo = tf.add(tf.matmul(predictionOne, weightTwo), biasTwo, name='predictionTwo')
        predictionThree = tf.add(tf.matmul(predictionTwo, weightThree), biasThree, name='predictionThree')
        predictionFour = tf.add(tf.matmul(predictionThree, weightFour), biasFour, name='predictionFour')
        predOut = tf.add(tf.matmul(predictionFour, weightOut), biasOut, name='predOut')

        prediction = tf.nn.softmax(predOut, name='prediction')

        # backpropagation
        theLogits = tf.nn.softmax_cross_entropy_with_logits(logits=predOut, labels=y_label, name='theLogits')
        loss = tf.reduce_mean(theLogits, name='loss')
        optimiser = tf.train.AdamOptimizer(self.learnRate).minimize(loss, name='optimiser')
        correct_prediction = tf.equal(tf.argmax(prediction, 1),
                                      tf.argmax(y_label, 1), name='correct_prediction')
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # actual
        init = tf.global_variables_initializer()

        x_train = imageset['data']
        y_train = imageset['labels']

        x_test = testset['data']
        y_test = testset['labels']

        x_inputs = x_inputs
        y_label = y_label

        sess.run(init)

        for ii in range(epoch):
            sess.run(optimiser,
                     feed_dict={x_inputs: x_train,
                                y_label: y_train})
            if ii % 100 == 0:
                # Analyse progress so far
                lossy = sess.run(loss,
                                 feed_dict={x_inputs: x_train,
                                            y_label: y_train})
                acc = sess.run(accuracy,
                               feed_dict={x_inputs: x_train,
                                          y_label: y_train})
                print('Training Step:' + str(ii) + ' out of ' +
                      str(epoch) + '  Accuracy =  ' + str(acc) +
                      '  Loss = ' + str(lossy))

        print("Training finished...")

        print("Integrated Testing")
        # actual testing
        check = sess.run(prediction, feed_dict={x_inputs: x_test, y_label: y_test})
        check_label = sess.run(y_label, feed_dict={y_label:y_test})

        predy = []
        labely = []
        for ii in range(len(y_test)):
            predy.append(check[ii].argmax())
            labely.append(check_label[ii].argmax())
            print("predy: ", predy[ii], " labely: ", labely[ii])
            #rocCurve = rc(check_label[ii], check[ii], pos_label=1)

        classes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

        confusionMatrix = cm(labely, predy, labels=range(15))

        report = cr(labely, predy, labels=range(15))

        # print("ROC Curve")
        # print(rocCurve)

        ut = Utility.Utility()

        ut.SaveConfusionMatrix(str(confusionMatrix), matrixFile)
        ut.SaveClassification(str(report), reportFile)

        print("confusion sk matrix: ", confusionMatrix)

        print("classification report ", report)

        lossy = sess.run(loss, feed_dict={x_inputs: x_test, y_label: y_test})
        accy = sess.run(accuracy, feed_dict={x_inputs: x_test, y_label: y_test})

        print('Accuracy = ' + str(accy) + ' Loss = ' + str(lossy))

        saver = tf.train.Saver()
        save_path = saver.save(sess, basename)
        # chkp.print_tensors_in_checkpoint_file(save_path, all_tensors=True, tensor_name='')
        return save_path