# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/24 下午10:18'

import numpy as np
import tensorflow as tf


class CNN(object):

    def __init__(self, logPath, trainSet, validationSet, testSet, lambda_=1e-4):
        """
        创建一个卷积神经网络
        """
        # 重置tensorflow的graph，确保神经网络可多次运行
        tf.reset_default_graph()
        tf.set_random_seed(1908)
        self.logPath = logPath
        self.trainSet = trainSet
        self.validationSet = validationSet
        self.testSet = testSet
        self.lambda_ = lambda_
        self.W = []

    def defineCNN(self):
        """
        定义 CNN 结构
        :return:
        """

        img = tf.reshape(self.input, [-1, 28, 28, 1])

        # 定义卷积层 1 和池化层 1
        convPool1 = self.defineConvPool(img, filterShape=[5, 5, 1, 20], poolSize=[1, 2, 2, 1])

        # 定义卷积层 2 和池化层 2
        convPool2 = self.defineConvPool(convPool1, filterShape=[5, 5, 20, 40], poolSize=[1, 2, 2, 1])

        # 将池化层 2 的输出变成行向量，后者将作为全连接层的输入
        convPool2 = tf.shape(convPool2, [-1, 40 * 4 * 4])

        # 定义全连接层
        self.out = self.defineFullConnected(convPool2, size=[30, 10])

    def defineConvPool(self, inputLayer, filterShape, poolSize):
        """
        定义卷积层和池化层
        :param inputLayer:
        :param filterShape:
        :param poolSize:
        :return:
        """
        weights = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1))
        self.W.append(weights)
        biases = tf.Variable(tf.zeros(filterShape[-1]))
        # 定义卷积层
        _conv2d = tf.nn.conv2d(inputLayer, weights, strides=[1, 1, 1, 1], padding="VALID")
        convOut = tf.nn.relu(_conv2d + biases)
        # 定义池化层
        poolOut = tf.nn.max_pool(convOut, ksize=poolSize, strides=poolSize, padding="VALID")
        return poolOut

    def defineFullConnected(self):
        """
        定义全连接层的结构
        :return:
        """
        prevSize = inputLayer.shape[1].value
        prevOut = inputLayer
        layer = 1
        # 定义隐藏层
        for currentSize in size[:-1]:
            weights = tf.Variable(tf.truncated_normal(
                [prevSize, currentSize], stddev=1.0 / np.sqrt(float(prevSize))),
                name="fc%s_weights" % layer)
            # 将模型中的权重项记录下来，用于之后的惩罚项
            self.W.append(weights)
            # 记录隐藏层的模型参数
            tf.summary.histogram("hidden%s" % layer, weights)
            biases = tf.Variable(tf.zeros([currentSize]),
                                 name="fc%s_biases" % layer)
            layer += 1
            # 定义这一层神经元的输出
            neuralOut = tf.nn.relu(tf.matmul(prevOut, weights) + biases)
            # 对隐藏层里的神经元使用dropout
            prevOut = tf.nn.dropout(neuralOut, self.keepProb)
            prevSize = currentSize
        # 定义输出层
        weights = tf.Variable(tf.truncated_normal(
            [prevSize, size[-1]], stddev=1.0 / np.sqrt(float(prevSize))),
            name="output_weights")
        # 将模型中的权重项记录下来，用于之后的惩罚项
        self.W.append(weights)
        biases = tf.Variable(tf.zeros([size[-1]]), name="output_biases")
        out = tf.matmul(prevOut, weights) + biases
        return out
