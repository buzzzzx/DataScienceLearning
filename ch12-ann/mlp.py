# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/23 下午10:36'

import numpy as np
import tensorflow as tf


class ANN(object):
    def __init__(self, size, logPath):
        tf.reset_default_graph()
        tf.set_random_seed(1908)
        self.logPath = logPath
        self.size = size
        self.layerNum = len(self.size)

    def definiteANN(self):
        """
        定义神经网络结构
        :return:
        """
        # 输入层
        prevSize = self.input.shape[1].value
        prevOut = self.input
        size = self.size

        # 隐藏层
        for currentSize in size[: -1]:
            weights = tf.Variable(tf.truncated_normal(
                [prevSize, currentSize],
                stddev=1.0 / np.sqrt(float(prevSize))
            ))
            biases = tf.Variable(tf.zeros([currentSize]))
            # prevOut = tf.nn.sigmoid(tf.matmul(prevOut, weights) + biases)
            prevOut = tf.nn.relu(tf.matmul(prevOut, weights) + biases)
            prevSize = currentSize

        # 输出层
        currentSize = size[-1]
        weights = tf.Variable(tf.truncated_normal(
            [prevSize, currentSize],
            stddev=1.0 / np.sqrt(float(prevSize))
        ))
        biases = tf.Variable(tf.zeros([currentSize]))
        self.out = tf.matmul(prevOut, weights)
        return self

    def definiteLoss(self):
        """
        定义神经网络的损失函数
        :return:
        """
        # 定义单点损失
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label,
            logits=self.out,
            name="loss"
        )
        # 定义整点损失
        self.loss = tf.reduce_mean(loss, name="average_loss")
        return self

    def SGD(self, X, Y, learningRate, miniBatchFraction, epoch):
        """
        使用随机梯度下降法训练模型
        :param X:
        :param Y:
        :param learningRate:
        :param miniBatchFraction:
        :param epoch:
        :return:
        """
        # 记录训练细节
        tf.summary.scalar(("loss", self.loss))
        summary = tf.summary.merge_all()

        method = tf.train.GradientDescentOptimizer(learningRate)
        optimizer = method.minimize(self.loss)
        batchSize = int(X.shape[0] * miniBatchFraction)
        batchNum = int(np.ceil(1 / miniBatchFraction))  # 迭代次数

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        summary_writer = tf.summary.FileWriter(self.logPath, graph=tf.get_default_graph())
        step = 0

        while (step < epoch):
            for i in range(batchNum):
                batchX = X[i * batchSize: batchSize * (i + 1)]
                batchY = Y[i * batchSize: batchSize * (i + 1)]

                sess.run(
                    [optimizer],
                    feed_dict={self.input: batchX, self.label: batchY}
                )

            step += 1

            # 将日志写入文件
            summary_str = sess.run(summary, feed_dict={self.input: X, self.label: Y})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        self.sess = sess

    def fit(self, X, Y, learningRate=0.3, miniBatchFraction=0.1, epoch=2500):
        self.input = tf.placeholder(tf.float32, shape=[None, X.shape[1]], name="X")
        self.label = tf.placeholder(tf.float64, shape=[None, self.size[-1]], name="Y")
        self.definiteANN()
        self.definiteLoss()
        self.SGD(X, Y, learningRate, miniBatchFraction, epoch)

    def predict_proba(self, X):
        """
        使用神经网络对未知数据进行预测
        :param X:
        :return:
        """
        sess = self.sess
        pred = tf.nn.softmax(logits=self.out, name="pred")
        prob = sess.run(pred, feed_dict={self.input: X})
        return prob
