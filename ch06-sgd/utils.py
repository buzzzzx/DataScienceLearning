# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/18 下午2:48'

import numpy as np
import tensorflow as tf


def generateLinearData(dimension, num):
    np.random.seed(1024)
    beta = np.array(range(dimension)) + 1
    x = np.random.random((num, dimension))
    epsilon = np.random.random((num, 1))

    y = x.dot(beta).reshape((-1, 1)) + epsilon

    return x, y


def createLinearModel(dimension):
    np.random.seed(1024)
    x = tf.placeholder(dtype=tf.float64, shape=[None, dimension], name='x')
    y = tf.placeholder(dtype=tf.float64, shape=[None, 1], name='y')

    betaPred = tf.Variable(np.random.random((dimension, 1)))
    yPred = tf.matmul(x, betaPred, name='y_pred')

    loss = tf.reduce_mean(tf.square(y - yPred))
    model = {
        "loss_function": loss,
        "independent_variable": x,
        "dependent_variable": y,
        "prediction": yPred,
        "model_params": betaPred,
    }
    return model


def createSummaryWriter(logPath):
    if tf.gfile.Exists(logPath):
        tf.gfile.DeleteRecursively(logPath)
    summarywriter = tf.summary.FileWriter(logPath, graph=tf.get_default_graph())
    return summarywriter
