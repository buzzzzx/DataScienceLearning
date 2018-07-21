# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/18 下午4:49'

from utils import generateLinearData, createLinearModel, createSummaryWriter

import math

import tensorflow as tf
import numpy as np


def stochasticGradientDescent(X, Y, model, learningRate=0.01,
                              miniBatchFraction=0.01, epoch=10000, tol=1.e-6):
    """
    随机梯度下降法训练模型，得到最优模型参数
    :param X:
    :param Y:
    :param model:
    :param learningRate:
    :param miniBatchFraction:
    :param epoch:
    :param tol:
    :return:
    """

    method = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    optimizer = method.minimize(model['loss_function'])

    # 增加日志
    tf.summary.scalar("loss_function", model["loss_function"])
    tf.summary.histogram("params", model["model_params"])
    tf.summary.scalar("first_param", tf.reduce_mean(model["model_params"][0]))
    tf.summary.scalar("last_param", tf.reduce_mean(model["model_params"][-1]))
    summary = tf.summary.merge_all()

    summaryWriter = createSummaryWriter("logs/stochastic_gradient_descent")

    # 开始运行 tensorflow
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # 每次迭代需要用到的数据量
    batchSize = int(X.shape[0] * miniBatchFraction)
    # 所有的数据都过一遍需要迭代的次数
    batchNum = int(math.ceil(1 / miniBatchFraction))

    step = 0
    preLoss = np.inf
    diff = np.inf

    while (step < epoch) & (diff > tol):
        for i in range(batchNum):
            # 选取小批次训练数据
            batchX = X[i * batchSize: (i + 1) * batchSize]
            batchY = Y[i * batchSize: (i + 1) * batchSize]

            # 迭代模型参数
            sess.run(
                [optimizer],
                feed_dict={
                    model['independent_variable']: batchX,
                    model['dependent_variable']: batchY,
                }
            )

            # 计算损失函数并写入日志
            summaryStr, loss = sess.run(
                [summary, model['loss_function']],
                feed_dict={
                    model['independent_variable']: X,
                    model['dependent_variable']: Y,
                }
            )

            # 将运作细节写入目录
            summaryWriter.add_summary(summaryStr, step * batchNum + 1)

            diff = abs(preLoss - loss)
            preLoss = loss
            if diff <= tol:
                break
        step += 1

    summaryWriter.close()

    print("模型参数：\n%s" % sess.run(model["model_params"]))
    print("训练轮次：%s" % step)
    print("损失函数值：%s" % loss)


def run():
    """
    程序入口
    """
    # dimension表示自变量的个数，num表示数据集里数据的个数。
    dimension = 30
    num = 10000
    # 随机产生模型数据
    X, Y = generateLinearData(dimension, num)
    # 定义模型
    model = createLinearModel(dimension)
    # 使用梯度下降法，估计模型参数
    stochasticGradientDescent(X, Y, model)


if __name__ == "__main__":
    run()
