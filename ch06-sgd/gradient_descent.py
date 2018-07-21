# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/18 下午2:46'

from utils import generateLinearData, createLinearModel, createSummaryWriter

import tensorflow as tf
import numpy as np


def run():
    """
    程序入口
    :return:
    """
    dimension = 30
    num = 10000

    # generate data
    X, Y = generateLinearData(dimension, num)

    # define model
    model = createLinearModel(dimension)

    # 得到模型参数
    gradientDescent(X, Y, model)


def gradientDescent(X, Y, model, learningRate=0.01, maxIter=10000, tol=1.e-6):
    """
    梯度下降法训练模型，得到最优模型参数
    :param X:
    :param Y:
    :param model:
    :return:
    """

    # 创建 GradientDescentOptimizer 对象
    method = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    optimizer = method.minimize(model['loss_function'])

    # 增加日志
    tf.summary.scalar("loss_function", model['loss_function'])
    tf.summary.histogram("params", model['model_params'])
    tf.summary.scalar("first_param", tf.reduce_mean(model['model_params'][0]))
    tf.summary.scalar("last_param", tf.reduce_mean(model['model_params'][-1]))
    summary = tf.summary.merge_all()
    # 在程序运行结束之后，运行如下命令，查看日志
    # tensorboard --logdir logs/

    summaryWriter = createSummaryWriter("logs/gradient_descent")

    # Tensorflow 开始运行
    sess = tf.Session()
    # 对 variable 变量进行初始化
    init = tf.global_variables_initializer()
    sess.run(init)

    step = 0
    prevLoss = np.inf
    diff = np.inf

    while (step < maxIter) & (diff > tol):
        _, summaryStr, loss = sess.run(
            [optimizer, summary, model['loss_function']],
            feed_dict={
                model['independent_variable']: X,
                model['dependent_variable']: Y
            }
        )
        # 将运作细节写入目录
        summaryWriter.add_summary(summaryStr, step)
        # 计算损失函数的变动
        diff = abs(prevLoss - loss)
        prevLoss = loss
        step += 1

    summaryWriter.close()


if __name__ == '__main__':
    run()
