# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/16 下午2:42'

"""
1. spit the data: trainData, testData
2. train the model
3. evaluate the model, get the MSE and COD
4. visualization
"""

import os
import sys

from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def linearModel(data):
    features = ["x"]
    labels = ["y"]

    trainData = data[:15]
    testData = data[15:]

    model = trainModel(trainData, features, labels)
    error, score = evaluateModel(model, testData, features, labels)
    visualizeModel(model, data, features, labels, error, score)


def trainModel(trainData, features, labels):
    model = linear_model.LinearRegression()
    model.fit(trainData[features], trainData[labels])
    return model


def evaluateModel(model, testData, features, labels):
    error = np.mean((model.predict(testData[features]) - testData[labels]) ** 2)
    score = model.score(testData[features], testData[labels])
    return error, score


def visualizeModel(model, data, features, labels, error, score):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(111)
    ax.set_title("线性回归示例")
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.scatter(data[features], data[labels], color='b', label=u'%s: $y = x + \epsilon$' % "真实值")
    if model.intercept_ > 0:
        # 画线图，用红色线条表示模型结果
        # 在Python3中，str不需要decode
        if sys.version_info[0] == 3:
            ax.plot(data[features], model.predict(data[features]), color='r',
                    label=u'%s: $y = %.3fx$ + %.3f' \
                          % ("预测值", model.coef_, model.intercept_))
        else:
            ax.plot(data[features], model.predict(data[features]), color='r',
                    label=u'%s: $y = %.3fx$ + %.3f' \
                          % ("预测值".decode("utf-8"), model.coef_, model.intercept_))
    ## coef: 系数，intercept: 截距
    else:
        # 在Python3中，str不需要decode
        if sys.version_info[0] == 3:
            ax.plot(data[features], model.predict(data[features]), color='r',
                    label=u'%s: $y = %.3fx$ - %.3f' \
                          % ("预测值", model.coef_, abs(model.intercept_)))
        else:
            ax.plot(data[features], model.predict(data[features]), color='r',
                    label=u'%s: $y = %.3fx$ - %.3f' \
                          % ("预测值".decode("utf-8"), model.coef_, abs(model.intercept_)))
    legend = plt.legend(shadow=True)
    legend.get_frame().set_facecolor('#6F93AE')
    # 显示均方差和决定系数
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.text(0.99, 0.01,
                u'%s%.3f\n%s%.3f' \
                % ("均方差：", error, "决定系数：", score),
                style='italic', verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, color='m', fontsize=13)
    else:
        ax.text(0.99, 0.01,
                u'%s%.3f\n%s%.3f' \
                % ("均方差：".decode("utf-8"), error, "决定系数：".decode("utf-8"), score),
                style='italic', verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, color='m', fontsize=13)
    # 展示上面所画的图片。图片将阻断程序的运行，直至所有的图片被关闭
    # 在Python shell里面，可以设置参数"block=False"，使阻断失效。
    plt.show()


if __name__ == '__main__':
    filepath = 'data/simple_example.csv'
    data = pd.read_csv(filepath)
    linearModel(data)
    # 选择列
    # data["x"] data[["x", "y"]]
    # 选择行
    # data[:10]
