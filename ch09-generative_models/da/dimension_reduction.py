# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/21 下午7:26'

import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def loadData():
    digits = datasets.load_digits()
    X = digits.data
    Y = digits.target
    return X, Y


def dimensionReduction(X, y):
    """
    使用 LDA 模型将数据降到 3 维
    :param X:
    :param y:
    :return:
    """
    model = LinearDiscriminantAnalysis(n_components=3)
    model.fit(X, y)
    newX = model.transform(X)
    return newX


def visualize(newX, y):
    """
    将结果可视化
    """
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里画一幅图
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    colors = ["r", "b", "k", "g"]
    markers = ["^", "x", "o", "*"]
    for color, marker, i in zip(colors, markers, [0, 1, 2, 3]):
        ax.scatter(newX[y == i, 0], newX[y == i, 1], newX[y == i, 2],
                   color=color, alpha=.8, lw=1, marker=marker, label=i)
    plt.legend(loc='best', shadow=True)
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        plt.title("利用LDA进行数据降维")
    else:
        plt.title("利用LDA进行数据降维".decode("utf-8"))
    plt.show()


if __name__ == "__main__":
    X, y = loadData()
    newX = dimensionReduction(X, y)
    visualize(newX, y)
