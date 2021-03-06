# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/23 下午3:13'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql import SparkSession


def generateData(n):
    """
    随机生成数据
    """
    x = np.random.randn(n)
    error = np.round(np.random.randn(n), 2)
    const = np.ones(n)
    y = 2 * x + 3 * const + error
    return pd.DataFrame({"x": x, "const": const, "y": y})


def startSpark():
    """
    创建 SparkSession，这是 Saprk 程序的入口
    :return:
    """
    spark = SparkSession.builder.appName("gd_example").getOrCreate()

    return spark


def gradientDescent(data, labelCol, featuresCol, stepSize=1, maxIter=100):
    """
    利用梯度下降法训练线性模型
    :param data: 训练模型的数据
    :param labelCol: 被预测量的名字
    :param featureCol: 自变量的名字列表
    :param stepSize: 梯度下降法里的学习速率
    :param maxIter: 梯度下降法的迭代次数
    :return:
    """
    w = np.random.randn(len(featuresCol))
    # 数据转换为dict形式，并将自变量转换为一个np.array
    df = data.rdd.map(
        lambda item: {"y": item[labelCol], "x": np.array([item[i] for i in featuresCol])})
    # 将数据放到内存里，提高运算速度
    df.cache()
    for i in range(maxIter):
        # 计算线性回归模型的单点梯度以及数据的个数
        _gradient, num = df \
            .map(lambda item: -1 * (item["y"] - w.dot(item["x"])) * item["x"]) \
            .map(lambda x: (x, 1)) \
            .reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        # 计算损失函数的平均单点梯度
        gradient = _gradient / num
        # 更新模型参数
        w -= stepSize * gradient
    return w


if __name__ == "__main__":
    np.random.seed(4889)
    spark = startSpark()
    data = spark.createDataFrame(generateData(1000))
    w = gradientDescent(data, "y", ["x", "const"])
    # 在Windows下运行此脚本需确保Windows下的命令提示符(cmd)能显示中文
    print("模型的参数为：%s" % [2, 3])
    print("参数的估计值为：%s" % w)
