# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/23 下午9:17'


from pyspark.ml.classification import LogisticRegression

def trainModel(data):
    """
    使用 spark ml 库中的逻辑回归算法
    :param data:
    :return:
    """
    lr = LogisticRegression()
    lrModel = lr.fit(data)
    print(lrModel.coefficients)
    print(lrModel.intercept)