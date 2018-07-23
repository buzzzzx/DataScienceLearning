# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/23 下午9:31'

from sklearn.linear_model import Lasso
from pyspark import SparkContext, SparkConf
from spark_sklearn import GridSearchCV
import numpy as np


def startSpark():
    conf = SparkConf().setAppName("grid search example")
    sc = SparkContext(conf=conf)
    return sc


def gridSearch(sc, data, label, features):
    """
    使用 grid search 寻找最优的超参数
    :param sc:
    :param data:
    :param label:
    :param features:
    :return: 
    """
    parameters = {"alpha": 10 ** np.linspace(-4, 0, 45)}
    la = Lasso()
    gs = GridSearchCV(sc, la, parameters)
    gs.fit(data[features], data[label])
    return gs
