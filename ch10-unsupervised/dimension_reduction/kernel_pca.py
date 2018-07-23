# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/22 下午5:37'

from sklearn.decomposition import KernelPCA


def trainModel(data):
    """
    使用带核函数的主成分分析将数据降维（先升维再降维）
    :param data:
    :return:
    """
    model = KernelPCA(n_components=2, kernel="rbf", gamma=25)
    model.fit(data)
    return model
