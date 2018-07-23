# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/22 下午5:24'

from sklearn.decomposition import PCA


def trainModel(data):
    """
    使用 PCA 对数据进行降维
    :param data:
    :return:
    """

    model = PCA(n_components=2)
    model.fit(data)
    return data
