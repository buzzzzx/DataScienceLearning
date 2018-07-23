# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/22 下午3:53'

from sklearn.mixture import GaussianMixture


def trainModel(data, clusterNum):
    """
    使用混合高斯模型 + EM 算法对数据进行聚类
    :param data:
    :param clusterNum:
    :return:
    """
    model = GaussianMixture(n_components=clusterNum, covariance_type="full")
    model.fit(data)
    return model
