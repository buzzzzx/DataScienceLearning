# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/22 下午4:30'

from sklearn.cluster import SpectralClustering


def trainModel(data, clusterNum):
    model = SpectralClustering(n_clusters=clusterNum, affinity="rbf", gamma=100, assign_labels="kmeans")
    model.fit(data)
    return model
