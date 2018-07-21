# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/16 下午5:48'

import pickle
from sklearn import linear_model


def trainAndSaveModel(data, modelPath):
    model = linear_model.LinearRegression()
    model.fit(data["x"], data["y"])
    pickle.dumps(model, open(modelPath, 'wb'))
    return model


def loadModel(modelPath):
    model = pickle.loads(open(modelPath, 'rb'))
    return model
