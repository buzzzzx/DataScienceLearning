# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/16 下午5:04'

import statsmodels.api as sm
import numpy as np
import pandas as pd


def generateData():
    np.random.seed(5320)
    x = np.array(range(0, 20)) / 2
    error = np.round(np.random.randn(20), 2)
    y = 0.05 * x + error
    z = np.zeros(20) + 1
    return pd.DataFrame({'x': x, 'z': z, 'y': y})


def wrongCoef():
    features = ['x', 'z']
    labels = ['y']
    data = generateData()
    X = data[features]
    Y = data[labels]

    model = sm.OLS(Y, X['x'])
    res = model.fit()
    print("没有加入新变量时：")
    print(res.summary())

    model1 = sm.OLS(Y, X)
    res1 = model1.fit()
    print("加入新变量时：")
    print(res1.summary())


if __name__ == '__main__':
    wrongCoef()
