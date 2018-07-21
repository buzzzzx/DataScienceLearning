# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/16 下午5:33'

import statsmodels.api as sm
import numpy as np
import pandas as pd


def trainRegulizedModel(X, Y, alpha):
    model = sm.OLS(Y, X)
    res = model.fit_regularized(alpha=alpha)
    return res



