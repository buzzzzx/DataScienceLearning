# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/17 下午4:11'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import pandas as pd
import numpy as np


def logitRegression(data):
    trainSet, testSet = train_test_split(data, test_size=0.2)
    re = trainModel(trainSet)
    modelSummary(re)
    interpretModel(re)
    makePrediction(re, testSet)
    evaluation(re)


def trainModel(trainSet):
    formula = "label_code ~ age + education_num + capital_gain + capital_loss + hours_per_week"
    model = sm.Logit.from_formula(formula, data=trainSet)
    re = model.fit()
    return re


def interpretModel(re):
    """
    理解模型关系
    :param re: 训练好的模型
    :return:
    """
    conf = re.conf_int()
    conf['OR'] = re.params
    conf.colums = ['2.5%', '97.5%', 'OR']
    print("各个变量对事件发生比的影响：")
    print(np.exp(conf))
    print("计算各个变量的边际效应：")
    print(re.get_margeff(at="overall").summary())


def modelSummary(re):
    print(re.summary())
    print(re.f_test("education_num=0"))
    print("检验假设 educat_num 的系数 = 0.32 和 hours_per_week 的系数 = 0.04 同时成立")
    print(re.f_test("education_num=0.32, hours_per_week=0.04"))


def makePrediction(re, testSet, alpha=0.5):
    testSet['prob'] = re.predict(testSet)
    print("事件发生概率（预测概率）大于0.6的数据个数：")
    print(testSet[testSet["prob"] > 0.6].shape[0])  # 输出值为576
    print("事件发生概率（预测概率）大于0.5的数据个数：")
    print(testSet[testSet["prob"] > 0.5].shape[0])  # 输出值为834
    testSet['pred'] = testSet.apply(lambda x: 1 if x['prob'] > alpha else 0, axis=1)
    return testSet


def evaluation(re):
    """
    计算预测结果的查准查全率以及f1
    参数
    ----
    re ：DataFrame，预测结果，里面包含两列：真实值‘lable_code’、预测值‘pred’
    """
    bins = np.array([0, 0.5, 1])
    label = re["label_code"]
    pred = re["pred"]
    tp, fp, fn, tn = np.histogram2d(label, pred, bins=bins)[0].flatten()
    precision = tp / (tp + fp)  # 0.951
    recall = tp / (tp + fn)  # 0.826
    f1 = 2 * precision * recall / (precision + recall)  # 0.884
    print("查准率: %.3f, 查全率: %.3f, f1: %.3f" % (precision, recall, f1))


def multiLogit(data):
    """
    解决多元分类问题：多元罗辑回归和 OvA
    :param data:
    :return:
    """
    features = ['x1', 'x2']
    labels = "label"
    methods = ['multinomial', 'ovr']
    for i in range(len(methods)):
        model = LogisticRegression(multi_class=methods[i], solver='sag', max_iter=1000, random_state=42)
        model.fit(data[features], data[labels])
        # ...


def balanceData(X, Y):
    """
    解决分类不均衡问题，修改权重
    :param data:
    :return:
    """
    positiveWeight = len(Y[Y > 0]) / float(len(Y))
    classWeight = {1: 1. / positiveWeight, 0: 1. / (1 - positiveWeight)}
    model = LogisticRegression(class_weight=classWeight, C=1e4)
    model.fit(X, Y.ravel())
    pred = model.predict(X)
    return pred


if __name__ == '__main__':
    data = pd.read_csv("data/adult.data")
    cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week", "label"]
    data = data[cols]
    data['label_code'] = pd.Categorical(data.label).codes
    logitRegression(data=data)
