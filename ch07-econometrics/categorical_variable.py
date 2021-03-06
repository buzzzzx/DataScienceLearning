# -*- coding: utf-8 -*-
__author__ = 'buzz'
__date__ = '2018/7/19 下午1:32'

import os
import sys

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from statsmodels.graphics.mosaicplot import mosaic
from patsy import ContrastMatrix


def readData(filepath):
    data = pd.read_csv(filepath)
    cols = ["workclass", "sex", "age", "education_num",
            "capital_gain", "capital_loss", "hours_per_week", "label"]
    return data[cols]


def transLabel(data):
    data['label_code'] = pd.Categorical(data.label).codes
    return data


def analyseData(data):
    cross1 = pd.crosstab(data['sex'], data['label'])
    print("显示sex, label交叉报表：")
    props = lambda key: {"color": "0.45"} if ' >50K' in key else {"color": "#C6E2FF"}
    mosaic(cross1[[" >50K", " <=50K"]].stack(), properties=props)
    plt.show()


def baseModel(data):
    """
    原有模型
    """
    formula = "label_code ~ education_num + capital_gain + capital_loss + hours_per_week"
    model = sm.Logit.from_formula(formula, data=data)
    re = model.fit()
    return re


def trainMode(data):
    """
    加入 Sex
    :param data:
    :return:
    """
    formula = """label_code ~ C(sex) + C(workclass) + education_num + capital_gain + capital_loss + hours_per_week"""
    model = sm.Logit.from_formula(formula, data=data)
    re = model.fit()
    return re


def trainModel2(data):
    """
    加入 workclass 变量，搭建逻辑回归模型，并训练模型
    """
    # 定义workclass的类别顺序，数组里的第一个值为基准类别
    l = [" ?", " Never-worked", " Without-pay", " State-gov",
         " Self-emp-not-inc", " Private", " Federal-gov",
         " Local-gov", " Self-emp-inc"]
    # 定义各个类别对应的虚拟变量
    contrast = np.eye(9, 6, k=-3)
    # 为每个虚拟变量命名
    contrast_mat = ContrastMatrix(contrast, l[3:])
    formula = """label_code ~ C(workclass, contrast_mat, levels=l)
            + C(sex) + education_num + capital_gain
            + capital_loss + hours_per_week"""
    model = sm.Logit.from_formula(formula, data=data)
    re = model.fit()
    return re


def trainModel3(data):
    """
    使用 sex 变量的 Ridit scoring 搭建逻辑回归模型，并训练模型
    :param data:
    :return:
    """
    l = [" Male", " Female"]
    contrast = [[-0.33], [0.67]]
    contrast_mat = ContrastMatrix(contrast, ["Ridit(sex)"])
    formula = """label_code ~ C(sex, contrast_mat, levels=l) + education_num
            + capital_gain + capital_loss + hours_per_week"""
    model = sm.Logit.from_formula(formula, data=data)
    re = model.fit()
    return re


def makePrediction(re, testSet, alpha=0.5):
    """
    使用训练好的模型对测试数据做预测
    """
    # 关闭pandas有关chain_assignment的警告
    pd.options.mode.chained_assignment = None
    # 计算事件发生的概率
    data = testSet.copy()
    data["prob"] = re.predict(data)
    # 根据预测的概率，得出最终的预测
    data["pred"] = data.apply(lambda x: 1 if x["prob"] > alpha else 0, axis=1)
    return data


def evaluation(newRe, baseRe):
    """
    展示加入性别变量后，模型效果的变化
    """
    fpr, tpr, _ = metrics.roc_curve(newRe["label_code"], newRe["prob"])
    auc = metrics.auc(fpr, tpr)
    # 为在Matplotlib中显示中文，设置特殊字体
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    # 创建一个图形框
    fig = plt.figure(figsize=(6, 6), dpi=80)
    # 在图形框里只画一幅图
    ax = fig.add_subplot(111)
    # 在Matplotlib中显示中文，需要使用unicode
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.set_title("%s" % "ROC曲线")
    else:
        ax.set_title("%s" % "ROC曲线".decode("utf-8"))
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.plot(fpr, tpr, "k", label="%s; %s = %0.2f" % ("加入性别后的ROC曲线",
                                                         "曲线下面积（AUC）", auc))
    else:
        ax.plot(fpr, tpr, "k",
                label="%s; %s = %0.2f" % ("加入性别后的ROC曲线".decode("utf-8"),
                                          "曲线下面积（AUC）".decode("utf-8"), auc))
    # 绘制原模型的ROC曲线
    fpr, tpr, _ = metrics.roc_curve(baseRe["label_code"], baseRe["prob"])
    auc = metrics.auc(fpr, tpr)
    # 在Python3中，str不需要decode
    if sys.version_info[0] == 3:
        ax.plot(fpr, tpr, "b-.", label="%s; %s = %0.2f" % ("加入性别前的ROC曲线",
                                                           "曲线下面积（AUC）", auc))
    else:
        ax.plot(fpr, tpr, "b-.",
                label="%s; %s = %0.2f" % ("加入性别前的ROC曲线".decode("utf-8"),
                                          "曲线下面积（AUC）".decode("utf-8"), auc))
    legend = plt.legend(shadow=True)
    plt.show()


def logitRegression(data):
    data = transLabel(data)
    analyseData(data)

    trainSet, testSet = train_test_split(data, test_size=0.2, random_state=2310)

    newRe = trainMode(trainSet)
    print(newRe.summary())
    newRe = makePrediction(newRe, testSet)

    # 计算原模型预测结果
    baseRe = baseModel(trainSet)
    baseRe = makePrediction(baseRe, testSet)
    evaluation(newRe, baseRe)


if __name__ == '__main__':
    filepath = "data/adult.data"
    data = readData(filepath)
    # analyseData(data)
    logitRegression(data)
