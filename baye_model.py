from __future__ import division, print_function
from sklearn import datasets
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from pylab import *
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

class NaiveBayes():
    """朴素贝叶斯分类模型. """

    def __init__(self):
        self.classes = None
        self.X = None
        self.y = None
        self.parameters = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        # 计算每一个类别每个特征的均值和方差
        for i in range(len(self.classes)):
            c = self.classes[i]
            # 选出该类别的数据集
            x_where_c = X[np.where(y == c)]
            # 计算该类别数据集的均值和方差
            self.parameters.append([])
            for j in range(len(x_where_c[0, :])):
                col = x_where_c[:, j]
                parameters = {}
                parameters["mean"] = col.mean()
                parameters["var"] = col.var()
                self.parameters[i].append(parameters)

    # 计算高斯分布密度函数的值
    def calculate_gaussian_probability(self, mean, var, x):
        coeff = (1.0 / (math.sqrt((2.0 * math.pi) * var)))
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var)))
        return coeff * exponent

    # 计算先验概率
    def calculate_priori_probability(self, c):
        x_where_c = self.X[np.where(self.y == c)]
        n_samples_for_c = x_where_c.shape[0]
        n_samples = self.X.shape[0]
        return n_samples_for_c / n_samples

    def classify(self, sample):
        posteriors = []

        # 遍历所有类别
        for i in range(len(self.classes)):
            c = self.classes[i]
            prior = self.calculate_priori_probability(c)
            posterior = np.log(prior)

            # probability = P(Y)*P(x1|Y)*P(x2|Y)*...*P(xN|Y)
            # 遍历所有特征
            for j, params in enumerate(self.parameters[i]):
                # 取出第i个类别第j个特征的均值和方差
                mean = params["mean"]
                var = params["var"]
                # 取出预测样本的第j个特征
                sample_feature = sample[j]
                # 按照高斯分布的密度函数计算密度值
                prob = self.calculate_gaussian_probability(mean, var, sample_feature)
                # 朴素贝叶斯模型假设特征之间条件独立，即P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y),
                # 并且用取对数的方法将累乘转成累加的形式
                posterior += np.log(prob)
            posteriors.append(posterior)

        # 对概率进行排序
        # print(np.exp(posteriors))
        index_of_max = np.argmax(posteriors)
        max_value = posteriors[index_of_max]

        return self.classes[index_of_max], np.exp(posteriors)

    # 对数据集进行类别预测
    def predict(self, X):
        y_pred = []
        prr = []
        for sample in X:
            y, pr = self.classify(sample)
            y_pred.append(y)
            prr.append(pr)
        return np.array(y_pred), prr


def create_dataset():
    data = pd.read_excel('data/data_54.xls', index_col='编号').values
    np.random.seed(15)
    np.random.shuffle(data)
    sz = int(0.8 * len(data))

    test_data = data[sz:, :]
    test_data = pd.DataFrame(test_data,columns=pd.read_excel('data/data_54.xls', index_col='编号').columns)
    # test_data.to_excel('data/test_data.xlsx')

    x_train = data[:sz, 1:]
    y_train = data[:sz, 0]
    x_test = data[sz:, 1:]
    y_test = data[sz:, 0]
    return x_train, x_test, y_train, y_test


def bates():
    X_train, X_test, y_train, y_test = create_dataset()

    clf = NaiveBayes()
    clf.fit(X_train, y_train)

    ##
    # X_test, y_test = X_train, y_train
    ##
    y_pred = np.array(clf.predict(X_test)[0])
    y_pr = np.array(clf.predict(X_test)[1])
    # print(y_pr)
    df = pd.DataFrame(y_pr)
    df.to_csv("data/bayes_y_pr.csv")

    accu = accuracy_score(y_test, y_pred)
    print(accu)
    auc = roc_auc_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    tp, fn, fp, tn = 0, 0, 0, 0
    for i, j in zip(y_test, y_pred):
        if i == 1 and j == 1:
            tp += 1
        elif i == 1 and j == 2:
            fn += 1
        elif i == 2 and j == 1:
            fp += 1
        elif i == 2 and j == 2:
            tn += 1
    hx = [tp, fn, fp, tn]

    return accu, hx, auc, kappa


bates()
