# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import random

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from SVM import SVM
from sklearn.svm import SVC, LinearSVC
import os, sys
import pandas as pd

poly_kernel_svm_clf = Pipeline([("scaler", StandardScaler()),
                                ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=0.5))
                                ])


def create_dataset():
    data = pd.read_excel('data/data_54.xls', index_col='编号').values
    np.random.seed(15)
    np.random.shuffle(data)
    sz = int(0.8 * len(data))
    x_train = data[:sz, 1:]
    y_train = data[:sz, 0]
    x_test = data[sz:, 1:]
    y_test = data[sz:, 0]
    return x_train, x_test, y_train, y_test


def train_svm():
    X_train, X_test, y_train, y_test = create_dataset()

    C = [0.001]
    for c in C:
        clf = svm.SVC(kernel="linear", probability=True)
        clf.fit(X_train, y_train)
        # print(clf.b, clf.w)

        pre = clf.predict(X_test)
        print(pre)
        df = pd.DataFrame(clf.predict_proba(X_test))
        print(df)

        df.to_csv("data/svm_y_pr.csv")
        print('correct:', np.mean((pre == y_test).astype(int)))


if __name__ == '__main__':
    train_svm()
