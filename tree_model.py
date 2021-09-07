from sklearn.datasets import load_iris
import numpy as np
import math
from collections import Counter
from sklearn.metrics import roc_auc_score, cohen_kappa_score, accuracy_score
import pandas as pd
import random
from sklearn import tree
import pickle


def create_dataset():
    data = pd.read_excel('data/data_54.xls', index_col='编号')
    cols = list(data.columns)
    data = data.values
    np.random.seed(15)
    np.random.shuffle(data)
    sz = int(0.8 * len(data))
    x_train = data[:sz, 1:]
    y_train = data[:sz, 0]
    x_test = data[sz:, 1:]
    y_test = data[sz:, 0]
    print(cols)
    rand_data = pd.DataFrame(data,columns=cols)
    rand_data.to_excel('data/rand_data.xlsx')
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = create_dataset()
# clf = tree.DecisionTreeClassifier(criterion="gini", min_samples_leaf=30, max_depth=15)
# clf.fit(x_train, y_train)
#
# y_pred = clf.predict(x_test)
# print(y_pred)
# print(accuracy_score(y_test, y_pred))
# y_pr = clf.predict_proba(x_test)
#
# df = pd.DataFrame(y_pr)
# df.to_csv("data/tree_y_pr.csv")
