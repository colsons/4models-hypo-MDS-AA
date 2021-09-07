import cv2
import time
import sys
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, roc_curve, auc
from matplotlib import rcParams

rcParams['font.sans-serif'] = ["Times New Roman"]
rcParams['font.size'] = 13


def eve(y_test, y_pred):
    a1 = accuracy_score(y_test, y_pred)
    a2 = roc_auc_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    tp, fn, fp, tn = 0, 0, 0, 0
    for i, j in zip(y_test, y_pred):
        if i == 0 and j == 0:
            tp += 1
        elif i == 0 and j == 1:
            fn += 1
        elif i == 1 and j == 0:
            fp += 1
        elif i == 1 and j == 1:
            tn += 1
    tpr = tp / (tp + fn)
    tnr = tn / (fp + tn)
    # 各项指标
    print("灵敏度 ", tp / (tp + fn))
    print("特异度 ", tn / (fp + tn))
    print("约登指数 ", tp / (tp + fn) + tn / (fp + tn) - 1)
    print("阳性似然比 ", tpr / (1 - tnr))
    print("阴性似然比 ", (1 - tpr) / tnr)
    print("acc ", a1)
    print("auc ", a2)
    print("kappa ", kappa)
    print("阳预 ", tp / (tp + fp))
    print("阴预 ", tn / (fn + tn))


def func(x):
    if x == 1:
        return 0
    if x == 0:
        return 1


y_test = [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
          1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
          0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1,
          1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
          0, 1, 1, 0, 0, 0]
y_test = [func(i) for i in y_test]

y_pr1 = pd.read_csv("data/tree_y_pr.csv", index_col='Unnamed: 0').values[:, [1, 0]]
y_pred1 = [list(i).index(max(i)) for i in y_pr1]

y_pr2 = pd.read_csv("data/bayes_y_pr.csv", index_col='Unnamed: 0').values[:, [1, 0]]
y_pred2 = [list(i).index(max(i)) for i in y_pr2]

y_pr3 = pd.read_csv("data/svm_y_pr.csv", index_col='Unnamed: 0').values[:, [1, 0]]
y_pred3 = [list(i).index(max(i)) for i in y_pr3]

y_pr4 = pd.read_csv("data/cnn_y_pr.csv", index_col='Unnamed: 0').values[:, [1, 0]]
y_pred4 = [list(i).index(max(i)) for i in y_pr4]

# print("tree:")
# eve(y_test, y_pred1)
# print("=======================")
# print("bayes:")
# eve(y_test, y_pred2)
# print("=======================")
# print("svm:")
# eve(y_test, y_pred3)
# print("=======================")
# print("cnn:")
# eve(y_test, y_pred4)
# print("=======================end")

# sys.exit(0)
# 画图ROC
y_test1 = []
for i in y_test:
    if i == 0:
        y_test1.append([1, 0])
    if i == 1:
        y_test1.append([0, 1])
y_test = np.array(y_test1)
print(y_test.shape)
plt.figure()
lw = 2

colors = ["-.", ":", "--", '-']
yps = [y_pr1, y_pr2, y_pr4, y_pr3]
models = ["DTree ", "Bayes ", "CNN ", "GA-SVM "]
for j in range(4):
    y_score = yps[j]
    print(y_score.shape)
    fpr = dict()
    tpr = dict()
    ro1 = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

        ro1[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[1], tpr[1], linestyle=colors[j], color='black',
             lw=lw, label=models[j] + 'ROC curve (area = %0.2f)' % ro1[1])

plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("data/roc1.png", dpi=300)
plt.show()
