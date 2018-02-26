# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
# from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline

def getConservation(filename):
    file = open(filename)
    listConservation = []
    str = file.readline()
    for astr in str.split(', '):
        listConservation.append(int(astr))
    return (listConservation)

listConservation = getConservation(r'skempiconservation.txt')
df = pd.read_excel(r'../../excell/450_3.xlsx')

# print (df)
# gbdt=GradientBoostingClassifier(n_estimators=200)

y_train = np.arange(400)
y_test = np.arange(50)
list1 = list(df.ix[0:400, 7])
list2 = list(df.ix[400:450, 7])
for i in range(400):
    y_train[i] = list1[i]
for i in range(50):
    y_test[i] = list2[i]
X_train = np.zeros((400, 6))
X_test = np.zeros((50, 6))
list3 = []
for j in range(5):
    # print(df.ix[:, j])
    # print(j)
    list3.append(list(df.ix[:, j + 8]))
# print(list3)
# list3.append(list(df.ix[: , 26]))
# list3.append(list(df.ix[: , 27]))
for j in range(5):
    for i in range(400):
        X_train[i][j] = list3[j][i]
for i in range(400):
    X_train[i][5] = listConservation[i]
print(X_train.shape)
for j in range(5):
    for i in range(50):
        X_test[i][j] = list3[j][400 + i]
for i in range(50):
    X_test[i][5] = listConservation[400+i]
X_train = X_train[:, [5]]
x_train = X_train.mean()
X_train = X_train-x_train
X_test = X_test[:, [5]]
x_test = X_test.mean()
X_test = X_test - x_test

# t = np.arctan2(X_test, y_test)

plt.plot(listConservation[0:400], list(df.ix[0:399, 7]), 'ro')
plt.axis([-8, 8, 0, 1])
plt.show()


# print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)
'''
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.5)
n_estimator = 10
grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)


# The gradient boosted model by itself
y_pred_grd = grd.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
#plt.plot(fpr_rf, tpr_rf, label='RF')
#plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.plot(fpr_grd, tpr_grd, label='GBT')
# plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.xlim(0, 1)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
#plt.plot(fpr_rf, tpr_rf, label='RF')
#plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.plot(fpr_grd, tpr_grd, label='GBT')
#plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
'''
params = {'n_estimators': 1000, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.015, 'min_samples_leaf': 1, 'random_state': 3}
clf = GradientBoostingClassifier(**params)

clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("Accuracy: {:.4f}".format(acc))
pred = clf.predict(X_test)
# print(pred,y_test)
acc1 = 0
tp = 0
fn = 0
fp = 0
'''
for i in range(49):
    if pred[i] == y_test[i]:
        acc1 = acc1 + 1
        if pred[i] == 1:
            tp = tp + 1
    elif pred[i] == 0 and y_test[i] == 1:
        fn = fn + 1
    else:
        fp = fp + 1
print(tp, fn, fp)
re = tp / (tp + fn)
pre = tp / (tp + fp)
fscore = 2 * re * pre / (re + pre)
print(re, pre, fscore)
print("Accuracy: {:.4f}".format(acc))
'''