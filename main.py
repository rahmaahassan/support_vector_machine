import pandas as pd  # loading data from datasets
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('notebook')   # window
sns.set_style('white')

from scipy.io import loadmat   # Define dataset.mat
from sklearn import svm      # Support Vector Classification

pd.set_option('display.notebook_repr_html', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)


def plotData(X, y, S):
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()

    plt.scatter(X[pos, 0], X[pos, 1], s=S, c='b', marker='+', linewidths=1)
    plt.scatter(X[neg, 0], X[neg, 1], s=S, c='r', marker='o', linewidths=1)
    plt.show()


def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plotData(X, y, 6)

    sv = svc.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='|', s=100, linewidths='5')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vector: ', svc.support_vectors_.size)

# Ex 1: linear SVM

# data1 = loadmat('ex1data1.mat')
# # print(data1)
#
# y1 = data1['y']
# X1 = data1['X']
#
# print('X1', X1.shape)
# print('y1', y1.shape)
#
# plotData(X1, y1, 50)
#
# clf = svm.SVC(C=100, kernel='linear')
# clf.fit(X1, y1.ravel())
# plot_svc(clf, X1, y1)

# Ex 2: Nonlinear SNM

# data2 = loadmat('ex2data2.mat')
# # print(data2)
#
# y2 = data2['y']
# X2 = data2['X']
# #
# # print('X2:', X2.shape)
# # print('y2', y2.shape)
#
# plotData(X2, y2, 8)
#
# clf2 = svm.SVC(C=50, kernel='rbf', gamma=25)
# clf2.fit(X2, y2.ravel())
# plot_svc(clf2, X2, y2)

# Training

spam_train = loadmat('spamTrain.mat')
spam_test = loadmat('spamTest.mat')

X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()

print(X.shape, y.shape, Xtest.shape, ytest.shape)

svc = svm.SVC()
svc.fit(X, y)
#
# Testing
print('Test Accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))
