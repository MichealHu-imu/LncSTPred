import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import string
import os
from pickle import load
import os
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


data = pd.read_csv('/Documents/predictive-dataset.csv')

# 分离标签和特征
labels = data.iloc[:, 0]
features = data.iloc[:, 1:]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


def models_N_weights(X, y, M, k):
    model = []
    model_weights = []
    training_errors = []

    N = X.shape[0]
    w = np.ones(N) / N

    for m in range(M):
        h = DecisionTreeClassifier(max_depth=5)
        h.fit(X, y, sample_weight=w)
        pred = h.predict(X)

        eps = w.dot(pred != y)
        alpha = (np.log((1 - eps) * (k - 1)) - np.log(eps))
        for i in range(N):
            if (y[i] == pred[i]):
                w[i] = w[i] * np.exp(-alpha)
            else:
                w[i] = w[i] * np.exp(alpha)
        w = w / w.sum()

        model.append(h)
        model_weights.append(alpha)

    return [model, model_weights]

def predict_joined_models(X, model, model_weights, frame, k):
    pred = model[k].predict(X)
    for i, idx in enumerate(frame.index):
        t = frame._get_value(idx, pred[i])
        frame._set_value(idx, pred[i], t + model_weights[k])
    return frame.idxmax(axis=1)

def error_func(y, y_hat):
    correct_pred = map(lambda t1, t2: t1 == t2, y, y_hat)
    correct_pred = list(correct_pred)
    Err = 1 - float(sum(correct_pred))/len(correct_pred)
    return Err

M = 5000
k = 5
columns = list(string.ascii_uppercase[:5])
M_list = []
train_err_list = []
test_err_list = []
N1= X_train.shape[0]
frame1 = DataFrame(np.zeros([N1,5]),columns=columns)
N2= X_test.shape[0]
frame2 = DataFrame(np.zeros([N2,5]),columns=columns)
model_fit = models_N_weights(X_train, y_train, M, k)
for m in range(M):
    y_hat_train = predict_joined_models(X_train, model_fit[0], model_fit[1], frame1, m)
    err = error_func(y_train, y_hat_train)
    train_err_list.append(err)

    y_hat_test = predict_joined_models(X_test, model_fit[0], model_fit[1], frame2, m)
    err = error_func(y_test, y_hat_test)
    test_err_list.append(err)
    M_list.append(m)
print(err)
y_hat_test.head()

err_df = DataFrame({}, columns=['train', 'test'])
err_df['train'] = train_err_list
err_df['test'] = test_err_list
err_df.head()
err_df.to_csv('/Documents/error.csv',encoding='utf-8', index=False)

plt.plot(M_list, train_err_list, c= 'red', label = 'train_error', linestyle='-')
plt.plot(M_list, test_err_list, c= 'green', label = 'test_error', linestyle='-')
plt.xlabel('number of weak learners')
plt.ylabel('Error')
plt.title('Error x Number of models')
plt.legend(loc = 'upper right')
plt.show()