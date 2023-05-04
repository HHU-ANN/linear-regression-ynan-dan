# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np


def ridge(data):
    x, y = read_data()
    i = np.eye(6)
    alpha = -0.1
    weight = np.dot(np.linalg.inv(np.dot(x.T, x) + np.dot(alpha, i)), np.dot(x.T, y))
    print(np.dot(alpha, i))
    return data @ weight
    pass

def lasso(data):
    x, y = read_data()
    weight = np.array([0, 0, 0, 0, 0, 0])
    label = 1e-9
    alpha = 0.6
    r = 4e-10
    loss_past = 0
    for i in range(int(1e6)):
        z = np.dot(x, weight)
        loss = np.dot((z - y).T, z - y) + alpha * np.sum(abs(weight))
        if abs(loss - loss_past) < label:
            break
        loss_past = loss
        dw = 2 * np.dot(z-y, x) + alpha * np.sign(weight)
        weight = weight - r * dw
    return data @ weight


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y



