import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import os
import errno


def sigmoid(t):  # Define the sigmoid function
    return (1 / (1 + np.exp(-t)))


def generate_logistic_data(dataSize, dT):
    X = np.random.randn(dataSize, 100)
    # beta = 10.0 / np.power(2, range(0,dataSize))
    beta = 10.0 / np.power(range(1, 100 + 1), 1)
    mu = X.dot(beta.reshape([100, 1]))
    y = np.random.binomial(1, sigmoid(mu), size=None)
    return X, y.ravel()


def getLoss(logistic, X, y):
    dataSize = X.shape[0]
    P = logistic.predict_proba(X)
    tmp = P[np.arange(dataSize), y]
    tmp[tmp == 0.0] = np.nextafter(0, 1)
    loss = -np.mean(np.log(tmp))
    return loss


def introplot():
    dataSize1 = 100
    dataSize2 = 200
    OracleSize = 10000
    dT = 50
    X1, y1 = generate_logistic_data(dataSize1, dT)
    X2, y2 = generate_logistic_data(dataSize2, dT)
    XO, yO = generate_logistic_data(OracleSize, dT)
    Result = {'1': [], '2': [], 'O1': [], 'O2': []}
    for i in range(dT):
        logistic1 = linear_model.LogisticRegression()
        logistic2 = linear_model.LogisticRegression()
        logistic1.fit(X1[:, :(i + 1)], y1)
        Result['1'].append(getLoss(logistic1, X1[:, :(i + 1)], y1))
        Result['O1'].append(getLoss(logistic1, XO[:, :(i + 1)], yO))
        logistic2.fit(X2[:, :(i + 1)], y2)
        Result['2'].append(getLoss(logistic2, X2[:, :(i + 1)], y2))
        Result['O2'].append(getLoss(logistic2, XO[:, :(i + 1)], yO))
    best_O1 = np.argmin(Result['O1'])
    best_O2 = np.argmin(Result['O2'])
    plt.figure(figsize=(3, 5), dpi=300, facecolor='w', edgecolor='k')
    plt.plot(range(1, dT + 1), Result['1'], 'b', label=('data size:%d' % dataSize1), linewidth=2)
    plt.plot(range(1, dT + 1), Result['2'], 'r', label=('data size:%d' % dataSize2), linewidth=2, linestyle='--')
    plt.xlim(1, dT)
    plt.xlabel('Dimension', fontsize=10, color='black')
    plt.ylabel('In sample Loss', fontsize=10, color='black')
    # plt.title('In sample Loss')
    plt.legend(loc='upper right', prop={'size': 10})
    plt.grid()
    plt.tight_layout()
    plt.savefig('./output/logistic/intro/intro1.png', bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(3, 5), dpi=300, facecolor='w', edgecolor='k')
    plt.plot(range(1, dT + 1), Result['O1'], 'b', label=('data size:%d' % dataSize1), linewidth=2)
    plt.scatter(best_O1 + 1, Result['O1'][best_O1], marker='o', facecolors='none', edgecolors='b', s=50)
    plt.plot(range(1, dT + 1), Result['O2'], 'r', label=('data size:%d' % dataSize2), linewidth=2, linestyle='--')
    plt.scatter(best_O2 + 1, Result['O2'][best_O2], marker='D', facecolors='none', edgecolors='r', s=50)
    plt.xlim(1, dT)
    plt.xlabel('Dimension', fontsize=10, color='black')
    plt.ylabel('Prediction Loss', fontsize=10, color='black')
    # plt.title('Prediction Loss')
    plt.legend(loc='upper right', prop={'size': 10})
    plt.grid()
    plt.tight_layout()
    plt.savefig('./output/logistic/intro/intro2.png', bbox_inches='tight')
    plt.show()


def main():
    try:
        os.makedirs('output/logistic')
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    introplot()


if __name__ == "__main__":
    main()