import numpy as np
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import matplotlib
import os
import errno


def crossvalition_loo(X, y):
    loo = KFold(n_splits=X.shape[0])
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for train_index, test_index in loo.split(X, y):
        X_train.append(X[train_index, :])
        y_train.append(y[train_index])
        X_test.append(X[test_index])
        y_test.append(y[test_index])
    # print(X_train.shape)
    # print(X_test[0].shape)
    # print(X.shape)
    return (X_train, X_test, y_train, y_test)


def generate_data(dataSize):
    # np.random.seed(seed)
    # X, y = datasets.make_moons(dataSize, noise=0.20)
    X, y = datasets.make_circles(n_samples=dataSize, shuffle=True, noise=0.1, random_state=None, factor=0.6)
    y = y.reshape((-1, 1))
    return X[:, :2], y[:, :2]


def getLoss(X, y, mlp):
    dataSize = X.shape[0]
    P = mlp.predict_proba(X)
    # tmp=P[np.arange(dataSize), y]
    # tmp[tmp==0.0] = np.nextafter(0, 1)
    loglik = np.mean(np.log(P))
    return -loglik


def introplot():
    dataSize1 = 100
    dataSize2 = 300
    X1, y1 = generate_data(dataSize1)
    y1 = y1[:, 0]
    X2, y2 = generate_data(dataSize2)
    y2 = y2[:, 0]
    dim = [1, 2, 3, 4, 5, 6, 10, 20]
    mlp1 = []
    mlp2 = []
    for i in range(len(dim)):
        print(i)
        cur_dim = dim[i]
        cur_mlp1 = MLPClassifier(hidden_layer_sizes=(cur_dim,), activation='tanh', solver='lbfgs')
        cur_mlp1.fit(X1, y1)
        mlp1.append(cur_mlp1)
        # X_train, X_test, y_train, y_test=crossvalition_loo(X1, y1)
        # Loss=np.zeros(len(X_train))
        # for i in range(len(X_train)):
        # cur_mlp1.fit(X_train[i], y_train[i])
        # mlp1.append(cur_mlp1)
        # Loss[i] = getLoss(X_test[i], y_test[i], cur_mlp1)
        # best_model1 = np.argmin(Loss)
        plot_decision_boundary(cur_mlp1, X1, y1, cur_dim)

        cur_mlp2 = MLPClassifier(hidden_layer_sizes=(cur_dim,), activation='tanh', solver='lbfgs')
        cur_mlp2.fit(X2, y2)
        mlp2.append(cur_mlp2)
        # X_train, X_test, y_train, y_test=crossvalition_loo(X2, y2)
        # Loss=np.zeros(len(X_train))
        # for i in range(len(X_train)):
        # cur_mlp2.fit(X_train[i], y_train[i])
        # mlp2.append(cur_mlp2)
        # Loss[i] = getLoss(X_test[i], y_test[i], cur_mlp1)
        # best_model2 = np.argmin(Loss)
        plot_decision_boundary(cur_mlp2, X2, y2, cur_dim)


def plot_decision_boundary(pred_func, X, y, dim):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    # Z[Z==0] = np.min(Z[Z!=0])
    # print(Z)
    plt.figure()
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.tight_layout()
    plt.grid()
    plt.savefig(('./output/neuralnet/intro/%d_%d.png' % (X.shape[0], dim)), bbox_inches='tight')
    # plt.show()


def main():
    try:
        os.makedirs('output/neuralnet')
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    introplot()


if __name__ == "__main__":
    main()