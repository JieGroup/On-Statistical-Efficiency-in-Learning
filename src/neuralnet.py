from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os

os.environ["THEANO_FLAGS"] = "floatX=float64"
import sys
import timeit
import numpy as np
from learning import getSeqEdvice
import theano
import theano.tensor as T
from theano.tensor.nlinalg import matrix_inverse
import matplotlib.animation as manimation
from mpl_toolkits import axes_grid1
from sklearn import datasets, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import time
import copy
import pickle
import os
import errno
import warnings

warnings.filterwarnings("ignore")


class HiddenLayer(object):
    def __init__(self, n_data, dim_in, dim_out, activation=T.tanh):
        self.n_data = n_data
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation

    def setParam(self, param_W, param_b):
        # beta=np.float32(param[0])
        # bias=np.float32(param[1])
        # DesignMatrix = np.concatenate((beta,bias.reshape((1,self.dim_out))),axis=0)
        # tmp_param = DesignMatrix.reshape(((self.dim_in+1)*self.dim_out,))
        # self.param = theano.shared(
        # value=tmp_param,
        # name='param',
        # borrow=True
        # )
        # tmp = self.param.reshape(((self.dim_in+1, self.dim_out)))
        # self.W = tmp[:self.dim_in,]
        # self.b = tmp[-1,]
        self.W = param_W
        self.b = param_b

    def setData(self, input, ifData):
        if (ifData):
            self.input = theano.shared(
                value=input,
                name='input',
                borrow=True
            )
        else:
            self.input = input
        # self.output = theano.shared(
        # value=np.zeros((self.n_data,self.dim_out),dtype=theano.config.floatX),
        # name='output',
        # borrow=True
        # )

    def setup(self, param_W, param_b, input, ifData):
        self.setParam(param_W, param_b)
        self.setData(input, ifData)
        lin_output = T.dot(self.input, self.W) + self.b
        # self.output.set_value(lin_output if activation is None else activation(lin_output))
        self.output = lin_output if self.activation is None else self.activation(lin_output)


class LogisticRegression(object):

    def __init__(self, n_data, dim_in, dim_out):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_data = n_data

    def setParam(self, param_W, param_b):
        # beta=np.float32(param[0])
        # bias=np.float32(param[1])
        # DesignMatrix = np.concatenate((beta,bias.reshape((1,self.dim_out))),axis=0)
        # tmp_param = DesignMatrix.reshape(((self.dim_in+1)*self.dim_out,))
        # self.param = theano.shared(
        # value=tmp_param,
        # name='param',
        # borrow=True
        # )
        # tmp = self.param.reshape(((self.dim_in+1, self.dim_out)))
        # self.W = tmp[:self.dim_in,]
        # self.b = tmp[-1,]
        self.W = param_W
        self.b = param_b

    def setData(self, input, dataY):
        self.input = input
        self.output = theano.shared(
            value=dataY,
            name='output',
            borrow=True
        )

    def setup(self, param_W, param_b, input, dataY):
        self.setParam(param_W, param_b)
        self.setData(input, dataY)
        self.p_y_given_x = T.nnet.sigmoid(T.dot(self.input, self.W) + self.b)
        # print(self.p_y_given_x.eval())

    def negative_log_likelihood_vector(self):
        if (self.dim_out == 1):
            tmp1 = T.concatenate([(1 - self.p_y_given_x), (self.p_y_given_x)], axis=1)[
                T.arange(self.n_data), self.output]
            # print(T.log(tmp1).eval())
            # tmp1[tmp1==0.0] = np.nextafter(0, 1)
            tmp = T.log(tmp1)
        else:
            tmp1 = self.p_y_given_x[T.arange(self.n_data), self.output]
            # print(tmp1.eval())
            # tmp1[tmp1==0.0] = np.nextafter(0, 1)
            tmp = T.log(tmp1)
        return -tmp

    def negative_log_likelihood_mean(self):
        if (self.dim_out == 1):
            tmp1 = T.concatenate([(1 - self.p_y_given_x), (self.p_y_given_x)], axis=1)[
                T.arange(self.n_data), self.output]
            # print(tmp1.eval())
            # tmp1[tmp1==0.0] = np.nextafter(0, 1)
            tmp = T.log(tmp1)
        else:
            tmp1 = self.p_y_given_x[T.arange(self.n_data), self.output]
            # print(tmp1.eval())
            # tmp1[tmp1==0.0] = np.nextafter(0, 1)
            tmp = T.log(tmp1)
            # print(tmp.shape.eval())
        return -T.mean(tmp)


class MLP(object):

    def __init__(self, n_data, dim_in, dim_hidden, dim_out):
        self.n_hidden = len(dim_hidden)
        self.n_data = n_data
        self.hiddenlayers = []
        self.numtotalparam = 0
        for i in range(self.n_hidden):
            if (i == 0):
                cur_dim_in = dim_in + 1
            else:
                cur_dim_in = dim_hidden[i - 1] + 1
            self.numtotalparam = self.numtotalparam + cur_dim_in * dim_hidden[i]
            self.hiddenlayers.append(HiddenLayer(
                n_data=n_data,
                dim_in=cur_dim_in,
                dim_out=dim_hidden[i],
                activation=T.tanh
            ))
        self.numtotalparam = self.numtotalparam + (dim_hidden[-1] + 1) * dim_out
        self.logRegressionLayer = LogisticRegression(
            n_data=n_data,
            dim_in=dim_hidden[-1] + 1,
            dim_out=dim_out
        )

    def setup(self, param, dataX, dataY):
        # print(dataY.shape)
        tmp_param = np.zeros(self.numtotalparam)
        pivot = 0
        tmp_DesignMatrixshape = []
        tmp_numparam = np.zeros(len(param), dtype=np.int)
        for i in range(len(param)):
            tmp_beta = param[0][i]
            tmp_bias = param[1][i]
            DesignMatrix = np.concatenate((tmp_beta, tmp_bias.reshape((1, tmp_bias.shape[0]))), axis=0)
            tmp_DesignMatrixshape.append(DesignMatrix.shape)
            tmp_numparam[i] = tmp_DesignMatrixshape[i][0] * tmp_DesignMatrixshape[i][1]
            tmp_param[pivot:(pivot + tmp_numparam[i])] = DesignMatrix.reshape((tmp_numparam[i],))
            pivot = pivot + tmp_numparam[i]

        # print(tmp_param)
        self.param = theano.shared(
            value=tmp_param,
            name='param',
            borrow=True
        )
        pivot = 0
        cur_input = dataX
        for i in range(len(param)):
            tmp = self.param[pivot:(pivot + tmp_numparam[i])].reshape(
                (tmp_DesignMatrixshape[i][0], tmp_DesignMatrixshape[i][1]))
            pivot = pivot + tmp_numparam[i]
            # print(tmp_DesignMatrixshape[i][0]-1)
            if (i == 0):
                self.hiddenlayers[i].setup(tmp[:tmp_DesignMatrixshape[i][0] - 1, ], tmp[-1,], cur_input, True)
                cur_input = self.hiddenlayers[i].output
            elif (i < (len(param) - 1)):
                self.hiddenlayers[i].setup(tmp[:tmp_DesignMatrixshape[i][0] - 1, ], tmp[-1,], cur_input, False)
                cur_input = self.hiddenlayers[i].output
            else:
                self.logRegressionLayer.setup(tmp[:tmp_DesignMatrixshape[i][0] - 1, ], tmp[-1,], cur_input, dataY)

    def negative_log_likelihood_vector(self):
        return (self.logRegressionLayer.negative_log_likelihood_vector())

    def negative_log_likelihood_mean(self):
        return (self.logRegressionLayer.negative_log_likelihood_mean())

    def GTIC_gradient(self):
        NLLV = self.negative_log_likelihood_vector()
        O = T.dmatrix('O')
        o = np.zeros((self.numtotalparam, self.numtotalparam))
        results, updates = theano.scan(lambda i, tmp: T.dot(T.grad(NLLV[i], self.param).reshape((-1, 1)),
                                                            T.grad(NLLV[i], self.param).reshape((-1, 1)).T) + tmp,
                                       sequences=[np.arange(self.n_data)],
                                       outputs_info=[O])
        result = results[-1]
        compute = theano.function([O], outputs=result)
        out = compute(o) / self.n_data
        return out

    def GTIC_hessian(self):
        NLLM = self.negative_log_likelihood_mean()
        # print(NLLM.eval())
        H_theta = theano.gradient.hessian(NLLM, self.param)
        # print(H_theta.eval())
        return H_theta

    def GTIC(self):
        # print(self.GTIC_hessian().eval())
        # your code that will (maybe) throw
        inverse = T.nlinalg.pinv(self.GTIC_hessian())
        tmp = T.nlinalg.trace(T.dot(inverse, self.GTIC_gradient())) / self.n_data
        return tmp


def sigmoid(t):  # Define the sigmoid function
    return (1 / (1 + np.exp(-t)))


def getLoss(X, y, mlp):
    P = mlp.predict_proba(X)
    n = X.shape[0]
    loglik = np.mean(np.log(P[np.arange(n), y]))
    return -loglik


def getAIC(n, k):
    return k / n


def getBIC(n, k):
    return (k * np.log(n)) / (2 * n)


def getDIC_1(n, k_AIC, Mn):
    adjust = 0
    for i in range(k_AIC):
        adjust = adjust + 1 / (i + 1)
    return Mn / n * adjust


def getDIC_2(n, k, kmax):
    adjust = 0
    for i in range(k):
        adjust = adjust + 1 / (i + 1)
    return kmax / n * adjust


def calculate_Loss(candModels, X_test, y_test):
    y_test = y_test[:, 0]
    N = len(candModels)
    loss = np.zeros((N, 1))
    dataSize = X_test.shape[0]
    for n in range(N):
        mlp = candModels[n]['mlp']
        # plot_decision_boundary(lambda x: mlp.predict(x), X_test, y_test)
        P = mlp.predict_proba(X_test)
        tmp = P[np.arange(dataSize), y_test]
        tmp[tmp == 0.0] = np.nextafter(0, 1)
        # plt.hist(tmp,bins=50)
        # plt.show()
        loss[n] = -np.mean(np.log(tmp))
    return loss


def generate_data(dataSize, seed):
    # np.random.seed(seed)
    # X, y = datasets.make_moons(dataSize, noise=0.20)
    X, y = datasets.make_circles(n_samples=dataSize, shuffle=True, noise=0.2, random_state=None, factor=0.8)
    y = y.reshape((-1, 1))
    return X[:, :2], y[:, :2]


def NeuralNetworkFit(candModels, Xt, yt, timing):
    GTICcompiletiming = 0
    N = len(candModels)
    kmax = 0
    Loss_AIC = np.inf
    for n in range(N):
        print(n)
        mlp_cv = MLPClassifier(hidden_layer_sizes=((n + 1),), activation='tanh', solver='lbfgs')

        if (n < np.floor((np.sqrt(Xt.shape[0]) / (Xt.shape[1] + 1)) - 1).astype(np.int) or n < 5):
            # Holdout
            start = time.time()
            X_train, X_test, y_train, y_test = crossvalition_p(Xt, yt, 0.7)
            mlp_cv.fit(X_train, y_train)
            Loss = getLoss(X_test, y_test, mlp_cv)
            candModels[n]['Holdout'] = Loss
            end = time.time()
            timing['Holdout'] = timing['Holdout'] + end - start
            # CV_10fold
            start = time.time()
            X_train, X_test, y_train, y_test = crossvalition_10fold(Xt, yt)
            Loss = np.zeros(len(X_train))
            for i in range(len(X_train)):
                mlp_cv.fit(X_train[i], y_train[i])
                Loss[i] = getLoss(X_test[i], y_test[i], mlp_cv)
            candModels[n]['CV_10fold'] = np.mean(Loss)
            end = time.time()
            timing['CV_10fold'] = timing['CV_10fold'] + end - start
            # CV_loo
            start = time.time()
            X_train, X_test, y_train, y_test = crossvalition_loo(Xt, yt)
            Loss = np.zeros(len(X_train))
            for i in range(len(X_train)):
                mlp_cv.fit(X_train[i], y_train[i])
                Loss[i] = getLoss(X_test[i], y_test[i], mlp_cv)
            candModels[n]['CV_loo'] = np.mean(Loss)
            end = time.time()
            timing['CV_loo'] = timing['CV_loo'] + end - start

        mlp = MLPClassifier(hidden_layer_sizes=((n + 1),), activation='tanh', solver='lbfgs')
        start = time.time()
        mlp.fit(Xt, yt)
        Loss = getLoss(Xt, yt, mlp)
        candModels[n]['mlp'] = mlp
        dataSize = Xt.shape[0]
        dim = (Xt.shape[1] + 1) * ((n + 1) + 1)
        end = time.time()
        time_loss = end - start

        if (n < np.floor((np.sqrt(Xt.shape[0]) / (Xt.shape[1] + 1)) - 1).astype(np.int) or n < 5):
            start_AIC = time.time()
            AIC = getAIC(dataSize, dim)
            candModels[n]['AIC'] = Loss + AIC
            end_AIC = time.time()
            timing['AIC'] = timing['AIC'] + time_loss + end_AIC - start_AIC

            start_BIC = time.time()
            BIC = getBIC(dataSize, dim)
            candModels[n]['BIC'] = Loss + BIC
            end_BIC = time.time()
            timing['BIC'] = timing['BIC'] + time_loss + end_BIC - start_BIC

            start_DIC_2 = time.time()
            DIC_2 = getDIC_2(dataSize, dim, dataSize ** (1 / 2))
            candModels[n]['DIC_2'] = Loss + DIC_2
            end_DIC_2 = time.time()
            timing['DIC_2'] = timing['DIC_2'] + time_loss + end_DIC_2 - start_DIC_2

            start_GTIC_compile = time.time()
            beta = mlp.coefs_
            bias = mlp.intercepts_
            dataX = Xt
            dataY = yt
            GTICKernel = MLP(n_data=dataX.shape[0], dim_in=dataX.shape[1], dim_hidden=[(n + 1)], dim_out=1)
            GTICKernel.setup([beta, bias], dataX, dataY)
            # loss = GTICKernel.negative_log_likelihood_mean()
            tic = GTICKernel.GTIC()
            GTIC_model = theano.function(
                inputs=[],
                outputs=tic,
            )
            end_GTIC_compile = time.time()
            GTICcompiletiming = GTICcompiletiming + end_GTIC_compile - start_GTIC_compile
            # print(GTIC_compile_timing)

            # print(Loss)
            # a=np.tanh(dataX.dot(beta[0])+bias[0]).dot(beta[1])+bias[1]
            # print(-(np.sum(a[dataY==1]) - np.sum(np.log(1+np.exp(a))))/dataX.shape[0])
            start_GTIC = time.time()
            GTIC = GTIC_model()
            # print(Loss)
            # print(GTIC)
            candModels[n]['GTIC'] = Loss + GTIC
            print(candModels[n]['GTIC'])
            end_GTIC = time.time()
            timing['GTIC'] = timing['GTIC'] + time_loss + end_GTIC - start_GTIC

            start_DIC_1 = time.time()
            if (candModels[n]['AIC'] < Loss_AIC):
                Loss_AIC = candModels[n]['AIC']
                kmax = n
            end_DIC_1 = time.time()
            timing['DIC_1'] = timing['DIC_1'] + time_loss + end_DIC_1 - start_DIC_1

    start_DIC_1 = time.time()
    for n in range(N):
        if (n < kmax):
            dataSize = Xt.shape[0]
            dim = Xt.shape[1] * ((n + 1) + 1)
            DIC_1 = getDIC_1(dataSize, dim, dataSize ** (1 / 3))
            candModels[n]['DIC_1'] = Loss + DIC_1
        else:
            candModels[n]['DIC_1'] = 0
    end_DIC_1 = time.time()
    timing['DIC_1'] = timing['DIC_1'] + end_DIC_1 - start_DIC_1

    return candModels, timing, GTICcompiletiming


def runLogisticExper():
    # %% pre-def the observation up to time T
    T = 301
    # dT = np.floor(T/2).astype(int) #largest candidate model considered
    # dT = np.sqrt(T).astype(int)
    dT = 20
    K = 3  # num of active models we maintain in sequential learning
    if dT < K:
        print('error in dT specification!')
    t_start = 101  # starting data size

    nJumpExpe = 1 * np.sqrt(T) / K  # expected #jump
    learningParams = {'eta': 1 * np.sqrt(8 * nJumpExpe / T), 'alpha': 2 * nJumpExpe / T,
                      'G': None}  # will use getSeqEdvice()
    actSet, move, thresh = range(K), False, 1 / K  # input to learning
    subW = np.zeros((K))
    subW[0] = 1
    N = 10000

    # pre-def the benchmark (testdata) to compute the expected loss
    X_test, y_test = generate_data(N, 100)  # testing data for loss computation
    X, y = generate_data(T, 0)  # all the training data
    # print(X.shape)

    # pre-def the fixed candidates
    candModels_fix = []
    for t in range(dT):
        candModels_fix.append({'mlp': None, 'AIC': None, 'BIC': None, 'DIC_1': None, 'DIC_2': None, 'GTIC': None, \
                               'Holdout': None, 'CV_10fold': None, 'CV_loo': None})
    L = -np.inf * np.ones((dT, T))  # store all the loss (each col) at each time
    candModels_time = []
    timing_image = {'AIC': [], 'BIC': [], 'DIC_1': [], 'DIC_2': [], 'GTIC': [], \
                    'Holdout': [], 'CV_10fold': [], 'CV_loo': []}
    # sequential procedure -- compute the loss for all candidates
    timing = {'AIC': 0, 'BIC': 0, 'DIC_1': 0, 'DIC_2': 0, 'GTIC': 0, \
              'Holdout': 0, 'CV_10fold': 0, 'CV_loo': 0}
    for t in range(t_start - 1, T):  # for each sample size
        print(t)
        Xt = X[range(t), :]
        yt = y[range(t), 0]
        candModels_fix, timing, GTICcompiletiming = NeuralNetworkFit(candModels_fix, Xt, yt, timing)
        # print(GTIC_compile_timing)
        for k in timing_image.keys():
            if (k == 'GTIC'):
                timing_image[k].append(timing[k] + GTICcompiletiming)
            else:
                timing_image[k].append(timing[k])
        # compute the loss matrix
        L[:, t] = np.squeeze(calculate_Loss(candModels_fix, X_test, y_test))
        print(L[:, t])
        for t in range(dT):
            candModels_time.append(copy.deepcopy(candModels_fix))
        # print(candModels_time[t-t_start+1][4]['GTIC'])
    print(L[:, 20])
    print(L[:, T - 20])
    L[L == -np.inf] = np.max(L)  # the none are set to be the max loss

    # sequential procedure -- compute the est loss and use graph-based learning
    actSet_start, actSet_end = np.zeros((T)), np.zeros((T))
    W_hy = np.zeros((dT, T))
    # seqPredLoss = np.zeros((T))  
    # loss_ratio =  np.zeros((1,T)) 

    output_filename = './output/neuralnet/output.pickle'
    # mode = ['AIC','BIC','DIC_1','DIC_2','Holdout','CV_10fold','CV_loo','GTIC']
    # mode = ['AIC','BIC','DIC_1','DIC_2','GTIC','Holdout','CV_10fold','CV_loo']
    mode = ['GTIC', 'Holdout', 'CV_10fold', 'CV_loo']
    # mode = ['GTIC','AIC','BIC','DIC_1']
    # mode = ['GTIC','Holdout','AIC','BIC']
    # mode = ['Holdout','AIC','BIC']
    candModels_Sequential = []
    for i in range(len(mode)):
        candModels_Sequential.append(
            {'learningParams': learningParams, 'actSet': actSet, 'actSet_start': actSet_start, 'actSet_end': actSet_end,
             'move': move, 'nummove': 0, 'thresh': thresh, \
             'subW': subW, 'W_hy': W_hy, 'seqPredLoss': np.zeros((T)), 'loss_ratio': np.zeros((T)),
             'batch_opt_model': np.zeros((T), dtype=np.int), 'batch_opt_loss': np.zeros((T)) + np.inf,
             'batch_loss_ratio': np.zeros((T))})

    with open(output_filename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([candModels_Sequential, candModels_time, L, mode, t_start, T, timing_image], f)

    for t in range(t_start - 1, T):  # for each sample size
        # cur_dT = 10
        cur_dT = np.floor((np.sqrt(t) / (X.shape[1] + 1)) - 1).astype(np.int)
        seq_dT = K if (cur_dT < K) else cur_dT
        # print(t)
        # print(candModels_time[t-t_start+1][4]['GTIC'])
        for m in range(len(mode)):
            candModels_Sequential[m]['seqPredLoss'][t] = np.sum(candModels_Sequential[m]['subW'] * L[
                candModels_Sequential[m]['actSet'], t])  # masterE is wrong! should be real loss numerically computed
            if t % 10 == 0:
                print("At iteration t = ", t)
            if candModels_Sequential[m]['move']:
                candModels_Sequential[m]['actSet'] = [(x + 1) for x in candModels_Sequential[m]['actSet']]
                candModels_Sequential[m]['nummove'] = candModels_Sequential[m]['nummove'] + 1
                movedsample = t / candModels_Sequential[m]['nummove']
                tmp_nJumpExpe = movedsample / K
                candModels_Sequential[m]['learningParams']['alpha'] = 1 * tmp_nJumpExpe / T
                candModels_Sequential[m]['learningParams']['eta'] = 1 * np.sqrt(8 * tmp_nJumpExpe / T)
                print(candModels_Sequential[m]['learningParams']['alpha'])
                if max(candModels_Sequential[m]['actSet']) >= seq_dT:
                    candModels_Sequential[m]['actSet'] = range(seq_dT - K, seq_dT)
                    candModels_Sequential[m]['move'] = False
            candModels_Sequential[m]['actSet_start'][t], candModels_Sequential[m]['actSet_end'][t] = min(
                candModels_Sequential[m]['actSet']), max(candModels_Sequential[m]['actSet'])
            subE = np.array(
                [(candModels_time[t - t_start + 1][j][mode[m]]) for j in candModels_Sequential[m]['actSet']]).reshape(
                K, )
            subE = np.array([1 if x is None else x for x in subE])
            candModels_Sequential[m]['subW'], masterE, candModels_Sequential[m]['move'] = getSeqEdvice(subE,
                                                                                                       candModels_Sequential[
                                                                                                           m]['subW'],
                                                                                                       candModels_Sequential[
                                                                                                           m][
                                                                                                           'learningParams'], \
                                                                                                       candModels_Sequential[
                                                                                                           m]['move'],
                                                                                                       candModels_Sequential[
                                                                                                           m]['thresh'],
                                                                                                       t)
            candModels_Sequential[m]['W_hy'][candModels_Sequential[m]['actSet'], t] = candModels_Sequential[m]['subW']
            weight = candModels_Sequential[m]['subW']
            # candModels_Sequential[m]['loss_ratio'][t] = np.sum(L[candModels_Sequential[m]['actSet'],
            # t]*weight)/np.min(L[:,t])
            candModels_Sequential[m]['loss_ratio'][t] = L[np.argmax(candModels_Sequential[m]['W_hy'][:, t],
                                                                    axis=0), t] / np.min(L[:, t])
            # print(mode[m])
            for l in range(cur_dT):
                # print(candModels_time[t-t_start+1][l][mode[m]])
                if ((candModels_time[t - t_start + 1][l][mode[m]] is not None) and (
                        candModels_time[t - t_start + 1][l][mode[m]] < candModels_Sequential[m]['batch_opt_loss'][t])):
                    candModels_Sequential[m]['batch_opt_model'][t] = l
                    candModels_Sequential[m]['batch_opt_loss'][t] = candModels_time[t - t_start + 1][l][mode[m]]
            # print(candModels_Sequential[m]['batch_opt_loss'][t])
            candModels_Sequential[m]['batch_loss_ratio'][t] = L[candModels_Sequential[m]['batch_opt_model'][
                                                                    t], t] / np.min(L[:, t])

    # summarize results
    viewLoss(np.log(L), candModels_Sequential[mode.index('GTIC')]['actSet_start'],
             candModels_Sequential[mode.index('GTIC')]['actSet_end'])
    viewSeqWeight(candModels_Sequential[mode.index('GTIC')]['W_hy'], L)  # print subW
    # viewSeqLoss_all(mode, candModels_Sequential, L, t_start)
    # viewBatchLoss_all(mode, candModels_Sequential, L, t_start)
    # viewLossRatio_all(mode, candModels_Sequential, t_start, T)
    # viewBatchLossRatio_all(mode, candModels_Sequential, t_start, T)
    viewSeqBatchLossRatio(mode, candModels_Sequential, t_start, T)
    viewTiming(mode, timing_image, t_start, T)
    # Saving the objects:
    with open(output_filename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([candModels_Sequential, candModels_time, L, mode, t_start, T, timing_image], f)


# Getting back the objects:
# with open('objs.pickle') as f:  # Python 3: open(..., 'rb')
# obj0, obj1, obj2 = pickle.load(f)


#########################################################################################################################################################################################################    
def crossvalition_p(X, y, p):
    dataSize = X.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - p), random_state=40)
    return (X_train, X_test, y_train, y_test)


def crossvalition_10fold(X, y):
    tenfold = KFold(n_splits=10)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for train_index, test_index in tenfold.split(X, y):
        X_train.append(X[train_index, :])
        y_train.append(y[train_index])
        X_test.append(X[test_index, :])
        y_test.append(y[test_index])
    return (X_train, X_test, y_train, y_test)


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


#########################################################################################################################################################################################################
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def viewLoss(L_transformed, actSet_start, actSet_end):
    nCandi, T = L_transformed.shape
    optModelIndex = np.argmin(L_transformed, axis=0)  # the first occurrence are returned.
    plt.figure(figsize=(10, 10), dpi=300, facecolor='w', edgecolor='k')
    bg = plt.imshow(L_transformed, cmap=plt.cm.Spectral)  # smaller the better
    # plt.colorbar()
    add_colorbar(bg)
    # plot along the active sets
    plt.scatter(range(T), optModelIndex, marker='o', color='k', s=1)
    #    plt.scatter(range(T), actSet_start, marker='x', color='b', s=30)
    #    plt.scatter(range(T), actSet_end, marker='x', color='b', s=30)
    plt.xlim(0, T - 1)
    plt.xlabel('Data Size', fontsize=10, color='black')
    plt.ylim(0, nCandi - 1)
    plt.ylabel('Model Complexity', fontsize=10, color='black')
    plt.title('Predictive Loss (in log)', fontsize=10)
    # plt.tight_layout()
    plt.savefig('./output/neuralnet/loss.png', bbox_inches='tight')
    # plt.show()


def videoLoss(L_transformed, actSet_start, actSet_end):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Loss', artist='VGROUP', comment='Loss')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    nCandi, T = L_transformed.shape
    tmp_L_transformed = np.zeros((nCandi, T)) * np.nan
    optModelIndex = np.argmin(L_transformed, axis=0)  # the first occurrence are returned.
    # tmp_optModelIndex = np.zeros(len(optModelIndex)))* np.nan
    # fig=plt.figure(figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k')
    fig = plt.figure(facecolor='w', edgecolor='k')
    # plt.imshow(L_transformed, cmap=plt.cm.Spectral)  #smaller the better
    bg = plt.imshow(L_transformed, cmap=plt.cm.Spectral)
    # divider = make_axes_locatable(fig)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    add_colorbar(bg)

    # plot along the active sets
    # plt.scatter(range(T), optModelIndex, marker='o', color='k', s=30)
    dots = plt.scatter(range(T), optModelIndex, marker='o', color='k', s=10)
    #    plt.scatter(range(T), actSet_start, marker='x', color='b', s=30)
    #    plt.scatter(range(T), actSet_end, marker='x', color='b', s=30)
    plt.xlim(0, T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylim(0, nCandi)
    plt.ylabel('Model Complexity', fontsize=14, color='black')
    plt.title('Predictive Loss (in log)')
    with writer.saving(fig, "Loss.mp4", T):
        for i in range(T):
            tmp_L_transformed[:, i] = L_transformed[:, i]
            bg.set_array(tmp_L_transformed)
            x = range(i + 1)
            y = optModelIndex[:i + 1]
            offsets = [[x[j], y[j]] for j in range(i + 1)]
            dots.set_offsets(offsets)
            writer.grab_frame()


def viewSeqWeight(W_hy, L):
    dT, T = W_hy.shape
    plt.figure(figsize=(10, 10), dpi=300, facecolor='w', edgecolor='k')
    bg = plt.imshow(W_hy, cmap='hot')  # smaller the better
    # plt.colorbar()
    add_colorbar(bg)
    plt.xlim(0, T - 1)
    plt.xlabel('Data Size', fontsize=10, color='black')
    plt.ylim(0, dT - 1)
    plt.ylabel('Model Complexity', fontsize=10, color='black')
    plt.title('Learning Weight', fontsize=10)

    # plot along the true best model
    optModelIndex = np.argmin(L, axis=0)
    plt.scatter(range(T), optModelIndex, marker='o', color='b', s=5)
    # plt.tight_layout()
    plt.savefig('./output/neuralnet/weight.png', bbox_inches='tight')
    # plt.show()


def videoSeqWeight(W_hy, L):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Loss', artist='VGROUP', comment='Loss')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    dT, T = W_hy.shape
    tmp_W_hy = np.zeros((dT, T)) * np.nan
    optModelIndex = np.argmin(L, axis=0)
    fig = plt.figure(facecolor='w', edgecolor='k')
    bg = plt.imshow(W_hy, cmap='hot')  # smaller the better
    add_colorbar(bg)
    dots = plt.scatter(range(T), optModelIndex, marker='o', color='b', s=5)
    plt.xlim(0, T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylim(0, dT)
    plt.ylabel('Model Complexity', fontsize=14, color='black')
    plt.title('Learning Weight')
    # plot along the true best model
    with writer.saving(fig, "SeqWeight.mp4", T):
        for i in range(T):
            tmp_W_hy[:, i] = W_hy[:, i]
            bg.set_array(tmp_W_hy)
            x = range(i + 1)
            y = optModelIndex[:i + 1]
            offsets = [[x[j], y[j]] for j in range(i + 1)]
            dots.set_offsets(offsets)
            writer.grab_frame()


def viewSeqLoss(predL_transformed, L_transformed, t_start):
    dT, T = L_transformed.shape
    plt.figure(figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(range(t_start - 1, T), np.min(L_transformed[:, t_start - 1:T], axis=0), 'k-', label='Optimum',
             linewidth=3)  # smaller the better
    plt.plot(range(t_start - 1, T), predL_transformed[t_start - 1:T], 'b-', label='Predictor',
             linewidth=3)  # smaller the better
    plt.xlim(t_start - 1, T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    #    plt.ylim(0,dT)
    plt.ylabel('Loss', fontsize=14, color='black')
    plt.title('Loss of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size': 10})
    # plt.show()


def viewSeqLoss_all(mode, candModels_Sequential, L_transformed, t_start):
    dT, T = L_transformed.shape
    plt.figure(figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(range(t_start - 1, T), np.min(L_transformed[:, t_start - 1:T], axis=0), 'k-', label='Optimum',
             linewidth=3)  # smaller the better
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    linestyles = ['-', '--', '-.', ':']
    for i in range(len(mode)):
        plt.plot(range(t_start - 1, T), candModels_Sequential[i]['seqPredLoss'][t_start - 1:T], color=colors[i],
                 label=mode[i], linewidth=1, linestyle=linestyles[i])  # smaller the better
    plt.xlim(t_start - 1, T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    #    plt.ylim(0,dT)
    plt.ylabel('Loss', fontsize=14, color='black')
    plt.title('Loss of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size': 10})
    # plt.show()


def viewBatchLoss_all(mode, candModels_Sequential, L_transformed, t_start):
    dT, T = L_transformed.shape
    plt.figure(figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(range(t_start - 1, T), np.min(L_transformed[:, t_start - 1:T], axis=0), 'k-', label='Optimum',
             linewidth=3)  # smaller the better
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    linestyles = ['-', '--', '-.', ':']
    for i in range(len(mode)):
        rows = candModels_Sequential[i]['batch_opt_model'][t_start - 1:T].tolist()
        cols = range(t_start - 1, T)
        # print(L_transformed[rows,cols])
        plt.plot(range(t_start - 1, T), L_transformed[rows, cols], color=colors[i], label=mode[i], linewidth=1,
                 linestyle=linestyles[i])  # smaller the better
    plt.xlim(t_start - 1, T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylabel('Batch Loss', fontsize=14, color='black')
    plt.title('Batch Loss of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size': 10})
    # plt.show()

    plt.figure(figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k')
    optModelIndex = np.argmin(L_transformed, axis=0)
    plt.scatter(range(T), optModelIndex, marker='o', color='k', s=30)
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    markers = ['s', '*', '^', 'D']
    for i in range(len(mode)):
        plt.scatter(range(t_start - 1, T), candModels_Sequential[i]['batch_opt_model'][t_start - 1:T],
                    marker=markers[i], color=colors[i], label=mode[i], s=30)
    plt.xlim(0, T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylim(0, dT)
    plt.ylabel('Batch Model', fontsize=14, color='black')
    plt.title('Batch Model of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size': 10})
    # plt.show()


def videoSeqLoss(predL_transformed, L_transformed, t_start):
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Loss', artist='VGROUP', comment='Loss')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    dT, T = L_transformed.shape
    fig = plt.figure(facecolor='w', edgecolor='k')
    # line1=plt.plot(range(t_start-1, T), np.min(L_transformed[:,t_start-1:T], axis=0), 'k-', label='Optimum',
    # linewidth=3)  #smaller the better
    # line2=plt.plot(range(t_start-1, T), predL_transformed[t_start-1:T], 'b-', label='Predictor', linewidth=3)
    # #smaller the better
    line1, = plt.plot(range(t_start - 1, T), np.min(L_transformed[:, t_start - 1:T], axis=0), 'k-', label='Optimum',
                      linewidth=3)  # smaller the better
    line2, = plt.plot(range(t_start - 1, T), predL_transformed[t_start - 1:T], 'b-', label='Predictor',
                      linewidth=3)  # smaller the better
    plt.xlim(t_start - 1, T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    #    plt.ylim(0,dT)
    plt.ylabel('Loss', fontsize=14, color='black')
    plt.title('Loss of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size': 10})
    with writer.saving(fig, "SeqLoss.mp4", T):
        for i in range(t_start - 1, T):
            line1.set_data(range(t_start - 1, i + 1), np.min(L_transformed[:, t_start - 1:i + 1], axis=0))
            line2.set_data(range(t_start - 1, i + 1), predL_transformed[t_start - 1:i + 1])
            writer.grab_frame()


def viewSeqBatchLossRatio(mode, candModels_Sequential, t_start, T):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    plt.figure(figsize=(10, 10), dpi=300, facecolor='w', edgecolor='k')
    linestyles = ['--', '--', '-.', ':']
    for i in range(len(mode)):
        plt.plot(range(t_start - 1, T), candModels_Sequential[i]['batch_loss_ratio'][t_start - 1:T], color=colors[i],
                 label=mode[i], linewidth=1, linestyle=linestyles[i])
    plt.plot(range(t_start - 1, T), candModels_Sequential[0]['loss_ratio'][t_start - 1:T], color='k',
             label='GTIC_Sequential', linewidth=1, linestyle='--')
    plt.xlim(t_start - 1, T - 1)
    plt.xlabel('Data Size', fontsize=10, color='black')
    #    plt.ylim(0,dT)
    plt.ylabel('Loss Ratio', fontsize=10, color='black')
    plt.title('Loss ratio of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size': 10})
    # plt.tight_layout()
    plt.grid()
    plt.savefig('./output/neuralnet/lossratio.png', bbox_inches='tight')
    # plt.show()


def viewLossRatio(loss_ratio, t_start, T):
    plt.figure(figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(range(t_start - 1, T), loss_ratio[t_start - 1:T], 'k-', label='Optimum', linewidth=3)
    # plt.show()


def viewLossRatio_all(mode, candModels_Sequential, t_start, T):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    plt.figure(figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k')
    linestyles = ['-', '--', '-.', ':']
    for i in range(len(mode)):
        plt.plot(range(t_start - 1, T), candModels_Sequential[i]['loss_ratio'][t_start - 1:T], color=colors[i],
                 label=mode[i], linewidth=1, linestyle=linestyles[i])
    plt.xlim(t_start - 1, T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    #    plt.ylim(0,dT)
    plt.ylabel('Loss Ratio', fontsize=14, color='black')
    plt.title('Loss ratio of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size': 10})
    # plt.show()


def viewBatchLossRatio_all(mode, candModels_Sequential, t_start, T):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    plt.figure(figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k')
    linestyles = ['-', '--', '-.', ':']
    for i in range(len(mode)):
        plt.plot(range(t_start - 1, T), candModels_Sequential[i]['batch_loss_ratio'][t_start - 1:T], color=colors[i],
                 label=mode[i], linewidth=1, linestyle=linestyles[i])
    plt.xlim(t_start - 1, T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    #    plt.ylim(0,dT)
    plt.ylabel('Batch Loss Ratio', fontsize=14, color='black')
    plt.title('Batch Loss ratio of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size': 10})
    # plt.show()


def viewTiming(mode, timing_image, t_start, T):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    plt.figure(figsize=(10, 10), dpi=300, facecolor='w', edgecolor='k')
    linestyles = ['--', '--', '-.', ':']
    for i in range(len(mode)):
        plt.plot(range(t_start - 1, T), timing_image[mode[i]], color=colors[i], label=mode[i], linewidth=1,
                 linestyle=linestyles[i])
    plt.xlim(t_start - 1, T - 1)
    plt.xlabel('Data Size', fontsize=10, color='black')
    #    plt.ylim(0,dT)
    plt.ylabel('Elapsed Time', fontsize=10, color='black')
    plt.title('Elapsed Time of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size': 10})
    # plt.tight_layout()
    plt.grid()
    plt.savefig('./output/neuralnet/timing.png', bbox_inches='tight')
    # plt.show()


def main():
    try:
        os.makedirs('output/neuralnet')
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    runLogisticExper()


if __name__ == "__main__":
    main()