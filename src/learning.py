__author__ = 'Jie'
import numpy as np


# E: col vec of expert prediction
# W: weight of them
# t: time
# learningParams: set of parameters for expert learning
# G: unweighted directed graph
# for test:
# E=np.ones((3))
# W=np.ones((3))
# learningParams={'eta': 1, 'alpha': 0.1, 'G':np.zeros((3,3), dtype=bool)}
def getEdvice(E, W, learningParams, t):
    eta, alpha, G = learningParams['eta'], learningParams['alpha'], learningParams['G']

    # updata
    logV = np.log(W) - eta * E  # in log scale
    V = np.exp(logV - np.max(logV))
    N = len(E)

    # redistribute
    share = np.zeros((N))
    outShareUnit = alpha * V / np.sum(G, 1)  # normalized G, the sum of out-flow is one at each node
    for n in range(0, N):
        share[n] = np.sum(outShareUnit[G[:, n]])
    W = V * (1 - alpha) + share
    W = W / np.sum(W)
    masterE = np.sum(E * W)
    # updated_W_masterE = {'W': W, 'masterE': masterE}
    return W, masterE  # updated_W_masterE


# specialized getEdvice(): where G is a path 0->1->... and subE corrs. to an active subset
# subW[0] is the smallest model (0->...)
# subE and subW are in undefinite size
def getSeqEdvice(subE, subW, learningParams, move, thresh, t):
    # boolean move indicates whether we first re-map the weight to a shifted set of candidates
    K = len(subW)
    if move:
        temp = subW[0]
        subW[0:K - 1] = subW[1:K]
        subW[K - 1] = temp
    eta, alpha = learningParams['eta'], learningParams['alpha']

    # updata
    logV = np.log(subW) - eta * subE  # in log scale
    V = np.exp(logV - np.max(logV))

    # redistribute
    for n in range(0, K):
        if n is 0:
            subW[n] = V[n] * (1 - (alpha))
        elif n > 0 and n < K - 1:
            subW[n] = V[n] * (1 - alpha) + (alpha) * V[n - 1]
        else:
            subW[n] = V[n] + (alpha) * V[n - 1]
    subW = subW / np.sum(subW)
    masterE = np.sum(subE * subW)
    if subW[0] < thresh and subW[K - 1] > 1 - thresh:
        move = True
    else:
        move = False
    return subW, masterE, move