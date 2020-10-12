from __future__ import print_function


import os
import sys
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

    

def init(p):
    candModels_init = []
    for k in range(p):
        candModels_init.append({'var': np.arange(0,k+1), 'beta': None, 'bias': None})
    return candModels_init

def getInsampleLoss(candModels_init, Xt, yt):  
    candModels = LogisticFit(candModels_init, Xt, yt)
    K = len(candModels)
    loss_AIC = np.zeros(K)
    loss_BIC = np.zeros(K)
    loss_TIC = np.zeros(K)
    loss_SH0 = np.zeros(K)
    ms = np.zeros(K)
    for k in range(K):
        var, beta, bias = candModels[k]['var'], candModels[k]['beta'], candModels[k]['bias']
        X, y = Xt[:,var], yt
        mu = X.dot( beta )+ bias
        X = np.concatenate((X,np.ones((X.shape[0],1))),axis=1)
        loglik = np.sum(mu[y==1]) - np.sum( np.log(1+np.exp(mu)) )
        m = beta.shape[0]+1
        n = y.shape[0]
        J = np.zeros((m,m))
        for i in range(n):
            J += ( y[i] - np.exp(mu[i])/(1+np.exp(mu[i])) )**2 * X[i,:].reshape((m,1)).dot( X[i,:].reshape((1,m)) ) 
        J /= n
        V = np.zeros((m,m))
        for i in range(n): 
            V += np.exp(mu[i]) / ((1+np.exp(mu[i]))**2) * X[i,:].reshape((m,1)).dot( X[i,:].reshape((1,m)) ) 
        V /= n
        tic = np.trace( np.linalg.inv(V).dot(J) ) / n 
        #print(-loglik/n)
        #print(pen)
        loss_AIC[k] = -loglik/n + 1.0 * m / n
        loss_BIC[k] = -loglik/n + 0.5 * m * np.log(n) / n 
        loss_TIC[k] = -loglik/n + tic
        loss_SH0[k] = -loglik/n
        ms[k] = m
    
    # selection
    AIC_sel = np.argmin(loss_AIC)
    BIC_sel = np.argmin(loss_BIC)
    TIC_sel = np.argmin(loss_TIC)
    cs = np.arange(10)/10.0
    ds = np.arange(10)
    for j in range(10):
        ds[j] = np.argmin(loss_SH0 + cs[j] * ms / n)
    idx = np.argmin(ds[:-1] - ds[1:])+1
    SH_sel = np.argmin(loss_SH0 + 2 * cs[idx] * ms / n)

    return candModels, loss_AIC, loss_BIC, loss_TIC, AIC_sel, BIC_sel, SH_sel, TIC_sel

def getOutsampleLoss(candModels, X_test, y_test, loss0):
    K = len(candModels)
    loss = np.zeros(K)
    acc = np.zeros(K)
    eff = np.zeros(K)
    
    for k in range(K):
        var, beta, bias = candModels[k]['var'], candModels[k]['beta'], candModels[k]['bias']
        if beta is not None:
            #print(beta[0],beta[1])
            mu = X_test[:,var].dot( beta ) + bias
            loss[k] = -(np.sum(mu[y_test==1]) - np.sum( np.log(1+np.exp(mu)) ))/y_test.shape[0]
            acc[k] = (np.sum((mu > 0).ravel() & (y_test == 1).ravel()) + 
                np.sum((mu > 0).ravel() & (y_test == 1).ravel())) / y_test.shape[0]
        else:
            loss[k] = -np.inf
            acc[k] = 0
            eff[k] = 0
    for k in range(K):
        eff[k] = np.min(loss - loss0) / (loss[k] - loss0)
    return loss, acc, eff #ave negative loglik
 
def LogisticFit(candModels, Xt, yt):
    N = len(candModels)
    logistic = linear_model.LogisticRegression()
    for n in range(N):
        #print(n)
        var = candModels[n]['var']
        if len(var) > len(yt):
            candModels[n]['beta'] = None
            candModels[n]['bias'] = None
        else:
            logistic.fit(Xt[:,var], yt)
            beta = logistic.coef_.reshape([len(var),1])
            bias = logistic.intercept_
            candModels[n]['beta'] = beta
            candModels[n]['bias'] = bias

    return candModels

def viewLoss(L_transformed, actSet_start, actSet_end):
    nCandi, T = L_transformed.shape
    optModelIndex = np.argmin(L_transformed, axis=0) #the first occurrence are returned.
    plt.figure(num=1, figsize=(40, 40), dpi=150, facecolor='w', edgecolor='k') 
    plt.imshow(L_transformed, cmap=plt.cm.Spectral)  #smaller the better
    plt.colorbar()
    
    #plot along the active sets 
    plt.scatter(range(T), optModelIndex, marker='o', color='k', s=30)
    plt.xlim(0,T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    plt.ylim(0,nCandi)
    plt.ylabel('Model Complexity', fontsize=14, color='black')
    plt.title('Predictive Loss (in log)')
    plt.show()
  
#if __name__ == '__main__':
    




