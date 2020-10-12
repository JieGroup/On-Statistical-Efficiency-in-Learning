from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from logistic import init, getInsampleLoss, getOutsampleLoss

def genY(mu):
    return np.random.binomial(1, 1 / (1 + np.exp(-mu)))

def generate_synthetic_data(n, p, gamma):

  # General a toy dataset: true logistic-based two classes but with small coefficients
  # np.random.seed(0)
  X = np.random.normal(size = (n, p))
  #beta = np.zeros((p, 1))
  #beta[0:7] = np.array([2,2,1,1,0.2,0.2,0.1,0.1])
  beta = 1 / np.power(np.arange(1,p+1), gamma)
  #beta = np.power(gamma, np.arange(1,p+1))
  mu = X.dot(beta)
  y = genY(mu)
  
  # compute the true loss Integral -log(density)
  loss0 = - (np.sum(mu[y==1]) - np.sum(np.log(1+np.exp(mu)))) / n
  return X, y, loss0

'''
    return mean and se
'''
def runSingleExper(n, p, gamma, nrep = 10, PRINT = True):
    
    loss_cmp = np.zeros((nrep, 4))
    acc_cmp = np.zeros((nrep, 4))
    eff_cmp = np.zeros((nrep, 4))
    
    for r in range(nrep):
        Xt, yt, _ = generate_synthetic_data(n ,p, gamma)
        candModels_init = init(p)
        candModels, loss_AIC, loss_BIC, loss_TIC, AIC_sel, BIC_sel, SH_sel, TIC_sel = getInsampleLoss(candModels_init, Xt, yt)
        
        # obtain the out sample predictive performance
        X_test, y_test, loss0 = generate_synthetic_data(1000, p, gamma)
        loss, acc, eff = getOutsampleLoss(candModels, X_test, y_test, loss0)
        
        loss_cmp[r,:] = np.array([loss[AIC_sel], loss[BIC_sel], loss[SH_sel], loss[TIC_sel]])
        acc_cmp[r,:] = np.array([acc[AIC_sel], acc[BIC_sel], acc[SH_sel], acc[TIC_sel]])
        eff_cmp[r,:] = np.array([eff[AIC_sel], eff[BIC_sel], eff[SH_sel], eff[TIC_sel]])
    
    if PRINT:
        print('pred loss of AIC, BIC, SH, TIC are \n', np.mean(loss_cmp, 0))
        print('accuracy of AIC, BIC, SH, TIC are \n', np.mean(acc_cmp, 0))
        print('effi of AIC, BIC, SH, TIC are \n', np.mean(eff_cmp, 0))
    
    c = np.sqrt(nrep)
    return np.mean(loss_cmp, 0), np.mean(acc_cmp, 0), np.mean(eff_cmp, 0), \
             np.std(loss_cmp, 0)/c, np.std(acc_cmp, 0)/c, np.std(eff_cmp, 0)/c
  
    
'''
    x is the control param
    b & c are the mean and se of some estimates
'''
def viewPlot(x, b, c, xlab, ylab):
    plt.plot(x, b[0,:], marker='.', color='k', label = 'AIC')
    plt.plot(x, b[1,:], marker='*', color='b', label = 'BIC')
    plt.plot(x, b[2,:], marker='o', color='r', label = 'TIC')
    #plt.xlim(, )s
    plt.xlabel(xlab, fontsize=14, color='black')
    plt.ylim(0,1)
    plt.ylabel(ylab, fontsize=14, color='black')
    #plt.title('Predictive Loss (in log)')
    plt.legend()

    ax = plt.gca()
    ax.fill_between(x, b[0,:]-c[0,:], b[0,:]+c[0,:], facecolor='k', interpolate=True, alpha=0.2)
    ax.fill_between(x, b[1,:]-c[1,:], b[1,:]+c[1,:], facecolor='b', interpolate=True, alpha=0.2)
    ax.fill_between(x, b[2,:]-c[2,:], b[2,:]+c[2,:], facecolor='r', interpolate=True, alpha=0.2)
    plt.show()

n = 100
p = 20
gamma = 0.1
runSingleExper(n, p, gamma)


def runVariousGamma(n, p):
    gammas = np.arange(0.1,1,0.1)
    ng = gammas.shape[0]
    # mean
    l, a, e = np.zeros((3, ng)), np.zeros((3, ng)), np.zeros((3, ng))
    # standard error
    ll, aa, ee = np.zeros((3, ng)), np.zeros((3, ng)), np.zeros((3, ng))
    for g in range(ng):
        gamma = gammas[g]
        l[:,g], a[:,g], e[:,g], ll[:,g], aa[:,g], ee[:,g] = runSingleExper(n, p, gamma, nrep = 10, PRINT = False)
    viewPlot(gammas, l, ll, 'gamma', 'loss')    
    viewPlot(gammas, a, aa, 'gamma', 'loss')    
    viewPlot(gammas, e, ee, 'gamma', 'loss')  
    

gamma = 0.999
def runVariousNP(gamma):
    ns = np.array([100, 300, 500])
    ps = np.floor(ns / 2).astype(int) #np.array([1, 1, 1]) #
    ng = ps.shape[0]
    # mean
    l, a, e = np.zeros((3, ng)), np.zeros((3, ng)), np.zeros((3, ng))
    # standard error
    ll, aa, ee = np.zeros((3, ng)), np.zeros((3, ng)), np.zeros((3, ng))
    for g in range(ng):
        print('g: ', g)
        n, p = ns[g], ps[g]
        l[:,g], a[:,g], e[:,g], ll[:,g], aa[:,g], ee[:,g] = runSingleExper(n, p, gamma, nrep = 10, PRINT = False)
    viewPlot(ns, l, ll/3, 'n', 'loss')    
    viewPlot(ns, a, aa/3, 'n', 'accuracy')    
    viewPlot(ns, e, ee/3, 'n', 'prediction efficiency')  
  
    # ReSULTS:
# case:  beta = 1 / np.power(np.arange(1,p+1), gamma)

#gamma = 0.1, n = [100, 300, 500, 1000], p = n/5
# l
#       [0.421521  , 0.34084834, 0.33969286],
#       [0.53112026, 0.64387475, 0.67797856],
#       [0.41874047, 0.34084834, 0.33969286]
# ll
#       [0.01459637, 0.0101405 , 0.00983788],
#       [0.03429068, 0.02385569, 0.00813174],
#       [0.01404431, 0.0101405 , 0.00983788]
# a
#       [0.7906, 0.834 , 0.8346],
#       [0.7436, 0.5988, 0.5844],
#       [0.7922, 0.834 , 0.8346]
# aa
#       [0.01298938, 0.00991968, 0.00884556],
#       [0.02207949, 0.03551248, 0.04733016],
#       [0.01241112, 0.00991968, 0.00884556]
# e
#       [0.95373569, 0.97798503, 0.98271539],
#       [0.55261912, 0.31805982, 0.31708338],
#       [0.97234726, 0.97798503, 0.98271539]
# ee
#       [0.01955293, 0.0143349 , 0.01135032],
#       [0.09203727, 0.06070138, 0.0218789 ],
#       [0.01437481, 0.0143349 , 0.01135032]
    
#gamma = 0.1, n = [100, 300, 500, 1000], p = n/2
# l
#       [0.6493764 , 0.80827891, 0.94920518],
#       [0.69097011, 0.69083992, 0.69464522],
#       [0.50152622, 0.53372287, 0.54205273]
# ll
#       [0.02410551, 0.04535307, 0.04364313],
#       [0.00455331, 0.00150002, 0.00131175],
#       [0.02183644, 0.02426556, 0.02244606]
# a
#       [0.6934, 0.7218, 0.7388],
#       [0.5456, 0.5488, 0.4968],
#       [0.78  , 0.7744, 0.7922]
# aa
#       [0.02691475, 0.01716147, 0.01311854],
#       [0.0723308 , 0.08611118, 0.08452796],
#       [0.0221504 , 0.01283994, 0.01126925]
#
# e
#       [0.64198818, 0.58215869, 0.50391191],
#       [0.57284832, 0.69772776, 0.72560492],
#       [0.96768052, 0.98655231, 0.98637116]
# ee
#       [0.03910324, 0.02663603, 0.01744821],
#       [0.03693174, 0.04091557, 0.03887935],
#       [0.01523912, 0.00754941, 0.00843784]
    
# case:  beta = np.power(gamma, np.arange(1,p+1))
    
#gamma = 0.1, ..., 0.9, n = 100, p = 20 
#    not good
    
#gamma = 0.999, n = [100, 300, 500], p = n/5 
# l
#       [0.38601163, 0.30416951, 0.29512452],
#       [0.52274729, 0.44474248, 0.43711171],
#       [0.38392059, 0.30066554, 0.28947708]   
# ll
#       [0.0116626 , 0.01040622, 0.00805659],
#       [0.03723766, 0.05177412, 0.04538468],
#       [0.01081052, 0.00980218, 0.0079788 ]

# a
#       [0.8328, 0.8744, 0.873 ],
#       [0.7476, 0.8016, 0.7666],
#       [0.8328, 0.8742, 0.8744]]    
# aa
#       [0.02167432, 0.00968628, 0.00741485],
#       [0.05025996, 0.03026787, 0.06199552],
#       [0.02167432, 0.00969722, 0.0077501 ]
#
# e
#       [0.95873169, 0.95642385, 0.965698  ],
#       [0.55855233, 0.64369755, 0.62276902],
#       [0.96958384, 0.97676611, 1.        ]
# ee
#       [0.02491018, 0.02179994, 0.01439715],
#       [0.0909716 , 0.09712408, 0.07735978],
#       [0.02409442, 0.01404765, 0.        ]

#gamma = 0.999, n = [100, 300, 500], p = n/2   
# l
#       [0.67661172, 0.81034165, 0.95882756],
#       [0.71382739, 0.69518798, 0.69292435],
#       [0.46890811, 0.50085189, 0.50602092]
# ll
#       [0.03182696, 0.02977199, 0.02802846],
#       [0.0067455 , 0.00127337, 0.00128078],
#       [0.01675207, 0.01797845, 0.01313404]

# a
#       [0.7292, 0.758 , 0.7294],
#       [0.6286, 0.6736, 0.4018],
#       [0.8026, 0.8062, 0.793 ]
# aa
#       [0.02638515, 0.01595995, 0.00513848],
#       [0.06406375, 0.07606723, 0.05469951],
#       [0.01646098, 0.01516694, 0.00464973]
#
# e
#       [0.58740368, 0.55572294, 0.48120486],
#       [0.53082758, 0.6621041 , 0.68496595],
#       [0.98223121, 0.99653429, 0.99247135]
# ee
#       [0.03737522, 0.02338216, 0.02343224],
#       [0.03346551, 0.03252834, 0.02199305],
#       [0.00928202, 0.00189169, 0.00638416]

    
    