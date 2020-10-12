import pickle
import numpy as np
from learning import getSeqEdvice
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import matplotlib
import matplotlib.animation as animation
import os
import errno


def runLogisticExper():
    # %% pre-def the observation up to time T
    T = 101
    # dT = np.floor(T/2).astype(int) #largest candidate model considered
    # dT = np.sqrt(T).astype(int)
    dT = 50
    K = 3  # num of active models we maintain in sequential learning
    # input_dim = 2
    # if dT < K:
    # print('error in dT specification!')
    # t_start = 101 #starting data size

    nJumpExpe = 1 * np.sqrt(T) / K  # expected #jump
    learningParams = {'eta': 2 * np.sqrt(8 * nJumpExpe / T), 'alpha': 5 * nJumpExpe / T,
                      'G': None}  # will use getSeqEdvice()
    actSet, move, thresh = range(K), False, 1 / K  # input to learning
    subW = np.zeros((K))
    subW[0] = 1
    # N=10000

    # #pre-def the benchmark (testdata) to compute the expected loss
    # X_test, y_test = generate_data(N,100) #testing data for loss computation 
    # X, y = generate_data(T,0) #all the training data 
    # #print(X.shape)

    # #pre-def the fixed candidates 
    # candModels_fix = []
    # for t in range(dT):
    # candModels_fix.append({'mlp':None, 'AIC': None, 'BIC': None, 'DIC_1': None, 'DIC_2': None, 'GTIC': None, \
    # 'Holdout': None, 'CV_10fold': None, 'CV_loo': None})
    # L = -np.inf * np.ones((dT,T)) #store all the loss (each col) at each time 
    # candModels_time = []
    # timing_image = {'AIC': [], 'BIC': [], 'DIC_1': [], 'DIC_2': [], 'GTIC': [], \
    #              'Holdout': [], 'CV_10fold': [], 'CV_loo': []}
    # #sequential procedure -- compute the loss for all candidates
    # timing = {'AIC': 0, 'BIC': 0, 'DIC_1': 0, 'DIC_2': 0, 'GTIC': 0, \
    # 'Holdout': 0, 'CV_10fold': 0, 'CV_loo': 0}
    # for t in range(t_start-1,T): #for each sample size
    # print(t)
    # Xt = X[range(t),:]
    # yt = y[range(t),0]
    # candModels_fix, timing, TICcompiletiming = NeuralNetworkFit(candModels_fix, Xt, yt, timing)
    # #print(TIC_compile_timing)
    # for k in timing_image.keys():
    # if(k=='GTIC'):
    # timing_image[k].append(timing[k]+TICcompiletiming)
    # else:
    # timing_image[k].append(timing[k])
    # #compute the loss matrix
    # L[:,t] = np.squeeze(calculate_Loss(candModels_fix, X_test, y_test))
    # print(L[:,t])
    # for t in range(dT):
    # candModels_time.append(copy.deepcopy(candModels_fix))
    # #print(candModels_time[t-t_start+1][4]['GTIC'])

    # L[L==-np.inf] = np.max(L) #the none are set to be the max loss

    # #sequential procedure -- compute the est loss and use graph-based learning 
    actSet_start, actSet_end = np.zeros((T)), np.zeros((T))
    W_hy = np.zeros((dT, T))
    # # seqPredLoss = np.zeros((T))  
    # # loss_ratio =  np.zeros((1,T)) 

    output_filename = './output/logistic/output.pickle'
    # mode = ['AIC','BIC','DIC_1','DIC_2','Holdout','CV_10fold','CV_loo','GTIC']
    # mode = ['AIC','BIC','DIC_1','DIC_2','GTIC','Holdout','CV_10fold','CV_loo']
    # mode = ['GTIC','Holdout','CV_10fold','CV_loo']
    # mode = ['GTIC','AIC','BIC','DIC_1']
    # mode = ['GTIC','Holdout','AIC','BIC']
    # mode = ['Holdout','AIC','BIC']
    with open(output_filename, 'rb') as f:  # Python 3: open(..., 'rb')
        candModels_Sequential1, candModels_time, L, mode, t_start, T, timing_image = pickle.load(f)
    # with open(output_filename, 'wb') as f:  # Python 3: open(..., 'wb')
    # pickle.dump([candModels_Sequential,candModels_time, L, mode, t_start, T], f)
    # print(L[:,0])
    candModels_Sequential = []
    for i in range(len(mode)):
        candModels_Sequential.append(
            {'learningParams': learningParams, 'actSet': actSet, 'actSet_start': actSet_start, 'actSet_end': actSet_end,
             'move': move, 'nummove': 0, 'thresh': thresh, \
             'subW': subW, 'W_hy': W_hy, 'seqPredLoss': np.zeros((T)), 'loss_ratio': np.zeros((T)),
             'batch_opt_model': np.zeros((T), dtype=np.int), 'batch_opt_loss': np.zeros((T)) + np.inf,
             'batch_loss_ratio': np.zeros((T))})
    for t in range(t_start - 1, T):  # for each sample size
        cur_dT = np.floor(np.sqrt(t)).astype(np.int)
        # cur_dT = dT
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
                # print(candModels_Sequential[m]['learningParams']['alpha'])
                if max(candModels_Sequential[m]['actSet']) >= dT:
                    candModels_Sequential[m]['actSet'] = range(dT - K, dT)
                    candModels_Sequential[m]['move'] = False
            candModels_Sequential[m]['actSet_start'][t], candModels_Sequential[m]['actSet_end'][t] = min(
                candModels_Sequential[m]['actSet']), max(candModels_Sequential[m]['actSet'])
            subE = np.array(
                [(candModels_time[j][t - t_start + 1][mode[m]]) for j in candModels_Sequential[m]['actSet']]).reshape(
                K, )
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
                if ((candModels_time[l][t - t_start + 1][mode[m]] is not None) and (
                        candModels_time[l][t - t_start + 1][mode[m]] < candModels_Sequential[m]['batch_opt_loss'][t])):
                    candModels_Sequential[m]['batch_opt_model'][t] = l
                    candModels_Sequential[m]['batch_opt_loss'][t] = candModels_time[l][t - t_start + 1][mode[m]]
            # print(candModels_Sequential[m]['batch_opt_loss'][t])
            candModels_Sequential[m]['batch_loss_ratio'][t] = L[candModels_Sequential[m]['batch_opt_model'][
                                                                    t], t] / np.min(L[:, t])

    # summarize results
    viewLoss(np.log(L), candModels_Sequential1[mode.index('GTIC')]['actSet_start'],
             candModels_Sequential[mode.index('GTIC')]['actSet_end'])
    # viewSeqWeight(candModels_Sequential[mode.index('GTIC')]['W_hy'], L) #print subW
    # viewSeqLoss_all(mode, candModels_Sequential, L, t_start)
    # viewBatchLoss_all(mode, candModels_Sequential, L, t_start)
    # viewLossRatio_all(mode, candModels_Sequential, t_start, T)
    viewBatchLossRatio_all(mode, candModels_Sequential, t_start, T)
    # viewSeqBatchLoss(mode, candModels_Sequential, L, t_start)
    # viewSeqBatchLossRatio(mode,candModels_Sequential, t_start, T)
    viewTiming(mode, timing_image, t_start, T)

    # with open(output_filename, 'wb') as f:  # Python 3: open(..., 'wb')
    # pickle.dump([candModels_Sequential,candModels_time, L, mode, t_start, T, timing_image], f)


# show efficiency by plottign loss_ratio
# viewLossRatio(loss_ratio, t_start)
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
    plt.figure(figsize=(10, 2), dpi=300, facecolor='w', edgecolor='k')
    bg = plt.imshow(L_transformed, cmap=plt.cm.Spectral)  # smaller the better
    # plt.colorbar()
    add_colorbar(bg)
    # plot along the active sets
    plt.scatter(range(10, T), optModelIndex[10:], marker='o', color='k', s=1)
    #    plt.scatter(range(T), actSet_start, marker='x', color='b', s=30)
    #    plt.scatter(range(T), actSet_end, marker='x', color='b', s=30)
    plt.xlim(0, T - 1)
    plt.xlabel('Data size', fontsize=10, color='black')
    plt.ylim(0, nCandi - 1)
    plt.ylabel('Model complexity', fontsize=10, color='black')
    plt.title('Expected prediction loss', fontsize=10)
    # plt.tight_layout()
    plt.savefig('./output/logistic/loss.png', bbox_inches='tight')
    # plt.show()


def videoLoss(L_transformed, actSet_start, actSet_end):
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Loss', artist='VGROUP', comment='Loss')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    nCandi, T = L_transformed.shape
    tmp_L_transformed = np.zeros((nCandi, T)) * np.nan
    optModelIndex = np.argmin(L_transformed, axis=0)  # the first occurrence are returned.
    # tmp_optModelIndex = np.zeros(len(optModelIndex)))* np.nan
    # fig=plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k')
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
    plt.figure(figsize=(10, 2), dpi=300, facecolor='w', edgecolor='k')
    bg = plt.imshow(W_hy, cmap='hot')  # smaller the better
    # plt.colorbar()
    add_colorbar(bg)
    plt.xlim(1, T - 1)
    plt.xlabel('Data size', fontsize=10, color='black')
    plt.ylim(0, dT - 1)
    plt.ylabel('Model complexity', fontsize=10, color='black')
    plt.title('Learning weight', fontsize=10)

    # plot along the true best model
    optModelIndex = np.argmin(L, axis=0)
    plt.scatter(range(10, T), optModelIndex[10:], marker='o', color='b', s=1)
    # plt.tight_layout()
    plt.savefig('./output/logistic/weight.png', bbox_inches='tight')
    # plt.show()


def videoSeqWeight(W_hy, L):
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Loss', artist='VGROUP', comment='Loss')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    dT, T = W_hy.shape
    tmp_W_hy = np.zeros((dT, T)) * np.nan
    optModelIndex = np.argmin(L, axis=0)
    fig = plt.figure(facecolor='w', edgecolor='k')
    bg = plt.imshow(W_hy, cmap='hot')  # smaller the better
    add_colorbar(bg)
    dots = plt.scatter(range(T), optModelIndex, marker='o', color='b', s=5)
    plt.xlim(1, T)
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
    plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k')
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
    plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k')
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
    plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k')
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

    plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k')
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
    FFMpegWriter = animation.writers['ffmpeg']
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
    plt.figure(figsize=(10, 2), dpi=300, facecolor='w', edgecolor='k')
    labels = ['GTIC', 'Holdout', '10 fold', 'LOO']
    linestyles = ['--', '--', '-.', ':']
    for i in range(len(mode)):
        plt.plot(range(t_start - 1, T), candModels_Sequential[i]['batch_loss_ratio'][t_start - 1:T], color=colors[i],
                 label=mode[i], linewidth=1, linestyle=linestyles[i])
    plt.plot(range(t_start - 1, T), candModels_Sequential[0]['loss_ratio'][t_start - 1:T], color='k',
             label='GTIC(Sequential)', linewidth=1, linestyle='--')
    plt.xlim(t_start - 1, T - 1)
    plt.xlabel('Data Size', fontsize=10, color='black')
    #    plt.ylim(0,dT)
    plt.ylabel('Loss ratio', fontsize=10, color='black')
    plt.title('Loss ratio of the selected model to the optimal model (oracle)')
    plt.legend(loc='upper right', prop={'size': 5})

    plt.tight_layout()
    plt.grid()
    plt.savefig('./output/logistic/lossratio.png')
    # plt.show()


def viewLossRatio(loss_ratio, t_start, T):
    plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k')
    plt.plot(range(t_start - 1, T), loss_ratio[t_start - 1:T], 'k-', label='Optimum', linewidth=3)
    # plt.show()


def viewLossRatio_all(mode, candModels_Sequential, t_start, T):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    plt.figure(figsize=(40, 40), dpi=300, facecolor='w', edgecolor='k')
    labels = ['GTIC', 'Holdout', '10 fold', 'LOO']
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
    plt.figure(figsize=(10, 2), dpi=300, facecolor='w', edgecolor='k')
    labels = ['GTIC', 'Holdout', '10 fold', 'LOO']
    linestyles = ['--', '--', '-.', ':']
    for i in range(len(mode)):
        plt.plot(range(t_start - 1, T), candModels_Sequential[i]['batch_loss_ratio'][t_start - 1:T], color=colors[i],
                 label=mode[i], linewidth=1, linestyle=linestyles[i])
    plt.plot(range(t_start - 1, T), candModels_Sequential[0]['loss_ratio'][t_start - 1:T], color='k',
             label='GTIC_Sequential', linewidth=1, linestyle='--')
    plt.xlim(t_start - 1, T)
    plt.xlabel('Data Size', fontsize=14, color='black')
    #    plt.ylim(0,dT)
    plt.ylabel('Loss Ratio', fontsize=14, color='black')
    plt.title('Loss ratio of the optimal model and our predictor at each time')
    plt.legend(loc='upper right', prop={'size': 7})

    plt.tight_layout()
    plt.grid()
    plt.savefig('./output/logistic/lossratio.png')
    # plt.show()


def viewTiming(mode, timing_image, t_start, T):
    colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(mode)))
    plt.figure(figsize=(10, 2), dpi=300, facecolor='w', edgecolor='k')
    linestyles = ['--', '--', '-.', ':']
    labels = ['GTIC', 'Holdout', '10 fold', 'LOO']
    for i in range(len(mode)):
        plt.plot(range(t_start - 1, T), timing_image[mode[i]], color=colors[i], label=mode[i], linewidth=1,
                 linestyle=linestyles[i])
    plt.xlim(t_start - 1, T - 1)
    plt.xlabel('Data Size', fontsize=10, color='black')
    #    plt.ylim(0,dT)
    plt.ylabel('Elapsed time (in seconds)', fontsize=10, color='black')
    plt.title('Computational complexity')
    plt.legend(loc='upper right', prop={'size': 7})
    plt.tight_layout()
    plt.grid()
    plt.savefig('./output/logistic/timing.png', bbox_inches='tight')
    # plt.show()


def main():
    try:
        os.makedirs('output/logistic')
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    runLogisticExper()


if __name__ == "__main__":
    main()