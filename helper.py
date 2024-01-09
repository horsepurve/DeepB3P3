import os
import re

import copy
import json
import numpy as np
import time
import pickle
import random
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax
#%%
from sklearn.metrics import accuracy_score, \
                            f1_score, \
                            matthews_corrcoef, \
                            roc_auc_score, \
                            roc_curve, \
                            precision_recall_curve, \
                            average_precision_score

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

#%% ========== ========== ========== ========== ========== ==========
#
# multi-ensemble
#
# .......... .......... ..........
def half_(v, num):
    return 0.5*(v[:num] + v[num:])
    
def read_pred_all(pred_dir, logs, label_true, prefix='/test_', aug=False,
                  mode='test'):  
    print('> load experiment from:', pred_dir)
    num = len(label_true)
    def half(v):
        return 0.5*(v[:num] + v[num:]) if aug else v
    for idx, epo in enumerate(logs):
        print("no.{} epoch {}".format(idx, epo), end=' ~ ')
        fname = pred_dir+prefix+str(epo)+ ('.txt' if mode=='test' else '_.txt')
        df = pd.read_csv(fname, sep='\t')
        _label = df['observed'].values[:num]
        assert sum(label_true - _label) == 0           
        if idx == 0:
            all_pred = half(df.values[:,1:])
        else:
            all_pred = np.dstack((all_pred, half(df.values[:,1:])))
    return all_pred

def checkacc(logit, label_true, mode='cls', threshold=0.5):
    if mode == 'cls':
        pred = np.argmax(logit, axis=-1)        
    elif mode == 'reg':
        pred = (logit > threshold).astype(int)
    acc = np.mean(pred == label_true)
    return acc

def checkall(logit, label_true, mode='cls', threshold=0.5):
    if mode == 'cls':
        pred = np.argmax(logit, axis=-1)        
    elif mode == 'reg':
        predd = (logit > threshold).astype(int)
    
    TP, FP, TN, FN = perf_measure(label_true, predd)
    SEN = TP/(TP+FN)
    SPE = TN/(TN+FP)
    MAT = matthews_corrcoef(label_true, predd)
    ACC = accuracy_score(label_true, predd)
    AUC = roc_auc_score(label_true, logit) # not predd
    F1  = f1_score(label_true, predd)
    return [ACC,SEN,SPE,AUC,MAT,F1]

def checkaccreg(logit, label_true, threshold=0.5):    
    pred = (logit > threshold).astype(int)
    acc = np.mean(pred == label_true)
    return acc
    
def m(acc_):
    return "{:.4f} @ {}".format(np.max(acc_), np.argmax(acc_))

def ens(all_pred, 
        label_true, 
        skip=0, 
        mode='cls', 
        threshold=0.5,
        prt_all=False, # True
        prt_auc=False,
        pr=False,
        plot_figure=True
        ):
    all_ = all_pred[:,:,skip:]
    acc_, acc_inc, acc_exc = [], [], []
    acc_all = []
    for cut in range(all_.shape[-1]):
        
        if mode == 'cls':
            inc_ = all_[:,:,:cut+1]
            exc_ = all_[:,:,cut+1:]
            sin_ = all_[:,:,cut]
            
            inc_ = np.mean(inc_, axis=-1)
            exc_ = np.mean(exc_, axis=-1)
            #inc_ = softmax(inc_, axis=1)
            #exc_ = softmax(exc_, axis=1)
        elif mode == 'reg':
            inc_ = all_[:,0,:cut+1]
            exc_ = all_[:,0,cut+1:]
            sin_ = all_[:,0,cut]
            inc_ = np.mean(inc_, axis=-1)
            exc_ = np.mean(exc_, axis=-1)
        
        acc_.append(checkacc(sin_, label_true, 
                             mode=mode, threshold=threshold))
        if prt_all:
            acc_all.append(checkall(sin_, label_true, 
                                    mode=mode, threshold=threshold))
        acc_inc.append(checkacc(inc_, label_true, 
                                mode=mode, threshold=threshold))        
        acc_exc.append(checkacc(exc_, label_true, 
                                mode=mode, threshold=threshold))
    if prt_all:
        for acc_epo in acc_all:
            print(acc_epo)
    
    if plot_figure:
        plt.plot(acc_, label='single '+m(acc_))
        plt.plot(acc_inc, label='include '+m(acc_inc))
        plt.plot(acc_exc, label='exclude '+m(acc_exc))
        plt.legend()
    print('single '+m(acc_))
    
    ########## 
    # check metrics
    cutat = np.argmax(acc_)
    sin_ = all_[:,0, cutat]
    predd = np.array(sin_>threshold, dtype=int)
    TP, FP, TN, FN = perf_measure(label_true, predd)
    SEN = TP/(TP+FN)
    SPE = TN/(TN+FP)
    MAT = matthews_corrcoef(label_true, predd)
    ACC = accuracy_score(label_true, predd)
    AUC = roc_auc_score(label_true, sin_) # not predd!
    print('@{}: ACC {:.4f}, SEN {:.4f}, SPE {:.4f}, AUC {:.4f}, MCC {:.4f}'.format(cutat, ACC, SEN, SPE, AUC, MAT))
    print(ACC, SEN, SPE, AUC, MAT, sep='\n')
    if prt_auc:
        if pr:
            fpr, tpr, _ = precision_recall_curve(label_true, sin_)
            area = average_precision_score(label_true, sin_)
            print('P-R area:', area)
        else:
            fpr, tpr, _ = roc_curve(label_true, sin_)
        return np.max(acc_), predd, fpr, tpr
    ##########
    return np.max(acc_), predd

def get_cmap(n, name='cool'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def where2cut(all_pred, label_true, rang=[0.8,1], step=0.0025,
              collect=False # return all accs
              ):
    num = int(( rang[1] - rang[0] ) / step)
    print('> generating grid:', num)
    steps = [rang[0] + i*step for i in range(num)]
    
    epo = all_pred.shape[-1]
    cuts = []
    bests = []
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = get_cmap(epo)
    accs_all = []
    for ep in range(epo):
        sin_ = all_pred[:,0,ep]
        accs = [checkaccreg(sin_, label_true, threshold=s) for s in steps]
        ax.plot(steps, accs, color=cmap(ep),
                alpha=0.8)
        if collect:
            accs_all.append(accs)
        best = np.max(accs)
        cut = steps[np.argmax(accs)]
        bests.append(best)
        cuts.append(cut)
        ax.scatter([cut],[best],color='r',s=15)
    ax.grid()
    # ax.set_ylim(0.9, 0.976)
    if collect:
        return bests, cuts, np.array(accs_all)
    return bests, cuts

def here2cut(all_pred, label_true, rangs):
    accs = []
    epo  = all_pred.shape[-1]
    for ep in range(epo):
        sin_ = all_pred[:,0,ep]
        accs.append( checkaccreg(sin_, label_true, threshold=rangs[ep]) )
    return accs

def ratio2cut(all_pred, label_true, r_num):
    print('using pos number:', r_num)
    accs = []
    epo  = all_pred.shape[-1]
    for ep in range(epo):
        sin_ = all_pred[:,0,ep]
        pred = np.zeros_like(label_true)
        pos  = np.argsort(sin_)[-r_num:]
        pred[pos] = 1
        accs.append( np.mean(pred == label_true) )
    return accs

def where2cutsin(soft, label_true, rang=[0.8,1], step=0.0025, plot=True):
    num = int(( rang[1] - rang[0] ) / step)
    if plot:
        print('> generating grid:', num)
    steps = [rang[0] + i*step for i in range(num)]
    accs = [checkaccreg(soft, label_true, threshold=s) for s in steps]
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(steps, accs, color='k')
        ax.set_title('max acc: {:.4f} @ {:.4f}'.format(np.max(accs),
                                                       steps[np.argmax(accs)]))
        ax.set_xlabel('threshold')
        ax.set_ylabel('acc')
    return steps, accs
    
def boxes(all_pred, label_true):
    print('> knowledge from data:', all_pred.shape)
    epo = all_pred.shape[-1]
    nu  = all_pred.shape[0]
    all_ = copy.deepcopy(all_pred[:,0,0])
    epo_ = np.array([0]*nu)
    lab_ = copy.deepcopy(label_true)
    accs = []
    for i in range(1,epo):
        sin_ = all_pred[:,0,i]
        accs.append( checkaccreg(sin_, label_true, threshold=0.5) )
        all_ = np.hstack((all_, all_pred[:,0,i]))
        epo_ = np.hstack((epo_, np.array([i]*nu)))
        lab_ = np.hstack((lab_, copy.deepcopy(label_true)))
    print('> scores:', all_.shape, '> epochs:', epo_.shape, '> labels:', lab_.shape)
    df = pd.DataFrame({'score':all_,
                       'epoch':epo_,
                       'label':lab_})
    fig, ax = plt.subplots(figsize=(18, 6))
    ax = sns.boxplot(data=df, x="epoch", y="score", hue="label")
    
#%%
from sklearn.manifold import TSNE
from numpy import linalg as LA

#%%
def read_log(filename):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    return lines

def read_pkl(filename):
    with open(filename, 'rb') as f:
        fea = pickle.load(f)
    return fea

def normalize_l2(x, axis=1):
    '''
    x.shape = (num_samples, feat_dim)
    see also:
        torch.linalg.norm(torch.from_numpy(embst[0]),dim=-1,keepdims=True)
    '''
    x_norm = np.linalg.norm(x, axis=axis, keepdims=True)
    x = x / (x_norm + 0.0) # 1e-8 | 0.0
    return x

def feat_trans(a, beta = 1., l2 = False, transform = 'beta',
               k = 1., bias = 0.):
    if l2: # do l2-norm firstly
        a = normalize_l2(a, axis=-1)
    a = k * a + bias
    if beta != 1.:
        if transform == 'beta':
            a = np.power(a[:, ] ,beta)
        elif transform == 'log':
            a = np.log(a) # np.log(a+0.04)
    return a
    
fun = lambda a,b : LA.norm(a[:,None,:]-b[None,:,:],axis=2)

def NC_o(support_data, support_label, 
         query_data, query_label, 
         beta        = 1., 
         l2          = False,
         use_mean    = True,
         transform   = 'beta', # beta | log
         k           = 1., # as in kx + b
         bias        = 0.
         ):
    '''
    NC_o: this version we *only* use mean
    given_cen: True -> *also* report upon this
    given_only: True -> *only* report upon this
    
    e.g.:
    beta        = 0.6 # 0.6
    l2          = True
    transform   = 'beta'
    use_mean    = True
    k           = 1.
    bias        = 2.
    
    support_data  = fea_tr
    support_label = lab_tr
    query_data    = fea_te
    query_label   = lab_te
    '''
    dime = support_data.shape[-1]   
    
    a = copy.deepcopy(support_data)
    a = feat_trans(a, beta=beta, l2=l2, transform=transform, k=k, bias=bias)
    
    if use_mean: # compute centroid 
        mea = np.zeros((0, dime))
        for i in range(2): # 4 for label augmentation
            dp = a[support_label == i]
            dp = np.mean(dp, axis=0)
            mea = np.vstack((mea, dp))
    
    b = copy.deepcopy(query_data)    
    b = feat_trans(b, beta=beta, l2=l2, transform=transform, k=k, bias=bias) 
    
    dist = fun(b, mea) # do distance
    print('distance:', dist.shape)
    min_id = np.argmin(dist, axis=-1)
    nn_acc = np.mean(min_id == query_label)
    print('~ acc: {:.4f}'.format(nn_acc))
    
    num   = dist.shape[0] // 2
    dist_ = np.zeros((num, 2))
    
    # for label augmentation:
    # dist_+= dist[:num,:2]+dist[:num,2:]+dist[num:,:2]+dist[num:,2:]
    dist_+= dist[:num,:]+dist[:num,:]
    
    print('distance:', dist_.shape)
    min_id = np.argmin(dist_, axis=-1)
    nn_acc = np.mean(min_id == query_label[:num])
    print('~ acc: {:.4f}'.format(nn_acc))
#%% ========== ========== ========== ========== ========== ==========
#
# try X augmentation
#
rep = lambda x,it: x[:it]+'X'+x[it+1:]

def df_aug(df_):
    df = copy.deepcopy(df_)
    num = df.shape[0]
    print('~~~ augmenting {} items ~~~'.format(num))
    for it in range(num):
        seq  = df.iloc[it,0]
        leng = len(seq)
        ran_id = random.randrange(leng)
        df.iloc[it,0] = rep(seq, ran_id)
    return df

def df_write(df,f):
    num = df.shape[0]
    for it in range(num):
        seq = df.iloc[it,0]
        rt  = df.iloc[it,1]
        f.write('{}\t{}\n'.format(seq,rt))

def x_aug(input_data, output_data, times=5):
    print('read input data: {}'.format(input_data))
    df = pd.read_csv(input_data, sep='\t')
    if os.path.isfile(output_data):
        print('ERROR: output file exists!')
        return 0
    with open(output_data, 'w') as f:
        f.write('sequence\tRT\n')
        df_write(df, f)
        for i in range(times-1):
            df_ = df_aug(df)
            df_write(df_, f)

def merge_aug(all_pred, nu, times, weight=1.):
    all_mer = copy.deepcopy(all_pred[:nu])
    wei_sum = 1
    for i in range(times-1):        
        all_mer += all_pred[nu*(i+1):nu*(i+2)]*weight
        wei_sum += weight
    return all_mer / wei_sum

def xx_aug(input_data, output_data, AA=2, times=4):
    '''
    change [2,3,...] AAs
    AA = 2: times = [4,6,8,10,...]
    AA = 3: times = [6,9,12,...]
    '''
    assert times % AA == 0
    assert times >= 2*AA
    times_pre = int(times / AA)
    in_data = input_data[:-4]+'_a1x{}.txt'.format(times_pre)

    print('read input data: {}'.format(in_data))
    df = pd.read_csv(in_data, sep='\t')
    if os.path.isfile(output_data):
        print('ERROR: output file exists!')
        return 0
    
    with open(output_data, 'w') as f:
        f.write('sequence\tRT\n')
        df_write(df, f) # write the original
        for i in range(AA-1):
            df_ = df_aug(df)
            df_write(df_, f)
            df = df_ # move to next level
#%%
import faiss 
def hit(d_, hit_):
    '''
    d_   = D[0]
    hit_ = np.array([True, True, True, False])
    '''
    hit_d = d_[hit_]    
    if len(hit_d) == 0:
        return None
    else:
        return hit_d.sum()
        
def influence(D, Ilabel, mode=0):
    num = D.shape[0]
    if mode == 0: # using raw distance sum
        result = np.zeros((num, 2))
        for i in range(num): # fill the scores
            result[i, 0] = hit(D[i], Ilabel[i] == 0)
            result[i, 1] = hit(D[i], Ilabel[i] == 1)
        result[np.isnan(result)] = -np.inf
    if mode == 1: # using 1/d^a as score [note: d can be very small]
        result = np.zeros((num, 2))
        D[D==0] = 1e-10
        D = 1/D
        for i in range(num): # fill the scores
            result[i, 0] = hit(D[i], Ilabel[i] == 0)
            result[i, 1] = hit(D[i], Ilabel[i] == 1)
        result[np.isnan(result)] = 0
    return result

def neighbor(fea_tr, fea_te, lab_tr, 
             k=4, mode=0
             ):
    '''
    k     = 4
    mode  = 0
    '''
    print('> train dim:', fea_tr.shape)
    print('> test dim:', fea_te.shape)
    assert fea_tr.shape[0] == lab_tr.shape[0]
    d  = 16
    xb = fea_tr
    xq = fea_te
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    D, I = index.search(xq, k)
    Ilabel = lab_tr[I]
    
    scores = influence(D, Ilabel, mode=mode)
    predic = np.argmax(scores, axis=-1)
    return predic
#%%
def w_l(values, v_mean):
    '''
    '''
    dis = np.sum((values - v_mean) ** 2, axis=-1)
    S   = np.sum(dis)
    w   = np.log(S / dis)
    return w / np.sum(w)

def truth(values, n_iter=5):
    v_mean = np.mean(values, axis=0) # 1st estimation
    w      = w_l(values, v_mean)
    v_     = np.dot(values.T, w) # 2nd estimation
    for i in range(n_iter):
        w  = w_l(values, v_)
        v_ = np.dot(values.T, w)
    return v_
#%%

def see_merge(all_pred, all_pred_, 
              label_true, label_true_, 
              nu, nu_, times, W):
    # W = 1
    print(all_pred.shape)
    all_mer = merge_aug(all_pred, nu, times=times, weight=W)
    print(all_mer.shape)
    
    #% % see trend
    bests, cuts = where2cut(all_mer[:nu,:,:], label_true[:nu], 
                            rang=[0.,1], step=0.001) # [0.2,0.7] | [0.9,1]
    print(np.max(bests))
    c_ts  = copy.deepcopy(cuts)
    b_sts = copy.deepcopy(bests)
    
    #% % merge it _
    print(all_pred_.shape)
    all_mer_ = merge_aug(all_pred_, nu_, times=times, weight=W)
    print(all_mer_.shape)
    
    #% % see trend _
    bests, cuts = where2cut(all_mer_[:nu_,:,:], label_true_[:nu_], 
                            rang=[0.,1], step=0.001) # [0.2,0.7] | [0.9,1]
    print(np.max(bests))
    
    #% %
    plt.scatter(c_ts, cuts, color='r')
    print(np.mean(c_ts))
    print(np.mean(cuts))
    
    #% %
    plt.scatter(b_sts, bests, color='k')            
    print(np.max(bests), np.argmax(bests))
    
    #% % regression mode
    acc_cut5 = ens(all_mer, label_true[:nu], skip=0, mode='reg', threshold=0.5)
    
    #% % regression mode
    acc_cutm = ens(all_mer, label_true[:nu], skip=0, mode='reg', threshold=np.mean(cuts))
    return acc_cut5, acc_cutm

#%%
#%%
#%%
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%% ========== ========== ========== ========== ========== ==========
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%