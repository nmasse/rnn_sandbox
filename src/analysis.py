def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def decode_signal(X, y, timesteps, k_folds=4):

    ''' Set up decoder, x-validate via k folds'''
    svm = SVC(C=1.0, kernel='linear', max_iter=200, decision_function_shape='ovr', 
        shrinking=False, tol=1e-3)
    if y.ndim < 2:
        y = y[:, np.newaxis]
    svm_acc = np.zeros((len(timesteps), y.shape[1], k_folds))
    skf = StratifiedKFold(n_splits=k_folds)

    for i, (tr_idx, te_idx) in enumerate(skf.split(X[:, 0, :], y[:,0])):
        y_tr, y_te = y[tr_idx], y[te_idx]
        for t, timestep in enumerate(timesteps):
            for j in range(y.shape[1]):
                X_tr, X_te = X[tr_idx, timestep, :], X[te_idx, timestep,  :]
                ss = StandardScaler()
                ss.fit(X_tr)
                X_tr = ss.transform(X_tr)
                X_te = ss.transform(X_te)
                svm.fit(X_tr, y_tr[:,j].squeeze())
                svm_acc[t, j, i] = svm.score(X_te, y_te[:,j].squeeze())

    return np.mean(svm_acc, axis=-1)

def accuracy_SL_all_tasks(policy, labels, mask, rule, 
    possible_rules, continue_resets=[]):

    accuracies = []
    for i in possible_rules:
        idx = np.where(rule == i)[0]
        if len(idx) == 0:
            accuracies.append(0.)
            continue
        #acc = accuracy_RL_like(policy[idx, ...], labels[idx, ...], mask[idx, ...], continue_resets)
        acc = accuracy_SL(policy[idx, ...], labels[idx, ...], mask[idx, ...]) 
        accuracies.append(acc)

    return accuracies

def accuracy_RL(rewards):

    correct = np.sum(rewards, axis=1)
    mean_correct = np.mean(np.float32(correct > 0))

    return mean_correct

def accuracy_SL(policy, labels, mask):

    labels_amax = np.argmax(labels,axis=-1)
    policy_amax = np.argmax(policy,axis=-1)
    non_fix_period = np.float32(labels_amax > 0)
    task_mask = mask * non_fix_period

    return np.sum(task_mask * (labels_amax == policy_amax)) / np.sum(task_mask)

def accuracy_RL_like(policy, labels, mask, continue_resets=[]):

    labels_amax = np.argmax(labels,axis=-1)
    policy_amax = np.argmax(policy,axis=-1)
    resp_period = np.float32(labels_amax > 0)
    batch_size, trial_length, _ = policy.shape
    correct = np.zeros((batch_size), dtype=np.float32)
    trial_continues = np.ones((batch_size), dtype=np.float32)
    correct_time = np.zeros_like(mask)
    for t in range(trial_length):
        # Check to see if we've started a new test period; if so, reset trial_continues
        if t in continue_resets:
            trial_continues = np.ones((batch_size), dtype=np.float32)
        correct_resp = trial_continues * np.float32(labels_amax[:,t] == policy_amax[:,t]) * resp_period[:, t]
        correct += mask[:,t] * correct_resp
        s =  mask[:,t] * correct_resp
        correct_time[:, t] = mask[:,t] * correct_resp
        trial_continues *= (1 - mask[:,t] * np.float32(policy_amax[:,t] > 0))

    return np.mean(correct) / max(1, len(continue_resets) + 1) # Make sure to normalize

def dimensionality(h, bounds):
    '''
    Compute dimensionality of data (participation ratio).
    '''
    dim_part_rat = []
    for i, b in enumerate(bounds):
        pca = PCA().fit(np.reshape(h[:,b,:], (-1, h.shape[-1])))
        e_vals = pca.explained_variance_ratio_
        dim_part_rat.append(np.sum(e_vals)**2 / np.sum(e_vals**2))

    return dim_part_rat