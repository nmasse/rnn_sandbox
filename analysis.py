def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def decode_signal(X, y, timesteps, k_folds=4):
    ''' Set up decoder, x-validate via k folds'''
    svm_acc = np.zeros((len(timesteps), k_folds))
    skf = StratifiedKFold(n_splits=k_folds)
    skf.get_n_splits(X[:, 0, :], y)

    for i, (tr_idx, te_idx) in enumerate(skf.split(X[:, 0, :], y)):
        y_tr, y_te = y[tr_idx], y[te_idx]
        for t, timestep in enumerate(timesteps):
            X_tr, X_te = X[tr_idx, timestep, :], X[te_idx, timestep,  :]
            ss = StandardScaler()
            ss.fit(X_tr)
            X_tr = ss.transform(X_tr)
            X_te = ss.transform(X_te)
            svm = SVC(C=1.0, kernel='linear', max_iter=1000, decision_function_shape='ovr', shrinking=False, tol=1e-3).fit(X_tr, y_tr)
            svm_acc[t, i] = svm.score(X_te, y_te)

    return np.mean(svm_acc, axis=-1)

def accuracy_SL(policy, labels, mask):

    labels_amax = np.argmax(labels,axis=-1)
    policy_amax = np.argmax(policy,axis=-1)
    non_fix_period = np.float32(labels_amax > 0)
    task_mask = mask * non_fix_period

    return np.sum(task_mask * (labels_amax == policy_amax)) / np.sum(task_mask)
