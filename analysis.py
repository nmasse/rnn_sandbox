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
            svm = SVC(C=1.0, kernel='linear', max_iter=2000, decision_function_shape='ovr', shrinking=False, tol=1e-3).fit(X_tr, y_tr)
            svm_acc[t, i] = svm.score(X_te, y_te)

    return np.mean(svm_acc, axis=-1)

def average_frs_by_condition(h, sample, test):
    # Identify sample/test conditions
    sample_rng = np.unique(sample)
    test_rng   = np.unique(test)
    conditions = np.hstack((sample.squeeze()[:,np.newaxis], 
        test.squeeze()[:,np.newaxis]))
    unique_conds = np.unique(conditions, axis=1)

    # Average activity of all neurons over trials with 
    # each unique combination of sample/test
    avg_fr = np.zeros((h.shape[2], len(unique_conds), h.shape[1])) # neur x cond x timestep
    for j, cond in enumerate(unique_conds):
        rel_trials    = np.where((conditions == cond).all(1))[0]
        avg_fr[:,j,:] = np.mean(h[rel_trials,:,:], axis=0).T

    return avg_fr, unique_conds
