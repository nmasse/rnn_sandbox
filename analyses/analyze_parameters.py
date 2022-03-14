import pickle, glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import warnings, argparse
import umap, os

parser = argparse.ArgumentParser('')
parser.add_argument('--data_path', type=str, default='/Users/mattrosen/param_data/')
parser.add_argument('--figure_save_path', type=str, default='../reliability_figures/')

warnings.filterwarnings('ignore')
args = parser.parse_args()

def load_data(data_path=args.data_path):
    keys_to_keep = ['tc_modulator',
                    'rnn_weight',
                    'input_weight',
                    'EE_kappa',
                    'EI_kappa',
                    'IE_kappa',
                    'II_kappa',
                    'alpha_EI',
                    'alpha_EE',
                    'alpha_II',
                    'inp_E_kappa',
                    'inp_I_kappa',
                    'inp_E_topo',
                    'inp_I_topo',
                    'EE_topo',
                    'EI_topo',
                    'IE_topo',
                    'II_topo',]

    X, y, fns, frs = [], [], [], []
    for fn in glob.glob(data_path + "*.npz"):
        d = np.load(fn)
        names = d.f.keys
        inds = [i for i in range(len(names)) if names[i] in keys_to_keep]
        X.append([float(j) for j in d.f.params[inds]])
        y.append(np.mean(d.f.acc))
        fns.append(d.f.params[-1])
        frs.append(d.f.fr)
        names = names[inds]

    X = np.array(X)
    y = np.array(y)
    frs = np.array(frs)

    #h = np.array(h)
    #X = X[np.where(y > 0.5)]
    #y = y[np.where(y > 0.85)]
    #frs = frs[np.where(y > 0.85)]

    return X, y, names, frs

def neighborhood_analysis(X, y, k=10):

    # For networks that are highly accurate, identify the nearest neighbors in parameter space 
    # 2 methods: 
    # 1) unnormalized (distance just in raw parameter space)
    # 2) normalized by feature dimension (to make distance in each dimension on the same scale)
    # 3) normalized, then scaled by correlation with accuracy

    # (A) Identify high and low accurate networks
    top_performers = np.argsort(y)[-50:]
    bad_performers = np.argsort(y)[:50]

    # (B) For each top performer, identify k nearest neighbors with each of the above methods
    hi_acc_neighbors_raw, hi_acc_neighbors_norm = {}, {}
    lo_acc_neighbors_raw, lo_acc_neighbors_norm = {}, {}
    closest_neighbor_lo_norm, closest_neighbor_hi_norm = {}, {}

    # Compute feature correlations with accuracy
    w = np.array([stats.pearsonr(X[:,i], y) for i in range(X.shape[1])])
    print(np.where(w[:,1] < 0.01))

    # B1. neighbors based on raw distance in parameter space
    raw_neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
    raw_neighbors.fit(X)
    for net in top_performers:
        net_X = X[net,:].reshape(1, -1)
        neighbors_d, neighbors_i = raw_neighbors.kneighbors(net_X)
        neighbors_i = neighbors_i.squeeze()[1:]
        hi_acc_neighbors_raw[(net, y[net])] = y[neighbors_i]
    for net in bad_performers:
        net_X = X[net,:].reshape(1, -1)
        neighbors_d, neighbors_i = raw_neighbors.kneighbors(net_X)
        neighbors_i = neighbors_i.squeeze()[1:]
        lo_acc_neighbors_raw[(net, y[net])] = y[neighbors_i]
    print("Raw neighbors found")
        
    # B2. neighbors based on norm distance in parameter space
    norm_neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
    X_sc = StandardScaler().fit_transform(X)
    norm_neighbors.fit(X_sc)
    for net in top_performers:
        net_X = X_sc[net,:].reshape(1, -1)
        neighbors_d, neighbors_i = norm_neighbors.kneighbors(net_X)
        neighbors_i = neighbors_i.squeeze()[1:] # leave out the first one
        hi_acc_neighbors_norm[(net, y[net])] = y[neighbors_i]
        closest_neighbor_hi_norm[(net, y[net])] = y[neighbors_i[np.argmin(neighbors_d)]]
    for net in bad_performers:
        net_X = X_sc[net,:].reshape(1, -1)
        neighbors_d, neighbors_i = norm_neighbors.kneighbors(net_X)
        neighbors_i = neighbors_i.squeeze()[1:] # leave out the first one 
        lo_acc_neighbors_norm[(net, y[net])] = y[neighbors_i]
        closest_neighbor_lo_norm[(net, y[net])] = y[neighbors_i[np.argmin(neighbors_d)]]

    # Demonstrate that high-accuracy nets occur in locally enriched regions of the parameter space
    fig, ax = plt.subplots(ncols=2, figsize=(12,6))
    bins = np.linspace(0.45, 1.0, 20)
    ax[0].hist(list(closest_neighbor_hi_norm.values()), 
        bins=bins, 
        alpha=0.5, label='High accuracy nets (n=50)')
    ax[0].hist(list(closest_neighbor_lo_norm.values()), 
        bins=bins,
        alpha=0.5, label='Low accuracy nets (n=50)')
    ax[0].legend()
    ax[0].set(xlabel='Accuracy, closest neighbor in parameter-space',
           ylabel='# networks')

    ax[1].hist([np.mean(list(hi_acc_neighbors_norm[(i, y[i])])) for i in top_performers], 
        bins=bins, 
        alpha=0.5, label='High accuracy nets (n=50)')
    ax[1].hist([np.mean(list(lo_acc_neighbors_norm[(i, y[i])])) for i in bad_performers], 
        bins=bins,
        alpha=0.5, label='Low accuracy nets (n=50)')
    ax[1].legend()
    ax[1].set(xlabel='Mean accuracy, 10 closest neighbors in parameter-space',
           ylabel='# networks')
    fig.suptitle('Parameters leading to high-accuracy networks cluster')
    plt.tight_layout()
    fig.savefig(os.path.join(args.figure_save_path, "param_neighboring_hist.png"), dpi=500)

    # For the top 10 highest-accuracy nets and lowest-accuracy params, plot 
    # accuracy of neighbors as a function of distance; do this as percentile
    # (e.g. for each param set, compute the distance wrt all others,
    # then do it in terms of percentile distance from there)
    pctls = np.linspace(1, 0, 100)

    acc_as_fxn_of_distance_pctile_hi, acc_as_fxn_of_distance_pctile_lo = [], []
    for net in top_performers[-20:]:
        net_X = X_sc[net].reshape(1,-1)
        neighbors_d, neighbors_i = norm_neighbors.kneighbors(net_X, n_neighbors = X_sc.shape[0])
        neighbors_d = neighbors_d.squeeze()
        neighbors_i = neighbors_i.squeeze()
        d_ranges = np.percentile(neighbors_d, pctls)
        net_accs = []
        for i in range(len(d_ranges)):
            relevant_nets = np.where(neighbors_d < d_ranges[i])[0]
            net_accs.append(np.mean(y[neighbors_i[relevant_nets]]))
        acc_as_fxn_of_distance_pctile_hi.append(np.array(net_accs))
    for net in bad_performers[:20]:
        net_X = X_sc[net].reshape(1,-1)
        neighbors_d, neighbors_i = norm_neighbors.kneighbors(net_X, n_neighbors = X_sc.shape[0])
        neighbors_d = neighbors_d.squeeze()
        neighbors_i = neighbors_i.squeeze()
        d_ranges = np.percentile(neighbors_d, pctls)
        net_accs = []
        for i in range(len(d_ranges)):
            relevant_nets = np.where(neighbors_d < d_ranges[i])[0]
            net_accs.append(np.mean(y[neighbors_i[relevant_nets]]))
        acc_as_fxn_of_distance_pctile_lo.append(np.array(net_accs))

    fig, ax = plt.subplots(1)
    ax.plot(np.array(acc_as_fxn_of_distance_pctile_lo).T, color='red', alpha=0.2)
    ax.plot(np.array(acc_as_fxn_of_distance_pctile_hi).T, color='blue', alpha=0.2)
    ax.plot(np.mean(np.array(acc_as_fxn_of_distance_pctile_lo).T, axis=1), color='red', linewidth=2)
    ax.plot(np.mean(np.array(acc_as_fxn_of_distance_pctile_hi).T, axis=1), color='blue', linewidth=2)
    plt.tight_layout()
    fig.savefig(os.path.join(args.figure_save_path, "param_neighboring_distances.png"), dpi=500)
 
def cluster_params(X, y):
    X_all = []
    for col_a in range(X.shape[1]):
        for col_b in range(col_a, X.shape[1]):
            X_all.append(np.divide(X[:,col_a], X[:,col_b]))
            X_all.append(np.divide(X[:,col_b], X[:,col_a]))
    X_all = np.array(X_all).T
    X_sc = StandardScaler().fit_transform(X_all)
    tsne = TSNE(2, 100).fit_transform(X_sc)
    clustering = DBSCAN(eps=1, min_samples=2).fit(X_sc[y > 0.8])
    
    fig = plt.figure()
    ax = fig.add_subplot()
    c = ax.scatter(tsne[y > 0.7,-0], tsne[y > 0.7,-1], c=y[y > 0.7])
    ax.scatter(tsne[y <= 0.7, -0], tsne[y <=0.7, -1], color='k', alpha=0.3)
    fig.colorbar(c)
    plt.show()


if __name__ == "__main__":
    X, y, names, frs = load_data()
    #neighborhood_analysis(X, y, k=5)
    cluster_params(X[y > 0.8], y[y > 0.8])
    #cluster_params(X, frs.mean(1))


    