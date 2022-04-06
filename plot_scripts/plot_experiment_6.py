import os, pickle, scipy, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
import matplotlib.transforms as transforms
import scipy.signal
import yaml, copy
from collections import defaultdict
import seaborn as sns, pandas as pd

#mpl.use('Agg')
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'CMU Sans Serif'
np.set_printoptions(precision=3, suppress=True)

parser = argparse.ArgumentParser('')
parser.add_argument('--data_dir', type=str, default='experiment6/')
parser.add_argument('--base_dir', type=str, default='/media/graphnets/reservoir/')
parser.add_argument('--save_path', type=str, default='figures/')

args = parser.parse_args()

def plot_results(data_dir, base_dir = args.base_dir):
    d = os.path.join(base_dir, data_dir)
    accs = []
    fns = os.listdir(d)
    for fn in fns:
        f = os.path.join(d, fn)
        if os.path.isdir(f) or not f.endswith(".pkl"):
            continue
        x = pickle.load(open(f,'rb'))

        if len(x.keys()) < 1:
            continue

        # Save accuracy
        print(np.array(x['task_accuracy']).shape)
        acc = np.array(x['task_accuracy']).squeeze().mean(-1)
        accs.append(acc)

    accs = np.vstack(accs)
    boxcar = np.ones((20,1), dtype=np.float32)/20.
    filtered_accs = scipy.signal.convolve(accs.T, boxcar, 'valid')
    filtered_accs[np.where(~np.isfinite(filtered_accs))] = 0.
    filt_acc_last_few = np.mean(filtered_accs[-5:,:], axis=0)[:,np.newaxis]


    # Make df
    acc_flat = filtered_accs.flatten()
    alphastr = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nets = np.tile(np.repeat(np.arange(9), 10), filtered_accs.shape[0]).astype(np.int16)
    iters = np.repeat(np.arange(filtered_accs.shape[0]), filtered_accs.shape[1]).astype(np.int16)

     # Make df for last iteration data
    acc_last_flat = filt_acc_last_few.flatten()
    alphastr = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nets_last = np.repeat(np.arange(9), 10).astype(np.int16)

    data = np.hstack((acc_flat[:,np.newaxis], iters[:,np.newaxis], nets[:,np.newaxis]))
    data_last = np.hstack((acc_last_flat[:,np.newaxis], nets_last[:,np.newaxis]))


    df = pd.DataFrame(data, columns=['Accuracy', 'Iteration', 'Network'])
    df_last = pd.DataFrame(data_last, columns=['Accuracy', 'Param. set'])
    df['Network'] = pd.Categorical(df['Network'].astype(int))
    df_last['Param. set'] = pd.Categorical(df_last['Param. set'].astype(int))
    fig, ax = plt.subplots(1, figsize=(4,3))
    sns.lineplot(data=df, x="Iteration", y="Accuracy", hue="Network", palette='colorblind', ax=ax, ci='sd')
    sns.move_legend(
            ax, "center left",
            bbox_to_anchor=(1, 0.5),
            ncol=1,
            frameon=False,
        )
    fig.tight_layout()
    fig.savefig("/Users/mattrosen/results_figs/exp6_acc.png", dpi=500)
    plt.close(fig)

    # Make into same subplot
    print(df_last)
    fig, ax = plt.subplots(1, figsize=(6,4))
    sns.histplot(data=df_last, x='Param. set', y='Accuracy',
        legend=False, cbar=True, cbar_kws={"label": "Count"})#,"ticks":np.arange(5).astype(np.int16)})
    ax.set(title='Learning (top-down bias frozen)', ylim=[0.1, 1.0])
    fig.tight_layout()
    fig.savefig("/Users/mattrosen/results_figs/exp6_reliability_singlesubplot.png", dpi=500)
    plt.close(fig)


if __name__ == "__main__":
    plot_results(args.data_dir)