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
parser.add_argument('--data_dir', type=str, default='experiment1/')
parser.add_argument('--base_dir', type=str, default='/media/graphnets/reservoir/')
parser.add_argument('--save_path', type=str, default='figures/')

args = parser.parse_args()

def add_mean(data, **kws):
    mean = np.mean(data['Accuracy'])
    ax = plt.gca()
    plt.axvline(x=mean)

def plot_results(data_dir, base_dir = args.base_dir):
    d = os.path.join(base_dir, data_dir)
    accuracy = []
    min_accuracy = []
    mean_accuracy = []
    sample_decoding = []
    mean_h = []
    initial_mean_h = []
    final_mean_h = []
    orig_accs = []
    param_fns = []
    dims = []

    fns = os.listdir(d)
    for fn in fns:
        f = os.path.join(d, fn)
        if os.path.isdir(f) or not f.endswith(".pkl"):
            continue
        x = pickle.load(open(f,'rb'))


        if len(x.keys()) < 1:
            continue

        # Identify the original accuracy
        orig_accuracy = os.path.basename(f)
        start = orig_accuracy.find("acc=")+4
        end = start + 6
        print(orig_accuracy)
        orig_accuracy = float(orig_accuracy[start:end])
        param_fns.append(os.path.basename(f))
        

        task_acc = np.array(x['task_accuracy'])
        min_accuracy.append(np.amin(task_acc[:, -10:, :].mean(axis=(1,2))))
        mean_accuracy.append(task_acc[:, -10:, :].mean(axis=(1,2)))
        final_mean_h.append(np.array(x['final_mean_h']).mean(1))
        initial_mean_h.append(np.array(x['initial_mean_h']).mean(1))
        sample_decoding.append(np.array(x['sample_decoding']))
        dims.append(np.array(x['dimensionality']))

        orig_accs.append(orig_accuracy)


    mean_accuracy = np.array(mean_accuracy)
    mean_accuracy_by_orig = copy.copy(mean_accuracy)
    orig_accs = np.array(orig_accs).flatten()
    for i in dims:
        print(np.mean(i, 0))
    print(0/0)
    
    # Compute reliability scores
    sd_acc_across_reps = np.std(mean_accuracy_by_orig, axis=1)
    fig, ax = plt.subplots(1)
    sns.scatterplot(x=orig_accs, y=sd_acc_across_reps, linewidth=0, ax=ax)
    ax.set(title=r'Original acc. vs. $\sigma$ acc. (N=50 nets)',
        xlabel='Original acc.',
        ylabel=r'$\sigma$ acc.')
    fig.tight_layout()
    fig.savefig("/Users/mattrosen/results_figs/orig_vs_sd_rep_acc.png", dpi=500)
    plt.close(fig)

    # One row per parameter set: plot histogram of accuracies of the 
    # repetitions, and mark the original mean for each distribution

    # First: make into pd dataframe
    net_ids = np.repeat(np.arange(mean_accuracy.shape[0]), mean_accuracy.shape[1])
    df = pd.DataFrame(np.hstack((mean_accuracy.flatten()[:,np.newaxis], net_ids[:,np.newaxis])), columns=['Accuracy', 'Param. set'])
    df = df.astype({"Accuracy": float, "Param. set": 'category'})
    g = sns.FacetGrid(df, row="Param. set", height=2, xlim=(0.5, 1.), aspect=4,)
    g.map_dataframe(sns.histplot, "Accuracy")

    g.map_dataframe(add_mean)

    '''
    fig, ax = plt.subplots(1, figsize=(3,3))
    x = np.repeat(orig_accs, task_acc.shape[0])

    sns.scatterplot(x, mean_accuracy, linewidth=0, ax=ax)
    ax.plot(np.linspace(mean_accuracy.min(), 1.0, 100), np.linspace(mean_accuracy.min(), 1.0, 100))
    ax.set(xlabel="Original acc.",
           ylabel=r"$\mu$ acc. (N=50 repeats)",
           title="Parameter reliability")
    '''
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig("/Users/mattrosen/results_figs/init_reliability_separatesubplots.png", dpi=500)
    plt.close(fig)

    # Make into same subplot
    fig, ax = plt.subplots(1, figsize=(6,4))
    sns.histplot(data=df, x='Param. set', y='Accuracy',
        legend=False, cbar=True, cbar_kws={"label": "Count"})
    ax.set(title='Initialization reliability', ylim=[0.5, 1.0])
    fig.tight_layout()
    fig.savefig("/Users/mattrosen/results_figs/init_reliability_singlesubplot.png", dpi=500)
    plt.close(fig)




if __name__ == "__main__":
    plot_results(args.data_dir)