import os, pickle, scipy, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
import matplotlib.transforms as transforms
import scipy.signal
import yaml
import seaborn as sns, pandas as pd

mpl.use('Agg')
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
import matplotlib as mpl
mpl.rcParams['font.family'] = 'CMU Sans Serif'
np.set_printoptions(precision=3, suppress=True)

parser = argparse.ArgumentParser('')
parser.add_argument('data_dir', type=str, default='./results/')
parser.add_argument('--base_dir', type=str, default='/home/mattrosen/rnn_sandbox/')
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/base_rnn_mod.yaml')
parser.add_argument('--params_range_fn', type=str, default='./rnn_params/param_ranges.yaml')
parser.add_argument('--param_save_path', type=str, default='../rnn_params/revised_tasks')
parser.add_argument('--save_path', type=str, default='./results/run_073021_best')
parser.add_argument('--do_save_params', type=bool, default=False)

args = parser.parse_args()

np.set_printoptions(precision=2, suppress=True)

def plot_results(data_dir, base_dir = args.base_dir):

    if not os.path.exists(args.param_save_path):
        os.makedirs(args.param_save_path)
    d = os.path.join(base_dir, data_dir)
    accuracy = []
    min_accuracy = []
    mean_accuracy = []
    sample_decoding = []
    mean_h = []
    initial_mean_h = []
    final_mean_h = []

    fns = os.listdir(d)
    print(len(fns))
    count = 0
    total_count = 0
    for fn in fns:
        f = os.path.join(d, fn)
        if os.path.isdir(f) or not f.endswith(".pkl"):
            continue
        x = pickle.load(open(f,'rb'))

        if 'sample_decoding' in x.keys():
            sample_decoding.append(x['sample_decoding'])

        if 'task_accuracy' not in x.keys():
            continue

        if len(x['task_accuracy']) > 0:
            total_count += 1
            if len(x['task_accuracy']) != 200:
                continue
            if np.mean(x['task_accuracy'][-10:]) > 0.9:
                initial_mean_h.append(x['initial_mean_h'][15:])
                final_mean_h.append(x['final_mean_h'][15:])

            task_acc = np.array(x['task_accuracy'])
            min_accuracy.append(np.amin(task_acc[-5:, :].mean(axis=0)))
            mean_accuracy.append(task_acc[-5:, :].mean())
            accuracy.append(np.stack(x['task_accuracy']))

            if np.mean(x['task_accuracy'][-25:]) > 0.95:
                print(f"{np.mean(x['task_accuracy'][-25:], axis=0)}, {x['final_mean_h'][15:].mean():.3f}, {sample_decoding[-1]}, params_acc={np.array(x['task_accuracy'][-2:]).mean():.4f}.yaml")
                #count += 1

            # Save out params to yaml
            if args.do_save_params:
            
                if np.mean(x['task_accuracy'][-25:]) > 0.95:
                    count += 1

                    p = vars(x['rnn_params'])
                    for k, v in p.items():
                        #print(k, type(v))
                        if type(v) == np.int64:
                            p[k] = int(v)
                        if type(v) == str and v != "none" and v != 'BPTT':
                            p[k] = float(v)
                    p['filename'] = os.path.basename(f)

                    with open(f"{args.param_save_path}/params_acc={np.array(x['task_accuracy'][-2:]).mean():.4f}.yaml", 'w') as outfile:
                        yaml.dump(p, outfile,default_flow_style=False)

        if np.isfinite(x['steady_state_h']) and x['steady_state_h'] < 1000:
            mean_h.append(x['steady_state_h'])
        else:
            mean_h.append(1000)

    print("N_NETS above 85%: ", count)
    print("N_NETS total: ", total_count)

    N_BINS = 19
    accuracy = np.stack(accuracy,axis=0)
    accuracy_all_tasks = np.mean(accuracy,axis=-1)
    boxcar = np.ones((10,1), dtype=np.float32)/10.
    filtered_acc = scipy.signal.convolve(accuracy_all_tasks.T, boxcar, 'valid')
    sample_decoding = np.stack(sample_decoding,axis=0)[:,1,:]
    f,ax = plt.subplots(3,2,figsize = (10,12))

    # Row 1: activity/sample decoding plots
    counts, bins, patches = ax[0,0].hist(np.log10(mean_h), N_BINS)
    ax[0,0].axvline(x=0., linestyle="--", color='red', label='Incl. thresh.')
    ax[0,0].set(xlabel="$\log_{10}$ mean hidden activity",
                ylabel="Count",
                title="$\log_{10}$ mean hidden activity " + f"(N={len(mean_h)})")
    ax[0,0].legend()
    

    # Make ticks to be the midpoints of the bins
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    bins_to_label = [bin_centers[i] for i in range(0, len(bin_centers), 2)]
    bin_labels = [f"{b:.1f}" for b in bins_to_label]
    bin_labels[-1] = f"{bin_labels[-1]}+"
    ax[0,0].set_xticks(bins_to_label)
    ax[0,0].set_xticklabels(bin_labels)

    ax[0,1].hist(sample_decoding, N_BINS)
    ax[0,1].axvline(x=1/8.0, linestyle='--', color='black', label=f'Chance ({float(1/8.)*100:.2f}%)')
    ax[0,1].axvline(x=0.75, linestyle="--", color='red', label='Incl. thresh. (75%)')
    ax[0,1].legend()
    ax[0,1].set(xlabel="Initial sample decoding",
                ylabel="Count",
                title=f"Initial sample decoding (N={len(sample_decoding)})")

    # Row 2: accuracy plots
    ax[1,0].plot(filtered_acc)
    ax[1,0].grid(True)
    ax[1,0].set_xlabel('Training batches (1 batch = 256 trials)')
    ax[1,0].set_ylabel('Task accuracy')
    ax[1,0].set_title(f'Accuracy during training N={accuracy.shape[0]}')
    ax[1,1].hist(filtered_acc[-1,:],N_BINS)
    ax[1,1].set_xlabel('Final task accuracy')
    ax[1,1].set_ylabel('Count')
    ax[1,1].set_title(f'Final task accuracy (N={accuracy.shape[0]})')
    print(accuracy.shape)

    # Row 3: Activity post-training, mean accuracy vs. min task accuracy 
    trans = ax[2,0].get_xaxis_transform()
    ax[2,0].plot(np.arange(-10*20, (final_mean_h[0].shape[0] - 10)*20, 20), np.array(final_mean_h).T)
    ax[2,0].set(ylim=[ax[2,0].get_ylim()[0], ax[2,0].get_ylim()[1]*1.1])
    ax[2,0].axvline(x=0, linestyle='--', color='grey')
    ax[2,0].text(20, ax[2,0].get_ylim()[1]*0.9, 'Sample\nON', fontweight='bold', fontsize=10, color='dimgrey')
    ax[2,0].axvline(x=300, linestyle='--', color='grey')
    ax[2,0].text(320, ax[2,0].get_ylim()[1]*0.9, 'Sample\nOFF', fontweight='bold', fontsize=10, color='dimgrey')
    ax[2,0].axvline(x=1300, linestyle='--', color='grey')
    ax[2,0].text(1320, ax[2,0].get_ylim()[1]*0.9, 'Response', fontweight='bold', fontsize=10, color='dimgrey')
    ax[2,0].set(xlabel='Time relative to sample onset (ms)',
                ylabel='Firing rate',
                title=f'Mean firing rates, DMS ({len(final_mean_h)} highest-accuracy networks)')
    scatter_colors = np.array(['blue']*len(mean_accuracy))
    scatter_colors[np.where(np.array(mean_accuracy) > 0.9)[0]] = 'red'
    ax[2,1].scatter(mean_accuracy, min_accuracy, alpha=0.3, color=scatter_colors)
    ax[2,1].plot()
    ax[2,1].set(xlabel='Mean accuracy across tasks',
                ylabel='Min. task accuracy',
                title=f'Min vs. mean task accuracy (N={accuracy.shape[0]})',
                xlim=[0.05, 1],
                ylim=[0.05, 1])
    
    plt.tight_layout()
    plt.savefig(f'/Users/mattrosen/results_figs/results_summary.png', dpi=300)

    ############################################################################
    # Figures for presentation (make each subplot its own figure)
    #
    # 1. Activity plot (most networks blow up!)
    # Row 1: activity/sample decoding plots
    colors = sns.color_palette('Set2', 8)
    fig, ax = plt.subplots(1, figsize=(6,4))
    log_act = np.log10(mean_h)
    bins = np.histogram(log_act, bins=N_BINS)[1]
    sns.histplot(x=log_act, bins=N_BINS, ax=ax, color=colors[0], element="step")
    ax.axvline(x=0., linestyle="--", color='#b3b3b3', label='Inclusion thr.')
    ax.set(xlabel="$log_{10}$ mean hidden activity",
           ylabel="Count",
           title="$log_{10}$ mean hidden activity " + f"(N={len(mean_h)})")

    # Make ticks to be the midpoints of the bins
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    bins_to_label = [bin_centers[i] for i in range(0, len(bin_centers), 2)]
    bin_labels = [f"{b:.1f}" for b in bins_to_label]
    bin_labels[-1] = f"{bin_labels[-1]}+"
    ax.set_xticks(bins_to_label)
    ax.set_xticklabels(bin_labels)

    ax.legend()
    plt.tight_layout()
    fig.savefig("/Users/mattrosen/results_figs/activity_hist.png", dpi=500)
    plt.close(fig)

    # 2. Stimulus encoding plot (another issue: most networks don't even
    # keep activity alive or non-chaotic well enough to provide basic
    # ingredients for task solution)

    # Make pandas df (one column per sample decode?)
    df = pd.DataFrame(sample_decoding, columns=['RF1', 'RF2'])
    df = pd.melt(df, id_vars=[], value_vars=['RF1', 'RF2'],
        var_name='RF', value_name='Decode')
    print(df.head())
    cmap = sns.color_palette('Set2', 2)
    fig, ax = plt.subplots(1, figsize=(6,4))
    sns.histplot(data=df, x='Decode', bins=N_BINS, ax=ax, hue='RF', multiple='dodge', palette=cmap)
    sns.move_legend(
            ax, "upper left",
            bbox_to_anchor=(1, 0.95),
            ncol=1,
            frameon=False,
        )
    
    from matplotlib.legend import Legend
    chance_line = ax.axvline(x=1/8.0, linestyle='--', color='black', label=f'Chance\n({float(1/8.)*100:.2f}%)')
    incl_thresh = ax.axvline(x=0.75, linestyle="--", color='#b3b3b3', label='Inclusion\nthr. (75%)')
    leg = Legend(ax, [chance_line, incl_thresh], 
        [f'Chance\n({float(1/8.)*100:.2f}%)', 'Inclusion\nthr. (75%)'],
                 loc='lower left', frameon=False, bbox_to_anchor=(1.0,0), title='Annot.')
    ax.add_artist(leg)


    ax.set(xlabel="Sample decode (pre-training)",
           ylabel='Count',
           title=f"Sample decode (pre-training) (N={len(sample_decoding)})")
    plt.tight_layout()
    fig.savefig("/Users/mattrosen/results_figs/sample_decoding_hist.png", dpi=500, bbox_inches="tight")
    plt.close(fig)

    # 3/3a. Accuracy plot (first with just the good ones, then with most networks 
    # in the middle)
    good_nets = np.where(accuracy_all_tasks[:,-25:].mean(1) > 0.9)[0]
    fig, ax = plt.subplots(1, figsize=(6,4))
    h0 = ax.plot(filtered_acc[:,good_nets], color='#fc8d62', linewidth=2)
    ax.set(title="Accuracy through training",
           xlabel="Batches",
           ylabel="Accuracy",
           ylim=[0.02,1.0])
    plt.tight_layout()
    plt.legend(handles=[h0[0]], labels=[r'Accuracy $>$ 0.9'])
    fig.savefig("/Users/mattrosen/results_figs/acc_traces.png", dpi=500)
    plt.close(fig)

    good_nets = np.where(accuracy_all_tasks[:,-25:].mean(1) > 0.9)[0]
    bad_nets = np.where(accuracy_all_tasks[:,-25:].mean(1) < 0.9)[0]
    fig, ax = plt.subplots(1, figsize=(6,4))
    h1 = ax.plot(filtered_acc[:,bad_nets], color='#b3b3b3', alpha=0.5, linewidth=0.25)
    h0 = ax.plot(filtered_acc[:,good_nets], color='#fc8d62', linewidth=2)
    ax.set(title="Accuracy through training",
           xlabel="Batches",
           ylabel="Accuracy",
           ylim=[0.02,1.0])
    plt.tight_layout()
    plt.legend(handles=[h0[0], h1[0]], labels=[r'Accuracy $>$ 0.9', r'Accuracy $<$ 0.9'])
    fig.savefig("/Users/mattrosen/results_figs/acc_traces_all.png", dpi=500)
    plt.close(fig)

    # 4. Accuracy histogram
    fig, ax = plt.subplots(1, figsize=(6,4))
    sns.histplot(filtered_acc[-1,:], bins=N_BINS, ax=ax, color=colors[0], element="step")
    ax.set(xlabel='Final accuracy',
           ylabel='Count',
           title=f'Final accuracy (N={accuracy.shape[0]})')
    plt.tight_layout()
    fig.savefig("/Users/mattrosen/results_figs/acc_hist.png", dpi=500)



if __name__ == "__main__":
    plot_results(args.data_dir)