import os, pickle, scipy, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
import matplotlib.transforms as transforms
import scipy.signal
import yaml
mpl.use('Agg')
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

parser = argparse.ArgumentParser('')
parser.add_argument('data_dir', type=str, default='./results/')
parser.add_argument('--base_dir', type=str, default='/home/mattrosen/rnn_sandbox/')
parser.add_argument('--n_stim_batches', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.02)
parser.add_argument('--adam_epsilon', type=float, default=1e-7)
parser.add_argument('--n_learning_rate_ramp', type=int, default=20)
parser.add_argument('--save_frs_by_condition', type=bool, default=False)
parser.add_argument('--gamma', type=float, default=0.0)
parser.add_argument('--lmbda', type=float, default=0.0)
parser.add_argument('--rnn_params_fn', type=str, default='./rnn_params/base_rnn_mod.yaml')
parser.add_argument('--params_range_fn', type=str, default='./rnn_params/param_ranges.yaml')
parser.add_argument('--save_path', type=str, default='./results/run_073021_best')

args = parser.parse_args()

def plot_results(data_dir, base_dir = args.base_dir):
    d = os.path.join(base_dir, data_dir)
    accuracy = []
    min_accuracy = []
    mean_accuracy = []
    sample_decoding = []
    mean_h = []
    initial_mean_h = []
    final_mean_h = []

    fns = os.listdir(d)
    for fn in fns:
        f = os.path.join(d, fn)
        if os.path.isdir(f):
            continue
        x = pickle.load(open(f,'rb'))

        if len(x['task_accuracy']) > 0:
            if len(x['task_accuracy'][-1]) != 7:
                continue
            if len(x['task_accuracy']) != 400:
                continue
            if np.mean(x['task_accuracy'][-10:]) > 0.9:
                print(x['task_accuracy'][-10:])
                initial_mean_h.append(x['initial_mean_h'][15:])
                final_mean_h.append(x['final_mean_h'][15:])

                # Save out params to yaml
                p = vars(x['rnn_params'])
                for k, v in p.items():
                    if type(v) == np.int64:
                        p[k] = int(v)
                    if type(v) == str:
                        p[k] = float(v)
                with open(f"rnn_params/params_acc={np.array(x['task_accuracy'][-2:]).mean():.4f}.yaml", 'w') as outfile:
                    yaml.dump(p, outfile,default_flow_style=False)
            task_acc = np.array(x['task_accuracy'])
            min_accuracy.append(np.amin(task_acc[-5:, :].mean(axis=0)))
            mean_accuracy.append(task_acc[-5:, :].mean())
            accuracy.append(np.stack(x['task_accuracy']))
            sample_decoding.append(x['sample_decoding'])
        if np.isfinite(x['steady_state_h']) and x['steady_state_h'] < 1000:
            mean_h.append(x['steady_state_h'])
        else:
            mean_h.append(1000)

    N_BINS = 19
    accuracy = np.stack(accuracy,axis=0)
    accuracy_all_tasks = np.mean(accuracy,axis=-1)
    boxcar = np.ones((10,1), dtype=np.float32)/10.
    filtered_acc = scipy.signal.convolve(accuracy_all_tasks.T, boxcar, 'valid')
    sample_decoding = np.stack(sample_decoding,axis=0)[:, 1]
    print(sample_decoding.shape, np.mean(sample_decoding))
    f,ax = plt.subplots(3,2,figsize = (10,12))

    # Row 1: activity/sample decoding plots
    counts, bins, patches = ax[0,0].hist(np.log10(mean_h), N_BINS)
    ax[0,0].set(xlabel="$\log_2$ mean hidden activity",
                ylabel="Count",
                title=f"$\log_2$ mean hidden activity (N={len(mean_h)})")
    

    # Make ticks to be the midpoints of the bins
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    bins_to_label = [bin_centers[i] for i in range(0, len(bin_centers), 2)]
    bin_labels = [f"{b:.1f}" for b in bins_to_label]
    bin_labels[-1] = f"{bin_labels[-1]}+"
    ax[0,0].set_xticks(bins_to_label)
    ax[0,0].set_xticklabels(bin_labels)

    ax[0,1].hist(sample_decoding, N_BINS)
    ax[0,1].axvline(x=1/6.0, linestyle='--', color='red', label=f'Chance ({float(1/6.)*100:.2f}%)')
    ax[0,1].legend()
    ax[0,1].set(xlabel="Initial sample decoding",
                ylabel="Count",
                title=f"Initial sample decoding (N={len(sample_decoding)})")
    

    # Row 2: accuracy plots
    ax[1,0].plot(filtered_acc)
    ax[1,0].grid(True)
    ax[1,0].set_xlabel('Training batches (1 batch = 256 trials)')
    ax[1,0].set_ylabel('Task accuracy')
    ax[1,0].set_title(f'Accuracy during training N={len(sample_decoding)}')
    ax[1,1].hist(filtered_acc[-1,:],N_BINS)
    ax[1,1].axvline(x=0.5, linestyle='--', color='red', label=f'Chance (50%)')
    ax[1,1].legend()
    ax[1,1].set_xlabel('Final task accuracy')
    ax[1,1].set_ylabel('Count')
    ax[1,1].set_title(f'Final task accuracy (N={len(sample_decoding)})')

    # Row 3: Activity post-training, mean accuracy vs. min task accuracy 
    trans = ax[2,0].get_xaxis_transform()
    ax[2,0].plot(np.arange(-10*20, (final_mean_h[0].shape[0] - 10)*20, 20), np.array(final_mean_h).T)
    ax[2,0].set(ylim=[ax[2,0].get_ylim()[0], ax[2,0].get_ylim()[1]*1.1])
    ax[2,0].axvline(x=0, linestyle='--', color='grey')
    ax[2,0].text(20, ax[2,0].get_ylim()[1]*0.925, 'Sample\nON', fontweight='bold', fontsize=10, color='dimgrey')
    ax[2,0].axvline(x=300, linestyle='--', color='grey')
    ax[2,0].text(320, ax[2,0].get_ylim()[1]*0.925, 'Sample\nOFF', fontweight='bold', fontsize=10, color='dimgrey')
    ax[2,0].axvline(x=1300, linestyle='--', color='grey')
    ax[2,0].text(1320, ax[2,0].get_ylim()[1]*0.925, 'Test\nON', fontweight='bold', fontsize=10, color='dimgrey')
    ax[2,0].set(xlabel='Time relative to sample onset (ms)',
                ylabel='Firing rate',
                title=f'Mean firing rates, DMS ({len(final_mean_h)} highest-accuracy networks)')
    ax[2,1].scatter(mean_accuracy, min_accuracy, alpha=0.3)
    ax[2,1].plot()
    ax[2,1].set(xlabel='Mean accuracy across tasks',
                ylabel='Min. task accuracy',
                title=f'Min vs. mean task accuracy (N={len(sample_decoding)})',
                xlim=[0.45, 1.0],
                ylim=[0.25, 1.0])
    
    plt.tight_layout()
    plt.savefig(f'results_{data_dir[:-1][data_dir[:-1].rfind("/")+1:]}.png', dpi=300)

if __name__ == "__main__":
    plot_results(args.data_dir)