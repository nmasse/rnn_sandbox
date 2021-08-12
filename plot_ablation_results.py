import os, pickle, scipy, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
import matplotlib.transforms as transforms
import scipy.signal
import yaml
from collections import defaultdict
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
    accuracy = defaultdict(list)
    min_accuracy = defaultdict(list)
    mean_accuracy = defaultdict(list)
    sample_decoding = defaultdict(list)
    mean_h = defaultdict(list)
    initial_mean_h = defaultdict(list)
    final_mean_h = defaultdict(list)

    names_to_check = ['mod', 'topo', 'alpha']

    fns = os.listdir(d)
    for fn in fns:
        f = os.path.join(d, fn)
        if os.path.isdir(f) or not f.endswith(".pkl"):
            continue
        x = pickle.load(open(f,'rb'))

        # Identify which ablation group the current net belongs to
        params = dict(vars(x['rnn_params']).items())
        keys = params.keys()
        keys = list(set(list(keys)).difference(set(['tc_modulator'])))
        for name in names_to_check:
            rel_keys = [k for k in keys if name in k]

            if all([params[k] == 0 for k in rel_keys]):
                # If all have been 0: break (we've found the one!)
                mode = name
                break


        if len(x['task_accuracy']) > 0:

            task_acc = np.array(x['task_accuracy'])
            min_accuracy[mode].append(np.amin(task_acc[-5:, :].mean(axis=0)))
            mean_accuracy[mode].append(task_acc[-5:, :].mean())
            accuracy[mode].append(np.stack(x['task_accuracy']))
            sample_decoding[mode].append(x['sample_decoding'][0])

        if np.isfinite(x['steady_state_h']) and x['steady_state_h'] < 1000:
            mean_h[mode].append(x['steady_state_h'])
        else:
            mean_h[mode].append(1000)

    fig, ax = plt.subplots(3, 3, figsize=(14,10))
    N_BINS = 19

    for i, mode in enumerate(names_to_check):
        if len(accuracy[mode]) == 0:
            continue
        accuracy_mode = np.stack(accuracy[mode], axis=0)
        accuracy_all_tasks = np.mean(accuracy_mode, axis=-1)
        boxcar = np.ones((10,1), dtype=np.float32)/10.
        filtered_acc = scipy.signal.convolve(accuracy_all_tasks.T, boxcar, 'valid')

        # Row 1: log mean h histogram, accuracy histogram, sample decode
        counts, bins, patches = ax[0,i].hist(np.log10(mean_h[mode]), N_BINS)
        ax[0,i].set(xlabel="$\log_{10}$ mean hidden activity",
                    ylabel="Count")
        # Make ticks to be the midpoints of the bins
        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        bins_to_label = [bin_centers[i] for i in range(0, len(bin_centers), 2)]
        bin_labels = [f"{b:.1f}" for b in bins_to_label]
        bin_labels[-1] = f"{bin_labels[-1]}+"
        ax[0,i].set_xticks(bins_to_label)
        ax[0,i].set_xticklabels(bin_labels)
        ax[0,i].set_title(r'$\bf{Ablation: ' + mode + '}$\n' + "$\log_{10}$ " + f"mean hidden activity (N={len(mean_h[mode])})")
        ax[1,i].hist(sample_decoding[mode], N_BINS)
        #print(sample_decoding[mode])
        ax[1,i].axvline(x=1/6.0, linestyle='--', color='red', label=f'Chance ({float(1/6.)*100:.2f}%)')
        ax[1,i].legend()
        ax[1,i].set(xlabel="Initial sample decoding",
                    ylabel="Count",
                    title=f"Initial sample decoding (N={len(sample_decoding[mode])})")
    
        ax[2,i].hist(filtered_acc[-1,:],N_BINS)
        ax[2,i].axvline(x=0.5, linestyle='--', color='red', label=f'Chance (50%)')
        ax[2,i].legend()
        ax[2,i].set_xlabel('Final task accuracy')
        ax[2,i].set_ylabel('Count')
        ax[2,i].set_title(f'Final task accuracy (N={len(sample_decoding[mode])})')
        ax[2,i].set_xlim([0.35, 1.0])

        
    plt.tight_layout()
    plt.savefig(f'results_7tasks_ablations.png', dpi=300)

if __name__ == "__main__":
    plot_results(args.data_dir)