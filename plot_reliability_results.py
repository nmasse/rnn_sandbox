import os, pickle, scipy, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
import matplotlib.transforms as transforms
import scipy.signal
import yaml, copy
from collections import defaultdict
#mpl.use('Agg')
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
    orig_accs = []
    param_fns = []

    fns = os.listdir(d)
    for fn in fns:
        f = os.path.join(d, fn)
        if os.path.isdir(f) or not f.endswith(".pkl"):
            continue
        x = pickle.load(open(f,'rb'))

        if len(x.keys()) < 1:
            continue
        if len(x['task_accuracy']) < 5:
            print(f, len(x['task_accuracy']))
            continue

        # Identify the original accuracy
        orig_accuracy = os.path.basename(f)
        start = orig_accuracy.find("acc=")+4
        end = start + 6
        orig_accuracy = float(orig_accuracy[start:end])
        param_fns.append(os.path.basename(f))
        

        if len(x['task_accuracy']) > 0:

            task_acc = np.array(x['task_accuracy'])
            min_accuracy.append(np.amin(task_acc[:, -10:, :].mean(axis=(1,2))))
            mean_accuracy.append(task_acc[:, -10:, :].mean(axis=(1,2)))
            final_mean_h.append(np.array(x['final_mean_h']).mean(1))



        og_results = yaml.load(open(f"./rnn_params/7tasks_high_accuracy_parameters/{os.path.basename(f)[:-4]}.yaml", 'rb'), Loader=yaml.FullLoader)
        og_results = og_results['filename']
        og_results = pickle.load(open(f"../7tasks/model_experimental/{og_results}", 'rb'))
        orig_accs.append(np.mean(og_results['task_accuracy'][-10:]))
        og_results = np.amax(og_results['initial_mean_h'])# + og_results['steady_state_h'] 
        if np.isfinite(og_results) and og_results < 1000:
            mean_h.append(og_results)
        else:
            mean_h.append(1000)


    mean_accuracy = np.array(mean_accuracy)
    mean_accuracy_by_orig = copy.copy(mean_accuracy)
    orig_accs = np.array(orig_accs).flatten()
    
    # Compute reliability scores
    sd_acc_across_reps = np.std(mean_accuracy_by_orig, axis=1)
    fig, ax = plt.subplots(1)
    ax.scatter(sd_acc_across_reps, orig_accs)
    fig.savefig("orig_acc_vs_sd_rep_acc.png", dpi=300)

    # Sort orig accuracies; print in order
    ordering = np.argsort(orig_accs)
    for i in ordering:
        print(param_fns[i], orig_accs[i], f"{sd_acc_across_reps[i]:.4f}")
    mean_accuracy = mean_accuracy.flatten()
    
    mean_h = np.array(mean_h).flatten()
    final_mean_h = np.array(final_mean_h).flatten()
    fig, ax = plt.subplots(ncols=3, figsize=(20,8))
    x = np.repeat(orig_accs, task_acc.shape[0])

    ax[0].scatter(x, mean_accuracy)
    ax[0].plot(np.linspace(mean_accuracy.min(), 1.0, 100), np.linspace(mean_accuracy.min(), 1.0, 100))
    ax[0].set(xlabel="Original accuracy (% correct, 7 tasks)",
           ylabel="Repeat variant task accuracy (% correct, 7 tasks)",
           title="Parameter reliability assay")

    # Low activity, high activity separately
    counts, bins = np.histogram(mean_h, 10)
    high_activity = np.where(mean_h >= bins[1])[0] 
    low_activity  = np.where(mean_h < bins[1])[0] 
    print(bins[1])
    low_x = np.repeat(orig_accs[low_activity], task_acc.shape[0])
    high_x = np.repeat(orig_accs[high_activity], task_acc.shape[0])
    low_inds = np.array([np.arange(i*task_acc.shape[0], (i + 1)*task_acc.shape[0]) for i in low_activity]).flatten()
    high_inds = np.array([np.arange(i*task_acc.shape[0], (i + 1)*task_acc.shape[0]) for i in high_activity]).flatten()
    ax[1].scatter(low_x, mean_accuracy[low_inds])
    ax[1].set(xlabel="Original accuracy (% correct, 7 tasks)",
           ylabel="Repeat variant task accuracy (% correct, 7 tasks)",
           title="Parameter reliability assay (low-activity networks)",
           ylim=[0.5, 1.0])
    ax[2].scatter(high_x, mean_accuracy[high_inds])
    ax[2].set(xlabel="Original accuracy (% correct, 7 tasks)",
           ylabel="Repeat variant task accuracy (% correct, 7 tasks)",
           title="Parameter reliability assay (high-activity networks)",
           ylim=[0.5, 1.0])

    fig.savefig("7tasks_parameter_reliability.png", dpi=300)

    fig, ax = plt.subplots(1, figsize=(8,8))

if __name__ == "__main__":
    plot_results(args.data_dir)