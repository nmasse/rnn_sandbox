import os, pickle, scipy, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
import matplotlib.transforms as transforms
import scipy.signal
import yaml
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

    fns = os.listdir(d)
    for fn in fns:
        f = os.path.join(d, fn)
        if os.path.isdir(f) or not f.endswith(".pkl"):
            continue
        x = pickle.load(open(f,'rb'))

        # Identify the original accuracy
        orig_accuracy = os.path.basename(f)
        start = orig_accuracy.find("acc=")+4
        end = start + 6
        orig_accuracy = float(orig_accuracy[start:end])
        orig_accs.append(orig_accuracy)

        if len(x['task_accuracy']) > 0:

            task_acc = np.array(x['task_accuracy'])
            min_accuracy.append(np.amin(task_acc[:, -2:, :].mean(axis=(1,2))))
            mean_accuracy.append(task_acc[:, -2:, :].mean(axis=(1,2)))

        for i in range(task_acc.shape[0]):
            if np.isfinite(np.mean(x['final_mean_h'][i])) and np.mean(x['final_mean_h'][i]) < 1000:
                mean_h.append(np.mean(x['final_mean_h'][i]))
            else:
                mean_h.append(1000)

    mean_accuracy = np.array(mean_accuracy).flatten()
    fig, ax = plt.subplots(1, figsize=(8,8))
    x = np.repeat(orig_accs, task_acc.shape[0])
    ax.scatter(x, mean_accuracy)
    ax.plot(np.linspace(mean_accuracy.min(), 1.0, 100), np.linspace(mean_accuracy.min(), 1.0, 100))
    ax.set(xlabel="Original accuracy (% correct, 7 tasks)",
           ylabel="Repeat variant task accuracy (% correct, 7 tasks)",
           title="Parameter generalizability assay",
           ylim=[0.5, 1.0],
           xlim=[0.5, 1.0])

    fig.savefig("7tasks_parameter_generalizability.png", dpi=300)

    

if __name__ == "__main__":
    plot_results(args.data_dir)