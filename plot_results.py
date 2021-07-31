import os, pickle, scipy, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
mpl.use('Agg')

parser = argparse.ArgumentParser('')
parser.add_argument('data_dir', type=str)
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
    sample_decoding = []
    mean_h = []
    fns = os.listdir(d)
    for fn in fns:
        f = os.path.join(d, fn)
        x = pickle.load(open(f,'rb'))
        if len(x['task_accuracy']) > 0:
            mean_h.append(x['steady_state_h'])
            accuracy.append(np.stack(x['task_accuracy']))
            sample_decoding.append(x['sample_decoding'])
    accuracy = np.stack(accuracy,axis=0)
    accuracy_all_tasks = np.mean(accuracy,axis=-1)
    boxcar = np.ones((10,1), dtype=np.float32)/10.
    filtered_acc = scipy.signal.convolve(accuracy_all_tasks.T, boxcar, 'valid')
    sample_decoding = np.stack(sample_decoding,axis=0)[:, 1]
    f,ax = plt.subplots(1,2,figsize = (10,4))
    ax[0].plot(filtered_acc)
    ax[0].grid(True)
    ax[0].set_xlabel('Training batches (1 batch = 256 trials)')
    ax[0].set_ylabel('Task accuracy')
    ax[0].set_title(f'Accuracy during training N={len(sample_decoding)}')
    ax[1].hist(filtered_acc[-1,:],20)
    ax[1].set_xlabel('Final task accuracy')
    ax[1].set_ylabel('Count')
    ax[1].set_title(f'Final task accuracy N={len(sample_decoding)}')
    plt.tight_layout()
    plt.savefig('results.jpg')

if __name__ == "__main__":
    plot_results(args.data_dir)