import os, pickle, scipy, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
import scipy.signal
mpl.use('Agg')

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
            if np.mean(x['task_accuracy'][-10:]) > 0.88:
                initial_mean_h.append(x['initial_mean_h'])
                final_mean_h.append(x['final_mean_h'])
            task_acc = np.array(x['task_accuracy'])
            min_accuracy.append(np.amin(task_acc[-5:, :].mean(axis=0)))
            mean_accuracy.append(task_acc[-5:, :].mean())
            mean_h.append(x['steady_state_h'])
            accuracy.append(np.stack(x['task_accuracy']))
            sample_decoding.append(x['sample_decoding'])
    accuracy = np.stack(accuracy,axis=0)
    accuracy_all_tasks = np.mean(accuracy,axis=-1)
    boxcar = np.ones((10,1), dtype=np.float32)/10.
    filtered_acc = scipy.signal.convolve(accuracy_all_tasks.T, boxcar, 'valid')
    sample_decoding = np.stack(sample_decoding,axis=0)[:, 1]
    print(sample_decoding.shape, np.mean(sample_decoding))
    f,ax = plt.subplots(2,2,figsize = (10,4))
    ax[0,0].plot(filtered_acc)
    ax[0,0].grid(True)
    ax[0,0].set_xlabel('Training batches (1 batch = 256 trials)')
    ax[0,0].set_ylabel('Task accuracy')
    ax[0,0].set_title(f'Accuracy during training N={len(sample_decoding)}')
    ax[0,1].hist(filtered_acc[-1,:],20)
    ax[0,1].set_xlabel('Final task accuracy')
    ax[0,1].set_ylabel('Count')
    ax[0,1].set_title(f'Final task accuracy N={len(sample_decoding)}')
    ax[1,0].plot(np.array(initial_mean_h).T, 'b')
    ax[1,0].plot(np.array(final_mean_h).T, 'r')
    ax[1,0].grid(True)
    ax[1,0].set_xlabel('Time (timesteps)')
    ax[1,0].set_ylabel('Firing rate')
    ax[1,0].set_title(f'Mean firing rates, DMS')
    ax[1,1].scatter(mean_accuracy, min_accuracy, alpha=0.3)
    ax[1,1].plot()
    ax[1,1].set(xlabel='Mean accuracy across tasks',
                ylabel='Min. task accuracy',
                title=f'Min vs. mean task accuracy, N={len(sample_decoding)}',
                xlim=[0.45, 1.0],
                ylim=[0.25, 1.0],
                yscale='log')
    plt.tight_layout()
    plt.savefig(f'results_{data_dir[:-1][data_dir[:-1].rfind("/")+1:]}.png', dpi=300)

if __name__ == "__main__":
    plot_results(args.data_dir)